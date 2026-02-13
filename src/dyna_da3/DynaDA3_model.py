from pyexpat import features
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.logger import logger

DA3_VITG_CHANNELS = 1536 
DA3_VITL_CHANNELS = 1024
DA3_VITG_FEAT_LAYERS=(21, 27, 33, 39)
DA3_VITL_FEAT_LAYERS=(11, 15, 19, 23)
DA3_VITG_CKPT_PATH = "../checkpoint/DA3-GIANT-1.1"
DA3_VITL_CKPT_PATH = "../checkpoint/DA3-LARGE-1.1"
DPT_EMBED_DIM = 2048

MODEL_CONFIGS = {
    'vitl': {
        'channels': DA3_VITL_CHANNELS,
        'feat_idxs': DA3_VITL_FEAT_LAYERS,
        'ckpt_path': DA3_VITL_CKPT_PATH
    },
    'vitg': {
        'channels': DA3_VITG_CHANNELS,
        'feat_idxs': DA3_VITG_FEAT_LAYERS,
        'ckpt_path': DA3_VITG_CKPT_PATH
    }
}

#######################################################
# UncertaintyDPT V3.2
# 
#######################################################
class UncertaintyDPT(nn.Module):
    """
    UncertaintyDPT
    Args:
        B: Batch Size=1
        N: Frame Sequence Length
        H, W: 修正过的图像分辨率:  H=14*h, W=14*w
        c_in: DINO 输入通道数
        c_embed: 嵌入维度
        feat_idxs: 从 DA3 提取的指定DINO特征层
    """
    def __init__(self, c_in, feat_idxs, c_embed=DPT_EMBED_DIM):
        super().__init__()
        self.feat_idxs = list(feat_idxs)

        # 投影层：将 DINO 特征投影到 c_embed 维度
        self.projects_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c_in, c_embed, kernel_size=1),
                    nn.ReLU(inplace=True),
                )
                for _ in range(len(self.feat_idxs))
            ]
        )

        # 深度/置信度特征头输入通道数：
        # 1. depth_norm (相对深度)
        # 2. conf_norm (相对置信度)
        # 3. |∇depth| (深度梯度)
        # 4. heuristic_uncertainty: (1-conf_norm) * (1-depth_norm)
        # 5. low_conf_indicator: (1-conf_norm)
        self.patch_size = 14
        geo_in_channels = 5 * (self.patch_size ** 2)
        
        # 将深度/置信度特征投影到 c_embed - 使用 1x1 卷积在 patch 内部混合信息
        self.geo_head = nn.Sequential(
            nn.Conv2d(geo_in_channels, c_embed, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        
        # 融合深度/置信度特征和图像特征融合
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(c_embed * 2, c_embed, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_embed),
            nn.ReLU(inplace=True),
        )

        num_classes = 2  # 类别: 0=Certain/Static, 1=Uncertain/Dynamic
        self.output_layer = nn.Sequential(
            nn.Conv2d(c_embed, c_embed // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_embed // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_embed // 2, num_classes * (self.patch_size ** 2), kernel_size=1),
            nn.PixelShuffle(self.patch_size)
        )
        
    def _geo_normalize(self, x):
            """
            基于百分位数的归一化，能自适应不同的数值范围 (如 conf >= 1 的情况)
            x: [N, 1, H, W]
            """
            N = x.shape[0]
            x_flat = x.view(N, -1)
            
            # 计算 5% 和 95% 分位数来确定有效范围，避免极值干扰
            q_low = torch.quantile(x_flat, 0.05, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            q_high = torch.quantile(x_flat, 0.95, dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
            
            # 归一化到 [0, 1]
            denom = q_high - q_low
            denom[denom < 1e-6] = 1e-6 # 防止除零
            
            x_norm = (x - q_low) / denom
            return x_norm.clamp(0.0, 1.0)

    def _geo_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        [N, C, H, W] -> [N, C*14*14, h, w]
        Pixel Unshuffle: 把空间维度信息 (patch_size x patch_size) 转移到通道维度
        """
        logger.debug(f"Geo preprocess input shape: {x.shape}")
        x = F.pixel_unshuffle(x, self.patch_size)
        logger.debug(f"Geo preprocess output shape: {x.shape}")
        return x

    def forward(
        self,
        feats: list[torch.Tensor],
        H: int,
        W: int,
        conf: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        构造先验特征
        假设:
            depth_norm 接近 0 -> 浅/近 (Shallow)
            conf_norm 接近 0 -> 低置信度 (Low Confidence)
        目标: 
            找出 "深度浅 且 置信度低" 的区域 -> 认为是 Uncertain
        Args:
            feats: List[[N, c_in, h, w]] 图像特征
            conf: 置信度图 Tensor, shape: [N, 1, H, W] 
            depth: 深度图 Tensor, shape: [N, 1, H, W] 
        """
        # 鲁棒归一化处理
        # 无论原始范围 归一化后到[0,1] 0=相对低值, 1=相对高值
        conf_norm = self._geo_normalize(conf)   
        depth_norm = self._geo_normalize(depth)
        
        # Feature: 深度梯度 (需在HW空间计算)
        grad_x = torch.abs(depth_norm[:, :, :, 1:] - depth_norm[:, :, :, :-1])
        grad_y = torch.abs(depth_norm[:, :, 1:, :] - depth_norm[:, :, :-1, :])
        depth_grad = F.pad(grad_x, (0, 1, 0, 0)) + F.pad(grad_y, (0, 0, 0, 1))
        
        # 预处理: HW -> hw [N, 1, H, W] -> [N, 1*14*14, h, w]
        depth_feat = self._geo_preprocess(depth_norm)
        conf_feat = self._geo_preprocess(conf_norm)
        grad_feat = self._geo_preprocess(depth_grad)

        # 融合深度和深度置信度 (在 hw 空间进行计算)
        # Feature: 如果深度浅 (1.0 - depth_norm 大) 且 置信度低 (1.0 - conf_norm 大)，则该项值大
        uncertainty_prior = (1.0 - conf_feat) * (1.0 - depth_feat)
        low_conf = (1.0 - conf_feat)

        geo_input = torch.cat(
            [depth_feat, conf_feat, grad_feat, uncertainty_prior, low_conf], dim=1
        )  # [N, 5*p^2, h, w]
        
        # 投影到 c_embed
        geo_feat = self.geo_head(geo_input)  # [N, c_embed, h, w]

        # 融合 DINO 特征
        img_feat_sum = None
        for i, feat in enumerate(feats):
            # [N, c_in, h, w] -> [N, c_embed, h, w]
            proj = self.projects_layer[i](feat)
            if img_feat_sum is None:
                img_feat_sum = proj
            else:
                img_feat_sum = img_feat_sum + proj

        # geo_feat与 img_feat 融合
        fused = torch.cat([geo_feat, img_feat_sum], dim=1) # [N, 2*c_embed, h, w]
        fused = self.fusion_layer(fused)                   # [N, c_embed, h, w]

        # 输出 output_layer 包含 PixelShuffle, 输出将回到 [N, num_classes, H, W]
        logits = self.output_layer(fused) 

        return logits


class DynaDA3(nn.Module):
    """
    DynaDA3
    """
    def __init__(
        self,
        model_name: str = 'vitl', # 'vitl' or 'vitg'
        uncertainty_head_ckpt_path: str = None, # 训练好的 uncertainty head 权重路径; 注意只有在训练时才可以不输入该参数
    ):
        super().__init__()

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"model_name must be one of {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_name]
        ckpt_path = config['ckpt_path']
        channels = config['channels']
        self.export_feat_idxs = list(config['feat_idxs'])

        print(f"Loading DA3 ({model_name}) from local path: {ckpt_path}...")
        self.da3 = DepthAnything3.from_pretrained(ckpt_path)

        # 冻结 DA3
        for p in self.da3.parameters():
            p.requires_grad = False
        self.da3.eval()

        # 初始化 UncertaintyDPT 
        self.uncertainty_head = UncertaintyDPT(
            c_in=channels,
            feat_idxs=range(len(self.export_feat_idxs)),
        )

        if uncertainty_head_ckpt_path:
            print(f"Loading uncertainty head from {uncertainty_head_ckpt_path}...")
            state_dict = torch.load(uncertainty_head_ckpt_path, map_location='cpu')
            missing_keys, _ = self.uncertainty_head.load_state_dict(state_dict, strict=True)
            if len(missing_keys) > 0:
                raise ValueError(f"Failed to load uncertainty_head weights. Missing: {missing_keys}")

        
    @staticmethod
    def _nhwc_to_nchw(feat: np.ndarray, device: torch.device) -> torch.Tensor:
        """
          feat shape: [N, h, w, C]  
        Returns:
          torch.Tensor [N, C, h, w] on device
        """
        assert isinstance(feat, np.ndarray), f"feat must be numpy, got {type(feat)}"
        assert feat.ndim == 4, f"feat must be [N,h,w,C], got {feat.shape}"
        
        t = torch.from_numpy(feat).to(device=device, dtype=torch.float32)     # [N,h,w,C]
        t = t.permute(0, 3, 1, 2).contiguous()                               # [N,C,h,w]
        return t

    def _run_uncertainty_head(self, prediction, device: torch.device):
        """
        从 prediction.aux 取特征，外加 conf/depth 跑 uncertainty_head
        并写回 prediction.uncertainty_seg_logits/uncertainty_seg_mask
        """
        # 用 processed_images 的分辨率作为最终输出 H,W
        H, W = prediction.processed_images.shape[1], prediction.processed_images.shape[2]

        # 按 export_feat_idxs 的顺序组织 feature list
        feats = []
        for layer in self.export_feat_idxs:
            k = f"feat_layer_{layer}"
            feats.append(self._nhwc_to_nchw(prediction.aux[k], device))

        # 处理 conf, prediction.conf [N, H, W]
        assert hasattr(prediction, "conf"), "prediction.conf missing"
        conf_np = prediction.conf # [N, H, W]
        conf_tensor = torch.from_numpy(conf_np).to(device=device, dtype=torch.float32).unsqueeze(1) # [N, H, W] -> [N, 1, H, W]

        # depth: prediction.depth [N, H, W]
        depth_np = prediction.depth
        depth_tensor = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32).unsqueeze(1) # [N, H, W] -> [N, 1, H, W]

        logits = self.uncertainty_head(  # [N,K,H,W]
            feats, H, W, conf=conf_tensor, depth=depth_tensor
        )
        mask = torch.argmax(logits, dim=1) # [N,H,W]

        prediction.uncertainty_seg_logits = logits
        prediction.uncertainty_seg_mask = mask

    @torch.no_grad()
    def inference(self, image, **kwargs):
        """
        在推理过程(需要所有其他头的输出)使用inferrence;
        由于只训练 uncertainty_head,所以这里inferrence需要返回uncertainty_head的输出
        Args:
            images: [B, N, 3, H, W] Tensor, normalized (ImageNet mean/std)
        Returns:
            prediction: DepthAnything3.Prediction  包含 uncertainty_seg_logits / uncertainty_seg_mask
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        device = next(self.uncertainty_head.parameters()).device

        output = self.da3.inference(
            image=image,
            export_feat_layers=self.export_feat_idxs,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

        self._run_uncertainty_head(output, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.time()

        logger.info(f"DynaDA3 Inference | Total: {t2-t0:.3f}s | Backbone: {t1-t0:.3f}s | Uncertainty Head: {t2-t1:.3f}s")

        return output

    def forward(self, image):
        """
        在训练过程(只训练特定任务头)中使用forward, 由于训练 uncertainty_head,所以这里forward需要返回uncertainty_head的logits; 
        在推理过程(需要所有其他头的输出)使用inferrence;
        Args:
            images: [B, N, 3, H, W] Tensor, normalized (ImageNet mean/std) 
                注意, 这里输入的尺寸是合法尺寸,也就是 H=14*h, W=14*w; 
                在训练时, 由DataLoader进行裁剪; 
                在推理时,由DA3的InputProcessor进行裁剪;
        Returns:
            logits: [B, num_classes, H, W]
                num_classes: uncertainty segmentation classes = 2 (moving / static)

        """
        # [B, N, 3, H, W]
        if image.ndim != 5: 
            logger.error(f"Input image must be 5-D Tensor [B, N, 3, H, W], got {image.shape}")
            assert False

        B, N, _, H, W = image.shape

        # 冻结 DA3 并提取特征
        with torch.no_grad():
            out = self.da3.model(
                image, 
                export_feat_layers=self.export_feat_idxs,
            )

        # feat_layers特征处理
        feats = [] 
        for layer in self.export_feat_idxs:
            feat = out['aux'][f"feat_layer_{layer}"]  # 原始 Shape: [B, N, h, w, C]
            _, _, h, w, C = feat.shape
            feat = feat.view(B * N, h, w, C) # 合并B,N: [B, N, h, w, C] -> [B*N, h, w, C]
            feat = feat.permute(0, 3, 1, 2).contiguous()# 调整顺序 channel first: [B*N, h, w, C] -> [B*N, C, h, w] 
            feats.append(feat)

        # conf 处理 (注意 da3的forward不输出conf, 只输出depth_conf)
        conf = out['depth_conf'] # [B, N, H, W] 
        conf = conf.view(B * N, 1, H, W) # [B, N, H, W] -> [B*N, 1, H, W]

        # depth 处理 (DA3 main head 输出 depth)
        depth = out['depth'] # [B, N, H, W]
        depth = depth.view(B * N, 1, H, W) # [B, N, H, W] -> [B*N, 1, H, W]
        

        # feats: List[[B*N, C, h, w]];  conf: [B*N, 1, H, W] 
        logits = self.uncertainty_head(
            feats, H, W, conf=conf, depth=depth
        ) #  logits: [B*N, 2, H, W]
        
        return logits
