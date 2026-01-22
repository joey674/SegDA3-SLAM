import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.logger import logger

class SegDPTHead(nn.Module):
    """
    DPT 分割头
    Args:
        list[Batch_tensor]，每个 Batch_tensor: [N, Feat_Dim, h, w]
    Return:
        logits [N, K, H, W]
    """
    def __init__(self, in_channels=1024, embed_dim=256, num_classes=2, readout_indices=(0, 1, 2, 3)):
        super().__init__()
        self.readout_indices = list(readout_indices)# 选择哪些 Transformer 层的特征用于分割头

        self.projects_layer = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim, kernel_size=1),# 1x1 卷积: 用于将原始高维特征（如 1024 维）降维到统一的 embed_dim（如 256 维），减少计算量。
                    nn.ReLU(inplace=True),
                )
                for _ in range(len(self.readout_indices))# 这个投影层有并行的N个卷积模块, 每个对应处理不同特征层出来的特征
            ]
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, num_classes, kernel_size=1),
        )

    def forward(self, features: list[torch.Tensor], H: int, W: int) -> torch.Tensor:
        # features: list of [N, Feat_Dim, h, w]
        selected_block_features = [features[i] for i in self.readout_indices]

        # 特征投影与对齐
        # 使用projects_layer对每个选定的特征块进行投影, 并将它们上采样到相同的空间分辨率
        target_h, target_w = selected_block_features[0].shape[-2:]
        proj = []
        for i, feat in enumerate(selected_block_features):
            x = self.projects_layer[i](feat)
            if x.shape[-2:] != (target_h, target_w):
                x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
            proj.append(x)

        # 特征融合
        # 这是一种 Simple Summation (简单加和) 的融合方式 将浅层（纹理细节丰富）和深层（语义信息丰富）的特征直接叠加。
        fused = 0
        for x in proj:
            fused = fused + x

        # 输出层
        # 融合后的特征通过一个小型卷积网络进行最后的预测：
        # 3x3 卷积 + BN + ReLU: 用于平滑特征，消除由于插值（Interpolation）产生的伪影，并进一步提取局部特征。
        # 1x1 卷积: 最终投影到 num_classes 通道，得到每个类别的置信度分数（Logits）。
        logits = self.output_layer(fused)  # [N, K, target_h, target_w]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)  # [N,K,H,W]
        return logits



class SegDA3(nn.Module):
    """
    SegDA3: 基于 DepthAnything3 + DPTHead 的运动分割模型
    - 主体输出 DepthAnything3.inference() 的 Prediction 原样
    - 额外输出motion_seg_logits / motion_seg_mask
    """
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 256,
        in_channels: int = 1024,
        export_feat_layers=(3, 7, 11, 15, 19, 23),
        seg_head_ckpt_path: str = None,
    ):
        super().__init__()
        model_path = "/home/zhouyi/repo/SegDA3/checkpoints/DA3-LARGE-1.1"
        print(f"Loading DA3 from local path: {model_path}...")
        self.da3 = DepthAnything3.from_pretrained(model_path)

        # 冻结 DA3，只训练 seg_head
        for p in self.da3.parameters():
            p.requires_grad = False
        self.da3.eval()

        self.export_feat_layers = list(export_feat_layers)

        self.seg_head = SegDPTHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
            readout_indices=range(len(self.export_feat_layers)),
        )

        if seg_head_ckpt_path:
            print(f"Loading trained head from {seg_head_ckpt_path}...")
            # 加载权重到内存
            state_dict = torch.load(seg_head_ckpt_path, map_location='cpu')
            
            # strict=False 可以避免因为 DA3 内部一些非训练参数不一致导致的报错 只要 seg_head 的 key 匹配即可
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            # 检查 seg_head 是否加载成功
            head_keys = [k for k in missing_keys if 'seg_head' in k]
            if len(head_keys) > 0:
                raise ValueError("Failed to load seg_head weights.")
        
    @staticmethod
    def _aux_feat_to_nchw(feat: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        写死按照你当前 API inference 的实际输出：
          feat shape: [N, h, w, C]  (例如 5,28,36,1024)
        返回：
          torch.Tensor [N, C, h, w] on device
        """
        assert isinstance(feat, np.ndarray), f"aux feat must be numpy, got {type(feat)}"
        assert feat.ndim == 4, f"aux feat must be [N,h,w,C], got {feat.shape}"
        t = torch.from_numpy(feat).to(device=device, dtype=torch.float32)     # [N,h,w,C]
        t = t.permute(0, 3, 1, 2).contiguous()                               # [N,C,h,w]
        return t

    def _run_motion_head(self, prediction, device: torch.device):
        """
        从 prediction.aux 取特征，跑 seg_head，写回 prediction.motion_seg_*
        """
        assert hasattr(prediction, "aux") and isinstance(prediction.aux, dict), "prediction.aux missing"

        # 用 processed_images 的分辨率作为最终输出 H,W
        assert prediction.processed_images is not None
        H, W = prediction.processed_images.shape[1], prediction.processed_images.shape[2]

        # 按 export_feat_layers 的顺序组织 feature list
        feats = []
        for layer in self.export_feat_layers:
            k = f"feat_layer_{layer}"
            assert k in prediction.aux, f"Missing {k} in prediction.aux"
            feats.append(self._aux_feat_to_nchw(prediction.aux[k], device))

        logits = self.seg_head(feats, H, W)                # [N,K,H,W]
        mask = torch.argmax(logits, dim=1)                 # [N,H,W]

        prediction.motion_seg_logits = logits
        prediction.motion_seg_mask = mask

    @torch.no_grad()
    def inference(self, image, **kwargs):
        """
        完全走官方 API inference（主体输出不改），只额外挂 motion seg。
        你原来的 demo 用 image=，这里也用 image=。
        """
        device = next(self.seg_head.parameters()).device

        output = self.da3.inference(
            image=image,
            export_feat_layers=self.export_feat_layers,
        )

        # 额外 head（只有 head 在 device 上跑）
        self._run_motion_head(output, device)

        return output

    def forward(self, image):
        """
        在训练过程(只训练特定任务头)中使用forward; 
        在推理过程(需要所有其他头的输出)使用inferrence;
        由于只训练 seg_head,所以这里forward只需要返回seg_head的logits
        Args:
            images: [B, N, 3, H, W] Tensor, normalized (ImageNet mean/std)
        Returns:
            logits: [B, num_classes, H, W]
                num_classes: motion segmentation classes = 2 (moving / static)

        """
        # 输入图片的tensor维度检查 
        # [B, N, 3, H, W]
        if image.ndim != 5: 
            logger.error(f"Input image must be 5-D Tensor [B, N, 3, H, W], got {image.shape}")
            assert False

        B, N, _, H, W = image.shape

        # 骨干提取特征(冻结状态，不计算梯度以节省显存) 
        # [B, N, h=H/14, w=W/14, Feat_Dim=1024]
        # 把 patch 的信息融合到Feat_Dim里
        with torch.no_grad():
            # 为什么要用 with torch.no_grad()：
            # 显存优化：DA3 Large 参数量巨大，如果不加这个，PyTorch 会记录每一层的激活值用于求导，普通显卡会立刻 OOM（显存溢出）。
            # 职责分离：我们相信预训练好的 DA3 提取特征的能力，所以“冻结”它，只让权重在 seg_head 里流动。
            out = self.da3.model(
                image, 
                export_feat_layers=self.export_feat_layers
            )

        # 特征处理
        feats = [] 
        for layer in self.export_feat_layers:
            feat = out['aux'][f"feat_layer_{layer}"]  # 原始 Shape: [B, N, h, w, Feat_Dim]
            
            # 不管 N 是多少，直接把 B 和 N 合并; 因为分割头（DPTHead）是基于 2D 卷积的，它不认识序列维度 N
            _, _, h, w, Feat_Dim = feat.shape
            
            # [B, N, h, w, Feat_Dim] -> [B*N, h, w, Feat_Dim]
            feat = feat.view(B * N, h, w, Feat_Dim)

            # [B*N, h, w, Feat_Dim] -> [B*N, Feat_Dim, h, w] (Channel-First 适配卷积层)
            feat = feat.permute(0, 3, 1, 2).contiguous()
            feats.append(feat)

        # 4. 运行分割头
        # 此时 feats 里的每个 Tensor 都是 [Batch_total, Feat_Dim, h, w]
        logits = self.seg_head(feats, H, W) # 返回 [B*N, 2, H, W]
        
        return logits
        
