import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from SegDA3_model import SegDA3

# ================= config =================
CONFIG = {
    "video_dirs": [
        "/home/zhouyi/repo/model_DepthAnythingV3/inputs/dancer",
        "/home/zhouyi/repo/model_DepthAnythingV3/inputs/2077scene1",
    ],
    "save_dir": "/home/zhouyi/repo/model_DepthAnythingV3/checkpoints/SegDA3",
    "num_classes": 2,
    "seq_range": (2, 5),
    "lr": 1e-4,
    "epochs": 5,
    "input_size": (518, 518),
    "num_workers": 4,
}

# ================= dataset =================
class MultiVideoDataset(Dataset):
    def __init__(self, video_dirs, input_size=(518, 518), seq_range=(2, 5)):
        self.input_size = input_size
        self.seq_min, self.seq_max = seq_range
        self.samples = []

        for v_dir in video_dirs:
            img_dir = os.path.join(v_dir, "images")
            mask_dir = os.path.join(v_dir, "masks")
            
            if not (os.path.exists(img_dir) and os.path.exists(mask_dir)):
                print(f"Skipping: {v_dir} (missing folders)")
                continue

            v_imgs = sorted(glob.glob(os.path.join(img_dir, "*")))
            v_masks = sorted(glob.glob(os.path.join(mask_dir, "*")))

            if len(v_imgs) < self.seq_max:
                continue

            for i in range(len(v_imgs) - self.seq_max + 1):
                self.samples.append({
                    "imgs": v_imgs,
                    "masks": v_masks,
                    "start": i
                })

        print(f"Dataset initialized: {len(self.samples)} samples.")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        start = s["start"]
        n = random.randint(self.seq_min, self.seq_max)
        
        clip_imgs, clip_masks = [], []
        for i in range(start, start + n):
            img = Image.open(s["imgs"][i]).convert('RGB').resize(self.input_size, Image.BILINEAR)
            mask = Image.open(s["masks"][i]).resize(self.input_size, Image.NEAREST)
            
            mask_arr = np.array(mask)
            if mask_arr.max() > 1: mask_arr = (mask_arr > 128).astype(int)
            
            clip_imgs.append(self.img_transform(img))
            clip_masks.append(torch.from_numpy(mask_arr).long())

        # ---------------------------------------------------------
        # 核心修改：确保输出维度是 [B=1, N, 3, H, W]
        # ---------------------------------------------------------
        imgs_tensor = torch.stack(clip_imgs).unsqueeze(0)   # [1, N, 3, H, W]
        masks_tensor = torch.stack(clip_masks).unsqueeze(0) # [1, N, H, W]
        
        return imgs_tensor, masks_tensor

# ================= vis =================
def calculate_iou(pred, label, num_classes):
    pred = torch.argmax(pred, dim=1)
    iou_list = []
    # 如果 label 是 [1, N, H, W]，展平以匹配预测
    label = label.view(-1, label.shape[-2], label.shape[-1])
    for cls in range(num_classes):
        intersection = ((pred == cls) & (label == cls)).sum().item()
        union = ((pred == cls) | (label == cls)).sum().item()
        if union == 0: iou_list.append(float('nan'))
        else: iou_list.append(float(intersection) / float(union))
    return np.nanmean(iou_list)

# ================= train =================
def train():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiVideoDataset(CONFIG["video_dirs"], CONFIG["input_size"], CONFIG["seq_range"])
    # 注意：由于 Dataset 已经自带 B 维度，DataLoader 的 batch_size 必须固定为 1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)

    model = SegDA3(num_classes=CONFIG["num_classes"]).to(device)
    model.train()

    optimizer = optim.AdamW(model.seg_head.parameters(), lr=CONFIG["lr"], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(CONFIG["epochs"]):
        epoch_loss, epoch_iou = 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

        for imgs, masks in pbar:
            # 此时 imgs 的形状从 dataloader 出来是 [1, 1, N, 3, H, W] 
            # (因为 Dataset 出一个 B=1，DataLoader 又套了一个 B=1)
            # 所以我们需要去掉 DataLoader 套的那一层
            imgs = imgs.squeeze(0).to(device)   # [1, N, 3, H, W]
            masks = masks.squeeze(0).to(device) # [1, N, H, W]
            
            optimizer.zero_grad()

            with autocast():
                # model 接收 [B, N, 3, H, W] 并返回 [B*N, 2, H, W]
                logits = model(imgs) 
                
                # 计算 Loss 前，将 masks 展平为 [B*N, H, W]
                masks_flatten = masks.view(-1, masks.shape[-2], masks.shape[-1])
                loss = criterion(logits, masks_flatten)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            with torch.no_grad():
                batch_iou = calculate_iou(logits, masks, CONFIG["num_classes"])
                epoch_iou += batch_iou
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "mIoU": f"{batch_iou:.4f}"})

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dataloader):.4f}, mIoU: {epoch_iou/len(dataloader):.4f}")
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "model.pth"))

if __name__ == "__main__":
    train()