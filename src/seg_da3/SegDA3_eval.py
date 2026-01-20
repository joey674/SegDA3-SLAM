import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from SegDA3_model import SegDA3
from depth_anything_3.utils.visualize import visualize_depth


# ================= config =================
# UKA
# IMG_PATHS = [
#     "/home/zhouyi/repo/dataset/UKA1/Case1Part1_1cropped/cropped_000956.jpg",
#     "/home/zhouyi/repo/dataset/UKA1/Case1Part1_1cropped/cropped_000957.jpg",
#     "/home/zhouyi/repo/dataset/UKA1/Case1Part1_1cropped/cropped_000958.jpg",
#     "/home/zhouyi/repo/dataset/UKA1/Case1Part1_1cropped/cropped_000959.jpg",
#     "/home/zhouyi/repo/dataset/UKA1/Case1Part1_1cropped/cropped_000960.jpg",
# ]
# 2077scene1
IMG_PATHS = [ 
    "/home/zhouyi/repo/model_DepthAnythingV3/inputs/2077scene1/images/00000.png",
    "/home/zhouyi/repo/model_DepthAnythingV3/inputs/2077scene1/images/00001.png",
    "/home/zhouyi/repo/model_DepthAnythingV3/inputs/2077scene1/images/00002.png",
    "/home/zhouyi/repo/model_DepthAnythingV3/inputs/2077scene1/images/00003.png",
    "/home/zhouyi/repo/model_DepthAnythingV3/inputs/2077scene1/images/00004.png",
    "/home/zhouyi/repo/model_DepthAnythingV3/inputs/2077scene1/images/00005.png",
    "/home/zhouyi/repo/model_DepthAnythingV3/inputs/2077scene1/images/00006.png",
    "/home/zhouyi/repo/model_DepthAnythingV3/inputs/2077scene1/images/00007.png",
    "/home/zhouyi/repo/model_DepthAnythingV3/inputs/2077scene1/images/00008.png",
]


SAVE_PATH = "/home/zhouyi/repo/VGGT-SLAM/outputs/segda3"
ckpt_path = "/home/zhouyi/repo/model_DepthAnythingV3/checkpoints/SegDA3/model.pth"
# ===========================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 加载模型（新 SegDA3 内部自己 from_pretrained + 冻结 DA3）
    print("Loading model...")
    model = SegDA3(seg_head_ckpt_path=ckpt_path
                    ).to(device)

    # 2) 推理（主体走 API inference：depth/processed_images/aux 都是 API 原样）
    print("Running inference...")
    pred = model.inference(image=IMG_PATHS)

    # 3) 打印关键信息，确认 DA3 输出是否正常
    print("== DA3 OUTPUT CHECK ==")
    print("processed_images:",
          type(pred.processed_images),
          getattr(pred.processed_images, "shape", None),
          getattr(pred.processed_images, "dtype", None))
    print("depth:",
          type(pred.depth),
          getattr(pred.depth, "shape", None),
          getattr(pred.depth, "dtype", None))
    print("aux keys:",
          list(pred.aux.keys()) if hasattr(pred, "aux") and isinstance(pred.aux, dict) else None)
    print("======================")

    # 4) 准备数据（注意：API inference 输出 depth 是 numpy，不要 .cpu().numpy()）
    images = pred.processed_images  # numpy uint8, [N,H,W,3] (通常是这个)
    if images is None:
        raise RuntimeError("pred.processed_images is None")
    if images.ndim == 4 and images.shape[1] == 3:
        # 兼容极少数情况下返回 [N,3,H,W]
        images = images.transpose(0, 2, 3, 1)

    depths = pred.depth  # numpy [N,H,W]
    if not isinstance(depths, np.ndarray):
        raise RuntimeError(f"pred.depth is not numpy: {type(depths)}")

    # motion mask 是 torch（来自你的 motion head）
    masks = pred.motion_seg_mask.detach().cpu().numpy()  # [N,H,W]

    # 5) 绘图（3 行 x N 列）
    print("Plotting...")
    num_imgs = len(IMG_PATHS)
    fig, axes = plt.subplots(3, num_imgs, figsize=(3 * num_imgs, 9))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(num_imgs):
        # 第一行：原图
        axes[0, i].imshow(images[i])
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Input RGB", fontsize=12, loc="left")

        # 第二行：深度图
        depth_vis = visualize_depth(depths[i], cmap="Spectral")
        axes[1, i].imshow(depth_vis)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Depth Prediction (DA3 API)", fontsize=12, loc="left")

        # 第三行：运动分割 Mask
        axes[2, i].imshow(masks[i], cmap="jet", interpolation="nearest", vmin=0, vmax=1)
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_title("Motion Mask (Extra Head)", fontsize=12, loc="left")

    # 6) 保存
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, bbox_inches="tight", dpi=150)
    print(f"Result saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
