import os
import sam3
import torch
import time
from sam3.model_builder import build_sam3_video_predictor

import glob
import cv2
import numpy as np
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
)
from tqdm import tqdm
# ==============================================================================
# static settings
# ==============================================================================
# video_path = "/home/zhouyi/repo/dataset/2077/scene1" 
# video_path = "/home/zhouyi/repo/model_sam3/assets/videos/dancer"
video_path = "/home/zhouyi/repo/dataset/UKA1/Case1Part1_1cropped"

output_dir = "/home/zhouyi/repo/model_DepthAnythingV3/inputs"
# text_prompt = "metal" 

# ==============================================================================
# setup
# ==============================================================================
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
# use all available GPUs on the machine
gpus_to_use = range(torch.cuda.device_count())
# # use only a single GPU
# gpus_to_use = [torch.cuda.current_device()]

predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

start_time = time.time()

video_path_name = os.path.basename(video_path)
output_dir = os.path.join(output_dir, video_path_name)
os.makedirs(output_dir, exist_ok=True)
output_image_dir = os.path.join(output_dir, "images")
os.makedirs(output_image_dir, exist_ok=True)
output_mask_dir = os.path.join(output_dir, "masks")
os.makedirs(output_mask_dir, exist_ok=True)
# ==============================================================================
# helper functions
# ==============================================================================
def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

# ==============================================================================
# load video/images
# ==============================================================================
# load "video_frames_for_vis" for visualization purposes (they are not used by the model)
if isinstance(video_path, str) and video_path.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    try:
        # integer sort instead of string sort (so that e.g. "2.jpg" is before "11.jpg")
        video_frames_for_vis.sort(
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
    except ValueError:
        # fallback to lexicographic sort if the format is not "<frame_index>.jpg"
        print(
            f'frame names are not in "<frame_index>.jpg" format: {video_frames_for_vis[:5]=}, '
            f"falling back to lexicographic sort."
        )
        video_frames_for_vis.sort()

# ==============================================================================
# Opening an inference session on this video
# ==============================================================================
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]

# ==============================================================================
# Video promptable concept segmentation with text

# Using SAM 3 you can describe objects using natural language, and the model will automatically detect and track all instances of that object throughout the video.

# In the example below, we add a text prompt on frame 0 and propagation throughout the video. Here we use the text prompt "person" to detect all people in the video. SAM 3 will automatically identify multiple person instances and assign each a unique object ID.

# Note that the first call might be slower due to setting up buffers. 
# ==============================================================================
# note: in case you already ran one text prompt and now want to switch to another text prompt
# it's required to reset the session first (otherwise the results would be wrong)
_ = predictor.handle_request(
    request=dict(
        type="reset_session",
        session_id=session_id,
    )
)

frame_idx = 0  # add a text prompt on frame 0
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=text_prompt,
    )
)
out = response["outputs"]

# now we propagate the outputs from frame 0 to the end of the video and collect all outputs
outputs_per_frame = propagate_in_video(predictor, session_id)

# finally, we reformat the outputs for visualization and plot the outputs
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

# ==============================================================================
# save outputs
# ==============================================================================
sorted_frame_indices = sorted(outputs_per_frame.keys())

for frame_idx in tqdm(sorted_frame_indices):
    if frame_idx >= len(video_frames_for_vis):
        print(f"Warning: Frame index {frame_idx} out of bounds for video_frames_for_vis")
        continue
        
    frame_data = video_frames_for_vis[frame_idx]
    
    # 处理图像数据，统一转为 BGR (因为 cv2.imwrite 需要 BGR)
    if isinstance(frame_data, str):
        # 如果是文件路径，读取它
        frame_bgr = cv2.imread(frame_data)
    elif isinstance(frame_data, np.ndarray):
        # 如果已经是数组，需要转为 BGR
        frame_bgr = cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR)
    else:
        continue
        
    if frame_bgr is None:
        continue

    h, w = frame_bgr.shape[:2]

    # 2. 合并当前帧的所有 Mask
    # 获取该帧下所有对象的 mask 字典: {obj_id: mask}
    masks_dict = outputs_per_frame[frame_idx]
    
    # 创建一个全黑的底图
    combined_mask = np.zeros((h, w), dtype=bool)
    
    # 遍历所有检测到的物体并合并
    for obj_id, mask in masks_dict.items():
        # 如果是 Tensor，转为 Numpy
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        
        # 确保维度匹配 (H, W)
        if mask.ndim > 2:
            mask = mask.squeeze()
            
        # 逻辑或合并 (只要有任意一个物体在该像素，就设为 True)
        combined_mask = np.logical_or(combined_mask, mask > 0.5)

    # 将布尔值转换为 0-255 的图像
    final_mask_img = (combined_mask.astype(np.uint8)) * 255
    
    # 3. 保存文件
    # 构造文件名 (使用帧索引，保证顺序)
    filename_base = f"{frame_idx:05d}"
    
    # 保存原图
    cv2.imwrite(os.path.join(output_image_dir, f"{filename_base}.png"), frame_bgr)
    
    # 保存 Mask
    cv2.imwrite(os.path.join(output_mask_dir, f"{filename_base}.png"), final_mask_img)

print("Processing completed. Outputs saved to:", output_dir)

# ==============================================================================
# time
# ==============================================================================
end_time = time.time()
elapsed_time = end_time - start_time
print(f"used time: {elapsed_time:.2f} s")


