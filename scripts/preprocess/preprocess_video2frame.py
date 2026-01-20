import os
import cv2
from PIL import Image

def video_to_frames(input_folder, output_folder, frame_interval=1):
    """
    简化版视频转帧函数
    
    Args:
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
        frame_interval (int): 帧间隔，每隔几帧提取一帧
    """
    os.makedirs(output_folder, exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        if not os.path.isfile(file_path):
            continue
            
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in video_extensions:
            continue
            
        print(f"处理: {filename}")
        
        # 创建子文件夹
        video_name = os.path.splitext(filename)[0]
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)
        
        # 使用OpenCV读取视频
        cap = cv2.VideoCapture(file_path)
        frame_idx = 0
        saved_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # 转换并保存
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frame_filename = f"{saved_idx:06d}.jpg"
                pil_image.save(os.path.join(video_output_folder, frame_filename))
                saved_idx += 1
            
            frame_idx += 1
        
        cap.release()
        print(f"保存了 {saved_idx} 帧")

# 使用示例
if __name__ == "__main__":
    input_folder = "/Users/guanzhouyi/repos/MA/DA3-SLAM/dataset/2077"
    output_folder = "/Users/guanzhouyi/repos/MA/DA3-SLAM/dataset/2077"
    
    video_to_frames(
        input_folder=input_folder,
        output_folder=output_folder,
        frame_interval=30  # 每隔10帧提取一帧
    )