import os
import cv2

def frames_to_video(input_folder, output_video_path, fps=30):
    """
    将文件夹下的图片序列合成视频
    """
    # 获取所有jpg图片并排序
    images = sorted([img for img in os.listdir(input_folder) if img.endswith(".jpg")])
    if not images:
        print("未发现图片")
        return

    # 读取第一张图获取分辨率
    first_frame = cv2.imread(os.path.join(input_folder, images[0]))
    h, w, _ = first_frame.shape

    # 初始化视频写入器 (使用 mp4v 编码)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print(f"开始合成: {output_video_path}")
    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()
    print("合成完毕")

if __name__ == "__main__":
    input_dir = "/home/zhouyi/repo/dataset/C3VD2/C3VD2_cropped" 
    output_path = "/home/zhouyi/repo/dataset/C3VD2/C3VD2.mp4"

    # 如果目录不存在则创建
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames_to_video(input_dir, output_path, fps=30)