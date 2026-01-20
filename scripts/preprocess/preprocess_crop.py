import os
from PIL import Image

def crop_images_in_folder(input_folder, output_folder):
    """
    处理文件夹中的所有图片，裁剪中心偏上的正方形区域
    正方形边长为原图高度的9/10
    """
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # 处理所有图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            try:
                # 打开图片
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)
                width, height = img.size
                
                # # 计算正方形边长：原图高度的n/10
                # #C3VD2
                ratio = 0.65#
                square_size = int(height * ratio)
                left = (width - square_size) // 2 -30  # 向左偏移 

                # 计算正方形边长：原图高度的n/10
                # UKA1
                # ratio = 0.8
                # square_size = int(height * ratio)
                # left = (width - square_size) // 2 +20

                
                # 垂直方向：向上偏移5%的高度，使得裁剪区域稍微偏上
                # 这样上方保留5%的高度，下方保留5%的高度
                top = int(height * (1-ratio)/2)
                
                right = left + square_size
                bottom = top + square_size
                
                # 确保裁剪区域不超出图像边界
                if left < 0:
                    left = 0
                    right = square_size
                if top < 0:
                    top = 0
                    bottom = square_size
                if right > width:
                    right = width
                    left = width - square_size
                if bottom > height:
                    bottom = height
                    top = height - square_size
                
                # 裁剪并保存
                cropped_img = img.crop((left, top, right, bottom))
                output_path = os.path.join(output_folder, f"{filename}")
                cropped_img.save(output_path)
                
                print(f"已处理: {filename} -> {square_size}x{square_size}")
                
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")
    
    print(f"\n处理完成! 结果保存在: {output_folder}")

# 使用示例
if __name__ == "__main__":
    # 在这里指定你的文件夹路径
    input_folder = "/home/zhouyi/repo/dataset/C3VD2/C3VD2"
    ouput_folder = "/home/zhouyi/repo/dataset/C3VD2/C3VD2_cropped"
    
    # 调用处理函数
    crop_images_in_folder(input_folder,ouput_folder)