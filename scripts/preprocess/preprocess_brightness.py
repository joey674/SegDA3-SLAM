import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def adjust_image_brightness(image, 
                           bright_threshold=230,  # 过亮阈值
                           dark_threshold=30,     # 过暗阈值
                           bright_reduction=0.7,  # 过亮区域调暗系数
                           dark_enhancement=1.5,  # 过暗区域提亮系数
                           clip_limit=2.0,        # CLAHE的对比度限制
                           grid_size=8):          # CLAHE的网格大小
    """
    调整图像亮度，使过亮区域变暗，过暗区域提亮
    
    Args:
        image: 输入图像 (BGR格式)
        bright_threshold: 过亮阈值 (0-255)
        dark_threshold: 过暗阈值 (0-255)
        bright_reduction: 过亮区域调暗系数
        dark_enhancement: 过暗区域提亮系数
        clip_limit: CLAHE对比度限制
        grid_size: CLAHE网格大小
        
    Returns:
        调整后的图像
    """
    # 转换为LAB颜色空间，分离亮度通道 （把RGB转换成LAB，L指的是亮度；这样就可以对亮度操作了）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 方法1: 使用CLAHE(对比度受限的自适应直方图均衡化)增强暗部
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    l_clahe = clahe.apply(l_channel)
    
    # 方法2: 检测并调整过亮区域
    # 创建过亮区域的掩码
    bright_mask = l_channel > bright_threshold
    
    # 如果有过亮区域，将其调暗
    if np.any(bright_mask):
        # 对过亮区域进行调暗
        l_adjusted = l_channel.copy().astype(np.float32)
        l_adjusted[bright_mask] = l_adjusted[bright_mask] * bright_reduction
        l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)
        
        # 将CLAHE结果和调整结果融合
        # 对过亮区域使用调整后的亮度，其他区域使用CLAHE增强的结果
        l_result = l_clahe.copy()
        l_result[bright_mask] = l_adjusted[bright_mask]
    else:
        l_result = l_clahe
    
    # 方法3: 增强过暗区域的细节
    # 检测过暗区域
    dark_mask = l_channel < dark_threshold
    
    if np.any(dark_mask):
        # 对过暗区域进行提亮
        l_result_f = l_result.copy().astype(np.float32)
        l_result_f[dark_mask] = l_result_f[dark_mask] * dark_enhancement
        l_result = np.clip(l_result_f, 0, 255).astype(np.uint8)
    
    # 方法4: 使用伽马校正进行全局亮度调整
    # 计算图像平均亮度
    mean_brightness = np.mean(l_result)
    
    # 根据平均亮度调整伽马值
    if mean_brightness < 100:  # 图像偏暗
        gamma = 0.8  # 提亮
    elif mean_brightness > 150:  # 图像偏亮
        gamma = 1.2  # 压暗
    else:  # 亮度适中
        gamma = 1.0
    
    if gamma != 1.0:
        # 应用伽马校正
        l_result_f = l_result.copy().astype(np.float32) / 255.0
        l_result_f = np.power(l_result_f, gamma)
        l_result = (l_result_f * 255).astype(np.uint8)
    
    # 方法5: 使用局部对比度增强
    # 创建局部对比度增强的核
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # 应用局部对比度增强
    l_enhanced = cv2.filter2D(l_result, -1, kernel)
    
    # 将增强的细节融合到原亮度通道
    alpha = 0.3  # 融合权重
    l_final = cv2.addWeighted(l_result, 1 - alpha, l_enhanced, alpha, 0)
    
    # 将处理后的亮度通道与原始的a、b通道合并
    lab_adjusted = cv2.merge([l_final, a_channel, b_channel])
    
    # 转换回BGR颜色空间
    result = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)
    
    return result

def adjust_brightness_in_folder(input_folder, output_folder, 
                               bright_threshold=230,
                               dark_threshold=30,
                               bright_reduction=0.7,
                               dark_enhancement=1.5,
                               clip_limit=2.0,
                               grid_size=8):
    """
    批量处理文件夹中的图像，调整亮度均匀性
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        bright_threshold: 过亮阈值 (0-255)
        dark_threshold: 过暗阈值 (0-255)
        bright_reduction: 过亮区域调暗系数
        dark_enhancement: 过暗区域提亮系数
        clip_limit: CLAHE对比度限制
        grid_size: CLAHE网格大小
    """
    # 创建输出文件夹
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP'}
    image_files = []
    
    input_path = Path(input_folder)
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'**/*{ext}'))
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 处理每张图像
    for img_path in tqdm(image_files, desc="处理图像"):
        try:
            # 读取图像
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"无法读取图像: {img_path}")
                continue
            
            # 调整亮度
            adjusted_img = adjust_image_brightness(
                img, 
                bright_threshold=bright_threshold,
                dark_threshold=dark_threshold,
                bright_reduction=bright_reduction,
                dark_enhancement=dark_enhancement,
                clip_limit=clip_limit,
                grid_size=grid_size
            )
            
            # 保存图像
            # 保持相对路径结构
            rel_path = img_path.relative_to(input_path)
            output_file = output_path / rel_path
            
            # 确保输出目录存在
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存图像
            cv2.imwrite(str(output_file), adjusted_img)
            
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
    
    print(f"处理完成! 结果已保存到: {output_folder}")

# 使用示例
if __name__ == "__main__":
    # 批量处理整个文件夹
    input_folder = "/home/zhouyi/repo/dataset/C3VD2/C3VD2_cropped"
    output_folder = "/home/zhouyi/repo/dataset/C3VD2/C3VD2_cropped_brightness"
    
    if os.path.exists(input_folder):
        print("\n批量处理文件夹:")
        adjust_brightness_in_folder(
            input_folder, 
            output_folder,
            bright_threshold=230,    # 过亮阈值
            dark_threshold=30,       # 过暗阈值
            bright_reduction=0.7,    # 过亮区域调暗系数
            dark_enhancement=1.5,    # 过暗区域提亮系数
            clip_limit=2.0,          # CLAHE对比度限制
            grid_size=8              # CLAHE网格大小
        )
    else:
        print(f"输入文件夹不存在: {input_folder}")
        print("请修改为正确的路径")