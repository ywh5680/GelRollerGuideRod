import cv2
import os
import numpy as np

# 路径设置
input_folder = 'F:/yxw/subtrat'         # 输入图片文件夹
reference_image_path = 'F:/yxw/subtrat/ref.jpg' # 参考图像路径
output_folder = 'F:/yxw/subtrat/output'        # 输出结果保存文件夹

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 加载参考图像
ref_img = cv2.imread(reference_image_path)

if ref_img is None:
    raise ValueError(f"无法加载参考图像：{reference_image_path}")

# 获取输入文件夹中的所有图像文件名
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

for img_file in image_files:
    img_path = os.path.join(input_folder, img_file)
    img = cv2.imread(img_path)

    if img is None:
        print(f"跳过无法读取的图像：{img_file}")
        continue

    # 确保参考图像和当前图像大小一致
    if img.shape != ref_img.shape:
        print(f"跳过尺寸不匹配的图像：{img_file}")
        continue

    # 图像减法（确保结果在可视范围内）
    result = cv2.subtract(img, ref_img)

    # 保存结果
    output_path = os.path.join(output_folder, img_file)
    cv2.imwrite(output_path, result*10)
    print(f"已保存：{output_path}")
