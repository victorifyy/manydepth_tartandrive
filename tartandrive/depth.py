import os
import numpy as np
import cv2

input_dir = "/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/depth_left"
output_dir = "dataset/depth"
os.makedirs(output_dir, exist_ok=True)

for file_name in os.listdir(input_dir):
    if file_name.endswith(".npy"):
        # 加载 .npy 文件
        depth = np.load(os.path.join(input_dir, file_name))

        # 转换深度值
        depth_mm = (np.abs(depth) * 1000).astype(np.uint16)

        # 保存为 .png 文件
        output_file = os.path.splitext(file_name)[0] + ".png"
        cv2.imwrite(os.path.join(output_dir, output_file), depth_mm)

