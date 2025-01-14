import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 数据路径配置
image_dir = "/home/ubuntu22/下载/TarTan dataset/image_left_color"  # 左视图图像路径
depth_dir = "/home/ubuntu22/下载/TarTan dataset/depth"    # 深度图路径
calib_dir = "/home/ubuntu22/下载/TarTan dataset/calib"    # 相机内参路径

# 加载相机内参函数
def load_calibration(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    P = []
    for line in lines:
        P.extend([float(x) for x in line.split()])
    P = np.array(P).reshape(3, 4)  # 相机内参是 3x4 矩阵
    return P

# 可视化函数
def visualize_alignment(image_path, depth_path, calib_path):
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB

    # 加载深度图
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # 加载相机内参
    calib_matrix = load_calibration(calib_path)

    # 打印相机内参
    print(f"Camera Intrinsics Matrix:\n{calib_matrix}")

    # 可视化
    plt.figure(figsize=(12, 6))

    # 图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.axis("off")

    # 深度图
    plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap="viridis")
    plt.title("Depth Map")
    plt.colorbar(label="Depth (mm)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# 遍历并检查数据
image_files = sorted(os.listdir(image_dir))
depth_files = sorted(os.listdir(depth_dir))
calib_files = sorted(os.listdir(calib_dir))

for i in range(len(image_files)):
    print(f"Checking file {i}: {image_files[i]}")
    image_path = os.path.join(image_dir, image_files[i])
    depth_path = os.path.join(depth_dir, depth_files[i])
    calib_path = os.path.join(calib_dir, calib_files[i])

    visualize_alignment(image_path, depth_path, calib_path)

    # 提示继续或退出
    user_input = input("Press Enter to check next file, or type 'q' to quit: ")
    if user_input.lower() == 'q':
        break
