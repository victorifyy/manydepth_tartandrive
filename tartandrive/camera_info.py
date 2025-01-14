import os

output_dir = "dataset/calib"
os.makedirs(output_dir, exist_ok=True)

# 从 /multisense/left/camera_info 中提取的参数
K = [455.77496337890625, 0.0, 497.1180114746094,
     0.0, 456.319091796875, 251.58502197265625,
     0.0, 0.0, 1.0]

# 生成内参文件
NUM_IMAGES = 1624  # 替换为你的图像数量
for i in range(NUM_IMAGES):
    with open(os.path.join(output_dir, f"{i:06d}.txt"), "w") as f:
        for j in range(0, len(K), 3):
            f.write(" ".join(map(str, K[j:j+3])) + " 0.0\n")
