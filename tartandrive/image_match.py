import numpy as np

# 读取 IMU 数据和时间戳
imu_data = np.load("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/multisense_imu/imu.npy")  # IMU 数据文件
imu_timestamps = np.loadtxt("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/multisense_imu/timestamps.txt")  # IMU 时间戳文件

# 读取图像时间戳
image_timestamps = np.loadtxt("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/image_left/timestamps.txt")  # 图像时间戳文件

# 找到最近的 IMU 时间戳索引
matched_imu_indices = [np.argmin(np.abs(imu_timestamps - t)) for t in image_timestamps]

# 提取匹配好的数据
matched_imu_data = imu_data[matched_imu_indices]            # 匹配的 IMU 数据
matched_imu_timestamps = imu_timestamps[matched_imu_indices]  # 匹配的 IMU 时间戳

# 保存包含时间戳的文件
output_with_timestamps = "./matched_IMU_with_timestamps.txt"
with open(output_with_timestamps, mode="w") as file:
    file.write("Image Timestamp, IMU Timestamp, IMU Data\n")
    for img_t, imu_t, imu_row in zip(image_timestamps, matched_imu_timestamps, matched_imu_data):
        file.write(f"{img_t:.6f}, {imu_t:.6f}, {', '.join(map(str, imu_row))}\n")
print(f"文件已保存（包含时间戳）: {output_with_timestamps}")

# 保存只有 IMU 数据的文件
output_only_imu = "./matched_imu_only.txt"
np.savetxt(output_only_imu, matched_imu_data, fmt="%.6f", delimiter=",", header="IMU Data", comments='')
print(f"文件已保存（只有 IMU 数据）: {output_only_imu}")
# 打印匹配的 IMU 数据的形状
print(f"Matched IMU data shape: {matched_imu_data.shape}")



# 加载 SuperOdom 数据和时间戳
superodom_data = np.load("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/super_odom/odometry.npy")
superodom_timestamps = np.loadtxt("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/super_odom/timestamps.txt")  # SuperOdom 时间戳

# 找到每个图像时间戳对应的最近 SuperOdom 数据索引
matched_indices = [np.argmin(np.abs(superodom_timestamps - t)) for t in image_timestamps]

# 提取匹配的 SuperOdom 数据
matched_superodom_data = superodom_data[matched_indices]  # 匹配的 SuperOdom 数据
matched_superodom_timestamps = superodom_timestamps[matched_indices]  # 匹配的 SuperOdom 时间戳

# 保存包含时间戳的文件
output_with_timestamps = "./matched_superodom_with_timestamps.txt"  # 替换为你的输出路径
with open(output_with_timestamps, mode="w") as file:
    file.write("Image Timestamp, SuperOdom Timestamp, SuperOdom Data\n")
    for img_t, odom_t, odom_row in zip(image_timestamps, matched_superodom_timestamps, matched_superodom_data):
        file.write(f"{img_t:.6f}, {odom_t:.6f}, {', '.join(map(str, odom_row))}\n")
print(f"文件已保存（包含时间戳）: {output_with_timestamps}")

# 保存只有 SuperOdom 数据的文件
output_only_superodom = "./matched_superodom_only.txt"  # 替换为你的输出路径
np.savetxt(output_only_superodom, matched_superodom_data, fmt="%.6f", delimiter=",", header="SuperOdom Data", comments='')
print(f"文件已保存（只有 SuperOdom 数据）: {output_only_superodom}")
# 打印匹配的 SuperOdom 数据的形状
print(f"Matched SuperOdom data shape: {matched_superodom_data.shape}")


from scipy.spatial.transform import Rotation as R


def superodom_to_transformation_matrix(superodom_data):
    """
    将 SuperOdom 数据转换为 4x4 Transformation Matrix
    :param superodom_data: 提取的 SuperOdom 数据，每行包含 (x, y, z, qx, qy, qz, qw)
    :return: 4x4 Transformation Matrices 数组
    """
    transformation_matrices = []
    for row in superodom_data:
        # 提取位置和四元数
        x, y, z = row[:3]
        qx, qy, qz, qw = row[3:7]

        # 将四元数转换为旋转矩阵
        rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

        # 构建 4x4 Transformation Matrix
        T = np.eye(4)
        T[:3, :3] = rotation_matrix  # 设置旋转部分
        T[:3, 3] = [x, y, z]  # 设置平移部分

        transformation_matrices.append(T)

    return np.array(transformation_matrices)


# 假设 matched_superodom_data 已提取完成
transformation_matrices = superodom_to_transformation_matrix(matched_superodom_data)

# 打印第一个变换矩阵
print(f"第一个 Transformation Matrix:\n{transformation_matrices[0]}")

# 旋转矩阵验证
R_matrix = transformation_matrices[0][:3, :3]
print("R^T * R =\n", R_matrix.T @ R_matrix)
# 如果结果接近单位矩阵和行列式为 1，则旋转矩阵有效。
print("Determinant of R =", np.linalg.det(R_matrix))

# 数据变换矩阵
output_pose_file = "./transformation_matrices.txt"
with open(output_pose_file, "w") as f:
    for i, T in enumerate(transformation_matrices):
        f.write(f"Pose {i}:\n")
        np.savetxt(f, T, fmt="%.6f")
        f.write("\n")
print(f"Transformation Matrices 已保存到: {output_pose_file}")
# 打印匹配的 SuperOdom 数据的形状
print(f"Transformation Matrices shape: {transformation_matrices.shape}")