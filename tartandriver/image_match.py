import numpy as np
import csv

# 读取 IMU 数据和时间戳
imu_data = np.load("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/multisense_imu/imu.npy")  # IMU 数据文件
imu_timestamps = np.loadtxt("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/multisense_imu/timestamps.txt")  # IMU 时间戳文件

# 读取图像时间戳
image_timestamps = np.loadtxt("/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/image_left/timestamps.txt")  # 图像时间戳文件

# 找到最近的 IMU 时间戳
matched_imu_indices = [np.argmin(np.abs(imu_timestamps - t)) for t in image_timestamps]
matched_imu_data = imu_data[matched_imu_indices]  # 提取对应的 IMU 数据

# 输出匹配结果
output_file = "/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/code/multisense_imu/matched_results.txt"
with open(output_file, mode="w") as file:
    file.write("Image Timestamp, Matched IMU Timestamp, IMU Data\n")
    for img_t, imu_t, imu_row in zip(image_timestamps, imu_timestamps[matched_imu_indices], matched_imu_data):
        file.write(f"{img_t:.6f}, {imu_t:.6f}, {', '.join(map(str, imu_row))}\n")

print(f"匹配结果已保存到文件: {output_file}")

def preprocess_imu(matched_imu_data, gravity=9.81):
    """
    校正 IMU 数据中的重力，拆分加速度和角速度
    """
    acc_data = matched_imu_data[:, :3]  # 加速度 (ax, ay, az)
    acc_data[:, 2] -= gravity           # 去除重力影响
    gyro_data = matched_imu_data[:, 3:]  # 角速度 (wx, wy, wz)
    return acc_data, gyro_data

from scipy.spatial.transform import Rotation as R

def compute_orientations(gyro_data, imu_timestamps):
    """
    从角速度计算旋转矩阵
    """
    orientations = []
    R_t = np.eye(3)  # 初始旋转矩阵
    orientations.append(R_t)

    for i in range(1, len(gyro_data)):
        dt = imu_timestamps[i] - imu_timestamps[i - 1]  # 时间间隔
        omega = gyro_data[i]
        omega_norm = np.linalg.norm(omega)

        if omega_norm > 0:
            axis = omega / omega_norm
            angle = omega_norm * dt
            delta_R = R.from_rotvec(angle * axis).as_matrix()  # 旋转矩阵
        else:
            delta_R = np.eye(3)  # 无旋转

        R_t = R_t @ delta_R
        orientations.append(R_t)

    return np.array(orientations)

def compute_positions(acc_data, imu_timestamps):
    """
    从加速度计算位置
    """
    positions = [np.zeros(3)]  # 初始位置
    velocities = [np.zeros(3)]  # 初始速度

    for i in range(1, len(acc_data)):
        dt = imu_timestamps[i] - imu_timestamps[i - 1]  # 时间间隔
        # 更新速度
        v_t = velocities[-1] + acc_data[i] * dt
        velocities.append(v_t)
        # 更新位置
        p_t = positions[-1] + v_t * dt
        positions.append(p_t)

    return np.array(positions)

def compute_poses(orientations, positions):
    """
    生成 4x4 位姿矩阵
    """
    poses = []
    for i in range(len(positions)):
        T = np.eye(4)
        T[:3, :3] = orientations[i]  # 旋转矩阵
        T[:3, 3] = positions[i]      # 平移向量
        poses.append(T)
    return np.array(poses)

# 提取加速度和角速度
acc_data, gyro_data = preprocess_imu(matched_imu_data)

# 计算旋转矩阵
orientations = compute_orientations(gyro_data, imu_timestamps[matched_imu_indices])

# 计算位置
positions = compute_positions(acc_data, imu_timestamps[matched_imu_indices])

# 组合位姿矩阵
poses = compute_poses(orientations, positions)

# 输出结果
print(f"第一个位姿矩阵:\n{poses[0]}")

import matplotlib.pyplot as plt

# 提取平面位置
positions_2d = positions[:, :2]
plt.plot(positions_2d[:, 0], positions_2d[:, 1], label="Trajectory")
plt.title("Device Trajectory")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.show()

output_pose_file = "/Users/fengyingying/Downloads/2023-11-14-14-45-20_gupta/code/multisense_imu/poses.txt"

with open(output_pose_file, "w") as f:
    for i, T in enumerate(poses):
        f.write(f"Pose {i}:\n")
        np.savetxt(f, T, fmt="%.6f")
        f.write("\n")

print(f"位姿矩阵已保存到文件: {output_pose_file}")