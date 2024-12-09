import os
import numpy as np
import PIL.Image as pil
import skimage.transform

from manydepth.datasets.mono_dataset import MonoDataset  # 假设 Manydepth 的 MonoDataset 可以作为基础类

class TartanDriveDataset(MonoDataset):
    """TartanDrive dataset loader, modified to only use left images and depth data
    """
    def __init__(self, *args, **kwargs):
        super(TartanDriveDataset, self).__init__(*args, **kwargs)

        # 内参矩阵，在multisense_intrinsics.txt文件中
        self.K = np.array([
            [455.77496337890625, 0.0, 497.1180114746094, 0.0],
            [0.0, 456.319091796875, 251.58502197265625, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self.full_res_shape = (1024, 544)  # 对应文件中的高度和宽度

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        # 检查是否存在 `.npy` 格式的深度文件
        depth_filename = os.path.join(
            self.data_path,
            scene_name,
            "depth_left/{:06d}.npy".format(int(frame_index)))

        return os.path.isfile(depth_filename)

    def index_to_folder_and_frame_idx(self, index):
        """
        Convert dataset index to folder, frame index, and side.
        Returns:
            folder (str): Scene name (directory name)
            frame_index (int): Index of the frame
            side (str): Camera side ('l' for left, 'r' for right)
        """
        # 假设文件列表每行格式为: "scene_name frame_index"
        line = self.filenames[index].split()  # 分割文件列表的行
        folder = line[0]  # 第一列是场景名称
        frame_index = int(line[1])  # 第二列是帧索引
        side = "l"  # 默认设置为左相机
        return folder, frame_index, side

    def get_image_path(self, folder, frame_index):
        # 假设图像存储在 `image_left_color` 文件夹中
        f_str = "{:06d}.png".format(frame_index)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_left_color",
            f_str)
        return image_path

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, do_flip):
        # 从 `.npy` 格式加载深度数据
        depth_filename = os.path.join(
            self.data_path,
            folder,
            "depth_left/{:06d}.npy".format(int(frame_index)))

        depth_gt = np.load(depth_filename)  # 加载深度数据

        # 调整深度图分辨率以匹配图像的原始分辨率
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_pose(self, folder, frame_index):
        """Load the pose information from the odometry file."""
        odom_path = os.path.join(self.data_path, folder, "matched_super_odom.txt")

        # 读取文件并解析位姿数据
        with open(odom_path, 'r') as f:
            lines = f.readlines()

        # 假设 frame_index 对应文件的行号（忽略文件的标题行）
        pose_data = lines[frame_index + 1].strip().split(", ")

        # 提取平移 (x, y, z) 和旋转四元数 (qx, qy, qz, qw)
        position = np.array([float(pose_data[2]), float(pose_data[3]), float(pose_data[4])])
        orientation = np.array([float(pose_data[5]), float(pose_data[6]), float(pose_data[7]), float(pose_data[8])])

        # 返回平移和旋转数据
        return position, orientation