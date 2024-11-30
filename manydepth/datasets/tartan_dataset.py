import os
import numpy as np
import PIL.Image as pil
import skimage.transform

from manydepth.mono_dataset import MonoDataset  # 假设 Manydepth 的 MonoDataset 可以作为基础类

class TartanDriveDataset(MonoDataset):
    """TartanDrive dataset loader, modified to only use left images and depth data
    """
    def __init__(self, *args, **kwargs):
        super(TartanDriveDataset, self).__init__(*args, **kwargs)

        # 内参矩阵，可以根据实际情况调整
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)  # 根据 TartanDrive 数据集的图像分辨率调整

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
        """Convert index in the dataset to a folder name and frame_idx
        """
        line = self.filenames[index].split()
        folder = line[0]

        frame_index = int(line[1]) if len(line) >= 2 else 0
        return folder, frame_index

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
        odom_path = os.path.join(self.data_path, folder, "gps_odom", "odometry.npy")
        odom_data = np.load(odom_path)
        pose = odom_data[frame_index]  # 假设 odometry.npy 中包含每一帧的位姿信息
        return pose