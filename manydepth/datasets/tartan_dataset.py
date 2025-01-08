import os
import torch
import numpy as np
import PIL.Image as pil
from torch.utils.data import Dataset
from manydepth.mono_dataset import MonoDataset  # 假设 Manydepth 的 MonoDataset 可以作为基础类


class TartanDriveDataset(MonoDataset):
    """TartanDrive dataset loader, modified to only use left images and depth data
    """

    def __init__(self, *args, **kwargs):
        super(TartanDriveDataset, self).__init__(*args, **kwargs)

        # 设置内参矩阵
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # 设置图像分辨率，根据实际数据集分辨率调整
        self.full_res_shape = (1242, 375)

    def check_depth(self):
        """
        检查深度图文件是否存在
        """
        for idx in range(len(self.filenames)):
            if not os.path.exists(self.filenames[idx]["depth"]):
                raise FileNotFoundError(f"Depth file {self.filenames[idx]['depth']} not found.")

    def load_depth(self, idx):
        """
        加载深度图
        """
        depth_path = self.filenames[idx]["depth"]
        depth_img = pil.open(depth_path)
        depth = np.array(depth_img, dtype=np.float32)
        return depth

    def load_image(self, idx, image_type="left"):
        """
        加载图像
        """
        image_path = self.filenames[idx][image_type]
        image = pil.open(image_path)
        image = np.array(image, dtype=np.float32)
        return image

    def get_intrinsics(self):
        """
        返回相机内参
        """
        return self.K

    def __len__(self):
        """
        返回数据集的样本数量
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        获取每个样本：图像、深度图、相机内参
        """
        # 加载图像和深度图
        left_image = self.load_image(idx, image_type="left")
        depth_map = self.load_depth(idx)

        # 调整大小和归一化
        left_image = skimage.transform.resize(left_image, self.full_res_shape)
        depth_map = skimage.transform.resize(depth_map, self.full_res_shape)

        # 转换为张量
        left_image = torch.tensor(left_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # 转换为 [C, H, W] 格式并归一化
        depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0)  # 添加深度维度

        # 获取相机内参矩阵
        intrinsics = torch.tensor(self.K, dtype=torch.float32)

        return {
            "image": left_image,
            "depth": depth_map,
            "intrinsics": intrinsics
        }
