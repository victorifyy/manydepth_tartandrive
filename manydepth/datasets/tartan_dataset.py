import os
import torch
import numpy as np
import torch.nn.functional as F
import PIL.Image as pil
from torch.utils.data import Dataset
from .mono_dataset import MonoDataset  # 假设 Manydepth 的 MonoDataset 可以作为基础类


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
        检查深度图文件是否存在，输出缺失文件的路径
        """
        for idx in range(len(self.filenames)):
            depth_path = self.filenames[idx]["depth"]
            if not os.path.exists(depth_path):
                print(f"Missing depth file: {depth_path}")

    def load_depth(self, idx):
        """
        加载深度图，支持 .npy 和 .jpg 格式
        """
        depth_path = self.filenames[idx]["depth"]

        # 根据文件后缀处理不同格式
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)  # 加载 .npy 文件
        elif depth_path.endswith('.jpg'):
            depth_img = pil.open(depth_path)  # 加载 .jpg 文件
            depth = np.array(depth_img, dtype=np.float32)

            # 如果深度值是归一化的，需要还原到实际值范围
            # 根据实际情况修改 min_depth 和 max_depth
            min_depth, max_depth = 0.1, 100.0
            depth = depth / 255.0 * (max_depth - min_depth) + min_depth
        else:
            raise ValueError(f"Unsupported depth file format: {depth_path}")

        return depth

    def load_image(self, idx, image_type="left"):
        """
        加载图像并处理异常
        """
        try:
            image_path = self.filenames[idx][image_type]
            image = pil.open(image_path)
            image = np.array(image, dtype=np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

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

        # 转换为张量
        left_image = torch.tensor(left_image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # 转换为 [C, H, W] 格式并归一化
        depth_map = torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0)  # 添加深度维度

        # 调整大小到 full_res_shape
        left_image = F.interpolate(left_image.unsqueeze(0), size=self.full_res_shape, mode='bilinear',
                                   align_corners=True).squeeze(0)
        depth_map = F.interpolate(depth_map.unsqueeze(0), size=self.full_res_shape, mode='nearest').squeeze(0)

        # 获取相机内参矩阵
        intrinsics = torch.tensor(self.K, dtype=torch.float32)

        return {
            "image": left_image,
            "depth": depth_map,
            "intrinsics": intrinsics
        }

