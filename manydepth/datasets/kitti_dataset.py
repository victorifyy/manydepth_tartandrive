# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402
import skimage.transform
import numpy as np
import PIL.Image as pil

from manydepth.kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
# KITTIDataset 是 KITTI 数据集加载器的基类。它包含了加载数据集的基本方法，其他子类可以根据特定数据集类型扩展此类。
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
# K 是相机的内参矩阵（焦距和主点位置），该矩阵经过归一化处理以适应图像尺寸。
# full_res_shape 定义了 KITTI 数据集的原始分辨率。
# side_map 是一个字典，用于将左右相机的标识符映射为数值。

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)
# 用于检查某一帧的深度文件是否存在，以判断是否有深度信息。
# 通过文件路径 velo_filename 判断文件是否存在，从而确定是否包含深度数据。

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        return folder, frame_index, side
# 将数据集中的索引 index 转换为 folder（场景文件夹名）、frame_index（帧索引）和 side（左右相机标识）。

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
# 	加载指定 folder、frame_index 和 side 的图像，如果 do_flip 为 True，则水平翻转图像。


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
# KITTIRAWDataset 继承自 KITTIDataset，用于加载 KITTI 原始深度数据集。
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path
# 生成图像路径，根据 folder 和 frame_index 组合生成文件名 f_str。

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
# get_depth 方法用于加载帧的深度图，通过 generate_depth_map 生成深度图，并调整分辨率以匹配图像的原始分辨率。
# 如果 do_flip 为 True，则水平翻转深度图。


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
# KITTIOdomDataset 继承自 KITTIDataset，专用于加载 KITTI 里程计数据。
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path
# 根据 folder、frame_index 和 side 生成图像路径。
# 使用 sequences 文件夹来区分不同的里程计序列。


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
# KITTIDepthDataset 继承自 KITTIDataset，用于加载更新后的 KITTI 深度数据集。
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path
# 根据 folder、frame_index 和 side 生成图像路径，适用于更新后的数据集格式。

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
# 加载指定帧的深度图并调整分辨率，适用于包含更新的地面真实深度数据的 KITTI 数据集。