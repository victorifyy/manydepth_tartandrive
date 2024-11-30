# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class CityscapesPreprocessedDataset(MonoDataset):
    """Cityscapes dataset - this expects triplets of images concatenated into a single wide image,
    which have had the ego car removed (bottom 25% of the image cropped)
    """

    RAW_WIDTH = 1024
    RAW_HEIGHT = 384

    def __init__(self, *args, **kwargs):
        super(CityscapesPreprocessedDataset, self).__init__(*args, **kwargs)

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            ulm ulm_000064_000012
        """
        city, frame_name = self.filenames[index].split()
        side = None
        return city, frame_name, side
# 该方法根据数据集中的索引 index，将文件名解析为 city（城市名称）和 frame_name（帧名称）。
# side 被设置为 None，表示不需要额外的视角信息。

    def check_depth(self):
        return False
# 该方法返回 False，表示不检查深度数据，因为此数据集不包含深度标签。

    def load_intrinsics(self, city, frame_name):
        # adapted from sfmlearner

        camera_file = os.path.join(self.data_path, city, "{}_cam.txt".format(frame_name))
        camera = np.loadtxt(camera_file, delimiter=",")
# 根据 city 和 frame_name 生成相机参数文件路径 camera_file。
# np.loadtxt 函数从文件中读取相机内参数据，假定文件使用逗号作为分隔符。
        fx = camera[0]
        fy = camera[4]
        u0 = camera[2]
        v0 = camera[5]
# 读取相机的焦距（fx 和 fy）以及主点坐标（u0 和 v0），这些参数用于构建内参矩阵。
        intrinsics = np.array([[fx, 0, u0, 0],
                               [0, fy, v0, 0],
                               [0,  0,  1, 0],
                               [0,  0,  0, 1]]).astype(np.float32)
# 以 4x4 矩阵的形式构建相机内参矩阵 intrinsics，并将其数据类型设置为 float32，便于后续计算。
        intrinsics[0, :] /= self.RAW_WIDTH
        intrinsics[1, :] /= self.RAW_HEIGHT
        return intrinsics
# 对相机内参矩阵中的水平和垂直坐标进行归一化处理，使其适应图像的原始宽度 RAW_WIDTH 和高度 RAW_HEIGHT。

    def get_colors(self, city, frame_name, side, do_flip):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")
# side 参数在此数据集中未使用，如果提供则会抛出错误。
        color = self.loader(self.get_image_path(city, frame_name))
        color = np.array(color)
# 通过 get_image_path 方法获取图像路径并加载图像，将图像转换为 numpy 数组，便于后续操作。
        w = color.shape[1] // 3
        inputs = {}
        inputs[("color", -1, -1)] = pil.fromarray(color[:, :w])
        inputs[("color", 0, -1)] = pil.fromarray(color[:, w:2*w])
        inputs[("color", 1, -1)] = pil.fromarray(color[:, 2*w:])
# 将加载的三联图像按照宽度均等分割为三个单独的图像，每个图像的宽度为 w。
        # 	•	inputs[("color", -1, -1)]：存储第一张图像。
        # 	•	inputs[("color", 0, -1)]：存储第二张图像。
        # 	•	inputs[("color", 1, -1)]：存储第三张图像。
        if do_flip:
            for key in inputs:
                inputs[key] = inputs[key].transpose(pil.FLIP_LEFT_RIGHT)
# 如果 do_flip 为 True，则水平翻转每张图像。
        return inputs

    def get_image_path(self, city, frame_name):
        return os.path.join(self.data_path, city, "{}.jpg".format(frame_name))
# 根据 city 和 frame_name 拼接生成图像文件路径