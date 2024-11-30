# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import json
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class CityscapesEvalDataset(MonoDataset):
    """Cityscapes evaluation dataset - here we are loading the raw, original images rather than
    preprocessed triplets, and so cropping needs to be done inside get_color.
    """
    RAW_HEIGHT = 1024
    RAW_WIDTH = 2048
# RAW_HEIGHT 和 RAW_WIDTH 定义了Cityscapes数据集中图像的原始分辨率，这在后续处理图像时用于归一化。

    def __init__(self, *args, **kwargs):
        super(CityscapesEvalDataset, self).__init__(*args, **kwargs)
# 这里的构造方法调用了父类 MonoDataset 的构造方法，继承了父类的初始化设置。

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits

        txt file is of format:
            aachen aachen_000000 4
        """
        city, frame_name = self.filenames[index].split()
        side = None

        return city, frame_name, side
# 此方法将数据集中的 index 转换为 city（城市名称）、frame_name（帧名）和 side，以便在读取图像时能知道其位置。
# side 设置为 None，表示不需要额外的信息。

    def check_depth(self):
        return False
# 此方法直接返回 False，表示不检查深度信息，因为Cityscapes数据集中没有与深度相关的标签。

    def load_intrinsics(self, city, frame_name):
        # adapted from sfmlearner
        split = "test"  # if self.is_train else "val"

        camera_file = os.path.join(self.data_path, 'camera_trainvaltest', 'camera',
                                   split, city, frame_name + '_camera.json')
        with open(camera_file, 'r') as f:
            camera = json.load(f)
# 这里定义了 split = "test"，用于选择评估集的相机参数文件路径。
# camera_file 是相机参数的路径，用于获取 fx、fy、u0 和 v0 等参数。
        fx = camera['intrinsic']['fx']
        fy = camera['intrinsic']['fy']
        u0 = camera['intrinsic']['u0']
        v0 = camera['intrinsic']['v0']
# 从相机参数JSON文件中读取焦距（fx 和 fy）和主点位置（u0 和 v0）。
        intrinsics = np.array([[fx, 0, u0, 0],
                               [0, fy, v0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).astype(np.float32)
# 以 4x4 矩阵的形式构建相机内参矩阵 intrinsics，用于后续图像处理。
        intrinsics[0, :] /= self.RAW_WIDTH
        intrinsics[1, :] /= self.RAW_HEIGHT * 0.75
        return intrinsics
# 通过归一化调整相机参数，使其适应Cityscapes图像的分辨率变化。此处将宽度除以原始宽度 RAW_WIDTH，高度按 RAW_HEIGHT 的 3/4 缩放。

    def get_color(self, city, frame_name, side, do_flip, is_sequence=False):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides yet")
# 加载特定城市和帧的图像。
# do_flip 表示是否水平翻转图像。
        color = self.loader(self.get_image_path(city, frame_name, side, is_sequence))

        # crop down to cityscapes size
        w, h = color.size
        crop_h = h * 3 // 4
        color = color.crop((0, 0, w, crop_h))
# 通过 get_image_path 方法获取图像路径并加载图像，裁剪图像高度至3/4以符合Cityscapes原始尺寸。
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
# 如果 do_flip 为真，将图像水平翻转。
        return color

    def get_offset_framename(self, frame_name, offset=-2):
        city, seq, frame_num = frame_name.split('_')

        frame_num = int(frame_num) + offset
        frame_num = str(frame_num).zfill(6)
        return '{}_{}_{}'.format(city, seq, frame_num)
# 用于根据给定的帧名和偏移量生成新的帧名。
# 解析 frame_name，提取城市名、序列和帧号，并调整帧号。

    def get_colors(self, city, frame_name, side, do_flip):
        if side is not None:
            raise ValueError("Cityscapes dataset doesn't know how to deal with sides")

        color = self.get_color(city, frame_name, side, do_flip)

        prev_name = self.get_offset_framename(frame_name, offset=-2)
        prev_color = self.get_color(city, prev_name, side, do_flip, is_sequence=True)
# 通过 get_color 加载当前帧和前一帧图像，存储到 inputs 字典中，便于后续模型输入和处理。
        inputs = {}
        inputs[("color", 0, -1)] = color
        inputs[("color", -1, -1)] = prev_color

        return inputs

    def get_image_path(self, city, frame_name, side, is_sequence=False):
        folder = "leftImg8bit" if not is_sequence else "leftImg8bit_sequence"
        split = "test"
        image_path = os.path.join(
            self.data_path, folder, split, city, frame_name + '_leftImg8bit.png')
        return image_path
# 生成给定帧的图像路径，依据 is_sequence 决定是使用静态帧还是序列图像