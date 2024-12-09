# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import random
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

cv2.setNumThreads(0)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
# pil_loader 函数用于加载图像。它打开指定路径的图像文件，并将其转换为 RGB 格式。
# Image.open(f) 使用 Pillow 库打开图像，确保图像在 RGB 模式下加载，避免潜在的资源警告。


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """
# MonoDataset 是一个用于单目数据集的基类，它包含图像加载、数据预处理、数据增强和相机内参调整等功能。
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png',  ###change Feng
                 ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales

        '''
        self.interp = Image.ANTIALIAS
        '''
        self.interp = Image.Resampling.LANCZOS  # 替换 ANTI_ALIAS 为 LANCZOS
        ### change Feng
        
        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
# data_path 是数据存储路径，filenames 是文件名列表。
# 	•	height 和 width 表示图像的高度和宽度，num_scales 表示图像的尺度数量。
# 	•	frame_idxs 表示需要加载的帧索引，用于多帧数据。
# 	•	is_train 指示数据是否用于训练，影响数据增强策略。
# 	•	img_ext 表示图像文件的扩展名。
# 	•	loader 和 to_tensor 分别用于加载图像和将图像转换为 tensor 格式。
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
# 这部分代码定义了颜色增强的参数范围（亮度、对比度、饱和度和色相），如果新版本的 ColorJitter 支持使用元组形式的范围，则直接使用元组，否则降级为标量参数。

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
# 根据 num_scales 创建不同的缩放比例，用于多尺度输入。

        self.load_depth = self.check_depth()
# 调用 check_depth 方法来检查数据集是否包含深度数据。
    '''
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
# preprocess 方法用于图像的缩放和增强。
# 遍历输入中的图像，并按照多尺度缩放生成不同分辨率的图像。

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                # check it isn't a blank frame - keep _aug as zeros so we can check for it
                if inputs[(n, im, i)].sum() == 0:
                    inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                else:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
# 将缩放后的图像转换为 tensor 格式，并应用颜色增强 color_aug。
# 如果图像为空，则增强版本保持为空。
    '''

    def preprocess(self, inputs, color_aug):
        """Resize color images to the required scales and augment if required."""
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                # check it isn't a blank frame - keep _aug as zeros so we can check for it
                if inputs[(n, im, i)].sum() == 0:
                    inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                else:
                    # color_aug 是 ColorJitter 实例，因此可以直接调用
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    ### chang Feng

    def __len__(self):
        return len(self.filenames)
# 返回数据集的长度，即文件名列表的长度。

    def load_intrinsics(self, folder, frame_index):
        return self.K.copy()
# 加载相机内参矩阵，这里是直接返回内参矩阵 K 的副本。
    '''
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
# 随机决定是否进行颜色增强和水平翻转。
# folder、frame_index 和 side 是当前帧的路径信息。

        poses = {}
        if type(self).__name__ in ["CityscapesPreprocessedDataset", "CityscapesEvalDataset"]:
            inputs.update(self.get_colors(folder, frame_index, side, do_flip))
        else:
            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(
                        folder, frame_index, other_side, do_flip)
                else:
                    try:
                        inputs[("color", i, -1)] = self.get_color(
                            folder, frame_index + i, side, do_flip)
                    except FileNotFoundError as e:
                        if i != 0:
                            # fill with dummy values
                            inputs[("color", i, -1)] = \
                                Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                            poses[i] = None
                        else:
                            raise FileNotFoundError(f'Cannot find frame - make sure your '
                                                    f'--data_path is set correctly, or try adding'
                                                    f' the --png flag. {e}')
# 如果数据集类型是 CityscapesPreprocessedDataset 或 CityscapesEvalDataset，直接调用 get_colors 方法。
# 	•	否则，根据帧索引 frame_idxs 加载帧图像。
# 	•	当某一帧不存在时，填充空白图像。

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.load_intrinsics(folder, frame_index)

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
# 针对每个尺度调整相机内参矩阵 K，并计算其逆矩阵 inv_K，方便多尺度特征的计算。

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
# 如果 do_color_aug 为 True，则随机生成颜色增强参数，并将增强后的图像保存到 inputs 中。

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
# 删除临时图像，以节省内存。

        if self.load_depth and False:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs
        '''

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder, frame_index, side = self.index_to_folder_and_frame_idx(index)

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        for scale in range(self.num_scales):
            K = self.load_intrinsics(folder, frame_index)
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # 直接使用 transforms.ColorJitter 而不是 get_params
        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = lambda x: x  # 确保 color_aug 始终是一个函数

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth and False:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
        inputs = {}
        inputs[("color", 0, 0)] = self.get_color(folder, frame_index, side, do_flip=False)

        # 调试输出
        print(f"Loaded sample {index}: {inputs.keys()}")

        return inputs
    ### chang feng

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
# 这些方法在基类中没有实现，需要在子类中根据具体数据集实现，如加载图像、检查深度数据和加载深度信息等。