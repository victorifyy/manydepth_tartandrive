# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch.nn as nn


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames
# num_input_frames：输入的图像帧数。通常用于表示输入图像序列的帧数，例如用于计算相邻帧的相机位姿。
# super(PoseCNN, self).__init__() 初始化 nn.Module 的基类。

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)
# self.convs 是一个字典，用于存储一系列卷积层。
# 该网络使用 7 层卷积层，每层卷积层通过 nn.Conv2d 实现。
# 	•	第一层卷积 self.convs[0] 的输入通道数为 3 * num_input_frames，因为每帧图像有 3 个颜色通道。
# 	•	每层卷积层的卷积核大小逐步缩小，第一层为 7x7，之后为 5x5，接着为 3x3，步幅和填充方式不同，以保持特征图的尺寸变化。

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)
# pose_conv 是一个 1x1 的卷积层，其输出通道数为 6 * (num_input_frames - 1)，表示 6 个参数（3 个旋转参数和 3 个位移参数）乘以帧数减去 1。
# 每对相邻帧需要 6 个自由度来描述旋转和平移，因此输出的通道数是 6 * (num_input_frames - 1)。
        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))
# num_convs 表示卷积层的数量（7 层）。
# relu 是一个激活函数，用于在每一层卷积后引入非线性。
# net 将卷积层的字典 convs 转为 ModuleList，便于在前向传播中逐层使用。

    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)
# 遍历 self.convs 中的每一层卷积，将输入 out 通过每一层卷积层并应用 ReLU 激活函数，逐步提取特征。

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)
# 将特征图输入到最后一层卷积 pose_conv，生成位姿估计的原始输出。
# 使用 mean(3).mean(2) 对特征图的空间维度求平均，将其压缩为每个帧对的位姿向量。
# out.view(-1, self.num_input_frames - 1, 1, 6) 将结果重新调整为 (-1, 帧数-1, 1, 6) 的维度，其中 6 表示旋转和位移参数，0.01 是一个缩放因子，防止位姿估计过大。

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
# 将输出 out 的前 3 个数值作为旋转参数 axisangle，后 3 个数值作为平移参数 translation，返回旋转角和位移两个张量。