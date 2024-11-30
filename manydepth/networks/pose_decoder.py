# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
# num_ch_enc：编码器每一层的通道数列表，用于确定输入的特征图大小。
# num_input_features：输入特征的数量，一般对应输入的帧数。
# num_frames_to_predict_for：需要预测位姿的帧数。如果未指定，则默认预测所有输入帧减去1的帧数。
# stride：卷积核的步幅，默认为1。

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
# 如果 num_frames_to_predict_for 参数为空，默认预测的帧数为 num_input_features - 1（用于相邻帧之间的位姿估计）。

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)
# squeeze 卷积层：将编码器最后一层特征的通道数缩小到 256，卷积核大小为 1x1。
# pose 系列卷积层：用于生成位姿估计的卷积层。
# 	•	pose, 0：将特征通道数扩展到 256，卷积核大小为 3x3。
# 	•	pose, 1：用于进一步处理特征图，保持通道数为 256。
# 	•	pose, 2：生成位姿预测结果，输出通道数为 6 * num_frames_to_predict_for，因为每帧需要 6 个参数（3 个旋转和 3 个位移），卷积核大小为 1x1。
        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))
# relu：用于非线性激活。
# net：将所有卷积层打包成 ModuleList，便于在前向传播中使用。

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]
# input_features：编码器提取的多层特征列表。
# last_features：取每个输入特征的最后一层（最高层）特征，便于后续的位姿预测。

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)
# 通过 squeeze 卷积层和 ReLU 激活函数处理每个特征图，将它们的通道数减少为 256。
# 将所有特征图在通道维度上拼接（concatenate），形成一个综合特征图。

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)
# 依次通过 pose 系列卷积层对拼接后的特征图进行卷积处理，只有最后一层 pose, 2 没有应用 ReLU 激活。
# pose, 2 生成的输出 out 包含了位姿估计信息。

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)
# 	out.mean(3).mean(2)：在特征图的空间维度上进行平均池化，将 out 压缩为每个帧对的位姿向量。
# out.view(-1, self.num_frames_to_predict_for, 1, 6)：调整输出形状，其中 6 表示每帧对的位姿参数（3 个旋转 + 3 个位移），0.01 是一个缩放因子，用于防止输出值过大。

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
# 将 out 的前 3 个数值作为旋转参数 axisangle，后 3 个数值作为位移参数 translation，并返回它们。