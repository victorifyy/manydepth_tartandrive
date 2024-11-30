# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from manydepth.layers import ConvBlock, Conv3x3, upsample
# 该代码使用了 torch 的神经网络模块 nn 来定义神经网络层。
# 	•	OrderedDict 用于存储网络层，以便按顺序初始化。
# 	•	ConvBlock、Conv3x3 和 upsample 是一些辅助模块，分别用于卷积块、3x3卷积和上采样操作。


class DepthDecoder(nn.Module):
# DepthDecoder 类继承自 nn.Module，实现了深度估计网络的解码器部分。解码器的目的是将编码器提取的特征进行逐层上采样并生成多尺度的深度图。
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()
#	•	num_ch_enc：编码器输出的每层特征图的通道数。
#	•	scales：多尺度的深度输出层次，默认为 range(4)，即生成 4 个不同分辨率的深度图。
#	•	num_output_channels：输出的通道数，默认为1（单通道深度图）。
#	•	use_skips：是否使用跳跃连接（skip connections），用于融合编码器和解码器的特征。

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
# 	•	upsample_mode 设置为 nearest，使用最近邻插值进行上采样。
# 	•	num_ch_dec 定义了解码器每一层的通道数，从低到高依次为 [16, 32, 64, 128, 256]。

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
# 	•	解码器层的卷积操作以递减顺序从高层到低层定义（i从4到0），每层包含两个卷积操作 upconv_0 和 upconv_1。
# 	•	upconv_0：首先上采样后对输入特征进行卷积。
# 	•	upconv_1：将跳跃连接的特征拼接后进行第二次卷积。
# 	•	如果 use_skips 为 True，并且层数 i > 0，则 upconv_1 的输入通道数会增加对应编码器层的通道数。

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
# 	•	定义多尺度输出层，每个尺度 s 对应一个 dispconv 卷积层，用于生成指定尺度的深度图。
# 	•	Conv3x3 是一个 3x3 卷积层，将通道数变为 num_output_channels。

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
# 将 convs 中的所有层存入 decoder 的 ModuleList 中，并初始化 sigmoid 激活函数用于深度输出。

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
# 	•	input_features 是编码器的特征图列表，input_features[-1] 表示编码器的最高层特征图。
# 	•	outputs 用于存储最终的深度图输出。
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
# 对于每一层 i，首先通过 upconv_0 卷积对特征图 x 进行上采样。
# 使用 upsample 函数对特征图 x 进行上采样，然后判断是否需要跳跃连接：
# 	•	若 use_skips 为 True 且 i > 0，将编码器的 input_features[i - 1] 拼接到上采样后的 x。
# 通过 upconv_1 卷积处理拼接后的特征图。
# 如果当前层在多尺度输出范围 scales 中，则计算深度图：
# 	•	使用 dispconv 卷积生成深度图，并通过 sigmoid 激活将深度图值约束在0到1之间。
# 	•	将结果存储到 outputs 字典中。
        return self.outputs
# 返回包含多尺度深度图的字典 outputs