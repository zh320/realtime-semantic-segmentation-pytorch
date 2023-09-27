"""
Paper:      Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation
Url:        https://arxiv.org/abs/1809.06323
Create by:  zh320
Date:       2023/09/24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, conv3x3, ConvBNAct, Activation


class EDANet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, k=40, num_b1=5, num_b2=8, act_type='relu'):
        super(EDANet, self).__init__()
        self.stage1 = DownsamplingBlock(n_channel, 15, act_type)
        self.stage2_d = DownsamplingBlock(15, 60, act_type)
        self.stage2 = EDABlock(60, k, num_b1, [1,1,1,2,2], act_type)
        self.stage3_d = ConvBNAct(260, 130, 3, 2, act_type=act_type)
        self.stage3 = EDABlock(130, k, num_b2, [2,2,4,4,8,8,16,16], act_type)
        self.project = conv1x1(130+k*num_b2, num_class)

    def forward(self, x):
        size = x.size()[2:]
        x = self.stage1(x)
        x = self.stage2_d(x)
        x = self.stage2(x)
        x = self.stage3_d(x)
        x = self.stage3(x)
        x = self.project(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(DownsamplingBlock, self).__init__()
        self.conv = conv3x3(in_channels, out_channels - in_channels, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn_act = nn.Sequential(
                                nn.BatchNorm2d(out_channels),
                                Activation(act_type)
                            )

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)
        return self.bn_act(x)


class EDABlock(nn.Module):
    def __init__(self, in_channels, k, num_block, dilations, act_type):
        super(EDABlock, self).__init__()
        assert len(dilations) == num_block, 'number of dilation rate should be equal to number of block'

        layers = []
        for i in range(num_block):
            dt = dilations[i]
            layers.append(EDAModule(in_channels, k, dt, act_type))
            in_channels += k
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EDAModule(nn.Module):
    def __init__(self, in_channels, k, dilation=1, act_type='relu'):
        super(EDAModule, self).__init__()
        self.conv = nn.Sequential(
                        ConvBNAct(in_channels, k, 1),
                        nn.Conv2d(k, k, (3, 1), padding=(1, 0), bias=False),
                        ConvBNAct(k, k, (1, 3), act_type=act_type),
                        nn.Conv2d(k, k, (3, 1), dilation=dilation, 
                                    padding=(dilation, 0), bias=False),
                        ConvBNAct(k, k, (1, 3), dilation=dilation, act_type=act_type)
                    )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = torch.cat([x, residual], dim=1)
        return x
