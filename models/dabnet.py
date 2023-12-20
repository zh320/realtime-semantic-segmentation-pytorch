"""
Paper:      DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/1907.11357
Create by:  zh320
Date:       2023/08/27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, DWConvBNAct, ConvBNAct
from .enet import InitialBlock


class DABNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='prelu'):
        super(DABNet, self).__init__()
        self.layer1 = ConvBNAct(n_channel, 32, 3, 2, act_type=act_type)
        self.layer2 = ConvBNAct(32, 32, 3, 1, act_type=act_type)
        self.layer3 = ConvBNAct(32, 32, 3, 1, act_type=act_type)
        self.layer4 = InitialBlock(32+n_channel, 64, act_type=act_type)
        self.layer5_7 = build_blocks(DABModule, 64, 3, dilation=2, act_type=act_type)
        self.layer8 = ConvBNAct(64*2+n_channel, 128, 3, 2, act_type=act_type)
        self.layer9_10 = build_blocks(DABModule, 128, 2, dilation=4, act_type=act_type)
        self.layer11_12 = build_blocks(DABModule, 128, 2, dilation=8, act_type=act_type)
        self.layer13_14 = build_blocks(DABModule, 128, 2, dilation=16, act_type=act_type)
        self.layer15 = conv1x1(128*2+n_channel, num_class)

    def forward(self, x):
        size = x.size()[2:]
        x_d2 = F.avg_pool2d(x, 3, 2, 1)
        x_d4 = F.avg_pool2d(x_d2, 3, 2, 1)
        x_d8 = F.avg_pool2d(x_d4, 3, 2, 1)

        # Stage 1
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.cat([x, x_d2], dim=1)

        # Stage 2
        x = self.layer4(x)
        x_block1 = x
        x = self.layer5_7(x)
        x = torch.cat([x, x_block1], dim=1)
        x = torch.cat([x, x_d4], dim=1)

        # Stage 3
        x = self.layer8(x)
        x_block2 = x
        x = self.layer9_10(x)
        x = self.layer11_12(x)
        x = self.layer13_14(x)
        x = torch.cat([x, x_block2], dim=1)
        x = torch.cat([x, x_d8], dim=1)

        # Stage 4
        x = self.layer15(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


def build_blocks(block, channels, num_block, dilation, act_type):
    layers = []
    for _ in range(num_block):
        layers.append(block(channels, dilation, act_type=act_type))
    return  nn.Sequential(*layers)


class DABModule(nn.Module):
    def __init__(self, channels, dilation, act_type):
        super(DABModule, self).__init__()
        assert channels % 2 == 0, 'Input channel of DABModule should be multiple of 2.\n'
        hid_channels = channels // 2
        self.init_conv = ConvBNAct(channels, hid_channels, 3, act_type=act_type)
        self.left_branch = nn.Sequential(
                                DWConvBNAct(hid_channels, hid_channels, (3,1), act_type=act_type),
                                DWConvBNAct(hid_channels, hid_channels, (1,3), act_type=act_type)
                            )
        self.right_branch = nn.Sequential(
                                DWConvBNAct(hid_channels, hid_channels, (3,1), dilation=dilation, act_type=act_type),
                                DWConvBNAct(hid_channels, hid_channels, (1,3), dilation=dilation, act_type=act_type)
                            )
        self.last_conv = ConvBNAct(hid_channels, channels, 1, act_type=act_type)

    def forward(self, x):
        residual = x
        x = self.init_conv(x)

        x_left = self.left_branch(x)
        x_right = self.right_branch(x)
        x = x_left + x_right

        x = self.last_conv(x)
        x += residual

        return x
