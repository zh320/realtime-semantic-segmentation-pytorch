"""
Paper:      LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation
Url:        https://arxiv.org/abs/1905.02423
Create by:  zh320
Date:       2023/04/23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct


class LEDNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3):
        super(LEDNet, self).__init__()
        self.encoder = Encoder(n_channel, 128)
        self.apn = AttentionPyramidNetwork(128, num_class)

    def forward(self, x):
        size = x.size()[2:]
        x = self.encoder(x)
        x = self.apn(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class Encoder(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__(
            DownsampleUint(in_channels, 32),
            SSnbtUnit(32),
            SSnbtUnit(32),
            SSnbtUnit(32),
            DownsampleUint(32, 64),
            SSnbtUnit(64),
            SSnbtUnit(64),
            DownsampleUint(64, out_channels),
            SSnbtUnit(out_channels, dilation=1),
            SSnbtUnit(out_channels, dilation=2),
            SSnbtUnit(out_channels, dilation=5),
            SSnbtUnit(out_channels, dilation=9),
            SSnbtUnit(out_channels, dilation=2),
            SSnbtUnit(out_channels, dilation=5),
            SSnbtUnit(out_channels, dilation=9),
            SSnbtUnit(out_channels, dilation=17),
        )


class DownsampleUint(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleUint, self).__init__()
        assert out_channels > in_channels, 'Output channel should be larger than input channel.'
        self.conv = ConvBNAct(in_channels, out_channels-in_channels, 3, 2)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)

        return x


class SSnbtUnit(nn.Module):
    def __init__(self, in_channels, dilation=1):
        super(SSnbtUnit, self).__init__()
        split_channels = in_channels // 2
        self.split_channels = split_channels
        self.left_branch = nn.Sequential(
                                nn.Conv2d(split_channels, split_channels, (3, 1), padding=(1,0)),
                                nn.ReLU(),
                                ConvBNAct(split_channels, split_channels, (1, 3)),
                                nn.Conv2d(split_channels, split_channels, (3, 1), 
                                            padding=(dilation,0), dilation=dilation),
                                nn.ReLU(),
                                ConvBNAct(split_channels, split_channels, (1, 3), dilation=dilation),
                            )
                            
        self.right_branch = nn.Sequential(
                                nn.Conv2d(split_channels, split_channels, (1, 3), padding=(0,1)),
                                nn.ReLU(),
                                ConvBNAct(split_channels, split_channels, (3, 1)),
                                nn.Conv2d(split_channels, split_channels, (1, 3), 
                                            padding=(0,dilation), dilation=dilation),
                                nn.ReLU(),
                                ConvBNAct(split_channels, split_channels, (3, 1), dilation=dilation),
                            )

    def forward(self, x):
        x_left = x[:, :self.split_channels]
        x_right = x[:, self.split_channels:]
        x_left = self.left_branch(x_left)
        x_right = self.right_branch(x_right)
        x_cat = torch.cat([x_left, x_right], dim=1)
        x += x_cat
        x = channel_shuffle(x)
        return x


class AttentionPyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionPyramidNetwork, self).__init__()
        self.left_conv1_1 = ConvBNAct(in_channels, in_channels, 3, 2)
        self.left_conv1_2 = ConvBNAct(in_channels, out_channels, 3)
        self.left_conv2_1 = ConvBNAct(in_channels, in_channels, 3, 2)
        self.left_conv2_2 = ConvBNAct(in_channels, out_channels, 3)
        self.left_conv3 = nn.Sequential(
                                ConvBNAct(in_channels, in_channels, 3, 2),
                                ConvBNAct(in_channels, out_channels, 3)
                            )

        self.mid_branch = ConvBNAct(in_channels, out_channels)
        self.right_branch = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),
                                ConvBNAct(in_channels, out_channels),
                            )

    def forward(self, x):
        size0 = x.size()[2:]
        
        x_left = self.left_conv1_1(x)
        size1 = x_left.size()[2:]
        
        x_left2 = self.left_conv2_1(x_left)
        size2 = x_left2.size()[2:]
        
        x_left3 = self.left_conv3(x_left2)
        x_left3 = F.interpolate(x_left3, size2, mode='bilinear', align_corners=True)
        
        x_left2 = self.left_conv2_2(x_left2)
        x_left2 += x_left3
        x_left2 = F.interpolate(x_left2, size1, mode='bilinear', align_corners=True)
        
        x_left = self.left_conv1_2(x_left)
        x_left += x_left2
        x_left = F.interpolate(x_left, size0, mode='bilinear', align_corners=True)
        
        x_mid = self.mid_branch(x)
        x_mid = torch.mul(x_left, x_mid)
        
        x_right = self.right_branch(x)
        x_right = F.interpolate(x_right, size0, mode='bilinear', align_corners=True)

        x_mid += x_right
        return x_mid


def channel_shuffle(x, groups=2):
    # Codes are borrowed from 
    # https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
