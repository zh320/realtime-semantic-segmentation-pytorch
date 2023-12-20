"""
Paper:      LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation
Url:        https://arxiv.org/abs/1905.02423
Create by:  zh320
Date:       2023/04/23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, Activation, channel_shuffle
from .enet import InitialBlock as DownsampleUint


class LEDNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='relu'):
        super(LEDNet, self).__init__()
        self.encoder = Encoder(n_channel, 128, act_type)
        self.apn = AttentionPyramidNetwork(128, num_class, act_type)

    def forward(self, x):
        size = x.size()[2:]
        x = self.encoder(x)
        x = self.apn(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class Encoder(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type):
        super(Encoder, self).__init__(
            DownsampleUint(in_channels, 32, act_type),
            SSnbtUnit(32, 1, act_type=act_type),
            SSnbtUnit(32, 1, act_type=act_type),
            SSnbtUnit(32, 1, act_type=act_type),
            DownsampleUint(32, 64, act_type),
            SSnbtUnit(64, 1, act_type=act_type),
            SSnbtUnit(64, 1, act_type=act_type),
            DownsampleUint(64, out_channels, act_type),
            SSnbtUnit(out_channels, 1, act_type=act_type),
            SSnbtUnit(out_channels, 2, act_type=act_type),
            SSnbtUnit(out_channels, 5, act_type=act_type),
            SSnbtUnit(out_channels, 9, act_type=act_type),
            SSnbtUnit(out_channels, 2, act_type=act_type),
            SSnbtUnit(out_channels, 5, act_type=act_type),
            SSnbtUnit(out_channels, 9, act_type=act_type),
            SSnbtUnit(out_channels, 17, act_type=act_type),
        )


class SSnbtUnit(nn.Module):
    def __init__(self, channels, dilation, act_type):
        super(SSnbtUnit, self).__init__()
        assert channels % 2 == 0, 'Input channel should be multiple of 2.\n'
        split_channels = channels // 2
        self.split_channels = split_channels
        self.left_branch = nn.Sequential(
                                nn.Conv2d(split_channels, split_channels, (3, 1), padding=(1,0)),
                                Activation(act_type),
                                ConvBNAct(split_channels, split_channels, (1, 3), act_type=act_type),
                                nn.Conv2d(split_channels, split_channels, (3, 1), 
                                            padding=(dilation,0), dilation=dilation),
                                Activation(act_type),
                                ConvBNAct(split_channels, split_channels, (1, 3), dilation=dilation, act_type=act_type),
                            )
                            
        self.right_branch = nn.Sequential(
                                nn.Conv2d(split_channels, split_channels, (1, 3), padding=(0,1)),
                                Activation(act_type),
                                ConvBNAct(split_channels, split_channels, (3, 1), act_type=act_type),
                                nn.Conv2d(split_channels, split_channels, (1, 3), 
                                            padding=(0,dilation), dilation=dilation),
                                Activation(act_type),
                                ConvBNAct(split_channels, split_channels, (3, 1), dilation=dilation, act_type=act_type),
                            )
        self.act = Activation(act_type)

    def forward(self, x):
        x_left = x[:, :self.split_channels].clone()
        x_right = x[:, self.split_channels:].clone()
        x_left = self.left_branch(x_left)
        x_right = self.right_branch(x_right)
        x_cat = torch.cat([x_left, x_right], dim=1)
        x += x_cat
        x = self.act(x)
        x = channel_shuffle(x)
        return x


class AttentionPyramidNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(AttentionPyramidNetwork, self).__init__()
        self.left_conv1_1 = ConvBNAct(in_channels, in_channels, 3, 2, act_type=act_type)
        self.left_conv1_2 = ConvBNAct(in_channels, out_channels, 3, act_type=act_type)
        self.left_conv2_1 = ConvBNAct(in_channels, in_channels, 3, 2, act_type=act_type)
        self.left_conv2_2 = ConvBNAct(in_channels, out_channels, 3, act_type=act_type)
        self.left_conv3 = nn.Sequential(
                                ConvBNAct(in_channels, in_channels, 3, 2, act_type=act_type),
                                ConvBNAct(in_channels, out_channels, 3, act_type=act_type)
                            )

        self.mid_branch = ConvBNAct(in_channels, out_channels, act_type=act_type)
        self.right_branch = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),
                                ConvBNAct(in_channels, out_channels, act_type=act_type),
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
