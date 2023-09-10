"""
Paper:      AGLNet: Towards real-time semantic segmentation of self-driving images 
                    via attention-guided lightweight network
Url:        https://www.sciencedirect.com/science/article/abs/pii/S1568494620306207
Create by:  zh320
Date:       2023/08/27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, Activation


class AGLNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='relu'):
        super(AGLNet, self).__init__()
        self.layer1 = DownsamplingUnit(n_channel, 32, act_type=act_type)
        self.layer2_4 = build_blocks(SSnbtUnit, 32, 3, act_type=act_type)
        self.layer5 = DownsamplingUnit(32, 64, act_type=act_type)
        self.layer6_7 = build_blocks(SSnbtUnit, 64, 2, act_type=act_type)
        self.layer8 = DownsamplingUnit(64, 128, act_type=act_type)
        self.layer9_16 = build_blocks(SSnbtUnit, 128, 8, dilations=[1,2,5,9,2,5,9,17], act_type=act_type)
        self.layer17 = FAPM(128, act_type=act_type)
        self.layer18 = GAUM(64, 128, 64, act_type=act_type)
        self.layer19 = GAUM(32, 64, 32, act_type=act_type)
        self.layer20 = conv1x1(32, num_class)

    def forward(self, x):
        size = x.size()[2:]
        
        # Stage 1
        x = self.layer1(x)
        x = self.layer2_4(x)
        x_s1 = x

        # Stage 2
        x = self.layer5(x)
        x = self.layer6_7(x)
        x_s2 = x

        # Stage 3
        x = self.layer8(x)
        x = self.layer9_16(x)
        
        x = self.layer17(x)
        x = self.layer18(x, x_s2)
        x = self.layer19(x, x_s1)
        
        x = self.layer20(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        return x


class DownsamplingUnit(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(DownsamplingUnit, self).__init__()
        self.conv = ConvBNAct(in_channels, out_channels - in_channels, 3, 2, act_type=act_type)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)

        return x


def build_blocks(block, channels, num_block, dilations=[], act_type='relu'):
    if len(dilations) == 0:
        dilations = [1 for _ in range(num_block)]
    else:
        if len(dilations) != num_block:
            raise ValueError(f'Number of dilation should be equal to number of blocks')

    layers = []
    for i in range(num_block):
        layers.append(block(channels, dilation=dilations[i], act_type=act_type))
    return  nn.Sequential(*layers)


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


class FAPM(nn.Module):
    def __init__(self, channels, act_type):
        super(FAPM, self).__init__()
        self.pfa = PyramidFeatureAttention(channels, act_type)
        self.conv = conv1x1(1, channels)
        self.gp = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            conv1x1(channels, channels),
                        )
        
    def forward(self, x):
        size = x.size()[2:]
        x_pfa = self.pfa(x)
        x_pfa = self.conv(x_pfa)

        x_gp = self.gp(x)
        x_gp = F.interpolate(x_gp, size, mode='bilinear', align_corners=True)

        x = x * x_pfa
        x += x_gp

        return x


class PyramidFeatureAttention(nn.Module):
    def __init__(self, channels, act_type):
        super(PyramidFeatureAttention, self).__init__()
        self.conv11 = ConvBNAct(channels, 1, (1,7), 2, act_type=act_type)
        self.conv12 = ConvBNAct(1, 1, (7,1), 1, act_type=act_type)
        self.conv21 = ConvBNAct(1, 1, (1,5), 2, act_type=act_type)
        self.conv22 = ConvBNAct(1, 1, (5,1), 1, act_type=act_type)
        self.conv31 = ConvBNAct(1, 1, (1,3), 2, act_type=act_type)
        self.conv32 = ConvBNAct(1, 1, (3,1), 1, act_type=act_type)

    def forward(self, x):
        size0 = x.size()[2:]

        x = self.conv11(x)
        size1 = x.size()[2:]
        x1 = self.conv12(x)
        
        x = self.conv21(x)
        size2 = x.size()[2:]
        x2 = self.conv22(x)
        
        x = self.conv31(x)
        x = self.conv32(x)
        x = F.interpolate(x, size2, mode='bilinear', align_corners=True)

        x += x2
        x = F.interpolate(x, size1, mode='bilinear', align_corners=True)
        
        x += x1
        x = F.interpolate(x, size0, mode='bilinear', align_corners=True)
        
        return x


class GAUM(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels, act_type):
        super(GAUM, self).__init__()
        self.up_conv = nn.Sequential(
                                nn.ConvTranspose2d(high_channels, low_channels, 3, 2, 1, 1),
                                nn.BatchNorm2d(low_channels),
                                Activation(act_type)
                        )
        self.sab = SpatialAttentionBlock(low_channels)
        self.cab = ChannelAttentionBlock(low_channels, out_channels)

    def forward(self, x_high, x_low):
        x_low = self.sab(x_low)
        x_high = self.up_conv(x_high)
        x_skip = x_high
        
        x_high = x_high * x_low
        x_skip2 = x_high
        
        x_high = self.cab(x_high)
        x_high = x_high * x_skip2
        
        x_high += x_skip
        return x_high


class SpatialAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(SpatialAttentionBlock, self).__init__()
        self.conv = conv1x1(channels, 1)
        
    def forward(self, x):
        x_s = self.conv(x)
        x_s = torch.sigmoid(x_s)
        x = x * x_s
        return x


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = conv1x1(in_channels, out_channels)
        
    def forward(self, x):
        x_c = self.pool(x)
        x_c = self.conv(x_c)
        x_c = torch.sigmoid(x_c)
        x = x * x_c
        return x
