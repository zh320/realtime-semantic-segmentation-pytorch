"""
Paper:      CFPNet: Channel-wise Feature Pyramid for Real-Time Semantic Segmentation
Url:        https://arxiv.org/abs/2103.12212
Create by:  zh320
Date:       2023/09/30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

from .modules import ConvBNAct
from .enet import InitialBlock as DownsamplingBlock


class CFPNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, n=2, m=6, dilations=[2,2,4,4,8,8,16,16], 
                    act_type='prelu'):
        super(CFPNet, self).__init__()
        assert len(dilations) == (n+m), f'Length of dilations should be equal to {n+m}.\n'
        self.conv_init = nn.Sequential(
                            ConvBNAct(n_channel, 32, stride=2, act_type=act_type),
                            ConvBNAct(32, 32, act_type=act_type),
                            ConvBNAct(32, 32, act_type=act_type)
                        )
        self.downsample1 = DownsamplingBlock(32+3, 64, act_type)
        self.cfp1 = build_blocks(CFPModule, 64, n, dilations[:n], act_type)
        self.downsample2 = DownsamplingBlock(64+3, 128, act_type)
        self.cfp2 = build_blocks(CFPModule, 128, m, dilations[n:], act_type)
        self.seg_head = ConvBNAct(128+3, num_class, 1, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]
        x_d2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x_d4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        x_d8 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=True)

        x = self.conv_init(x)
        x = torch.cat([x, x_d2], dim=1)

        x = self.downsample1(x)
        x = self.cfp1(x)
        x = torch.cat([x, x_d4], dim=1)

        x = self.downsample2(x)
        x = self.cfp2(x)
        x = torch.cat([x, x_d8], dim=1)

        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


def build_blocks(block, channels, num_block, dilations=[], act_type='relu'):
    if len(dilations) == 0:
        dilations = [1 for _ in range(num_block)]
    else:
        if len(dilations) != num_block:
            raise ValueError(f'Number of dilation should be equal to number of blocks')

    layers = []
    for i in range(num_block):
        layers.append(block(channels, dilations[i], act_type=act_type))
    return  nn.Sequential(*layers)


class CFPModule(nn.Module):
    def __init__(self, channels, rk, K=4, rk_ratio=None, act_type='prelu',):
        super(CFPModule, self).__init__()
        if rk_ratio is None:
            rk_ratio = [1/rk, 1/4, 1/2, 1]
        assert len(rk_ratio) == K, f'Length of rk_ratio should be {K}.\n'

        self.K = K
        channel_kn = channels // K

        self.conv_init = ConvBNAct(channels, channel_kn, 1, act_type=act_type)

        self.layers = nn.ModuleList()
        for k in range(K):
            dt = ceil(rk * rk_ratio[k])    # dilation
            self.layers.append(FeaturePyramidChannel(channel_kn, dt, act_type=act_type))

        self.conv_last = ConvBNAct(channels, channels, 1, act_type=act_type)

    def forward(self, x):
        residual = x

        x = self.conv_init(x)       # Projection

        transform_feats = []        # Parallel FP channels
        for i in range(self.K):
            transform_feats.append(self.layers[i](x))

        for j in range(1, self.K):
            transform_feats[j] += transform_feats[j-1]

        x = torch.cat(transform_feats, dim=1)   # Concatenation

        x = self.conv_last(x)

        x += residual

        return x


class FeaturePyramidChannel(nn.Module):
    def __init__(self, channels, dilation, act_type, channel_split=[1,1,2]):
        super(FeaturePyramidChannel, self).__init__()
        split_num = sum(channel_split)
        assert channels % split_num == 0, f'Channel of FPC should be multiple of {split_num}.\n'
        ch_b1 = (channels // split_num) * channel_split[0]
        ch_b2 = (channels // split_num) * channel_split[1]
        ch_b3 = (channels // split_num) * channel_split[2]

        self.block1 = nn.Sequential(
                            ConvBNAct(channels, ch_b1, (3, 1), dilation=dilation, act_type=act_type),
                            ConvBNAct(ch_b1, ch_b1, (1, 3), dilation=dilation, act_type=act_type),
                        )
        self.block2 = nn.Sequential(
                            ConvBNAct(ch_b1, ch_b2, (3, 1), dilation=dilation, act_type=act_type),
                            ConvBNAct(ch_b2, ch_b2, (1, 3), dilation=dilation, act_type=act_type),
                        )
        self.block3 = nn.Sequential(
                            ConvBNAct(ch_b2, ch_b3, (3, 1), dilation=dilation, act_type=act_type),
                            ConvBNAct(ch_b3, ch_b3, (1, 3), dilation=dilation, act_type=act_type),
                        )

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        
        x = torch.cat([x1, x2, x3], dim=1)

        return x
