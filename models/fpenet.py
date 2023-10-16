"""
Paper:      Feature Pyramid Encoding Network for Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/1909.08599
Create by:  zh320
Date:       2023/10/08
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DWConvBNAct, ConvBNAct


class FPENet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, p=3, q=9, k=4, act_type='relu'):
        super(FPENet, self).__init__()
        self.stage1 = nn.Sequential(
                            ConvBNAct(n_channel, 16, 3, 2, act_type=act_type, inplace=True),
                            FPEBlock(16, 16, 1, 1, act_type=act_type)
                        )
        self.stage2_0 = FPEBlock(16, 32, k, 2, act_type=act_type)
        self.stage2 = build_blocks(FPEBlock, 32, p-1, k, act_type)
        self.stage3_0 = FPEBlock(32, 64, k, 2, act_type=act_type)
        self.stage3 = build_blocks(FPEBlock, 64, q-1, k, act_type)
        self.decoder2 = MEUModule(32, 64, 64, act_type)
        self.decoder1 = MEUModule(16, 64, 32, act_type)
        self.final = ConvBNAct(32, num_class, 1, act_type=act_type, inplace=True)

    def forward(self, x):
        size = x.size()[2:]
        x1 = self.stage1(x)
        x = self.stage2_0(x1)
        x2 = self.stage2(x)
        x = self.stage3_0(x2)
        x = self.stage3(x)
        x = self.decoder2(x2, x)
        x = self.decoder1(x1, x)
        x = self.final(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


def build_blocks(block, channels, num_block, expansion, act_type):
    layers = []
    for i in range(num_block):
        layers.append(block(channels, channels, expansion, 1, act_type=act_type))
    return  nn.Sequential(*layers)


class FPEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, dilations=[1,2,4,8], 
                    act_type='relu'):
        super(FPEBlock, self).__init__()
        assert len(dilations) > 0, 'Length of dilations should be larger than 0.\n'
        self.K = len(dilations)
        self.use_skip = (in_channels == out_channels) and (stride == 1)
        expand_channels = out_channels * expansion
        self.ch = expand_channels // self.K

        self.conv_init = ConvBNAct(in_channels, expand_channels, 1, act_type=act_type, inplace=True)

        self.layers = nn.ModuleList()
        for i in range(self.K):
            self.layers.append(DWConvBNAct(self.ch, self.ch, 3, stride, dilations[i], act_type=act_type))

        self.conv_last = ConvBNAct(expand_channels, out_channels, 1, act_type=act_type)

    def forward(self, x):
        if self.use_skip:
            residual = x

        x = self.conv_init(x)

        transform_feats = []
        for i in range(self.K):
            transform_feats.append(self.layers[i](x[:, i*self.ch:(i+1)*self.ch]))

        for j in range(1, self.K):
            transform_feats[j] += transform_feats[j-1]

        x = torch.cat(transform_feats, dim=1)

        x = self.conv_last(x)

        if self.use_skip:
            x += residual

        return x


class MEUModule(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels, act_type):
        super(MEUModule, self).__init__()
        self.conv_low = ConvBNAct(low_channels, out_channels, 1, act_type=act_type, inplace=True)
        self.conv_high = ConvBNAct(high_channels, out_channels, 1, act_type=act_type, inplace=True)
        self.sa = SpatialAttentionBlock(act_type)
        self.ca = ChannelAttentionBlock(out_channels, act_type)

    def forward(self, x_low, x_high):
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)

        x_sa = self.sa(x_low)
        x_ca = self.ca(x_high)

        x_low = x_low * x_ca
        x_high = F.interpolate(x_high, scale_factor=2, mode='bilinear', align_corners=True)
        x_high = x_high * x_sa

        return x_low + x_high


class SpatialAttentionBlock(nn.Module):
    def __init__(self, act_type):
        super(SpatialAttentionBlock, self).__init__()
        self.conv = ConvBNAct(1, 1, 1, act_type=act_type, inplace=True)

    def forward(self, x):
        x = self.conv(torch.mean(x, dim=1, keepdim=True))

        return x


class ChannelAttentionBlock(nn.Sequential):
    def __init__(self, channels, act_type):
        super(ChannelAttentionBlock, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(channels, channels, 1, act_type=act_type, inplace=True)
        )
