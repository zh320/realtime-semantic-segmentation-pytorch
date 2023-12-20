"""
Paper:      Fast Semantic Segmentation for Scene Perception
Url:        https://ieeexplore.ieee.org/document/8392426
Create by:  zh320
Date:       2023/10/22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ConvBNAct, DeConvBNAct, Activation
from .enet import InitialBlock as InitBlock


class FSSNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='prelu'):
        super(FSSNet, self).__init__()
        # Encoder
        self.init_block = InitBlock(n_channel, 16, act_type)
        self.down1 = DownsamplingBlock(16, 64, act_type)
        self.factorized = build_blocks(FactorizedBlock, 64, 4, act_type=act_type)
        self.down2 = DownsamplingBlock(64, 128, act_type)
        self.dilated = build_blocks(DilatedBlock, 128, 6, [2,5,9,2,5,9], act_type)
        # Decoder
        self.up2 = UpsamplingBlock(128, 64, act_type)
        self.bottleneck2 = build_blocks(DilatedBlock, 64, 2, act_type=act_type)
        self.up1 = UpsamplingBlock(64, 16, act_type)
        self.bottleneck1 = build_blocks(DilatedBlock, 16, 2, act_type=act_type)
        self.full_conv = DeConvBNAct(16, num_class, act_type=act_type)

    def forward(self, x):
        x = self.init_block(x)      # 2x down
        x_d1 = self.down1(x)        # 4x down
        x = self.factorized(x_d1)
        x_d2 = self.down2(x)        # 8x down
        x = self.dilated(x_d2)

        x = self.up2(x, x_d2)       # 8x up
        x = self.bottleneck2(x)
        x = self.up1(x, x_d1)       # 4x up
        x = self.bottleneck1(x)
        x = self.full_conv(x)       # 2x up

        return x


def build_blocks(block, channels, num_block, dilations=[], act_type='relu'):
    if len(dilations) == 0:
        dilations = [1 for _ in range(num_block)]
    else:
        if len(dilations) != num_block:
            raise ValueError(f'Number of dilation should be equal to number of blocks')

    layers = []
    for i in range(num_block):
        layers.append(block(channels, dilations[i], act_type))
    return  nn.Sequential(*layers)


class FactorizedBlock(nn.Module):
    def __init__(self, channels, dilation=1, act_type='relu'):
        super(FactorizedBlock, self).__init__()
        hid_channels = channels // 4
        self.conv = nn.Sequential(
                        ConvBNAct(channels, hid_channels, 1, act_type=act_type),
                        ConvBNAct(hid_channels, hid_channels, (1,3), act_type='none'),
                        ConvBNAct(hid_channels, hid_channels, (3,1), act_type=act_type),
                        ConvBNAct(hid_channels, channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x

        x = self.conv(x)
        x += residual

        return self.act(x)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation, act_type):
        super(DilatedBlock, self).__init__()
        hid_channels = channels // 4
        self.conv = nn.Sequential(
                        ConvBNAct(channels, hid_channels, 1, act_type=act_type),
                        ConvBNAct(hid_channels, hid_channels, 3, dilation=dilation, act_type=act_type),
                        ConvBNAct(hid_channels, channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)
        
    def forward(self, x):
        residual = x

        x = self.conv(x)
        x += residual

        return self.act(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(DownsamplingBlock, self).__init__()
        hid_channels = out_channels // 4
        self.conv = nn.Sequential(
                        ConvBNAct(in_channels, hid_channels, 2, 2, act_type=act_type),
                        ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type),
                        ConvBNAct(hid_channels, out_channels, 1, act_type='none')
                    )
        self.pool = nn.Sequential(
                        nn.MaxPool2d(3, 2, 1),
                        ConvBNAct(in_channels, out_channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        x_pool = self.pool(x)
        x = self.conv(x)
        x += x_pool

        return self.act(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(UpsamplingBlock, self).__init__()
        hid_channels = in_channels // 4
        self.deconv = nn.Sequential(
                            ConvBNAct(in_channels, hid_channels, 1, act_type=act_type),
                            DeConvBNAct(hid_channels, hid_channels, act_type=act_type),
                            ConvBNAct(hid_channels, out_channels, 1, act_type='none')
                        )
        self.conv = ConvBNAct(in_channels, out_channels, 1, act_type='none')
        self.act = Activation(act_type)

    def forward(self, x, pool_feat):
        x_deconv = self.deconv(x)

        x = x + pool_feat
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x += x_deconv

        return self.act(x)
