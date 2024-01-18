"""
Paper:      Rethinking Dilated Convolution for Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/2111.09957
Create by:  zh320
Date:       2024/01/13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, Activation


class RegSeg(nn.Module):
    def __init__(self, num_class=1, n_channel=3, dilations=None, act_type='relu'):
        super(RegSeg, self).__init__()
        if dilations is None:
            dilations = [[1,1], [1,2], [1,2], [1,3], [2,3], [2,7], [2,3],
                         [2,6], [2,5], [2,9], [2,11], [4,7], [5,14]]
        else:
            if len(dilations) != 13:
                raise ValueError("Dilation pairs' length should be 13\n")

        # Backbone-1
        self.conv_init = ConvBNAct(n_channel, 32, 3, 2, act_type=act_type)
        
        # Backbone-2
        self.stage_d4 = DBlock(32, 48, 2, act_type=act_type)

        # Backbone-3
        layers = [DBlock(48, 128, 2, act_type=act_type)]
        for _ in range(3-1):
            layers.append(DBlock(128, 128, 1, r1=1, r2=1, act_type=act_type))
        self.stage_d8 = nn.Sequential(*layers)

        # Backbone-4
        layers = [DBlock(128, 256, 2, act_type=act_type)]
        for i in range(13-1):
            layers.append(DBlock(256, 256, 1, r1=dilations[i][0], r2=dilations[i][1], act_type=act_type))

        # Backbone-5
        layers.append(DBlock(256, 320, 2, r1=dilations[-1][0], r2=dilations[-1][1], act_type=act_type))
        self.stage_d16 = nn.Sequential(*layers)

        # Decoder
        self.decoder = Decoder(num_class, 48, 128, 320, act_type)

    def forward(self, x):
        size = x.size()[2:]

        x = self.conv_init(x)               # 2x down
        x_d4 = self.stage_d4(x)             # 4x down
        x_d8 = self.stage_d8(x_d4)          # 8x down
        x_d16 = self.stage_d16(x_d8)        # 16x down
        x = self.decoder(x_d4, x_d8, x_d16) # 4x down
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, r1=None, r2=None, 
                    g=16, se_ratio=0.25, act_type='relu'):
        super(DBlock, self).__init__()
        assert stride in [1, 2], f'Unsupported stride: {stride}'
        self.stride = stride

        self.conv1 = ConvBNAct(in_channels, out_channels, 1, act_type=act_type)
        if stride == 1:
            assert in_channels == out_channels, 'In_channels should be the same as out_channels when stride = 1'
            split_ch = out_channels // 2
            assert split_ch % g == 0, 'Group width `g` should be evenly divided by split_ch'
            groups = split_ch // g
            self.split_channels = split_ch
            self.conv_left = ConvBNAct(split_ch, split_ch, 3, dilation=r1, groups=groups, act_type=act_type)
            self.conv_right = ConvBNAct(split_ch, split_ch, 3, dilation=r2, groups=groups, act_type=act_type)
        else:   # stride == 2
            assert out_channels % g == 0, 'Group width `g` should be evenly divided by out_channels'
            groups = out_channels // g
            self.conv_left = ConvBNAct(out_channels, out_channels, 3, 2, groups=groups, act_type=act_type)
            self.conv_skip = nn.Sequential(
                                nn.AvgPool2d(2, 2, 0),
                                ConvBNAct(in_channels, out_channels, 1, act_type='none')
                            )
        self.conv2 = nn.Sequential(
                        SEBlock(out_channels, se_ratio, act_type),
                        ConvBNAct(out_channels, out_channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        if self.stride == 1:
            x_left = self.conv_left(x[:, :self.split_channels])
            x_right = self.conv_right(x[:,self.split_channels:])
            x = torch.cat([x_left, x_right], dim=1)
        else:
            x = self.conv_left(x)
            residual = self.conv_skip(residual)

        x = self.conv2(x)
        x += residual

        return self.act(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio, act_type):
        super(SEBlock, self).__init__()
        squeeze_channels = int(channels * reduction_ratio)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se_block = nn.Sequential(
                            nn.Linear(channels, squeeze_channels),
                            Activation(act_type),
                            nn.Linear(squeeze_channels, channels),
                            Activation('sigmoid')
                        )

    def forward(self, x):
        residual = x
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.se_block(x).unsqueeze(-1).unsqueeze(-1)
        x = x * residual

        return x


class Decoder(nn.Module):
    def __init__(self, num_class, d4_channel, d8_channel, d16_channel, act_type):
        super(Decoder, self).__init__()
        self.conv_d16 = ConvBNAct(d16_channel, 128, 1, act_type=act_type)
        self.conv_d8_stage1 = ConvBNAct(d8_channel, 128, 1, act_type=act_type)
        self.conv_d4_stage1 = ConvBNAct(d4_channel, 8, 1, act_type=act_type)
        self.conv_d8_stage2 = ConvBNAct(128, 64, 3, act_type=act_type)
        self.conv_d4_stage2 = nn.Sequential(
                                    ConvBNAct(64+8, 64, 3, act_type=act_type),
                                    conv1x1(64, num_class)
                                )

    def forward(self, x_d4, x_d8, x_d16):
        size_d4 = x_d4.size()[2:]
        size_d8 = x_d8.size()[2:]

        x_d16 = self.conv_d16(x_d16)
        x_d16 = F.interpolate(x_d16, size_d8, mode='bilinear', align_corners=True)

        x_d8 = self.conv_d8_stage1(x_d8)
        x_d8 += x_d16
        x_d8 = self.conv_d8_stage2(x_d8)
        x_d8 = F.interpolate(x_d8, size_d4, mode='bilinear', align_corners=True)

        x_d4 = self.conv_d4_stage1(x_d4)
        x_d4 = torch.cat([x_d4, x_d8], dim=1)
        x_d4 = self.conv_d4_stage2(x_d4)

        return x_d4
