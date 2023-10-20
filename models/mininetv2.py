"""
Paper:      MiniNet: An Efficient Semantic Segmentation ConvNet for Real-Time Robotic Applications
Url:        https://ieeexplore.ieee.org/abstract/document/9023474
Create by:  zh320
Date:       2023/10/15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import DWConvBNAct, PWConvBNAct, ConvBNAct, Activation


class MiniNetv2(nn.Module):
    def __init__(self, num_class=1, n_channel=3, feat_dt=[1,2,1,4,1,8,1,16,1,1,1,2,1,4,1,8],
                    act_type='relu'):
        super(MiniNetv2, self).__init__()
        self.d1_2 = nn.Sequential(
                        DownsamplingUnit(n_channel, 16, act_type),
                        DownsamplingUnit(16, 64, act_type),
                    )
        self.ref = nn.Sequential(
                        DownsamplingUnit(n_channel, 16, act_type),
                        DownsamplingUnit(16, 64, act_type)
                    )
        self.m1_10 = build_blocks(MultiDilationDSConv, 64, 10, act_type=act_type)
        self.d3 = DownsamplingUnit(64, 128, act_type)
        self.feature_extractor = build_blocks(MultiDilationDSConv, 128, len(feat_dt), feat_dt, act_type)
        self.up1 = Upsample(128, 64, act_type=act_type)
        self.m26_29 = build_blocks(MultiDilationDSConv, 64, 4, act_type=act_type)
        self.output = Upsample(64, num_class, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]

        x_ref = self.ref(x)

        x = self.d1_2(x)
        x = self.m1_10(x)
        x = self.d3(x)
        x = self.feature_extractor(x)
        x = self.up1(x)
        x += x_ref

        x = self.m26_29(x)
        x = self.output(x)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class DownsamplingUnit(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(DownsamplingUnit, self).__init__()
        assert out_channels > in_channels, 'out_channels should be larger than in_channels.\n'
        self.conv = ConvBNAct(in_channels, out_channels - in_channels, 3, 2, act_type=act_type, inplace=True)
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
        layers.append(block(channels, channels, 3, 1, dilations[i], act_type))
    return  nn.Sequential(*layers)


class MultiDilationDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, act_type='relu'):
        super(MultiDilationDSConv, self).__init__()
        self.dilated = dilation > 1
        self.dw_conv = DWConvBNAct(in_channels, in_channels, kernel_size, stride, 1, act_type)
        self.pw_conv = PWConvBNAct(in_channels, out_channels, act_type, inplace=True)
        if self.dilated:
            self.ddw_conv = DWConvBNAct(in_channels, in_channels, kernel_size, stride, dilation, act_type, inplace=True)

    def forward(self, x):
        x_dw = self.dw_conv(x)
        if self.dilated:
            x_ddw = self.ddw_conv(x)
            x_dw += x_ddw
        x = self.pw_conv(x_dw)

        return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None, padding=None,
                    upsample_type='deconvolution', act_type='relu'):
        super(Upsample, self).__init__()
        if upsample_type == 'deconvolution':
            if kernel_size is None:
                kernel_size = 2*scale_factor - 1
            if padding is None:    
                padding = (kernel_size - 1) // 2
            output_padding = scale_factor - 1
            self.up_conv = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels, out_channels, 
                                                        kernel_size=kernel_size, 
                                                        stride=scale_factor, padding=padding,
                                                        output_padding=output_padding),
                                    nn.BatchNorm2d(out_channels),
                                    Activation(act_type)
                            )
        else:
            self.up_conv = nn.Sequential(
                                    ConvBNAct(in_channels, out_channels, 1, act_type=act_type),
                                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                            )

    def forward(self, x):
        return self.up_conv(x)
