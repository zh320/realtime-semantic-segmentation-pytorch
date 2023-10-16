"""
Paper:      FDDWNet: A Lightweight Convolutional Neural Network for Real-time 
            Sementic Segmentation
Url:        https://arxiv.org/abs/1911.00632
Create by:  zh320
Date:       2023/10/08
"""

import torch
import torch.nn as nn

from .modules import DWConvBNAct, ConvBNAct, Activation


class FDDWNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, ks=3, act_type='relu'):
        super(FDDWNet, self).__init__()
        self.layer1 = DownsamplingUnit(n_channel, 16, act_type)
        self.layer2 = DownsamplingUnit(16, 64, act_type)
        self.layer3_7 = build_blocks(EERMUnit, 64, 5, ks, [1,1,1,1,1], act_type)
        self.layer8 = DownsamplingUnit(64, 128, act_type)
        self.layer9_16 = build_blocks(EERMUnit, 128, 8, ks, [1,2,5,9,1,2,5,9], act_type)
        self.layer17_24 = build_blocks(EERMUnit, 128, 8, ks, [2,5,9,17,2,5,9,17], act_type)
        self.layer25 = Upsample(128, 64, act_type=act_type)
        self.layer26_27 = build_blocks(EERMUnit, 64, 2, ks, [1,1], act_type)
        self.layer28 = Upsample(64, 16, act_type=act_type)
        self.layer29_30 = build_blocks(EERMUnit, 16, 2, ks, [1,1], act_type)
        self.layer31 = Upsample(16, num_class, act_type=act_type)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        residual = self.layer3_7(x)
        x = self.layer8(residual)
        x = self.layer9_16(x)
        x = self.layer17_24(x)
        x = self.layer25(x)
        x = self.layer26_27(x)
        x += residual
        x = self.layer28(x)
        x = self.layer29_30(x)
        x = self.layer31(x)

        return x


def build_blocks(block, channels, num_block, kernel_size, dilations=[], act_type='relu'):
    if len(dilations) == 0:
        dilations = [1 for _ in range(num_block)]
    else:
        if len(dilations) != num_block:
            raise ValueError(f'Number of dilation should be equal to number of blocks')

    layers = []
    for i in range(num_block):
        layers.append(block(channels, kernel_size, dilations[i], act_type))
    return  nn.Sequential(*layers)


class EERMUnit(nn.Module):
    def __init__(self, channels, ks, dt, act_type):
        super(EERMUnit, self).__init__()
        self.conv = nn.Sequential(
                        DWConvBNAct(channels, channels, (ks, 1), act_type='none'),
                        DWConvBNAct(channels, channels, (1, ks), act_type='none'),
                        ConvBNAct(channels, channels, 1, act_type=act_type, inplace=True),
                        DWConvBNAct(channels, channels, (ks, 1), dilation=dt, act_type='none'),
                        DWConvBNAct(channels, channels, (1, ks), dilation=dt, act_type='none'),
                        ConvBNAct(channels, channels, 1, act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x

        x = self.conv(x)
        x += residual

        return self.act(x)


class DownsamplingUnit(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(DownsamplingUnit, self).__init__()
        assert out_channels > in_channels, 'out_channels should be larger than in_channels.\n'
        self.conv = ConvBNAct(in_channels, out_channels - in_channels, 3, 2, act_type=act_type, inplace=True)
        self.pool = nn.MaxPool2d(3, 2, 1)

    def forward(self, x):
        x = torch.cat([self.conv(x), self.pool(x)], dim=1)
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
                                    ConvBNAct(in_channels, out_channels, 1, act_type=act_type, inplace=True),
                                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                            )

    def forward(self, x):
        return self.up_conv(x)
