"""
Paper:      ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/1906.09826
Create by:  zh320
Date:       2023/09/24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ConvBNAct, DeConvBNAct, Activation
from .enet import InitialBlock as DownsamplingUnit


class ESNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='relu'):
        super(ESNet, self).__init__()
        self.block1_down = DownsamplingUnit(n_channel, 16, act_type)
        self.block1 = build_blocks('fcu', 16, 3, K=3, act_type=act_type)
        self.block2_down = DownsamplingUnit(16, 64, act_type)
        self.block2 = build_blocks('fcu', 64, 2, K=5, act_type=act_type)
        self.block3_down = DownsamplingUnit(64, 128, act_type)
        self.block3 = build_blocks('pfcu', 128, 3, r1=2, r2=5, r3=9, act_type=act_type)
        self.block4_up = DeConvBNAct(128, 64, act_type=act_type)
        self.block4 = build_blocks('fcu', 64, 2, K=5, act_type=act_type)
        self.block5_up = DeConvBNAct(64, 16, act_type=act_type)
        self.block5 = build_blocks('fcu', 16, 2, K=3, act_type=act_type)
        self.full_conv = DeConvBNAct(16, num_class, act_type=act_type)

    def forward(self, x):
        x = self.block1_down(x)
        x = self.block1(x)

        x = self.block2_down(x)
        x = self.block2(x)

        x = self.block3_down(x)
        x = self.block3(x)

        x = self.block4_up(x)
        x = self.block4(x)

        x = self.block5_up(x)
        x = self.block5(x)

        x = self.full_conv(x)

        return x


def build_blocks(block_type, channels, num_block, K=None, r1=None, r2=None, r3=None, 
                act_type='relu'):
    layers = []
    for _ in range(num_block):
        if block_type == 'fcu':
            layers.append(FCU(channels, K, act_type))
        elif block_type == 'pfcu':
            layers.append(PFCU(channels, r1, r2, r3, act_type))
        else:
            raise NotImplementedError(f'Unsupported block type: {block_type}.\n')
    return nn.Sequential(*layers)


class FCU(nn.Module):
    def __init__(self, channels, K, act_type):
        super(FCU, self).__init__()
        assert K is not None, 'K should not be None.\n'
        padding = (K - 1) // 2
        self.conv = nn.Sequential(
                        nn.Conv2d(channels, channels, (K, 1), padding=(padding, 0), bias=False),
                        Activation(act_type, inplace=True),
                        ConvBNAct(channels, channels, (1, K), act_type=act_type, inplace=True),
                        nn.Conv2d(channels, channels, (K, 1), padding=(padding, 0), bias=False),
                        Activation(act_type, inplace=True),
                        ConvBNAct(channels, channels, (1, K), act_type='none')
                    )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x

        x = self.conv(x)
        x += residual

        return self.act(x)


class PFCU(nn.Module):
    def __init__(self, channels, r1, r2, r3, act_type):
        super(PFCU, self).__init__()
        assert (r1 is not None) and (r2 is not None) and (r3 is not None)

        self.conv0 = nn.Sequential(
                        nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), bias=False),
                        Activation(act_type, inplace=True),
                        ConvBNAct(channels, channels, (1, 3), act_type=act_type, inplace=True)
                    )
        self.conv_left = nn.Sequential(
                            nn.Conv2d(channels, channels, (3, 1), padding=(r1, 0), 
                                        dilation=r1, bias=False),
                            Activation(act_type, inplace=True),
                            ConvBNAct(channels, channels, (1, 3), dilation=r1, act_type='none')
                        )
        self.conv_mid = nn.Sequential(
                            nn.Conv2d(channels, channels, (3, 1), padding=(r2, 0), 
                                        dilation=r2, bias=False),
                            Activation(act_type, inplace=True),
                            ConvBNAct(channels, channels, (1, 3), dilation=r2, act_type='none')
                        )
        self.conv_right = nn.Sequential(
                            nn.Conv2d(channels, channels, (3, 1), padding=(r3, 0), 
                                        dilation=r3, bias=False),
                            Activation(act_type, inplace=True),
                            ConvBNAct(channels, channels, (1, 3), dilation=r3, act_type='none')
                        )
        self.act = Activation(act_type)

    def forward(self, x):
        residual = x

        x = self.conv0(x)

        x_left = self.conv_left(x)
        x_mid = self.conv_mid(x)
        x_right = self.conv_right(x)

        x = x_left + x_mid + x_right + residual

        return  self.act(x)
