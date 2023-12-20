"""
Paper:      ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation
Url:        https://ieeexplore.ieee.org/document/8063438
Create by:  zh320
Date:       2023/08/20
"""

import torch
import torch.nn as nn

from .modules import ConvBNAct, DeConvBNAct, Activation
from .enet import InitialBlock as DownsamplerBlock


class ERFNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='relu'):
        super(ERFNet, self).__init__()
        self.layer1 = DownsamplerBlock(n_channel, 16, act_type=act_type)

        self.layer2 = DownsamplerBlock(16, 64, act_type=act_type)
        self.layer3_7 = build_blocks(NonBt1DBlock, 64, 5, act_type=act_type)
        
        self.layer8 = DownsamplerBlock(64, 128, act_type=act_type)
        self.layer9_16 = build_blocks(NonBt1DBlock, 128, 8, 
                                        dilations=[2,4,8,16,2,4,8,16], act_type=act_type)
        
        self.layer17 = DeConvBNAct(128, 64, act_type=act_type)
        self.layer18_19 = build_blocks(NonBt1DBlock, 64, 2, act_type=act_type)

        self.layer20 = DeConvBNAct(64, 16, act_type=act_type)
        self.layer21_22 = build_blocks(NonBt1DBlock, 16, 2, act_type=act_type)
        
        self.layer23 = DeConvBNAct(16, num_class, act_type=act_type)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3_7(x)
        x = self.layer8(x)
        x = self.layer9_16(x)
        x = self.layer17(x)
        x = self.layer18_19(x)
        x = self.layer20(x)
        x = self.layer21_22(x)
        x = self.layer23(x)
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


class NonBt1DBlock(nn.Module):
    def __init__(self, channels, dilation=1, act_type='relu'):
        super(NonBt1DBlock, self).__init__()
        self.conv = nn.Sequential(
                                ConvBNAct(channels, channels, (3, 1), inplace=True),
                                ConvBNAct(channels, channels, (1, 3), inplace=True),
                                ConvBNAct(channels, channels, (3, 1), dilation=dilation, inplace=True),
                                nn.Conv2d(channels, channels, (1, 3), dilation=dilation, 
                                            padding=(0, dilation), bias=False)
                            )
        self.bn_act = nn.Sequential(
                                nn.BatchNorm2d(channels),
                                Activation(act_type, inplace=True)
                            )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual
        x = self.bn_act(x)
        return x
