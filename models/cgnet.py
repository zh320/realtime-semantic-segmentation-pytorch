"""
Paper:      CGNet: A Light-weight Context Guided Network for Semantic Segmentation
Url:        https://arxiv.org/abs/1811.08201
Create by:  zh320
Date:       2023/09/24
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, Activation


class CGNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, M=3, N=15, act_type='prelu'):
        super(CGNet, self).__init__()
        self.stage1 = InitBlock(n_channel, 32, act_type=act_type)
        self.stage2_down = CGBlock(64, 64, 2, 2, act_type=act_type)
        self.stage2 = build_blocks(CGBlock, 64+3, 64, 2, M-1, act_type)
        self.stage3_down = CGBlock(128, 128, 2, 4, act_type=act_type)
        self.stage3 = build_blocks(CGBlock, 128+3, 128, 4, N-1, act_type)
        self.seg_head = conv1x1(128*2, num_class)

    def forward(self, x):
        size = x.size()[2:]
        x_d4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        x_d8 = F.interpolate(x, scale_factor=0.125, mode='bilinear', align_corners=True)

        x, x1 = self.stage1(x)

        x = torch.cat([x, x1], dim=1)
        x2 = self.stage2_down(x)
        x = torch.cat([x2, x_d4], dim=1)        # Input injection
        x = self.stage2(x)

        x = torch.cat([x, x2], dim=1)
        x3 = self.stage3_down(x)
        x = torch.cat([x3, x_d8], dim=1)        # Input injection
        x = self.stage3(x)

        x = torch.cat([x, x3], dim=1)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class InitBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(InitBlock, self).__init__()
        self.conv0 = ConvBNAct(in_channels, out_channels, stride=2, act_type=act_type)
        self.conv1 = ConvBNAct(out_channels, out_channels, act_type=act_type)
        self.conv2 = ConvBNAct(out_channels, out_channels, act_type=act_type)
        
    def forward(self, x):
        x0 = self.conv0(x)
        x = self.conv1(x0)
        x = self.conv2(x)
        return x, x0


def build_blocks(block, in_channels, out_channels, dilation, num_block, act_type):
    layers = []
    for _ in range(num_block):
        layers.append(block(in_channels, out_channels, 1, dilation, act_type=act_type))
        in_channels = out_channels
    return  nn.Sequential(*layers)


class CGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation, res_type='GRL', act_type='prelu'):
        super(CGBlock, self).__init__()
        if res_type not in ['GRL', 'LRL']:
            raise ValueError('Residual learning only support GRL and LRL type.\n')
        self.res_type = res_type
        self.use_skip = (stride == 1) and (in_channels == out_channels)
        self.conv = conv1x1(in_channels, out_channels//2)
        self.loc = nn.Conv2d(out_channels//2, out_channels//2, 3, stride, padding=1, 
                                groups=out_channels//2, bias=False)
        self.sur = nn.Conv2d(out_channels//2, out_channels//2, 3, stride, padding=dilation, 
                                dilation=dilation, groups=out_channels//2, bias=False)
        self.joi = nn.Sequential(
                                nn.BatchNorm2d(out_channels),
                                Activation(act_type)
                            )
        self.glo = nn.Sequential(
                                nn.Linear(out_channels, out_channels//8),
                                nn.Linear(out_channels//8, out_channels)
                            )

    def forward(self, x):
        residual = x
        x = self.conv(x)

        x_loc = self.loc(x)
        x_sur = self.sur(x)

        x = torch.cat([x_loc, x_sur], dim=1)
        x = self.joi(x)

        if self.use_skip and self.res_type == 'LRL':
            x += residual

        x_glo = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x_glo = torch.sigmoid(self.glo(x_glo))
        x_glo = x_glo.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x = x * x_glo

        if self.use_skip and self.res_type == 'GRL':
            x += residual

        return x
