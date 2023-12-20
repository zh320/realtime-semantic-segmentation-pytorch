"""
Paper:      Cross Attention Network for Semantic Segmentation
Url:        https://arxiv.org/abs/1907.10958
Create by:  zh320
Date:       2023/09/30
"""

import torch
import torch.nn as nn

from .modules import ConvBNAct, DeConvBNAct, Activation
from .backbone import ResNet, Mobilenetv2


class CANet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='mobilenet_v2', act_type='relu'):
        super(CANet, self).__init__()
        self.spatial_branch = SpatialBranch(n_channel, 64, act_type)
        self.context_branch = ContextBranch(64*4, backbone_type)
        self.fca = FeatureCrossAttentionModule(64*4, num_class, act_type)
        self.up = DeConvBNAct(num_class, num_class, scale_factor=8)

    def forward(self, x):
        size = x.size()[2:]
        x_s = self.spatial_branch(x)
        x_c = self.context_branch(x)
        x = self.fca(x_s, x_c)
        x = self.up(x)

        return x


class SpatialBranch(nn.Sequential):
    def __init__(self, n_channel, channels, act_type):
        super(SpatialBranch, self).__init__(
            ConvBNAct(n_channel, channels, 3, 2, act_type=act_type, inplace=True),
            ConvBNAct(channels, channels*2, 3, 2, act_type=act_type, inplace=True),
            ConvBNAct(channels*2, channels*4, 3, 2, act_type=act_type, inplace=True),
        )


class ContextBranch(nn.Module):
    def __init__(self, out_channels, backbone_type, hid_channels=192):
        super(ContextBranch, self).__init__()
        if 'mobilenet' in backbone_type:
            self.backbone = Mobilenetv2()
            channels = [320, 96]
        elif 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [512, 256] if (('18' in backbone_type) or ('34' in backbone_type)) else [2048, 1024]
        else:
            raise NotImplementedError()
            
        self.up1 = DeConvBNAct(channels[0], hid_channels)
        self.up2 = DeConvBNAct(channels[1] + hid_channels, out_channels)

    def forward(self, x):
        _, _, x_d16, x = self.backbone(x)
        x = self.up1(x)

        x = torch.cat([x, x_d16], dim=1)
        x = self.up2(x)

        return x


class FeatureCrossAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(FeatureCrossAttentionModule, self).__init__()
        self.conv_init = ConvBNAct(2*in_channels, in_channels, act_type=act_type, inplace=True)
        self.sa = SpatialAttentionBlock(in_channels)
        self.ca = ChannelAttentionBlock(in_channels)
        self.conv_last = ConvBNAct(in_channels, out_channels, inplace=True)

    def forward(self, x_s, x_c):
        x = torch.cat([x_s, x_c], dim=1)
        x_s = self.sa(x_s)
        x_c = self.ca(x_c)

        x = self.conv_init(x)
        residual = x

        x = x * x_s
        x = x * x_c
        x += residual

        x = self.conv_last(x)

        return x


class SpatialAttentionBlock(nn.Sequential):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__(
            ConvBNAct(in_channels, 1, act_type='sigmoid')
        )


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        x_max = self.max_pool(x).view(-1, self.in_channels)
        x_avg = self.avg_pool(x).view(-1, self.in_channels)

        x_max = self.fc(x_max)
        x_avg = self.fc(x_avg)

        x = x_max + x_avg
        x = torch.sigmoid(x)

        return x.unsqueeze(-1).unsqueeze(-1)
