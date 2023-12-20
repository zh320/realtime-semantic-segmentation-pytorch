"""
Paper:      BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/1808.00897
Create by:  zh320
Date:       2023/09/03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, SegHead
from .backbone import ResNet


class BiSeNetv1(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='resnet18', act_type='relu',):
        super(BiSeNetv1, self).__init__()
        self.spatial_path = SpatialPath(n_channel, 128, act_type=act_type)
        self.context_path = ContextPath(256, backbone_type, act_type=act_type)
        self.ffm = FeatureFusionModule(384, 256, act_type=act_type)
        self.seg_head = SegHead(256, num_class, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]
        x_s = self.spatial_path(x)
        x_c = self.context_path(x)
        x = self.ffm(x_s, x_c)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class SpatialPath(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type):
        super(SpatialPath, self).__init__(
            ConvBNAct(in_channels, out_channels, 3, 2, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, 2, act_type=act_type),
            ConvBNAct(out_channels, out_channels, 3, 2, act_type=act_type),
        )


class ContextPath(nn.Module):
    def __init__(self, out_channels, backbone_type, act_type):
        super(ContextPath, self).__init__()
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [256, 512] if ('18' in backbone_type) or ('34' in backbone_type) else [1024, 2048]
        else:
            raise NotImplementedError()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.arm_16 = AttentionRefinementModule(channels[0])
        self.arm_32 = AttentionRefinementModule(channels[1])

        self.conv_16 = conv1x1(channels[0], out_channels)
        self.conv_32 = conv1x1(channels[1], out_channels)

    def forward(self, x):
        _, _, x_16, x_32 = self.backbone(x)
        x_32_avg = self.pool(x_32)
        x_32 = self.arm_32(x_32)
        x_32 += x_32_avg
        x_32 = self.conv_32(x_32)
        x_32 = F.interpolate(x_32, scale_factor=2, mode='bilinear', align_corners=True)

        x_16 = self.arm_16(x_16)
        x_16 = self.conv_16(x_16)
        x_16 += x_32
        x_16 = F.interpolate(x_16, scale_factor=2, mode='bilinear', align_corners=True)
        
        return x_16


class AttentionRefinementModule(nn.Module):
    def __init__(self, channels):
        super(AttentionRefinementModule, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = ConvBNAct(channels, channels, 1, act_type='sigmoid')

    def forward(self, x):
        x_pool = self.pool(x)
        x_pool = x_pool.expand_as(x)
        x_pool = self.conv(x_pool)
        x = x * x_pool

        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(FeatureFusionModule, self).__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, act_type=act_type)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Sequential(
                                conv1x1(out_channels, out_channels),
                                nn.ReLU(),
                                conv1x1(out_channels, out_channels),
                                nn.Sigmoid(),
                            )

    def forward(self, x_low, x_high):
        x = torch.cat([x_low, x_high], dim=1)
        x = self.conv1(x)

        x_pool = self.pool(x)
        x_pool = x_pool.expand_as(x)
        x_pool = self.conv2(x_pool)

        x_pool = x * x_pool
        x = x + x_pool

        return x
