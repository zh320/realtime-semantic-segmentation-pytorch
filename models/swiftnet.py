"""
Paper:      In Defense of Pre-trained ImageNet Architectures for Real-time 
            Semantic Segmentation of Road-driving Images
Url:        https://arxiv.org/abs/1903.08469
Create by:  zh320
Date:       2023/10/22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, PWConvBNAct, ConvBNAct, PyramidPoolingModule
from .backbone import ResNet, Mobilenetv2


class SwiftNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='resnet18', up_channels=128, 
                    act_type='relu'):
        super(SwiftNet, self).__init__()
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [64, 128, 256, 512] if backbone_type in ['resnet18', 'resnet34'] else [256, 512, 1024, 2048]
        elif backbone_type == 'mobilenet_v2':
            self.backbone = Mobilenetv2()
            channels = [24, 32, 96, 320]
        else:
            raise NotImplementedError()

        self.connection1 = ConvBNAct(channels[0], up_channels, 1, act_type=act_type)
        self.connection2 = ConvBNAct(channels[1], up_channels, 1, act_type=act_type)
        self.connection3 = ConvBNAct(channels[2], up_channels, 1, act_type=act_type)
        self.spp = PyramidPoolingModule(channels[3], up_channels, act_type, bias=True)
        self.decoder = Decoder(up_channels, num_class, act_type)

    def forward(self, x):
        size = x.size()[2:]

        x1, x2, x3, x4 = self.backbone(x)

        x1 = self.connection1(x1)
        x2 = self.connection2(x2)
        x3 = self.connection3(x3)
        x4 = self.spp(x4)

        x = self.decoder(x4, x1, x2, x3)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class Decoder(nn.Module):
    def __init__(self, channels, num_class, act_type):
        super(Decoder, self).__init__()
        self.up_stage3 = ConvBNAct(channels, channels, 3, act_type=act_type)
        self.up_stage2 = ConvBNAct(channels, channels, 3, act_type=act_type)
        self.up_stage1 = ConvBNAct(channels, num_class, 3, act_type=act_type)
        
    def forward(self, x, x1, x2, x3):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x += x3
        x = self.up_stage3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x += x2
        x = self.up_stage2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x += x1
        x = self.up_stage1(x)

        return x
