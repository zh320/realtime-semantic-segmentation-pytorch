"""
Paper:      LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
Url:        https://arxiv.org/abs/1707.03718
Create by:  zh320
Date:       2023/04/23
"""

import torch
import torch.nn as nn

from .modules import ConvBNAct, DeConvBNAct, Activation
from .backbone import ResNet


class LinkNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='resnet18', act_type='relu'):
        super(LinkNet, self).__init__()
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [64, 128, 256, 512] if backbone_type in ['resnet18', 'resnet34'] else [256, 512, 1024, 2048]
        else:
            raise NotImplementedError()

        self.dec_block4 = DecoderBlock(channels[3], channels[2], act_type)
        self.dec_block3 = DecoderBlock(channels[2], channels[1], act_type)
        self.dec_block2 = DecoderBlock(channels[1], channels[0], act_type)
        self.dec_block1 = DecoderBlock(channels[0], channels[0], act_type, scale_factor=1)
        self.seg_head = SegHead(channels[0], num_class, act_type)

    def forward(self, x):
        x_1, x_2, x_3, x_4 = self.backbone(x)
        x = self.dec_block4(x_4)
        x = self.dec_block3(x + x_3)
        x = self.dec_block2(x + x_2)
        x = self.dec_block1(x + x_1)
        x = self.seg_head(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, scale_factor=2):
        super(DecoderBlock, self).__init__()
        hid_channels = in_channels // 4
        self.conv1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        if scale_factor > 1:
            self.full_conv = DeConvBNAct(hid_channels, hid_channels, scale_factor, act_type=act_type)
        else:
            self.full_conv = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.conv2 = ConvBNAct(hid_channels, out_channels, 1, act_type=act_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.full_conv(x)
        x = self.conv2(x)

        return x


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, scale_factor=2):
        hid_channels = in_channels // 2
        super(SegHead, self).__init__(
                DeConvBNAct(in_channels, hid_channels, scale_factor, act_type=act_type),
                ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type),
                DeConvBNAct(hid_channels, num_class, scale_factor, act_type=act_type)
        )
