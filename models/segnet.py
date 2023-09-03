"""
Paper:      SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
Url:        https://arxiv.org/abs/1511.00561
Create by:  zh320
Date:       2023/08/20
"""

import torch
import torch.nn as nn

from .modules import ConvBNAct


class SegNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, hid_channel=64, act_type='relu'):
        super(SegNet, self).__init__()
        self.down_stage1 = DownsampleBlock(n_channel, hid_channel, act_type, False)
        self.down_stage2 = DownsampleBlock(hid_channel, hid_channel*2, act_type, False)
        self.down_stage3 = DownsampleBlock(hid_channel*2, hid_channel*4, act_type, True)
        self.down_stage4 = DownsampleBlock(hid_channel*4, hid_channel*8, act_type, True)
        self.down_stage5 = DownsampleBlock(hid_channel*8, hid_channel*8, act_type, True)
        self.up_stage5 = UpsampleBlock(hid_channel*8, hid_channel*8, act_type, True)
        self.up_stage4 = UpsampleBlock(hid_channel*8, hid_channel*4, act_type, True)
        self.up_stage3 = UpsampleBlock(hid_channel*4, hid_channel*2, act_type, True)
        self.up_stage2 = UpsampleBlock(hid_channel*2, hid_channel, act_type, False)
        self.up_stage1 = UpsampleBlock(hid_channel, hid_channel, act_type, False)
        self.classifier = ConvBNAct(hid_channel, num_class, act_type=act_type)

    def forward(self, x):
        x, indices1 = self.down_stage1(x)
        x, indices2 = self.down_stage2(x)
        x, indices3 = self.down_stage3(x)
        x, indices4 = self.down_stage4(x)
        x, indices5 = self.down_stage5(x)
        x = self.up_stage5(x, indices5)
        x = self.up_stage4(x, indices4)
        x = self.up_stage3(x, indices3)
        x = self.up_stage2(x, indices2)
        x = self.up_stage1(x, indices1)
        x = self.classifier(x)
        
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu', extra_conv=False):
        super(DownsampleBlock, self).__init__()
        layers = [ConvBNAct(in_channels, out_channels, 3, act_type=act_type, inplace=True),
                  ConvBNAct(out_channels, out_channels, 3, act_type=act_type, inplace=True)]
        if extra_conv:
            layers.append(ConvBNAct(out_channels, out_channels, 3, act_type=act_type, inplace=True))
        self.conv = nn.Sequential(*layers)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        x, indices = self.pool(x)
        return x, indices


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu', extra_conv=False):
        super(UpsampleBlock, self).__init__()
        self.pool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        
        hid_channel = in_channels if extra_conv else out_channels
        
        layers = [ConvBNAct(in_channels, in_channels, 3, act_type=act_type, inplace=True),
                  ConvBNAct(in_channels, hid_channel, 3, act_type=act_type, inplace=True)]
                  
        if extra_conv:
            layers.append(ConvBNAct(in_channels, out_channels, 3, act_type=act_type, inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x, indices):
        x = self.pool(x, indices)
        x = self.conv(x)
        
        return x
