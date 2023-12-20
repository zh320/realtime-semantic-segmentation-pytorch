"""
Paper:      Speeding up Semantic Segmentation for Autonomous Driving
Url:        https://openreview.net/pdf?id=S1uHiFyyg
Create by:  zh320
Date:       2023/10/22
"""

import torch
import torch.nn as nn

from .modules import ConvBNAct, DeConvBNAct, Activation


class SQNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='elu'):
        super(SQNet, self).__init__()
        # Encoder, SqueezeNet-1.1
        self.conv = ConvBNAct(n_channel, 64, 3, 2, act_type=act_type)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.fire1 = nn.Sequential(
                            FireModule(64, 16, 64, 64, act_type),
                            FireModule(128, 16, 64, 64, act_type)
                        )
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        self.fire2 = nn.Sequential(
                            FireModule(128, 32, 128, 128, act_type),
                            FireModule(256, 32, 128, 128, act_type)
                        )
        self.pool3 = nn.MaxPool2d(3, 2, 1)
        self.fire3 = nn.Sequential(
                            FireModule(256, 48, 192, 192, act_type),
                            FireModule(384, 48, 192, 192, act_type),
                            FireModule(384, 64, 256, 256, act_type),
                            FireModule(512, 64, 256, 256, act_type)
                        )
        # Decoder
        self.pdc = ParallelDilatedConv(512, 128, [1,2,4,8], act_type)
        self.up1 = DeConvBNAct(128, 128, act_type=act_type)
        self.refine1 = BypassRefinementModule(256, 128, 128, act_type)
        self.up2 = DeConvBNAct(128, 128, act_type=act_type)
        self.refine2 = BypassRefinementModule(128, 128, 64, act_type=act_type)
        self.up3 = DeConvBNAct(64, 64, act_type=act_type)
        self.refine3 = BypassRefinementModule(64, 64, num_class, act_type=act_type)
        self.up4 = DeConvBNAct(num_class, num_class, act_type=act_type)

    def forward(self, x):
        x1 = self.conv(x)
        x = self.pool1(x1)
        x2 = self.fire1(x)
        x = self.pool2(x2)
        x3 = self.fire2(x)
        x = self.pool3(x3)
        x = self.fire3(x)
        x = self.pdc(x)
        x = self.up1(x)
        x = self.refine1(x3, x)
        x = self.up2(x)
        x = self.refine2(x2, x)
        x = self.up3(x)
        x = self.refine3(x1, x)
        x = self.up4(x)

        return x


class FireModule(nn.Module):
    def __init__(self, in_channels, sq_channels, ex1_channels, ex3_channels, act_type):
        super(FireModule, self).__init__()
        self.conv_squeeze = ConvBNAct(in_channels, sq_channels, 1, act_type=act_type)
        self.conv_expand1 = ConvBNAct(sq_channels, ex1_channels, 1, act_type=act_type)
        self.conv_expand3 = ConvBNAct(sq_channels, ex3_channels, 3, act_type=act_type)

    def forward(self, x):
        x = self.conv_squeeze(x)
        x1 = self.conv_expand1(x)
        x3 = self.conv_expand3(x)
        x = torch.cat([x1, x3], dim=1)

        return x


class ParallelDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, act_type):
        super(ParallelDilatedConv, self).__init__()
        assert len(dilations) == 4, 'Length of dilations should be 4.\n'
        self.conv0 = ConvBNAct(in_channels, out_channels, 3, dilation=dilations[0], act_type=act_type)
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, dilation=dilations[1], act_type=act_type)
        self.conv2 = ConvBNAct(in_channels, out_channels, 3, dilation=dilations[2], act_type=act_type)
        self.conv3 = ConvBNAct(in_channels, out_channels, 3, dilation=dilations[3], act_type=act_type)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = x0 + x1 + x2 + x3

        return x


class BypassRefinementModule(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels, act_type):
        super(BypassRefinementModule, self).__init__()
        self.conv_low = ConvBNAct(low_channels, low_channels, 3, act_type=act_type)
        self.conv_cat = ConvBNAct(low_channels + high_channels, out_channels, 3, act_type=act_type)

    def forward(self, x_low, x_high):
        x_low = self.conv_low(x_low)
        x = torch.cat([x_low, x_high], dim=1)
        x = self.conv_cat(x)

        return x
