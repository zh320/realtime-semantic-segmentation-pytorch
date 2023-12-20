"""
Paper:      ADSCNet: asymmetric depthwise separable convolution for semantic 
            segmentation in real-time
Url:        https://link.springer.com/article/10.1007/s10489-019-01587-1
Create by:  zh320
Date:       2023/09/30
"""

import torch
import torch.nn as nn

from .modules import conv1x1, ConvBNAct, DWConvBNAct, DeConvBNAct, Activation


class ADSCNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='relu6'):
        super(ADSCNet, self).__init__()
        self.conv0 = ConvBNAct(n_channel, 32, 3, 2, act_type=act_type, inplace=True)
        self.conv1 = ADSCModule(32, 1, act_type=act_type)
        self.conv2_4 = nn.Sequential(
                            ADSCModule(32, 1, act_type=act_type),
                            ADSCModule(32, 2, act_type=act_type),
                            ADSCModule(64, 1, act_type=act_type)
                        )
        self.conv5 = ADSCModule(64, 2, act_type=act_type)
        self.ddcc = DDCC(128, [3, 5, 9, 13], act_type)
        self.up1 = nn.Sequential(
                        DeConvBNAct(128, 64),
                        ADSCModule(64, 1, act_type=act_type)
                    )
        self.up2 = nn.Sequential(
                        ADSCModule(64, 1, act_type=act_type),
                        DeConvBNAct(64, 32)
                    )
        self.up3 = nn.Sequential(
                        ADSCModule(32, 1, act_type=act_type),
                        DeConvBNAct(32, num_class)
                    )

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x4 = self.conv2_4(x1)
        x = self.conv5(x4)
        x = self.ddcc(x)
        x = self.up1(x)
        x += x4
        x = self.up2(x)
        x += x1
        x = self.up3(x)

        return x


class ADSCModule(nn.Module):
    def __init__(self, channels, stride, dilation=1, act_type='relu'):
        super(ADSCModule, self).__init__()
        assert stride in [1, 2], 'Unsupported stride type.\n'
        self.use_skip = stride == 1
        self.conv = nn.Sequential(
                        DWConvBNAct(channels, channels, (3, 1), stride, dilation, act_type, inplace=True),
                        conv1x1(channels, channels),
                        DWConvBNAct(channels, channels, (1, 3), 1, dilation, act_type, inplace=True),
                        conv1x1(channels, channels)
                    )
        if not self.use_skip:
            self.pool = nn.AvgPool2d(3, 2, 1)

    def forward(self, x):
        x_conv = self.conv(x)

        if self.use_skip:
            x = x + x_conv
        else:
            x_pool = self.pool(x)
            x = torch.cat([x_conv, x_pool], dim=1)

        return x


class DDCC(nn.Module):
    def __init__(self, channels, dilations, act_type):
        super(DDCC, self).__init__()
        assert len(dilations)==4, 'Length of dilations should be 4.\n'
        self.block1 = nn.Sequential(
                            nn.AvgPool2d(dilations[0], 1, dilations[0]//2),
                            ADSCModule(channels, 1, dilations[0], act_type)
                        )

        self.block2 = nn.Sequential(
                            conv1x1(2*channels, channels),
                            nn.AvgPool2d(dilations[1], 1, dilations[1]//2),
                            ADSCModule(channels, 1, dilations[1], act_type)
                        )

        self.block3 = nn.Sequential(
                            conv1x1(3*channels, channels),
                            nn.AvgPool2d(dilations[2], 1, dilations[2]//2),
                            ADSCModule(channels, 1, dilations[2], act_type)
                        )

        self.block4 = nn.Sequential(
                            conv1x1(4*channels, channels),
                            nn.AvgPool2d(dilations[3], 1, dilations[3]//2),
                            ADSCModule(channels, 1, dilations[3], act_type)
                        )

        self.conv_last = conv1x1(5*channels, channels)

    def forward(self, x):
        x1 = self.block1(x)
        
        x2 = torch.cat([x, x1], dim=1)
        x2 = self.block2(x2)
        
        x3 = torch.cat([x, x1, x2], dim=1)
        x3 = self.block3(x3)
        
        x4 = torch.cat([x, x1, x2, x3], dim=1)
        x4 = self.block4(x4)
        
        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.conv_last(x)

        return x
