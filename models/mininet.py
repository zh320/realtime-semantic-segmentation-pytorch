"""
Paper:      Enhancing V-SLAM Keyframe Selection with an Efficient ConvNet for Semantic Analysis
Url:        https://ieeexplore.ieee.org/abstract/document/8793923
Create by:  zh320
Date:       2023/10/15
"""

import torch
import torch.nn as nn

from .modules import conv1x1, DSConvBNAct, ConvBNAct, DeConvBNAct, Activation


class MiniNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='selu'):
        super(MiniNet, self).__init__()
        # Downsample block
        self.down1 = DSConvBNAct(n_channel, 12, 3, 2, act_type=act_type)
        self.down2 = DSConvBNAct(12, 24, 3, 2, act_type=act_type)
        self.down3 = DSConvBNAct(24, 48, 3, 2, act_type=act_type)
        self.down4 = DSConvBNAct(48, 96, 3, 2, act_type=act_type)
        # Branch 1
        self.branch1 = nn.Sequential(
                            ConvModule(96, 1, act_type),
                            ConvModule(96, 2, act_type),
                            ConvModule(96, 4, act_type),
                            ConvModule(96, 8, act_type),
                        )
        # Branch 2
        self.branch2_down = DSConvBNAct(96, 192, 3, 2, act_type=act_type)
        self.branch2 = nn.Sequential(
                            ConvModule(192, 1, act_type),
                            DSConvBNAct(192, 386, 3, 2, act_type=act_type),
                            ConvModule(386, 1, act_type),
                            ConvModule(386, 1, act_type),
                            DeConvBNAct(386, 192, act_type=act_type),
                            ConvModule(192, 1, act_type),
                        )
        self.branch2_up = DeConvBNAct(192*2, 96, act_type=act_type)
        # Upsample Block
        self.up4 = nn.Sequential(
                        DeConvBNAct(96*3, 96, act_type=act_type),
                        ConvModule(96, 1, act_type),
                        conv1x1(96, 48)
                    )
        self.up3 = DeConvBNAct(48*2, 24, act_type=act_type)
        self.up2 = DeConvBNAct(24*2, 12, act_type=act_type)
        self.up1 = DeConvBNAct(12*2, num_class, act_type=act_type)

    def forward(self, x):
        x_d1 = self.down1(x)
        x_d2 = self.down2(x_d1)
        x_d3 = self.down3(x_d2)
        x_d4 = self.down4(x_d3)

        x_b1 = self.branch1(x_d4)

        x_d5 = self.branch2_down(x_d4)
        x_b2 = self.branch2(x_d5)
        x_b2 = torch.cat([x_b2, x_d5], dim=1)
        x_b2 = self.branch2_up(x_b2)

        x = torch.cat([x_b1, x_b2, x_d4], dim=1)
        x = self.up4(x)
        x = torch.cat([x, x_d3], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x_d2], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x_d1], dim=1)
        x = self.up1(x)

        return x


class ConvModule(nn.Module):
    def __init__(self, channels, dilation, act_type):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(channels, channels, (1,3), padding=(0, dilation), 
                                    dilation=dilation, groups=channels, bias=False),
                        Activation(act_type),
                        nn.Conv2d(channels, channels, (3,1), padding=(dilation, 0), 
                                    dilation=dilation, groups=channels, bias=False),
                        Activation(act_type),
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(channels, channels, (3,1), padding=(dilation, 0), 
                                    dilation=dilation, groups=channels, bias=False),
                        Activation(act_type),
                        nn.Conv2d(channels, channels, (1,3), padding=(0, dilation), 
                                    dilation=dilation, groups=channels, bias=False),
                    )
        self.dropout = nn.Dropout(p=0.25)
        self.act = Activation(act_type)
    
    def forward(self, x):
        residual = x

        x1 = self.conv1(x)
        x = self.conv2(x1)

        x += x1
        x = self.dropout(x)
        x += residual

        return self.act(x)
