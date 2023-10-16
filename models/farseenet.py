"""
Paper:      FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale 
            Context Aggregation and Feature Space Super-resolution
Url:        https://arxiv.org/abs/2003.03913
Create by:  zh320
Date:       2023/10/08
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, DWConvBNAct, ConvBNAct


class FarSeeNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='resnet18', act_type='relu'):
        super(FarSeeNet, self).__init__()
        if 'resnet' in backbone_type:
            self.frontend_network = ResNet(backbone_type)
            high_channels = 512 if backbone_type in ['resnet18', 'resnet34'] else 2048
            low_channels = 256 if backbone_type in ['resnet18', 'resnet34'] else 1024
        else:
            raise NotImplementedError()

        self.backend_network = FASPP(high_channels, low_channels, num_class, act_type)

    def forward(self, x):
        size = x.size()[2:]

        x_high, x_low = self.frontend_network(x)

        x = self.backend_network(x_high, x_low)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class FASPP(nn.Module):
    def __init__(self, high_channels, low_channels, num_class, act_type, 
                    dilations=[6,12,18], hid_channels=256):
        super(FASPP, self).__init__()
        # High level convolutions
        self.conv_high = nn.ModuleList([
                                ConvBNAct(high_channels, hid_channels, 1, act_type=act_type)
                            ])
        for dt in dilations:
            self.conv_high.append(
                nn.Sequential(
                    ConvBNAct(high_channels, hid_channels, 1, act_type=act_type),
                    DWConvBNAct(hid_channels, hid_channels, 3, dilation=dt, act_type=act_type)
                )
            )

        self.sub_pixel_high = nn.Sequential(
                                    conv1x1(hid_channels*4, hid_channels*2*(2**2)),
                                    nn.PixelShuffle(2)
                                )

        # Low level convolutions
        self.conv_low_init = ConvBNAct(low_channels, 48, 1, act_type=act_type)
        self.conv_low = nn.ModuleList([
                            ConvBNAct(hid_channels*2+48, hid_channels//2, 1, act_type=act_type)
                        ])
        for dt in dilations[:-1]:
            self.conv_low.append(
                nn.Sequential(
                    ConvBNAct(hid_channels*2+48, hid_channels//2, 1, act_type=act_type),
                    DWConvBNAct(hid_channels//2, hid_channels//2, 3, dilation=dt, act_type=act_type)
                )
            )

        self.conv_low_last = nn.Sequential(
                                ConvBNAct(hid_channels//2*3, hid_channels*2, 1, act_type=act_type),
                                ConvBNAct(hid_channels*2, hid_channels*2, act_type=act_type)
                            )

        self.sub_pixel_low = nn.Sequential(
                                conv1x1(hid_channels*2, num_class*(4**2)),
                                nn.PixelShuffle(4)
                            )

    def forward(self, x_high, x_low):
        # High level features
        high_feats = []
        for conv_high in self.conv_high:
            high_feats.append(conv_high(x_high))

        x = torch.cat(high_feats, dim=1)
        x = self.sub_pixel_high(x)

        # Low level features
        x_low = self.conv_low_init(x_low)
        x = torch.cat([x, x_low], dim=1)

        low_feats = []
        for conv_low in self.conv_low:
            low_feats.append(conv_low(x))
            
        x = torch.cat(low_feats, dim=1)
        x = self.conv_low_last(x)
        x = self.sub_pixel_low(x)

        return x


class ResNet(nn.Module):
    def __init__(self, resnet_type, pretrained=True):
        super(ResNet, self).__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101

        resnet_hub = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50,
                        'resnet101':resnet101,}
        if resnet_type not in resnet_hub:
            raise ValueError(f'Unsupported ResNet type: {resnet_type}.\n')

        resnet = resnet_hub[resnet_type](pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)       # 2x down
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 4x down
        x = self.layer1(x)
        x = self.layer2(x)      # 8x down
        x3 = self.layer3(x)      # 16x down
        x = self.layer4(x3)      # 32x down

        return x, x3
