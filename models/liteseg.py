"""
Paper:      LiteSeg: A Novel Lightweight ConvNet for Semantic Segmentation
Url:        https://arxiv.org/abs/1912.06683
Create by:  zh320
Date:       2023/10/15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct


class LiteSeg(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='mobilenet_v2', act_type='relu'):
        super(LiteSeg, self).__init__()
        if backbone_type == 'mobilenet_v2':
            self.backbone = Mobilenetv2()
            channels = [320, 32]
        elif 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [512, 128] if backbone_type in ['resnet18', 'resnet34'] else [2048, 512]
        else:
            raise NotImplementedError()

        self.daspp = DASPPModule(channels[0], 512, act_type)
        self.seg_head = SegHead(512 + channels[1], num_class, act_type)

    def forward(self, x):
        size = x.size()[2:]

        x, x1 = self.backbone(x)
        size1 = x1.size()[2:]

        x = self.daspp(x)
        x = F.interpolate(x, size1, mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)

        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class DASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(DASPPModule, self).__init__()
        hid_channels = in_channels // 5
        last_channels = in_channels - hid_channels * 4
        self.stage1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        self.stage2 = ConvBNAct(in_channels, hid_channels, 3, dilation=3, act_type=act_type)
        self.stage3 = ConvBNAct(in_channels, hid_channels, 3, dilation=6, act_type=act_type)
        self.stage4 = ConvBNAct(in_channels, hid_channels, 3, dilation=9, act_type=act_type)
        self.stage5 = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            conv1x1(in_channels, last_channels)
                        )
        self.conv = ConvBNAct(2*in_channels, out_channels, 1, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]

        x1 = self.stage1(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x)
        x5 = self.stage5(x)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=True)

        x = self.conv(torch.cat([x, x1, x2, x3, x4, x5], dim=1))
        return x


class Mobilenetv2(nn.Module):
    def __init__(self, pretrained=True):
        super(Mobilenetv2, self).__init__()
        from torchvision.models import mobilenet_v2
        
        mobilenet = mobilenet_v2(pretrained=pretrained)

        self.layer1 = mobilenet.features[:7]     # 8x down
        self.layer2 = mobilenet.features[7:14]   # 16x down
        self.layer3 = mobilenet.features[14:18]    # 32x down
        
    def forward(self, x):
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        return x, x1


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
        x2 = self.layer2(x)      # 8x down
        x = self.layer3(x2)      # 16x down
        x = self.layer4(x)      # 32x down

        return x, x2


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type, hid_channels=256):
        super(SegHead, self).__init__(
            ConvBNAct(in_channels, hid_channels, 3, act_type=act_type),
            ConvBNAct(hid_channels, hid_channels//2, 3, act_type=act_type),
            conv1x1(hid_channels//2, num_class)
        )
