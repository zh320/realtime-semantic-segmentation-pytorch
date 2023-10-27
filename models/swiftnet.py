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

from .modules import conv1x1, PWConvBNAct, ConvBNAct


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
        self.spp = PyramidPoolingModule(channels[3], up_channels, act_type)
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


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(PyramidPoolingModule, self).__init__()
        hid_channels = int(in_channels // 4)
        self.stage1 = self._make_stage(in_channels, hid_channels, 1)
        self.stage2 = self._make_stage(in_channels, hid_channels, 2)
        self.stage3 = self._make_stage(in_channels, hid_channels, 4)
        self.stage4 = self._make_stage(in_channels, hid_channels, 6)
        self.conv = PWConvBNAct(2*in_channels, out_channels, act_type)

    def _make_stage(self, in_channels, out_channels, pool_size):
        return nn.Sequential(
                        nn.AdaptiveAvgPool2d(pool_size),
                        conv1x1(in_channels, out_channels)
                )

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.stage1(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.stage2(x), size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.stage3(x), size, mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.stage4(x), size, mode='bilinear', align_corners=True)
        x = self.conv(torch.cat([x, x1, x2, x3, x4], dim=1))
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
        x1 = self.layer1(x)
        x2 = self.layer2(x1)      # 8x down
        x3 = self.layer3(x2)      # 16x down
        x4 = self.layer4(x3)      # 32x down

        return x1, x2, x3, x4


class Mobilenetv2(nn.Module):
    def __init__(self, pretrained=True):
        super(Mobilenetv2, self).__init__()
        from torchvision.models import mobilenet_v2

        mobilenet = mobilenet_v2(pretrained=pretrained)

        self.layer1 = mobilenet.features[:4]        # 4x down
        self.layer2 = mobilenet.features[4:7]       # 8x down
        self.layer3 = mobilenet.features[7:14]      # 16x down
        self.layer4 = mobilenet.features[14:18]     # 32x down
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4
