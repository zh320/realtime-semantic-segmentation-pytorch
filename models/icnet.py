"""
Paper:      ICNet for Real-Time Semantic Segmentation on High-Resolution Images
Url:        https://arxiv.org/abs/1704.08545
Create by:  zh320
Date:       2023/10/15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, Activation, PyramidPoolingModule, SegHead


class ICNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='resnet18', act_type='relu', 
                    use_aux=True):
        super(ICNet, self).__init__()
        if 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            ch1 = 512 if backbone_type in ['resnet18', 'resnet34'] else 2048
            ch2 = 128 if backbone_type in ['resnet18', 'resnet34'] else 512
        else:
            raise NotImplementedError()

        self.use_aux = use_aux
        self.bottom_branch = HighResolutionBranch(n_channel, 128, act_type=act_type)
        self.ppm = PyramidPoolingModule(ch1, 256, act_type=act_type)
        self.cff42 = CascadeFeatureFusionUnit(256, ch2, 128, num_class, act_type, use_aux)
        self.cff21 = CascadeFeatureFusionUnit(128, 128, 128, num_class, act_type, use_aux)
        self.seg_head = SegHead(128, num_class, act_type)

    def forward(self, x, is_training=False):
        size = x.size()[2:]
        x_d2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x_d4 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)

        # Lowest resolution branch
        x_d4, _ = self.backbone(x_d4)           # 32x down
        x_d4 = self.ppm(x_d4)

        # Medium resolution branch
        _, x_d2 = self.backbone(x_d2)           # 16x down

        # High resolution branch
        x = self.bottom_branch(x)               # 8x down

        # Cascade feature fusion
        if self.use_aux:
            x_d2, aux2 = self.cff42(x_d4, x_d2) # 16x down
            x, aux3 = self.cff21(x_d2, x)       # 8x down
        else:
            x_d2 = self.cff42(x_d4, x_d2)
            x = self.cff21(x_d2, x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)   # 4x down
        x = self.seg_head(x)                    # 4x down

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        if self.use_aux and is_training:
            return x, (aux2, aux3)
        else:
            return x


class CascadeFeatureFusionUnit(nn.Module):
    def __init__(self, channel1, channel2, out_channels, num_class, act_type, use_aux):
        super(CascadeFeatureFusionUnit, self).__init__()
        self.use_aux = use_aux
        self.conv1 = ConvBNAct(channel1, out_channels, 3, 1, 2, act_type='none')
        self.conv2 = ConvBNAct(channel2, out_channels, 1, act_type='none')
        self.act = Activation(act_type)
        if use_aux:
            self.classifier = SegHead(channel1, num_class, act_type)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        if self.use_aux:
            x_aux = self.classifier(x1)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        x = self.act(x1 + x2)

        if self.use_aux:
            return x, x_aux
        else:
            return x


class HighResolutionBranch(nn.Sequential):
    def __init__(self, in_channels, out_channels, hid_channels=32, act_type='relu'):
        super(HighResolutionBranch, self).__init__(
            ConvBNAct(in_channels, hid_channels, 3, 2, act_type=act_type),
            ConvBNAct(hid_channels, hid_channels*2, 3, 2, act_type=act_type),
            ConvBNAct(hid_channels*2, out_channels, 3, 2, act_type=act_type)
        )


class ResNet(nn.Module):
    def __init__(self, resnet_type, pretrained=True):
        super(ResNet, self).__init__()
        from torchvision.models import resnet18, resnet34, resnet50, resnet101

        resnet_hub = {'resnet18':resnet18, 'resnet34':resnet34, 'resnet50':resnet50,
                        'resnet101':resnet101,}
        if resnet_type not in resnet_hub.keys():
            raise ValueError(f'Unsupported ResNet type: {resnet_type}.\n')

        use_basicblock = resnet_type in ['resnet18', 'resnet34']

        resnet = resnet_hub[resnet_type](pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Change stride-2 conv to dilated conv
        layers = [[self.layer3[0], resnet.layer3[0]], [self.layer4[0], resnet.layer4[0]]]        
        for i in range(1,3):
            ch = 128 if use_basicblock else 512
            resnet_downsample = layers[i-1][1].downsample[0]
            resnet_conv = layers[i-1][1].conv1 if use_basicblock else layers[i-1][1].conv2

            layers[i-1][0].downsample[0] = nn.Conv2d(ch*i, ch*i*2, 1, 1, bias=False)
            if use_basicblock:
                layers[i-1][0].conv1 = nn.Conv2d(ch*i, ch*i*2, 3, 1, 2*i, 2*i, bias=False)
            else:
                layers[i-1][0].conv2 = nn.Conv2d(ch//2*i, ch//2*i, 3, 1, 2*i, 2*i, bias=False)

            with torch.no_grad():
                layers[i-1][1].downsample[0].weight.copy_(resnet_downsample.weight)
                if use_basicblock:
                    layers[i-1][1].conv1.weight.copy_(resnet_conv.weight)
                else:
                    layers[i-1][1].conv2.weight.copy_(resnet_conv.weight)

    def forward(self, x):
        x = self.conv1(x)       # 2x down
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 4x down
        x = self.layer1(x)
        x2 = self.layer2(x)      # 8x down
        x = self.layer3(x2)      # 8x down with dilation 2
        x = self.layer4(x)      # 8x down with dilation 4

        return x, x2
