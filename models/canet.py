"""
Paper:      Cross Attention Network for Semantic Segmentation
Url:        https://arxiv.org/abs/1907.10958
Create by:  zh320
Date:       2023/09/30
"""

import torch
import torch.nn as nn

from .modules import ConvBNAct, Activation


class CANet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='mobilenet_v2', act_type='relu'):
        super(CANet, self).__init__()
        self.spatial_branch = SpatialBranch(n_channel, 64, act_type)
        self.context_branch = ContextBranch(64*4, backbone_type)
        self.fca = FeatureCrossAttentionModule(64*4, num_class, act_type)
        self.up = Upsample(num_class, num_class, scale_factor=8)

    def forward(self, x):
        size = x.size()[2:]
        x_s = self.spatial_branch(x)
        x_c = self.context_branch(x)
        x = self.fca(x_s, x_c)
        x = self.up(x)

        return x


class SpatialBranch(nn.Sequential):
    def __init__(self, n_channel, channels, act_type):
        super(SpatialBranch, self).__init__(
            ConvBNAct(n_channel, channels, 3, 2, act_type=act_type, inplace=True),
            ConvBNAct(channels, channels*2, 3, 2, act_type=act_type, inplace=True),
            ConvBNAct(channels*2, channels*4, 3, 2, act_type=act_type, inplace=True),
        )


class ContextBranch(nn.Module):
    def __init__(self, out_channels, backbone_type, hid_channels=192):
        super(ContextBranch, self).__init__()
        if 'mobilenet' in backbone_type:
            self.backbone = Mobilenetv2()
            channels = [320, 96]
        elif 'resnet' in backbone_type:
            self.backbone = ResNet(backbone_type)
            channels = [512, 256] if (('18' in backbone_type) or ('34' in backbone_type)) else [2048, 1024]
        else:
            raise NotImplementedError()
            
        self.up1 = Upsample(channels[0], hid_channels)
        self.up2 = Upsample(channels[1] + hid_channels, out_channels)

    def forward(self, x):
        x, x_d16 = self.backbone(x)
        x = self.up1(x)

        x = torch.cat([x, x_d16], dim=1)
        x = self.up2(x)

        return x


class FeatureCrossAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(FeatureCrossAttentionModule, self).__init__()
        self.conv_init = ConvBNAct(2*in_channels, in_channels, act_type=act_type, inplace=True)
        self.sa = SpatialAttentionBlock(in_channels)
        self.ca = ChannelAttentionBlock(in_channels)
        self.conv_last = ConvBNAct(in_channels, out_channels, inplace=True)

    def forward(self, x_s, x_c):
        x = torch.cat([x_s, x_c], dim=1)
        x_s = self.sa(x_s)
        x_c = self.ca(x_c)

        x = self.conv_init(x)
        residual = x

        x = x * x_s
        x = x * x_c
        x += residual

        x = self.conv_last(x)

        return x


class SpatialAttentionBlock(nn.Sequential):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__(
            ConvBNAct(in_channels, 1, act_type='sigmoid')
        )


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        x_max = self.max_pool(x).view(-1, self.in_channels)
        x_avg = self.avg_pool(x).view(-1, self.in_channels)

        x_max = self.fc(x_max)
        x_avg = self.fc(x_avg)

        x = x_max + x_avg
        x = torch.sigmoid(x)

        return x.unsqueeze(-1).unsqueeze(-1)


class Mobilenetv2(nn.Module):
    def __init__(self, pretrained=True):
        super(Mobilenetv2, self).__init__()
        from torchvision.models import mobilenet_v2

        mobilenet = mobilenet_v2(pretrained=pretrained)

        self.layer1 = mobilenet.features[:7]
        self.layer2 = mobilenet.features[7:14]
        self.layer3 = mobilenet.features[14:18]
        
    def forward(self, x):
        x = self.layer1(x)      # 8x down
        x2 = self.layer2(x)     # 16x down
        x = self.layer3(x2)     # 32x down

        return x, x2


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


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None, padding=None,
                    upsample_type='deconvolution', act_type='relu'):
        super(Upsample, self).__init__()
        if upsample_type == 'deconvolution':
            if kernel_size is None:
                kernel_size = 2*scale_factor - 1
            if padding is None:    
                padding = (kernel_size - 1) // 2
            output_padding = scale_factor - 1
            self.up_conv = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels, out_channels, 
                                                        kernel_size=kernel_size, 
                                                        stride=scale_factor, padding=padding,
                                                        output_padding=output_padding),
                                    nn.BatchNorm2d(out_channels),
                                    Activation(act_type)
                            )
        else:
            self.up_conv = nn.Sequential(
                                    ConvBNAct(in_channels, out_channels, 1, act_type=act_type),
                                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                            )

    def forward(self, x):
        return self.up_conv(x)
