"""
Paper:      Real-time Semantic Segmentation with Fast Attention
Url:        https://arxiv.org/abs/2007.03815
Create by:  zh320
Date:       2024/04/06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from .modules import ConvBNAct, DeConvBNAct, SegHead, Activation
from .backbone import ResNet


class FANet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, att_channel=32, backbone_type='resnet18', cat_feat=True,
                    act_type='relu'):
        super(FANet, self).__init__()
        if backbone_type in ['resnet18', 'resnet34']:
            self.backbone = ResNet(backbone_type)
            channels = [64, 128, 256, 512]
            self.num_stage = len(channels)

            # Reduce spatial dimension for Res-1
            downsample = ConvBNAct(channels[0], channels[0], 1, 2, act_type='none')
            self.backbone.layer1[0] = BasicBlock(channels[0], channels[0], 2, downsample)
        else:
            raise NotImplementedError()
        self.cat_feat = cat_feat

        self.fast_attention = nn.ModuleList([FastAttention(channels[i], att_channel, act_type) for i in range(self.num_stage)])

        layers = [FuseUp(att_channel, att_channel, act_type=act_type) for _ in range(self.num_stage-1)]
        layers.append(FuseUp(att_channel, att_channel, has_up=False, act_type=act_type))
        self.fuse_up = nn.ModuleList(layers)

        last_channel = 4*att_channel if cat_feat else att_channel
        self.seg_head = SegHead(last_channel, num_class, act_type)

    def forward(self, x):
        size = x.size()[2:]
        x1, x2, x3, x4 = self.backbone(x)

        x4 = self.fast_attention[3](x4)
        x4 = self.fuse_up[3](x4)

        x3 = self.fast_attention[2](x3)
        x3 = self.fuse_up[2](x3, x4)

        x2 = self.fast_attention[1](x2)
        x2 = self.fuse_up[1](x2, x3)

        x1 = self.fast_attention[0](x1)
        x1 = self.fuse_up[0](x1, x2)

        if self.cat_feat:
            size1 = x1.size()[2:]
            x4 = F.interpolate(x4, size1, mode='bilinear', align_corners=True)
            x3 = F.interpolate(x3, size1, mode='bilinear', align_corners=True)
            x2 = F.interpolate(x2, size1, mode='bilinear', align_corners=True)

            x = torch.cat([x1, x2, x3, x4], dim=1)
            x = self.seg_head(x)
        else:
            x = self.seg_head(x1)

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class FastAttention(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(FastAttention, self).__init__()
        self.conv_q = ConvBNAct(in_channels, out_channels, 3, act_type='none')
        self.conv_k = ConvBNAct(in_channels, out_channels, 3, act_type='none')
        self.conv_v = ConvBNAct(in_channels, out_channels, 3, act_type='none')
        self.conv_fuse = ConvBNAct(out_channels, out_channels, 3, act_type=act_type)

    def forward(self, x):
        x_q = self.conv_q(x)
        x_k = self.conv_k(x)
        x_v = self.conv_v(x)
        residual = x_v

        B, C, H, W = x_q.size()
        n = H * W

        x_q = x_q.view(B, C, n)
        x_k = x_k.view(B, C, n)
        x_v = x_v.view(B, C, n)

        x_q = F.normalize(x_q, p=2, dim=1)
        x_k = F.normalize(x_k, p=2, dim=1).permute(0,2,1)

        y = (x_q @ (x_k @ x_v)) / n
        y = y.view(B, C, H, W)
        y = self.conv_fuse(y)
        y += residual

        return y


class FuseUp(nn.Module):
    def __init__(self, in_channels, out_channels, has_up=True, act_type='relu'):
        super(FuseUp, self).__init__()
        self.has_up = has_up
        if has_up:
            self.up = DeConvBNAct(in_channels, in_channels, act_type=act_type, inplace=True)

        self.conv = ConvBNAct(in_channels, out_channels, 3, act_type=act_type, inplace=True)

    def forward(self, x_fa, x_up=None):
        if self.has_up:
            if x_up is None:
                raise RuntimeError('Missing input from Up layer.\n')
            else:
                x_up = self.up(x_up)
            x_fa += x_up

        x_fa = self.conv(x_fa)

        return x_fa
