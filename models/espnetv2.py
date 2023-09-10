"""
Paper:      ESPNetv2: A Light-weight, Power Efficient, and General Purpose 
            Convolutional Neural Network
Url:        https://arxiv.org/abs/1811.11431
Create by:  zh320
Date:       2023/09/03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, DSConvBNAct, PWConvBNAct, ConvBNAct


class ESPNetv2(nn.Module):
    def __init__(self, num_class=1, n_channel=3, K=4, alpha3=3, alpha4=7, act_type='prelu'):
        super(ESPNetv2, self).__init__()
        self.pool = nn.AvgPool2d(3, 2, 1)
        self.l1_block = ConvBNAct(n_channel, 32, 3, 2, act_type=act_type)
        self.l2_block = EESPModule(32, stride=2, act_type=act_type)
        self.l3_block1 = EESPModule(64, stride=2, act_type=act_type)
        self.l3_block2 = build_blocks(EESPModule, 128, alpha3, act_type=act_type)
        self.l4_block1 = EESPModule(128, stride=2, act_type=act_type)
        self.l4_block2 = build_blocks(EESPModule, 256, alpha4, act_type=act_type)

        self.convl4_l3 = ConvBNAct(256, 128, 1)
        self.ppm = PyramidPoolingModule(256, 256, act_type=act_type)
        self.decoder = SegHead(256, num_class, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]
        x_d4 = self.pool(self.pool(x))
        x_d8 = self.pool(x_d4)
        x_d16 = self.pool(x_d8)
        
        x = self.l1_block(x)
        x = self.l2_block(x, x_d4)

        x = self.l3_block1(x, x_d8)
        x3 = self.l3_block2(x)
        size_l3 = x3.size()[2:]
        
        x = self.l4_block1(x3, x_d16)
        x = self.l4_block2(x)
        x = F.interpolate(x, size_l3, mode='bilinear', align_corners=True)
        x = self.convl4_l3(x)
        x = torch.cat([x, x3], dim=1)
        
        x = self.ppm(x)
        x = self.decoder(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


def build_blocks(block, channels, num_block, act_type='relu'):
    layers = []
    for _ in range(num_block):
        layers.append(block(channels, act_type=act_type))
    return  nn.Sequential(*layers)
    
    
class EESPModule(nn.Module):
    def __init__(self, channels, K=4, ks=3, stride=1, act_type='prelu'):
        super(EESPModule, self).__init__()
        assert channels % K == 0, 'Input channels should be integer multiples of K.\n'
        
        self.K = K
        channel_k = channels // K
        self.use_skip = stride == 1

        self.conv_init = nn.Conv2d(channels, channel_k, 1, groups=K, bias=False)
        self.layers = nn.ModuleList()
        for k in range(1, K+1):
            dt = 2**(k-1)       # dilation
            self.layers.append(DSConvBNAct(channel_k, channel_k, ks, stride, dt, act_type=act_type))
        self.conv_last = nn.Conv2d(channels, channels, 1, groups=K, bias=False)

        if not self.use_skip:
            self.pool = nn.AvgPool2d(3, 2, 1)
            self.conv_stride = nn.Sequential(
                                            ConvBNAct(3, 3, 3),
                                            conv1x1(3, channels*2)
                                        )

    def forward(self, x, img=None):
        if not self.use_skip and img is None:
            raise ValueError('Strided EESP unit needs downsampled input image.\n')

        residual = x
        transform_feats = []

        x = self.conv_init(x)     # Reduce
        for i in range(self.K):
            transform_feats.append(self.layers[i](x))   # Split --> Transform
            
        for j in range(1, self.K):
            transform_feats[j] += transform_feats[j-1]      # Merge: Sum

        x = torch.cat(transform_feats, dim=1)               # Merge: Concat
        x = self.conv_last(x)

        if self.use_skip:
            x += residual
        else:
            residual = self.pool(residual)
            x = torch.cat([x, residual], dim=1)
            img = self.conv_stride(img)
            x += img

        return x

    
class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(PyramidPoolingModule, self).__init__()
        hid_channels = int(in_channels // 4)
        self.stage1 = self._make_stage(in_channels, hid_channels, 1)
        self.stage2 = self._make_stage(in_channels, hid_channels, 2)
        self.stage3 = self._make_stage(in_channels, hid_channels, 4)
        self.stage4 = self._make_stage(in_channels, hid_channels, 6)
        self.conv = PWConvBNAct(2*in_channels, out_channels, act_type=act_type)

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


class SegHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, act_type, hid_channels=128):
        super(SegHead, self).__init__(
            ConvBNAct(in_channels, hid_channels, 3, act_type=act_type),
            conv1x1(hid_channels, out_channels)
        )
