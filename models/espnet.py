"""
Paper:      ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation
Url:        https://arxiv.org/abs/1803.06815
Create by:  zh320
Date:       2023/08/06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, conv3x3, ConvBNAct, DeConvBNAct, Activation


class ESPNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, arch_type='espnet', K=5, alpha2=2, 
                    alpha3=8, block_channel=[16, 64, 128], act_type='prelu'):
        super(ESPNet, self).__init__()
        arch_hub = ['espnet', 'espnet-a', 'espnet-b', 'espnet-c']
        if arch_type not in arch_hub:
            raise ValueError(f'Unsupport architecture type: {arch_type}.\n')
        self.arch_type = arch_type

        use_skip = arch_type in ['espnet', 'espnet-b', 'espnet-c']
        reinforce = arch_type in ['espnet', 'espnet-c']
        use_decoder = arch_type in ['espnet']
        
        if arch_type == 'espnet-a':
            block_channel[2] = block_channel[1]

        self.use_skip = use_skip
        self.reinforce = reinforce
        self.use_decoder = use_decoder

        self.l1_block = ConvBNAct(n_channel, block_channel[0], 3, 2, act_type=act_type)
        self.l2_block = L2Block(block_channel[0], block_channel[1], arch_type, alpha2, use_skip, reinforce, act_type)
        self.l3_block = L3Block(block_channel[2], num_class, arch_type, alpha3, use_skip, reinforce, use_decoder, act_type)

        if use_decoder:
            self.decoder = Decoder(num_class, 19, 131, act_type)

    def forward(self, x):
        x_input = x
        x = self.l1_block(x)
        if self.reinforce:
            size = x.size()[2:]
            x_half = F.interpolate(x_input, size, mode='bilinear')
            x = torch.cat([x, x_half], dim=1)
            if self.use_decoder:
                x_l1 = x

        if self.reinforce:
            x = self.l2_block(x, x_input)
            if self.use_decoder:
                x_l2 = x
        else:
            x = self.l2_block(x)

        x = self.l3_block(x)

        if self.use_decoder:
            x = self.decoder(x, x_l1, x_l2)
        else:
            size = x_input.size()[2:]
            x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class L2Block(nn.Module):
    def __init__(self, in_channels, hid_channels, arch_type, alpha, use_skip, 
                    reinforce, act_type='prelu'):
        super(L2Block, self).__init__()
        self.arch_type = arch_type
        self.alpha = alpha
        self.use_skip = use_skip
        self.reinforce = reinforce
        
        if reinforce:
            in_channels += 3

        self.conv1 = ESPModule(in_channels, hid_channels, stride=2, act_type=act_type)

        layers = []
        for _ in range(alpha):
            layers.append(ESPModule(hid_channels, hid_channels, act_type=act_type))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x, x_input=None):
        x = self.conv1(x)
        if self.use_skip:
            skip = x

        x = self.layers(x)
        
        if self.use_skip:
            x = torch.cat([x, skip], dim=1)

        if self.reinforce:
            size = x.size()[2:]
            x_quarter = F.interpolate(x_input, size, mode='bilinear')
            x = torch.cat([x, x_quarter], dim=1)

        return x


class L3Block(nn.Module):
    def __init__(self, in_channels, out_channels, arch_type, alpha, use_skip, 
                    reinforce, use_decoder, act_type='prelu'):
        super(L3Block, self).__init__()
        self.arch_type = arch_type
        self.alpha = alpha
        self.use_skip = use_skip
        
        if reinforce:
            in_channels += 3

        self.conv1 = ESPModule(in_channels, 128, stride=2, act_type=act_type)

        layers = []
        for _ in range(alpha):
            layers.append(ESPModule(128, 128, act_type=act_type))
        self.layers = nn.Sequential(*layers)

        if use_decoder:
            self.conv_last = ConvBNAct(256, out_channels, 1, act_type=act_type)
        elif use_skip:
            self.conv_last = conv1x1(256, out_channels)
        else:
            self.conv_last = conv1x1(128, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_skip:
            skip = x

        x = self.layers(x)

        if self.use_skip:
            x = torch.cat([x, skip], dim=1)

        x = self.conv_last(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_class, l1_channel, l2_channel, act_type='prelu'):
        super(Decoder, self).__init__()
        self.upconv_l3 = DeConvBNAct(num_class, num_class, act_type=act_type)
        self.conv_cat_l2 = ConvBNAct(l2_channel, num_class, 1)
        self.conv_l2 = ESPModule(2*num_class, num_class)
        self.upconv_l2 = DeConvBNAct(num_class, num_class, act_type=act_type)
        self.conv_cat_l1 = ConvBNAct(l1_channel, num_class, 1)
        self.conv_l1 = ESPModule(2*num_class, num_class)
        self.upconv_l1 = DeConvBNAct(num_class, num_class)
        
    def forward(self, x, x_l1, x_l2):
        x = self.upconv_l3(x)
        x_l2 = self.conv_cat_l2(x_l2)
        x = torch.cat([x, x_l2], dim=1)
        x = self.conv_l2(x)

        x = self.upconv_l2(x)
        x_l1 = self.conv_cat_l1(x_l1)
        x = torch.cat([x, x_l1], dim=1)
        x = self.conv_l1(x)

        x = self.upconv_l1(x)

        return x


class ESPModule(nn.Module):
    def __init__(self, in_channels, out_channels, K=5, ks=3, stride=1, act_type='prelu',):
        super(ESPModule, self).__init__()
        self.K = K
        self.stride = stride
        self.use_skip = (in_channels == out_channels) and (stride == 1)
        channel_kn = out_channels // K
        channel_k1 = out_channels - (K -1) * channel_kn
        self.perfect_divisor = channel_k1 == channel_kn

        if self.perfect_divisor:
            self.conv_kn = conv1x1(in_channels, channel_kn, stride)
        else:
            self.conv_kn = conv1x1(in_channels, channel_kn, stride)
            self.conv_k1 = conv1x1(in_channels, channel_k1, stride)

        self.layers = nn.ModuleList()
        for k in range(1, K+1):
            dt = 2**(k-1)       # dilation
            channel = channel_k1 if k == 1 else channel_kn
            self.layers.append(ConvBNAct(channel, channel, ks, 1, dt, act_type=act_type))

    def forward(self, x):
        if self.use_skip:
            residual = x

        transform_feats = []
        if self.perfect_divisor:
            x = self.conv_kn(x)     # Reduce
            for i in range(self.K):
                transform_feats.append(self.layers[i](x))   # Split --> Transform
                
            for j in range(1, self.K):
                transform_feats[j] += transform_feats[j-1]      # Merge: Sum
        else:
            x1 = self.conv_k1(x)    # Reduce
            xn = self.conv_kn(x)    # Reduce
            transform_feats.append(self.layers[0](x1))      # Split --> Transform
            for i in range(1, self.K):
                transform_feats.append(self.layers[i](xn))   # Split --> Transform

            for j in range(2, self.K):
                transform_feats[j] += transform_feats[j-1]      # Merge: Sum

        x = torch.cat(transform_feats, dim=1)               # Merge: Concat

        if self.use_skip:
            x += residual

        return x
