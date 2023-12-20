"""
Paper:      ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time
Url:        https://arxiv.org/abs/1805.04554
Create by:  zh320
Date:       2023/05/13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, DSConvBNAct, DWConvBNAct, PWConvBNAct, ConvBNAct, Activation


class ContextNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='relu'):
        super(ContextNet, self).__init__()
        self.full_res_branch = Branch_1(n_channel, [32, 64, 128], 128, act_type=act_type)
        self.lower_res_branch = Branch_4(n_channel, 128, act_type=act_type)
        self.feature_fusion = FeatureFusion(128, 128, 128, act_type=act_type)
        self.classifier = ConvBNAct(128, num_class, 1, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]
        x_lower = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        full_res_feat = self.full_res_branch(x)
        lower_res_feat = self.lower_res_branch(x_lower)
        x = self.feature_fusion(full_res_feat, lower_res_feat)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        return x


class Branch_1(nn.Sequential):
    def __init__(self, in_channels, hid_channels, out_channels, act_type='relu'):
        assert len(hid_channels) == 3
        super(Branch_1, self).__init__(
                ConvBNAct(in_channels, hid_channels[0], 3, 2, act_type=act_type),
                DWConvBNAct(hid_channels[0], hid_channels[0], 3, 1, act_type='none'),
                PWConvBNAct(hid_channels[0], hid_channels[1], act_type=act_type),
                DWConvBNAct(hid_channels[1], hid_channels[1], 3, 1, act_type='none'),
                PWConvBNAct(hid_channels[1], hid_channels[2], act_type=act_type),
                DWConvBNAct(hid_channels[2], hid_channels[2], 3, 1, act_type='none'),
                PWConvBNAct(hid_channels[2], out_channels, act_type=act_type)
        )


class Branch_4(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super(Branch_4, self).__init__()
        self.conv_init = ConvBNAct(in_channels, 32, 3, 2, act_type=act_type)
        inverted_residual_setting = [
                # t, c, n, s
                [1, 32, 1, 1],
                [6, 32, 1, 1],
                [6, 48, 3, 2],
                [6, 64, 3, 2],
                [6, 96, 2, 1],
                [6, 128, 2, 1],
            ]
        
        # Building inverted residual blocks, codes borrowed from 
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        features = []
        in_channels = 32
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_channels, c, stride, t, act_type=act_type))
                in_channels = c
        self.bottlenecks = nn.Sequential(*features)
        self.conv_last = ConvBNAct(128, out_channels, 3, 1, act_type=act_type)
        
    def forward(self, x):
        x = self.conv_init(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)
        
        return x


class FeatureFusion(nn.Module):
    def __init__(self, branch_1_channels, branch_4_channels, out_channels, act_type='relu'):
        super(FeatureFusion, self).__init__()
        self.branch_1_conv = conv1x1(branch_1_channels, out_channels)
        self.branch_4_conv = nn.Sequential(
                                DSConvBNAct(branch_4_channels, out_channels, 3, dilation=4, act_type='none'),
                                conv1x1(out_channels, out_channels)
                                )
        self.act = Activation(act_type=act_type)                                 
        
    def forward(self, branch_1_feat, branch_4_feat):
        size = branch_1_feat.size()[2:]
        
        branch_1_feat = self.branch_1_conv(branch_1_feat)
        
        branch_4_feat = F.interpolate(branch_4_feat, size, mode='bilinear', align_corners=True)
        branch_4_feat = self.branch_4_conv(branch_4_feat)
        
        res = branch_1_feat + branch_4_feat
        res = self.act(res)
        
        return res


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6, act_type='relu'):
        super(InvertedResidual, self).__init__()
        hid_channels = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
                        PWConvBNAct(in_channels, hid_channels, act_type=act_type),
                        DWConvBNAct(hid_channels, hid_channels, 3, stride, act_type=act_type),
                        ConvBNAct(hid_channels, out_channels, 1, act_type='none')
                    )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
