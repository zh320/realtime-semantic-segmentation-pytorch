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
    def __init__(self, num_class=1, n_channel=3, act_type='relu', use_aux=False):
        super(ContextNet, self).__init__()
        self.use_aux = use_aux
        self.full_res_branch = Branch_1(n_channel, [32, 64, 128], 128, act_type=act_type)
        self.lower_res_branch = Branch_4(n_channel, 128, num_class, act_type=act_type, use_aux=use_aux)
        self.feature_fusion = FeatureFusion(128, 128, 128, act_type=act_type)
        self.classifier = ConvBNAct(128, num_class, 1, act_type=act_type)

    def forward(self, x, is_training=False):
        size = x.size()[2:]
        x_lower = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=True)
        full_res_feat = self.full_res_branch(x)
        if self.use_aux:
            lower_res_feat, aux1, aux2, aux3, aux4 = self.lower_res_branch(x_lower)
        else:
            lower_res_feat = self.lower_res_branch(x_lower)
        x = self.feature_fusion(full_res_feat, lower_res_feat)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        if self.use_aux and is_training:
            return x, (aux1, aux2, aux3, aux4)
        else:    
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
    def __init__(self, in_channels, out_channels, num_class, act_type='relu', use_aux=False):
        super(Branch_4, self).__init__()
        self.use_aux = use_aux
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
        all_features = []
        in_channels = 32
        for setting in inverted_residual_setting:
            features = []
            t, c, n, s = setting
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_channels, c, stride, t, act_type=act_type))
                in_channels = c
            all_features.append(features)
        
        self.bottleneck0 = nn.Sequential(*all_features[0])
        self.bottleneck1 = nn.Sequential(*all_features[1])
        self.bottleneck2 = nn.Sequential(*all_features[2])
        self.bottleneck3 = nn.Sequential(*all_features[3])
        self.bottleneck4 = nn.Sequential(*all_features[4])
        self.bottleneck5 = nn.Sequential(*all_features[5])
        
        if use_aux:
            self.aux_head1 = AuxHead(32, num_class, act_type=act_type)
            self.aux_head2 = AuxHead(48, num_class, act_type=act_type)
            self.aux_head3 = AuxHead(64, num_class, act_type=act_type)
            self.aux_head4 = AuxHead(96, num_class, act_type=act_type)

        self.conv_last = ConvBNAct(128, out_channels, 3, 1, act_type=act_type)
        
    def forward(self, x):
        x = self.conv_init(x)
        x = self.bottleneck0(x)
        
        x = self.bottleneck1(x)
        if self.use_aux:
            aux1 = self.aux_head1(x)
            
        x = self.bottleneck2(x)
        if self.use_aux:
            aux2 = self.aux_head2(x)

        x = self.bottleneck3(x)
        if self.use_aux:
            aux3 = self.aux_head3(x)
            
        x = self.bottleneck4(x)
        if self.use_aux:
            aux4 = self.aux_head4(x)

        x = self.bottleneck5(x)
        
        x = self.conv_last(x)
        
        if self.use_aux:
            return x, aux1, aux2, aux3, aux4
        else:
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


class AuxHead(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type='relu'):
        super(AuxHead, self).__init__(
            DSConvBNAct(in_channels, in_channels, 3, 1, act_type=act_type),
            DSConvBNAct(in_channels, in_channels, 3, 1, act_type=act_type),
            PWConvBNAct(in_channels, num_class, act_type=act_type),
        )        