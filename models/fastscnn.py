"""
Paper:      Fast-SCNN: Fast Semantic Segmentation Network
Url:        https://arxiv.org/abs/1902.04502
Create by:  zh320
Date:       2023/04/16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, DSConvBNAct, DWConvBNAct, PWConvBNAct, ConvBNAct, Activation, \
                        PyramidPoolingModule


class FastSCNN(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='relu'):
        super(FastSCNN, self).__init__()
        self.learning_to_downsample = LearningToDownsample(n_channel, 64, act_type=act_type)
        self.global_feature_extractor = GlobalFeatureExtractor(64, 128, act_type=act_type)
        self.feature_fusion = FeatureFusionModule(64, 128, 128, act_type=act_type)
        self.classifier = Classifier(128, num_class, act_type=act_type)

    def forward(self, x):
        size = x.size()[2:]
        higher_res_feat = self.learning_to_downsample(x)
        lower_res_feat = self.global_feature_extractor(higher_res_feat)
        x = self.feature_fusion(higher_res_feat, lower_res_feat)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class LearningToDownsample(nn.Sequential):
    def __init__(self, in_channels, out_channels, hid_channels=[32, 48], act_type='relu'):
        super(LearningToDownsample, self).__init__(
            ConvBNAct(in_channels, hid_channels[0], 3, 2, act_type=act_type),
            DSConvBNAct(hid_channels[0], hid_channels[1], 3, 2, act_type=act_type),
            DSConvBNAct(hid_channels[1], out_channels, 3, 2, act_type=act_type),
        )


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super(GlobalFeatureExtractor, self).__init__()
        inverted_residual_setting = [
                # t, c, n, s
                [6, 64, 3, 2],
                [6, 96, 2, 2],
                [6, 128, 3, 1],
            ]

        # Building inverted residual blocks, codes borrowed from 
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        features = []
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_channels, c, stride, t, act_type=act_type))
                in_channels = c
        self.bottlenecks = nn.Sequential(*features)

        self.ppm = PyramidPoolingModule(in_channels, out_channels, act_type=act_type, bias=True)

    def forward(self, x):
        x = self.bottlenecks(x)
        x = self.ppm(x)

        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, higher_channels, lower_channels, out_channels, act_type='relu'):
        super(FeatureFusionModule, self).__init__()
        self.higher_res_conv = conv1x1(higher_channels, out_channels)
        self.lower_res_conv = nn.Sequential(
                                DWConvBNAct(lower_channels, lower_channels, 3, 1, act_type=act_type),
                                conv1x1(lower_channels, out_channels)
                            )
        self.non_linear = nn.Sequential(
                                nn.BatchNorm2d(out_channels),
                                Activation(act_type)
                        )

    def forward(self, higher_res_feat, lower_res_feat):
        size = higher_res_feat.size()[2:]
        higher_res_feat = self.higher_res_conv(higher_res_feat)
        lower_res_feat = F.interpolate(lower_res_feat, size, mode='bilinear', align_corners=True)
        lower_res_feat = self.lower_res_conv(lower_res_feat)
        x = self.non_linear(higher_res_feat + lower_res_feat)

        return x


class Classifier(nn.Sequential):
    def __init__(self, in_channels, num_class, act_type='relu'):
        super(Classifier, self).__init__(
            DSConvBNAct(in_channels, in_channels, 3, 1, act_type=act_type),
            DSConvBNAct(in_channels, in_channels, 3, 1, act_type=act_type),
            PWConvBNAct(in_channels, num_class, act_type=act_type),
        )


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
