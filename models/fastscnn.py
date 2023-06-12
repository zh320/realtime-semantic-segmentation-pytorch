"""
Paper:      Fast-SCNN: Fast Semantic Segmentation Network
Url:        https://arxiv.org/abs/1902.04502
Create by:  zh320
Date:       2023/04/16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, DSConvBNAct, DWConvBNAct, PWConvBNAct, ConvBNAct


class FastSCNN(nn.Module):
    def __init__(self, num_class=1, n_channel=3, act_type='relu', use_aux=False):
        super(FastSCNN, self).__init__()
        self.use_aux = use_aux
        self.learning_to_downsample = LearningToDownsample(n_channel, 64, act_type=act_type)
        self.global_feature_extractor = GlobalFeatureExtractor(64, 128, num_class, act_type=act_type, use_aux=use_aux)
        self.feature_fusion = FeatureFusionModule(64, 128, 128, act_type=act_type)
        self.classifier = Classifier(128, num_class, act_type=act_type)
        
    def forward(self, x, is_training=False):
        size = x.size()[2:]
        higher_res_feat = self.learning_to_downsample(x)
        if self.use_aux:
            lower_res_feat, aux1, aux2, aux3 = self.global_feature_extractor(higher_res_feat)
        else:
            lower_res_feat = self.global_feature_extractor(higher_res_feat)
            
        x = self.feature_fusion(higher_res_feat, lower_res_feat)
        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        if self.use_aux and is_training:
            return x, (aux1, aux2, aux3)
        else:    
            return x
        
        
class LearningToDownsample(nn.Sequential):
    def __init__(self, in_channels, out_channels, hid_channels=[32, 48], act_type='relu'):
        super(LearningToDownsample, self).__init__(
            ConvBNAct(in_channels, hid_channels[0], 3, 2, act_type=act_type),
            DSConvBNAct(hid_channels[0], hid_channels[1], 3, 2, act_type=act_type),
            DSConvBNAct(hid_channels[1], out_channels, 3, 2, act_type=act_type),
        )


class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, act_type='relu', use_aux=False):
        super(GlobalFeatureExtractor, self).__init__()
        self.use_aux = use_aux
        inverted_residual_setting = [
                # t, c, n, s
                [6, 64, 3, 2],
                [6, 96, 2, 2],
                [6, 128, 3, 1],
            ]
        
        # Building inverted residual blocks, codes borrowed from 
        # https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        all_features = []
        for setting in inverted_residual_setting:
            features = []
            t, c, n, s = setting
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(in_channels, c, stride, t, act_type=act_type))
                in_channels = c
            all_features.append(features)    

        self.bottleneck1 = nn.Sequential(*all_features[0])
        self.bottleneck2 = nn.Sequential(*all_features[1])
        self.bottleneck3 = nn.Sequential(*all_features[2])
        
        if use_aux:
            self.aux_head1 = Classifier(inverted_residual_setting[0][1], num_class, act_type=act_type)
            self.aux_head2 = Classifier(inverted_residual_setting[1][1], num_class, act_type=act_type)
            self.aux_head3 = Classifier(inverted_residual_setting[2][1], num_class, act_type=act_type)
        
        self.ppm = PyramidPoolingModule(in_channels, out_channels, act_type=act_type)

    def forward(self, x):
        x = self.bottleneck1(x)
        if self.use_aux:
            aux1 = self.aux_head1(x)
            
        x = self.bottleneck2(x)
        if self.use_aux:
            aux2 = self.aux_head2(x)
            
        x = self.bottleneck3(x)
        if self.use_aux:
            aux3 = self.aux_head3(x)

        x = self.ppm(x)
        if self.use_aux:
            return x, aux1, aux2, aux3
        else:    
            return x


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
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
                                nn.ReLU()
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
