"""
Paper:      Rethinking BiSeNet For Real-time Semantic Segmentation
Url:        https://arxiv.org/abs/2104.13188
Create by:  zh320
Date:       2024/01/20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, SegHead
from .bisenetv1 import AttentionRefinementModule, FeatureFusionModule


class STDC(nn.Module):
    def __init__(self, num_class=1, n_channel=3, encoder_type='stdc1', use_detail_head=False, use_aux=False, 
                    act_type='relu'):
        super(STDC, self).__init__()
        repeat_times_hub = {'stdc1': [1,1,1], 'stdc2': [3,4,2]}
        if encoder_type not in repeat_times_hub.keys():
            raise ValueError('Unsupported encoder type.\n')
        repeat_times = repeat_times_hub[encoder_type]
        assert not use_detail_head * use_aux, 'Currently only support either aux-head or detail head.\n'
        self.use_detail_head = use_detail_head
        self.use_aux = use_aux

        self.stage1 = ConvBNAct(n_channel, 32, 3, 2)
        self.stage2 = ConvBNAct(32, 64, 3, 2)
        self.stage3 = self._make_stage(64, 256, repeat_times[0], act_type)
        self.stage4 = self._make_stage(256, 512, repeat_times[1], act_type)
        self.stage5 = self._make_stage(512, 1024, repeat_times[2], act_type)
        
        if use_aux:
            self.aux_head3 = SegHead(256, num_class, act_type)
            self.aux_head4 = SegHead(512, num_class, act_type)
            self.aux_head5 = SegHead(1024, num_class, act_type)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.arm4 = AttentionRefinementModule(512)
        self.arm5 = AttentionRefinementModule(1024)
        self.conv4 = conv1x1(512, 256)
        self.conv5 = conv1x1(1024, 256)
        
        self.ffm = FeatureFusionModule(256+256, 128, act_type)

        self.seg_head = SegHead(128, num_class, act_type)
        if use_detail_head:
            self.detail_head = SegHead(256, 1, act_type)
            self.detail_conv = conv1x1(3, 1)

    def _make_stage(self, in_channels, out_channels, repeat_times, act_type):
        layers = [STDCModule(in_channels, out_channels, 2, act_type)]
        
        for _ in range(repeat_times):
            layers.append(STDCModule(out_channels, out_channels, 1, act_type))
        return nn.Sequential(*layers)

    def forward(self, x, is_training=False):
        size = x.size()[2:]

        x = self.stage1(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        if self.use_aux:
            aux3 = self.aux_head3(x3)

        x4 = self.stage4(x3)
        if self.use_aux:
            aux4 = self.aux_head4(x4)

        x5 = self.stage5(x4)
        if self.use_aux:
            aux5 = self.aux_head5(x5)

        x5_pool = self.pool(x5)
        x5 = x5_pool + self.arm5(x5)
        x5 = self.conv5(x5)
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)

        x4 = self.arm4(x4)
        x4 = self.conv4(x4)
        x4 += x5
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)

        x = self.ffm(x4, x3)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        if torch.onnx.is_in_onnx_export():
            # output_data = x.softmax(dim=1)
            max_probs, predictions = x.max(1, keepdim=True)
            return predictions.to(torch.int8)

        if self.use_detail_head and is_training:
            x_detail = self.detail_head(x3)
            return x, x_detail
        elif self.use_aux and is_training:
            return x, (aux3, aux4, aux5)
        else:
            return x


class STDCModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super(STDCModule, self).__init__()
        if out_channels % 8 != 0:
            raise ValueError('Output channel should be evenly divided by 8.\n')
        if stride not in [1, 2]:
            raise ValueError(f'Unsupported stride: {stride}\n')

        self.stride = stride
        self.block1 = ConvBNAct(in_channels, out_channels//2, 1)
        self.block2 = ConvBNAct(out_channels//2, out_channels//4, 3, stride)
        if self.stride == 2:
            self.pool = nn.AvgPool2d(3, 2, 1)
        self.block3 = ConvBNAct(out_channels//4, out_channels//8, 3)
        self.block4 = ConvBNAct(out_channels//8, out_channels//8, 3)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        if self.stride == 2:
            x1 = self.pool(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        return torch.cat([x1, x2, x3, x4], dim=1)


class LaplacianConv(nn.Module):
    def __init__(self, device):
        super(LaplacianConv, self).__init__()
        self.laplacian_kernel = torch.tensor([[[[-1.,-1.,-1.],[-1.,8.,-1.],[-1.,-1.,-1.]]]]).to(device)

    def forward(self, lbl):
        size = lbl.size()[2:]
        lbl_1x = F.conv2d(lbl, self.laplacian_kernel, stride=1, padding=1)
        lbl_2x = F.conv2d(lbl, self.laplacian_kernel, stride=2, padding=1)
        lbl_4x = F.conv2d(lbl, self.laplacian_kernel, stride=4, padding=1)

        lbl_2x = F.interpolate(lbl_2x, size, mode='nearest')
        lbl_4x = F.interpolate(lbl_4x, size, mode='nearest')

        lbl = torch.cat([lbl_1x, lbl_2x, lbl_4x], dim=1)

        return lbl
