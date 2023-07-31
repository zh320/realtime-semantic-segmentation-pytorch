"""
Paper:      PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model
Url:        https://arxiv.org/abs/2204.02681
Create by:  zh320
Date:       2023/07/15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, conv3x3, ConvBNAct


class PPLiteSeg(nn.Module):
    def __init__(self, num_class=1, n_channel=3, encoder_channels=[32, 64, 256, 512, 1024],
                    encoder_type='stdc1', fusion_type='spatial', act_type='relu'):
        super(PPLiteSeg, self).__init__()
        decoder_channel_hub = {'stdc1': [32, 64, 128], 'stdc2': [64, 96, 128]}
        decoder_channels = decoder_channel_hub[encoder_type]

        self.encoder = Encoder(n_channel, encoder_channels, encoder_type, act_type)
        self.sppm = SPPM(encoder_channels[-1], decoder_channels[0], act_type)
        self.decoder = FLD(encoder_channels, decoder_channels, num_class, fusion_type, act_type)

    def forward(self, x):
        size = x.size()[2:]
        x3, x4, x5 = self.encoder(x)
        x5 = self.sppm(x5)
        x = self.decoder(x3, x4, x5, size)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, encoder_channels, encoder_type, act_type):
        super(Encoder, self).__init__()
        encoder_hub = {'stdc1':STDCBackbone, 'stdc2':STDCBackbone}
        if encoder_type not in encoder_hub.keys():
            raise ValueError(f'Unsupport encoder type: {encoder_type}.\n')
        
        self.encoder = encoder_hub[encoder_type](in_channels, encoder_channels, encoder_type, act_type)
        
    def forward(self, x):
        x3, x4, x5 = self.encoder(x)

        return x3, x4, x5


class SPPM(nn.Module):
    def __init__(self, in_channels, out_channels, act_type):
        super(SPPM, self).__init__()
        hid_channels = int(in_channels // 4)
        self.act_type = act_type

        self.pool1 = self._make_pool_layer(in_channels, hid_channels, 1)
        self.pool2 = self._make_pool_layer(in_channels, hid_channels, 2)
        self.pool3 = self._make_pool_layer(in_channels, hid_channels, 4)
        self.conv = conv3x3(hid_channels, out_channels)

    def _make_pool_layer(self, in_channels, out_channels, pool_size):
        return nn.Sequential(
                        nn.AdaptiveAvgPool2d(pool_size),
                        ConvBNAct(in_channels, out_channels, 1, act_type=self.act_type)
                )

    def forward(self, x):
        size = x.size()[2:]
        x1 = F.interpolate(self.pool1(x), size, mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.pool2(x), size, mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.pool3(x), size, mode='bilinear', align_corners=True)
        x = self.conv(x1 + x2 + x3)
        
        return x


class FLD(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_class, fusion_type, act_type):
        super(FLD, self).__init__()
        self.stage6 = ConvBNAct(decoder_channels[0], decoder_channels[0])
        self.fusion1 = UAFM(encoder_channels[3], decoder_channels[0], fusion_type)
        self.stage7 = ConvBNAct(decoder_channels[0], decoder_channels[1])
        self.fusion2 = UAFM(encoder_channels[2], decoder_channels[1], fusion_type)
        self.stage8 = ConvBNAct(decoder_channels[1], decoder_channels[2])
        self.seg_head = ConvBNAct(decoder_channels[2], num_class, 3, act_type=act_type)

    def forward(self, x3, x4, x5, size):
        x = self.stage6(x5)
        x = self.fusion1(x, x4)
        x = self.stage7(x)
        x = self.fusion2(x, x3)
        x = self.stage8(x)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class STDCBackbone(nn.Module):
    def __init__(self, in_channels, encoder_channels, encoder_type, act_type):
        super(STDCBackbone, self).__init__()
        repeat_times_hub = {'stdc1': [1,1,1], 'stdc2': [3,4,2]}
        repeat_times = repeat_times_hub[encoder_type]
        self.stage1 = ConvBNAct(in_channels, encoder_channels[0], 3, 2)
        self.stage2 = ConvBNAct(encoder_channels[0], encoder_channels[1], 3, 2)
        self.stage3 = self._make_stage(encoder_channels[1], encoder_channels[2], repeat_times[0], act_type)
        self.stage4 = self._make_stage(encoder_channels[2], encoder_channels[3], repeat_times[1], act_type)
        self.stage5 = self._make_stage(encoder_channels[3], encoder_channels[4], repeat_times[2], act_type)

    def _make_stage(self, in_channels, out_channels, repeat_times, act_type):
        layers = [STDCModule(in_channels, out_channels, 2, act_type)]
        
        for _ in range(repeat_times):
            layers.append(STDCModule(out_channels, out_channels, 1, act_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        return x3, x4, x5


class STDCModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super(STDCModule, self).__init__()
        if out_channels % 8 != 0:
            raise ValueError('Output channel should be evenly divided by 8.\n')
        self.stride = stride
        self.block1 = ConvBNAct(in_channels, out_channels//2, 1)
        self.block2 = ConvBNAct(out_channels//2, out_channels//4, 3, stride)
        if self.stride == 2:
            self.pool = nn.AvgPool2d(3, 2, 1)
        self.block3 = ConvBNAct(out_channels//4, out_channels//8, 3)
        self.block4 = ConvBNAct(out_channels//8, out_channels//8, 3)
        
    def forward(self, x):
        x = self.block1(x)
        x2 = self.block2(x)
        if self.stride == 2:
            x = self.pool(x)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        return torch.cat([x, x2, x3, x4], dim=1)


class UAFM(nn.Module):
    def __init__(self, in_channels, out_channels, fusion_type):
        super(UAFM, self).__init__()
        fusion_hub = {'spatial': SpatialAttentionModule, 'channel': ChannelAttentionModule}
        if fusion_type not in fusion_hub.keys():
            raise ValueError(f'Unsupport fusion type: {fusion_type}.\n')

        self.conv = conv1x1(in_channels, out_channels)
        self.attention = fusion_hub[fusion_type](out_channels)

    def forward(self, x_high, x_low):
        size = x_low.size()[2:]
        x_low = self.conv(x_low)
        x_up = F.interpolate(x_high, size, mode='bilinear', align_corners=True)
        alpha = self.attention(x_up, x_low)
        x = alpha * x_up + (1 - alpha) * x_low
        
        return x


class SpatialAttentionModule(nn.Module):
    def __init__(self, out_channels):
        super(SpatialAttentionModule, self).__init__()
        self.conv = conv1x1(4, 1)
        
    def forward(self, x_up, x_low):
        mean_up = torch.mean(x_up, dim=1, keepdim=True)
        max_up, _ = torch.max(x_up, dim=1, keepdim=True)
        mean_low = torch.mean(x_low, dim=1, keepdim=True)
        max_low, _ = torch.max(x_low, dim=1, keepdim=True)
        x = self.conv(torch.cat([mean_up, max_up, mean_low, max_low], dim=1))
        x = torch.sigmoid(x)    # [N, 1, H, W]
        
        return x


class ChannelAttentionModule(nn.Module):
    def __init__(self, out_channels):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = conv1x1(4*out_channels, out_channels)

    def forward(self, x_up, x_low):
        avg_up = self.avg_pool(x_up)
        max_up = self.max_pool(x_up)
        avg_low = self.avg_pool(x_low)
        max_low = self.max_pool(x_low)
        x = self.conv(torch.cat([avg_up, max_up, avg_low, max_low], dim=1))
        x = torch.sigmoid(x)    # [N, C, 1, 1]

        return x
