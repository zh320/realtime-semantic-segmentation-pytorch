"""
Paper:      DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation
Url:        https://arxiv.org/abs/1904.02216
Create by:  zh320
Date:       2023/10/22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, DSConvBNAct, DWConvBNAct, ConvBNAct, Activation, SegHead


class DFANet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='XceptionA', expansion=4, 
                    repeat_times=[4,6,4], use_extra_backbone=True, act_type='relu'):
        super(DFANet, self).__init__()
        assert len(repeat_times) == 3
        if backbone_type == 'XceptionA':
            channels = [48, 96, 192]
        elif backbone_type == 'XceptionB':
            channels = [32, 64, 128]
        else:
            raise NotImplementedError()
        self.use_extra_backbone = use_extra_backbone

        self.conv1 = ConvBNAct(n_channel, 8, 3, 2, act_type=act_type)

        in_channels = [8, channels[0], channels[1]]
        self.backbone1 = Encoder(in_channels, channels, expansion, repeat_times, act_type)

        if self.use_extra_backbone:
            # Rotate the channels to perform features fusion
            new_channels = channels[2:] + channels[:2]
            in_channels = [(channels[i] + new_channels[i]) for i in range(len(channels))]
            self.backbone2 = Encoder(in_channels, channels, expansion, repeat_times, act_type)
            self.backbone3 = Encoder(in_channels, channels, expansion, repeat_times, act_type)

            self.decoder = Decoder(channels[0], channels[2], num_class, act_type)
        else:
            self.seg_head = SegHead(channels[2], num_class, act_type)

    def forward(self, x):
        x = self.conv1(x)

        x, x_enc2, x_enc3, x_enc4 = self.backbone1(x)
        
        if self.use_extra_backbone:
            enc_x1, fc_x1 = x_enc2, x
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

            x, x_enc2, x_enc3, x_enc4 = self.backbone2(x, x_enc2, x_enc3, x_enc4)
            enc_x2, fc_x2 = x_enc2, x
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

            fc_x3, enc_x3, _, _ = self.backbone3(x, x_enc2, x_enc3, x_enc4)

            x = self.decoder(enc_x1, enc_x2, enc_x3, fc_x1, fc_x2, fc_x3)
        else:
            x = self.seg_head(x)
            x = F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, channels, expansion, repeat_times, act_type):
        super(Encoder, self).__init__()
        assert len(in_channels) == 3
        self.enc2 = EncoderBlock(in_channels[0], channels[0], expansion, repeat_times[0], act_type)
        self.enc3 = EncoderBlock(in_channels[1], channels[1], expansion, repeat_times[1], act_type)
        self.enc4 = EncoderBlock(in_channels[2], channels[2], expansion, repeat_times[2], act_type)
        self.fc_attention = FCAttention(channels[2], act_type)

    def forward(self, x, x_enc2=None, x_enc3=None, x_enc4=None):
        if x_enc2 is not None:
            x = torch.cat([x, x_enc2], dim=1)
        x = self.enc2(x)
        x_enc2 = x

        if x_enc3 is not None:
            x = torch.cat([x, x_enc3], dim=1)
        x = self.enc3(x)
        x_enc3 = x

        if x_enc4 is not None:
            x = torch.cat([x, x_enc4], dim=1)
        x = self.enc4(x)
        x_enc4 = x

        x = self.fc_attention(x)

        return x, x_enc2, x_enc3, x_enc4


class Decoder(nn.Module):
    def __init__(self, enc_channels, fc_channels, num_class, act_type, hid_channels=48):
        super(Decoder, self).__init__()
        self.enc_conv1 = ConvBNAct(enc_channels, hid_channels, 3, act_type=act_type, inplace=True)
        self.enc_conv2 = ConvBNAct(enc_channels, hid_channels, 3, act_type=act_type, inplace=True)
        self.enc_conv3 = ConvBNAct(enc_channels, hid_channels, 3, act_type=act_type, inplace=True)
        self.conv_enc = conv1x1(hid_channels, num_class)
        
        self.fc_conv1 = SegHead(fc_channels, num_class, act_type)
        self.fc_conv2 = SegHead(fc_channels, num_class, act_type)
        self.fc_conv3 = SegHead(fc_channels, num_class, act_type)
        
    def forward(self, enc_x1, enc_x2, enc_x3, fc_x1, fc_x2, fc_x3):
        enc_x1 = self.enc_conv1(enc_x1)
        enc_x2 = self.enc_conv2(enc_x2)
        enc_x2 = F.interpolate(enc_x2, scale_factor=2, mode='bilinear', align_corners=True)
        enc_x3 = self.enc_conv3(enc_x3)
        enc_x3 = F.interpolate(enc_x3, scale_factor=4, mode='bilinear', align_corners=True)

        enc_x = enc_x1 + enc_x2 + enc_x3
        enc_x = self.conv_enc(enc_x)

        fc_x1 = self.fc_conv1(fc_x1)
        fc_x1 = F.interpolate(fc_x1, scale_factor=4, mode='bilinear', align_corners=True)
        fc_x2 = self.fc_conv2(fc_x2)
        fc_x2 = F.interpolate(fc_x2, scale_factor=8, mode='bilinear', align_corners=True)
        fc_x3 = self.fc_conv3(fc_x3)
        fc_x3 = F.interpolate(fc_x3, scale_factor=16, mode='bilinear', align_corners=True)

        x = enc_x + fc_x1 + fc_x2 + fc_x3
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, repeat_times, act_type):
        super(EncoderBlock, self).__init__()
        layers = [XceptionBlock(in_channels, out_channels, 2, expansion, act_type)]

        for _ in range(repeat_times-1):
            layers.append(XceptionBlock(out_channels, out_channels, 1, expansion, act_type))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class FCAttention(nn.Module):
    def __init__(self, channels, act_type, linear_channels=1000):
        super(FCAttention, self).__init__()
        self.channels = channels
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.linear = nn.Linear(channels, linear_channels)
        self.conv = ConvBNAct(linear_channels, channels, 1, act_type=act_type, inplace=True)

    def forward(self, x):
        attention = self.pool(x).view(-1, self.channels)
        attention = self.linear(attention)
        attention = attention.unsqueeze(-1).unsqueeze(-1)
        attention = self.conv(attention)
        x = x * attention

        return x


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, act_type):
        super(XceptionBlock, self).__init__()
        self.use_skip = (in_channels == out_channels) and (stride == 1)
        self.stride = stride
        hid_channels = out_channels // expansion
        self.conv = nn.Sequential(
                        # Activation(act_type, inplace=True),
                        DSConvBNAct(in_channels, hid_channels, 3, act_type=act_type),
                        DSConvBNAct(hid_channels, hid_channels, 3, act_type=act_type),
                        DWConvBNAct(hid_channels, out_channels, 3, stride, act_type=act_type, inplace=True),
                        conv1x1(out_channels, out_channels),
                        Activation(act_type),
                    )
        if stride > 1:
            self.conv_stride = conv1x1(in_channels, out_channels, 2)

    def forward(self, x):
        if self.use_skip:
            residual = x

        x_right = self.conv(x)

        if self.stride > 1:
            x_left = self.conv_stride(x)
            x_right += x_left

        if self.use_skip:
            x_right += residual

        return x_right
