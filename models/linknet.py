"""
Paper:      LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
Url:        https://arxiv.org/abs/1707.03718
Create by:  zh320
Date:       2023/04/23
"""

import torch
import torch.nn as nn

from .modules import ConvBNAct, Activation


class LinkNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3):
        super(LinkNet, self).__init__()
        self.init_block = InitBlock(n_channel, 64)
        self.enc_block1 = EncoderBlock(64, 64)
        self.enc_block2 = EncoderBlock(64, 128)
        self.enc_block3 = EncoderBlock(128, 256)
        self.enc_block4 = EncoderBlock(256, 512)
        self.dec_block4 = DecoderBlock(512, 256)
        self.dec_block3 = DecoderBlock(256, 128)
        self.dec_block2 = DecoderBlock(128, 64)
        self.dec_block1 = DecoderBlock(64, 64)
        self.seg_head = SegHead(64, num_class)

    def forward(self, x):
        x = self.init_block(x)
        x1 = self.enc_block1(x)
        x2 = self.enc_block2(x1)
        x3 = self.enc_block3(x2)
        x = self.enc_block4(x3)
        x = self.dec_block4(x)
        x = self.dec_block3(x + x3)
        x = self.dec_block2(x + x2)
        x = self.dec_block1(x + x1)
        x = self.seg_head(x)
        return x


class InitBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(InitBlock, self).__init__(
                ConvBNAct(in_channels, out_channels, 7, 2),
                nn.MaxPool2d(3, 2, 1)
        )


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBNAct(in_channels, out_channels, 1, 2, act_type='none')
        self.block1 = nn.Sequential(
                            ConvBNAct(in_channels, out_channels, 3, 2),
                            ConvBNAct(out_channels, out_channels, 3, 1, act_type='none'),
                        )
        self.act1 = nn.ReLU()
        self.block2 = nn.Sequential(
                            ConvBNAct(out_channels, out_channels, 3, 1),
                            ConvBNAct(out_channels, out_channels, 3, 1, act_type='none')
                        )
        self.act2 = nn.ReLU()

    def forward(self, x):
        res1 = self.block1(x)
        x = self.conv(x)
        res1 += x
        res1 = self.act1(res1)
        
        res2 = self.block2(res1)
        res2 += res1
        res2 = self.act2(res2)
        return res2


class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        hid_channels = in_channels // 4
        super(DecoderBlock, self).__init__(
                ConvBNAct(in_channels, hid_channels, 1),
                Upsample(hid_channels, hid_channels, 2, 4, 1),
                ConvBNAct(hid_channels, out_channels, 1)
        )


class SegHead(nn.Sequential):
    def __init__(self, in_channels, num_class):
        hid_channels = in_channels // 2
        super(SegHead, self).__init__(
                Upsample(in_channels, hid_channels, 2, 4, 1),
                ConvBNAct(hid_channels, hid_channels, 3),
                Upsample(hid_channels, num_class, 2, 4, 1)
        )


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, kernel_size=None, padding=None,
                    upsample_type='deconvolution', act_type='relu'):
        super(Upsample, self).__init__()
        if upsample_type == 'deconvolution':
            if kernel_size is None:
                kernel_size = 2*scale_factor - 1
            if padding is None:    
                padding = (kernel_size - 1) // 2
            self.up_conv = nn.Sequential(
                                    nn.ConvTranspose2d(in_channels, out_channels, 
                                                        kernel_size=kernel_size, 
                                                        stride=scale_factor, padding=padding,),
                                    nn.BatchNorm2d(out_channels),
                                    Activation(act_type)
                            )
        else:
            self.up_conv = nn.Sequential(
                                    ConvBNAct(in_channels, out_channels, 1, act_type=act_type),
                                    nn.Upsample(scale_factor=scale_factor, mode='bilinear')
                            )

    def forward(self, x):
        return self.up_conv(x)
