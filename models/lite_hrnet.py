"""
Paper:      Lite-HRNet: A Lightweight High-Resolution Network
Url:        https://arxiv.org/abs/2104.06403
Create by:  zh320
Date:       2024/03/10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, DSConvBNAct, DWConvBNAct, ConvBNAct, channel_shuffle


class LiteHRNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, base_ch=40, arch_type='litehrnet18', 
                    repeat=2, act_type='relu'):
        super(LiteHRNet, self).__init__()
        arch_hub = {'litehrnet18': [2,4,2], 'litehrnet30': [3,8,3],}
        if arch_type not in arch_hub.keys():
            raise ValueError(f'Unsupport architecture type: {arch_type}.\n')
        num_modules = arch_hub[arch_type]

        self.stem = nn.Sequential(
                        ConvBNAct(n_channel, 32, 3, 2, act_type=act_type),
                        ShuffleBlock(32, base_ch, 2, act_type)
                    )
        self.stage1_down = DSConvBNAct(base_ch, base_ch*2, 3, 2, act_type=act_type)
        self.stage2 = StageBlock(base_ch, 2, repeat, num_modules[0], act_type)
        self.stage3 = StageBlock(base_ch, 3, repeat, num_modules[1], act_type)
        self.stage4 = StageBlock(base_ch, 4, repeat, num_modules[2], act_type)
        self.rep_head = RepresentationHead(base_ch, num_class, 4, act_type)

    def forward(self, x):
        size = x.size()[2:]

        x = self.stem(x)
        x2 = self.stage1_down(x)
        feats = [x, x2]

        feats = self.stage2(feats)
        feats = self.stage3(feats)
        feats = self.stage4(feats)
        x = self.rep_head(feats)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class StageBlock(nn.Module):
    def __init__(self, base_ch, stage, repeat, num_modules, act_type):
        super(StageBlock, self).__init__()
        assert stage >= 2
        assert repeat > 0
        assert num_modules > 0
        chs = [2**i*base_ch for i in range(stage)]
        crw_ch = sum(chs) // 2

        self.stage_blocks = nn.ModuleList([])
        for i in range(num_modules):
            crw_module = CrossResolutionWeightModule(crw_ch, act_type)

            # CCW Block
            ccw_blocks = nn.ModuleList([])
            for j in range(stage):
                ccw_block = nn.ModuleList([CCWBlock(chs[j], chs[j], 1, act_type) for _ in range(repeat)])
                ccw_blocks.append(ccw_block)

            # Fusion Block
            extra_output = (i == num_modules - 1) and (stage != 4)
            fusion_block = FusionBlock(base_ch, stage, extra_output, act_type)

            self.stage_blocks.append(crw_module)
            self.stage_blocks.append(ccw_blocks)
            self.stage_blocks.append(fusion_block)
            assert len(self.stage_blocks) % 3 == 0

    def forward(self, feats):
        for i in range(len(self.stage_blocks) // 3):
            crw_module = self.stage_blocks[i*3]
            ccw_blocks = self.stage_blocks[i*3+1]
            fusion_block = self.stage_blocks[i*3+2]

            cr_weight = crw_module(feats)
            for j, ccw_block in enumerate(ccw_blocks):
                for m in ccw_block:
                    feats[j] = m(feats[j], cr_weight[j])
            feats = fusion_block(feats)

        return feats


class RepresentationHead(nn.Module):
    def __init__(self, base_ch, num_class, num_stage, act_type, hid_ch=128):
        super(RepresentationHead, self).__init__()
        self.up = nn.ModuleList([nn.Identity()])
        for i in range(num_stage-1):
            self.up.append(nn.Upsample(scale_factor=2**(i+1), mode='bilinear', align_corners=True))

        in_ch = sum([2**i for i in range(num_stage)]) * base_ch
        self.seg_head = nn.Sequential(
                            DSConvBNAct(in_ch, hid_ch, 3, act_type=act_type),
                            conv1x1(hid_ch, num_class)
                        )
        # self.seg_head = conv1x1(in_ch, num_class)

    def forward(self, feats):
        for i, m in enumerate(self.up):
            feats[i] = m(feats[i])

        x = torch.cat(feats, dim=1)
        x = self.seg_head(x)

        return x


class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super(ShuffleBlock, self).__init__()
        assert stride in [1, 2]
        in_ch_l, out_ch_l = in_channels//2, out_channels//2
        in_ch_r, out_ch_r = in_channels-in_ch_l, out_channels-out_ch_l
        self.in_ch_l = in_ch_l

        if stride != 1 or in_ch_l != out_ch_l:
            self.left_branch = ConvBNAct(in_ch_l, out_ch_l, 1, stride, act_type=act_type)
        else:
            self.left_branch = nn.Identity()

        self.right_branch = nn.Sequential(
                                ConvBNAct(in_ch_r, out_ch_r, 1, act_type=act_type),
                                DWConvBNAct(out_ch_r, out_ch_r, 3, stride, act_type=act_type),
                                ConvBNAct(out_ch_r, out_ch_r, 1, act_type=act_type)
                            )

    def forward(self, x):
        x_l = x[:, :self.in_ch_l]
        x_r = x[:, self.in_ch_l:]

        x_l = self.left_branch(x_l)
        x_r = self.right_branch(x_r)

        x = torch.cat([x_l, x_r], dim=1)
        x = channel_shuffle(x)

        return x


class CCWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, act_type):
        super(CCWBlock, self).__init__()
        assert stride in [1, 2]
        in_ch_l, out_ch_l = in_channels//2, out_channels//2
        in_ch_r, out_ch_r = in_channels-in_ch_l, out_channels-out_ch_l

        self.split_ch = [in_ch_l, in_ch_r]
        if stride != 1 or in_ch_l != out_ch_l:
            self.left_branch = ConvBNAct(in_ch_l, out_ch_l, 1, stride, act_type=act_type)
        else:
            self.left_branch = nn.Identity()

        self.right_branch = DWConvBNAct(in_ch_r, out_ch_r, 3, stride, act_type=act_type)
        self.sw = SpatialWeightModule(out_ch_r, act_type)

    def forward(self, feats, cr_weight):
        feats_l, feats_r = torch.split(feats, self.split_ch, dim=1)

        # Left branch
        feats_l = self.left_branch(feats_l)

        # Right branch
        size = feats_r.size()[2:]
        cr_weight = F.interpolate(cr_weight, size, mode='nearest')
        feats_r = self.right_branch(feats_r * cr_weight)
        spatial_weight = self.sw(feats_r)
        feats_r = feats_r * spatial_weight

        feats = torch.cat([feats_l, feats_r], dim=1)
        feats = channel_shuffle(feats)

        return feats


class CrossResolutionWeightModule(nn.Module):
    def __init__(self, channels, act_type, ch_reduction=8, pool_size=None):
        super(CrossResolutionWeightModule, self).__init__()
        hid_channels = channels // ch_reduction
        self.pool_size = pool_size
        self.conv = nn.Sequential(
                        ConvBNAct(channels, hid_channels, 1, act_type=act_type),
                        ConvBNAct(hid_channels, channels, 1, act_type='sigmoid')
                    )

    def forward(self, feats):
        pool_size = feats[-1].size()[2:] if self.pool_size is None else self.pool_size
        ch_r = [feat.size()[1]//2 for feat in feats]

        cr_weight = []
        for i, feat in enumerate(feats):
            if i == len(feats)-1:
                cr_weight.append(feat[:, ch_r[i]:])
            else:
                cr_weight.append(F.adaptive_avg_pool2d(feat[:, ch_r[i]:], pool_size))
        cr_weight = torch.cat(cr_weight, dim=1)
        cr_weight = self.conv(cr_weight)
        cr_weight = torch.split(cr_weight, ch_r, dim=1)

        return cr_weight


class FusionBlock(nn.Module):
    def __init__(self, base_ch, stage, extra_output, act_type):
        super(FusionBlock, self).__init__()
        assert stage in [2,3,4]
        self.stage = stage
        self.extra_output = extra_output

        channels = list(range(stage))
        if extra_output:
            channels.append(stage)
        channels = [2**ch * base_ch for ch in channels]

        # Stream 1
        num_stage = stage + 1 if extra_output else stage
        self.stream1 = nn.ModuleList([nn.Identity()])
        self.stream1.extend([DownsampleBlock(channels[0], channels[i], i, act_type) for i in range(1, num_stage)])

        # Stream 2
        num_stage = stage if extra_output else stage-1
        self.stream2 = nn.ModuleList([UpsampleBlock(channels[1], channels[0], 2, act_type), nn.Identity()])
        self.stream2.extend([DownsampleBlock(channels[1], channels[i+1], i, act_type) for i in range(1, num_stage)])

        # Stream 3
        if stage in [3, 4]:
            self.stream3 = nn.ModuleList([UpsampleBlock(channels[2], channels[2-i], 2**i, act_type) for i in range(2, 0, -1)])
            self.stream3.append(nn.Identity())
            if extra_output or stage == 4:
                self.stream3.append(DownsampleBlock(channels[2], channels[3], 1, act_type))

        # Stream 4
        if stage == 4:
            self.stream4 = nn.ModuleList([UpsampleBlock(channels[3], channels[3-i], 2**i, act_type) for i in range(3, 0, -1)])
            self.stream4.append(nn.Identity())

    def forward(self, feats):
        assert len(feats) == self.stage
        x3, x4 = None, None

        x1 = self.stream1[0](feats[0]) + self.stream2[0](feats[1])
        x2 = self.stream1[1](feats[0]) + self.stream2[1](feats[1])
        if self.stage in [3, 4] or self.extra_output:
            x3 = self.stream1[2](feats[0]) + self.stream2[2](feats[1])

        if self.stage in [3, 4]:
            x1 += self.stream3[0](feats[2])
            x2 += self.stream3[1](feats[2])
            x3 += self.stream3[2](feats[2])
            if self.stage == 4 or self.extra_output:
                x4 = self.stream1[3](feats[0]) + self.stream2[3](feats[1]) + self.stream3[3](feats[2])

                if self.stage == 4:
                    x1 += self.stream4[0](feats[3])
                    x2 += self.stream4[1](feats[3])
                    x3 += self.stream4[2](feats[3])
                    x4 += self.stream4[3](feats[3])

        res = [x1, x2]
        if x3 is not None:
            res.append(x3)

        if x4 is not None:
            res.append(x4)

        return res


class SpatialWeightModule(nn.Module):
    def __init__(self, channels, act_type, ch_reduction=8):
        super(SpatialWeightModule, self).__init__()
        hid_channels = channels // ch_reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                        ConvBNAct(channels, hid_channels, 1, act_type=act_type),
                        ConvBNAct(hid_channels, channels, 1, act_type='sigmoid')
                    )

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_avg = self.fc(x_avg)

        return x_avg


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, scale_factor, act_type):
        super(UpsampleBlock, self).__init__(
            ConvBNAct(in_ch, out_ch, 1, act_type=act_type),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        )


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, act_type):
        super(DownsampleBlock, self).__init__()
        assert num_block >= 1

        layers = []
        if num_block > 1:
            for i in range(num_block):
                hid_ch = in_ch if i != num_block - 1 else out_ch
                layers.append(DSConvBNAct(in_ch, hid_ch, 3, 2, act_type=act_type))
        else:
            layers.append(DSConvBNAct(in_ch, out_ch, 3, 2, act_type=act_type))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)

        return x
