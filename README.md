# Introduction

PyTorch implementation of realtime semantic segmentation models, support multi-gpu training and validating, automatic mixed precision training, knowledge distillation etc.  
\
<img src="https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/enet_800epoch.gif" width="100%" height="100%" />

# Requirements

torch == 1.8.1  
segmentation-models-pytorch  
torchmetrics  
albumentations  
loguru  
tqdm  

# Supported models

- [ADSCNet](models/adscnet.py) [^adscnet]  
- [AGLNet](models/aglnet.py) [^aglnet]  
- [BiSeNetv1](models/bisenetv1.py) [^bisenetv1]  
- [BiSeNetv2](models/bisenetv2.py) [^bisenetv2]  
- [CANet](models/canet.py) [^canet]  
- [CFPNet](models/cfpnet.py) [^cfpnet]  
- [CGNet](models/cgnet.py) [^cgnet]  
- [ContextNet](models/contextnet.py)[^contextnet]  
- [DABNet](models/dabnet.py)[^dabnet]  
- [DDRNet](models/ddrnet.py)[^ddrnet]  
- [DFANet](models/dfanet.py)[^dfanet]  
- [EDANet](models/edanet.py) [^edanet]  
- [ENet](models/enet.py) [^enet]  
- [ERFNet](models/erfnet.py) [^erfnet]  
- [ESNet](models/esnet.py) [^esnet]  
- [ESPNet](models/espnet.py) [^espnet]  
- [ESPNetv2](models/espnetv2.py) [^espnetv2]  
- [FarseeNet](models/farseenet.py) [^farseenet]  
- [FastSCNN](models/fastscnn.py) [^fastscnn]  
- [FDDWNet](models/fddwnet.py) [^fddwnet]  
- [FPENet](models/fpenet.py) [^fpenet]  
- [FSSNet](models/fssnet.py) [^fssnet]  
- [ICNet](models/icnet.py) [^icnet]  
- [LEDNet](models/lednet.py) [^lednet]  
- [LinkNet](models/linknet.py)[^linknet]  
- [Lite-HRNet](models/lite_hrnet.py)[^lite_hrnet]
- [LiteSeg](models/liteseg.py)[^liteseg]  
- [MiniNet](models/mininet.py)[^mininet]  
- [MiniNetv2](models/mininetv2.py)[^mininetv2]  
- [PP-LiteSeg](models/pp_liteseg.py)[^ppliteseg]  
- [RegSeg](models/regseg.py)[^regseg]  
- [SegNet](models/segnet.py)[^segnet]  
- [ShelfNet](models/shelfnet.py)[^shelfnet]  
- [SQNet](models/sqnet.py)[^sqnet]  
- [STDC](models/stdc.py)[^stdc]  
- [SwiftNet](models/swiftnet.py)[^swiftnet]  

If you want to use encoder-decoder structure with pretrained encoders, you may refer to: segmentation-models-pytorch[^smp]. This repo also provides easy access to SMP. Just modify the [config file](configs/my_config.py) to (e.g. if you want to train DeepLabv3Plus with ResNet-101 backbone as teacher model to perform knowledge distillation)  

```
self.model = 'smp'
self.encoder = 'resnet101'
self.decoder = 'deeplabv3p'
```

or use [command-line arguments](configs/parser.py)  

```
python main.py --model smp --encoder resnet101 --decoder deeplabv3p
```

Details of the configurations can also be found in this [file](configs/parser.py).  

[^adscnet]: [ADSCNet: asymmetric depthwise separable convolution for semantic segmentation in real-time](https://link.springer.com/article/10.1007/s10489-019-01587-1)  
[^aglnet]: [AGLNet: Towards real-time semantic segmentation of self-driving images via attention-guided lightweight network](https://www.sciencedirect.com/science/article/abs/pii/S1568494620306207)  
[^bisenetv1]: [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897)  
[^bisenetv2]: [BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](https://arxiv.org/abs/2004.02147)  
[^canet]: [Cross Attention Network for Semantic Segmentation](https://arxiv.org/abs/1907.10958)  
[^cfpnet]: [CFPNet: Channel-wise Feature Pyramid for Real-Time Semantic Segmentation](https://arxiv.org/abs/2103.12212)  
[^cgnet]: [CGNet: A Light-weight Context Guided Network for Semantic Segmentation](https://arxiv.org/abs/1811.08201)  
[^contextnet]: [ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time](https://arxiv.org/abs/1805.04554)  
[^dabnet]: [DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation](https://arxiv.org/abs/1907.11357)  
[^ddrnet]: [Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes](https://arxiv.org/abs/2101.06085)  
[^dfanet]: [DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation](https://arxiv.org/abs/1904.02216)  
[^edanet]: [Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation](https://arxiv.org/abs/1809.06323)  
[^enet]: [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147)  
[^erfnet]: [ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation](https://ieeexplore.ieee.org/document/8063438)  
[^esnet]: [ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1906.09826)  
[^espnet]: [ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation](https://arxiv.org/abs/1803.06815)  
[^espnetv2]: [ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network](https://arxiv.org/abs/1811.11431)  
[^farseenet]: [FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale Context Aggregation and Feature Space Super-resolution](https://arxiv.org/abs/2003.03913)  
[^fastscnn]: [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502)  
[^fddwnet]: [FDDWNet: A Lightweight Convolutional Neural Network for Real-time Sementic Segmentation](https://arxiv.org/abs/1911.00632)  
[^fpenet]: [Feature Pyramid Encoding Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1909.08599)  
[^fssnet]: [Fast Semantic Segmentation for Scene Perception](https://ieeexplore.ieee.org/document/8392426)  
[^icnet]: [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)  
[^lednet]: [LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/1905.02423)  
[^linknet]: [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/abs/1707.03718)  
[^lite_hrnet]: [Lite-HRNet: A Lightweight High-Resolution Network](https://arxiv.org/abs/2104.06403)  
[^liteseg]: [LiteSeg: A Novel Lightweight ConvNet for Semantic Segmentation](https://arxiv.org/abs/1912.06683)  
[^mininet]: [Enhancing V-SLAM Keyframe Selection with an Efficient ConvNet for Semantic Analysis](https://ieeexplore.ieee.org/abstract/document/8793923)  
[^mininetv2]: [MiniNet: An Efficient Semantic Segmentation ConvNet for Real-Time Robotic Applications](https://ieeexplore.ieee.org/abstract/document/9023474)  
[^ppliteseg]: [PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model](https://arxiv.org/abs/2204.02681)  
[^regseg]: [Rethinking Dilated Convolution for Real-time Semantic Segmentation](https://arxiv.org/abs/2111.09957)  
[^segnet]: [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)  
[^shelfnet]: [ShelfNet for Fast Semantic Segmentation](https://arxiv.org/abs/1811.11254)  
[^sqnet]: [Speeding up Semantic Segmentation for Autonomous Driving](https://openreview.net/pdf?id=S1uHiFyyg)  
[^stdc]: [Rethinking BiSeNet For Real-time Semantic Segmentation](https://arxiv.org/abs/2104.13188v1)  
[^swiftnet]: [In Defense of Pre-trained ImageNet Architectures for Real-time Semantic Segmentation of Road-driving Images](https://arxiv.org/abs/1903.08469)  
[^smp]: [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)  

# Knowledge Distillation

Currently only support the original knowledge distillation method proposed by Geoffrey Hinton.[^kd]  

[^kd]: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)  

# How to use

## DDP training (recommend)

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
```

## DP training

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

# Performances and checkpoints

## full resolution on Cityscapes

| Model      | Year | Encoder        | Params(M) <br> paper/my | FPS<sup>1</sup> | mIoU(paper) <br> val/test         | mIoU(my) val<sup>2</sup>                                                                                                      |
| ---------- |:----:|:--------------:|:----------------------- |:---------------:| --------------------------------- |:-----------------------------------------------------------------------------------------------------------------------------:|
| ADSCNet    | 2019 | None           | n.a./0.51               | 89              | n.a./67.5                         | [69.06](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/adscnet.pth)                   |
| AGLNet     | 2020 | None           | 1.12/1.02               | 61              | 69.39/70.1                        | [73.58](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/aglnet.pth)                    |
| BiSeNetv1  | 2018 | ResNet18       | 49.0/13.32              | 88              | 74.8/74.7                         | [74.91](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/bisenetv1.pth)                 |
| BiSeNetv2  | 2020 | None           | n.a./2.27               | 142             | 73.4/72.6                         | [73.73](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/bisenetv2-aux.pth)<sup>3</sup> |
| CANet      | 2019 | MobileNetv2    | 4.8/4.77                | 76              | 73.4/73.5                         | [76.59](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.1/canet.pth)                     |
| CFPNet     | 2021 | None           | 0.55/0.27               | 64              | n.a./70.1                         | [70.08](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/cfpnet.pth)                    |
| CGNet      | 2018 | None           | 0.41/0.24               | 157             | 59.7/64.8<sup>4</sup>             | [67.25](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/cgnet.pth)                     |
| ContextNet | 2018 | None           | 0.85/1.01               | 80              | 65.9/66.1                         | [66.61](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/contextnet.pth)                |
| DABNet     | 2019 | None           | 0.76/0.75               | 140             | n.a./70.1                         | [70.78](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/dabnet.pth)                    |
| DDRNet     | 2021 | None           | 5.7/5.54                | 233             | 77.8/77.4                         | [74.34](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/ddrnet-23-slim.pth)            |
| DFANet     | 2019 | XceptionA      | 7.8/3.05                | 60              | 71.9/71.3                         | [65.28](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/dfanet-a.pth)                  |
| EDANet     | 2018 | None           | 0.68/0.69               | 125             | n.a./67.3                         | [70.76](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/edanet.pth)                    |
| ENet       | 2016 | None           | 0.37/0.37               | 140             | n.a./58.3                         | [71.31](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/enet.pth)                      |
| ERFNet     | 2017 | None           | 2.06/2.07               | 60              | 70.0/68.0                         | [76.00](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/erfnet.pth)                    |
| ESNet      | 2019 | None           | 1.66/1.66               | 66              | n.a./70.7                         | [71.82](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/esnet.pth)                     |
| ESPNet     | 2018 | None           | 0.36/0.38               | 111             | n.a./60.3                         | [66.39](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/espnet.pth)                    |
| ESPNetv2   | 2018 | None           | 1.25/0.86               | 101             | 66.4/66.2                         | [70.35](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/espnetv2.pth)                  |
| FarseeNet  | 2020 | ResNet18       | n.a./16.75              | 130             | 73.5/70.2                         | [77.35](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/farseenet.pth)                 |
| FastSCNN   | 2019 | None           | 1.11/1.02               | 358             | 68.6/68.0                         | [69.37](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/fastscnn.pth)                  |
| FDDWNet    | 2019 | None           | 0.80/0.77               | 51              | n.a./71.5                         | [75.86](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/fddwnet.pth)                   |
| FPENet     | 2019 | None           | 0.38/0.36               | 90              | n.a./70.1                         | [72.05](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/fpenet.pth)                    |
| FSSNet     | 2018 | None           | 0.2/0.20                | 121             | n.a./58.8                         | [65.44](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/fssnet.pth)                    |
| ICNet      | 2017 | ResNet18       | 26.5<sup>5</sup>/12.42  | 102             | 67.7<sup>5</sup>/69.5<sup>5</sup> | [69.65](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/icnet.pth)                     |
| LEDNet     | 2019 | None           | 0.94/1.46               | 76              | n.a./70.6                         | [72.63](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.1/lednet.pth)                    |
| LinkNet    | 2017 | ResNet18       | 11.5/11.54              | 106             | n.a./76.4                         | [73.39](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.1/linknet.pth)                   |
| Lite-HRNet | 2021 | None           | 1.1/1.09                | 30              | 73.8/72.8                         | [70.66](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/lite_hrnet.pth)                |
| LiteSeg    | 2019 | MobileNetv2    | 4.38/4.29               | 117             | 70.0/67.8                         | [76.10](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.1/liteseg.pth)                   |
| MiniNet    | 2019 | None           | 3.1/1.41                | 254             | n.a./40.7                         | [61.47](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.1/mininet.pth)                   |
| MiniNetv2  | 2020 | None           | 0.5/0.51                | 86              | n.a./70.5                         | [71.79](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/mininetv2.pth)                 |
| PP-LiteSeg | 2022 | STDC1          | n.a./6.33               | 201             | 76.0/74.9                         | [72.49](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/ppliteseg_stdc1.pth)           |
| PP-LiteSeg | 2022 | STDC2          | n.a./10.56              | 136             | 78.2/77.5                         | [74.37](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/ppliteseg_stdc2.pth)           |
| RegSeg     | 2021 | None           | 3.34/3.37               | 104             | 78.5/78.3                         | [74.28](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/regseg.pth)                    |
| SegNet     | 2015 | None           | 29.46/29.48             | 14              | n.a./56.1                         | [70.77](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/segnet.pth)                    |
| ShelfNet   | 2018 | ResNet18       | 23.5/16.04              | 110             | n.a./74.8                         | [77.63](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/shelfnet.pth)                  |
| SQNet      | 2016 | SqueezeNet-1.1 | n.a./4.81               | 69              | n.a./59.8                         | [69.55](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/sqnet.pth)                     |
| STDC       | 2021 | STDC1          | n.a./7.79               | 163             | 74.5/75.3                         | [75.25](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/stdc1.pth)<sup>6</sup>         |
| STDC       | 2021 | STDC2          | n.a./11.82              | 119             | 77.0/76.8                         | [76.78](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/stdc2.pth)<sup>6</sup>         |
| SwiftNet   | 2019 | ResNet18       | 11.8/11.95              | 141             | 75.4/75.5                         | [75.43](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/swiftnet.pth)                  |

[<sup>1</sup>FPSs are evaluated on RTX 2080 at resolution 1024x512 using this [script](tools/test_speed.py). Please note that FPSs vary between devices and hardwares and also depend on other factors (e.g. whether to use cudnn or not). To obtain accurate FPSs, please test them on your device accordingly.]  
[<sup>2</sup>These results are obtained by training 800 epochs with crop-size 1024x1024]  
[<sup>3</sup>These results are obtained by using auxiliary heads]  
[<sup>4</sup>This result is obtained by using deeper model, i.e. CGNet_M3N21]  
[<sup>5</sup>The original encoder of ICNet is ResNet50]  
[<sup>6</sup>In my experiments, detail loss does not improve the performances. However, using auxiliary heads does contribute to the improvements]  

## SMP performance on Cityscapes

| Decoder       | Params (M) | mIoU (200 epoch) | mIoU (800 epoch)                                                                                               |
|:-------------:|:----------:|:----------------:|:--------------------------------------------------------------------------------------------------------------:|
| DeepLabv3     | 15.90      | 75.22            | [77.16](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/deeplabv3.pth)  |
| DeepLabv3Plus | 12.33      | 73.97            | [75.90](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/deeplabv3p.pth) |
| FPN           | 13.05      | 73.44            | [74.94](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/fpn.pth)        |
| LinkNet       | 11.66      | 71.17            | [73.19](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/linknet.pth)    |
| MANet         | 21.68      | 74.59            | [76.14](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/manet.pth)      |
| PAN           | 11.37      | 70.25            | [72.46](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/pan.pth)        |
| PSPNet        | 11.41      | 61.63            | [67.26](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/pspnet.pth)     |
| UNet          | 14.33      | 72.99            | [74.45](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/unet.pth)       |
| UNetPlusPlus  | 15.97      | 74.31            | [75.57](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/unetpp.pth)     |

[For comparison, the above results are all using ResNet-18 as encoders.]  

## Knowledge distillation

| Model | Encoder       | Decoder                 | kd_training | mIoU(200 epoch) | mIoU(800 epoch) |
|:-----:|:-------------:|:-----------------------:|:-----------:|:---------------:|:---------------:|
| SMP   | DeepLabv3Plus | ResNet-101 <br> teacher | -           | 78.10           | 79.20           |
| SMP   | DeepLabv3Plus | ResNet-18 <br> student  | False       | 73.97           | 75.90           |
| SMP   | DeepLabv3Plus | ResNet-18 <br> student  | True        | 75.20           | 76.41           |

# Prepare the dataset

```
/Cityscapes
    /gtFine
    /leftImg8bit
```

# References