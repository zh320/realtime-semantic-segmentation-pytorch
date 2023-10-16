# Introduction
PyTorch Implementation of realtime semantic segmentation models, support multi-gpu training and validating, automatic mixed precision training, knowledge distillation etc.  
\
<img src="demo/enet_800epoch.gif" width="100%" height="100%" />


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
- [EDANet](models/edanet.py) [^edanet]  
- [ENet](models/enet.py) [^enet]  
- [ERFNet](models/erfnet.py) [^erfnet]  
- [ESNet](models/esnet.py) [^esnet]  
- [ESPNet](models/espnet.py) [^espnet]  
- [ESPNetv2](models/espnetv2.py) [^espnetv2]  
- [FarseeNet](models/farseenet.py) [^farseenet]  
- [FastSCNN](models/fastscnn.py) [^fastscnn]  
- [FPENet](models/fpenet.py) [^fpenet]  
- [LEDNet](models/lednet.py) [^lednet]  
- [LinkNet](models/linknet.py)[^linknet]  
- [PP-LiteSeg](models/pp_liteseg.py)[^ppliteseg]  
- [SegNet](models/segnet.py)[^segnet]  

If you want to use encoder-decoder structure with pretrained encoders, you may refer to: segmentation-models-pytorch[^smp]. This repo also provides easy access to SMP. Just modify the [config file](configs/my_config.py) to (e.g. if you want to train DeepLabv3Plus with ResNet-101 backbone as teacher model to perform knowledge distillation)  
```
self.model = 'smp'
self.encoder = 'resnet101'
self.decoder = 'deeplabv3p'
```


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
[^edanet]: [Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation](https://arxiv.org/abs/1809.06323)  
[^enet]: [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147)  
[^erfnet]: [ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation](https://ieeexplore.ieee.org/document/8063438)  
[^esnet]: [ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1906.09826)  
[^espnet]: [ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation](https://arxiv.org/abs/1803.06815)  
[^espnetv2]: [ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network](https://arxiv.org/abs/1811.11431)  
[^farseenet]: [FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale Context Aggregation and Feature Space Super-resolution](https://arxiv.org/abs/2003.03913)  
[^fastscnn]: [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502)  
[^fpenet]: [Feature Pyramid Encoding Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1909.08599)  
[^lednet]: [LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/1905.02423)  
[^linknet]: [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/abs/1707.03718)  
[^ppliteseg]: [PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model](https://arxiv.org/abs/2204.02681)  
[^segnet]: [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)  
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
| Model | Params (M) <br> paper / my | mIoU (paper) <br> val / test| mIoU (my) val*|
| :---: | :---: | :---: | :---: |
| ADSCNet | n.a. / 0.51 | n.a. / 67.5 | [69.06](weights/adscnet_crop-1024_800epoch.pth) |
| AGLNet | 1.12 / 1.02 | 69.39 / 70.1 | [73.58](weights/aglnet_crop-1024_800epoch.pth) |
| BiSeNetv1 | 49.0 / 13.32 | 74.8 / 74.7 | [74.91](weights/bisenetv1_crop-1024_800epoch.pth) |
| BiSeNetv2 | n.a. / 2.53 | 73.4 / 72.6 | [73.73**](weights/bisenetv2-aux_crop-1024_800epoch.pth) |
| CANet | 4.8 / 4.77 | 73.4 / 73.5 | [73.76](weights/canet_crop-1024_800epoch.pth) |
| CFPNet | 0.55 / 0.27 | n.a. / 70.1 | [70.08](weights/cfpnet_crop-1024_800epoch.pth) |
| CGNet | 0.41 / 0.24 | 59.7 / 64.8*** | [67.25](weights/cgnet_crop-1024_800epoch.pth) |
| ContextNet | 0.85 / 1.01 | 65.9 / 66.1 | [66.61](weights/contextnet_crop-1024_800epoch.pth) |
| DABNet | 0.76 / 0.75 | n.a. / 70.1 | [70.78](weights/dabnet_crop-1024_800epoch.pth) |
| DDRNet | 5.7 / 5.54 | 77.8 / 77.4 | [74.34](weights/ddrnet-23-slim_crop-1024_800epoch.pth) |
| EDANet | 0.68 / 0.69 | n.a. / 67.3 | [70.76](weights/edanet_crop-1024_800epoch.pth) |
| ENet | 0.37 / 0.37 | n.a. / 58.3 | [71.31](weights/enet_crop-1024_800epoch.pth) |
| ERFNet | 2.06 / 2.07 | 70.0 / 68.0 | [76.00](weights/erfnet_crop-1024_800epoch.pth) |
| ESNet | 1.66 / 1.66 | n.a. / 70.7 | [71.82](weights/esnet_crop-1024_800epoch.pth) |
| ESPNet | 0.36 / 0.38 | n.a. / 60.3 | [66.39](weights/espnet_crop-1024_800epoch.pth) |
| ESPNetv2 | 1.25 / 0.86 | 66.4 / 66.2 | [70.35](weights/espnetv2_crop-1024_800epoch.pth) |
| FarseeNet | n.a. / 16.75 | 73.5 / 70.2 | [77.35](weights/farseenet_crop-1024_800epoch.pth) |
| FastSCNN | 1.11 / 1.02 | 68.6 / 68.0 | [69.37](weights/fastscnn_crop-1024_800epoch.pth) |
| FPENet | 0.38 / 0.36 | n.a. / 70.1 | [72.05](weights/fpenet_crop-1024_800epoch.pth) |
| LEDNet | 0.94 / 1.46 | n.a. / 70.6 | [71.40](weights/lednet_crop-1024_800epoch.pth) |
| LinkNet | 11.5 / 11.71 | n.a. / 76.4| [71.72](weights/linknet_crop-1024_800epoch.pth) |
| PP-LiteSeg <br> Enc: STDC1 | n.a. / 6.33 | 76.0 / 74.9| [72.49](weights/ppliteseg_stdc1_crop-1024_800epoch.pth) |
| PP-LiteSeg <br> Enc: STDC2 | n.a. / 10.56 | 78.2 / 77.5| [74.37](weights/ppliteseg_stdc2_crop-1024_800epoch.pth) |
| SegNet | 29.46 / 29.48 | n.a. / 56.1| [70.77](weights/segnet_crop-1024_800epoch.zip.001) |

[*These results are obtained by training 800 epochs with crop-size 1024x1024]  
[**These results are obtained by using auxiliary heads]  
[***This result is obtained by using deeper model, i.e. CGNet_M3N21]  


## SMP performance on Cityscapes  
| Decoder | Params (M) | mIoU (200 epoch) | mIoU (800 epoch) |
| :---: | :---: | :---: | :---: |
| DeepLabv3 | 15.90 | 75.22 | 77.16 |
| DeepLabv3Plus | 12.33 | 73.97 | 75.90 |
| FPN | 13.05 | 73.44 | 74.94 |
| LinkNet | 11.66 | 71.17 | 73.19 |
| MANet | 21.68 | 74.59 | 76.14 |
| PAN | 11.37 | 70.25 | 72.46 |
| PSPNet | 11.41 | 61.63 | 67.26 |
| UNet | 14.33 | 72.99 | 74.45 |
| UNetPlusPlus | 15.97 | 74.31 | 75.57 |

[For comparison, the above results are all using ResNet-18 as encoders.]  


## Knowledge distillation  
| Model | Encoder | Decoder | kd_training | mIoU (200 epoch) | mIoU (800 epoch) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| SMP | DeepLabv3Plus | ResNet-101 <br> teacher | - | 78.10 | 79.20 |
| SMP | DeepLabv3Plus | ResNet-18 <br> student | False | 73.97 | 75.90 |
| SMP | DeepLabv3Plus | ResNet-18 <br> student | True | 75.20 | 76.41 |


# Prepare the dataset
```
/Cityscapes
    /gtFine
    /leftImg8bit
```



# References