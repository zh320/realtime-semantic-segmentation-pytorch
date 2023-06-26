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
- [BiSeNetv2](models/bisenetv2.py) [^bisenetv2]  
- [ContextNet](models/contextnet.py)[^contextnet]  
- [ENet](models/enet.py) [^enet]  
- [FastSCNN](models/fastscnn.py) [^fastscnn]  
- [LEDNet](models/lednet.py) [^lednet]  
- [LinkNet](models/linknet.py)[^linknet]  

If you want to use encoder-decoder structure with pretrained encoders, you may refer to: segmentation-models-pytorch[^smp]. This repo also provides easy access to SMP. Just modify the [config file](configs/my_config.py) to (e.g. if you want to train DeepLabv3Plus with ResNet-101 backbone as teacher model to perform knowledge distillation)  
```
self.model = 'smp'
self.encoder = 'resnet101'
self.decoder = 'deeplabv3p'
```


[^bisenetv2]: [BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](https://arxiv.org/abs/2004.02147)  
[^contextnet]: [ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time](https://arxiv.org/abs/1805.04554)  
[^enet]: [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147)  
[^fastscnn]: [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502)  
[^lednet]: [LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/1905.02423)  
[^linknet]: [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/abs/1707.03718)  
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
| Model | Params (M) <br> my / paper | mIoU (paper) <br> val / test| mIoU (200 epoch) | mIoU (800 epoch) |
| :---: | :---: | :---: | :---: | :---: |
| BiSeNetv2 | 2.53 / - | 73.4 / 72.6 | [64.41](weights/bisenetv2_200epoch.pth) / [69.36*](weights/bisenetv2-aux_200epoch.pth) | [68.68](weights/bisenetv2_800epoch.pth) / [72.15*](weights/bisenetv2-aux_800epoch.pth) |
| ContextNet | 1.01 / 0.85 | 65.9 / 66.1 | [62.17](weights/contextnet_200epoch.pth) | [66.15](weights/contextnet_800epoch.pth) |
| ENet | 0.37 / 0.37 | - / 58.3 | [62.03](weights/enet_200epoch.pth) | [69.65](weights/enet_800epoch.pth) |
| FastSCNN | 1.02 / 1.11 | 68.6 / 68.0 | [61.31](weights/fastscnn_200epoch.pth) | [66.75](weights/fastscnn_800epoch.pth) |
| LEDNet | 1.46 / 0.94 | - / 70.6 | [65.91](weights/lednet_200epoch.pth) | [71.76](weights/lednet_800epoch.pth) |
| LinkNet | 11.71 / 11.5 | - / 76.4| [63.82](weights/linknet_200epoch.pth) | [70.86](weights/linknet_800epoch.pth) |

[*These results are obtained by using auxiliary heads]  


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


# Prepare the dataset
```
/Cityscapes
    /gtFine
    /leftImg8bit
```



# References