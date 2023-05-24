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
If you want to use encoder-decoder structure with pretrained encoders, you may refer to this repo: segmentation-models-pytorch[^smp]

[BiSeNetv2](https://github.com/zh320/realtime-semantic-segmentation-pytorch/blob/main/models/bisenetv2.py) [^bisenetv2]  
[ENet](https://github.com/zh320/realtime-semantic-segmentation-pytorch/blob/main/models/enet.py) [^enet]  
[FastSCNN](https://github.com/zh320/realtime-semantic-segmentation-pytorch/blob/main/models/fastscnn.py) [^fastscnn]  
[LEDNet](https://github.com/zh320/realtime-semantic-segmentation-pytorch/blob/main/models/lednet.py) [^lednet]  

More models and benchmarks are coming.


[^bisenetv2]: [BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](https://arxiv.org/abs/2004.02147)  
[^enet]: [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147)  
[^fastscnn]: [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502)  
[^lednet]: [LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/1905.02423)  
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
(full resolution on Cityscapes)
| Model | Params | mIoU (200 epoch) | mIoU (800 epoch) |
| :---: | :---: | :---: | :---: |
| BiSeNetv2 | - | [0.6441](weights/bisenetv2_200epoch.pth) | [0.6868](weights/bisenetv2_800epoch.pth) |
| ENet | - | [0.6203](weights/enet_200epoch.pth) | [0.6965](weights/enet_800epoch.pth) |
| FastSCNN | - | [0.6131](weights/fastscnn_200epoch.pth) | [0.6675](weights/fastscnn_800epoch.pth) |
| LEDNet | - | [0.6591](weights/lednet_200epoch.pth) | [0.7176](weights/lednet_800epoch.pth) |


# Prepare the dataset
```
/Cityscapes
    /gtFine
    /leftImg8bit
```


# References