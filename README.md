# Introduction

PyTorch implementation of realtime semantic segmentation models, support multi-gpu training and validating, automatic mixed precision training, knowledge distillation, hyperparameter optimization using Optuna etc.  
\
<img2 src="https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.0/enet_800epoch.gif" width="100%" height="100%" />

# Requirements

torch == 1.8.1  
segmentation-models-pytorch  
torchmetrics  
albumentations  
loguru  
tqdm  
optuna == 4.0.0 (optional)  
optuna-integration == 4.0.0 (optional)  

If you find any version conflicts, see [requirements](./requirements.txt). This repo may also work with torch > 1.8.1, but it has not been verified yet.  

If you want a minimally reproducible environment, you may run
```
pip install -r requirements.txt
```

# Supported models

<details><summary>ADSCNet</summary>

[ADSCNet: asymmetric depthwise separable convolution for semantic segmentation in real-time](https://link.springer.com/article/10.1007/s10489-019-01587-1) [[codes](models/adscnet.py)]  

> Abstract: Semantic segmentation can be considered as a per-pixel localization and classification problem, which gives a meaningful label to each pixel in an input image. Deep convolutional neural networks have made extremely successful in semantic segmentation in recent years. However, some challenges still exist. The first challenge task is that most current networks are complex and it is hard to deploy these models on mobile devices because of the limitation of computational cost and memory. Getting more contextual information from downsampled feature maps is another challenging task. To this end, we propose an asymmetric depthwise separable convolution network (ADSCNet) which is a lightweight neural network for real-time semantic segmentation. To facilitating information propagation, Dense Dilated Convolution Connections (DDCC), which connects a set of dilated convolutional layers in a dense way, is introduced in the network. Pooling operation is inserted before ADSCNet unit to cover more contextual information in prediction. Extensive experimental results validate the superior performance of our proposed method compared with other network architectures. Our approach achieves mean intersection over union (mIOU) of 67.5% on Cityscapes dataset at 76.9 frames per second.  
</details>


<details><summary>AGLNet</summary>

[AGLNet: Towards real-time semantic segmentation of self-driving images via attention-guided lightweight network](https://www.sciencedirect.com/science/article/abs/pii/S1568494620306207) [[codes](models/aglnet.py)]  

> Abstract: The extensive computational burden limits the usage of convolutional neural networks (CNNs) in edge devices for image semantic segmentation, which plays a significant role in many real-world applications, such as augmented reality, robotics, and self-driving. To address this problem, this paper presents an attention-guided lightweight network, namely AGLNet, which employs an encoder–decoder architecture for real-time semantic segmentation. Specifically, the encoder adopts a novel residual module to abstract feature representations, where two new operations, channel split and shuffle, are utilized to greatly reduce computation cost while maintaining higher segmentation accuracy. On the other hand, instead of using complicated dilated convolution and artificially designed architecture, two types of attention mechanism are subsequently employed in the decoder to upsample features to match input resolution. Specifically, a factorized attention pyramid module (FAPM) is used to explore hierarchical spatial attention from high-level output, still remaining fewer model parameters. To delineate object shapes and boundaries, a global attention upsample module (GAUM) is adopted as global guidance for high-level features. The comprehensive experiments demonstrate that our approach achieves state-of-the-art results in terms of speed and accuracy on three self-driving datasets: CityScapes, CamVid, and Mapillary Vistas. AGLNet achieves 71.3%, 69.4%, and 30.7% mean IoU on these datasets with only 1.12M model parameters. Our method also achieves 52 FPS, 90 FPS, and 53 FPS inference speed, respectively, using a single GTX 1080Ti GPU. Our code is open-source and available at https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks.

</details>


<details><summary>BiSeNetv1</summary>

[BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1808.00897) [[codes](models/bisenetv1.py)]  

> Abstract: Semantic segmentation requires both rich spatial information and sizeable receptive field. However, modern approaches usually compromise spatial resolution to achieve real-time inference speed, which leads to poor performance. In this paper, we address this dilemma with a novel Bilateral Segmentation Network (BiSeNet). We first design a Spatial Path with a small stride to preserve the spatial information and generate high-resolution features. Meanwhile, a Context Path with a fast downsampling strategy is employed to obtain sufficient receptive field. On top of the two paths, we introduce a new Feature Fusion Module to combine features efficiently. The proposed architecture makes a right balance between the speed and segmentation performance on Cityscapes, CamVid, and COCO-Stuff datasets. Specifically, for a 2048x1024 input, we achieve 68.4% Mean IOU on the Cityscapes test dataset with speed of 105 FPS on one NVIDIA Titan XP card, which is significantly faster than the existing methods with comparable performance.

</details>


<details><summary>BiSeNetv2</summary>

[BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](https://arxiv.org/abs/2004.02147) [[codes](models/bisenetv2.py)]  

> Abstract: The low-level details and high-level semantics are both essential to the semantic segmentation task. However, to speed up the model inference, current approaches almost always sacrifice the low-level details, which leads to a considerable accuracy decrease. We propose to treat these spatial details and categorical semantics separately to achieve high accuracy and high efficiency for realtime semantic segmentation. To this end, we propose an efficient and effective architecture with a good trade-off between speed and accuracy, termed Bilateral Segmentation Network (BiSeNet V2). This architecture involves: (i) a Detail Branch, with wide channels and shallow layers to capture low-level details and generate high-resolution feature representation; (ii) a Semantic Branch, with narrow channels and deep layers to obtain high-level semantic context. The Semantic Branch is lightweight due to reducing the channel capacity and a fast-downsampling strategy. Furthermore, we design a Guided Aggregation Layer to enhance mutual connections and fuse both types of feature representation. Besides, a booster training strategy is designed to improve the segmentation performance without any extra inference cost. Extensive quantitative and qualitative evaluations demonstrate that the proposed architecture performs favourably against a few state-of-the-art real-time semantic segmentation approaches. Specifically, for a 2,048x1,024 input, we achieve 72.6% Mean IoU on the Cityscapes test set with a speed of 156 FPS on one NVIDIA GeForce GTX 1080 Ti card, which is significantly faster than existing methods, yet we achieve better segmentation accuracy.

</details>


<details><summary>CANet</summary>

[Cross Attention Network for Semantic Segmentation](https://arxiv.org/abs/1907.10958) [[codes](models/canet.py)]  

> Abstract: In this paper, we address the semantic segmentation task with a deep network that combines contextual features and spatial information. The proposed Cross Attention Network is composed of two branches and a Feature Cross Attention (FCA) module. Specifically, a shallow branch is used to preserve low-level spatial information and a deep branch is employed to extract high-level contextual features. Then the FCA module is introduced to combine these two branches. Different from most existing attention mechanisms, the FCA module obtains spatial attention map and channel attention map from two branches separately, and then fuses them. The contextual features are used to provide global contextual guidance in fused feature maps, and spatial features are used to refine localizations. The proposed network outperforms other real-time methods with improved speed on the Cityscapes and CamVid datasets with lightweight backbones, and achieves state-of-the-art performance with a deep backbone.

</details>


<details><summary>CFPNet</summary>

[CFPNet: Channel-wise Feature Pyramid for Real-Time Semantic Segmentation](https://arxiv.org/abs/2103.12212) [[codes](models/cfpnet.py)]  

> Abstract: Real-time semantic segmentation is playing a more important role in computer vision, due to the growing demand for mobile devices and autonomous driving. Therefore, it is very important to achieve a good trade-off among performance, model size and inference speed. In this paper, we propose a Channel-wise Feature Pyramid (CFP) module to balance those factors. Based on the CFP module, we built CFPNet for real-time semantic segmentation which applied a series of dilated convolution channels to extract effective features. Experiments on Cityscapes and CamVid datasets show that the proposed CFPNet achieves an effective combination of those factors. For the Cityscapes test dataset, CFPNet achieves 70.1% class-wise mIoU with only 0.55 million parameters and 2.5 MB memory. The inference speed can reach 30 FPS on a single RTX 2080Ti GPU with a 1024x2048-pixel image.

</details>


<details><summary>CGNet</summary>

[CGNet: A Light-weight Context Guided Network for Semantic Segmentation](https://arxiv.org/abs/1811.08201) [[codes](models/cgnet.py)]  

> Abstract: The demand of applying semantic segmentation model on mobile devices has been increasing rapidly. Current state-of-the-art networks have enormous amount of parameters hence unsuitable for mobile devices, while other small memory footprint models follow the spirit of classification network and ignore the inherent characteristic of semantic segmentation. To tackle this problem, we propose a novel Context Guided Network (CGNet), which is a light-weight and efficient network for semantic segmentation. We first propose the Context Guided (CG) block, which learns the joint feature of both local feature and surrounding context, and further improves the joint feature with the global context. Based on the CG block, we develop CGNet which captures contextual information in all stages of the network and is specially tailored for increasing segmentation accuracy. CGNet is also elaborately designed to reduce the number of parameters and save memory footprint. Under an equivalent number of parameters, the proposed CGNet significantly outperforms existing segmentation networks. Extensive experiments on Cityscapes and CamVid datasets verify the effectiveness of the proposed approach. Specifically, without any post-processing and multi-scale testing, the proposed CGNet achieves 64.8% mean IoU on Cityscapes with less than 0.5 M parameters. The source code for the complete system can be found at [this https URL](https://github.com/wutianyiRosun/CGNet).

</details>


<details><summary>ContextNet</summary>

[ContextNet: Exploring Context and Detail for Semantic Segmentation in Real-time](https://arxiv.org/abs/1805.04554) [[codes](models/contextnet.py)]  

> Abstract: Modern deep learning architectures produce highly accurate results on many challenging semantic segmentation datasets. State-of-the-art methods are, however, not directly transferable to real-time applications or embedded devices, since naive adaptation of such systems to reduce computational cost (speed, memory and energy) causes a significant drop in accuracy. We propose ContextNet, a new deep neural network architecture which builds on factorized convolution, network compression and pyramid representation to produce competitive semantic segmentation in real-time with low memory requirement. ContextNet combines a deep network branch at low resolution that captures global context information efficiently with a shallow branch that focuses on high-resolution segmentation details. We analyse our network in a thorough ablation study and present results on the Cityscapes dataset, achieving 66.1% accuracy at 18.3 frames per second at full (1024x2048) resolution (41.9 fps with pipelined computations for streamed data).

</details>


<details><summary>DABNet</summary>

[DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation](https://arxiv.org/abs/1907.11357) [[codes](models/dabnet.py)]  

> Abstract: As a pixel-level prediction task, semantic segmentation needs large computational cost with enormous parameters to obtain high performance. Recently, due to the increasing demand for autonomous systems and robots, it is significant to make a tradeoff between accuracy and inference speed. In this paper, we propose a novel Depthwise Asymmetric Bottleneck (DAB) module to address this dilemma, which efficiently adopts depth-wise asymmetric convolution and dilated convolution to build a bottleneck structure. Based on the DAB module, we design a Depth-wise Asymmetric Bottleneck Network (DABNet) especially for real-time semantic segmentation, which creates sufficient receptive field and densely utilizes the contextual information. Experiments on Cityscapes and CamVid datasets demonstrate that the proposed DABNet achieves a balance between speed and precision. Specifically, without any pretrained model and postprocessing, it achieves 70.1% Mean IoU on the Cityscapes test dataset with only 0.76 million parameters and a speed of 104 FPS on a single GTX 1080Ti card.

</details>


<details><summary>DDRNet</summary>

[Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes](https://arxiv.org/abs/2101.06085) [[codes](models/ddrnet.py)]  

> Abstract: Semantic segmentation is a key technology for autonomous vehicles to understand the surrounding scenes. The appealing performances of contemporary models usually come at the expense of heavy computations and lengthy inference time, which is intolerable for self-driving. Using light-weight architectures (encoder-decoder or two-pathway) or reasoning on low-resolution images, recent methods realize very fast scene parsing, even running at more than 100 FPS on a single 1080Ti GPU. However, there is still a significant gap in performance between these real-time methods and the models based on dilation backbones. To tackle this problem, we proposed a family of efficient backbones specially designed for real-time semantic segmentation. The proposed deep dual-resolution networks (DDRNets) are composed of two deep branches between which multiple bilateral fusions are performed. Additionally, we design a new contextual information extractor named Deep Aggregation Pyramid Pooling Module (DAPPM) to enlarge effective receptive fields and fuse multi-scale context based on low-resolution feature maps. Our method achieves a new state-of-the-art trade-off between accuracy and speed on both Cityscapes and CamVid dataset. In particular, on a single 2080Ti GPU, DDRNet-23-slim yields 77.4% mIoU at 102 FPS on Cityscapes test set and 74.7% mIoU at 230 FPS on CamVid test set. With widely used test augmentation, our method is superior to most state-of-the-art models and requires much less computation. Codes and trained models are available online.

</details>


<details><summary>DFANet</summary>

[DFANet: Deep Feature Aggregation for Real-Time Semantic Segmentation](https://arxiv.org/abs/1904.02216) [[codes](models/dfanet.py)]  

> Abstract: This paper introduces an extremely efficient CNN architecture named DFANet for semantic segmentation under resource constraints. Our proposed network starts from a single lightweight backbone and aggregates discriminative features through sub-network and sub-stage cascade respectively. Based on the multi-scale feature propagation, DFANet substantially reduces the number of parameters, but still obtains sufficient receptive field and enhances the model learning ability, which strikes a balance between the speed and segmentation performance. Experiments on Cityscapes and CamVid datasets demonstrate the superior performance of DFANet with 8× less FLOPs and 2× faster than the existing state-of-the-art real-time semantic segmentation methods while providing comparable accuracy. Specifically, it achieves 70.3\% Mean IOU on the Cityscapes test dataset with only 1.7 GFLOPs and a speed of 160 FPS on one NVIDIA Titan X card, and 71.3\% Mean IOU with 3.4 GFLOPs while inferring on a higher resolution image.

</details>


<details><summary>EDANet</summary>

[Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation](https://arxiv.org/abs/1809.06323) [[codes](models/edanet.py)]  

> Abstract: Real-time semantic segmentation plays an important role in practical applications such as self-driving and robots. Most semantic segmentation research focuses on improving estimation accuracy with little consideration on efficiency. Several previous studies that emphasize high-speed inference often fail to produce high-accuracy segmentation results. In this paper, we propose a novel convolutional network named Efficient Dense modules with Asymmetric convolution (EDANet), which employs an asymmetric convolution structure and incorporates dilated convolution and dense connectivity to achieve high efficiency at low computational cost and model size. EDANet is 2.7 times faster than the existing fast segmentation network, ICNet, while it achieves a similar mIoU score without any additional context module, post-processing scheme, and pretrained model. We evaluate EDANet on Cityscapes and CamVid datasets, and compare it with the other state-of-art systems. Our network can run with the high-resolution inputs at the speed of 108 FPS on one GTX 1080Ti.

</details>


<details><summary>ENet</summary>

[ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/abs/1606.02147) [[codes](models/enet.py)]  

> Abstract: The ability to perform pixel-wise semantic segmentation in real-time is of paramount importance in mobile applications. Recent deep neural networks aimed at this task have the disadvantage of requiring a large number of floating point operations and have long run-times that hinder their usability. In this paper, we propose a novel deep neural network architecture named ENet (efficient neural network), created specifically for tasks requiring low latency operation. ENet is up to 18× faster, requires 75× less FLOPs, has 79× less parameters, and provides similar or better accuracy to existing models. We have tested it on CamVid, Cityscapes and SUN datasets and report on comparisons with existing state-of-the-art methods, and the trade-offs between accuracy and processing time of a network. We present performance measurements of the proposed architecture on embedded systems and suggest possible software improvements that could make ENet even faster.

</details>


<details><summary>ERFNet</summary>

[ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation](https://ieeexplore.ieee.org/document/8063438) [[codes](models/erfnet.py)]  

> Abstract: Semantic segmentation is a challenging task that addresses most of the perception needs of intelligent vehicles (IVs) in an unified way. Deep neural networks excel at this task, as they can be trained end-to-end to accurately classify multiple object categories in an image at pixel level. However, a good tradeoff between high quality and computational resources is yet not present in the state-of-the-art semantic segmentation approaches, limiting their application in real vehicles. In this paper, we propose a deep architecture that is able to run in real time while providing accurate semantic segmentation. The core of our architecture is a novel layer that uses residual connections and factorized convolutions in order to remain efficient while retaining remarkable accuracy. Our approach is able to run at over 83 FPS in a single Titan X, and 7 FPS in a Jetson TX1 (embedded device). A comprehensive set of experiments on the publicly available Cityscapes data set demonstrates that our system achieves an accuracy that is similar to the state of the art, while being orders of magnitude faster to compute than other architectures that achieve top precision. The resulting tradeoff makes our model an ideal approach for scene understanding in IV applications. The code is publicly available at: https://github.com/Eromera/erfnet.

</details>


<details><summary>ESNet</summary>

[ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1906.09826) [[codes](models/esnet.py)]  

> Abstract: The recent years have witnessed great advances for semantic segmentation using deep convolutional neural networks (DCNNs). However, a large number of convolutional layers and feature channels lead to semantic segmentation as a computationally heavy task, which is disadvantage to the scenario with limited resources. In this paper, we design an efficient symmetric network, called (ESNet), to address this problem. The whole network has nearly symmetric architecture, which is mainly composed of a series of factorized convolution unit (FCU) and its parallel counterparts (PFCU). On one hand, the FCU adopts a widely-used 1D factorized convolution in residual layers. On the other hand, the parallel version employs a transform-split-transform-merge strategy in the designment of residual module, where the split branch adopts dilated convolutions with different rate to enlarge receptive field. Our model has nearly 1.6M parameters, and is able to be performed over 62 FPS on a single GTX 1080Ti GPU. The experiments demonstrate that our approach achieves state-of-the-art results in terms of speed and accuracy trade-off for real-time semantic segmentation on CityScapes dataset.

</details>


<details><summary>ESPNet</summary>

[ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation](https://arxiv.org/abs/1803.06815) [[codes](models/espnet.py)]  

> Abstract: We introduce a fast and efficient convolutional neural network, ESPNet, for semantic segmentation of high resolution images under resource constraints. ESPNet is based on a new convolutional module, efficient spatial pyramid (ESP), which is efficient in terms of computation, memory, and power. ESPNet is 22 times faster (on a standard GPU) and 180 times smaller than the state-of-the-art semantic segmentation network PSPNet, while its category-wise accuracy is only 8% less. We evaluated ESPNet on a variety of semantic segmentation datasets including Cityscapes, PASCAL VOC, and a breast biopsy whole slide image dataset. Under the same constraints on memory and computation, ESPNet outperforms all the current efficient CNN networks such as MobileNet, ShuffleNet, and ENet on both standard metrics and our newly introduced performance metrics that measure efficiency on edge devices. Our network can process high resolution images at a rate of 112 and 9 frames per second on a standard GPU and edge device, respectively.

</details>


<details><summary>ESPNetv2</summary>

[ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network](https://arxiv.org/abs/1811.11431) [[codes](models/espnetv2.py)]  

> Abstract: We introduce a light-weight, power efficient, and general purpose convolutional neural network, ESPNetv2, for modeling visual and sequential data. Our network uses group point-wise and depth-wise dilated separable convolutions to learn representations from a large effective receptive field with fewer FLOPs and parameters. The performance of our network is evaluated on four different tasks: (1) object classification, (2) semantic segmentation, (3) object detection, and (4) language modeling. Experiments on these tasks, including image classification on the ImageNet and language modeling on the PenTree bank dataset, demonstrate the superior performance of our method over the state-of-the-art methods. Our network outperforms ESPNet by 4-5% and has 2-4x fewer FLOPs on the PASCAL VOC and the Cityscapes dataset. Compared to YOLOv2 on the MS-COCO object detection, ESPNetv2 delivers 4.4% higher accuracy with 6x fewer FLOPs. Our experiments show that ESPNetv2 is much more power efficient than existing state-of-the-art efficient methods including ShuffleNets and MobileNets. Our code is open-source and available at [this https URL](https://github.com/sacmehta/ESPNetv2)

</details>


<details><summary>FANet</summary>

[Real-time Semantic Segmentation with Fast Attention](https://arxiv.org/abs/2007.03815) [[codes](models/fanet.py)]  

> Abstract: In deep CNN based models for semantic segmentation, high accuracy relies on rich spatial context (large receptive fields) and fine spatial details (high resolution), both of which incur high computational costs. In this paper, we propose a novel architecture that addresses both challenges and achieves state-of-the-art performance for semantic segmentation of high-resolution images and videos in real-time. The proposed architecture relies on our fast spatial attention, which is a simple yet efficient modification of the popular self-attention mechanism and captures the same rich spatial context at a small fraction of the computational cost, by changing the order of operations. Moreover, to efficiently process high-resolution input, we apply an additional spatial reduction to intermediate feature stages of the network with minimal loss in accuracy thanks to the use of the fast attention module to fuse features. We validate our method with a series of experiments, and show that results on multiple datasets demonstrate superior performance with better accuracy and speed compared to existing approaches for real-time semantic segmentation. On Cityscapes, our network achieves 74.4% mIoU at 72 FPS and 75.5% mIoU at 58 FPS on a single Titan X GPU, which is~∼50% faster than the state-of-the-art while retaining the same accuracy.

</details>


<details><summary>FarseeNet</summary>

[FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale Context Aggregation and Feature Space Super-resolution](https://arxiv.org/abs/2003.03913) [[codes](models/farseenet.py)]  

> Abstract: Real-time semantic segmentation is desirable in many robotic applications with limited computation resources. One challenge of semantic segmentation is to deal with the object scale variations and leverage the context. How to perform multi-scale context aggregation within limited computation budget is important. In this paper, firstly, we introduce a novel and efficient module called Cascaded Factorized Atrous Spatial Pyramid Pooling (CF-ASPP). It is a lightweight cascaded structure for Convolutional Neural Networks (CNNs) to efficiently leverage context information. On the other hand, for runtime efficiency, state-of-the-art methods will quickly decrease the spatial size of the inputs or feature maps in the early network stages. The final high-resolution result is usually obtained by non-parametric up-sampling operation (e.g. bilinear interpolation). Differently, we rethink this pipeline and treat it as a super-resolution process. We use optimized super-resolution operation in the up-sampling step and improve the accuracy, especially in sub-sampled input image scenario for real-time applications. By fusing the above two improvements, our methods provide better latency-accuracy trade-off than the other state-of-the-art methods. In particular, we achieve 68.4% mIoU at 84 fps on the Cityscapes test set with a single Nivida Titan X (Maxwell) GPU card. The proposed module can be plugged into any feature extraction CNN and benefits from the CNN structure development.

</details>


<details><summary>FastSCNN</summary>

[Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502) [[codes](models/fastscnn.py)]  

> Abstract: The encoder-decoder framework is state-of-the-art for offline semantic image segmentation. Since the rise in autonomous systems, real-time computation is increasingly desirable. In this paper, we introduce fast segmentation convolutional neural network (Fast-SCNN), an above real-time semantic segmentation model on high resolution image data (1024x2048px) suited to efficient computation on embedded devices with low memory. Building on existing two-branch methods for fast segmentation, we introduce our `learning to downsample' module which computes low-level features for multiple resolution branches simultaneously. Our network combines spatial detail at high resolution with deep features extracted at lower resolution, yielding an accuracy of 68.0% mean intersection over union at 123.5 frames per second on Cityscapes. We also show that large scale pre-training is unnecessary. We thoroughly validate our metric in experiments with ImageNet pre-training and the coarse labeled data of Cityscapes. Finally, we show even faster computation with competitive results on subsampled inputs, without any network modifications.

</details>


<details><summary>FDDWNet</summary>

[FDDWNet: A Lightweight Convolutional Neural Network for Real-time Sementic Segmentation](https://arxiv.org/abs/1911.00632) [[codes](models/fddwnet.py)]  

> Abstract: This paper introduces a lightweight convolutional neural network, called FDDWNet, for real-time accurate semantic segmentation. In contrast to recent advances of lightweight networks that prefer to utilize shallow structure, FDDWNet makes an effort to design more deeper network architecture, while maintains faster inference speed and higher segmentation accuracy. Our network uses factorized dilated depth-wise separable convolutions (FDDWC) to learn feature representations from different scale receptive fields with fewer model parameters. Additionally, FDDWNet has multiple branches of skipped connections to gather context cues from intermediate convolution layers. The experiments show that FDDWNet only has 0.8M model size, while achieves 60 FPS running speed on a single RTX 2080Ti GPU with a 1024x512 input image. The comprehensive experiments demonstrate that our model achieves state-of-the-art results in terms of available speed and accuracy trade-off on CityScapes and CamVid datasets.

</details>


<details><summary>FPENet</summary>

[Feature Pyramid Encoding Network for Real-time Semantic Segmentation](https://arxiv.org/abs/1909.08599) [[codes](models/fpenet.py)]  

> Abstract: Although current deep learning methods have achieved impressive results for semantic segmentation, they incur high computational costs and have a huge number of parameters. For real-time applications, inference speed and memory usage are two important factors. To address the challenge, we propose a lightweight feature pyramid encoding network (FPENet) to make a good trade-off between accuracy and speed. Specifically, we use a feature pyramid encoding block to encode multi-scale contextual features with depthwise dilated convolutions in all stages of the encoder. A mutual embedding upsample module is introduced in the decoder to aggregate the high-level semantic features and low-level spatial details efficiently. The proposed network outperforms existing real-time methods with fewer parameters and improved inference speed on the Cityscapes and CamVid benchmark datasets. Specifically, FPENet achieves 68.0\% mean IoU on the Cityscapes test set with only 0.4M parameters and 102 FPS speed on an NVIDIA TITAN V GPU.

</details>


<details><summary>FSSNet</summary>

[Fast Semantic Segmentation for Scene Perception](https://ieeexplore.ieee.org/document/8392426) [[codes](models/fssnet.py)]  

> Abstract: Semantic segmentation is a challenging problem in computer vision. Many applications, such as autonomous driving and robot navigation with urban road scene, need accurate and efficient segmentation. Most state-of-the-art methods focus on accuracy, rather than efficiency. In this paper, we propose a more efficient neural network architecture, which has fewer parameters, for semantic segmentation in the urban road scene. An asymmetric encoder-decoder structure based on ResNet is used in our model. In the first stage of encoder, we use continuous factorized block to extract low-level features. Continuous dilated block is applied in the second stage, which ensures that the model has a larger view field, while keeping the model small-scale and shallow. The down sampled features from encoder are up sampled with decoder to the same-size output as the input image and the details refined. Our model can achieve end-to-end and pixel-to-pixel training without pretraining from scratch. The parameters of our model are only 0.2M, 100× less than those of others such as SegNet, etc. Experiments are conducted on five public road scene datasets (CamVid, CityScapes, Gatech, KITTI Road Detection, and KITTI Semantic Segmentation), and the results demonstrate that our model can achieve better performance.

</details>


<details><summary>ICNet</summary>

[ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545) [[codes](models/icnet.py)]  

> Abstract: We focus on the challenging task of real-time semantic segmentation in this paper. It finds many practical applications and yet is with fundamental difficulty of reducing a large portion of computation for pixel-wise label inference. We propose an image cascade network (ICNet) that incorporates multi-resolution branches under proper label guidance to address this challenge. We provide in-depth analysis of our framework and introduce the cascade feature fusion unit to quickly achieve high-quality segmentation. Our system yields real-time inference on a single GPU card with decent quality results evaluated on challenging datasets like Cityscapes, CamVid and COCO-Stuff.

</details>


<details><summary>LEDNet</summary>

[LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation](https://arxiv.org/abs/1905.02423) [[codes](models/lednet.py)]  

> Abstract: The extensive computational burden limits the usage of CNNs in mobile devices for dense estimation tasks. In this paper, we present a lightweight network to address this problem,namely LEDNet, which employs an asymmetric encoder-decoder architecture for the task of real-time semantic segmentation.More specifically, the encoder adopts a ResNet as backbone network, where two new operations, channel split and shuffle, are utilized in each residual block to greatly reduce computation cost while maintaining higher segmentation accuracy. On the other hand, an attention pyramid network (APN) is employed in the decoder to further lighten the entire network complexity. Our model has less than 1M parameters,and is able to run at over 71 FPS in a single GTX 1080Ti GPU. The comprehensive experiments demonstrate that our approach achieves state-of-the-art results in terms of speed and accuracy trade-off on CityScapes dataset.

</details>


<details><summary>LinkNet</summary>

[LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/abs/1707.03718) [[codes](models/linknet.py)]  

> Abstract: Pixel-wise semantic segmentation for visual scene understanding not only needs to be accurate, but also efficient in order to find any use in real-time application. Existing algorithms even though are accurate but they do not focus on utilizing the parameters of neural network efficiently. As a result they are huge in terms of parameters and number of operations; hence slow too. In this paper, we propose a novel deep neural network architecture which allows it to learn without any significant increase in number of parameters. Our network uses only 11.5 million parameters and 21.2 GFLOPs for processing an image of resolution 3x640x360. It gives state-of-the-art performance on CamVid and comparable results on Cityscapes dataset. We also compare our networks processing time on NVIDIA GPU and embedded system device with existing state-of-the-art architectures for different image resolutions.

</details>


<details><summary>Lite-HRNet</summary>

[Lite-HRNet: A Lightweight High-Resolution Network](https://arxiv.org/abs/2104.06403) [[codes](models/lite_hrnet.py)]  

> Abstract: We present an efficient high-resolution network, Lite-HRNet, for human pose estimation. We start by simply applying the efficient shuffle block in ShuffleNet to HRNet (high-resolution network), yielding stronger performance over popular lightweight networks, such as MobileNet, ShuffleNet, and Small HRNet.
We find that the heavily-used pointwise (1x1) convolutions in shuffle blocks become the computational bottleneck. We introduce a lightweight unit, conditional channel weighting, to replace costly pointwise (1x1) convolutions in shuffle blocks. The complexity of channel weighting is linear w.r.t the number of channels and lower than the quadratic time complexity for pointwise convolutions. Our solution learns the weights from all the channels and over multiple resolutions that are readily available in the parallel branches in HRNet. It uses the weights as the bridge to exchange information across channels and resolutions, compensating the role played by the pointwise (1x1) convolution. Lite-HRNet demonstrates superior results on human pose estimation over popular lightweight networks. Moreover, Lite-HRNet can be easily applied to semantic segmentation task in the same lightweight manner. The code and models have been publicly available at [this https URL](https://github.com/HRNet/Lite-HRNet).

</details>


<details><summary>LiteSeg</summary>

[LiteSeg: A Novel Lightweight ConvNet for Semantic Segmentation](https://arxiv.org/abs/1912.06683) [[codes](models/liteseg.py)]  

> Abstract: Semantic image segmentation plays a pivotal role in many vision applications including autonomous driving and medical image analysis. Most of the former approaches move towards enhancing the performance in terms of accuracy with a little awareness of computational efficiency. In this paper, we introduce LiteSeg, a lightweight architecture for semantic image segmentation. In this work, we explore a new deeper version of Atrous Spatial Pyramid Pooling module (ASPP) and apply short and long residual connections, and depthwise separable convolution, resulting in a faster and efficient model. LiteSeg architecture is introduced and tested with multiple backbone networks as Darknet19, MobileNet, and ShuffleNet to provide multiple trade-offs between accuracy and computational cost. The proposed model LiteSeg, with MobileNetV2 as a backbone network, achieves an accuracy of 67.81% mean intersection over union at 161 frames per second with 640×360 resolution on the Cityscapes dataset.

</details>


<details><summary>MiniNet</summary>

[Enhancing V-SLAM Keyframe Selection with an Efficient ConvNet for Semantic Analysis](https://ieeexplore.ieee.org/abstract/document/8793923) [[codes](models/mininet.py)]  

> Abstract: Selecting relevant visual information from a video is a challenging task on its own and even more in robotics, due to strong computational restrictions. This work proposes a novel keyframe selection strategy based on image quality and semantic information, which boosts strategies currently used in Visual-SLAM (V-SLAM). Commonly used V-SLAM methods select keyframes based only on relative displacements and amount of tracked feature points. Our strategy to select more carefully these keyframes allows the robotic systems to make better use of them. With minimal computational cost, we show that our selection includes more relevant keyframes, which are useful for additional posterior recognition tasks, without penalizing the existing ones, mainly place recognition. A key ingredient is our novel CNN architecture to run a quick semantic image analysis at the onboard CPU of the robot. It provides sufficient accuracy significantly faster than related works. We demonstrate our hypothesis with several public datasets with challenging robotic data.

</details>


<details><summary>MiniNetv2</summary>

[MiniNet: An Efficient Semantic Segmentation ConvNet for Real-Time Robotic Applications](https://ieeexplore.ieee.org/abstract/document/9023474) [[codes](models/mininetv2.py)]  

> Abstract: Efficient models for semantic segmentation, in terms of memory, speed, and computation, could boost many robotic applications with strong computational and temporal restrictions. This article presents a detailed analysis of different techniques for efficient semantic segmentation. Following this analysis, we have developed a novel architecture, MiniNet-v2, an enhanced version of MiniNet. MiniNet-v2 is built considering the best option depending on CPU or GPU availability. It reaches comparable accuracy to the state-of-the-art models but uses less memory and computational resources. We validate and analyze the details of our architecture through a comprehensive set of experiments on public benchmarks (Cityscapes, Camvid, and COCO-Text datasets), showing its benefits over relevant prior work. Our experiments include a sample application where these models can boost existing robotic applications.

</details>


<details><summary>PP-LiteSeg</summary>

[PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model](https://arxiv.org/abs/2204.02681) [[codes](models/pp_liteseg.py)]  

> Abstract: Real-world applications have high demands for semantic segmentation methods. Although semantic segmentation has made remarkable leap-forwards with deep learning, the performance of real-time methods is not satisfactory. In this work, we propose PP-LiteSeg, a novel lightweight model for the real-time semantic segmentation task. Specifically, we present a Flexible and Lightweight Decoder (FLD) to reduce computation overhead of previous decoder. To strengthen feature representations, we propose a Unified Attention Fusion Module (UAFM), which takes advantage of spatial and channel attention to produce a weight and then fuses the input features with the weight. Moreover, a Simple Pyramid Pooling Module (SPPM) is proposed to aggregate global context with low computation cost. Extensive evaluations demonstrate that PP-LiteSeg achieves a superior trade-off between accuracy and speed compared to other methods. On the Cityscapes test set, PP-LiteSeg achieves 72.0% mIoU/273.6 FPS and 77.5% mIoU/102.6 FPS on NVIDIA GTX 1080Ti. Source code and models are available at PaddleSeg: [this https URL](https://github.com/PaddlePaddle/PaddleSeg).

</details>


<details><summary>RegSeg</summary>

[Rethinking Dilated Convolution for Real-time Semantic Segmentation](https://arxiv.org/abs/2111.09957) [[codes](models/regseg.py)]  

> Abstract: The field-of-view is an important metric when designing a model for semantic segmentation. To obtain a large field-of-view, previous approaches generally choose to rapidly downsample the resolution, usually with average poolings or stride 2 convolutions. We take a different approach by using dilated convolutions with large dilation rates throughout the backbone, allowing the backbone to easily tune its field-of-view by adjusting its dilation rates, and show that it's competitive with existing approaches. To effectively use the dilated convolution, we show a simple upper bound on the dilation rate in order to not leave gaps in between the convolutional weights, and design an SE-ResNeXt inspired block structure that uses two parallel 3×3 convolutions with different dilation rates to preserve the local details. Manually tuning the dilation rates for every block can be difficult, so we also introduce a differentiable neural architecture search method that uses gradient descent to optimize the dilation rates. In addition, we propose a lightweight decoder that restores local information better than common alternatives. To demonstrate the effectiveness of our approach, our model RegSeg achieves competitive results on real-time Cityscapes and CamVid datasets. Using a T4 GPU with mixed precision, RegSeg achieves 78.3 mIOU on Cityscapes test set at 37 FPS, and 80.9 mIOU on CamVid test set at 112 FPS, both without ImageNet pretraining.

</details>


<details><summary>SegNet</summary>

[SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561) [[codes](models/segnet.py)]  

> Abstract: We present a novel and practical deep fully convolutional neural network architecture for semantic pixel-wise segmentation termed SegNet. This core trainable segmentation engine consists of an encoder network, a corresponding decoder network followed by a pixel-wise classification layer. The architecture of the encoder network is topologically identical to the 13 convolutional layers in the VGG16 network. The role of the decoder network is to map the low resolution encoder feature maps to full input resolution feature maps for pixel-wise classification. The novelty of SegNet lies is in the manner in which the decoder upsamples its lower resolution input feature map(s). Specifically, the decoder uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear upsampling. This eliminates the need for learning to upsample. The upsampled maps are sparse and are then convolved with trainable filters to produce dense feature maps. We compare our proposed architecture with the widely adopted FCN and also with the well known DeepLab-LargeFOV, DeconvNet architectures. This comparison reveals the memory versus accuracy trade-off involved in achieving good segmentation performance.
SegNet was primarily motivated by scene understanding applications. Hence, it is designed to be efficient both in terms of memory and computational time during inference. It is also significantly smaller in the number of trainable parameters than other competing architectures. We also performed a controlled benchmark of SegNet and other architectures on both road scenes and SUN RGB-D indoor scene segmentation tasks. We show that SegNet provides good performance with competitive inference time and more efficient inference memory-wise as compared to other architectures. We also provide a Caffe implementation of SegNet and a web demo at this http URL.

</details>


<details><summary>ShelfNet</summary>

[ShelfNet for Fast Semantic Segmentation](https://arxiv.org/abs/1811.11254) [[codes](models/shelfnet.py)]  

> Abstract: In this paper, we present ShelfNet, a novel architecture for accurate fast semantic segmentation. Different from the single encoder-decoder structure, ShelfNet has multiple encoder-decoder branch pairs with skip connections at each spatial level, which looks like a shelf with multiple columns. The shelf-shaped structure can be viewed as an ensemble of multiple deep and shallow paths, thus improving accuracy. We significantly reduce computation burden by reducing channel number, at the same time achieving high accuracy with this unique structure. In addition, we propose a shared-weight strategy in the residual block which reduces parameter number without sacrificing performance. Compared with popular non real-time methods such as PSPNet, our ShelfNet achieves 4× faster inference speed with similar accuracy on PASCAL VOC dataset. Compared with real-time segmentation models such as BiSeNet, our model achieves higher accuracy at comparable speed on the Cityscapes Dataset, enabling the application in speed-demanding tasks such as street-scene understanding for autonomous driving. Furthermore, our ShelfNet achieves 79.0\% mIoU on Cityscapes Dataset with ResNet34 backbone, outperforming PSPNet and BiSeNet with large backbones such as ResNet101. Through extensive experiments, we validated the superior performance of ShelfNet. We provide link to the implementation \url{[this https URL](https://github.com/juntang-zhuang/ShelfNet-lw-cityscapes)}.

</details>


<details><summary>SQNet</summary>

[Speeding up Semantic Segmentation for Autonomous Driving](https://openreview.net/pdf?id=S1uHiFyyg) [[codes](models/sqnet.py)]  

> Abstract: Deep learning has considerably improved semantic image segmentation. However, its high accuracy is traded against larger computational costs which makes it unsuitable for embedded devices in self-driving cars. We propose a novel deep network architecture for image segmentation that keeps the high accuracy while being efficient enough for embedded devices. The architecture consists of ELU activation functions, a SqueezeNet-like encoder, followed by parallel dilated convolutions, and a decoder with SharpMask-like refinement modules. On the Cityscapes dataset, the new network achieves higher segmentation accuracy than other networks that are tailored to embedded devices. Simultaneously the frame-rate is still sufficiently high for the deployment in autonomous vehicles.

</details>


<details><summary>STDC</summary>

[Rethinking BiSeNet For Real-time Semantic Segmentation](https://arxiv.org/abs/2104.13188v1) [[codes](models/stdc.py)]  

> Abstract: BiSeNet has been proved to be a popular two-stream network for real-time segmentation. However, its principle of adding an extra path to encode spatial information is time-consuming, and the backbones borrowed from pretrained tasks, e.g., image classification, may be inefficient for image segmentation due to the deficiency of task-specific design. To handle these problems, we propose a novel and efficient structure named Short-Term Dense Concatenate network (STDC network) by removing structure redundancy. Specifically, we gradually reduce the dimension of feature maps and use the aggregation of them for image representation, which forms the basic module of STDC network. In the decoder, we propose a Detail Aggregation module by integrating the learning of spatial information into low-level layers in single-stream manner. Finally, the low-level features and deep features are fused to predict the final segmentation results. Extensive experiments on Cityscapes and CamVid dataset demonstrate the effectiveness of our method by achieving promising trade-off between segmentation accuracy and inference speed. On Cityscapes, we achieve 71.9% mIoU on the test set with a speed of 250.4 FPS on NVIDIA GTX 1080Ti, which is 45.2% faster than the latest methods, and achieve 76.8% mIoU with 97.0 FPS while inferring on higher resolution images.

</details>


<details><summary>SwiftNet</summary>

[In Defense of Pre-trained ImageNet Architectures for Real-time Semantic Segmentation of Road-driving Images](https://arxiv.org/abs/1903.08469) [[codes](models/swiftnet.py)]  

> Abstract: Recent success of semantic segmentation approaches on demanding road driving datasets has spurred interest in many related application fields. Many of these applications involve real-time prediction on mobile platforms such as cars, drones and various kinds of robots. Real-time setup is challenging due to extraordinary computational complexity involved. Many previous works address the challenge with custom lightweight architectures which decrease computational complexity by reducing depth, width and layer capacity with respect to general purpose architectures. We propose an alternative approach which achieves a significantly better performance across a wide range of computing budgets. First, we rely on a light-weight general purpose architecture as the main recognition engine. Then, we leverage light-weight upsampling with lateral connections as the most cost-effective solution to restore the prediction resolution. Finally, we propose to enlarge the receptive field by fusing shared features at multiple resolutions in a novel fashion. Experiments on several road driving datasets show a substantial advantage of the proposed approach, either with ImageNet pre-trained parameters or when we learn from scratch. Our Cityscapes test submission entitled SwiftNetRN-18 delivers 75.5% MIoU and achieves 39.9 Hz on 1024x2048 images on GTX1080Ti.

</details>
<br>

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

[^smp]: [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)  

# Knowledge Distillation

Currently only support the original knowledge distillation method proposed by Geoffrey Hinton.[^kd]  

[^kd]: [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)  

# Hyperparameter Optimization

This repo also supports hyperparameter optimization using Optuna.[^optuna] For example, if you want to search hyperparameters using BiSeNetv1, you may simply run

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 optuna_search.py
```

[^optuna]: [Optuna: A hyperparameter optimization framework](https://github.com/optuna/optuna)

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
| FANet      | 2020 | ResNet18       | n.a./12.26              | 100             | 75.0/74.4                         | [74.92](https://github.com/zh320/realtime-semantic-segmentation-pytorch/releases/download/v1.2/fanet.pth)                     |
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

## Hyperparameter Optimization

| Model     | Encoder     | Method                                  | mIoU(200 epoch) |
|:---------:|:-----------:|:---------------------------------------:|:---------------:|
| BiSeNetv1 | ResNet18    | Random                                  | 72.71           |
|           |             | [Optuna](optuna_results/bisenetv1.json) | 74.40           |
| DDRNet    | None        | Random                                  | 71.18           |
|           |             | [Optuna](optuna_results/ddrnet.json)    | 72.22           |
| LiteSeg   | MobileNetv2 | Random                                  | 75.29           |
|           |             | [Optuna](optuna_results/liteseg.json)   | 75.47           |

[When using random search, the hyperparameters were chosen from the default config. For Optuna search, each experiment was performed 100 trials.]  

# Prepare the dataset

```
Cityscapes/
├── gtFine/
└── leftImg8bit/
```

# References