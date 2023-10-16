import os, torch
import segmentation_models_pytorch as smp

from .bisenetv2 import BiSeNetv2
from .enet import ENet
from .fastscnn import FastSCNN
from .lednet import LEDNet
from .linknet import LinkNet
from .contextnet import ContextNet
from .pp_liteseg import PPLiteSeg
from .ddrnet import DDRNet
from .espnet import ESPNet
from .erfnet import ERFNet
from .segnet import SegNet
from .dabnet import DABNet
from .bisenetv1 import BiSeNetv1
from .espnetv2 import ESPNetv2
from .aglnet import AGLNet
from .cgnet import CGNet
from .edanet import EDANet
from .esnet import ESNet
from .adscnet import ADSCNet
from .canet import CANet
from .cfpnet import CFPNet


decoder_hub = {'deeplabv3':smp.DeepLabV3, 'deeplabv3p':smp.DeepLabV3Plus, 'fpn':smp.FPN,
               'linknet':smp.Linknet, 'manet':smp.MAnet, 'pan':smp.PAN, 'pspnet':smp.PSPNet,
               'unet':smp.Unet, 'unetpp':smp.UnetPlusPlus}


def get_model(config):
    model_hub = {'bisenetv2':BiSeNetv2, 'enet':ENet, 'fastscnn':FastSCNN, 'lednet':LEDNet,
                 'linknet':LinkNet, 'contextnet':ContextNet, 'ppliteseg':PPLiteSeg,
                 'ddrnet':DDRNet, 'espnet':ESPNet, 'erfnet':ERFNet, 'segnet':SegNet,
                 'dabnet':DABNet, 'bisenetv1':BiSeNetv1, 'espnetv2':ESPNetv2,
                 'aglnet':AGLNet, 'cgnet':CGNet, 'edanet':EDANet, 'esnet':ESNet,
                 'adscnet':ADSCNet, 'canet':CANet, 'cfpnet':CFPNet,}

    # The following models currently support auxiliary heads
    aux_models = ['bisenetv2', 'contextnet', 'fastscnn', 'ddrnet']
    
    if config.model == 'smp':   # Use segmentation models pytorch
        if config.decoder not in decoder_hub:
            raise ValueError(f"Unsupported decoder type: {config.decoder}")

        model = decoder_hub[config.decoder](encoder_name=config.encoder, 
                                            encoder_weights=config.encoder_weights, 
                                            in_channels=3, classes=config.num_class)

    elif config.model in model_hub.keys():
        if config.model in aux_models:
            model = model_hub[config.model](num_class=config.num_class, use_aux=config.use_aux)
        elif config.model == 'ppliteseg':
            model = model_hub[config.model](num_class=config.num_class, encoder_type=config.encoder)
        else:
            model = model_hub[config.model](num_class=config.num_class)

    else:
        raise NotImplementedError(f"Unsupport model type: {config.model}")

    return model


def get_teacher_model(config, device):
    if config.kd_training:
        if not os.path.isfile(config.teacher_ckpt):
            raise ValueError(f'Could not find teacher checkpoint at path {config.teacher_ckpt}.')
        
        if config.teacher_decoder not in decoder_hub.keys():
            raise ValueError(f"Unsupported teacher decoder type: {config.teacher_decoder}")      

        model = decoder_hub[config.teacher_decoder](encoder_name=config.teacher_encoder, 
                            encoder_weights=None, in_channels=3, classes=config.num_class)        

        teacher_ckpt = torch.load(config.teacher_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(teacher_ckpt['state_dict'])
        del teacher_ckpt
            
        model = model.to(device)    
        model.eval()
    else:
        model = None
        
    return model