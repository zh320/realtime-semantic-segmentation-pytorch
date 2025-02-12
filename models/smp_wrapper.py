import segmentation_models_pytorch as smp


encoder_hub = smp.encoders.get_encoder_names()
decoder_hub = {'deeplabv3':smp.DeepLabV3, 'deeplabv3p':smp.DeepLabV3Plus, 'fpn':smp.FPN,
               'linknet':smp.Linknet, 'manet':smp.MAnet, 'pan':smp.PAN, 'pspnet':smp.PSPNet,
               'unet':smp.Unet, 'unetpp':smp.UnetPlusPlus}


def get_smp_model(encoder_name, decoder_name, encoder_weights, num_class):
    if encoder_name not in encoder_hub:
        raise ValueError(f'Unsupported encoder: {encoder_name} for SMP model. Available encoders are:\n {encoder_hub}.')

    if decoder_name not in decoder_hub:
        raise ValueError(f'Unsupported decoder: {decoder_name} for SMP model. Available decoders are:\n {decoder_hub.keys()}.')

    model = decoder_hub[decoder_name](encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=num_class)

    return model