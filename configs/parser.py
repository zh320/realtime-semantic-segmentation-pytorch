import argparse

from datasets import list_available_datasets
from models import list_available_models


def load_parser(config):
    args = get_parser()

    for k,v in vars(args).items():
        if v is not None:
            try:
                exec(f"config.{k} = v")
            except:
                raise RuntimeError(f'Unable to assign value to config.{k}')
    return config


def get_parser():
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument('--task', type=str, default=None, choices = ['train', 'val', 'predict'],
        help='choose which task you want to use')

    # Dataset
    dataset_list = list_available_datasets()
    parser.add_argument('--dataset', type=str, default=None, choices=dataset_list,
        help='choose which dataset you want to use')
    parser.add_argument('--dataroot', type=str, default=None, 
        help='path to your dataset')
    parser.add_argument('--num_class', type=int, default=None, 
        help='number of classes')
    parser.add_argument('--ignore_index', type=int, default=None, 
        help='ignore index used for cross_entropy/ohem loss')

    # Model
    model_list = list_available_models()
    parser.add_argument('--model', type=str, default=None, choices=model_list,
        help='choose which model you want to use')
    parser.add_argument('--encoder', type=str, default=None, 
        help='choose which encoder of SMP model you want to use (please refer to SMP repo)')
    parser.add_argument('--decoder', type=str, default=None, 
        choices = ['deeplabv3', 'deeplabv3p', 'fpn', 'linknet', 'manet', 
                   'pan', 'pspnet', 'unet', 'unetpp'],
        help='choose which decoder of SMP model you want to use (please refer to SMP repo)')
    parser.add_argument('--encoder_weights', type=str, default=None, 
        help='choose which pretrained weight of SMP encoder you want to use (please refer to SMP repo)')

    # Training
    parser.add_argument('--total_epoch', type=int, default=None, 
        help='number of total training epochs')
    parser.add_argument('--base_lr', type=float, default=None, 
        help='base learning rate for single GPU, total learning rate *= gpu number')
    parser.add_argument('--train_bs', type=int, default=None, 
        help='training batch size for single GPU, total batch size *= gpu number')
    parser.add_argument('--use_aux', action='store_true', default=None,
        help='whether to use auxiliary heads or not if exist (default: False)')
    parser.add_argument('--aux_coef', type=tuple, default=None, 
        help='coefficients of auxiliary losses, must have the same length as auxiliary heads')

    # Validating
    parser.add_argument('--val_bs', type=int, default=None, 
        help='validating batch size for single GPU, total batch size *= gpu number')    
    parser.add_argument('--begin_val_epoch', type=int, default=None, 
        help='which epoch to start validating')    
    parser.add_argument('--val_interval', type=int, default=None, 
        help='epoch interval between two validations')

    # Testing
    parser.add_argument('--test_bs', type=int, default=None, 
        help='testing batch size (currently only support single GPU)')
    parser.add_argument('--test_data_folder', type=str, default=None, 
        help='path to your testing image folder')
    parser.add_argument('--colormap', type=str, default=None, choices = ['cityscapes', 'custom'],
        help='choose which colormap of visulization you want to use')
    parser.add_argument('--save_mask', action='store_false', default=None,
        help='whether to save the predicted mask or not (default: True)')
    parser.add_argument('--blend_prediction', action='store_false', default=None,
        help='whether to blend the image and mask using colormap for visualization or not (default: True)')
    parser.add_argument('--blend_alpha', type=float, default=None, 
        help='coefficient to blend the mask with the image')

    # Loss
    parser.add_argument('--loss_type', type=str, default=None, choices = ['ce', 'ohem'],
        help='choose which loss you want to use')
    parser.add_argument('--class_weights', type=tuple, default=None, 
        help='class weights for cross entropy loss')
    parser.add_argument('--ohem_thrs', type=float, default=None, 
        help='filtering threshold for ohem loss')

    # Scheduler
    parser.add_argument('--lr_policy', type=str, default=None, 
        choices = ['cos_warmup', 'linear', 'step'],
        help='choose which learning rate policy you want to use')
    parser.add_argument('--warmup_epochs', type=int, default=None, 
        help='warmup epoch number for `cos_warmup` learning rate policy')

    # Optimizer
    parser.add_argument('--optimizer_type', type=str, default=None, 
        choices = ['sgd', 'adam', 'adamw'],
        help='choose which optimizer you want to use')
    parser.add_argument('--momentum', type=float, default=None, 
        help='momentum of SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=None, 
        help='weight decay rate of SGD optimizer')

    # Monitoring
    parser.add_argument('--save_ckpt', action='store_false', default=None,
        help='whether to save checkpoint or not (default: True)')
    parser.add_argument('--save_dir', type=str, default=None, 
        help='path to save checkpoints and training configurations etc.')
    parser.add_argument('--use_tb', action='store_false', default=None,
        help='whether to use tensorboard or not (default: True)')
    parser.add_argument('--tb_log_dir', type=str, default=None, 
        help='path to save tensorboard logs')
    parser.add_argument('--ckpt_name', type=str, default=None, 
        help='given name of the saved checkpoint, otherwise use `last` and `best`')

    # Training setting
    parser.add_argument('--amp_training', action='store_true', default=None,
        help='whether to use automatic mixed precision training or not (default: False)')
    parser.add_argument('--resume_training', action='store_false', default=None,
        help='whether to load training state from specific checkpoint or not if present (default: True)')
    parser.add_argument('--load_ckpt', action='store_false', default=None,
        help='whether to load given checkpoint or not if exist (default: True)')
    parser.add_argument('--load_ckpt_path', type=str, default=None, 
        help='path to load specific checkpoint, otherwise try to load `last.pth`')
    parser.add_argument('--base_workers', type=int, default=None, 
        help='number of workers for single GPU, total workers *= number of GPU')
    parser.add_argument('--random_seed', type=int, default=None, 
        help='random seed')
    parser.add_argument('--use_ema', action='store_true', default=None,
        help='whether to use exponetial moving average to update weights or not (default: False)')

    # Augmentation
    parser.add_argument('--crop_size', type=int, default=None, 
        help='crop size for RandomCrop augmentation if crop_h or crop_w is not given')
    parser.add_argument('--crop_h', type=int, default=None, 
        help='crop height for RandomCrop augmentation')
    parser.add_argument('--crop_w', type=int, default=None, 
        help='crop width for RandomCrop augmentation')
    parser.add_argument('--scale', type=float, default=None, 
        help='resize the input images and masks accordingly')
    parser.add_argument('--randscale', type=tuple, default=None, 
        help='scale limit for RandomScale augmentation')
    parser.add_argument('--brightness', type=float, default=None, 
        help='brightness limit for ColorJitter augmentation')
    parser.add_argument('--contrast', type=float, default=None, 
        help='contrast limit for ColorJitter augmentation')
    parser.add_argument('--saturation', type=float, default=None, 
        help='saturation limit for ColorJitter augmentation')
    parser.add_argument('--h_flip', type=float, default=None, 
        help='probability to perform HorizontalFlip')
    parser.add_argument('--v_flip', type=float, default=None, 
        help='probability to perform VerticalFlip')

    # DDP
    parser.add_argument('--synBN', action='store_false', default=None, 
        help='whether to use SyncBatchNorm or not if trained with DDP (default: True)')
    parser.add_argument('--local_rank', type=int, default=None, 
        help='used for DDP, DO NOT CHANGE')

    # Knowledge Distillation
    parser.add_argument('--kd_training', action='store_true', default=None,
        help='whether to use knowledge distillation or not (default: False)')
    parser.add_argument('--teacher_ckpt', type=str, default=None, 
        help='path to your teacher checkpoint')
    parser.add_argument('--teacher_model', type=str, default=None, 
        help='name of your teacher model')
    parser.add_argument('--teacher_encoder', type=str, default=None, 
        help='name of your teacher encoder if use SMP model')
    parser.add_argument('--teacher_decoder', type=str, default=None, 
        help='name of your teacher decoder if use SMP model')
    parser.add_argument('--kd_loss_type', type=str, default=None, choices = ['kl_div', 'mse'],
        help='choose which loss you want to perform knowledge distillation')
    parser.add_argument('--kd_loss_coefficient', type=float, default=None, 
        help='coefficient of knowledge distillation loss')
    parser.add_argument('--kd_temperature', type=float, default=None, 
        help='temperature used for KL divergence loss')

    # Export
    parser.add_argument('--export_format', type=str, default=None, choices = ['onnx'],
        help='choose which `export_format` you want to use')
    parser.add_argument('--export_size', type=tuple, default=None, 
        help='input shape for exportation. Required by static graph format like ONNX.')
    parser.add_argument('--export_name', type=str, default=None, 
        help='given name for the target exported file')
    parser.add_argument('--onnx_opset', type=int, default=None, 
        help='ONNX opset version')
    parser.add_argument('--load_onnx_path', type=str, default=None, 
        help='path to load a specific ONNX file, otherwise try to export a dummy one according to `model`')

    args = parser.parse_args()
    return args