import os, random, torch, json
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    
def get_writer(config, main_rank):
    if config.use_tb and main_rank:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.tb_log_dir)
    else:
        writer = None
    return writer
    
    
def get_logger(config, main_rank):
    if main_rank:
        import sys
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")

        log_path = f'{config.save_dir}/{config.logger_name}.log'
        logger.add(log_path, format="[{time:YYYY-MM-DD HH:mm}] {message}", level="INFO")
    else:
        logger = None
    return logger


def save_config(config):
    config_dict = vars(config)
    with open(f'{config.save_dir}/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)


def log_config(config, logger):
    keys = ['dataset', 'num_class', 'model', 'encoder', 'decoder', 'loss_type', 
            'optimizer_type', 'lr_policy', 'total_epoch', 'train_bs', 'val_bs',  
            'train_num', 'val_num', 'gpu_num', 'num_workers', 'amp_training', 
            'DDP', 'kd_training', 'synBN', 'use_ema', 'use_aux']
            
    config_dict = vars(config)
    infos = f"\n\n\n{'#'*25} Config Informations {'#'*25}\n" 
    infos += '\n'.join('%s: %s' % (k, config_dict[k]) for k in keys)
    infos += f"\n{'#'*71}\n\n"
    logger.info(infos)
    

def get_colormap(config):
    if config.colormap == 'cityscapes':
        colormap = {0:(128, 64,128), 1:(244, 35,232), 2:( 70, 70, 70), 3:(102,102,156),
                    4:(190,153,153), 5:(153,153,153), 6:(250,170, 30), 7:(220,220,  0),
                    8:(107,142, 35), 9:(152,251,152), 10:( 70,130,180), 11:(220, 20, 60),
                    12:(255,  0,  0), 13:(  0,  0,142), 14:(  0,  0, 70), 15:(  0, 60,100),
                    16:(  0, 80,100), 17:(  0,  0,230), 18:(119, 11, 32)}

    elif config.colormap == 'custom':
        raise NotImplementedError()
        
    else:
        raise ValueError(f'Unsupport colormap type: {config.colormap}.')

    colormap = [color for color in colormap.values()]
    
    if len(colormap) < config.num_class:
        raise ValueError('Length of colormap is smaller than the number of class.')
    else:
        return colormap[:config.num_class]