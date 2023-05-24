from torch.optim.lr_scheduler import OneCycleLR, StepLR
from math import ceil


def get_scheduler(config, optimizer):
    if config.DDP:
        config.iters_per_epoch = ceil(config.train_num/config.train_bs/config.gpu_num)
    else:
        config.iters_per_epoch = ceil(config.train_num/config.train_bs)
    config.total_itrs = int(config.total_epoch*config.iters_per_epoch)

    if config.lr_policy == 'cos_warmup':
        warmup_ratio = config.warmup_epochs / config.total_epoch
        scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.total_itrs, 
                                pct_start=warmup_ratio)

    elif config.lr_policy == 'linear':
        scheduler = OneCycleLR(optimizer, max_lr=config.lr, total_steps=config.total_itrs, 
                                pct_start=0., anneal_strategy='linear')

    elif config.lr_policy == 'step':
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=0.1)

    else:
        raise NotImplementedError(f'Unsupported scheduler type: {config.lr_policy}')
    return scheduler