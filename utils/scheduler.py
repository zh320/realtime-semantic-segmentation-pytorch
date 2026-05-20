from torch.optim.lr_scheduler import OneCycleLR, StepLR


def get_scheduler(config, optimizer, max_lr, total_itrs):
    if config.lr_policy == 'cos_warmup':
        warmup_ratio = config.warmup_epochs / config.total_epoch
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_itrs,
                                pct_start=warmup_ratio)

    elif config.lr_policy == 'linear':
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_itrs,
                                pct_start=0., anneal_strategy='linear')

    elif config.lr_policy == 'step':
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=0.1)

    else:
        raise NotImplementedError(f'Unsupported scheduler type: {config.lr_policy}')
    return scheduler
