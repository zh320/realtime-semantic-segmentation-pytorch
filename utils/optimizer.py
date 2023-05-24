from torch.optim import SGD, Adam, AdamW


def get_optimizer(config, model):    
    optimizer_hub = {'sgd':SGD, 'adam':Adam, 'adamw':AdamW}
    params = model.parameters()

    if config.optimizer_type == 'sgd':
        config.lr = config.base_lr * config.gpu_num
        optimizer = optimizer_hub[config.optimizer_type](params=params, lr=config.lr, 
                                                    momentum=config.momentum, 
                                                    weight_decay=config.weight_decay)

    elif config.optimizer_type in ['adam', 'adamw']:
        config.lr = 0.001 * config.gpu_num
        optimizer = optimizer_hub[config.optimizer_type](params=params, lr=config.lr)

    else:
        raise NotImplementedError(f'Unsupported optimizer type: {config.optimizer_type}')

    return optimizer