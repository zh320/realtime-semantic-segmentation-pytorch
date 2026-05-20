from torch.optim import SGD, Adam, AdamW


def get_optimizer(config, model, gpu_num):
    optimizer_hub = {'sgd':SGD, 'adam':Adam, 'adamw':AdamW}
    params = model.parameters()

    if config.optimizer_type == 'sgd':
        max_lr = config.base_lr * gpu_num
        optimizer = optimizer_hub[config.optimizer_type](params=params, lr=max_lr,
                                                    momentum=config.momentum,
                                                    weight_decay=config.weight_decay)

    elif config.optimizer_type in ['adam', 'adamw']:
        max_lr = 0.1 * config.base_lr * gpu_num
        optimizer = optimizer_hub[config.optimizer_type](params=params, lr=max_lr)

    else:
        raise NotImplementedError(f'Unsupported optimizer type: {config.optimizer_type}')

    return optimizer, max_lr
