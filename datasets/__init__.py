from torch.utils.data import DataLoader
from .cityscapes import Cityscapes

dataset_hub = {'cityscapes':Cityscapes,}


def get_dataset(config):
    if config.dataset in dataset_hub.keys():
        train_dataset = dataset_hub[config.dataset](config=config, mode='train')
        val_dataset = dataset_hub[config.dataset](config=config, mode='val')
    else:
        raise NotImplementedError('Unsupported dataset!')
        
    return train_dataset, val_dataset
    
    
def get_loader(config, rank, pin_memory=True):
    train_dataset, val_dataset = get_dataset(config)
    
    # Make sure train number is divisible by train batch size
    config.train_num = int(len(train_dataset) // config.train_bs * config.train_bs)
    config.val_num = len(val_dataset)
    
    if config.DDP:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=config.gpu_num, 
                                            rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=config.gpu_num,
                                            rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=config.train_bs, shuffle=False, 
                                    num_workers=config.num_workers, pin_memory=pin_memory, 
                                    sampler=train_sampler, drop_last=True)
                                    
        val_loader = DataLoader(val_dataset, batch_size=config.val_bs, shuffle=False,
                                    num_workers=config.num_workers, pin_memory=pin_memory,
                                    sampler=val_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.train_bs, 
                                    shuffle=True, num_workers=config.num_workers, drop_last=True)
                                    
        val_loader = DataLoader(val_dataset, batch_size=config.val_bs, 
                                    shuffle=False, num_workers=config.num_workers)

    return train_loader, val_loader


def get_test_loader(config): 
    from .test_dataset import TestDataset
    dataset = TestDataset(config)

    config.test_num = len(dataset)

    if config.DDP:
        raise NotImplementedError()

    else:
        test_loader = DataLoader(dataset, batch_size=config.test_bs, 
                                    shuffle=False, num_workers=config.num_workers)

    return test_loader