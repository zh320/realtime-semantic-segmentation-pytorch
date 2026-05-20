from torch.utils.data import DataLoader

from .cityscapes import Cityscapes
from .dataset_registry import dataset_hub


def get_dataset(config, mode):
    if config.dataset not in dataset_hub.keys():
        raise NotImplementedError('Unsupported dataset!')

    return dataset_hub[config.dataset](config=config, mode=mode)


def get_loader(config, mode, is_DDP, batch_size, rank, gpu_num, num_workers, pin_memory=True):
    dataset = get_dataset(config, mode)

    is_train = mode == 'train'

    # Make sure train number is divisible by train batch size
    num_samples = int(len(dataset) // batch_size * batch_size) if is_train else len(dataset)

    if is_DDP:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=gpu_num, rank=rank, shuffle=is_train)
    else:   # DP
        sampler = None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(is_train and not is_DDP),
                        sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                        drop_last=is_train)

    return loader, num_samples


def get_test_loader(config, is_DDP, num_workers):
    from .test_dataset import TestDataset
    dataset = TestDataset(config)

    test_num = len(dataset)

    if is_DDP:
        raise NotImplementedError()

    else:
        test_loader = DataLoader(dataset, batch_size=config.test_bs,
                                    shuffle=False, num_workers=num_workers)

    return test_loader, test_num


def list_available_datasets():
    dataset_list = list(dataset_hub.keys())

    return dataset_list
