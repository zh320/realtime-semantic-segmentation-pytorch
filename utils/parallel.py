import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def set_device(config, rank):
    if config.DDP:
        torch.cuda.set_device(rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=dist.Backend.NCCL, init_method='env://')
        device = torch.device('cuda', rank)
        config.gpu_num = dist.get_world_size()
    else:   # DP
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config.gpu_num = torch.cuda.device_count()
        config.train_bs *= config.gpu_num

    # Setup num_workers
    config.num_workers = config.gpu_num * config.base_workers

    return device


def parallel_model(config, model, rank, device):
    if config.DDP:
        if config.synBN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model.to(rank), device_ids=[rank], output_device=rank)    
    else:
        model = nn.DataParallel(model)
        model.to(device)

    return model


def destroy_ddp_process(config):
    if config.DDP and config.destroy_ddp_process:
        dist.destroy_process_group()


def sampler_set_epoch(config, loader, cur_epochs):
    if config.DDP:
        loader.sampler.set_epoch(cur_epochs)