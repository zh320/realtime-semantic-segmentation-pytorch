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


def set_device(config, is_DDP, rank):
    if is_DDP:
        torch.cuda.set_device(rank)
        if not dist.is_initialized():
            dist.init_process_group(backend=dist.Backend.NCCL, init_method='env://')
        device = torch.device('cuda', rank)
        gpu_num = dist.get_world_size()
    else:   # DP
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_num = torch.cuda.device_count()

    # Setup num_workers
    num_workers = gpu_num * config.base_workers

    return device, gpu_num, num_workers


def parallel_model(config, is_DDP, model, rank, device):
    if is_DDP:
        if config.synBN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model.to(rank), device_ids=[rank], output_device=rank)
    else:
        model = nn.DataParallel(model)
        model.to(device)

    return model


def destroy_ddp_process(config, is_DDP):
    if is_DDP and config.destroy_ddp_process:
        dist.destroy_process_group()
