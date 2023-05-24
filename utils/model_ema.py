'''
Codes are based on
https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py
'''

import torch
import torch.nn as nn
from copy import deepcopy
from .parallel import de_parallel


def get_ema_model(config, model, device):
    return ModelEmaV2(config, model, device=device)


class ModelEmaV2(nn.Module):
    def __init__(self, config, model, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(de_parallel(model))
        self.ema.eval()
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.ema.to(device=device)
        self.use_ema = config.use_ema    
        self.total_itrs = config.total_itrs    

    @torch.no_grad()
    def _update(self, model, update_fn):
        for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if self.device is not None:
                model_v = model_v.to(device=self.device)
            ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model, cur_itrs):
        if self.use_ema:
            decay = min(max(cur_itrs / self.total_itrs, 0), 1)
            self._update(de_parallel(model), update_fn=lambda e, m: decay * e + (1. - decay) * m)
        else:
            self._update(de_parallel(model), update_fn=lambda e, m: m)
