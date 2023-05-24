import torch
import torch.nn as nn
import torch.nn.functional as F


class OhemCELoss(nn.Module):
    def __init__(self, thresh, ignore_index=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_index].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


def get_loss_fn(config, device):
    if config.class_weights is None:
        weights = None
    else:
        weights = torch.Tensor(config.class_weights).to(device)

    if config.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index, 
                                        reduction=config.reduction, weight=weights)
    
    elif config.loss_type == 'ohem':
        criterion = OhemCELoss(thresh=config.ohem_thrs, ignore_index=config.ignore_index)  

    else:
        raise NotImplementedError(f"Unsupport loss type: {config.loss_type}")
        
    return criterion
    
    
def kd_loss_fn(config, outputs, outputsT):
    if config.kd_loss_type == 'kl_div':
        lossT = F.kl_div(F.log_softmax(outputs/config.kd_temperature, dim=1),
                    F.softmax(outputsT.detach()/config.kd_temperature, dim=1)) * config.kd_temperature ** 2
                    
    elif config.kd_loss_type == 'mse':
        lossT = F.mse_loss(outputs, outputsT.detach())
        
    return lossT