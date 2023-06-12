import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.cuda import amp
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from .loss import kd_loss_fn
from models import get_teacher_model
from utils import (get_seg_metrics, sampler_set_epoch, get_colormap)


class SegTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        if config.is_testing:
            self.colormap = torch.tensor(get_colormap(config)).to(self.device)
        else:
            self.teacher_model = get_teacher_model(config, self.device)
            self.metrics = get_seg_metrics(config).to(self.device)

    def train_one_epoch(self, config):
        self.model.train()
        
        sampler_set_epoch(config, self.train_loader, self.cur_epoch) 
    
        pbar = tqdm(self.train_loader) if self.main_rank else self.train_loader

        for cur_itrs, (images, masks) in enumerate(pbar):
            self.cur_itrs = cur_itrs
            self.train_itrs += 1

            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)    

            self.optimizer.zero_grad()
            
            # Forward path
            if config.use_aux:
                with amp.autocast(enabled=config.amp_training):
                    preds, preds_aux = self.model(images, is_training=True)
                    loss = self.loss_fn(preds, masks)
                    
                masks_auxs = masks.unsqueeze(1).float()
                if config.aux_coef is None:
                    config.aux_coef = torch.ones(len(preds_aux))
                elif len(preds_aux) != len(config.aux_coef):
                    raise ValueError('Auxiliary loss coefficient length does not match.')

                for i in range(len(preds_aux)):
                    aux_size = preds_aux[i].size()[2:]
                    masks_aux = F.interpolate(masks_auxs, aux_size, mode='nearest')
                    masks_aux = masks_aux.squeeze(1).to(self.device, dtype=torch.long)

                    with amp.autocast(enabled=config.amp_training):
                        loss += config.aux_coef[i] * self.loss_fn(preds_aux[i], masks_aux)

            else:
                with amp.autocast(enabled=config.amp_training):
                    preds = self.model(images)
                    loss = self.loss_fn(preds, masks)

            if config.use_tb and self.main_rank:
                self.writer.add_scalar('train/loss', loss.detach(), self.train_itrs)
            
            # Knowledge distillation
            if config.kd_training:
                with amp.autocast(enabled=config.amp_training):
                    with torch.no_grad():
                        teacher_preds = self.teacher_model(images)   # Teacher predictions
                        
                    loss_kd = kd_loss_fn(config, preds, teacher_preds.detach())
                    loss += config.kd_loss_coefficient * loss_kd

                if config.use_tb and self.main_rank:
                    self.writer.add_scalar('train/loss_kd', loss_kd.detach(), self.train_itrs)  
                    self.writer.add_scalar('train/loss_total', loss.detach(), self.train_itrs)      
                   
            # Backward path
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            self.ema_model.update(self.model, self.train_itrs)

            if self.main_rank:
                pbar.set_description(('%s'*2) % 
                                (f'Epoch:{self.cur_epoch}/{config.total_epoch}{" "*4}|',
                                f'Loss:{loss.detach():4.4g}{" "*4}|',)
                                )

        return

    @torch.no_grad()
    def validate(self, config, val_best=False):
        pbar = tqdm(self.val_loader) if self.main_rank else self.val_loader
        for (images, masks) in pbar:
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)

            preds = self.ema_model.ema(images)
            self.metrics.update(preds.detach(), masks)

            if self.main_rank:
                pbar.set_description(('%s'*1) % (f'Validating:{" "*4}|',))

        iou = self.metrics.compute()
        score = iou.mean()  # mIoU

        if self.main_rank:
            if val_best:
                self.logger.info(f'\n\nTrain {config.total_epoch} epochs finished.' + 
                                 f'\n\nBest mIoU is: {score:.4f}\n')
            else:
                self.logger.info(f' Epoch{self.cur_epoch} mIoU: {score:.4f}    | ' + 
                                 f'best mIoU so far: {self.best_score:.4f}\n')

            if config.use_tb and self.cur_epoch < config.total_epoch:
                self.writer.add_scalar('val/mIoU', score.cpu(), self.cur_epoch+1)
                for i in range(config.num_class):
                    self.writer.add_scalar(f'val/IoU_cls{i:02f}', iou[i].cpu(), self.cur_epoch+1)
        self.metrics.reset()
        return score

    @torch.no_grad()
    def predict(self, config):
        if config.DDP:
            raise ValueError('Predict mode currently does not support DDP.')
            
        self.logger.info('\nStart predicting...\n')

        for (images, images_aug, img_names) in tqdm(self.test_loader):
            images_aug = images_aug.to(self.device, dtype=torch.float32)
            
            preds = self.model(images_aug)
                        
            preds = self.colormap[preds.max(dim=1)[1]].cpu().numpy()
            
            images = images.cpu().numpy()

            # Saving results
            for i in range(preds.shape[0]):
                save_path = os.path.join(config.save_dir, img_names[i])
                save_suffix = img_names[i].split('.')[-1]
                 
                pred = Image.fromarray(preds[i].astype(np.uint8))
                
                if config.save_mask:
                    pred.save(save_path)
                
                if config.blend_prediction:
                    save_blend_path = save_path.replace(f'.{save_suffix}', f'_blend.{save_suffix}')
                    
                    image = Image.fromarray(images[i].astype(np.uint8))
                    image = Image.blend(image, pred, config.blend_alpha)
                    image.save(save_blend_path)
