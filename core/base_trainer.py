import os
import torch
from torch.cuda import amp
from copy import deepcopy
from .loss import get_loss_fn
from models import get_model
from datasets import get_loader, get_test_loader
from utils import (get_optimizer, get_scheduler, parallel_model, de_parallel, 
                    get_ema_model, set_seed, set_device, get_writer, get_logger, 
                    destroy_ddp_process, mkdir, save_config, log_config,)


class BaseTrainer:
    def __init__(self, config):
        super(BaseTrainer, self).__init__()
        # DDP parameters, DO NOT CHANGE
        self.rank = int(os.getenv('RANK', -1))
        self.local_rank = int(os.getenv('LOCAL_RANK', -1))
        self.world_size = int(os.getenv('WORLD_SIZE', 1))
        config.DDP = self.local_rank != -1
        self.main_rank = self.local_rank in [-1, 0]

        # Logger compatible with ddp training
        self.logger = get_logger(config, self.main_rank)

        # Select device to train the model
        self.device = set_device(config, self.local_rank)

        # Automatic mixed precision training scaler
        self.scaler = amp.GradScaler(enabled=config.amp_training)
        
        # Create directory to save checkpoints and logs
        mkdir(config.save_dir)
        
        # Set random seed to obtain reproducible results
        set_seed(config.random_seed)

        # Define model and put it to the selected device
        self.model = get_model(config).to(self.device)

        if config.is_testing:
            self.test_loader = get_test_loader(config)
        else:
            # Tensorboard monitor
            self.writer = get_writer(config, self.main_rank)
        
            # Define loss function
            self.loss_fn = get_loss_fn(config, self.device)
            
            # Get train and validate loader
            self.train_loader, self.val_loader = get_loader(config, self.local_rank)
        
            # Define optimizer
            self.optimizer = get_optimizer(config, self.model)
            
            # Define scheduler to control how learning rate changes
            self.scheduler = get_scheduler(config, self.optimizer)
            
            # Define variables to monitor training process
            self.best_score = 0.
            self.cur_epoch = 0
            self.train_itrs = 0
        
        # Load specific checkpoints if needed
        self.load_ckpt(config)

        # Use exponential moving average of checkpoint update if needed
        if not config.is_testing:
            self.ema_model = get_ema_model(config, self.model, self.device)

    def run(self, config):
        # Parallel the model using DP or DDP
        self.parallel_model(config)
        
        # Output the training/validating configs (only in rank 0 if DDP)
        if self.main_rank:
            save_config(config)
            log_config(config, self.logger)
        
        # Start training from the latest epoch or from scratch
        start_epoch = self.cur_epoch
        for cur_epoch in range(start_epoch, config.total_epoch):
            self.cur_epoch = cur_epoch
            
            self.train_one_epoch(config)
            
            if cur_epoch >= config.begin_val_epoch and cur_epoch % config.val_interval == 0:
                val_score = self.validate(config)
                
                if self.main_rank and val_score > self.best_score:
                    # Save best model
                    self.best_score = val_score
                    if config.save_ckpt:
                        self.save_ckpt(config, save_best=True) 

            if self.main_rank and config.save_ckpt:
                # Save last model    
                self.save_ckpt(config)

        # Close tensorboard after training
        if config.use_tb and self.main_rank:
            self.writer.flush()
            self.writer.close()

        # Validate for the best model
        if config.save_ckpt:
            self.val_best(config)
        
        destroy_ddp_process(config)
        
    def parallel_model(self, config):
        self.model = parallel_model(config, self.model, self.local_rank, self.device)

    def train_one_epoch(self, config):
        '''You may implement whatever training process you like here, 
            e.g. knowledge distillation, self-supervised learning or 
            semi-supervised learning.'''
        raise NotImplementedError()

    def validate(self, config):
        raise NotImplementedError()   

    def predict(self, config):
        raise NotImplementedError()

    def load_ckpt(self, config):
        if config.load_ckpt and os.path.isfile(config.load_ckpt_path):
            checkpoint = torch.load(config.load_ckpt_path, map_location=torch.device(self.device))
            self.model.load_state_dict(checkpoint['state_dict'])
            if self.main_rank:
                self.logger.info(f"Load model state dict from {config.load_ckpt_path}")

            if not config.is_testing and config.resume_training:
                self.cur_epoch = checkpoint['cur_epoch'] + 1
                self.best_score = checkpoint['best_score']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.train_itrs = self.cur_epoch * config.iters_per_epoch

                if self.main_rank:
                    self.logger.info(f"Resume training from {config.load_ckpt_path}")
   
            del checkpoint
        else:
            if config.is_testing:
                raise ValueError(f'Could not find any pretrained checkpoint at path: {config.load_ckpt_path}.')
            else:
                if self.main_rank:
                    self.logger.info('[!] Train from scratch')

    def save_ckpt(self, config, save_best=False):
        if config.ckpt_name is None:
            save_name = 'best.pth' if save_best else 'last.pth'
        save_path = f'{config.save_dir}/{save_name}'
        state_dict = self.ema_model.ema.state_dict() if save_best else de_parallel(self.model).state_dict()

        torch.save({
            'cur_epoch': self.cur_epoch,
            'best_score': self.best_score,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict() if not save_best else None,
            'scheduler': self.scheduler.state_dict() if not save_best else None,
        }, save_path)

    def val_best(self, config, ckpt_path=None):
        ckpt_path = f"{config.save_dir}/best.pth" if ckpt_path is None else ckpt_path
        if not os.path.isfile(ckpt_path):
            raise ValueError(f'Best checkpoint does not exist at {ckpt_path}')
        
        if self.main_rank:
            self.logger.info(f"\nTrain {config.total_epoch} epochs finished!\n")
            self.logger.info(f'{"#"*50}\nValidation for the best checkpoint...')

        self.model = de_parallel(self.model)
        checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['state_dict'])
        
        self.model.to(self.device)
        del checkpoint
        
        self.ema_model.ema = deepcopy(de_parallel(self.model)).eval()

        val_score = self.validate(config, val_best=True)

        if self.main_rank:
            self.logger.info(f'Best validation score is {val_score}.\n')
