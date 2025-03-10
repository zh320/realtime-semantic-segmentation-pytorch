import os


class BaseConfig:
    def __init__(self,):
        # Task
        self.task = 'train' # train, val, predict

        # Dataset
        self.dataset = None
        self.dataroot = None
        self.num_class = -1
        self.ignore_index = 255

        # Model
        self.model = None
        self.encoder = None
        self.decoder = None
        self.encoder_weights = 'imagenet'

        # Detail Head (For STDC)
        self.use_detail_head = False
        self.detail_thrs = 0.1
        self.detail_loss_coef = 1.0
        self.dice_loss_coef = 1.0
        self.bce_loss_coef = 1.0

        # Training
        self.total_epoch = 200
        self.base_lr = 0.01
        self.train_bs = 16      # For each GPU
        self.use_aux = False
        self.aux_coef = None

        # Validating
        self.val_bs = 16        # For each GPU
        self.begin_val_epoch = 0    # Epoch to start validation
        self.val_interval = 1   # Epoch interval between validation

        # Testing
        self.test_bs = 16
        self.test_data_folder = None
        self.colormap = 'cityscapes'
        self.save_mask = True
        self.blend_prediction = True
        self.blend_alpha = 0.3

        # Loss
        self.loss_type = 'ohem'
        self.class_weights = None
        self.ohem_thrs = 0.7
        self.reduction = 'mean'

        # Scheduler
        self.lr_policy = 'cos_warmup'
        self.warmup_epochs = 3

        # Optimizer
        self.optimizer_type = 'sgd'
        self.momentum = 0.9         # For SGD
        self.weight_decay = 1e-4    # For SGD

        # Monitoring
        self.save_ckpt = True
        self.save_dir = 'save'
        self.use_tb = True          # tensorboard
        self.tb_log_dir = None
        self.ckpt_name = None
        self.logger_name = None

        # Training setting
        self.amp_training = False
        self.resume_training = True
        self.load_ckpt = True
        self.load_ckpt_path = None
        self.base_workers = 8
        self.random_seed = 1
        self.use_ema = False

        # Augmentation
        self.crop_size = 512
        self.crop_h = None
        self.crop_w = None
        self.scale = 1.0
        self.randscale = 0.0
        self.brightness = 0.0
        self.contrast = 0.0
        self.saturation = 0.0
        self.h_flip = 0.0
        self.v_flip = 0.0

        # DDP
        self.synBN = True
        self.destroy_ddp_process = True

        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = ''
        self.teacher_model = 'smp'
        self.teacher_encoder = None
        self.teacher_decoder = None
        self.kd_loss_type = 'kl_div'
        self.kd_loss_coefficient = 1.0
        self.kd_temperature = 4.0

        # Export
        self.export_format = 'onnx'
        self.export_size = (512, 1024)
        self.export_name = None
        self.onnx_opset = 11
        self.load_onnx_path = None

    def init_dependent_config(self):
        if self.load_ckpt_path is None and self.task == 'train':
            self.load_ckpt_path = f'{self.save_dir}/last.pth'

        if self.tb_log_dir is None:
            self.tb_log_dir = f'{self.save_dir}/tb_logs/'

        if self.crop_h is None:
            self.crop_h = self.crop_size

        if self.crop_w is None:
            self.crop_w = self.crop_size

        if self.export_name is None:
            suffix = os.path.basename(self.load_ckpt_path).replace('.pth', '') if self.load_ckpt_path else 'dummy'
            self.export_name = f'{self.model}_{suffix}'