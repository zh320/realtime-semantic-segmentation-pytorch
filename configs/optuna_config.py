try:
    import optuna
except:
    raise RuntimeError('Unable to import Optuna. Please check whether you have installed it correctly.\n')
from .base_config import BaseConfig


class OptunaConfig(BaseConfig):
    def __init__(self,):
        super().__init__()
        # Task
        self.task = 'train'

        # Dataset
        self.dataset = 'cityscapes'
        self.data_root = '/path/to/your/dataset'
        self.num_class = 19

        # Model
        self.model = 'bisenetv1'

        # Training
        self.total_epoch = 200
        self.train_bs = 5
        self.loss_type = 'ohem'
        self.logger_name = 'seg_trainer'

        # Validating
        self.val_bs = 4

        # Training setting
        self.load_ckpt = False

        # DDP
        self.synBN = True
        self.destroy_ddp_process = False

        # Augmentation
        self.scale = 1.0
        self.crop_size = 1024

        # Optuna
        self.study_name = 'optuna-study'
        self.study_direction = 'maximize'
        self.num_trial = 100
        self.save_every_trial = True

    def get_trial_params(self, trial):
        self.optimizer_type = trial.suggest_categorical('optimizer', ['sgd', 'adam', 'adamw'])
        self.base_lr = trial.suggest_loguniform('base_lr', 1e-3, 1e-1)
        self.use_ema = trial.suggest_categorical('use_ema', [True, False])
        self.scale_max = trial.suggest_float('scale_max', 0.25, 1.5)
        self.scale_min = trial.suggest_float('scale_min', 0.1, 0.8)
        self.brightness = trial.suggest_float('brightness', 0.0, 0.9)
        self.contrast = trial.suggest_float('contrast', 0.0, 0.9)
        self.saturation = trial.suggest_float('saturation', 0.0, 0.9)
        self.h_flip = trial.suggest_float('h_flip', 0.0, 0.5)

        self.randscale = [-self.scale_min, self.scale_max]