import os
from collections import namedtuple
import yaml
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2
from utils import transforms


class Custom(Dataset):
    '''
    demo for load custom datasets.
    '''

    def __init__(self, config, mode='train'):
        data_root = os.path.expanduser(config.data_root)
        dataset_config_filepath = os.path.join(data_root, 'data.yaml')
        if not os.path.exists(dataset_config_filepath):
            raise Exception(f"{dataset_config_filepath} not exists.")
        with open(dataset_config_filepath, 'r', encoding='utf-8') as yaml_file:
            dataset_config = yaml.safe_load(yaml_file)
        print('dataset_config: ', dataset_config)
        data_root = dataset_config['path']
        # self.num_classes = len(dataset_config['names'])
        self.id_to_train_id = dict()
        for i in range(len(dataset_config['names'])):
            self.id_to_train_id[i] = i
        # self.train_id_to_name = dict()
        # for k, v in dataset_config['names'].items():
        #     self.train_id_to_name[k] = str(v)
            
        img_dir = os.path.join(data_root, mode, 'imgs')
        msk_dir = os.path.join(data_root, mode, 'masks')

        if not os.path.isdir(img_dir):
            raise RuntimeError(f'Image directory: {img_dir} does not exist.')
            
        if not os.path.isdir(msk_dir):
            raise RuntimeError(f'Mask directory: {msk_dir} does not exist.')
        
        if mode == 'train':
            self.transform = AT.Compose([
                transforms.ResizeToSquare(size=config.train_size),
                transforms.Scale(scale=config.scale),
                AT.RandomScale(scale_limit=config.randscale),
                AT.PadIfNeeded(min_height=config.crop_h, min_width=config.crop_w, value=(114,114,114), mask_value=(0,0,0)),
                AT.RandomCrop(height=config.crop_h, width=config.crop_w),
                AT.ColorJitter(brightness=config.brightness, contrast=config.contrast, saturation=config.saturation),
                AT.HorizontalFlip(p=config.h_flip),
                AT.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2(),                
            ])
            
        elif mode == 'val':
            self.transform = AT.Compose([
                transforms.ResizeToSquare(size=config.test_size),
                transforms.Scale(scale=config.scale),
                AT.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2(),
            ])

        self.images = []
        self.masks = []

        for img_file_name in os.listdir(img_dir):
            img_file_basename = os.path.splitext(img_file_name)[0]

            self.images.append(os.path.join(img_dir, img_file_name))
            self.masks.append(os.path.join(msk_dir, img_file_basename + '.png'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        mask = np.asarray(Image.open(self.masks[index]).convert('L'))
        
        # Perform augmentation and normalization
        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        
        return image, mask
  