import os
from collections import namedtuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as AT
from albumentations.pytorch import ToTensorV2
from utils import transforms


class CamVid(Dataset):
    # Codes are based on https://github.com/mcordts/cityscapesScripts
      

    def __init__(self, config, mode='train'):
        data_root = os.path.expanduser(config.data_root)
        dir = mode + 'annot'
        img_dir = os.path.join(data_root, dir)
        msk_dir = os.path.join(data_root, dir)

        if not os.path.isdir(img_dir):
            raise RuntimeError(f'Image directory: {img_dir} does not exist.')
            
        if not os.path.isdir(msk_dir):
            raise RuntimeError(f'Mask directory: {msk_dir} does not exist.')
        
        if mode == 'train':
            self.transform = AT.Compose([
                transforms.Scale(scale=config.scale),
                AT.RandomScale(scale_limit=config.randscale),
                AT.PadIfNeeded(min_height=config.crop_h, min_width=config.crop_w, value=(114,114,114), mask_value=(0,0,0)),
                AT.RandomCrop(height=config.crop_h, width=config.crop_w),
                AT.ColorJitter(brightness=config.brightness, contrast=config.contrast, saturation=config.saturation),
                AT.HorizontalFlip(p=config.h_flip),
                AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),                
            ])
            
        elif mode == 'val':
            self.transform = AT.Compose([
                transforms.Scale(scale=config.scale),
                AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        self.images = []
        self.masks = []
       
        for file_name in os.listdir(img_dir):
            self.images.append(os.path.join(img_dir, file_name))

        for mask_name in os.listdir(msk_dir):     
            self.masks.append(os.path.join(msk_dir, mask_name))    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        mask = np.asarray(Image.open(self.masks[index]).convert('L'))
        
        # Perform augmentation and normalization
        # 执行增强和规范化
        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']
        
        # Encode mask using trainId
        # 使用 trainId 对掩码进行编码 
        return image, mask

    @classmethod
    def encode_target(cls, mask):
        return cls.id_to_train_id[np.array(mask)]    