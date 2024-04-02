import numpy as np
import albumentations as AT
import torch.nn.functional as F


def to_numpy(array):
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    return array    


class Scale:
    def __init__(self, scale, interpolation=1, p=1, is_testing=False):
        self.scale = scale
        self.interpolation = interpolation
        self.p = p
        self.is_testing = is_testing
        
    def __call__(self, image, mask=None):
        img = to_numpy(image)
        if not self.is_testing:
            msk = to_numpy(mask)
        
        imgh, imgw, _ = img.shape
        new_imgh, new_imgw = int(imgh * self.scale), int(imgw * self.scale)
    
        aug = AT.Resize(height=new_imgh, width=new_imgw, interpolation=self.interpolation, p=self.p)
        
        if self.is_testing:
            augmented = aug(image=img)
        else:
            augmented = aug(image=img, mask=msk)
        return augmented


class ResizeToSquare:
    def __init__(self, size, interpolation=1, p=1, is_testing=False):
        self.size = size
        self.interpolation = interpolation
        self.p = p
        self.is_testing = is_testing

    def __call__(self, image, mask=None):
        img = to_numpy(image)

        h, w, _ = img.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = ((vp, vp), (hp, hp), (0, 0))
        img = np.pad(img, padding, mode='constant', constant_values=0)

        if not self.is_testing:
            msk = to_numpy(mask)
            msk_h, msk_w, _ = img.shape
            msk_max_wh = np.max([msk_w, msk_h])
            msk_hp = int((msk_max_wh - msk_w) / 2)
            msk_vp = int((msk_max_wh - msk_h) / 2)
            msk_padding = ((msk_vp, msk_vp), (msk_hp, msk_hp))
            msk = np.pad(msk, msk_padding, mode='constant', constant_values=0)

        aug = AT.Resize(height=self.size, width=self.size, interpolation=self.interpolation, p=self.p)

        if self.is_testing:
            augmented = aug(image=img)
        else:
            augmented = aug(image=img, mask=msk)
        return augmented
