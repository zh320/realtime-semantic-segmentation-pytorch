import numpy as np
import albumentations as AT


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