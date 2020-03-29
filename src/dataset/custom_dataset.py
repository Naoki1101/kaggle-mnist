import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as album


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(album, transform.name):
            return getattr(album, transform.name)
        else:
            return eval(transform.name)
    if cfg.transforms:
        transforms = [get_object(transform)(**transform.params) for name, transform in cfg.transforms.items()]
        return album.Compose(transforms)
    else:
        return None


class CustomDataset(Dataset):
    
    def __init__(self, df, labels, cfg, is_train=True):
        self.cfg = cfg
        self.data = df.values
        self.labels = labels.values
        self.n_channels = cfg.n_channels
        self.transforms = get_transforms(self.cfg)
        self.is_train = is_train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx, :].reshape(28, 28).astype(np.uint8)
        image = (image*(255.0/image.max())).astype(np.uint8)
        image = cv2.resize(image, dsize=(self.cfg.img_size.height, self.cfg.img_size.width))
        if self.transforms:
            image = self.transforms(image=image)['image']
        image = image.reshape(1, self.cfg.img_size.width, self.cfg.img_size.height).astype(np.float32)
        if self.n_channels == 3:
            image = np.concatenate([image for i in range(self.n_channels)], axis=0)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            return image