import sys
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as album

sys.path.append("./src")
import const


def get_transforms(cfg):
    def get_object(transform):
        if hasattr(album, transform.name):
            return getattr(album, transform.name)
        else:
            return eval(transform.name)

    if cfg.transforms:
        transforms = [
            get_object(transform)(**transform.params)
            for name, transform in cfg.transforms.items()
        ]
        return album.Compose(transforms)
    else:
        return None


class CustomDataset(Dataset):
    def __init__(self, df, target_df, cfg):
        self.cfg = cfg
        self.data = df[const.PIXEL_COLS].values
        self.transforms = get_transforms(self.cfg)
        self.is_train = False

        if target_df is not None:
            self.labels = df[const.TARGET_COL].values
            self.is_train = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(28, 28).astype(np.uint8)
        image = (image * (255.0 / image.max())).astype(np.uint8)
        image = cv2.resize(
            image, dsize=(self.cfg.img_size.height, self.cfg.img_size.width)
        )
        if self.transforms:
            image = self.transforms(image=image)["image"]

        image = image[np.newaxis, :, :].astype(np.float32)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            return image
