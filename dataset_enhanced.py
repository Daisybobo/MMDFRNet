import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.train = train
        img_dir = os.path.join(root_dir, 'img')
        label_dir = os.path.join(root_dir, 'labels')

        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.tif')])

        if train:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.GridDistortion(p=0.2),
                A.ElasticTransform(p=0.2)
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.img_files)

    def normalize_data(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'img', self.img_files[idx])
        label_path = os.path.join(self.root_dir, 'labels', self.label_files[idx])

        with rasterio.open(img_path) as img_src:
            img = img_src.read()
            img = np.moveaxis(img, 0, -1)

        with rasterio.open(label_path) as label_src:
            label = label_src.read()
            label = np.squeeze(label)

        sar_data = img[:, :, 10:]
        optical_data = img[:, :, :10]

        sar_data = self.normalize_data(sar_data)
        optical_data = self.normalize_data(optical_data)

        if self.transform and self.train:
            augmented = self.transform(image=np.concatenate([optical_data, sar_data], axis=-1), mask=label)
            combined_data = augmented['image']
            label = augmented['mask']
            optical_data = combined_data[:, :, :10]
            sar_data = combined_data[:, :, 10:]

        sar_data = torch.from_numpy(sar_data).float().permute(2, 0, 1)
        optical_data = torch.from_numpy(optical_data).float().permute(2, 0, 1)
        label = torch.from_numpy(label).long()

        return sar_data, optical_data, label


# 创建数据集实例
root_dir = './Rice/test'

dataset = MyDataset(root_dir)
