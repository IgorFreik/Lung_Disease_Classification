import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision
import os
import numpy as np
import torch

class ImageDataset(Dataset):
    def __init__(self, imgs_path, labels_path, transforms):
        self.img_f_names = os.listdir(imgs_path)
        self.imgs_path = imgs_path
        self.labels = np.load(labels_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.img_f_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.imgs_path, self.img_f_names[idx])
        img = torchvision.io.read_image(img_path) / 255
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        else:
            img = img[:3]
        img = self.transforms(img)
        label = self.labels[idx]
        return img, label
