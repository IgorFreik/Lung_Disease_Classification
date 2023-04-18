from torch.utils.data import Dataset
import torchvision
import os
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, imgs_path, labels_path):
        self.img_f_names = os.listdir(imgs_path)
        self.imgs_path = imgs_path
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.img_f_names)

    def __getitem__(self, idx):
        img = torchvision.io.read_image(os.path.join(self.imgs_path, self.img_f_names[idx]))
        label = self.labels[idx]
        print(img.shape, label)
        return img, label