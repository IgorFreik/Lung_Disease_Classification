from torch.utils.data import DataLoader
from p1_preprocessing.image_dataset import ImageDataset
import torch


def get_loaders(train_batch_size=64, test_batch_size=32):
    IMGS_PATH = './data/images'
    LABELS_PATH = './data/labels.npy'

    full_dataset = ImageDataset(IMGS_PATH, LABELS_PATH)
    train_size = int(0.8 * len(full_dataset))

    train_data = torch.utils.data.Subset(full_dataset, range(train_size))
    test_data = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))

    # train_data.dataset.transform = tt.Compose([
    #     tt.Resize((224, 224)),
    #     tt.RandomHorizontalFlip(p=0.5),
    #     tt.RandomRotation(degrees=10),
    #     tt.ToTensor(),
    #     tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # ])
    #
    # test_data.dataset.transform = tt.Compose([
    #     tt.Resize((224, 224)),
    #     tt.ToTensor(),
    #     tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # ])

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader
