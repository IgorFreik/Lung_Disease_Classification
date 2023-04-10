import torch.cuda
import torch.optim as optim
import torch.nn as nn
from models import BaselineSimple
from data_preparation import download_data
from interface import create_interface
from train_test import train_model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {device} device.')

    # Download data -- 100'000 images in high resolution. Will take significant time and space.
    download_data()

    # Define model
    model = BaselineSimple().to(device)

    # Define things for training
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Train the model
    train_model(model, 1, None, None, optimizer, loss_fn, scheduler, device)

    # Show model stats
    ...

    # Open web interface
    create_interface(model, device)

