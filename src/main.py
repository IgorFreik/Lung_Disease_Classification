import torch.cuda
import torch.optim as optim
import torch.nn as nn
from models import BaselineSimple
from data_preparation import download_data, get_loaders
from interface import create_interface
from train_test import train_model
from model_ananlysis import show_confusion_matrix, print_statistical_metrics


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {device} device.')

    # Download data -- 100'000 images in high resolution. Will take significant time and space.
    download_data()
    train_loader, test_loader = get_loaders()

    # Define model
    model = BaselineSimple().to(device)

    # Define fun stuff for training
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    nb_epochs = 1

    # Train the model
    train_model(model, nb_epochs, train_loader, test_loader, optimizer, loss_fn, scheduler, device)

    # Show model stats
    show_confusion_matrix(model, test_loader, device)
    print_statistical_metrics(model, test_loader, device)

    # Open web interface
    create_interface(model, device)
