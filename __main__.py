import torch.optim as optim
import os

from p1_preprocessing.get_loaders import get_loaders
from p1_preprocessing.download_data import download_data_if_none
from p2_models.models import *
from p2_models.train_test import train_model
from p3_analysis.model_ananlysis import print_statistical_metrics
from web_interface import show_web_interface


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {device} device.')

    # Define model
    model = CheXNet(n_classes=15).to(device)

    # Download data -- 100'000 images in high resolution. Will take significant time and space.
    download_data_if_none()
    train_loader, test_loader = get_loaders()

    if os.getenv('RE-TRAIN') == "True":
        # Define fun stuff for training
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        loss_fn = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        nb_epochs = 1

        # Train the model
        train_model(model, nb_epochs, train_loader, test_loader, optimizer, loss_fn, scheduler, device)
    else:
        model.load_state_dict(torch.load('./model_weights/best_checkpoint.model'))

    # Show model stats
    print_statistical_metrics(model, test_loader, device)

    # Open web interface
    show_web_interface(model, device)
