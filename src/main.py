import torch.cuda
import torch.optim as optim
import torch.nn as nn
from models import BaselineSimple


if __name__ == '__main__':
    # Download data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {device} device.')
    ...

    # Set model
    model = BaselineSimple().to(device)

    # Train the model
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Open the interface & show the stats
    ...
