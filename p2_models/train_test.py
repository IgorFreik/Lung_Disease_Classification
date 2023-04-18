from tqdm import tqdm
import torch
from p2_models.models import *
from typing import Callable, Iterable
import torch.nn as nn
from sklearn.metrics import f1_score
import os
from pathlib import Path
import matplotlib.pyplot as plt
import plotext
import datetime


def train_step(
        model: nn.Module,
        train_sampler: Iterable,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
):
    train_accuracy = 0.0
    losses = []
    model.train()

    y_true = []
    y_pred = []

    for (imgs, targets) in tqdm(train_sampler):
        imgs, targets = imgs.to(device), targets.type(torch.LongTensor).to(device)

        output = model(imgs)
        loss = loss_function(output, targets)
        losses.append(loss)

        with torch.no_grad():
            predictions = torch.argmax(output, dim=1)
            train_accuracy += (predictions == targets).float().mean()

            # Append true and predicted labels
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return losses, train_accuracy, macro_f1


def test_step(
        model: nn.Module,
        test_sampler: Iterable,
        loss_function: Callable[..., torch.Tensor],
        device: str,
):
    model.eval()
    losses = []
    test_accuracy = 0.0
    with torch.no_grad():
        y_true = []
        y_pred = []
        for (imgs, targets) in tqdm(test_sampler):
            imgs, targets = imgs.to(device), targets.type(torch.LongTensor).to(device)
            output = model.forward(imgs)
            predictions = torch.argmax(output, dim=1)
            test_accuracy += (predictions == targets).float().mean()

            # Append true and predicted labels
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

            loss = loss_function(output, targets)
            losses.append(loss)

    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return losses, test_accuracy, macro_f1


def train_model(model, n_epochs, train_sampler, test_sampler, optimizer, loss_function, scheduler, device):
    best_f1 = 0.0
    mean_losses_train = []
    mean_losses_test = []

    for e in range(n_epochs):
        # Training:
        losses, train_acc, train_macro_f1 = train_step(model, train_sampler, optimizer, loss_function, device)
        # Calculating and printing statistics:
        train_acc /= len(train_sampler)
        mean_loss = sum(losses) / len(losses)
        mean_losses_train.append(mean_loss)
        print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}, macro f1 on train set: {train_macro_f1:.4f}\n")

        # Testing:
        losses, test_acc, test_macro_f1 = test_step(model, test_sampler, loss_function, device)
        # # Calculating and printing statistics:
        test_acc /= len(test_sampler)
        mean_loss = sum(losses) / len(losses)
        mean_losses_test.append(mean_loss)
        print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}, macro f1 on test set: {test_macro_f1:.4f}\n")

        if scheduler is not None:
            scheduler.step(mean_loss)

        # Save the best model
        if test_macro_f1 > best_f1:
            torch.save(model.state_dict(), 'model_weights/best_checkpoint.model')
            best_f1 = test_macro_f1

        # Plotting during training
        plotext.clf()
        plotext.scatter(mean_losses_train, label="train")
        plotext.scatter(mean_losses_test, label="test")
        plotext.title("Train and test loss")

        plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

        plotext.show()

    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists

    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

    # Create plot of losses
    plt.figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")
    fig.legend()

    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")
