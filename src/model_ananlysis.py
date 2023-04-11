import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


def get_preds_targets(model, test_loader, device):
    all_predictions = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).long()
        predictions = model(images)
        all_predictions += predictions.argmax(dim=1).cpu().numpy().tolist()
        all_labels += labels.cpu().numpy().tolist()
    return all_predictions, all_labels


def show_confusion_matrix(model, test_loader, device):
    classes = ['Atelectasis', 'Effusion', 'Infiltration', 'No Finding', 'Nodule', 'Pneumothorax']

    all_predictions, all_labels = get_preds_targets(model, test_loader, device)

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmax=1, vmin=0)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def print_statistical_metrics(model, test_loader, device):

    all_predictions, all_labels = get_preds_targets(model, test_loader, device)

    # Rename the input arrays so that they do not clash with any other variables
    y_true = list(all_labels)
    y_pred = list(all_predictions)

    # Calculate micro-averaged metrics
    micro_f1_score = f1_score(y_true, y_pred, average='micro')
    micro_precision = precision_score(y_true, y_pred, average='micro')
    micro_recall = recall_score(y_true, y_pred, average='micro')

    # Calculate macro-averaged metrics
    macro_f1_score = f1_score(y_true, y_pred, average='macro')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    print(
        f'Micro precision: {micro_precision} \nMicro recall: {micro_recall} \nMicro f1 score: {micro_f1_score} \n')
    print(
        f'Macro precision: {macro_precision} \nMacro recall: {macro_recall} \nMacro f1 score: {macro_f1_score} \n')
    print(f'Accuracy: {accuracy}')
