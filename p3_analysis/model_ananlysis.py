from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error


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


def print_statistical_metrics(model, test_loader, device):

    all_predictions, all_labels = get_preds_targets(model, test_loader, device)

    # Rename the input arrays so that they do not clash with any other variables
    y_true = list(all_labels)
    y_pred = list(all_predictions)

    # Calculate metrics
    f1_score = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f'MAE: {mae}, F1-score: {f1_score}, Accuracy: {accuracy}')
