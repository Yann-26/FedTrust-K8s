"""Evaluation metrics for federated learning."""

import torch
from torch.utils.data import DataLoader


def evaluate_accuracy(model, test_dataset, device, batch_size=64):
    """Evaluate model accuracy on test dataset."""
    model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return 100 * correct / total


def compute_confusion_matrix(model, test_dataset, device, batch_size=64):
    """Compute confusion matrix for model predictions."""
    model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels
