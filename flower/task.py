import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================================================
# Configuration
# ============================================================

NUM_CLIENTS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOCAL_EPOCHS = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model
# ============================================================


class SimpleCNN(nn.Module):
    """Simple CNN used in FedTrust-K8s experiments."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# Dataset
# ============================================================


def load_datasets(data_root="./datasets"):
    """Load MNIST training and test datasets."""

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


# ============================================================
# IID partitioning
# ============================================================


def create_iid_datasets(train_dataset, num_clients=NUM_CLIENTS):
    """
    Reproduce the IID partition used in Experiment 01.

    Each client receives an equal contiguous partition
    of the MNIST training dataset.
    """

    client_size = len(train_dataset) // num_clients

    client_datasets = []

    for client_id in range(num_clients):
        start = client_id * client_size
        end = start + client_size

        indices = list(range(start, end))

        client_dataset = Subset(
            train_dataset,
            indices,
        )

        client_datasets.append(client_dataset)

    return client_datasets


# ============================================================
# Local training
# ============================================================


def train_model(
    model,
    trainloader,
    epochs=LOCAL_EPOCHS,
    learning_rate=LEARNING_RATE,
):
    """Train a model locally."""

    model.to(DEVICE)
    model.train()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    batches = 0

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            batches += 1

    average_loss = total_loss / batches

    return average_loss


# ============================================================
# Evaluation
# ============================================================


def evaluate_model(model, testloader):
    """Evaluate a model on the global MNIST test set."""

    model.to(DEVICE)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (predictions == labels).sum().item()

    accuracy = correct / total

    average_loss = total_loss / len(testloader)

    return average_loss, accuracy


# ============================================================
# Data loading helper
# ============================================================


def get_client_loaders(
    client_id,
    num_clients=NUM_CLIENTS,
    batch_size=BATCH_SIZE,
):
    """Return the local DataLoader for a Flower client."""

    train_dataset, _ = load_datasets()

    client_datasets = create_iid_datasets(
        train_dataset,
        num_clients,
    )

    trainloader = DataLoader(
        client_datasets[client_id],
        batch_size=batch_size,
        shuffle=True,
    )

    return trainloader


def get_testloader(batch_size=BATCH_SIZE):
    """Return the global MNIST test DataLoader."""

    _, test_dataset = load_datasets()

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
