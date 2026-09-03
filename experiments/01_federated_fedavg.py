import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ============================================================
# Configuration
# ============================================================

NUM_CLIENTS = 5  # number of clients participating in the federated learning process
ROUNDS = 5  # number of federated learning rounds (iterations) to perform

LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model
# ============================================================


# A simple convolutional neural network (CNN) for image classification on the MNIST dataset
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # Convolutional layer with 1 input channel (grayscale), 16 output channels, kernel size of 3, and padding of 1
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

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./datasets", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root="./datasets", train=False, download=True, transform=transform
)


# ============================================================
# Split training data between clients
# ============================================================

client_size = (
    len(train_dataset) // NUM_CLIENTS
)  # Each client will receive an equal portion of the training dataset

client_datasets = []

for client_id in range(NUM_CLIENTS):
    start = client_id * client_size

    end = start + client_size

    indices = list(range(start, end))

    client_dataset = Subset(train_dataset, indices)

    client_datasets.append(client_dataset)


# ============================================================
# Client training
# ============================================================


# Each client trains its local model on its own dataset and returns the updated model weights
def train_client(global_model, client_dataset):
    # Create a local model for the client by copying the global model

    local_model = copy.deepcopy(
        global_model
    )  # Each client starts with a copy of the current global model

    local_model.to(DEVICE)

    local_model.train()

    loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # The optimizer is responsible for updating the model's parameters based on the computed gradients
    optimizer = optim.Adam(local_model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(LOCAL_EPOCHS):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = local_model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            # each client produces its own local model

    return local_model.state_dict()  # returns the model's learned parameters.


# ============================================================
# FedAvg
# ============================================================


# The FedAvg algorithm aggregates the model weights from multiple clients by averaging them to create a new global model
def fedavg(client_models):  # takes the models from the five clients

    # Create a new global model by averaging the weights of the client models
    global_model = copy.deepcopy(client_models[0])

    for key in global_model.keys():  # Iterate over each parameter in the model's state dictionary (weights and biases)
        global_model[key] = (
            torch.stack(  # puts the corresponding parameters together
                # Stack the corresponding parameters from all client models
                # along a new dimension and compute the mean to obtain the averaged parameter for the global model
                [client_model[key].float() for client_model in client_models],
                dim=0,
            ).mean(dim=0) # calculates their average 
        )

    return global_model


# ============================================================
# Evaluation
# ============================================================


def evaluate(model):

    model.to(DEVICE)

    model.eval()

    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (predictions == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


# ============================================================
# Federated Training
# ============================================================

print("=" * 65)

print("FedTrust-K8s — Experiment 01: Federated Learning")

print("=" * 65)

print(f"Clients          : {NUM_CLIENTS}")

print(f"Federated rounds : {ROUNDS}")

print(f"Local epochs     : {LOCAL_EPOCHS}")

print(f"Device           : {DEVICE}")

print("=" * 65)


global_model = SimpleCNN()


for round_number in range(1, ROUNDS + 1):
    print(f"\n--- Federated Round {round_number}/{ROUNDS} ---")

    client_models = []

    for client_id in range(NUM_CLIENTS):
        print(f"Training Client {client_id + 1}...")

        local_weights = train_client(global_model, client_datasets[client_id])

        client_models.append(local_weights)

    global_weights = fedavg(client_models)

    global_model.load_state_dict(global_weights)

    accuracy = evaluate(global_model)

    print(f"Global model accuracy: {accuracy:.2f}%")


print("\n" + "=" * 65)

print("Federated training completed.")

print("=" * 65)
