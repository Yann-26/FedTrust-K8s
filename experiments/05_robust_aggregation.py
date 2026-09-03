import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ============================================================
# Configuration
# ============================================================

NUM_CLIENTS = 5
ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.001
BYZANTINE_CLIENT_ID = 4  # Client 5 is Byzantine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model
# ============================================================


class SimpleCNN(nn.Module):
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

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./datasets", train=True, download=True, transform=transform
)

test_dataset = datasets.MNIST(
    root="./datasets", train=False, download=True, transform=transform
)


# ============================================================
# Non-IID Dataset Partition
# ============================================================

# Group MNIST samples by their labels.
label_indices = {label: [] for label in range(10)}

for index, (_, label) in enumerate(train_dataset):
    label_indices[label].append(index)


# Each client receives two digit classes.
client_labels = {
    0: [0, 1],
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
    4: [8, 9],
}

client_datasets = []

for client_id in range(NUM_CLIENTS):
    indices = []
    for label in client_labels[client_id]:
        indices.extend(label_indices[label])

    client_dataset = Subset(train_dataset, indices)
    client_datasets.append(client_dataset)


# ============================================================
# Client training
# ============================================================


def train_client(global_model, client_dataset):
    local_model = copy.deepcopy(global_model)
    local_model.to(DEVICE)
    local_model.train()

    loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

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

    return local_model.state_dict()


# ============================================================
# Byzantine Attack
# ============================================================


def byzantine_attack(model_weights):
    """Simulates a Byzantine client by flipping the sign of gradients."""
    attacked_weights = {}
    for key in model_weights.keys():
        attacked_weights[key] = -model_weights[key]
    return attacked_weights


# ============================================================
# Aggregation Methods
# ============================================================


def fedavg(client_models):
    """Standard FedAvg - average all models."""
    global_model = copy.deepcopy(client_models[0])

    for key in global_model.keys():
        global_model[key] = torch.stack(
            [client_model[key].float() for client_model in client_models], dim=0
        ).mean(dim=0)

    return global_model


def krum(client_models, num_clients_to_select=1):
    """
    Krum: Select the model most consistent with others.
    Assumes at most 1 Byzantine client (f=1).
    """
    f = 1
    num_neighbors = len(client_models) - f - 2

    if num_neighbors < 1:
        num_neighbors = 1

    # Convert state_dicts to flat vectors
    flat_models = []
    for model in client_models:
        flat = []
        for key in model.keys():
            flat.append(model[key].flatten())
        flat_models.append(torch.cat(flat))

    # Compute distances
    num_clients = len(client_models)
    distances = torch.zeros((num_clients, num_clients))

    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = torch.norm(flat_models[i] - flat_models[j], p=2)
            distances[i, j] = dist
            distances[j, i] = dist

    # Compute scores (sum of distances to nearest neighbors)
    scores = []
    for i in range(num_clients):
        dist_to_others = distances[i, :]
        sorted_indices = torch.argsort(dist_to_others)
        nearest_indices = sorted_indices[1 : num_neighbors + 1]
        score = dist_to_others[nearest_indices].sum()
        scores.append(score)

    best_idx = torch.argmin(torch.tensor(scores)).item()
    return client_models[best_idx]


def trimmed_mean(client_models, trim_ratio=0.2):
    """
    Trimmed Mean: Remove extreme values and average the rest.
    trim_ratio: fraction of values to remove from each end.
    """
    global_model = copy.deepcopy(client_models[0])

    for key in global_model.keys():
        # Stack all client parameters for this layer
        stacked = torch.stack(
            [client_model[key].float() for client_model in client_models], dim=0
        )

        # Sort along the client dimension
        sorted_values, _ = torch.sort(stacked, dim=0)

        # Calculate how many to trim from each end
        n = len(client_models)
        trim_count = int(n * trim_ratio)

        if trim_count > 0:
            # Remove extremes and average the rest
            trimmed = sorted_values[trim_count : n - trim_count]
            global_model[key] = trimmed.mean(dim=0)
        else:
            global_model[key] = stacked.mean(dim=0)

    return global_model


def coordinate_median(client_models):
    """Coordinate-wise Median: Take median for each parameter."""
    global_model = copy.deepcopy(client_models[0])

    for key in global_model.keys():
        stacked = torch.stack(
            [client_model[key].float() for client_model in client_models], dim=0
        )
        # Take median along the client dimension
        global_model[key] = torch.median(stacked, dim=0)[0]

    return global_model


def geometric_median(client_models, max_iter=100, tol=1e-5):
    """
    Geometric Median: Find point minimizing sum of Euclidean distances.
    Uses Weiszfeld algorithm.
    """
    # Convert state_dicts to flat vectors
    flat_models = []
    for model in client_models:
        flat = []
        for key in model.keys():
            flat.append(model[key].flatten())
        flat_models.append(torch.cat(flat))

    # Stack into matrix
    X = torch.stack(flat_models, dim=0)  # Shape: (n_clients, n_params)
    n_clients, n_params = X.shape

    # Initialize with mean
    y = X.mean(dim=0)

    for _ in range(max_iter):
        # Compute distances
        distances = torch.norm(X - y, dim=1)

        # Handle zero distances
        mask = distances < tol
        if mask.any():
            # If point is exactly at a data point, return it
            return client_models[mask.nonzero()[0].item()]

        # Weiszfeld update
        weights = 1.0 / distances
        weights = weights / weights.sum()
        y_new = (weights.unsqueeze(1) * X).sum(dim=0)

        if torch.norm(y_new - y) < tol:
            break

        y = y_new

    # Reconstruct state_dict from flat vector
    global_model = copy.deepcopy(client_models[0])
    idx = 0
    for key in global_model.keys():
        param_size = global_model[key].numel()
        global_model[key] = y[idx : idx + param_size].reshape(global_model[key].shape)
        idx += param_size

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

    return 100 * correct / total


# ============================================================
# Helper function to run a single aggregation method
# ============================================================


def run_experiment(aggregation_method, method_name):
    """Run federated training with a specific aggregation method."""

    print(f"\n{'=' * 65}")
    print(f"Testing: {method_name}")
    print("=" * 65)

    global_model = SimpleCNN()
    round_accuracies = []

    for round_number in range(1, ROUNDS + 1):
        print(f"\n--- Round {round_number}/{ROUNDS} ---")

        client_models = []

        for client_id in range(NUM_CLIENTS):
            is_byzantine = client_id == BYZANTINE_CLIENT_ID

            if is_byzantine:
                print(f"⚠️  Client {client_id + 1} (BYZANTINE) preparing attack...")
            else:
                print(f"Training Client {client_id + 1}...")

            local_weights = train_client(global_model, client_datasets[client_id])

            if is_byzantine:
                print(f"💀 Client {client_id + 1} performing Byzantine attack!")
                local_weights = byzantine_attack(local_weights)

            client_models.append(local_weights)

        # Aggregate using the specified method
        global_weights = aggregation_method(client_models)
        global_model.load_state_dict(global_weights)

        accuracy = evaluate(global_model)
        round_accuracies.append(accuracy)
        print(f"Global model accuracy: {accuracy:.2f}%")

    return round_accuracies


# ============================================================
# Main Experiment
# ============================================================

print("=" * 65)
print("FedTrust-K8s — Experiment 05: Robust Aggregation Comparison")
print("=" * 65)
print(f"Clients          : {NUM_CLIENTS}")
print(f"Federated rounds : {ROUNDS}")
print(f"Local epochs     : {LOCAL_EPOCHS}")
print(f"Device           : {DEVICE}")
print("=" * 65)
print("Client label distributions:")
print("Client 1 → digits 0, 1")
print("Client 2 → digits 2, 3")
print("Client 3 → digits 4, 5")
print("Client 4 → digits 6, 7")
print("Client 5 → digits 8, 9 (BYZANTINE)")
print("=" * 65)

# Dictionary of aggregation methods to test
aggregation_methods = {
    "FedAvg": fedavg,
    "Krum": krum,
    "Trimmed Mean (20%)": trimmed_mean,
    "Coordinate Median": coordinate_median,
    "Geometric Median": geometric_median,
}

# Store all results
all_results = {}

# Run each method
for method_name, method_func in aggregation_methods.items():
    accuracies = run_experiment(method_func, method_name)
    all_results[method_name] = accuracies

# ============================================================
# Results Summary
# ============================================================

print("\n" + "=" * 65)
print("SUMMARY: All Aggregation Methods Compared")
print("=" * 65)

# Create header
print(
    f"\n{'Method':<20} {'R1':<8} {'R2':<8} {'R3':<8} {'R4':<8} {'R5':<8} {'Final':<8}"
)
print("-" * 70)

# Print results for each method
for method_name, accuracies in all_results.items():
    final_acc = accuracies[-1]
    print(
        f"{method_name:<20} "
        f"{accuracies[0]:.2f}%   "
        f"{accuracies[1]:.2f}%   "
        f"{accuracies[2]:.2f}%   "
        f"{accuracies[3]:.2f}%   "
        f"{accuracies[4]:.2f}%   "
        f"{final_acc:.2f}%"
    )

print("\n" + "=" * 65)
print("Experiment completed!")
print("=" * 65)
