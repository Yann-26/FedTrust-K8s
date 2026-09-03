"""Experiment 06: Differential Privacy using modular structure."""

import sys
sys.path.append('.')

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from src modules
from src.models import SimpleCNN
from src.data import get_mnist_datasets, create_non_iid_datasets
from src.evaluation import evaluate_accuracy
from src.privacy import DPTrainer, compute_privacy_budget
from src.aggregation import fedavg
from src.utils import get_device, print_header, print_config, set_seed

# ============================================================
# Configuration
# ============================================================

CONFIG = {
    'num_clients': 5,
    'rounds': 5,
    'local_epochs': 1,
    'batch_size': 64,
    'learning_rate': 0.001,
    'dp_enabled': True,
    'sensitivity': 1.0,
    'noise_multipliers': [0.0, 0.5, 1.0, 2.0],
    'delta': 1e-5,
    'seed': 42
}

device = get_device()
set_seed(CONFIG['seed'])


# ============================================================
# Client Training Function
# ============================================================

def train_client(global_model, client_dataset, config, dp_trainer):
    """Train a client with optional Differential Privacy."""
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
    
    loader = DataLoader(
        client_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    optimizer = optim.Adam(
        local_model.parameters(),
        lr=config['learning_rate']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(config['local_epochs']):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Apply Differential Privacy
            dp_trainer.apply(local_model)
            
            optimizer.step()
    
    return local_model.state_dict()


# ============================================================
# Main Experiment
# ============================================================

print_header("FedTrust-K8s — Experiment 06: Differential Privacy")
print_config(CONFIG)

# Load datasets
train_dataset, test_dataset = get_mnist_datasets()
client_datasets = create_non_iid_datasets(train_dataset, CONFIG['num_clients'])

print("\nClient label distributions:")
for i in range(CONFIG['num_clients']):
    labels = []
    for idx in range(len(client_datasets[i])):
        _, label = client_datasets[i][idx]
        labels.append(label)
    unique_labels = sorted(set(labels))
    print(f"Client {i+1} → digits {', '.join(map(str, unique_labels))}")

print("=" * 65)

# Run with different noise multipliers
results = {}

for noise_mult in CONFIG['noise_multipliers']:
    print(f"\n{'='*65}")
    print(f"Testing DP with noise multiplier: {noise_mult}")
    print('='*65)
    
    dp_trainer = DPTrainer(
        enabled=(noise_mult > 0),
        sensitivity=CONFIG['sensitivity'],
        noise_multiplier=noise_mult
    )
    
    global_model = SimpleCNN()
    accuracies = []
    
    for round_num in range(1, CONFIG['rounds'] + 1):
        print(f"\n--- Round {round_num}/{CONFIG['rounds']} ---")
        
        client_models = []
        
        for client_id in range(CONFIG['num_clients']):
            print(f"Training Client {client_id + 1}...")
            
            local_weights = train_client(
                global_model,
                client_datasets[client_id],
                CONFIG,
                dp_trainer
            )
            client_models.append(local_weights)
        
        # Aggregate
        global_weights = fedavg(client_models)
        global_model.load_state_dict(global_weights)
        
        # Evaluate
        accuracy = evaluate_accuracy(global_model, test_dataset, device)
        accuracies.append(accuracy)
        print(f"Global model accuracy: {accuracy:.2f}%")
    
    results[noise_mult] = accuracies

# ============================================================
# Results Summary
# ============================================================

print("\n" + "=" * 65)
print("SUMMARY: Differential Privacy Results")
print("=" * 65)

print("\nPrivacy-Utility Tradeoff:")
print(f"{'Noise (σ)':<12} {'Privacy Level':<20} {'Final Accuracy':<15}")
print("-" * 50)

for noise_mult, accuracies in results.items():
    if noise_mult == 0:
        privacy = "No privacy"
    elif noise_mult <= 0.5:
        privacy = "Low privacy"
    elif noise_mult <= 1.0:
        privacy = "Medium privacy"
    else:
        privacy = "High privacy"
    
    final_acc = accuracies[-1]
    print(f"{noise_mult:<12} {privacy:<20} {final_acc:.2f}%")

print("\n" + "=" * 65)
print("Experiment completed!")
print("=" * 65)
