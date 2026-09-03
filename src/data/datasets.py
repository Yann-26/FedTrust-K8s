"""Dataset utilities for FedTrust-K8s."""

from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_mnist_datasets(data_root="./datasets"):
    """Load MNIST train and test datasets."""
    transform = transforms.ToTensor()
    
    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset


def create_iid_datasets(train_dataset, num_clients=5):
    """Create IID partitions for clients."""
    total_size = len(train_dataset)
    client_size = total_size // num_clients
    
    client_datasets = []
    for client_id in range(num_clients):
        start = client_id * client_size
        end = start + client_size
        indices = list(range(start, end))
        client_datasets.append(Subset(train_dataset, indices))
    
    return client_datasets


def create_non_iid_datasets(train_dataset, num_clients=5):
    """Create non-IID partitions for clients.
    
    Each client gets 2 digit classes:
    Client 0: digits 0, 1
    Client 1: digits 2, 3
    Client 2: digits 4, 5
    Client 3: digits 6, 7
    Client 4: digits 8, 9
    """
    client_labels = {
        0: [0, 1],
        1: [2, 3],
        2: [4, 5],
        3: [6, 7],
        4: [8, 9],
    }
    
    # Group indices by label
    label_indices = {label: [] for label in range(10)}
    for idx, (_, label) in enumerate(train_dataset):
        label_indices[label].append(idx)
    
    # Create client datasets
    client_datasets = []
    for client_id in range(num_clients):
        indices = []
        for label in client_labels[client_id]:
            indices.extend(label_indices[label])
        client_datasets.append(Subset(train_dataset, indices))
    
    return client_datasets


def create_poisoned_dataset(dataset, target_label, poison_fraction, seed=42):
    """Poison a dataset by flipping labels to target class."""
    import random
    random.seed(seed)
    
    # Get all indices
    all_indices = list(range(len(dataset)))
    
    # Separate by current label
    poisoned_indices = []
    clean_indices = []
    for idx in all_indices:
        _, label = dataset[idx]
        if label != target_label:
            poisoned_indices.append(idx)
        else:
            clean_indices.append(idx)
    
    # Determine how many to poison
    num_to_poison = int(len(poisoned_indices) * poison_fraction)
    selected_indices = random.sample(poisoned_indices, num_to_poison)
    
    # Create wrapper that returns poisoned labels
    class PoisonedDataset:
        def __init__(self, original_dataset, poison_indices, target_label):
            self.original_dataset = original_dataset
            self.poison_indices = set(poison_indices)
            self.target_label = target_label
            
        def __getitem__(self, idx):
            data, label = self.original_dataset[idx]
            if idx in self.poison_indices:
                return data, self.target_label
            return data, label
            
        def __len__(self):
            return len(self.original_dataset)
    
    return PoisonedDataset(dataset, selected_indices, target_label)
