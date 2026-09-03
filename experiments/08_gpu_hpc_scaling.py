"""Experiment 08: GPU / HPC Scaling for Federated Learning.

This experiment benchmarks and scales federated learning:
1. CPU vs GPU performance comparison
2. Scaling to larger models
3. Throughput and training time analysis
4. Memory usage profiling
"""

import sys
sys.path.append('.')

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import psutil
import gc

# Import from src modules
from src.models import SimpleCNN
from src.data import get_mnist_datasets, create_non_iid_datasets
from src.evaluation import evaluate_accuracy
from src.aggregation import fedavg
from src.utils import get_device, print_header, set_seed


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    'num_clients': 5,
    'rounds': 3,
    'local_epochs': 1,
    'batch_sizes': [32, 64, 128, 256],  # For scaling tests
    'learning_rate': 0.001,
    'seed': 42,
}

set_seed(CONFIG['seed'])


# ============================================================
# Larger Model for Scaling Tests
# ============================================================

class ResNetStyleCNN(nn.Module):
    """Larger CNN model for scaling tests."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


# ============================================================
# Model Factories
# ============================================================

def get_model(model_type="small"):
    """Get model by type."""
    if model_type == "small":
        return SimpleCNN()
    elif model_type == "medium":
        return ResNetStyleCNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Performance Benchmarking
# ============================================================

class PerformanceBenchmark:
    """Benchmark federated learning performance."""
    
    def __init__(self, device, model_type="small"):
        self.device = device
        self.model_type = model_type
        self.results = defaultdict(list)
        
    def benchmark_training(self, train_dataset, test_dataset, client_datasets, batch_size=64):
        """Benchmark training time and throughput."""
        print(f"\n{'='*65}")
        print(f"Benchmarking: {self.model_type.upper()} model on {self.device}")
        print(f"Batch size: {batch_size}")
        print('='*65)
        
        # Initialize model
        global_model = get_model(self.model_type).to(self.device)
        
        # Track metrics
        times = []
        memory_usage = []
        throughputs = []
        
        for round_num in range(CONFIG['rounds']):
            round_start = time.time()
            client_models = []
            
            for client_id in range(CONFIG['num_clients']):
                # Monitor memory before training
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated(self.device)
                
                # Train client
                local_model = self._train_client(
                    global_model,
                    client_datasets[client_id],
                    batch_size
                )
                client_models.append(local_model)
                
                # Monitor memory after training
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated(self.device)
                    memory_usage.append((memory_after - memory_before) / 1024**2)  # MB
            
            # Aggregate
            global_weights = fedavg(client_models)
            global_model.load_state_dict(global_weights)
            
            round_time = time.time() - round_start
            times.append(round_time)
            
            # Calculate throughput (images per second)
            total_images = CONFIG['num_clients'] * batch_size * CONFIG['local_epochs']
            throughput = total_images / round_time
            throughputs.append(throughput)
            
            # Evaluate
            accuracy = evaluate_accuracy(global_model, test_dataset, self.device)
            print(f"Round {round_num + 1}: {round_time:.2f}s, {throughput:.1f} img/s, Acc: {accuracy:.2f}%")
        
        # Store results
        self.results['times'].extend(times)
        self.results['throughputs'].extend(throughputs)
        self.results['memory_usage'] = memory_usage
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_throughput': np.mean(throughputs),
            'std_throughput': np.std(throughputs),
            'avg_memory': np.mean(memory_usage) if memory_usage else 0,
            'total_parameters': count_parameters(global_model),
        }
    
    def _train_client(self, global_model, client_dataset, batch_size):
        """Train a single client."""
        local_model = copy.deepcopy(global_model)
        local_model.train()
        
        loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        optimizer = optim.Adam(local_model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(CONFIG['local_epochs']):
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        return local_model.state_dict()


# ============================================================
# System Information
# ============================================================

def get_system_info(device):
    """Get system information."""
    info = {
        'device': str(device),
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_total': psutil.virtual_memory().total / 1024**3,  # GB
        'memory_available': psutil.virtual_memory().available / 1024**3,  # GB
    }
    
    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info['gpu_memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3
        info['gpu_memory_cached'] = torch.cuda.memory_reserved(0) / 1024**3
    
    return info


# ============================================================
# Visualization
# ============================================================

def visualize_results(cpu_results, gpu_results):
    """Visualize benchmarking results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training Time
    ax1 = axes[0, 0]
    models = ['Small CNN', 'ResNet-Style']
    cpu_times = [cpu_results['small']['avg_time'], cpu_results['medium']['avg_time']]
    gpu_times = [gpu_results['small']['avg_time'], gpu_results['medium']['avg_time']]
    
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, cpu_times, width, label='CPU', color='blue', alpha=0.7)
    ax1.bar(x + width/2, gpu_times, width, label='GPU', color='red', alpha=0.7)
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Throughput (Images/Second)
    ax2 = axes[0, 1]
    cpu_throughput = [cpu_results['small']['avg_throughput'], cpu_results['medium']['avg_throughput']]
    gpu_throughput = [gpu_results['small']['avg_throughput'], gpu_results['medium']['avg_throughput']]
    
    ax2.bar(x - width/2, cpu_throughput, width, label='CPU', color='blue', alpha=0.7)
    ax2.bar(x + width/2, gpu_throughput, width, label='GPU', color='red', alpha=0.7)
    ax2.set_ylabel('Throughput (images/sec)')
    ax2.set_title('Throughput Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameters vs Time
    ax3 = axes[1, 0]
    params = [cpu_results['small']['total_parameters'], cpu_results['medium']['total_parameters']]
    cpu_times = [cpu_results['small']['avg_time'], cpu_results['medium']['avg_time']]
    gpu_times = [gpu_results['small']['avg_time'], gpu_results['medium']['avg_time']]
    
    ax3.plot(params, cpu_times, 'bo-', label='CPU', linewidth=2, markersize=8)
    ax3.plot(params, gpu_times, 'ro-', label='GPU', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Parameters')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Scaling: Parameters vs Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Memory Usage
    ax4 = axes[1, 1]
    if 'avg_memory' in cpu_results['small']:
        cpu_memory = [cpu_results['small']['avg_memory'], cpu_results['medium']['avg_memory']]
        gpu_memory = [gpu_results['small']['avg_memory'], gpu_results['medium']['avg_memory']]
        
        ax4.bar(x - width/2, cpu_memory, width, label='CPU', color='blue', alpha=0.7)
        ax4.bar(x + width/2, gpu_memory, width, label='GPU', color='red', alpha=0.7)
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Memory Usage Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = "results/figures/gpu_scaling_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {fig_path}")
    
    return fig


# ============================================================
# Main Experiment
# ============================================================

def main():
    """Run the complete GPU/HPC scaling experiment."""
    device = get_device()
    print_header("FedTrust-K8s — Experiment 08: GPU / HPC Scaling")
    
    # Get system info
    print("\nSystem Information:")
    print("-" * 50)
    sys_info = get_system_info(device)
    for key, value in sys_info.items():
        print(f"{key}: {value}")
    print("-" * 50)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset, test_dataset = get_mnist_datasets()
    client_datasets = create_non_iid_datasets(train_dataset, CONFIG['num_clients'])
    
    # Model types to test
    model_types = ["small", "medium"]
    model_names = {
        "small": "SimpleCNN",
        "medium": "ResNetStyleCNN"
    }
    
    results = {
        'cpu': {},
        'gpu': {}
    }
    
    # Test on CPU
    print("\n" + "="*65)
    print("Testing on CPU")
    print("="*65)
    cpu_device = torch.device("cpu")
    
    for model_type in model_types:
        benchmark = PerformanceBenchmark(cpu_device, model_type)
        result = benchmark.benchmark_training(
            train_dataset,
            test_dataset,
            client_datasets,
            batch_size=CONFIG['batch_sizes'][0]
        )
        results['cpu'][model_type] = result
        print(f"\n{model_names[model_type]} Results:")
        print(f"  Avg Time: {result['avg_time']:.2f}s")
        print(f"  Avg Throughput: {result['avg_throughput']:.1f} img/s")
        print(f"  Parameters: {result['total_parameters']:,}")
    
    # Test on GPU (if available)
    if torch.cuda.is_available():
        print("\n" + "="*65)
        print("Testing on GPU")
        print("="*65)
        gpu_device = torch.device("cuda")
        
        for model_type in model_types:
            benchmark = PerformanceBenchmark(gpu_device, model_type)
            result = benchmark.benchmark_training(
                train_dataset,
                test_dataset,
                client_datasets,
                batch_size=CONFIG['batch_sizes'][0]
            )
            results['gpu'][model_type] = result
            print(f"\n{model_names[model_type]} Results:")
            print(f"  Avg Time: {result['avg_time']:.2f}s")
            print(f"  Avg Throughput: {result['avg_throughput']:.1f} img/s")
            print(f"  Parameters: {result['total_parameters']:,}")
    else:
        print("\n⚠️  No GPU detected. GPU tests skipped.")
        results['gpu'] = results['cpu']  # Fallback
    
    # Visualize results
    print("\n" + "="*65)
    print("Visualizing Results")
    print("="*65)
    fig = visualize_results(results['cpu'], results['gpu'])
    
    # Summary
    print("\n" + "="*65)
    print("EXPERIMENT 08 COMPLETE")
    print("="*65)
    
    print("\nKey Findings:")
    
    # Speedup calculation
    if torch.cuda.is_available():
        for model_type in model_types:
            cpu_time = results['cpu'][model_type]['avg_time']
            gpu_time = results['gpu'][model_type]['avg_time']
            speedup = cpu_time / gpu_time
            print(f"  • {model_names[model_type]}: {speedup:.1f}x speedup on GPU")
    
    print(f"  • Small model parameters: {results['cpu']['small']['total_parameters']:,}")
    print(f"  • Medium model parameters: {results['cpu']['medium']['total_parameters']:,}")
    print(f"  • CPU throughput: {results['cpu']['small']['avg_throughput']:.1f} img/s")
    
    if torch.cuda.is_available():
        print(f"  • GPU throughput: {results['gpu']['small']['avg_throughput']:.1f} img/s")
    
    print("\nRecommendations:")
    print("  • Use GPU for large models and datasets")
    print("  • Use batch size 128-256 for optimal GPU utilization")
    print("  • Monitor memory usage for large models")
    
    print("\n" + "="*65)
    print("✅ Experiment 08 completed successfully!")
    print("="*65)


if __name__ == "__main__":
    main()
