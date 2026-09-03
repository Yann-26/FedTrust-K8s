"""Shared utilities for FedTrust-K8s."""

import copy
import torch
import random
import numpy as np


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get available device (GPU/CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_header(title, char="=", width=65):
    """Print a formatted header."""
    print(char * width)
    print(title.center(width))
    print(char * width)


def print_config(config):
    """Print configuration dictionary."""
    print_header("Configuration", "-", 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 50)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
