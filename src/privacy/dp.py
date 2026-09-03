"""Differential Privacy utilities for federated learning."""

import torch
import numpy as np


def clip_gradients(model, max_norm=1.0):
    """Clip gradients to bound sensitivity."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def add_gaussian_noise(model, noise_multiplier=0.5, sensitivity=1.0):
    """Add Gaussian noise to model parameters."""
    noise_scale = noise_multiplier * sensitivity
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.normal(mean=0, std=noise_scale, size=param.shape).to(
                param.device
            )
            param.data += noise


def compute_privacy_budget(num_rounds, noise_multiplier, sensitivity, delta):
    """Compute approximate (ε, δ) privacy budget using moments accountant."""
    sigma = noise_multiplier * sensitivity
    epsilon = np.sqrt(2 * num_rounds * np.log(1 / delta)) / sigma
    return epsilon


class DPTrainer:
    """Differential Privacy wrapper for client training."""

    def __init__(self, enabled=True, sensitivity=1.0, noise_multiplier=0.5):
        self.enabled = enabled
        self.sensitivity = sensitivity
        self.noise_multiplier = noise_multiplier

    def apply(self, model):
        """Apply DP operations to model."""
        if not self.enabled:
            return

        clip_gradients(model, self.sensitivity)
        add_gaussian_noise(model, self.noise_multiplier, self.sensitivity)
