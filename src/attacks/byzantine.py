"""Byzantine attack implementations for federated learning."""

import torch


def sign_flipping(model_weights):
    """Flip the sign of all parameters."""
    attacked_weights = {}
    for key in model_weights.keys():
        attacked_weights[key] = -model_weights[key]
    return attacked_weights


def random_noise(model_weights, noise_scale=0.1):
    """Add random Gaussian noise to model weights."""
    attacked_weights = {}
    for key in model_weights.keys():
        noise = torch.normal(mean=0, std=noise_scale, size=model_weights[key].shape)
        attacked_weights[key] = model_weights[key] + noise
    return attacked_weights


def label_flipping_update(model_weights):
    """Labels flipped in local training, so no gradient manipulation needed."""
    # This is a placeholder - actual label flipping happens in data preprocessing
    return model_weights


class ByzantineClient:
    """Base class for Byzantine client behavior."""

    def __init__(self, attack_type="sign_flipping", **kwargs):
        self.attack_type = attack_type
        self.kwargs = kwargs

    def attack(self, model_weights):
        """Apply Byzantine attack to model weights."""
        if self.attack_type == "sign_flipping":
            return sign_flipping(model_weights)
        elif self.attack_type == "random_noise":
            return random_noise(model_weights, **self.kwargs)
        else:
            return model_weights
