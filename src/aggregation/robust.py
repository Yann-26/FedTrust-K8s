"""Robust aggregation methods for Byzantine-resilient FL."""

import copy
import torch


def krum(client_models, f=1):
    """Krum: Select the model most consistent with others.
    
    Args:
        client_models: List of model state_dicts
        f: Number of Byzantine clients (assumed)
    """
    num_neighbors = len(client_models) - f - 2
    if num_neighbors < 1:
        num_neighbors = 1
    
    # Convert to flat vectors
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
    
    # Compute scores
    scores = []
    for i in range(num_clients):
        dist_to_others = distances[i, :]
        sorted_indices = torch.argsort(dist_to_others)
        nearest_indices = sorted_indices[1:num_neighbors + 1]
        score = dist_to_others[nearest_indices].sum()
        scores.append(score)
    
    best_idx = torch.argmin(torch.tensor(scores)).item()
    return client_models[best_idx]


def trimmed_mean(client_models, trim_ratio=0.2):
    """Trimmed Mean: Remove extreme values and average the rest."""
    global_model = copy.deepcopy(client_models[0])
    n = len(client_models)
    trim_count = int(n * trim_ratio)
    
    for key in global_model.keys():
        stacked = torch.stack(
            [model[key].float() for model in client_models],
            dim=0
        )
        sorted_values, _ = torch.sort(stacked, dim=0)
        
        if trim_count > 0:
            trimmed = sorted_values[trim_count:n - trim_count]
            global_model[key] = trimmed.mean(dim=0)
        else:
            global_model[key] = stacked.mean(dim=0)
    
    return global_model


def coordinate_median(client_models):
    """Coordinate-wise Median: Take median for each parameter."""
    global_model = copy.deepcopy(client_models[0])
    
    for key in global_model.keys():
        stacked = torch.stack(
            [model[key].float() for model in client_models],
            dim=0
        )
        global_model[key] = torch.median(stacked, dim=0)[0]
    
    return global_model


def geometric_median(client_models, max_iter=100, tol=1e-5):
    """Geometric Median: Find point minimizing sum of Euclidean distances."""
    # Convert to flat vectors
    flat_models = []
    for model in client_models:
        flat = []
        for key in model.keys():
            flat.append(model[key].flatten())
        flat_models.append(torch.cat(flat))
    
    X = torch.stack(flat_models, dim=0)
    y = X.mean(dim=0)
    
    for _ in range(max_iter):
        distances = torch.norm(X - y, dim=1)
        
        # Handle zero distances
        mask = distances < tol
        if mask.any():
            return client_models[mask.nonzero()[0].item()]
        
        # Weiszfeld update
        weights = 1.0 / distances
        weights = weights / weights.sum()
        y_new = (weights.unsqueeze(1) * X).sum(dim=0)
        
        if torch.norm(y_new - y) < tol:
            break
        y = y_new
    
    # Reconstruct state_dict
    global_model = copy.deepcopy(client_models[0])
    idx = 0
    for key in global_model.keys():
        param_size = global_model[key].numel()
        global_model[key] = y[idx:idx + param_size].reshape(global_model[key].shape)
        idx += param_size
    
    return global_model
