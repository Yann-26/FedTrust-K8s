"""Federated Averaging implementations."""

import copy
import torch


def fedavg(client_models):
    """Standard FedAvg - average all models."""
    global_model = copy.deepcopy(client_models[0])
    
    for key in global_model.keys():
        global_model[key] = torch.stack(
            [model[key].float() for model in client_models],
            dim=0
        ).mean(dim=0)
    
    return global_model

