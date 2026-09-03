"""Aggregation methods for federated learning."""

from src.aggregation.fedavg import fedavg
from src.aggregation.robust import (
    krum,
    trimmed_mean,
    coordinate_median,
    geometric_median,
)

__all__ = ["fedavg", "krum", "trimmed_mean", "coordinate_median", "geometric_median"]
