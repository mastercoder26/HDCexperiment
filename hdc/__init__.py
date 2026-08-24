"""Public API for the modular HDC simulator."""

from .config import HDCConfig
from .costs import CostModel, OperationCost
from .ops import bind, bundle, permute, similarity
from .simulator import HDCSimulator
from .vectors import DIM, Hypervector, random_vector

__all__ = [
    "CostModel",
    "DIM",
    "HDCConfig",
    "HDCSimulator",
    "Hypervector",
    "OperationCost",
    "bind",
    "bundle",
    "permute",
    "random_vector",
    "similarity",
]
