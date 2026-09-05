"""The three names most users need from this small HDC project."""

from .core import DEFAULT_DIMENSIONS, HDC
from .costs import CostProfile, available_cost_profiles, get_cost_profile
from .model import HDCClassifier

__version__ = "1.2.2"

__all__ = [
    "DEFAULT_DIMENSIONS",
    "CostProfile",
    "HDC",
    "HDCClassifier",
    "__version__",
    "available_cost_profiles",
    "get_cost_profile",
]
