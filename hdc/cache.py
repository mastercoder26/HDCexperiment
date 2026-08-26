"""Caching utilities for HDC."""

from functools import lru_cache
import numpy as np

def memoize_vector_op(func):
    """Generic LRU cache decorator for HDC vector operations."""
    cache = {}
    def wrapper(*args):
        # Hashable arguments only (convert arrays to bytes)
        key = tuple(arg.tobytes() if isinstance(arg, np.ndarray) else arg for arg in args)
        if key not in cache:
            cache[key] = func(*args)
        return cache[key]
    return wrapper
