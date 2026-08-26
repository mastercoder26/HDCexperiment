"""Distributed and multiprocessing utilities for HDC."""

import multiprocessing
from typing import Callable, Iterable, Any

def run_distributed(func: Callable, items: Iterable, num_workers: int = None) -> list[Any]:
    """Run a function across multiple processes."""
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_workers) as pool:
        return pool.map(func, items)
