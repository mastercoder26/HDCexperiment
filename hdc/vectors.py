"""Creation of random bipolar hypervectors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DIM = 10_000
Hypervector = NDArray[np.int8]


def random_vector(
    dimensions: int = DIM,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> Hypervector:
    """Create a random bipolar hypervector.

    Pass a generator when several calls must share one deterministic random
    stream. Pass a seed for a convenient reproducible standalone call.
    """

    if isinstance(dimensions, bool) or not isinstance(dimensions, int):
        raise ValueError("dimensions must be a positive integer")
    if dimensions <= 0:
        raise ValueError("dimensions must be a positive integer")
    if rng is not None and seed is not None:
        raise ValueError("provide either rng or seed, not both")

    generator = rng if rng is not None else np.random.default_rng(seed)
    values = np.array([-1, 1], dtype=np.int8)
    return generator.choice(values, size=dimensions)
#
