"""Memory optimization utilities for HDC vectors."""

import numpy as np

def pack_bipolar_vector(vector: np.ndarray) -> np.ndarray:
    """Pack a bipolar (-1, 1) vector into an array of bytes."""
    # Convert -1, 1 to 0, 1
    binary = np.where(vector == 1, 1, 0).astype(np.uint8)
    return np.packbits(binary)

def unpack_bipolar_vector(packed: np.ndarray, length: int) -> np.ndarray:
    """Unpack an array of bytes back into a bipolar (-1, 1) vector."""
    binary = np.unpackbits(packed)[:length]
    return np.where(binary == 1, 1, -1).astype(np.int8)
