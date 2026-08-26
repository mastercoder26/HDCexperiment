"""Parallel operations for HDC using joblib."""

import numpy as np
import concurrent.futures

def parallel_bundle(vectors: list[np.ndarray], chunk_size: int = 100) -> np.ndarray:
    """Bundle vectors in parallel chunks."""
    if not vectors:
        raise ValueError("No vectors to bundle")
        
    def bundle_chunk(chunk):
        totals = np.sum(chunk, axis=0)
        return np.where(totals >= 0, 1, -1).astype(np.int8)

    chunks = [vectors[i:i + chunk_size] for i in range(0, len(vectors), chunk_size)]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        chunk_results = list(executor.map(bundle_chunk, chunks))
        
    # Bundle the chunk results
    return bundle_chunk(chunk_results)
