"""
The 3 things you do with hypervectors:

bind(a, b)       -> combine two vectors into something NEW, different from both
bundle(a, b)     -> combine two vectors into something similar to BOTH of them
similarity(a, b) -> how alike two vectors are (-1 = opposite, 1 = identical)
"""
import numpy as np


def bind(a, b):
    return a * b


def bundle(a, b):
    return np.sign(a.astype(np.int64) + b.astype(np.int64))


def similarity(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
