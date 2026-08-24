"""
A "hypervector" is just a big list of random -1s and 1s.

Two random hypervectors are almost always very different from each other.
That's the one trick the whole simulator is built on.
"""
import numpy as np

DIM = 10000  # how many numbers are in each hypervector


def random_vector():
    """Make a new random hypervector."""
    return np.random.choice([-1, 1], size=DIM)
#
