"""The five basic operations used by this beginner HDC project."""

from __future__ import annotations

import numpy as np


# Default hypervector dimensionality.
DEFAULT_DIMENSIONS = 10_000

OPERATIONS = ("random", "bind", "bundle", "permute", "similarity")


class HDC:
    """Create and operate on bipolar hypervectors containing -1 and +1."""

    def __init__(
        self,
        dimensions: int = DEFAULT_DIMENSIONS,
        seed: int = 42,
        energy_costs: dict[str, float] | None = None,
        latency_costs: dict[str, float] | None = None,
    ) -> None:
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")

        self.dimensions = dimensions
        self.random = np.random.default_rng(seed)
        self.energy_costs = dict(energy_costs or {})
        self.latency_costs = dict(latency_costs or {})

        self.calls = {name: 0 for name in OPERATIONS}
        self.work_units = {name: 0 for name in OPERATIONS}

    def _check_vector(self, vector: np.ndarray) -> np.ndarray:
        """Make sure a value is one valid hypervector for this HDC object."""
        checked = np.asarray(vector)
        if checked.ndim != 1 or len(checked) != self.dimensions:
            raise ValueError(
                f"expected a vector with {self.dimensions} dimensions"
            )
        if not np.all((checked == -1) | (checked == 1)):
            raise ValueError("bipolar hypervectors may contain only -1 and +1")
        return checked

    def _record(self, operation: str, work: int) -> None:
        """Add one operation to the simulator counters."""
        self.calls[operation] += 1
        self.work_units[operation] += work

    def random_vector(self) -> np.ndarray:
        """Create a random hypervector."""
        self._record("random", self.dimensions)
        return self.random.choice(
            np.array([-1, 1], dtype=np.int8), size=self.dimensions
        )

    def bind(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Join two concepts by multiplying their matching positions."""
        left = self._check_vector(left)
        right = self._check_vector(right)
        self._record("bind", self.dimensions)
        return (left * right).astype(np.int8)

    def bundle(
        self,
        *vectors: np.ndarray,
        weights: list[float] | None = None,
    ) -> np.ndarray:
        """Combine several hypervectors into one majority-vote hypervector."""
        if not vectors:
            raise ValueError("bundle needs at least one vector")
        if weights is not None and len(weights) != len(vectors):
            raise ValueError("weights must match the number of vectors")

        checked = [self._check_vector(vector) for vector in vectors]
        if weights is not None:
            weighted = [
                vector * weight
                for vector, weight in zip(checked, weights)
            ]
            totals = np.sum(weighted, axis=0)
        else:
            totals = np.sum(checked, axis=0)

        bundled = np.where(totals >= 0, 1, -1).astype(np.int8)
        self._record("bundle", (len(checked) - 1) * self.dimensions)
        return bundled

    def permute(self, vector: np.ndarray, shifts: int = 1) -> np.ndarray:
        """Move positions to represent order or sequence."""
        vector = self._check_vector(vector)
        self._record("permute", self.dimensions)
        return np.roll(vector, shifts).astype(np.int8)

    def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        """Return cosine similarity for bipolar vectors, from -1 to +1."""
        left = self._check_vector(left)
        right = self._check_vector(right)
        self._record("similarity", self.dimensions)
        return float(np.dot(left.astype(float), right.astype(float))) / self.dimensions

    def report(self) -> dict[str, object]:
        """Return operation counts plus optional energy and latency estimates."""
        operations: dict[str, dict[str, float | int]] = {}
        for name in OPERATIONS:
            work = self.work_units[name]
            operations[name] = {
                "calls": self.calls[name],
                "work_units": work,
                "energy_pj": work * self.energy_costs.get(name, 0.0),
                "latency_ns": work * self.latency_costs.get(name, 0.0),
            }

        return {
            "operations": operations,
            "total_work_units": sum(self.work_units.values()),
            "total_energy_pj": sum(
                details["energy_pj"] for details in operations.values()
            ),
            "total_latency_ns": sum(
                details["latency_ns"] for details in operations.values()
            ),
        }
