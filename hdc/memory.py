"""Item and associative memories for bipolar hypervectors."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .ops import validate_hypervector
from .simulator import HDCSimulator
from .vectors import Hypervector


def _copy_for_storage(vector: NDArray[np.generic]) -> Hypervector:
    stored = np.asarray(vector, dtype=np.int8).copy()
    stored.setflags(write=False)
    return stored


class ItemMemory:
    """Assign a stable random hypervector to each atomic symbol."""

    def __init__(self, simulator: HDCSimulator) -> None:
        self.simulator = simulator
        self._items: dict[str, Hypervector] = {}

    def get(self, symbol: str) -> Hypervector:
        """Return a defensive copy of a symbol's stable hypervector."""

        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a nonempty string")
        if symbol not in self._items:
            self._items[symbol] = _copy_for_storage(
                self.simulator.random_hypervector()
            )
        return self._items[symbol].copy()

    @property
    def memory_bytes(self) -> int:
        return sum(vector.nbytes for vector in self._items.values())

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True, slots=True)
class MemoryMatch:
    """Nearest associative-memory entry and all comparison scores."""

    label: str
    similarity: float
    scores: Mapping[str, float]
    margin: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scores",
            MappingProxyType(dict(self.scores)),
        )


class AssociativeMemory:
    """Store labeled hypervectors and retrieve the nearest by similarity."""

    def __init__(self, simulator: HDCSimulator) -> None:
        self.simulator = simulator
        self._entries: dict[str, Hypervector] = {}

    def add(self, label: str, vector: NDArray[np.generic]) -> None:
        """Add or replace a labeled vector."""

        if not isinstance(label, str) or not label:
            raise ValueError("label must be a nonempty string")
        validated = validate_hypervector(vector)
        if validated.size != self.simulator.config.dimensions:
            raise ValueError(
                "hypervector dimension does not match associative-memory dimension"
            )
        self._entries[label] = _copy_for_storage(validated)

    def query(self, vector: NDArray[np.generic]) -> MemoryMatch:
        """Return the closest entry, with a deterministic label tie-break."""

        if not self._entries:
            raise RuntimeError("cannot query an empty associative memory")

        scores = {
            label: self.simulator.similarity(vector, stored_vector)
            for label, stored_vector in sorted(self._entries.items())
        }
        ordered = sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        winning_label, winning_similarity = ordered[0]
        margin = (
            winning_similarity - ordered[1][1]
            if len(ordered) > 1
            else 0.0
        )
        return MemoryMatch(
            label=winning_label,
            similarity=winning_similarity,
            scores=scores,
            margin=margin,
        )

    @property
    def memory_bytes(self) -> int:
        return sum(vector.nbytes for vector in self._entries.values())

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))
