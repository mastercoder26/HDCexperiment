"""Configuration for a reproducible bipolar HDC simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TIE_BREAKERS = frozenset({"random", "positive", "negative"})


@dataclass(frozen=True, slots=True)
class HDCConfig:
    """Immutable settings shared by all modules in one simulation."""

    dimensions: int = 10_000
    seed: int | None = 0
    tie_breaker: str = "positive"

    def __post_init__(self) -> None:
        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int):
            raise ValueError("dimensions must be a positive integer")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer or None")
        if self.tie_breaker not in TIE_BREAKERS:
            supported = ", ".join(sorted(TIE_BREAKERS))
            raise ValueError(f"tie_breaker must be one of: {supported}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)
