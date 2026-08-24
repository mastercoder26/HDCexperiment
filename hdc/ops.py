"""Pure operations for bipolar hypervectors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import TIE_BREAKERS
from .vectors import Hypervector


def validate_hypervector(
    vector: NDArray[np.generic],
    *,
    name: str = "vector",
) -> NDArray[np.generic]:
    """Validate that a value is a one-dimensional bipolar hypervector."""

    array = np.asarray(vector)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional hypervector")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all((array == -1) | (array == 1)):
        raise ValueError(f"{name} must contain only bipolar values -1 and +1")
    return array


def _validate_same_shape(*vectors: NDArray[np.generic]) -> None:
    if len({np.asarray(vector).shape for vector in vectors}) > 1:
        raise ValueError("hypervectors must have the same shape")


def _validated_vectors(
    vectors: tuple[NDArray[np.generic], ...],
) -> tuple[NDArray[np.generic], ...]:
    _validate_same_shape(*vectors)
    return tuple(
        validate_hypervector(vector, name=f"vector[{index}]")
        for index, vector in enumerate(vectors)
    )


def bind(
    first: NDArray[np.generic],
    second: NDArray[np.generic],
) -> Hypervector:
    """Associate two vectors through bipolar element-wise multiplication."""

    validated_first, validated_second = _validated_vectors((first, second))
    return np.multiply(validated_first, validated_second, dtype=np.int8)


def bundle(
    *vectors: NDArray[np.generic],
    tie_breaker: str = "positive",
    rng: np.random.Generator | None = None,
) -> Hypervector:
    """Superpose one or more vectors and return a bipolar hypervector."""

    if not vectors:
        raise ValueError("bundle requires at least one hypervector")
    if tie_breaker not in TIE_BREAKERS:
        supported = ", ".join(sorted(TIE_BREAKERS))
        raise ValueError(f"tie_breaker must be one of: {supported}")
    if tie_breaker == "random" and rng is None:
        raise ValueError(
            "random tie_breaker requires an explicit random-number generator"
        )

    validated = _validated_vectors(vectors)
    accumulator = np.sum(np.stack(validated), axis=0, dtype=np.int64)
    result = np.sign(accumulator).astype(np.int8)
    tie_mask = result == 0
    tie_count = int(np.count_nonzero(tie_mask))

    if tie_count:
        if tie_breaker == "positive":
            result[tie_mask] = 1
        elif tie_breaker == "negative":
            result[tie_mask] = -1
        else:
            assert rng is not None
            result[tie_mask] = rng.choice(
                np.array([-1, 1], dtype=np.int8),
                size=tie_count,
            )

    return result


def permute(vector: NDArray[np.generic], shifts: int = 1) -> Hypervector:
    """Cyclically shift a vector to represent order or position."""

    validated = validate_hypervector(vector)
    if isinstance(shifts, bool) or not isinstance(shifts, int):
        raise ValueError("shifts must be an integer")
    return np.roll(validated, shifts).astype(np.int8, copy=False)


def similarity(
    first: NDArray[np.generic],
    second: NDArray[np.generic],
) -> float:
    """Return normalized dot-product similarity in the range [-1, 1]."""

    validated_first, validated_second = _validated_vectors((first, second))
    first_float = validated_first.astype(np.float64, copy=False)
    second_float = validated_second.astype(np.float64, copy=False)
    return float(np.dot(first_float, second_float) / validated_first.size)
