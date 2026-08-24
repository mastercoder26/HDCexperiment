"""Instrumented execution of HDC operations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any, Callable, Mapping, TypeVar

import numpy as np
from numpy.typing import NDArray

from .config import HDCConfig
from .costs import CostModel
from .ops import bind, bundle, permute, similarity, validate_hypervector
from .vectors import Hypervector, random_vector


ResultType = TypeVar("ResultType")


@dataclass(slots=True)
class OperationStats:
    """Aggregate measured and modeled values for one operation."""

    calls: int = 0
    work_units: int = 0
    measured_runtime_ns: int = 0
    modeled_energy_pj: float = 0.0
    modeled_latency_ns: float = 0.0

    def record(
        self,
        *,
        work_units: int,
        measured_runtime_ns: int,
        modeled_energy_pj: float,
        modeled_latency_ns: float,
    ) -> None:
        self.calls += 1
        self.work_units += work_units
        self.measured_runtime_ns += measured_runtime_ns
        self.modeled_energy_pj += modeled_energy_pj
        self.modeled_latency_ns += modeled_latency_ns

    def to_dict(self, *, include_measured_runtime: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "calls": self.calls,
            "work_units": self.work_units,
            "modeled_energy_pj": self.modeled_energy_pj,
            "modeled_latency_ns": self.modeled_latency_ns,
        }
        if include_measured_runtime:
            payload["measured_runtime_ns"] = self.measured_runtime_ns
        return payload


@dataclass(frozen=True, slots=True)
class SimulationReport:
    """Immutable snapshot of one simulator's aggregate metrics."""

    config: HDCConfig
    cost_profile: str
    operations: Mapping[str, OperationStats]

    @property
    def total_calls(self) -> int:
        return sum(stats.calls for stats in self.operations.values())

    @property
    def total_work_units(self) -> int:
        return sum(stats.work_units for stats in self.operations.values())

    @property
    def total_measured_runtime_ns(self) -> int:
        return sum(
            stats.measured_runtime_ns for stats in self.operations.values()
        )

    @property
    def total_modeled_energy_pj(self) -> float:
        return sum(stats.modeled_energy_pj for stats in self.operations.values())

    @property
    def total_modeled_latency_ns(self) -> float:
        return sum(stats.modeled_latency_ns for stats in self.operations.values())

    def to_dict(self, *, include_measured_runtime: bool = True) -> dict[str, Any]:
        totals: dict[str, Any] = {
            "calls": self.total_calls,
            "work_units": self.total_work_units,
            "modeled_energy_pj": self.total_modeled_energy_pj,
            "modeled_latency_ns": self.total_modeled_latency_ns,
        }
        if include_measured_runtime:
            totals["measured_runtime_ns"] = self.total_measured_runtime_ns

        return {
            "config": self.config.to_dict(),
            "cost_profile": self.cost_profile,
            "operations": {
                operation: stats.to_dict(
                    include_measured_runtime=include_measured_runtime
                )
                for operation, stats in sorted(self.operations.items())
            },
            "totals": totals,
        }


class HDCSimulator:
    """Run bipolar HDC operations with reproducible randomness and accounting."""

    def __init__(
        self,
        config: HDCConfig | None = None,
        *,
        cost_model: CostModel | None = None,
    ) -> None:
        self.config = config if config is not None else HDCConfig()
        self.cost_model = cost_model if cost_model is not None else CostModel()
        self._rng = np.random.default_rng(self.config.seed)
        self._operations: dict[str, OperationStats] = {}

    def _run_recorded(
        self,
        operation: str,
        work_units: int,
        action: Callable[[], ResultType],
    ) -> ResultType:
        start_ns = perf_counter_ns()
        result = action()
        measured_runtime_ns = max(1, perf_counter_ns() - start_ns)
        estimate = self.cost_model.estimate(operation, work_units)
        stats = self._operations.setdefault(operation, OperationStats())
        stats.record(
            work_units=work_units,
            measured_runtime_ns=measured_runtime_ns,
            modeled_energy_pj=estimate.energy_pj,
            modeled_latency_ns=estimate.latency_ns,
        )
        return result

    def _validate_dimension(
        self,
        vector: NDArray[np.generic],
    ) -> NDArray[np.generic]:
        validated = validate_hypervector(vector)
        if validated.size != self.config.dimensions:
            raise ValueError(
                "hypervector dimension does not match configured dimension "
                f"{self.config.dimensions}"
            )
        return validated

    def random_hypervector(self) -> Hypervector:
        """Generate and record one hypervector."""

        return self._run_recorded(
            "random",
            self.config.dimensions,
            lambda: random_vector(
                dimensions=self.config.dimensions,
                rng=self._rng,
            ),
        )

    def bind(
        self,
        first: NDArray[np.generic],
        second: NDArray[np.generic],
    ) -> Hypervector:
        """Bind two configured-dimension vectors."""

        self._validate_dimension(first)
        self._validate_dimension(second)
        return self._run_recorded(
            "bind",
            self.config.dimensions,
            lambda: bind(first, second),
        )

    def bundle(self, *vectors: NDArray[np.generic]) -> Hypervector:
        """Bundle configured-dimension vectors using the configured tie policy."""

        for vector in vectors:
            self._validate_dimension(vector)
        work_units = max(0, len(vectors) - 1) * self.config.dimensions
        return self._run_recorded(
            "bundle",
            work_units,
            lambda: bundle(
                *vectors,
                tie_breaker=self.config.tie_breaker,
                rng=self._rng if self.config.tie_breaker == "random" else None,
            ),
        )

    def permute(
        self,
        vector: NDArray[np.generic],
        shifts: int = 1,
    ) -> Hypervector:
        """Permute one configured-dimension vector."""

        self._validate_dimension(vector)
        return self._run_recorded(
            "permute",
            self.config.dimensions,
            lambda: permute(vector, shifts),
        )

    def similarity(
        self,
        first: NDArray[np.generic],
        second: NDArray[np.generic],
    ) -> float:
        """Compare two configured-dimension vectors."""

        self._validate_dimension(first)
        self._validate_dimension(second)
        return self._run_recorded(
            "similarity",
            self.config.dimensions,
            lambda: similarity(first, second),
        )

    def reset_metrics(self) -> None:
        """Clear accounting without changing configuration or random state."""

        self._operations.clear()

    def report(self) -> SimulationReport:
        """Return a snapshot that cannot mutate live simulator statistics."""

        copied_operations = {
            operation: replace(stats)
            for operation, stats in self._operations.items()
        }
        return SimulationReport(
            config=self.config,
            cost_profile=self.cost_model.name,
            operations=MappingProxyType(copied_operations),
        )
