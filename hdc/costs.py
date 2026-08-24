"""Replaceable analytical energy and latency cost profiles."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


SUPPORTED_OPERATIONS = (
    "random",
    "bind",
    "bundle",
    "permute",
    "similarity",
)


def _validate_nonnegative_finite(value: float, name: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative finite number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite number")


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Modeled cost for one recorded operation."""

    energy_pj: float
    latency_ns: float


@dataclass(frozen=True, slots=True)
class OperationCost:
    """Fixed and work-proportional cost assumptions for one operation."""

    fixed_energy_pj: float = 0.0
    energy_pj_per_work_unit: float = 0.0
    fixed_latency_ns: float = 0.0
    latency_ns_per_work_unit: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _validate_nonnegative_finite(value, name)

    def estimate(self, work_units: int) -> CostEstimate:
        """Calculate modeled cost for a nonnegative amount of work."""

        if isinstance(work_units, bool) or not isinstance(work_units, int):
            raise ValueError("work_units must be a nonnegative integer")
        if work_units < 0:
            raise ValueError("work_units must be a nonnegative integer")
        return CostEstimate(
            energy_pj=(
                float(self.fixed_energy_pj)
                + work_units * float(self.energy_pj_per_work_unit)
            ),
            latency_ns=(
                float(self.fixed_latency_ns)
                + work_units * float(self.latency_ns_per_work_unit)
            ),
        )

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible cost coefficients."""

        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class CostModel:
    """Named per-operation analytical cost assumptions."""

    name: str = "unconfigured"
    operation_costs: Mapping[str, OperationCost] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("cost model name must be a nonempty string")

        copied_costs: dict[str, OperationCost] = {}
        for operation, cost in self.operation_costs.items():
            if not isinstance(operation, str) or not operation:
                raise ValueError("operation names must be nonempty strings")
            if not isinstance(cost, OperationCost):
                raise ValueError("operation costs must be OperationCost instances")
            copied_costs[operation] = cost
        object.__setattr__(
            self,
            "operation_costs",
            MappingProxyType(copied_costs),
        )

    def estimate(self, operation: str, work_units: int) -> CostEstimate:
        """Estimate an operation, using zero cost when it is unconfigured."""

        cost = self.operation_costs.get(operation, OperationCost())
        return cost.estimate(work_units)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible named profile."""

        return {
            "name": self.name,
            "operations": {
                operation: cost.to_dict()
                for operation, cost in sorted(self.operation_costs.items())
            },
        }

    @classmethod
    def uniform(
        cls,
        *,
        name: str,
        fixed_energy_pj: float = 0.0,
        energy_pj_per_work_unit: float = 0.0,
        fixed_latency_ns: float = 0.0,
        latency_ns_per_work_unit: float = 0.0,
    ) -> "CostModel":
        """Create a simple profile with the same coefficients for every operation."""

        cost = OperationCost(
            fixed_energy_pj=fixed_energy_pj,
            energy_pj_per_work_unit=energy_pj_per_work_unit,
            fixed_latency_ns=fixed_latency_ns,
            latency_ns_per_work_unit=latency_ns_per_work_unit,
        )
        return cls(
            name=name,
            operation_costs={
                operation: cost for operation in SUPPORTED_OPERATIONS
            },
        )
