"""Named analytical cost assumptions for HDC operations."""

from __future__ import annotations

from dataclasses import dataclass

from .core import OPERATIONS


@dataclass(frozen=True)
class CostProfile:
    """Energy and latency assumptions per processed vector position."""

    name: str
    description: str
    energy_pj: dict[str, float]
    latency_ns: dict[str, float]


def _uniform_cost(value: float) -> dict[str, float]:
    return {operation: value for operation in OPERATIONS}


_PROFILES = {
    "unconfigured": CostProfile(
        name="unconfigured",
        description="No hardware costs configured; reported estimates are zero.",
        energy_pj=_uniform_cost(0.0),
        latency_ns=_uniform_cost(0.0),
    ),
    "example_edge": CostProfile(
        name="example_edge",
        description=(
            "Illustrative operation-specific values for demonstrations only; "
            "they are not measurements from a real device."
        ),
        energy_pj={
            "random": 0.20,
            "bind": 0.05,
            "bundle": 0.12,
            "permute": 0.08,
            "similarity": 0.15,
        },
        latency_ns={
            "random": 0.30,
            "bind": 0.10,
            "bundle": 0.25,
            "permute": 0.12,
            "similarity": 0.40,
        },
    ),
}


def available_cost_profiles() -> tuple[str, ...]:
    """Return cost-profile names accepted by the command line."""
    return tuple(_PROFILES)


def get_cost_profile(name: str) -> CostProfile:
    """Return an independent copy of a named cost profile."""
    try:
        profile = _PROFILES[name]
    except KeyError as error:
        choices = ", ".join(available_cost_profiles())
        raise ValueError(f"unknown cost profile '{name}'; choose from {choices}") from error
    return CostProfile(
        name=profile.name,
        description=profile.description,
        energy_pj=dict(profile.energy_pj),
        latency_ns=dict(profile.latency_ns),
    )
