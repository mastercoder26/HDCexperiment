"""Task-level encoders that compose the core HDC operations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .memory import ItemMemory
from .simulator import HDCSimulator
from .vectors import Hypervector


class RecordEncoder:
    """Encode categorical field-value mappings as one bipolar hypervector."""

    def __init__(
        self,
        simulator: HDCSimulator,
        *,
        item_memory: ItemMemory | None = None,
    ) -> None:
        self.simulator = simulator
        self.item_memory = (
            item_memory if item_memory is not None else ItemMemory(simulator)
        )

    @staticmethod
    def _serialized_value(value: Any) -> str:
        try:
            return json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "feature values must be JSON-serializable categorical values"
            ) from error

    @classmethod
    def value_token(cls, feature_name: str, value: Any) -> str:
        return f"value:{feature_name}:{cls._serialized_value(value)}"

    def encode(self, features: Mapping[str, Any]) -> Hypervector:
        """Bind each field role to its value and bundle the pairs."""

        if not isinstance(features, Mapping) or not features:
            raise ValueError("a record must contain at least one feature")

        pairs: list[Hypervector] = []
        for feature_name in sorted(features):
            if not isinstance(feature_name, str) or not feature_name:
                raise ValueError("feature name must be a nonempty string")
            role = self.item_memory.get(f"field:{feature_name}")
            value = self.item_memory.get(
                self.value_token(feature_name, features[feature_name])
            )
            pairs.append(self.simulator.bind(role, value))

        return self.simulator.bundle(*pairs)
