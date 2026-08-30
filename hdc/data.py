"""Load labeled categorical records from JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .model import LabeledRecord


def _read_records(payload: object, section: str) -> list[LabeledRecord]:
    if not isinstance(payload, list):
        raise ValueError(f"dataset '{section}' must be a list")

    records: list[LabeledRecord] = []
    for index, item in enumerate(payload):
        record_name = f"{section} record {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{record_name} must be an object")
        label = item.get("label")
        features = item.get("features")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{record_name} needs a label")
        if not isinstance(features, dict) or not features:
            raise ValueError(f"{record_name} needs features")
        for field, value in features.items():
            if not isinstance(field, str) or not field:
                raise ValueError(f"{record_name} feature names must be strings")
            if value is None or not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"{record_name} feature '{field}' "
                    "must be a categorical scalar"
                )
        records.append((label, features))
    return records


def load_dataset(path: Path) -> tuple[list[LabeledRecord], list[LabeledRecord]]:
    """Load and validate ``training`` and ``test`` records from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset root must be an object")

    training = _read_records(payload.get("training"), "training")
    test = _read_records(payload.get("test"), "test")
    if not training:
        raise ValueError("dataset needs at least one training record")
    return training, test
