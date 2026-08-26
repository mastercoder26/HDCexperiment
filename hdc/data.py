"""Load labeled categorical records from a small JSON dataset."""

from __future__ import annotations

import json
from pathlib import Path

from .model import LabeledRecord


def _read_records(payload: object, section: str) -> list[LabeledRecord]:
    if not isinstance(payload, list):
        raise ValueError(f"dataset '{section}' must be a list")

    records: list[LabeledRecord] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"{section} record {index} must be an object")
        label = item.get("label")
        features = item.get("features")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{section} record {index} needs a label")
        if not isinstance(features, dict) or not features:
            raise ValueError(f"{section} record {index} needs features")
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
