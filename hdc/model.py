"""A small HDC classifier for generic categorical records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .core import HDC


Record = Mapping[str, object]
LabeledRecord = tuple[str, Record]


class HDCClassifier:
    """Learn one prototype hypervector for each label."""

    def __init__(self, hdc: HDC) -> None:
        self.hdc = hdc

        # item_memory remembers the random vector assigned to each word/value.
        self.item_memory: dict[str, np.ndarray] = {}

        # prototypes will hold one learned vector per class, such as "normal".
        self.prototypes: dict[str, np.ndarray] = {}

    def vector_for(self, token: str) -> np.ndarray:
        """Return the same random vector every time a token is requested."""
        if token not in self.item_memory:
            self.item_memory[token] = self.hdc.random_vector()
        return self.item_memory[token].copy()

    def encode(self, record: Record) -> np.ndarray:
        """Turn a categorical dictionary into one hypervector."""
        if not record:
            raise ValueError("cannot encode an empty record")

        field_vectors = []
        for field in sorted(record):
            field_vector = self.vector_for(f"field:{field}")
            value_vector = self.vector_for(f"value:{field}:{record[field]}")
            field_vectors.append(self.hdc.bind(field_vector, value_vector))

        return self.hdc.bundle(*field_vectors)

    def train(self, training_records: Sequence[LabeledRecord]) -> None:
        """Bundle all training examples for each label into a prototype."""
        if not training_records:
            raise ValueError("training data cannot be empty")

        examples_by_label: dict[str, list[np.ndarray]] = {}
        for label, record in training_records:
            examples_by_label.setdefault(label, []).append(self.encode(record))

        self.prototypes = {
            label: self.hdc.bundle(*examples)
            for label, examples in examples_by_label.items()
        }

    def predict(self, record: Record) -> dict[str, object]:
        """Choose the prototype most similar to the encoded record."""
        if not self.prototypes:
            raise RuntimeError("train the classifier before making predictions")

        encoded = self.encode(record)
        scores = {
            label: self.hdc.similarity(encoded, prototype)
            for label, prototype in self.prototypes.items()
        }
        ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        margin = scores[ranked[0]]
        if len(ranked) > 1:
            margin -= scores[ranked[1]]

        return {"label": ranked[0], "scores": scores, "margin": margin}

    def evaluate(self, test_records: Sequence[LabeledRecord]) -> dict[str, object]:
        """Predict several labeled examples and calculate accuracy."""
        predictions = []
        correct = 0

        for true_label, record in test_records:
            result = self.predict(record)
            is_correct = result["label"] == true_label
            correct += int(is_correct)
            predictions.append(
                {
                    "true_label": true_label,
                    "predicted_label": result["label"],
                    "correct": is_correct,
                    "scores": result["scores"],
                    "margin": result["margin"],
                }
            )

        total = len(test_records)
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "predictions": predictions,
        }

    def memory_bytes(self) -> int:
        """Count bytes occupied by remembered and learned hypervectors."""
        vectors = list(self.item_memory.values()) + list(self.prototypes.values())
        return sum(vector.nbytes for vector in vectors)
