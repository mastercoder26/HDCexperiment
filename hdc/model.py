"""An HDC classifier for categorical records."""

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

        self.item_memory: dict[str, np.ndarray] = {}
        self.prototypes: dict[str, np.ndarray] = {}

    def vector_for(self, token: str) -> np.ndarray:
        """Return a stable random vector for a token."""
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
            if not isinstance(label, str) or not label:
                raise ValueError("training labels must be non-empty strings")
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
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        return {
            "label": best_label,
            "scores": scores,
            "margin": best_score - second_score,
        }

    def predict_batch(self, records: Sequence[Record]) -> list[dict[str, object]]:
        """Predict labels for a batch of records."""
        return [self.predict(record) for record in records]

    def evaluate(self, test_records: Sequence[LabeledRecord]) -> dict[str, object]:
        """Predict several labeled examples and calculate accuracy."""
        predictions = []
        correct = 0
        margin_total = 0.0
        class_totals: dict[str, dict[str, int]] = {}

        for true_label, record in test_records:
            result = self.predict(record)
            is_correct = result["label"] == true_label
            correct += int(is_correct)
            margin_total += float(result["margin"])
            class_stats = class_totals.setdefault(
                true_label,
                {"correct": 0, "total": 0},
            )
            class_stats["correct"] += int(is_correct)
            class_stats["total"] += 1
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
        per_class = {}
        for label in sorted(class_totals):
            stats = class_totals[label]
            per_class[label] = {
                **stats,
                "accuracy": stats["correct"] / stats["total"],
            }

        labels = sorted(
            set(self.prototypes)
            | {label for label, _ in test_records}
            | {str(prediction["predicted_label"]) for prediction in predictions}
        )
        confusion_rows = {
            true_label: {predicted_label: 0 for predicted_label in labels}
            for true_label in labels
        }
        for prediction in predictions:
            true_label = str(prediction["true_label"])
            predicted_label = str(prediction["predicted_label"])
            confusion_rows[true_label][predicted_label] += 1

        classification_report = {}
        for label in labels:
            true_positives = confusion_rows[label][label]
            support = sum(confusion_rows[label].values())
            predicted_total = sum(row[label] for row in confusion_rows.values())
            precision = true_positives / predicted_total if predicted_total else 0.0
            recall = true_positives / support if support else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            classification_report[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }

        macro_f1 = (
            sum(metrics["f1"] for metrics in classification_report.values())
            / len(classification_report)
            if classification_report
            else 0.0
        )
        return {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "average_margin": margin_total / total if total else 0.0,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "classification_report": classification_report,
            "confusion_matrix": {
                "labels": labels,
                "rows": confusion_rows,
            },
            "predictions": predictions,
        }

    def memory_bytes(self) -> int:
        """Count bytes occupied by remembered and learned hypervectors."""
        vectors = list(self.item_memory.values()) + list(self.prototypes.values())
        return sum(vector.nbytes for vector in vectors)
