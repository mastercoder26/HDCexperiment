"""Prototype classification and a small end-to-end IoT baseline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .config import HDCConfig
from .costs import CostModel
from .encoding import RecordEncoder
from .memory import AssociativeMemory
from .simulator import HDCSimulator, SimulationReport


@dataclass(frozen=True, slots=True)
class LabeledRecord:
    """One labeled categorical record."""

    label: str
    features: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a nonempty string")
        if not isinstance(self.features, Mapping) or not self.features:
            raise ValueError("features must be a nonempty mapping")
        object.__setattr__(
            self,
            "features",
            MappingProxyType(dict(self.features)),
        )


@dataclass(frozen=True, slots=True)
class Prediction:
    """One classifier prediction without a ground-truth label."""

    label: str
    similarity: float
    scores: Mapping[str, float]
    margin: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))


@dataclass(frozen=True, slots=True)
class EvaluatedPrediction:
    """Prediction paired with its expected label."""

    true_label: str
    predicted_label: str
    similarity: float
    margin: float
    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_label": self.true_label,
            "predicted_label": self.predicted_label,
            "correct": self.true_label == self.predicted_label,
            "similarity": self.similarity,
            "margin": self.margin,
            "scores": dict(self.scores),
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Serializable accuracy and simulator output for one experiment."""

    experiment_name: str
    dataset_name: str
    predictions: tuple[EvaluatedPrediction, ...]
    simulation_report: SimulationReport
    item_memory_bytes: int
    prototype_memory_bytes: int

    @property
    def correct_predictions(self) -> int:
        return sum(
            prediction.true_label == prediction.predicted_label
            for prediction in self.predictions
        )

    @property
    def accuracy(self) -> float:
        if not self.predictions:
            return 0.0
        return self.correct_predictions / len(self.predictions)

    def to_dict(
        self,
        *,
        include_measured_runtime: bool = True,
    ) -> dict[str, Any]:
        return {
            "experiment": self.experiment_name,
            "dataset": self.dataset_name,
            "config": self.simulation_report.config.to_dict(),
            "accuracy": self.accuracy,
            "correct_predictions": self.correct_predictions,
            "total_predictions": len(self.predictions),
            "predictions": [
                prediction.to_dict() for prediction in self.predictions
            ],
            "memory": {
                "item_memory_bytes": self.item_memory_bytes,
                "prototype_memory_bytes": self.prototype_memory_bytes,
                "total_bytes": (
                    self.item_memory_bytes + self.prototype_memory_bytes
                ),
            },
            "simulation": self.simulation_report.to_dict(
                include_measured_runtime=include_measured_runtime
            ),
        }


class PrototypeClassifier:
    """One bipolar prototype per class with nearest-similarity inference."""

    def __init__(self, encoder: RecordEncoder) -> None:
        self.encoder = encoder
        self.simulator = encoder.simulator
        self.memory = AssociativeMemory(self.simulator)
        self._is_fitted = False

    def fit(self, records: Iterable[LabeledRecord]) -> None:
        """Encode records and bundle them into one prototype per class."""

        training_records = tuple(records)
        if not training_records:
            raise ValueError("training data must contain at least one record")

        grouped_vectors: dict[str, list[Any]] = defaultdict(list)
        for record in training_records:
            grouped_vectors[record.label].append(
                self.encoder.encode(record.features)
            )

        self.memory = AssociativeMemory(self.simulator)
        for label, vectors in sorted(grouped_vectors.items()):
            prototype = self.simulator.bundle(*vectors)
            self.memory.add(label, prototype)
        self._is_fitted = True

    def predict(self, features: Mapping[str, Any]) -> Prediction:
        """Encode one record and retrieve its nearest class prototype."""

        if not self._is_fitted:
            raise RuntimeError("fit the classifier before making predictions")
        query = self.encoder.encode(features)
        match = self.memory.query(query)
        return Prediction(
            label=match.label,
            similarity=match.similarity,
            scores=match.scores,
            margin=match.margin,
        )


def _training_records() -> tuple[LabeledRecord, ...]:
    return (
        LabeledRecord(
            "normal",
            {"protocol": "mqtt", "encryption": "enabled", "packet_rate": "low"},
        ),
        LabeledRecord(
            "normal",
            {"protocol": "coap", "encryption": "enabled", "packet_rate": "low"},
        ),
        LabeledRecord(
            "normal",
            {
                "protocol": "mqtt",
                "encryption": "enabled",
                "packet_rate": "medium",
            },
        ),
        LabeledRecord(
            "anomaly",
            {
                "protocol": "telnet",
                "encryption": "disabled",
                "packet_rate": "high",
            },
        ),
        LabeledRecord(
            "anomaly",
            {"protocol": "ftp", "encryption": "disabled", "packet_rate": "high"},
        ),
        LabeledRecord(
            "anomaly",
            {
                "protocol": "telnet",
                "encryption": "disabled",
                "packet_rate": "medium",
            },
        ),
    )


def _test_records() -> tuple[LabeledRecord, ...]:
    return (
        LabeledRecord(
            "normal",
            {"protocol": "mqtt", "encryption": "enabled", "packet_rate": "low"},
        ),
        LabeledRecord(
            "normal",
            {"protocol": "https", "encryption": "enabled", "packet_rate": "low"},
        ),
        LabeledRecord(
            "anomaly",
            {
                "protocol": "telnet",
                "encryption": "disabled",
                "packet_rate": "high",
            },
        ),
        LabeledRecord(
            "anomaly",
            {"protocol": "ssh", "encryption": "disabled", "packet_rate": "high"},
        ),
    )


def run_iot_baseline(
    config: HDCConfig | None = None,
    *,
    cost_model: CostModel | None = None,
) -> ExperimentResult:
    """Run a deterministic toy IoT normal-versus-anomaly experiment."""

    simulator = HDCSimulator(config, cost_model=cost_model)
    encoder = RecordEncoder(simulator)
    classifier = PrototypeClassifier(encoder)
    classifier.fit(_training_records())

    evaluated_predictions: list[EvaluatedPrediction] = []
    for record in _test_records():
        prediction = classifier.predict(record.features)
        evaluated_predictions.append(
            EvaluatedPrediction(
                true_label=record.label,
                predicted_label=prediction.label,
                similarity=prediction.similarity,
                margin=prediction.margin,
                scores=prediction.scores,
            )
        )

    return ExperimentResult(
        experiment_name="HDC IoT baseline",
        dataset_name="toy_iot_categorical_v1",
        predictions=tuple(evaluated_predictions),
        simulation_report=simulator.report(),
        item_memory_bytes=encoder.item_memory.memory_bytes,
        prototype_memory_bytes=classifier.memory.memory_bytes,
    )
