import unittest

import numpy as np

from hdc.core import HDC
from hdc.model import HDCClassifier


TRAINING_RECORDS = [
    ("class_a", {"color": "red", "shape": "circle", "size": "small"}),
    ("class_a", {"color": "orange", "shape": "circle", "size": "small"}),
    ("class_a", {"color": "red", "shape": "square", "size": "medium"}),
    ("class_b", {"color": "blue", "shape": "triangle", "size": "large"}),
    ("class_b", {"color": "green", "shape": "triangle", "size": "large"}),
    ("class_b", {"color": "blue", "shape": "square", "size": "medium"}),
]


class SimpleClassifierTests(unittest.TestCase):
    def setUp(self):
        self.hdc = HDC(dimensions=2_000, seed=42)
        self.classifier = HDCClassifier(self.hdc)

    def test_same_token_gets_same_vector(self):
        first = self.classifier.vector_for("color")
        second = self.classifier.vector_for("color")

        np.testing.assert_array_equal(first, second)

    def test_record_order_does_not_change_encoding(self):
        first = self.classifier.encode({"color": "red", "shape": "circle"})
        second = self.classifier.encode({"shape": "circle", "color": "red"})

        self.assertEqual(self.hdc.similarity(first, second), 1.0)

    def test_classifier_trains_and_predicts(self):
        self.classifier.train(TRAINING_RECORDS)

        result = self.classifier.predict(
            {"color": "red", "shape": "circle", "size": "small"}
        )

        self.assertEqual(result["label"], "class_a")
        self.assertGreater(result["margin"], 0)
        self.assertEqual(set(result["scores"]), {"class_a", "class_b"})

    def test_evaluate_returns_accuracy_and_predictions(self):
        self.classifier.train(TRAINING_RECORDS)
        test_records = [
            ("class_a", {"color": "orange", "shape": "circle", "size": "small"}),
            ("class_b", {"color": "green", "shape": "triangle", "size": "large"}),
        ]

        result = self.classifier.evaluate(test_records)

        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(len(result["predictions"]), 2)

    def test_predict_batch_returns_one_result_per_record(self):
        self.classifier.train(TRAINING_RECORDS)
        records = [
            {"color": "red", "shape": "circle", "size": "small"},
            {"color": "blue", "shape": "triangle", "size": "large"},
        ]

        results = self.classifier.predict_batch(records)

        self.assertEqual(
            [result["label"] for result in results],
            ["class_a", "class_b"],
        )

    def test_prediction_before_training_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "train"):
            self.classifier.predict({"color": "red"})

    def test_memory_size_counts_items_and_prototypes(self):
        self.classifier.train(TRAINING_RECORDS)

        expected_bytes = (
            len(self.classifier.item_memory) + len(self.classifier.prototypes)
        ) * self.hdc.dimensions
        self.assertEqual(self.classifier.memory_bytes(), expected_bytes)


if __name__ == "__main__":
    unittest.main()
