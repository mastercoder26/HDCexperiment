import unittest

import numpy as np

from hdc.core import HDC
from hdc.model import HDCClassifier


TRAINING_RECORDS = [
    ("normal", {"protocol": "mqtt", "encryption": "on", "rate": "low"}),
    ("normal", {"protocol": "coap", "encryption": "on", "rate": "low"}),
    ("normal", {"protocol": "mqtt", "encryption": "on", "rate": "medium"}),
    ("anomaly", {"protocol": "telnet", "encryption": "off", "rate": "high"}),
    ("anomaly", {"protocol": "ftp", "encryption": "off", "rate": "high"}),
    ("anomaly", {"protocol": "telnet", "encryption": "off", "rate": "medium"}),
]


class SimpleClassifierTests(unittest.TestCase):
    def setUp(self):
        self.hdc = HDC(dimensions=2_000, seed=42)
        self.classifier = HDCClassifier(self.hdc)

    def test_same_token_gets_same_vector(self):
        first = self.classifier.vector_for("protocol")
        second = self.classifier.vector_for("protocol")

        np.testing.assert_array_equal(first, second)

    def test_record_order_does_not_change_encoding(self):
        first = self.classifier.encode({"protocol": "mqtt", "rate": "low"})
        second = self.classifier.encode({"rate": "low", "protocol": "mqtt"})

        self.assertEqual(self.hdc.similarity(first, second), 1.0)

    def test_classifier_trains_and_predicts(self):
        self.classifier.train(TRAINING_RECORDS)

        result = self.classifier.predict(
            {"protocol": "mqtt", "encryption": "on", "rate": "low"}
        )

        self.assertEqual(result["label"], "normal")
        self.assertGreater(result["margin"], 0)
        self.assertEqual(set(result["scores"]), {"normal", "anomaly"})

    def test_evaluate_returns_accuracy_and_predictions(self):
        self.classifier.train(TRAINING_RECORDS)
        test_records = [
            ("normal", {"protocol": "https", "encryption": "on", "rate": "low"}),
            ("anomaly", {"protocol": "ssh", "encryption": "off", "rate": "high"}),
        ]

        result = self.classifier.evaluate(test_records)

        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(len(result["predictions"]), 2)

    def test_prediction_before_training_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "train"):
            self.classifier.predict({"protocol": "mqtt"})

    def test_memory_size_counts_items_and_prototypes(self):
        self.classifier.train(TRAINING_RECORDS)

        expected_bytes = (
            len(self.classifier.item_memory) + len(self.classifier.prototypes)
        ) * self.hdc.dimensions
        self.assertEqual(self.classifier.memory_bytes(), expected_bytes)


if __name__ == "__main__":
    unittest.main()
