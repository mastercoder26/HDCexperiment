import unittest

import numpy as np

from hdc import HDCConfig, HDCSimulator
from hdc.encoding import RecordEncoder
from hdc.memory import AssociativeMemory, ItemMemory


class ItemMemoryTests(unittest.TestCase):
    def setUp(self):
        self.simulator = HDCSimulator(HDCConfig(dimensions=2_000, seed=13))
        self.memory = ItemMemory(self.simulator)

    def test_same_symbol_returns_same_vector(self):
        first = self.memory.get("value:mqtt")
        second = self.memory.get("value:mqtt")

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(self.memory), 1)

    def test_returned_vector_cannot_mutate_stored_item(self):
        returned = self.memory.get("value:mqtt")
        original_first_value = int(returned[0])
        returned[0] *= -1

        stored = self.memory.get("value:mqtt")

        self.assertEqual(int(stored[0]), original_first_value)

    def test_empty_symbol_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "symbol"):
            self.memory.get("")


class AssociativeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.simulator = HDCSimulator(HDCConfig(dimensions=2_000, seed=14))
        self.memory = AssociativeMemory(self.simulator)

    def test_query_returns_nearest_vector_and_margin(self):
        normal = self.simulator.random_hypervector()
        anomaly = self.simulator.random_hypervector()
        self.memory.add("normal", normal)
        self.memory.add("anomaly", anomaly)

        match = self.memory.query(normal)

        self.assertEqual(match.label, "normal")
        self.assertAlmostEqual(match.similarity, 1.0)
        self.assertGreater(match.margin, 0)
        self.assertEqual(set(match.scores), {"normal", "anomaly"})

    def test_query_requires_at_least_one_entry(self):
        query = self.simulator.random_hypervector()

        with self.assertRaisesRegex(RuntimeError, "empty"):
            self.memory.query(query)

    def test_add_validates_label_and_dimension(self):
        vector = self.simulator.random_hypervector()

        with self.assertRaisesRegex(ValueError, "label"):
            self.memory.add("", vector)

        with self.assertRaisesRegex(ValueError, "dimension"):
            self.memory.add("wrong", np.ones(10, dtype=np.int8))


class RecordEncoderTests(unittest.TestCase):
    def setUp(self):
        self.simulator = HDCSimulator(HDCConfig(dimensions=4_000, seed=15))
        self.encoder = RecordEncoder(self.simulator)

    def test_record_encoding_is_independent_of_mapping_order(self):
        first = self.encoder.encode({"protocol": "mqtt", "rate": "low"})
        second = self.encoder.encode({"rate": "low", "protocol": "mqtt"})

        self.assertAlmostEqual(self.simulator.similarity(first, second), 1.0)

    def test_record_can_retrieve_bound_value_through_cleanup_memory(self):
        record = self.encoder.encode(
            {
                "protocol": "mqtt",
                "encryption": "enabled",
                "rate": "low",
            }
        )
        protocol_role = self.encoder.item_memory.get("field:protocol")
        noisy_protocol = self.simulator.bind(record, protocol_role)
        candidates = AssociativeMemory(self.simulator)
        candidates.add(
            "mqtt",
            self.encoder.item_memory.get('value:protocol:"mqtt"'),
        )
        candidates.add(
            "telnet",
            self.encoder.item_memory.get('value:protocol:"telnet"'),
        )

        match = candidates.query(noisy_protocol)

        self.assertEqual(match.label, "mqtt")
        self.assertGreater(match.similarity, 0)

    def test_empty_record_and_invalid_feature_name_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.encoder.encode({})

        with self.assertRaisesRegex(ValueError, "feature name"):
            self.encoder.encode({"": "value"})


if __name__ == "__main__":
    unittest.main()
