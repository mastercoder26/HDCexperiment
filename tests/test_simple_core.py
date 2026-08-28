import unittest

import numpy as np

from hdc.core import DEFAULT_DIMENSIONS, HDC


class SimpleHDCTests(unittest.TestCase):
    def test_dimension_from_constructor_controls_vector_size(self):
        hdc = HDC(dimensions=30, seed=42)

        vector = hdc.random_vector()

        self.assertEqual(len(vector), 30)
        self.assertEqual(hdc.report()["operations"]["random"]["work_units"], 30)

    def test_default_dimension_has_one_shared_value(self):
        hdc = HDC(seed=42)

        self.assertEqual(hdc.dimensions, DEFAULT_DIMENSIONS)

    def test_seed_makes_vectors_repeatable(self):
        first = HDC(dimensions=100, seed=7).random_vector()
        second = HDC(dimensions=100, seed=7).random_vector()

        np.testing.assert_array_equal(first, second)

    def test_bind_can_be_reversed(self):
        hdc = HDC(dimensions=100, seed=8)
        first = hdc.random_vector()
        second = hdc.random_vector()

        recovered = hdc.bind(hdc.bind(first, second), first)

        np.testing.assert_array_equal(recovered, second)

    def test_bundle_stays_bipolar_and_similar_to_inputs(self):
        hdc = HDC(dimensions=1_000, seed=9)
        first = hdc.random_vector()
        second = hdc.random_vector()
        third = hdc.random_vector()

        combined = hdc.bundle(first, second, third)

        self.assertEqual(set(np.unique(combined)), {-1, 1})
        self.assertGreater(hdc.similarity(combined, first), 0)
        self.assertGreater(hdc.similarity(combined, second), 0)
        self.assertGreater(hdc.similarity(combined, third), 0)

    def test_permutation_can_be_reversed(self):
        hdc = HDC(dimensions=100, seed=10)
        vector = hdc.random_vector()

        restored = hdc.permute(hdc.permute(vector, 5), -5)

        np.testing.assert_array_equal(restored, vector)

    def test_weighted_bundle_can_prioritize_one_vector(self):
        hdc = HDC(dimensions=3, seed=10)
        preferred = np.array([1, 1, 1], dtype=np.int8)
        other = np.array([-1, -1, -1], dtype=np.int8)

        combined = hdc.bundle(preferred, other, weights=[2.0, 1.0])

        np.testing.assert_array_equal(combined, preferred)

    def test_report_uses_changeable_operation_costs(self):
        hdc = HDC(
            dimensions=10,
            seed=11,
            energy_costs={"bind": 0.5},
            latency_costs={"bind": 2.0},
        )
        first = hdc.random_vector()
        second = hdc.random_vector()

        hdc.bind(first, second)
        report = hdc.report()

        self.assertEqual(report["operations"]["bind"]["calls"], 1)
        self.assertEqual(report["operations"]["bind"]["work_units"], 10)
        self.assertEqual(report["operations"]["bind"]["energy_pj"], 5.0)
        self.assertEqual(report["operations"]["bind"]["latency_ns"], 20.0)

    def test_bad_dimensions_and_vectors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            HDC(dimensions=0)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            HDC(dimensions=True)

        hdc = HDC(dimensions=3)
        with self.assertRaisesRegex(ValueError, "bipolar"):
            hdc.bind(
                np.array([1, 0, -1]),
                np.array([1, 1, -1]),
            )

    def test_bundle_rejects_invalid_weights(self):
        hdc = HDC(dimensions=3, seed=12)
        first = np.array([1, 1, -1], dtype=np.int8)
        second = np.array([-1, 1, 1], dtype=np.int8)

        with self.assertRaisesRegex(ValueError, "weights"):
            hdc.bundle(first, second, weights=[1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            hdc.bundle(first, second, weights=[1.0, np.nan])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            hdc.bundle(first, second, weights=[1.0, -1.0])
        with self.assertRaisesRegex(ValueError, "positive"):
            hdc.bundle(first, second, weights=[0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
