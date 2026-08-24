import unittest

import numpy as np

from hdc import bind, bundle, permute, random_vector, similarity


class HypervectorOperationTests(unittest.TestCase):
    def test_random_vector_is_reproducible_and_configurable(self):
        first = random_vector(dimensions=1_024, seed=42)
        second = random_vector(dimensions=1_024, seed=42)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (1_024,))
        self.assertEqual(first.dtype, np.int8)
        self.assertEqual(set(np.unique(first)), {-1, 1})

    def test_random_vector_accepts_an_existing_generator(self):
        first_rng = np.random.default_rng(7)
        second_rng = np.random.default_rng(7)

        first = random_vector(dimensions=256, rng=first_rng)
        second = random_vector(dimensions=256, rng=second_rng)

        np.testing.assert_array_equal(first, second)

    def test_random_vector_rejects_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            random_vector(dimensions=0)

        with self.assertRaisesRegex(ValueError, "either rng or seed"):
            random_vector(
                dimensions=100,
                rng=np.random.default_rng(1),
                seed=1,
            )

    def test_binding_is_self_inverse_for_bipolar_vectors(self):
        first = random_vector(dimensions=1_024, seed=1)
        second = random_vector(dimensions=1_024, seed=2)

        recovered = bind(bind(first, second), first)

        np.testing.assert_array_equal(recovered, second)

    def test_bundle_handles_multiple_vectors(self):
        first = random_vector(dimensions=2_000, seed=3)
        second = random_vector(dimensions=2_000, seed=4)
        third = random_vector(dimensions=2_000, seed=5)

        bundled = bundle(first, second, third)

        self.assertEqual(set(np.unique(bundled)), {-1, 1})
        self.assertGreater(similarity(bundled, first), 0)
        self.assertGreater(similarity(bundled, second), 0)
        self.assertGreater(similarity(bundled, third), 0)

    def test_bundle_randomly_resolves_even_ties_without_zeros(self):
        vector = random_vector(dimensions=1_024, seed=6)

        bundled = bundle(
            vector,
            -vector,
            tie_breaker="random",
            rng=np.random.default_rng(8),
        )

        self.assertEqual(set(np.unique(bundled)), {-1, 1})

    def test_bundle_positive_policy_resolves_ties_to_positive(self):
        vector = np.array([1, -1, 1, -1], dtype=np.int8)

        bundled = bundle(vector, -vector, tie_breaker="positive")

        np.testing.assert_array_equal(
            bundled,
            np.ones(vector.shape, dtype=np.int8),
        )

    def test_bundle_validates_tie_configuration(self):
        vector = random_vector(dimensions=128, seed=9)

        with self.assertRaisesRegex(ValueError, "random-number generator"):
            bundle(vector, -vector, tie_breaker="random")

        with self.assertRaisesRegex(ValueError, "tie_breaker"):
            bundle(vector, -vector, tie_breaker="unsupported")

        with self.assertRaisesRegex(ValueError, "at least one"):
            bundle()

    def test_permutation_can_be_reversed(self):
        vector = random_vector(dimensions=1_024, seed=10)

        shifted = permute(vector, shifts=17)
        restored = permute(shifted, shifts=-17)

        np.testing.assert_array_equal(restored, vector)

    def test_similarity_has_expected_bounds(self):
        vector = random_vector(dimensions=1_024, seed=11)

        self.assertAlmostEqual(similarity(vector, vector), 1.0)
        self.assertAlmostEqual(similarity(vector, -vector), -1.0)

    def test_operations_reject_invalid_vectors(self):
        invalid = np.array([1, 0, -1], dtype=np.int8)
        short = np.array([1, -1], dtype=np.int8)

        with self.assertRaisesRegex(ValueError, "bipolar"):
            similarity(invalid, invalid)

        with self.assertRaisesRegex(ValueError, "same shape"):
            bind(invalid, short)

        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            permute(np.ones((2, 2), dtype=np.int8))


if __name__ == "__main__":
    unittest.main()
