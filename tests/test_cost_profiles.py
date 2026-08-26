import unittest

from hdc.core import OPERATIONS
from hdc.costs import available_cost_profiles, get_cost_profile


class CostProfileTests(unittest.TestCase):
    def test_unconfigured_profile_has_zero_costs(self):
        profile = get_cost_profile("unconfigured")

        self.assertEqual(profile.name, "unconfigured")
        self.assertEqual(set(profile.energy_pj), set(OPERATIONS))
        self.assertEqual(set(profile.latency_ns), set(OPERATIONS))
        self.assertTrue(all(cost == 0.0 for cost in profile.energy_pj.values()))
        self.assertTrue(all(cost == 0.0 for cost in profile.latency_ns.values()))

    def test_example_edge_profile_is_operation_specific(self):
        profile = get_cost_profile("example_edge")

        self.assertTrue(all(cost > 0.0 for cost in profile.energy_pj.values()))
        self.assertTrue(all(cost > 0.0 for cost in profile.latency_ns.values()))
        self.assertGreater(len(set(profile.energy_pj.values())), 1)
        self.assertGreater(len(set(profile.latency_ns.values())), 1)

    def test_available_profiles_are_listed(self):
        self.assertEqual(
            available_cost_profiles(),
            ("unconfigured", "example_edge"),
        )

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown cost profile"):
            get_cost_profile("missing")


if __name__ == "__main__":
    unittest.main()
