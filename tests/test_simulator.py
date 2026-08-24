import json
import unittest

import numpy as np

from hdc import HDCConfig, HDCSimulator
from hdc.costs import CostModel, OperationCost


class HDCSimulatorTests(unittest.TestCase):
    def setUp(self):
        cost_model = CostModel(
            name="test-profile",
            operation_costs={
                "random": OperationCost(
                    fixed_energy_pj=2.0,
                    energy_pj_per_work_unit=0.5,
                    fixed_latency_ns=3.0,
                    latency_ns_per_work_unit=0.25,
                ),
                "bind": OperationCost(
                    energy_pj_per_work_unit=1.0,
                    latency_ns_per_work_unit=2.0,
                ),
            },
        )
        self.simulator = HDCSimulator(
            HDCConfig(dimensions=100, seed=12),
            cost_model=cost_model,
        )

    def test_seed_makes_simulation_reproducible(self):
        other = HDCSimulator(HDCConfig(dimensions=100, seed=12))

        np.testing.assert_array_equal(
            self.simulator.random_hypervector(),
            other.random_hypervector(),
        )

    def test_report_tracks_calls_work_and_modeled_costs(self):
        first = self.simulator.random_hypervector()
        second = self.simulator.random_hypervector()
        self.simulator.bind(first, second)

        report = self.simulator.report()

        self.assertEqual(report.operations["random"].calls, 2)
        self.assertEqual(report.operations["random"].work_units, 200)
        self.assertEqual(report.operations["random"].modeled_energy_pj, 104.0)
        self.assertEqual(report.operations["random"].modeled_latency_ns, 56.0)
        self.assertEqual(report.operations["bind"].calls, 1)
        self.assertEqual(report.operations["bind"].work_units, 100)
        self.assertEqual(report.total_modeled_energy_pj, 204.0)
        self.assertEqual(report.total_modeled_latency_ns, 256.0)
        self.assertGreater(report.total_measured_runtime_ns, 0)

    def test_all_operations_are_instrumented(self):
        first = self.simulator.random_hypervector()
        second = self.simulator.random_hypervector()
        third = self.simulator.random_hypervector()

        bound = self.simulator.bind(first, second)
        bundled = self.simulator.bundle(first, second, third)
        shifted = self.simulator.permute(bound, shifts=2)
        self.simulator.similarity(bundled, shifted)

        report = self.simulator.report()

        self.assertEqual(
            set(report.operations),
            {"random", "bind", "bundle", "permute", "similarity"},
        )
        self.assertEqual(report.operations["bundle"].work_units, 200)

    def test_reset_metrics_keeps_configuration_but_clears_report(self):
        self.simulator.random_hypervector()

        self.simulator.reset_metrics()

        report = self.simulator.report()
        self.assertEqual(report.operations, {})
        self.assertEqual(report.config.dimensions, 100)

    def test_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "dimensions"):
            HDCConfig(dimensions=0)

        with self.assertRaisesRegex(ValueError, "tie_breaker"):
            HDCConfig(tie_breaker="unsupported")

    def test_simulator_rejects_wrong_vector_dimension(self):
        wrong_size = np.ones(10, dtype=np.int8)

        with self.assertRaisesRegex(ValueError, "configured dimension"):
            self.simulator.permute(wrong_size)

    def test_report_is_json_serializable(self):
        self.simulator.random_hypervector()

        payload = self.simulator.report().to_dict()
        serialized = json.dumps(payload)

        self.assertIn("test-profile", serialized)
        self.assertEqual(payload["config"]["dimensions"], 100)
        self.assertIn("totals", payload)


if __name__ == "__main__":
    unittest.main()
