import unittest

from hdc.costs import CostModel, OperationCost


class CostModelTests(unittest.TestCase):
    def test_operation_cost_estimates_fixed_and_variable_components(self):
        cost = OperationCost(
            fixed_energy_pj=2.0,
            energy_pj_per_work_unit=0.5,
            fixed_latency_ns=3.0,
            latency_ns_per_work_unit=0.25,
        )

        estimate = cost.estimate(100)

        self.assertEqual(estimate.energy_pj, 52.0)
        self.assertEqual(estimate.latency_ns, 28.0)

    def test_cost_model_returns_zero_for_unconfigured_operation(self):
        model = CostModel(name="empty")

        estimate = model.estimate("bind", 1_000)

        self.assertEqual(estimate.energy_pj, 0.0)
        self.assertEqual(estimate.latency_ns, 0.0)

    def test_uniform_cost_model_applies_to_all_supported_operations(self):
        model = CostModel.uniform(
            name="example",
            energy_pj_per_work_unit=0.5,
            latency_ns_per_work_unit=1.5,
        )

        for operation in ("random", "bind", "bundle", "permute", "similarity"):
            estimate = model.estimate(operation, 10)
            self.assertEqual(estimate.energy_pj, 5.0)
            self.assertEqual(estimate.latency_ns, 15.0)

    def test_cost_values_and_work_units_must_be_nonnegative(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            OperationCost(fixed_energy_pj=-1)

        with self.assertRaisesRegex(ValueError, "work_units"):
            OperationCost().estimate(-1)

    def test_cost_model_can_be_serialized(self):
        model = CostModel(
            name="paper-profile",
            operation_costs={"bind": OperationCost(energy_pj_per_work_unit=2)},
        )

        payload = model.to_dict()

        self.assertEqual(payload["name"], "paper-profile")
        self.assertEqual(
            payload["operations"]["bind"]["energy_pj_per_work_unit"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
