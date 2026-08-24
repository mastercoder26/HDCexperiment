import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hdc import HDCConfig
from hdc.costs import CostModel
from hdc.encoding import RecordEncoder
from hdc.experiment import (
    LabeledRecord,
    PrototypeClassifier,
    run_iot_baseline,
)
from hdc.simulator import HDCSimulator


class PrototypeClassifierTests(unittest.TestCase):
    def test_classifier_learns_symbolic_records(self):
        simulator = HDCSimulator(HDCConfig(dimensions=4_000, seed=16))
        classifier = PrototypeClassifier(RecordEncoder(simulator))
        classifier.fit(
            [
                LabeledRecord("normal", {"protocol": "mqtt", "rate": "low"}),
                LabeledRecord("normal", {"protocol": "coap", "rate": "low"}),
                LabeledRecord("anomaly", {"protocol": "telnet", "rate": "high"}),
                LabeledRecord("anomaly", {"protocol": "ftp", "rate": "high"}),
            ]
        )

        prediction = classifier.predict({"protocol": "mqtt", "rate": "low"})

        self.assertEqual(prediction.label, "normal")
        self.assertGreater(prediction.similarity, 0)

    def test_classifier_rejects_prediction_before_training(self):
        simulator = HDCSimulator(HDCConfig(dimensions=1_000, seed=17))
        classifier = PrototypeClassifier(RecordEncoder(simulator))

        with self.assertRaisesRegex(RuntimeError, "fit"):
            classifier.predict({"protocol": "mqtt"})

    def test_classifier_rejects_empty_training_data(self):
        simulator = HDCSimulator(HDCConfig(dimensions=1_000, seed=18))
        classifier = PrototypeClassifier(RecordEncoder(simulator))

        with self.assertRaisesRegex(ValueError, "training"):
            classifier.fit([])


class BaselineExperimentTests(unittest.TestCase):
    def test_iot_baseline_produces_repeatable_preliminary_results(self):
        config = HDCConfig(dimensions=4_000, seed=19)

        first = run_iot_baseline(config)
        second = run_iot_baseline(config)

        self.assertEqual(first.accuracy, second.accuracy)
        self.assertEqual(
            [prediction.predicted_label for prediction in first.predictions],
            [prediction.predicted_label for prediction in second.predictions],
        )
        self.assertGreaterEqual(first.accuracy, 0.75)
        self.assertEqual(len(first.predictions), 4)
        self.assertIn("similarity", first.simulation_report.operations)

    def test_experiment_result_can_exclude_nondeterministic_runtime(self):
        result = run_iot_baseline(HDCConfig(dimensions=1_000, seed=20))

        payload = result.to_dict(include_measured_runtime=False)

        self.assertNotIn(
            "measured_runtime_ns",
            payload["simulation"]["operations"]["random"],
        )
        json.dumps(payload)

    def test_uniform_cost_profile_reaches_experiment_report(self):
        cost_model = CostModel.uniform(
            name="example-device",
            energy_pj_per_work_unit=0.1,
            latency_ns_per_work_unit=0.2,
        )

        result = run_iot_baseline(
            HDCConfig(dimensions=1_000, seed=21),
            cost_model=cost_model,
        )

        self.assertEqual(result.simulation_report.cost_profile, "example-device")
        self.assertGreater(result.simulation_report.total_modeled_energy_pj, 0)

    def test_cli_runs_baseline_and_writes_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "result.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--dimensions",
                    "1000",
                    "--seed",
                    "22",
                    "--output",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("HDC IoT baseline", completed.stdout)
        self.assertEqual(payload["config"]["dimensions"], 1_000)
        self.assertEqual(len(payload["predictions"]), 4)


if __name__ == "__main__":
    unittest.main()
