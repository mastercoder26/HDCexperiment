import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SimpleCommandLineTests(unittest.TestCase):
    def test_command_line_dimension_reaches_saved_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "result.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--dimensions",
                    "30",
                    "--seed",
                    "42",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertIn("dimensions: 30", completed.stdout)
        self.assertIn("Generic HDC baseline", completed.stdout)
        self.assertEqual(result["experiment"], "generic_categorical_baseline")
        self.assertEqual(result["dimensions"], 30)
        self.assertEqual(result["total"], 4)
        self.assertIn("operations", result["simulation"])
        self.assertGreaterEqual(result["timing_ms"]["training"], 0.0)
        self.assertGreaterEqual(result["timing_ms"]["inference"], 0.0)
        self.assertGreaterEqual(result["timing_ms"]["total"], 0.0)
        self.assertIn("training time:", completed.stdout)
        self.assertIn("average margin:", completed.stdout)
        self.assertEqual(result["dataset"]["training_records"], 6)
        self.assertEqual(result["dataset"]["test_records"], 4)
        self.assertEqual(result["dataset"]["classes"], ["class_a", "class_b"])
        self.assertGreater(result["model"]["item_vectors"], 0)
        self.assertEqual(result["model"]["prototype_vectors"], 2)
        self.assertEqual(result["model"]["memory_bytes"], result["memory_bytes"])

    def test_command_line_runs_dimension_and_seed_sweep(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "sweep.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--sweep-dimensions",
                    "30,60",
                    "--sweep-seeds",
                    "1,2",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result["mode"], "sweep")
        self.assertEqual(len(result["runs"]), 4)
        self.assertEqual(
            {(run["dimensions"], run["seed"]) for run in result["runs"]},
            {(30, 1), (30, 2), (60, 1), (60, 2)},
        )
        self.assertIn("average_accuracy", result["summary"])
        self.assertIn("sweep runs: 4", completed.stdout)

    def test_command_line_demo_explains_the_built_in_example(self):
        completed = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--demo",
                "--dimensions",
                "64",
                "--seed",
                "42",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("HDC demo", completed.stdout)
        self.assertIn("1. Encode each record as a hypervector.", completed.stdout)
        self.assertIn(
            "2. Bundle the training records into class prototypes.", completed.stdout
        )
        self.assertIn(
            "3. Compare each test record with those prototypes.", completed.stdout
        )
        self.assertIn("training records: 6", completed.stdout)
        self.assertIn("test records:     4", completed.stdout)
        self.assertIn("accuracy:   4/4", completed.stdout)

    def test_command_line_uses_named_cost_profile(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "costs.json"
            subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--dimensions",
                    "100",
                    "--cost-profile",
                    "example_edge",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result["config"]["cost_profile"], "example_edge")
        self.assertGreater(result["simulation"]["total_energy_pj"], 0.0)
        self.assertGreater(result["simulation"]["total_latency_ns"], 0.0)


if __name__ == "__main__":
    unittest.main()
