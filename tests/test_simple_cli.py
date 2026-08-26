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


if __name__ == "__main__":
    unittest.main()
