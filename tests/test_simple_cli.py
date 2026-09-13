import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SimpleCommandLineTests(unittest.TestCase):
    def test_command_line_reports_package_version(self):
        completed = subprocess.run(
            [sys.executable, "main.py", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout.strip(), "hdc-baseline 1.4.0")

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

        self.assertIn("HDC CLASSIFIER — RESULTS", completed.stdout)
        self.assertIn("Dimensions: 30", completed.stdout)
        self.assertEqual(result["experiment"], "generic_categorical_baseline")
        self.assertEqual(result["dimensions"], 30)
        self.assertEqual(result["total"], 4)
        self.assertIn("operations", result["simulation"])
        self.assertGreaterEqual(result["timing_ms"]["training"], 0.0)
        self.assertGreaterEqual(result["timing_ms"]["inference"], 0.0)
        self.assertGreaterEqual(result["timing_ms"]["total"], 0.0)
        self.assertIn("Training time:", completed.stdout)
        self.assertIn("Average margin:", completed.stdout)
        self.assertIn("Overall score", completed.stdout)
        self.assertIn("Breakdown by type:", completed.stdout)
        self.assertIn("Prediction grid", completed.stdout)
        self.assertIn("6 training examples", completed.stdout)
        self.assertIn("Confidence:", completed.stdout)
        self.assertIn("correct true=", completed.stdout)
        self.assertIn("WRONG   true=", completed.stdout)
        self.assertIn(f"Saved results: {output}", completed.stdout)
        self.assertEqual(result["dataset"]["training_records"], 6)
        self.assertEqual(result["dataset"]["test_records"], 4)
        self.assertEqual(result["dataset"]["classes"], ["class_a", "class_b"])
        self.assertGreater(result["model"]["item_vectors"], 0)
        self.assertEqual(result["model"]["prototype_vectors"], 2)
        self.assertEqual(result["model"]["memory_bytes"], result["memory_bytes"])
        self.assertIn("Model size:", completed.stdout)
        self.assertIn("Learned patterns:", completed.stdout)
        self.assertIn("Type averages:", completed.stdout)
        self.assertIn("Memory used:", completed.stdout)

    def test_command_line_exports_predictions_to_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_output = Path(temporary_directory) / "predictions.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--dimensions",
                    "30",
                    "--seed",
                    "42",
                    "--csv-output",
                    str(csv_output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with csv_output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            list(rows[0]),
            [
                "record",
                "true_label",
                "predicted_label",
                "correct",
                "margin",
                "score_class_a",
                "score_class_b",
            ],
        )
        self.assertEqual(rows[0]["record"], "1")
        self.assertEqual(rows[0]["true_label"], "class_a")
        self.assertIn(rows[0]["correct"], {"True", "False"})
        self.assertIn(f"Saved CSV: {csv_output}", completed.stdout)

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
        self.assertIn("average_macro_f1", result["summary"])
        self.assertIn("average_margin", result["summary"])
        self.assertIn("best_run", result["summary"])
        self.assertEqual(
            set(result["summary"]["best_run"]),
            {"dimensions", "seed", "accuracy", "average_margin"},
        )
        self.assertIn("HDC CLASSIFIER — EXPERIMENT SWEEP", completed.stdout)
        self.assertIn("Runs: 4", completed.stdout)
        self.assertIn("Average accuracy:", completed.stdout)
        self.assertIn("Average F1 score:", completed.stdout)
        self.assertIn("Run details:", completed.stdout)
        self.assertIn("dimensions=30 seed=1", completed.stdout)
        self.assertIn("dimensions=60 seed=2", completed.stdout)
        self.assertIn("Best configuration:", completed.stdout)
        self.assertIn(f"Saved results: {output}", completed.stdout)
        self.assertIn("=" * 50, completed.stdout)

    def test_command_line_exports_sweep_summary_to_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            csv_output = Path(temporary_directory) / "sweep.csv"
            completed = subprocess.run(
                [
                    sys.executable,
                    "main.py",
                    "--sweep-dimensions",
                    "30,60",
                    "--sweep-seeds",
                    "1,2",
                    "--csv-output",
                    str(csv_output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with csv_output.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            list(rows[0]),
            [
                "dimensions",
                "seed",
                "accuracy",
                "macro_f1",
                "average_margin",
                "memory_bytes",
                "total_work_units",
                "total_energy_pj",
                "total_latency_ns",
                "training_ms",
                "inference_ms",
            ],
        )
        self.assertEqual(
            {(row["dimensions"], row["seed"]) for row in rows},
            {("30", "1"), ("30", "2"), ("60", "1"), ("60", "2")},
        )
        self.assertIn(f"Saved CSV: {csv_output}", completed.stdout)

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

        self.assertIn("HDC CLASSIFIER — DEMO", completed.stdout)
        self.assertIn(
            "It learns to tell class_a and class_b apart",
            completed.stdout,
        )
        self.assertIn("Training examples (what it learns from):", completed.stdout)
        self.assertIn(
            "class_a: color=red, shape=circle, size=small", completed.stdout
        )
        self.assertIn(
            "class_b: color=blue, shape=square, size=medium", completed.stdout
        )
        self.assertIn("How it works:", completed.stdout)
        self.assertIn("1. Each record is turned into a long list of numbers.", completed.stdout)
        self.assertIn(
            "2. Records of the same type are grouped into an average.", completed.stdout
        )
        self.assertIn(
            "3. New records are compared to those averages to guess their type.", completed.stdout
        )
        self.assertIn("Training data: 6 examples", completed.stdout)
        self.assertIn("Test data:     4 examples", completed.stdout)
        self.assertIn("Test examples (the computer will try to guess these):", completed.stdout)
        self.assertIn("color=orange, shape=square, size=small", completed.stdout)
        self.assertIn("color=blue, shape=triangle, size=medium", completed.stdout)
        self.assertIn("=" * 50, completed.stdout)

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
