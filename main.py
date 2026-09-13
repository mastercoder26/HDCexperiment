"""Command-line interface for the HDC baseline classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from hdc import DEFAULT_DIMENSIONS, HDC, CostProfile, HDCClassifier, __version__
from hdc.core import OPERATIONS
from hdc.costs import available_cost_profiles, get_cost_profile
from hdc.data import LabeledRecord, load_dataset

Result = dict[str, Any]

EXAMPLE_DATASET: list[LabeledRecord] = [
    ("class_a", {"color": "red", "shape": "circle", "size": "small"}),
    ("class_a", {"color": "orange", "shape": "circle", "size": "small"}),
    ("class_a", {"color": "red", "shape": "square", "size": "medium"}),
    ("class_b", {"color": "blue", "shape": "triangle", "size": "large"}),
    ("class_b", {"color": "green", "shape": "triangle", "size": "large"}),
    ("class_b", {"color": "blue", "shape": "square", "size": "medium"}),
]

EXAMPLE_TEST_RECORDS: list[LabeledRecord] = [
    ("class_a", {"color": "orange", "shape": "square", "size": "small"}),
    ("class_a", {"color": "red", "shape": "circle", "size": "medium"}),
    ("class_b", {"color": "green", "shape": "square", "size": "large"}),
    ("class_b", {"color": "blue", "shape": "triangle", "size": "medium"}),
]


def parse_int_list(raw_value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    values: list[int] = []
    for chunk in raw_value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        try:
            value = int(stripped)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"expected integer, got {chunk!r}"
            ) from error
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"expected positive integer, got {value}"
            )
        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError(
            "expected at least one positive integer"
        )
    return values


def read_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hyperdimensional Computing (HDC) baseline classifier.\n"
            "A brain-inspired, ultra-lightweight classification system "
            "designed for IoT and edge computing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hdc-baseline {__version__}",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--demo",
        action="store_true",
        help="explain what HDC is, walk through the built-in dataset, and run the classification",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"hypervector length / dimensions (default: {DEFAULT_DIMENSIONS:,})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for repeatable hypervector generation (default: 42)",
    )
    parser.add_argument(
        "--sweep-dimensions",
        type=parse_int_list,
        default=None,
        help="comma-separated dimensions for an experiment sweep (e.g. 1000,5000)",
    )
    parser.add_argument(
        "--sweep-seeds",
        type=parse_int_list,
        default=None,
        help="comma-separated seeds for an experiment sweep (e.g. 7,42)",
    )
    parser.add_argument(
        "--cost-profile",
        choices=available_cost_profiles(),
        default="unconfigured",
        help="named operation-specific analytical cost assumptions (default: unconfigured)",
    )
    parser.add_argument(
        "--energy-cost",
        type=float,
        default=None,
        help="uniform picojoules per work unit; overrides the named profile",
    )
    parser.add_argument(
        "--latency-cost",
        type=float,
        default=None,
        help="uniform nanoseconds per work unit; overrides the named profile",
    )
    mode_group.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="optional JSON file containing custom training and test records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON output path to save complete experiment metrics",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="optional CSV path for prediction rows or sweep summaries",
    )
    return parser.parse_args()


def _label_status(correct: bool) -> str:
    return "correct" if correct else "WRONG"


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def print_run_intro(
    training_records: Sequence[LabeledRecord],
    test_records: Sequence[LabeledRecord],
    dimensions: int,
) -> None:
    labels = sorted({label for label, _ in training_records})
    print("=" * 50)
    print("  HYPERDIMENSIONAL COMPUTING (HDC) CLASSIFIER")
    print("=" * 50)
    print()
    print("What is Hyperdimensional Computing?")
    print("  HDC is a brain-inspired computing method that encodes data into")
    print(f"  high-dimensional hypervectors ({dimensions:,} numbers of -1 and +1).")
    print("  Instead of training deep neural networks with backpropagation,")
    print("  HDC learns in a single fast pass by combining vectors with simple")
    print("  math (bind & bundle), and classifies by vector similarity.")
    print()
    print("What this run is doing:")
    print(f"  1. Learning from {len(training_records)} examples across {len(labels)} classes ({', '.join(labels)}).")
    print("  2. Bundling learned patterns into an average prototype vector for each class.")
    print(f"  3. Testing on {len(test_records)} held-out examples to measure accuracy and confidence.")
    print()


def print_demo_intro(
    training_records: Sequence[LabeledRecord],
    test_records: Sequence[LabeledRecord],
) -> None:
    labels = sorted({label for label, _ in training_records})
    feature_names = sorted(
        {name for _, record in training_records for name in record}
    )
    label_text = " and ".join(labels)
    if len(feature_names) > 2:
        feature_text = (
            ", ".join(feature_names[:-1]) + f", and {feature_names[-1]}"
        )
    else:
        feature_text = " and ".join(feature_names)

    print("=" * 50)
    print("  HDC CLASSIFIER — DEMO")
    print("=" * 50)
    print()
    print("What is Hyperdimensional Computing?")
    print("  HDC is a brain-inspired computing method that encodes patterns into")
    print("  long vectors of numbers (hypervectors). Instead of heavy neural")
    print("  networks, it combines patterns with simple math (bind & bundle) and")
    print("  classifies by comparing similarity—ideal for low-power edge devices.")
    print()
    print("What this does:")
    print(f"  It learns to tell {label_text} apart by looking at {feature_text}.")
    print()
    print("Training examples (what it learns from):")
    for label, record in training_records:
        features = ", ".join(
            f"{name}={record[name]}" for name in feature_names if name in record
        )
        print(f"    {label}: {features}")

    print()
    print("How it works:")
    print("    1. Each record is turned into a long list of numbers.")
    print("    2. Records of the same type are grouped into an average.")
    print("    3. New records are compared to those averages to guess their type.")
    print()
    print(f"    Training data: {len(training_records)} examples")
    print(f"    Test data:     {len(test_records)} examples")

    print()
    print("Test examples (the computer will try to guess these):")
    for index, (_, record) in enumerate(test_records, start=1):
        features = ", ".join(
            f"{name}={record[name]}" for name in feature_names if name in record
        )
        print(f"    {index}. {features}")
    print()
    print("=" * 50)
    print()


def print_result(result: Result) -> None:
    accuracy = result["accuracy"]
    filled = round(accuracy * 20)
    bar = "#" * filled + "-" * (20 - filled)

    print("=" * 50)
    print("  HDC CLASSIFIER — RESULTS")
    print("=" * 50)
    print()
    print(f"  Dimensions: {result['dimensions']:,}")
    print(f"  Random seed: {result['seed']}")
    print()

    dataset = result["dataset"]
    print("  Dataset:")
    print(
        f"    {dataset['training_records']} training examples, "
        f"{dataset['test_records']} test examples, "
        f"{len(dataset['classes'])} types"
    )
    print()

    print("  How well it did:")
    print(f"    Correct:    {result['correct']} out of {result['total']}")
    print(f"    Accuracy:   {_format_percent(accuracy)}")
    print(f"    Confidence: [{bar}] {_format_percent(accuracy)}")
    print(f"    Average margin: {result['average_margin']:.3f}")
    print(f"    Overall score (F1): {result['macro_f1']:.3f}")
    print()

    model = result["model"]
    memory_bytes = model["memory_bytes"]
    print("  Model size:")
    print(f"    Learned patterns: {model['item_vectors']}")
    print(f"    Type averages:    {model['prototype_vectors']}")
    print(f"    Memory used:      {memory_bytes / 1_024:.1f} KiB ({memory_bytes} bytes)")
    print()

    report = result["classification_report"]
    print("  Breakdown by type:")
    for label, metrics in report.items():
        print(
            f"    {label}: "
            f"precision={_format_percent(metrics['precision'])} "
            f"recall={_format_percent(metrics['recall'])} "
            f"F1={_format_percent(metrics['f1'])} "
            f"examples={metrics['support']}"
        )
    print()

    confusion_matrix = result["confusion_matrix"]
    labels = confusion_matrix["labels"]
    label_width = max((len(label) for label in labels), default=7)
    label_width = max(label_width, 7)
    cell_width = max(label_width, 7)
    print("  Prediction grid (rows = actual type, columns = guessed type):")
    header = " " * (label_width + 1) + " ".join(
        f"{label:^{cell_width}}" for label in labels
    )
    print(f"    {header}")
    for label in labels:
        row = confusion_matrix["rows"][label]
        counts = " ".join(
            f"{row[predicted]:^{cell_width}}" for predicted in labels
        )
        print(f"    {label:<{label_width}} {counts}")
    print()

    print("  Individual predictions:")
    for prediction in result["predictions"]:
        status = _label_status(prediction["correct"])
        scores = ", ".join(
            f"{label}={_format_percent(score)}"
            for label, score in sorted(prediction["scores"].items())
        )
        print(
            f"    {status:<7} true={prediction['true_label']:<{label_width}} "
            f"guessed={prediction['predicted_label']:<{label_width}} "
            f"margin={prediction['margin']:.3f} ({scores})"
        )
    print()

    simulation = result["simulation"]
    timing = result["timing_ms"]
    print("  Performance:")
    print(
        f"    Total work units:    {simulation['total_work_units']:,}"
    )
    print(
        f"    Estimated energy:    {simulation['total_energy_pj']:.3f} pJ"
    )
    print(
        f"    Estimated latency:   {simulation['total_latency_ns']:.3f} ns"
    )
    print(
        f"    Training time:       {timing['training']:.3f} ms"
    )
    print(
        f"    Prediction time:     {timing['inference']:.3f} ms"
    )
    print()

    print("  Interpretation & what this means:")
    if accuracy == 1.0:
        print(
            f"    • Accuracy ({_format_percent(accuracy)}): Perfect score! "
            f"All {result['correct']} out of {result['total']} test examples were correctly identified."
        )
    elif accuracy >= 0.75:
        print(
            f"    • Accuracy ({_format_percent(accuracy)}): Strong performance. "
            f"The model got {result['correct']} out of {result['total']} test examples right."
        )
    else:
        print(
            f"    • Accuracy ({_format_percent(accuracy)}): The model made errors "
            f"({result['correct']}/{result['total']} correct). Increasing dimensions "
            f"or adding training examples helps separate overlapping classes."
        )

    margin = result["average_margin"]
    if margin >= 0.3:
        print(
            f"    • Confidence margin ({margin:.3f}): High. Winning classes had "
            "significantly higher cosine similarity than runners-up, indicating decisive predictions."
        )
    elif margin > 0.0:
        print(
            f"    • Confidence margin ({margin:.3f}): Moderate. Predictions were correct, "
            "but similarity scores between competing classes were relatively close."
        )
    else:
        print(
            f"    • Confidence margin ({margin:.3f}): Low. Class representations overlapped, "
            "indicating uncertain classification decisions."
        )

    work_units = simulation["total_work_units"]
    print(
        f"    • Edge efficiency: Completed in {timing['total']:.2f} ms total using {work_units:,} "
        f"work units and {memory_bytes / 1_024:.1f} KiB of RAM. Because HDC uses simple vector "
        "operations rather than deep neural network backpropagation, it is ideal for low-power edge and IoT chips."
    )
    print()
    print("=" * 50)


def write_csv_result(result: Result, path: Path) -> None:
    """Write prediction details or sweep summaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if result.get("mode") == "sweep":
        fieldnames = [
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
        ]
        rows = []
        for run in result["runs"]:
            simulation = run["simulation"]
            timing = run["timing_ms"]
            rows.append(
                {
                    "dimensions": run["dimensions"],
                    "seed": run["seed"],
                    "accuracy": run["accuracy"],
                    "macro_f1": run["macro_f1"],
                    "average_margin": run["average_margin"],
                    "memory_bytes": run["memory_bytes"],
                    "total_work_units": simulation["total_work_units"],
                    "total_energy_pj": simulation["total_energy_pj"],
                    "total_latency_ns": simulation["total_latency_ns"],
                    "training_ms": timing["training"],
                    "inference_ms": timing["inference"],
                }
            )
    else:
        labels = result["confusion_matrix"]["labels"]
        fieldnames = [
            "record",
            "true_label",
            "predicted_label",
            "correct",
            "margin",
            *(f"score_{label}" for label in labels),
        ]
        rows = []
        for index, prediction in enumerate(result["predictions"], start=1):
            scores = prediction["scores"]
            rows.append(
                {
                    "record": index,
                    "true_label": prediction["true_label"],
                    "predicted_label": prediction["predicted_label"],
                    "correct": prediction["correct"],
                    "margin": prediction["margin"],
                    **{f"score_{label}": scores.get(label, "") for label in labels},
                }
            )

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    dimensions: int,
    seed: int,
    training_records: Sequence[LabeledRecord],
    test_records: Sequence[LabeledRecord],
    cost_profile_name: str = "unconfigured",
    energy_cost: float | None = None,
    latency_cost: float | None = None,
) -> Result:
    """Train and evaluate one HDC configuration."""
    profile = get_cost_profile(cost_profile_name)
    energy_costs = profile.energy_pj
    latency_costs = profile.latency_ns
    if energy_cost is not None:
        energy_costs = {name: energy_cost for name in OPERATIONS}
    if latency_cost is not None:
        latency_costs = {name: latency_cost for name in OPERATIONS}
    hdc = HDC(
        dimensions=dimensions,
        seed=seed,
        energy_costs=energy_costs,
        latency_costs=latency_costs,
    )

    classifier = HDCClassifier(hdc)
    training_started = perf_counter()
    classifier.train(training_records)
    training_ms = (perf_counter() - training_started) * 1_000

    inference_started = perf_counter()
    evaluation = classifier.evaluate(test_records)
    inference_ms = (perf_counter() - inference_started) * 1_000
    memory_bytes = classifier.memory_bytes()
    labels = sorted(
        {label for label, _ in training_records}
        | {label for label, _ in test_records}
    )

    return {
        "experiment": "generic_categorical_baseline",
        "dimensions": dimensions,
        "seed": seed,
        **evaluation,
        "memory_bytes": memory_bytes,
        "dataset": {
            "training_records": len(training_records),
            "test_records": len(test_records),
            "classes": labels,
        },
        "model": {
            "item_vectors": len(classifier.item_memory),
            "prototype_vectors": len(classifier.prototypes),
            "memory_bytes": memory_bytes,
        },
        "simulation": hdc.report(),
        "timing_ms": {
            "training": training_ms,
            "inference": inference_ms,
            "total": training_ms + inference_ms,
        },
        "config": {
            "dimensions": dimensions,
            "seed": seed,
            "cost_profile": profile.name,
            "cost_profile_description": profile.description,
            "energy_cost": energy_cost,
            "latency_cost": latency_cost,
        },
    }


def run_experiment_sweep(
    dimensions_list: Sequence[int],
    seed_list: Sequence[int],
    training_records: Sequence[LabeledRecord],
    test_records: Sequence[LabeledRecord],
    cost_profile_name: str = "unconfigured",
    energy_cost: float | None = None,
    latency_cost: float | None = None,
) -> Result:
    """Run an experiment sweep across dimensions and random seeds."""
    runs: list[Result] = []
    for dimensions in dimensions_list:
        for seed in seed_list:
            runs.append(
                run_experiment(
                    dimensions=dimensions,
                    seed=seed,
                    training_records=training_records,
                    test_records=test_records,
                    cost_profile_name=cost_profile_name,
                    energy_cost=energy_cost,
                    latency_cost=latency_cost,
                )
            )

    best_run = max(
        runs,
        key=lambda run: (
            float(run["accuracy"]),
            float(run["average_margin"]),
            float(run["macro_f1"]),
        ),
    )
    average_accuracy = sum(float(run["accuracy"]) for run in runs) / len(runs)
    average_macro_f1 = sum(float(run["macro_f1"]) for run in runs) / len(runs)
    average_margin = sum(float(run["average_margin"]) for run in runs) / len(
        runs
    )

    return {
        "experiment": "generic_categorical_baseline",
        "mode": "sweep",
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "average_accuracy": average_accuracy,
            "average_macro_f1": average_macro_f1,
            "average_margin": average_margin,
            "best_run": {
                "dimensions": best_run["dimensions"],
                "seed": best_run["seed"],
                "accuracy": best_run["accuracy"],
                "average_margin": best_run["average_margin"],
            },
        },
    }


def main() -> int:
    args = read_arguments()
    is_sweep = bool(args.sweep_dimensions or args.sweep_seeds)

    if args.dataset:
        dataset = load_dataset(args.dataset)
        training_records = dataset["training"]
        test_records = dataset["test"]
    else:
        training_records = EXAMPLE_DATASET
        test_records = EXAMPLE_TEST_RECORDS

    if args.demo:
        print_demo_intro(training_records, test_records)
    elif not is_sweep:
        print_run_intro(training_records, test_records, args.dimensions)

    if is_sweep:
        sweep_dimensions = args.sweep_dimensions or [args.dimensions]
        sweep_seeds = args.sweep_seeds or [args.seed]
        result = run_experiment_sweep(
            dimensions_list=sweep_dimensions,
            seed_list=sweep_seeds,
            training_records=training_records,
            test_records=test_records,
            cost_profile_name=args.cost_profile,
            energy_cost=args.energy_cost,
            latency_cost=args.latency_cost,
        )
    else:
        result = run_experiment(
            dimensions=args.dimensions,
            seed=args.seed,
            training_records=training_records,
            test_records=test_records,
            cost_profile_name=args.cost_profile,
            energy_cost=args.energy_cost,
            latency_cost=args.latency_cost,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.csv_output:
        write_csv_result(result, args.csv_output)

    if is_sweep:
        summary = result["summary"]
        print("=" * 50)
        print("  HDC CLASSIFIER — EXPERIMENT SWEEP")
        print("=" * 50)
        print()
        print(f"  Runs: {summary['run_count']}")
        print(f"  Average accuracy:  {_format_percent(summary['average_accuracy'])}")
        print(f"  Average F1 score:  {summary['average_macro_f1']:.3f}")
        print(f"  Average margin:    {summary['average_margin']:.3f}")
        print()
        print("  Run details:")
        for run in result["runs"]:
            print(
                f"    dimensions={run['dimensions']} seed={run['seed']} "
                f"accuracy={_format_percent(run['accuracy'])} "
                f"F1={run['macro_f1']:.3f} "
                f"margin={run['average_margin']:.3f}"
            )
        best_run = summary["best_run"]
        print()
        print(
            f"  Best configuration: "
            f"dimensions={best_run['dimensions']}, "
            f"seed={best_run['seed']}, "
            f"accuracy={_format_percent(best_run['accuracy'])}, "
            f"margin={best_run['average_margin']:.3f}"
        )
        print()
        print("  Interpretation & sweep insights:")
        print(
            f"    • Top performer: dimensions={best_run['dimensions']} with seed={best_run['seed']} "
            f"achieved {_format_percent(best_run['accuracy'])} accuracy and {best_run['average_margin']:.3f} margin."
        )
        print(
            "    • Dimensionality: In HDC, higher dimensions provide more orthogonal vector capacity, "
            "reducing interference between learned patterns."
        )
        print()
        print("=" * 50)
    else:
        print_result(result)

    if args.output:
        print(f"\nSaved results: {args.output}")
    if args.csv_output:
        print(f"Saved CSV: {args.csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
