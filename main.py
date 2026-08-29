"""Run a generic HDC classifier on a small categorical example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from hdc.core import DEFAULT_DIMENSIONS, OPERATIONS, HDC
from hdc.costs import available_cost_profiles, get_cost_profile
from hdc.data import load_dataset
from hdc.model import HDCClassifier


TRAINING_RECORDS = [
    ("class_a", {"color": "red", "shape": "circle", "size": "small"}),
    ("class_a", {"color": "orange", "shape": "circle", "size": "small"}),
    ("class_a", {"color": "red", "shape": "square", "size": "medium"}),
    ("class_b", {"color": "blue", "shape": "triangle", "size": "large"}),
    ("class_b", {"color": "green", "shape": "triangle", "size": "large"}),
    ("class_b", {"color": "blue", "shape": "square", "size": "medium"}),
]

TEST_RECORDS = [
    ("class_a", {"color": "orange", "shape": "square", "size": "small"}),
    ("class_a", {"color": "red", "shape": "circle", "size": "medium"}),
    ("class_b", {"color": "green", "shape": "square", "size": "large"}),
    ("class_b", {"color": "blue", "shape": "triangle", "size": "medium"}),
]


def comma_separated_integers(value: str) -> list[int]:
    """Convert a comma-separated string into a list of integers."""
    try:
        numbers = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not numbers:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return numbers


def read_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HDC baseline classifier.")
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"hypervector length (default: {DEFAULT_DIMENSIONS})",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sweep-dimensions",
        type=comma_separated_integers,
        help="comma-separated dimensions for an experiment sweep",
    )
    parser.add_argument(
        "--sweep-seeds",
        type=comma_separated_integers,
        help="comma-separated seeds for an experiment sweep",
    )
    parser.add_argument(
        "--cost-profile",
        choices=available_cost_profiles(),
        default="unconfigured",
        help="named operation-specific analytical cost assumptions",
    )
    parser.add_argument(
        "--energy-cost",
        type=float,
        help="uniform picojoules per work unit; overrides the named profile",
    )
    parser.add_argument(
        "--latency-cost",
        type=float,
        help="uniform nanoseconds per work unit; overrides the named profile",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="optional JSON file containing training and test records",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def print_result(result: dict[str, object]) -> None:
    print("Generic HDC baseline")
    print(f"dimensions: {result['dimensions']}")
    print(f"seed:       {result['seed']}")
    print(
        f"accuracy:   {result['correct']}/{result['total']} "
        f"({result['accuracy']:.1%})"
    )
    print(f"average margin: {result['average_margin']:.3f}")
    print("per-class accuracy:")
    for label, stats in result["per_class"].items():
        print(
            f"  {label}: {stats['correct']}/{stats['total']} "
            f"({stats['accuracy']:.1%})"
        )

    print("\npredictions:")
    for prediction in result["predictions"]:
        scores = ", ".join(
            f"{label}={score:.3f}"
            for label, score in sorted(prediction["scores"].items())
        )
        print(
            f"  true={prediction['true_label']:<7} "
            f"predicted={prediction['predicted_label']:<7} "
            f"margin={prediction['margin']:.3f} ({scores})"
        )

    simulation = result["simulation"]
    print("\noperation counts:")
    for name, details in simulation["operations"].items():
        if details["calls"]:
            print(
                f"  {name:<10} calls={details['calls']:<3} "
                f"work_units={details['work_units']}"
            )
    print(
        "estimated totals: "
        f"{simulation['total_energy_pj']:.3f} pJ, "
        f"{simulation['total_latency_ns']:.3f} ns"
    )
    timing = result["timing_ms"]
    print(f"training time:  {timing['training']:.3f} ms")
    print(f"inference time: {timing['inference']:.3f} ms")


def run_experiment(
    dimensions: int,
    seed: int,
    training_records,
    test_records,
    cost_profile_name: str = "unconfigured",
    energy_cost: float | None = None,
    latency_cost: float | None = None,
) -> dict[str, object]:
    """Train and evaluate one fully isolated HDC configuration."""
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

    result = {
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
        },
    }
    return result


def run_sweep(
    args: argparse.Namespace,
    training_records,
    test_records,
) -> dict[str, object]:
    """Run every requested dimension and seed combination."""
    dimensions = args.sweep_dimensions or [args.dimensions]
    seeds = args.sweep_seeds or [args.seed]
    runs = [
        run_experiment(
            dimension,
            seed,
            training_records,
            test_records,
            cost_profile_name=args.cost_profile,
            energy_cost=args.energy_cost,
            latency_cost=args.latency_cost,
        )
        for dimension in dimensions
        for seed in seeds
    ]
    return {
        "mode": "sweep",
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "average_accuracy": sum(run["accuracy"] for run in runs) / len(runs),
            "average_margin": (
                sum(run["average_margin"] for run in runs) / len(runs)
            ),
        },
    }


def main() -> int:
    args = read_arguments()
    if args.dataset:
        training_records, test_records = load_dataset(args.dataset)
    else:
        training_records, test_records = TRAINING_RECORDS, TEST_RECORDS

    is_sweep = args.sweep_dimensions is not None or args.sweep_seeds is not None
    if is_sweep:
        result = run_sweep(args, training_records, test_records)
    else:
        result = run_experiment(
            args.dimensions,
            args.seed,
            training_records,
            test_records,
            cost_profile_name=args.cost_profile,
            energy_cost=args.energy_cost,
            latency_cost=args.latency_cost,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if is_sweep:
        print("Generic HDC experiment sweep")
        print(f"sweep runs: {result['summary']['run_count']}")
        print(f"average accuracy: {result['summary']['average_accuracy']:.1%}")
    else:
        print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
