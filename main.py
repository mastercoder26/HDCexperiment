"""Run the beginner HDC classifier on a tiny IoT example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hdc.core import DEFAULT_DIMENSIONS, OPERATIONS, HDC
from hdc.model import HDCClassifier


# These small examples make the program easy to demonstrate and inspect.
TRAINING_RECORDS = [
    ("normal", {"protocol": "mqtt", "encryption": "on", "rate": "low"}),
    ("normal", {"protocol": "https", "encryption": "on", "rate": "low"}),
    ("normal", {"protocol": "ssh", "encryption": "on", "rate": "medium"}),
    ("anomaly", {"protocol": "telnet", "encryption": "off", "rate": "high"}),
    ("anomaly", {"protocol": "ftp", "encryption": "off", "rate": "high"}),
    ("anomaly", {"protocol": "ssh", "encryption": "off", "rate": "high"}),
]

TEST_RECORDS = [
    ("normal", {"protocol": "mqtt", "encryption": "on", "rate": "low"}),
    ("normal", {"protocol": "https", "encryption": "on", "rate": "low"}),
    ("anomaly", {"protocol": "telnet", "encryption": "off", "rate": "high"}),
    ("anomaly", {"protocol": "ssh", "encryption": "off", "rate": "high"}),
]


def read_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small HDC IoT example.")
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        help=f"hypervector length (default: {DEFAULT_DIMENSIONS})",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--energy-cost",
        type=float,
        default=0.0,
        help="estimated picojoules per work unit",
    )
    parser.add_argument(
        "--latency-cost",
        type=float,
        default=0.0,
        help="estimated nanoseconds per work unit",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def print_result(result: dict[str, object]) -> None:
    print("HDC IoT baseline")
    print(f"dimensions: {result['dimensions']}")
    print(f"seed:       {result['seed']}")
    print(
        f"accuracy:   {result['correct']}/{result['total']} "
        f"({result['accuracy']:.1%})"
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


def main() -> int:
    args = read_arguments()

    # The CLI uses one simple cost for every operation. Python users can pass
    # different values per operation directly to HDC if an experiment needs it.
    energy_costs = {name: args.energy_cost for name in OPERATIONS}
    latency_costs = {name: args.latency_cost for name in OPERATIONS}
    hdc = HDC(
        dimensions=args.dimensions,
        seed=args.seed,
        energy_costs=energy_costs,
        latency_costs=latency_costs,
    )

    classifier = HDCClassifier(hdc)
    classifier.train(TRAINING_RECORDS)
    evaluation = classifier.evaluate(TEST_RECORDS)

    result = {
        "dimensions": args.dimensions,
        "seed": args.seed,
        **evaluation,
        "memory_bytes": classifier.memory_bytes(),
        "simulation": hdc.report(),
        # Kept so older saved-report readers can still find these two values.
        "config": {"dimensions": args.dimensions, "seed": args.seed},
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
