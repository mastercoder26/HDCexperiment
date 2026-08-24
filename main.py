"""Command-line entry point for the toy IoT HDC baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hdc import CostModel, HDCConfig
from hdc.experiment import ExperimentResult, run_iot_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the modular bipolar HDC IoT baseline.",
    )
    parser.add_argument("--dimensions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tie-breaker",
        choices=("random", "positive", "negative"),
        default="positive",
    )
    parser.add_argument(
        "--energy-pj-per-work-unit",
        type=float,
        default=0.0,
        help="Example analytical coefficient; not a measured value.",
    )
    parser.add_argument(
        "--latency-ns-per-work-unit",
        type=float,
        default=0.0,
        help="Example analytical coefficient; not a measured value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the full JSON report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report instead of the readable summary.",
    )
    return parser


def _cost_model_from_args(args: argparse.Namespace) -> CostModel:
    if (
        args.energy_pj_per_work_unit == 0
        and args.latency_ns_per_work_unit == 0
    ):
        return CostModel()
    return CostModel.uniform(
        name="cli-example-uniform",
        energy_pj_per_work_unit=args.energy_pj_per_work_unit,
        latency_ns_per_work_unit=args.latency_ns_per_work_unit,
    )


def print_summary(result: ExperimentResult) -> None:
    report = result.simulation_report
    print(result.experiment_name)
    print(f"dataset:    {result.dataset_name}")
    print(f"dimensions: {report.config.dimensions}")
    print(f"seed:       {report.config.seed}")
    print(
        "accuracy:   "
        f"{result.correct_predictions}/{len(result.predictions)} "
        f"({result.accuracy:.1%})"
    )
    print()
    print("predictions:")
    for prediction in result.predictions:
        scores = ", ".join(
            f"{label}={score:.3f}"
            for label, score in sorted(prediction.scores.items())
        )
        print(
            f"  true={prediction.true_label:<7} "
            f"predicted={prediction.predicted_label:<7} "
            f"margin={prediction.margin:.3f} ({scores})"
        )
    print()
    print("operation metrics:")
    for operation, stats in sorted(report.operations.items()):
        print(
            f"  {operation:<10} calls={stats.calls:<3} "
            f"work_units={stats.work_units:<8} "
            f"runtime_ns={stats.measured_runtime_ns}"
        )
    print(
        "modeled totals: "
        f"{report.total_modeled_energy_pj:.3f} pJ, "
        f"{report.total_modeled_latency_ns:.3f} ns "
        f"(profile: {report.cost_profile})"
    )


def main() -> int:
    args = build_parser().parse_args()
    config = HDCConfig(
        dimensions=args.dimensions,
        seed=args.seed,
        tie_breaker=args.tie_breaker,
    )
    result = run_iot_baseline(
        config,
        cost_model=_cost_model_from_args(args),
    )
    payload = result.to_dict()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_summary(result)
        if args.output is not None:
            print(f"report:     {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
