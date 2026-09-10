"""Run the matched baseline, preparation-only, and combined replay matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def _dump(path: Path, value) -> None:
    """Write a JSON progress artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _reduction(before: float, after: float) -> dict:
    """Describe an absolute and percentage runtime reduction."""
    return {
        "before_s": before,
        "after_s": after,
        "reduction_s": before - after,
        "reduction_percent": 100.0 * (before - after) / before if before else None,
    }


def _metrics(sweep: dict) -> dict:
    """Select whole-sweep and per-flight timing measures for comparison."""
    stages = sweep["flight_stage_totals_s"]
    flights = sweep["flights"]
    return {
        "wall_s": sweep["wall_s"],
        "task_total_s": sweep["task_total_s"],
        "task_s_per_flight": sweep["task_total_s"] / flights,
        "bootstrap_total_s": stages.get("bootstrap_s", 0.0),
        "bootstrap_s_per_flight": stages.get("bootstrap_s", 0.0) / flights,
        "goal_total_s": stages.get("bootstrap_goal_s", 0.0),
        "goal_s_per_flight": stages.get("bootstrap_goal_s", 0.0) / flights,
        "pool_setup_s": sweep["pool_setup_s"],
        "bootstrap_goal_calls": sweep["bootstrap_goal_calls"],
        "bootstrap_goal_native": sweep["bootstrap_goal_native"],
        "bootstrap_goal_declines": sweep["bootstrap_goal_declines"],
        "flight_stage_totals_s": stages,
    }


def main() -> int:
    """Execute forward three-round arms and a reverse-order late-round repeat."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--prep-source", type=Path, required=True)
    parser.add_argument("--combined-source", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--harness",
        type=Path,
        default=Path(__file__).with_name("benchmark_captured_sweep.py"),
    )
    parser.add_argument("--budget", type=float, default=1800.0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a fresh output directory")
    args.out.mkdir(parents=True)
    sources = {
        "baseline": args.baseline_source.resolve(),
        "prep": args.prep_source.resolve(),
        "combined": args.combined_source.resolve(),
    }
    harness = args.harness.resolve()
    plan = [
        *(('forward', arm, round_number) for arm in ('baseline', 'prep', 'combined')
          for round_number in (1, 6, 11)),
        *(('reverse', arm, 11) for arm in ('combined', 'prep', 'baseline')),
    ]
    manifest = {
        "status": "running",
        "sources": {key: str(value) for key, value in sources.items()},
        "capture_root": str(args.capture_root.resolve()),
        "harness": str(harness),
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "budget_s_per_sweep": args.budget,
        "plan": plan,
        "timing_policy": "Every sweep runs serially in a fresh process with PYTHONHASHSEED=0.",
    }
    _dump(args.out / "manifest.json", manifest)
    completed_records = []
    forward_dirs = {}
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    for sequence, (direction, arm, round_number) in enumerate(plan, start=1):
        run_name = f"{sequence:02d}_{direction}_{arm}_round_{round_number:04d}"
        run_out = args.out / run_name
        command = [
            sys.executable,
            str(harness),
            "--source-root",
            str(sources[arm]),
            "--capture",
            str(args.capture_root / f"round_{round_number:04d}.pkl"),
            "--out",
            str(run_out),
            "--budget",
            str(args.budget),
        ]
        if not (direction == "forward" and arm == "baseline"):
            command.extend(["--reference", str(forward_dirs[("baseline", round_number)])])
        log_path = args.out / f"{run_name}.log"
        with log_path.open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {
            "sequence": sequence,
            "direction": direction,
            "arm": arm,
            "round": round_number,
            "out": str(run_out),
            "log": str(log_path),
            "returncode": result.returncode,
        }
        if (run_out / "sweep.json").exists():
            record["metrics"] = _metrics(json.loads((run_out / "sweep.json").read_text()))
        completed_records.append(record)
        _dump(args.out / "progress.json", completed_records)
        if result.returncode:
            manifest.update(status="failed", failed_sequence=sequence)
            _dump(args.out / "manifest.json", manifest)
            return result.returncode
        if direction == "forward":
            forward_dirs[(arm, round_number)] = run_out

    by_key = {
        (record["direction"], record["arm"], record["round"]): record["metrics"]
        for record in completed_records
    }
    comparisons = {}
    for round_number in (1, 6, 11):
        baseline = by_key[("forward", "baseline", round_number)]
        prep = by_key[("forward", "prep", round_number)]
        combined = by_key[("forward", "combined", round_number)]
        comparisons[str(round_number)] = {
            "prep_vs_baseline": {
                key: _reduction(baseline[key], prep[key])
                for key in ("wall_s", "task_total_s", "task_s_per_flight", "bootstrap_total_s", "goal_total_s")
            },
            "combined_vs_prep": {
                key: _reduction(prep[key], combined[key])
                for key in ("wall_s", "task_total_s", "task_s_per_flight", "bootstrap_total_s", "goal_total_s")
            },
        }
    drift = {}
    for arm in ("baseline", "prep", "combined"):
        forward = by_key[("forward", arm, 11)]
        reverse = by_key[("reverse", arm, 11)]
        drift[arm] = {
            key: {
                "forward_s": forward[key],
                "reverse_s": reverse[key],
                "delta_s": reverse[key] - forward[key],
                "delta_percent": (
                    100.0 * (reverse[key] - forward[key]) / forward[key]
                    if not math.isclose(forward[key], 0.0)
                    else None
                ),
            }
            for key in ("wall_s", "task_total_s", "task_s_per_flight", "goal_total_s")
        }
    _dump(
        args.out / "summary.json",
        {"runs": completed_records, "comparisons": comparisons, "round11_order_drift": drift},
    )
    manifest.update(status="complete", completed_runs=len(completed_records))
    _dump(args.out / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
