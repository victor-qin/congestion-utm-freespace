"""Compare three GPU verification modes on identical fresh hard-88 LP/IP solves.

Each mode runs in a new process. Shared-lock waiting is excluded from timings.
The unchecked modes retain CPU preparation/screening, canonical seeds, route construction/scoring,
post-pricing audits and final independent feasibility checks; they cannot certify.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


MODES = ("full", "epsilon_border", "no_checks")


def summarize_run(directory: Path) -> dict[str, Any]:
    """Read final quality and stopping evidence without treating IP optimality as certification.

    Parameters
    ------------
    - directory (Path): one validation harness's output directory.

    Return
    --------
    - output (dict[str, Any]): solver wall time, quality, certificates and search work.
    """
    result = json.loads((directory / "result.json").read_text())
    stats = json.loads((directory / "planner_stats.json").read_text())
    pricing_path = directory / "pricing.json"
    records = json.loads(pricing_path.read_text())["results"] if pricing_path.exists() else []
    return {
        "solver_wall_s": result["elapsed_s"], "total_cost": stats["objective"],
        "objective_name": stats["objective_name"], "accepted": result["accepted"],
        "feasible": result["verified"], "validation_passed": result["passed"],
        "pricing_complete": stats.get("pricing_search_complete"),
        "pricing_bound_certified": stats.get("pricing_bound_certified"),
        "cost_lower_bound": stats.get("cost_lower_bound"),
        "global_cost_gap": stats.get("global_integer_gap_cost"),
        "termination": stats.get("termination_reason"),
        "pricing_failure": stats.get("pricing_failure_status"),
        "iterations": stats.get("iterations"), "pricing_results": len(records),
        "complete_results": sum(r["search_complete"] for r in records),
        "certified_results": sum(r["certified"] for r in records),
        "first_incomplete_flight": next(
            (r["flight_id"] for r in records if not r["search_complete"]), None),
        "cpu_full_pricing_guard": result["cpu_full_pricing_guard"],
        "failure_reasons": result["failure_reasons"],
        **{name: sum(r["stats"].get(name, 0) for r in records) for name in (
            "candidates", "cpu_recomputations", "canonical_validations",
            "canonical_rejections", "gpu_dispatches", "prepare_s", "price_upload_bytes",
        )},
    }


def save_summary(output: Path, report: dict[str, Any]) -> None:
    """Persist every run and compare costs only when both schedules accept the full cohort.

    Parameters
    ------------
    - output (Path): fresh comparison directory containing per-run artifacts.
    - report (dict[str, Any]): metadata and progressively collected outcomes.

    Return
    --------
    - output (None): writes comparison.json and a compact comparison.csv.
    """
    rows = report["runs"]
    for row in rows:
        baseline = next((r for r in rows if r["repeat"] == row["repeat"]
                         and r["mode"] == "full"), {})
        comparable = all(r.get("feasible") is True and not r.get("error")
                         and r.get("accepted") == len(report["flight_ids"])
                         and r.get("objective_name") == "total_cost" for r in (row, baseline))
        row["cost_minus_full"] = (row["total_cost"] - baseline["total_cost"]
                                  if comparable else None)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = ["repeat", "mode", "process_wall_s", "solver_wall_s", "total_cost",
               "cost_minus_full", "accepted", "feasible", "pricing_complete",
               "pricing_bound_certified", "cost_lower_bound", "termination", "pricing_failure",
               "first_incomplete_flight", "pricing_results", "complete_results",
               "certified_results", "candidates", "cpu_recomputations", "canonical_validations",
               "exit_code", "error"]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Run serialized, process-isolated comparisons, preserving incomplete and failed outcomes.

    Parameters
    ------------
    - argv (list[str] | None): CLI options; None uses this process's arguments.

    Return
    --------
    - output (int): nonzero for execution, input-matching or feasibility failures, not label caps.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True, help="Trusted original early capture")
    parser.add_argument("--out", type=Path, required=True, help="Fresh comparison directory")
    parser.add_argument("--metal-lock", type=Path, required=True, help="Shared advisory lock path")
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--budget", type=float, default=900.0, help="Seconds per solve, including IP reserve")
    parser.add_argument("--max-labels", type=int, default=250_000)
    parser.add_argument("--repeats", type=int, default=1, help="Rotate mode order across fresh-process repeats")
    parser.add_argument("--flight-ids", type=int, nargs="+", help="Override hard-88 cohort for a smoke test")
    args = parser.parse_args(argv)
    if not math.isfinite(args.epsilon) or args.epsilon < 0:
        parser.error("--epsilon must be finite and nonnegative")
    if not math.isfinite(args.budget) or args.budget <= 65:
        parser.error("--budget must exceed the 65-second final IP reserve")
    if min(args.max_labels, args.repeats) < 1:
        parser.error("--max-labels and --repeats must be positive")
    if args.out.exists() or not args.capture.is_file():
        parser.error("Choose an existing capture and a fresh output directory")
    ids = args.flight_ids
    if ids is None:
        evidence = json.loads(Path(__file__).with_name("gpu_hard88_measurements.json").read_text())
        ids = evidence["cohort"]
    if len(set(ids)) != len(ids):
        parser.error("--flight-ids must be distinct")
    root = Path(__file__).resolve().parents[2]
    output, capture = args.out.resolve(), args.capture.resolve()
    output.mkdir(parents=True)
    report: dict[str, Any] = {
        "flight_ids": ids, "capture": str(capture), "epsilon": args.epsilon,
        "budget_s": args.budget, "max_labels": args.max_labels, "batch_size": 1,
        "repeats": args.repeats, "runs": [],
        "notes": [
            "Fresh LP/IP solve per mode; captured duals are not reused.",
            "Process wall includes startup; solver wall includes validation/reporting overhead.",
            "Lock waiting is excluded. Warm pricing is not measured by this command.",
            "Epsilon/border and no-check search are uncertified; final feasibility audits remain.",
            "A retained certified bound does not imply the final pricing sweep completed.",
            "Compare stopping/completion before interpreting runtime differences as speedups.",
        ],
    }
    reference = None
    failed = False
    print("Waiting for shared Metal benchmark lock", flush=True)
    with args.metal_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for repeat in range(args.repeats):
            offset = repeat % len(MODES)
            for mode in MODES[offset:] + MODES[:offset]:
                name = f"{repeat + 1:02d}-{mode}"
                target = output / name
                command = [sys.executable, "-m"]
                if mode == "full":
                    command += ["analysis.colgen.gpu_validation", "solve"]
                else:
                    command += ["analysis.colgen.gpu_epsilon_experiment", "solve", "--epsilon"]
                    command += ([str(args.epsilon), "--verify-border"] if mode == "epsilon_border"
                                else ["0", "--skip-completion-validation"])
                command += ["--capture", str(capture), "--out", str(target),
                            "--budget", str(args.budget), "--max-labels", str(args.max_labels),
                            "--chunk-size", "2048", "--batch-size", "1", "--flight-ids", *map(str, ids)]
                row: dict[str, Any] = dict(repeat=repeat + 1, mode=mode, command=command)
                print(f"Starting {name}; log: {output / (name + '.log')}", flush=True)
                started = time.perf_counter()
                with (output / (name + ".log")).open("w") as log:
                    process = subprocess.run(command, cwd=root, env=dict(os.environ, PYTHONPATH=str(root)),
                                             stdout=log, stderr=subprocess.STDOUT, check=False)
                row.update(exit_code=process.returncode, process_wall_s=time.perf_counter() - started)
                try:
                    row.update(summarize_run(target))
                    manifest = json.loads((target / "manifest.json").read_text())
                    signature = {key: manifest[key] for key in (
                        "capture_sha256", "requests_sha256", "config_sha256", "flight_ids",
                        "static_terminals_sha256", "source_sha256", "harness_sha256", "run_params",
                    )}
                    if reference is None:
                        reference = signature
                        report["matched_inputs"] = reference
                    if signature != reference:
                        raise ValueError("Source, configuration or selected inputs changed between runs")
                    if process.returncode not in (0, 1) or row["feasible"] is not True:
                        raise ValueError("Execution or final feasibility audit failed; inspect per-run reports")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    row["error"] = str(exc)
                    failed = True
                report["runs"].append(row)
                save_summary(output, report)
                print(f"{name}: solver={row.get('solver_wall_s')}s cost={row.get('total_cost')} "
                      f"accepted={row.get('accepted')}/{len(ids)} feasible={row.get('feasible')} "
                      f"pricing_complete={row.get('pricing_complete')} stop={row.get('termination')} "
                      f"error={row.get('error')}", flush=True)
    print(f"Comparison: {output / 'comparison.csv'}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
