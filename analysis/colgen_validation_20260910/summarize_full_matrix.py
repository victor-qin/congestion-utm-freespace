"""Summarize two-seed full validation runs and their named solver stages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def _write(path: Path, value) -> None:
    """Write one JSON artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _distribution(values: list[float]) -> dict[str, float | int]:
    """Summarize a complete per-flight timing population."""
    ordered = sorted(values)

    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "total_s": math.fsum(ordered),
        "mean_s": math.fsum(ordered) / len(ordered),
        "median_s": percentile(0.5),
        "p95_s": percentile(0.95),
        "max_s": ordered[-1],
    }


def _flight_profile(directory: Path, planner: dict) -> dict:
    """Aggregate raw flight diagnostics and persistent worker load."""
    records = []
    for path in sorted((directory / "flight_records").glob("round_*.jsonl.gz")):
        with gzip.open(path, "rt") as handle:
            records.extend(json.loads(line) for line in handle)
    fields = ("task_s", "bootstrap_s", "bootstrap_goal_s", "compiled_s", "fallback_s")
    timings = {
        key: _distribution([float(record.get(key, 0.0)) for record in records])
        for key in fields
    }
    workers = {}
    for record in records:
        key = f"{record.get('worker')}:{record.get('pid')}"
        item = workers.setdefault(key, {"records": 0, "task_s": 0.0})
        item["records"] += 1
        item["task_s"] += float(record.get("task_s", 0.0))
    worker_totals = [item["task_s"] for item in workers.values()]
    return {
        "records": len(records),
        "timings": timings,
        "workers": workers,
        "worker_task_load": _distribution(worker_totals),
        "pricing_task_total_s": float(planner.get("pricing_task_total_s", 0.0)),
        "pricing_wall_s": float(planner.get("pricing_wall_s", 0.0)),
        "observed_worker_utilization": (
            math.fsum(worker_totals) / (float(planner["pricing_wall_s"]) * len(workers))
            if workers and planner.get("pricing_wall_s")
            else None
        ),
        "native_diagnostics": {
            "kernel_fell_back": int(planner.get("kernel_fell_back", 0)),
            "kernel_label_restarts": int(planner.get("kernel_label_restarts", 0)),
            "pricing_worker_lost": int(planner.get("pricing_worker_lost", 0)),
            "goal_native_calls": sum(bool(record.get("bootstrap_goal_native")) for record in records),
            "goal_declines": sum(bool(record.get("bootstrap_goal_decline_reason")) for record in records),
        },
    }


def _case(directory: Path) -> dict:
    """Load one full result and aggregate per-iteration stage counters."""
    result = json.loads((directory / "result.json").read_text())
    planner = json.loads((directory / "planner_stats.json").read_text())
    iterations = json.loads((directory / "iterations.json").read_text())
    capture = json.loads((directory / "capture_manifest.json").read_text())
    stage_names = sorted({key for row in iterations for key in row.get("stage_s", {})})
    count_names = sorted({key for row in iterations for key in row.get("stage_n", {})})
    stage_s = {
        key: math.fsum(float(row.get("stage_s", {}).get(key, 0.0)) for row in iterations)
        for key in stage_names
    }
    stage_n = {
        key: sum(int(row.get("stage_n", {}).get(key, 0)) for row in iterations)
        for key in count_names
    }
    columns_accepted = sum(int(row.get("columns_added", 0)) for row in iterations)
    revalidated = int(planner.get("pricing_revalidated_columns", stage_n.get("canonical_priced", 0)))
    certified = int(planner.get("pricing_certified_columns", 0))
    paths_sha256 = hashlib.sha256((directory / "paths.jsonl").read_bytes()).hexdigest()
    column_identities = []
    column_numbers = []
    with (directory / "columns.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            column_identities.append(
                {key: row.get(key) for key in ("index", "flight_id", "departure_step", "path_id", "geometry_id", "phase", "sweep")}
            )
            column_numbers.append(
                {key: row.get(key) for key in ("index", "cost", "reduced_cost")}
            )
    semantic_iterations = [
        {
            key: row.get(key)
            for key in (
                "iteration", "lp_objective", "upper_bound", "cost_lower_bound", "cost_upper_bound",
                "lp_gap_cost", "heuristic_cost", "heuristic_gap_cost", "columns", "columns_added",
                "rc_sum", "rc_n_positive", "rc_max", "rc_p50", "rc_p90", "dual_l2", "dual_linf",
                "dual_nonzero", "pricing_exact",
            )
        }
        for row in iterations
    ]
    def digest(value) -> str:
        """Hash one normalized JSON value."""
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return {
        "directory": str(directory.resolve()),
        "result": result,
        "planner_stats": planner,
        "per_flight": _flight_profile(directory, planner),
        "capture": capture,
        "stage_s": stage_s,
        "stage_n": stage_n,
        "final_lp_gap_cost": iterations[-1].get("lp_gap_cost") if iterations else None,
        "pricing_columns_accepted": columns_accepted,
        "pricing_certified_columns_derived": certified,
        "pricing_revalidated_columns_derived": revalidated,
        "trace_hashes": {
            "paths_jsonl_sha256": paths_sha256,
            "column_identities_sha256": digest(column_identities),
            "column_numbers_sha256": digest(column_numbers),
            "semantic_iterations_sha256": digest(semantic_iterations),
        },
        "simulation_wall_excluding_capture_s": result.get(
            "simulation_wall_excluding_capture_s", result["simulation_wall_s"]
        ),
    }


def _change(before: float, after: float) -> dict[str, float | None]:
    """Report absolute and percentage change with a zero-safe percentage."""
    return {
        "baseline": before,
        "combined": after,
        "delta": after - before,
        "percent": None if before == 0 else 100.0 * (after / before - 1.0),
    }


def _present_equal(before: dict, after: dict, key: str) -> bool:
    """Require a non-null value in both records and compare it exactly."""
    return (
        key in before
        and before[key] is not None
        and key in after
        and after[key] is not None
        and before[key] == after[key]
    )


def _zero(mapping: dict, key: str) -> bool:
    """Require one numeric diagnostic to be present and equal to zero."""
    value = mapping.get(key)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


def _empty(mapping: dict, key: str) -> bool:
    """Require one collection diagnostic to be present and empty."""
    value = mapping.get(key)
    return isinstance(value, (list, tuple)) and not value


def main() -> int:
    """Write JSON and Markdown comparisons for completed seed pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path(__file__).parent / "full_matrix")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "full_comparison")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = {
        f"{arm}_seed{seed}": _case(args.matrix / f"{arm}_seed{seed}")
        for seed in (0, 1)
        for arm in ("baseline", "combined")
    }
    metrics = (
        "simulation_wall_s",
        "solver_wall_s",
        "pricing_wall_s",
        "ip_wall_s",
        "iteration_ip_wall_s",
        "ip_setup_s",
    )
    stage_metrics = (
        "add_violated_rows",
        "canonical_priced",
        "canonical_heuristic",
        "canonical_iteration_ip",
        "pricing_certificate_check",
        "add_column",
        "solve_lp",
        "iteration_ip",
    )
    comparisons = {}
    for seed in (0, 1):
        baseline = cases[f"baseline_seed{seed}"]
        combined = cases[f"combined_seed{seed}"]
        b_result, c_result = baseline["result"], combined["result"]
        input_checks = {
            key: _present_equal(b_result, c_result, key)
            for key in (
                "requests_sha",
                "generated_requests_sha",
                "demand_config_sha",
                "input_sha",
                "algorithm_params_sha",
                "n_requests",
                "n_static_terminals",
            )
        }
        quality = {
            key: {"baseline": b_result.get(key), "combined": c_result.get(key)}
            for key in (
                "verified",
                "accepted",
                "denied",
                "filed_cost",
                "congestion_cost",
                "total_delay_s",
                "solver_objective",
                "global_cost_gap",
                "termination",
                "ip_status",
                "iterations",
                "columns",
                "trajectory_sha",
                "peak_process_rss_mib",
            )
        }
        quality_deltas = {
            key: float(c_result[key]) - float(b_result[key])
            for key in ("congestion_cost", "solver_objective")
            if key in b_result
            and b_result[key] is not None
            and key in c_result
            and c_result[key] is not None
        }
        quality["final_lp_gap_cost"] = {
            "baseline": baseline["final_lp_gap_cost"],
            "combined": combined["final_lp_gap_cost"],
        }
        quality_within_tolerance = {
            key: key in quality_deltas and abs(quality_deltas[key]) <= 1e-6
            for key in ("congestion_cost", "solver_objective")
        }
        safety = {
            arm: {
                "verified": case["result"].get("verified") is True,
                "all_requests_accepted": (
                    isinstance(case["result"].get("accepted"), int)
                    and isinstance(case["result"].get("n_requests"), int)
                    and case["result"]["accepted"] == case["result"]["n_requests"]
                ),
                "repair_added_zero": _zero(case["planner_stats"], "repair_added"),
                "kernel_fallback_zero": _zero(case["planner_stats"], "kernel_fell_back"),
                "kernel_restart_zero": _zero(case["planner_stats"], "kernel_label_restarts"),
                "worker_loss_zero": _zero(case["planner_stats"], "pricing_worker_lost"),
                "goal_declines_zero": case["per_flight"]["native_diagnostics"]["goal_declines"] == 0,
                "budget_denied_zero": _empty(
                    case["planner_stats"], "budget_denied_flight_ids"
                ),
                "search_exhausted_zero": _empty(
                    case["planner_stats"], "search_exhausted_flight_ids"
                ),
                "objective_consistent": abs(
                    float(case["result"].get("congestion_cost_minus_solver_objective", math.inf))
                ) <= 1e-6,
                "ip_status_optimal": case["planner_stats"].get("ip_status") == "optimal",
                "ip_optimal_true": case["planner_stats"].get("ip_optimal") is True,
                "restricted_ip_gap_finite": (
                    isinstance(case["planner_stats"].get("restricted_ip_gap"), (int, float))
                    and math.isfinite(float(case["planner_stats"]["restricted_ip_gap"]))
                ),
            }
            for arm, case in (("baseline", baseline), ("combined", combined))
        }
        trace_parity = {
            key: baseline["trace_hashes"][key] == combined["trace_hashes"][key]
            for key in baseline["trace_hashes"]
        }
        wall = {
            key: _change(float(b_result.get(key, 0.0) or 0.0), float(c_result.get(key, 0.0) or 0.0))
            for key in metrics
        }
        wall["simulation_wall_excluding_capture_s"] = _change(
            baseline["simulation_wall_excluding_capture_s"],
            combined["simulation_wall_excluding_capture_s"],
        )
        stages = {
            key: _change(
                baseline["stage_s"].get(key, 0.0), combined["stage_s"].get(key, 0.0)
            )
            for key in stage_metrics
        }
        comparisons[f"seed{seed}"] = {
            "input_checks": input_checks,
            "inputs_match": all(input_checks.values()),
            "quality": quality,
            "quality_deltas": quality_deltas,
            "quality_within_tolerance": quality_within_tolerance,
            "safety": safety,
            "correctness_passed": (
                all(input_checks.values())
                and all(all(checks.values()) for checks in safety.values())
            ),
            "trace_parity": trace_parity,
            "trace_note": "Finite 30-second per-round IPs may change incumbents and final trajectories; differences are reported, not silently accepted as exact parity.",
            "wall": wall,
            "stages": stages,
            "stage_counts": {
                key: {
                    "baseline": baseline["stage_n"].get(key, 0),
                    "combined": combined["stage_n"].get(key, 0),
                }
                for key in stage_metrics
            },
            "pricing_acceptance": {
                arm: {
                    "accepted": cases[f"{arm}_seed{seed}"]["pricing_columns_accepted"],
                    "certified": int(cases[f"{arm}_seed{seed}"]["planner_stats"].get("pricing_certified_columns", 0)),
                    "revalidated": int(cases[f"{arm}_seed{seed}"]["planner_stats"].get("pricing_revalidated_columns", cases[f"{arm}_seed{seed}"]["stage_n"].get("canonical_priced", 0))),
                }
                for arm in ("baseline", "combined")
            },
        }
    summary = {"cases": cases, "comparisons": comparisons}
    _write(args.out / "summary.json", summary)
    for seed in (0, 1):
        item = comparisons[f"seed{seed}"]
        _write(
            args.out / f"seed{seed}_parity.json",
            {
                "correctness_passed": item["correctness_passed"],
                "input_checks": item["input_checks"],
                "safety": item["safety"],
                "trace_parity": item["trace_parity"],
                "quality": item["quality"],
                "quality_deltas": item["quality_deltas"],
                "quality_within_tolerance": item["quality_within_tolerance"],
                "trace_note": item["trace_note"],
            },
        )
    lines = [
        "# Full validation benchmark",
        "",
        "| Seed | Input match | Verified B/C | Accepted B/C | Final LP gap B/C | Adjusted wall B → C | Pricing B → C |",
        "|---:|:---:|:---:|---:|---:|---:|---:|",
    ]
    for seed in (0, 1):
        item = comparisons[f"seed{seed}"]
        quality = item["quality"]
        wall = item["wall"]
        lines.append(
            f"| {seed} | {item['inputs_match']} | {quality['verified']['baseline']}/"
            f"{quality['verified']['combined']} | {quality['accepted']['baseline']}/"
            f"{quality['accepted']['combined']} | "
            f"{quality['final_lp_gap_cost']['baseline']:.6g}/"
            f"{quality['final_lp_gap_cost']['combined']:.6g} | "
            f"{wall['simulation_wall_excluding_capture_s']['baseline']:.3f}s → "
            f"{wall['simulation_wall_excluding_capture_s']['combined']:.3f}s "
            f"({wall['simulation_wall_excluding_capture_s']['percent']:.2f}%) | "
            f"{wall['pricing_wall_s']['baseline']:.3f}s → "
            f"{wall['pricing_wall_s']['combined']:.3f}s "
            f"({wall['pricing_wall_s']['percent']:.2f}%) |"
        )
    lines.extend([
        "",
        "| Seed | Arm | Task mean / median / p95 / max | Bootstrap mean / median / p95 / max | Main compiled mean / median / p95 / max | Worker task total | Utilization |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for seed in (0, 1):
        for arm in ("baseline", "combined"):
            profile = cases[f"{arm}_seed{seed}"]["per_flight"]
            task = profile["timings"]["task_s"]
            bootstrap = profile["timings"]["bootstrap_s"]
            compiled = profile["timings"]["compiled_s"]
            lines.append(
                f"| {seed} | {arm} | {task['mean_s']:.4f} / {task['median_s']:.4f} / {task['p95_s']:.4f} / {task['max_s']:.4f} | "
                f"{bootstrap['mean_s']:.4f} / {bootstrap['median_s']:.4f} / {bootstrap['p95_s']:.4f} / {bootstrap['max_s']:.4f} | "
                f"{compiled['mean_s']:.4f} / {compiled['median_s']:.4f} / {compiled['p95_s']:.4f} / {compiled['max_s']:.4f} | "
                f"{profile['pricing_task_total_s']:.3f}s | {profile['observed_worker_utilization']:.3f} |"
            )
    lines.extend([
        "",
        "Named stage totals and exact quality fields are in `summary.json`. Capture overhead is",
        "reported by each run; adjusted wall is supporting evidence, while matched sweep and row",
        "replays isolate the two changes. Single runs at each seed do not estimate variance.",
        "",
    ])
    (args.out / "report.md").write_text("\n".join(lines))
    return 0 if all(item["correctness_passed"] for item in comparisons.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
