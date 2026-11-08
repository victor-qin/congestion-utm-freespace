"""Summarize guarded integer replays without equating capped and completed work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

from analysis.colgen.common import write_json


TOTALS = (
    "prepare_s", "integer_prepare_s", "integer_gather_s", "bootstrap_s", "future_s",
    "labels", "full_search_labels", "candidates", "cpu_recomputations", "integer_audited",
    "canonical_validations", "canonical_rejections", "gpu_direct_prunes", "gpu_dispatches",
    "integer_roots", "integer_cost_roots",
)


def replay_summary(path: Path, numeric_hashes: dict[str, str]) -> dict[str, Any]:
    """Preserve per-flight certificates and all relevant timing/allocation observations.

    Parameters
    ------------
    - path (Path): one execution_replay JSON file from a fresh process.
    - numeric_hashes (dict[str, str]): current numeric implementation, including allocation policy.

    Return
    --------
    - output (dict[str, Any]): compact evidence; batch size one prevents duplicate batch counters.
    """
    raw = path.read_bytes()
    data = json.loads(raw)
    if data["batch_size"] != 1:
        raise ValueError("This summary requires batch size one for additive per-flight counters")
    mode = ("audit" if data.get("gpu_integer_audit") else
            "integer" if data.get("gpu_integer_bounds") else "reference")
    if not data["cpu_full_pricing_guard"] or (mode == "integer" and not data["cpu_numeric_verifier_guard"]):
        raise ValueError("Required full-CPU/per-prune verifier guards were not enabled")
    phases = []
    for run in data["runs"]:
        rows = run["results"]
        phases.append({
            "label": run["label"], "elapsed_s": run["elapsed_s"],
            "after_base_preparation_s": (run["elapsed_s"] - run["measured"].get("prepare_s", 0.)
                                         - run["measured"].get("independent_validation_s", 0.)),
            "certified": sum(row["certified"] for row in rows), "count": len(rows),
            "all_complete": all(row["search_complete"] for row in rows),
            "max_bound_gap": max((row["upper_bound"] - row["reduced_cost"]
                                   for row in rows if row["certified"]), default=None),
            "totals": {key: sum(row["stats"].get(key, 0) for row in rows) for key in TOTALS},
            "resident_bytes_max": max((row["stats"].get("resident_bytes", 0) for row in rows), default=0),
            "integer_bytes_max": max((row["stats"].get("integer_bytes", 0) for row in rows), default=0),
            "process_peak_rss_bytes": run.get("process_peak_rss_bytes"),
            "measured": run["measured"], "counters": run["counters"],
            "rows": [{key: row[key] for key in (
                "flight_id", "reduced_cost", "upper_bound", "search_complete", "certified", "status",
            )} for row in rows],
        })
    return {
        "file": str(path), "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "cohort": path.name.split("-")[0], "mode": mode,
        "current_numeric_source": all(data["source_sha256"].get(name) == digest
                                      for name, digest in numeric_hashes.items()),
        **{key: data[key] for key in ("batch_size", "max_labels", "chunk_size", "graph_s",
                                    "snapshot_index_s", "init_s", "capture_sha256", "source_sha256",
                                    "cpu_full_pricing_guard", "cpu_numeric_verifier_guard")},
        "phases": phases,
    }


def solver_summary(path: Path) -> dict[str, Any]:
    """Report solver quality and stopping conditions independently of its timing.

    Parameters
    ------------
    - path (Path): gpu_validation output directory containing a completed result report.

    Return
    --------
    - output (dict[str, Any]): feasible objective, bounds and exact cap/completion evidence.
    """
    result = json.loads((path / "result.json").read_text())
    stats_path = path / "planner_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    records = json.loads((path / "pricing.json").read_text())["results"]
    return {
        "directory": str(path), **result,
        **{key: stats.get(key) for key in ("objective", "cost_lower_bound", "global_integer_gap_cost",
                                         "termination_reason", "pricing_failure_status", "iterations",
                                         "pricing_search_complete", "pricing_bound_certified")},
        "certified_results": sum(row["certified"] for row in records),
        "result_count": len(records),
        "incomplete": [{key: row[key] for key in ("flight_id", "status", "reduced_cost")}
                       for row in records if not row["search_complete"]],
        "source_sha256": json.loads((path / "manifest.json").read_text())["source_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    """Validate matched replay outcomes and emit compact reviewable source-linked evidence.

    Parameters
    ------------
    - argv (list[str] | None): input directory, optional solver directories and output JSON.

    Return
    --------
    - output (int): zero after checking cohort identity and completed objective agreement.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--solver", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    package = Path(__file__).resolve().parents[2] / "freespace_sim/planner/colgen"
    numeric_hashes = {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                      for name in ("gpu_integer.py", "gpu_integer_bounds.py", "gpu_integer_search.py",
                                   "gpu_integer_source.py", "pricing_integer.h",
                                   "pricing_integer_bounds.h", "pricing_integer_search.h")}
    runs = [replay_summary(path, numeric_hashes) for path in sorted(args.replays.glob("*.json"))]
    for run in runs:
        references = [other for other in runs if other["cohort"] == run["cohort"]
                      and other["mode"] == "reference"]
        if not references:
            continue
        reference = references[0]
        for key in ("capture_sha256", "batch_size", "max_labels", "chunk_size"):
            if run[key] != reference[key]:
                raise ValueError(f"Unmatched replay input: {key}")
        for phase in run["phases"]:
            other = next((p for p in reference["phases"] if p["label"] == phase["label"]), None)
            if other is None:
                continue
            if [r["flight_id"] for r in phase["rows"]] != [r["flight_id"] for r in other["rows"]]:
                raise ValueError("Unmatched cohort ordering")
            complete = (phase["all_complete"] and other["all_complete"]
                        and phase["certified"] == other["certified"] == phase["count"])
            differences = [abs(a["reduced_cost"] - b["reduced_cost"])
                           for a, b in zip(phase["rows"], other["rows"], strict=True)]
            phase["matched_complete_certificates"] = complete
            phase["max_rc_difference_from_reference"] = max(differences, default=0.)
            if complete and phase["max_rc_difference_from_reference"] > 1e-6:
                raise AssertionError("Completed integer/reference objectives differ beyond tolerance")
    comparisons = []
    for cohort in sorted({run["cohort"] for run in runs}):
        labels = sorted({phase["label"] for run in runs if run["cohort"] == cohort
                         for phase in run["phases"]})
        for label in labels:
            modes = {mode: [phase for run in runs if run["cohort"] == cohort and run["mode"] == mode
                            and (mode == "reference" or run["current_numeric_source"])
                            for phase in run["phases"] if phase["label"] == label]
                     for mode in ("reference", "integer")}
            if not all(modes.values()):
                continue
            complete = all(phase.get("matched_complete_certificates", False)
                           for phases in modes.values() for phase in phases)
            times = {mode: [phase["elapsed_s"] for phase in phases] for mode, phases in modes.items()}
            medians = {mode: median(values) for mode, values in times.items()}
            comparisons.append({
                "cohort": cohort, "phase": label, "all_matching_complete_certificates": complete,
                "elapsed_s": times, "median_s": medians,
                "median_wall_time_reduction_percent": (
                    100 * (1 - medians["integer"] / medians["reference"]) if complete else None),
            })
    sources = {}
    for run in runs:
        hashes = run.pop("source_sha256")
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        sources[fingerprint] = hashes
        run["source_fingerprint"] = fingerprint
        run["flight_ids"] = [row["flight_id"] for row in run["phases"][0]["rows"]]
        for phase in run["phases"]:
            rows = phase.pop("rows")
            phase["result_rows_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            phase["incomplete"] = [row for row in rows if not row["search_complete"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, {
        "timers_may_overlap": True, "cold_is_first_pricing_call": True,
        "graph_snapshot_and_backend_setup_reported_separately": True,
        "numeric_ambiguity_policy": "retain; no separate refinement queue or ambiguity counter",
        "cuda_hardware_tested": False, "replays": runs,
        "comparisons": comparisons,
        "current_numeric_source_sha256": numeric_hashes,
        "source_manifests": sources,
        "solvers": [solver_summary(path) for path in args.solver],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
