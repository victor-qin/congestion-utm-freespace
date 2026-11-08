"""Summarize optional approach-bound trials without equating capped and completed work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analysis.colgen.common import source_hashes, write_json
from analysis.colgen.integer_pruning_report import replay_summary, solver_summary


def checked_pair(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Check identical replay inputs and compare independent complete certificates.

    Parameters
    ------------
    - before (dict): execution replay with the optional approach bound disabled.
    - after (dict): replay with the same numerical mode and the optional bound enabled.

    Return
    --------
    - output (list[dict]): per-phase timing, objective and completion comparisons.
    """
    for key in ("source_sha256", "capture_sha256", "batch_size", "max_labels", "chunk_size",
                "gpu_integer_bounds"):
        if before[key] != after[key]:
            raise ValueError(f"Replay input mismatch: {key}")
    comparisons = []
    for a, b in zip(before["runs"], after["runs"], strict=True):
        if a["label"] != b["label"]:
            raise ValueError("Phase mismatch")
        pairs = list(zip(a["results"], b["results"], strict=True))
        if any(x["flight_id"] != y["flight_id"] for x, y in pairs):
            raise ValueError("Ordered cohort mismatch")
        complete = all(row["certified"] and row["search_complete"]
                       for pair in pairs for row in pair)
        difference = max((abs(x["reduced_cost"] - y["reduced_cost"]) for x, y in pairs), default=0.)
        if complete and difference > 1e-6:
            raise AssertionError("Complete objectives differ beyond tolerance")
        comparisons.append({
            "phase": a["label"], "matched_complete_certificates": complete,
            "max_rc_difference": difference,
            "same_returned_columns": all(x["column"] == y["column"] for x, y in pairs),
            "before_s": a["elapsed_s"], "after_s": b["elapsed_s"],
            "wall_reduction_percent": 100 * (1 - b["elapsed_s"] / a["elapsed_s"]) if complete else None,
        })
    return comparisons


def main() -> None:
    """Collect source-linked direct, cohort and canonical-solver evidence.

    Parameters
    ------------
    - none: CLI specifies the job directory and output JSON.

    Return
    --------
    - output (None): writes compact checked evidence and raw-file hashes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[2] / "freespace_sim/planner/colgen"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package.iterdir() if p.suffix in (".py", ".h")}
    all_hashes = source_hashes(package.parents[2])
    report: dict[str, Any] = {"source_sha256": hashes, "direct_2224": {}, "replays": {},
                              "comparisons": {}, "solvers": {}}
    direct_hash = None
    for mode in ("integer-before", "integer-after", "audit", "reference-before", "reference-after"):
        path = args.jobs / f"approach-2224-{mode}.json"
        data = json.loads(path.read_bytes())
        if data["source_sha256"] != hashes:
            raise ValueError(f"Stale source: {path}")
        direct_hash = direct_hash or data["problem_sha256"]
        if data["problem_sha256"] != direct_hash:
            raise ValueError("Direct replay problem mismatch")
        report["direct_2224"][mode] = {
            "raw_path": str(path), "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "problem_sha256": direct_hash, "options": data["options"],
            "runs": [{key: value for key, value in run.items() if key != "column"}
                     for run in data["runs"]],
        }
    for mode in ("integer-after", "audit", "reference-after"):
        for run in report["direct_2224"][mode]["runs"]:
            if not (run["certified"] and run["search_complete"]
                    and run["stats"].get("approach_root_prunes", 0) > 0):
                raise AssertionError("Direct replay did not exercise a certified approach closure")
    for mode in ("integer", "reference"):
        before, after = (report["direct_2224"][f"{mode}-{suffix}"] for suffix in ("before", "after"))
        options = [{key: value for key, value in data["options"].items()
                    if key != "gpu_approach_bounds"} for data in (before, after)]
        if options[0] != options[1]:
            raise ValueError("Direct replay option mismatch")
        if any(a["reduced_cost"] != b["reduced_cost"]
               for a, b in zip(before["runs"], after["runs"], strict=True)):
            raise AssertionError("Direct replay incumbent changed")
    for cohort in ("hard88", "late18", "controls4"):
        raw = {}
        for mode in (("after",) if cohort == "controls4" else ("before", "after", "audit")):
            path = args.jobs / f"approach-{cohort}-{mode}.json"
            data = raw[mode] = json.loads(path.read_bytes())
            summary = replay_summary(path, hashes)
            if not summary["current_numeric_source"]:
                raise ValueError(f"Stale source: {path}")
            for phase, run in zip(summary["phases"], data["runs"], strict=True):
                phase["totals"].update({key: sum(row["stats"].get(key, 0) for row in run["results"])
                                        for key in ("approach_prepare_s", "approach_work", "approach_roots",
                                                    "approach_root_prunes")})
                phase.pop("rows")
            summary.pop("source_sha256")
            report["replays"][f"{cohort}-{mode}"] = summary
        if "before" in raw:
            report["comparisons"][cohort] = checked_pair(raw["before"], raw["after"])
    for cohort in ("ten", "hard88"):
        inputs = []
        for mode in ("before", "after"):
            directory = args.jobs / f"approach-solver-{cohort}-{mode}"
            summary = solver_summary(directory)
            manifest = json.loads((directory / "manifest.json").read_bytes())
            keys = ("capture_sha256", "config_sha256", "requests_sha256",
                    "static_terminals_sha256", "flight_ids", "run_params")
            identity = {key: manifest[key] for key in keys}
            identity["run_params"] = {key: value for key, value in identity["run_params"].items()
                                      if key != "pricing_gpu_approach_bounds"}
            inputs.append(identity)
            summary["input_identity"] = identity
            summary["manifest_sha256"] = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
            summary["current_source"] = all(all_hashes.get(name) == digest
                                            for name, digest in summary.pop("source_sha256").items())
            if not summary["current_source"]:
                raise ValueError("Stale solver source")
            report["solvers"][f"{cohort}-{mode}"] = summary
        if inputs[0] != inputs[1]:
            raise ValueError("Canonical solver inputs differ")
    write_json(args.out, report)


if __name__ == "__main__":
    main()
