"""Check original-capture cache comparisons and summarize matched complete pricing work."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

from analysis.colgen.common import digest, write_json
from analysis.colgen.integer_pruning_report import replay_summary


def check_results(reference: dict[str, Any], current: dict[str, Any]) -> None:
    """Require identical inputs, ordered cohorts, canonical objectives and complete certificates.

    Parameters
    ------------
    - reference (dict): original stacked-source replay using the same saved captures.
    - current (dict): cached or disabled-cache replay with independent canonical validation.

    Return
    --------
    - output (None): raises on any changed cohort, incomplete result or numerical/column mismatch.
    """
    for key in ("capture_sha256", "batch_size", "max_labels", "chunk_size",
                "gpu_integer_bounds", "gpu_integer_audit", "gpu_approach_bounds", "budget_s"):
        if key in reference and current[key] != reference[key]:
            raise ValueError(f"Replay inputs differ: {key}")
    for before, after in zip(reference["runs"], current["runs"], strict=True):
        if before["label"] != after["label"]:
            raise ValueError("Replay phases differ")
        for a, b in zip(before["results"], after["results"], strict=True):
            if not all(r["certified"] and r["search_complete"] for r in (a, b)):
                raise AssertionError("A capped or incomplete result cannot enter a timing comparison")
            for key in ("flight_id", "reduced_cost", "column", "upper_bound", "status"):
                if a[key] != b[key]:
                    raise AssertionError(f"Exact result mismatch in {after['label']}: {key}")


def summarize(path: Path, hashes: dict[str, str]) -> dict[str, Any]:
    """Extend existing replay summaries with actual retained cache storage and exact result hashes.

    Parameters
    ------------
    - path (Path): one independently serialized original-capture replay.
    - hashes (dict): current production source hashes, including cache ownership code.

    Return
    --------
    - output (dict): compact source-linked evidence with gauges kept separate from cumulative work.
    """
    data = json.loads(path.read_bytes())
    result = replay_summary(path, hashes)
    result["matches_measured_source"] = result.pop("current_numeric_source")
    result["cache_enabled"] = data.get("integer_cache_enabled", False)
    result["cache_limits"] = data.get("integer_cache_limits")
    result["budget_s"] = data.get("budget_s")
    result["replay_script_sha256"] = data.get("replay_script_sha256")
    result["flight_ids"] = [r["flight_id"] for r in data["runs"][0]["results"]]
    for phase, raw in zip(result["phases"], data["runs"], strict=True):
        phase.pop("rows")
        phase["exact_results_sha256"] = digest([
            {k: r[k] for k in ("flight_id", "reduced_cost", "column", "upper_bound", "status")}
            for r in raw["results"]
        ])
        phase["totals"].update({
            k: sum(r["stats"].get(k, 0) for r in raw["results"])
            for k in ("bootstrap_labels", "initial_full_search_labels", "approach_prepare_s")
        })
        phase["retained_operands"] = raw.get("operand_cache", {})
        phase["cache_host_bytes_peak"] = max(
            (r["stats"].get("integer_cache_host_bytes", 0) for r in raw["results"]), default=0,
        )
        phase["cache_device_bytes_peak"] = max(
            (r["stats"].get("integer_cache_device_bytes", 0) for r in raw["results"]), default=0,
        )
    return result


def main() -> None:
    """Validate two independent paired repeats and retain rejected trials as separate evidence.

    Parameters
    ------------
    - none: CLI selects the replay directory and destination JSON.

    Return
    --------
    - output (None): writes checked comparisons; stale final sources are an error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[2] / "freespace_sim/planner/colgen"
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package.iterdir() if p.suffix in (".py", ".h")}
    cohorts = json.loads(Path(__file__).with_name("arrival_union_results.json").read_bytes())
    report: dict[str, Any] = {
        "measured_source_sha256": hashes, "original_base": "d7ad96861f866dededecd6d2c0ebee5ef02f0541",
        "timers_overlap": True, "avoided_encoding_time_is_prior_build_time": True,
        "memory_gauges_are_not_summed_per_flight_peaks": True,
        "cuda_hardware_tested": False, "original_baselines": {}, "replays": {},
        "comparisons": [], "rejected_trials": {}, "work_count_differences": [],
    }
    for cohort in ("hard88", "late18"):
        base_path = args.replays / f"{cohort}-base-1.json"
        base = json.loads(base_path.read_bytes())
        ids = [r["flight_id"] for r in base["runs"][0]["results"]]
        if ids != [r["flight_id"] for r in cohorts[cohort]["after"]["rows"]]:
            raise ValueError("Cohort differs from the original saved order")
        report["original_baselines"][cohort] = summarize(base_path, hashes)
        phases: dict[str, dict[str, list[float]]] = {}
        for repeat in (1, 2):
            controls = {}
            for mode in ("disabled", "cached"):
                name = f"{cohort}-{mode}-{repeat}"
                path = args.replays / f"{name}.json"
                data = controls[mode] = json.loads(path.read_bytes())
                summary = summarize(path, hashes)
                if not summary["matches_measured_source"]:
                    raise ValueError(f"Final measurement has stale production source: {path}")
                if summary["cache_enabled"] != (mode == "cached"):
                    raise ValueError(f"Cache control does not match its label: {path}")
                check_results(base, data)
                report["replays"][name] = summary
                for phase in summary["phases"]:
                    phases.setdefault(phase["label"], {"disabled": [], "cached": []})[mode].append(
                        phase["elapsed_s"],
                    )
            check_results(controls["disabled"], controls["cached"])
            for before, after in zip(controls["disabled"]["runs"], controls["cached"]["runs"], strict=True):
                for a, b in zip(before["results"], after["results"], strict=True):
                    differences = {key: {"disabled": a["stats"].get(key, 0),
                                         "cached": b["stats"].get(key, 0)}
                                   for key in ("labels", "full_search_labels", "bootstrap_labels",
                                               "initial_full_search_labels", "candidates",
                                               "canonical_validations")
                                   if a["stats"].get(key, 0) != b["stats"].get(key, 0)}
                    if differences:
                        report["work_count_differences"].append({
                            "cohort": cohort, "repeat": repeat, "phase": after["label"],
                            "flight_id": a["flight_id"], "differences": differences,
                        })
        for label, modes in phases.items():
            medians = {mode: median(times) for mode, times in modes.items()}
            original = next(r["elapsed_s"] for r in base["runs"] if r["label"] == label)
            report["comparisons"].append({
                "cohort": cohort, "phase": label, "elapsed_s": modes, "median_s": medians,
                "exact_columns_objectives_and_certificate_uppers": True,
                "wall_reduction_percent": 100 * (1 - medians["cached"] / medians["disabled"]),
                "original_base_s": original,
                "reduction_from_original_base_percent": 100 * (1 - medians["cached"] / original),
            })
    for name in ("hard88-trial-1", "hard88-trial-2"):
        report["rejected_trials"][name] = summarize(args.replays / f"{name}.json", hashes)
    # Keep one source manifest per version instead of repeating it for every process.
    manifests = {}
    for section in ("original_baselines", "replays", "rejected_trials"):
        for summary in report[section].values():
            source = summary.pop("source_sha256")
            fingerprint = digest(source)
            manifests[fingerprint] = source
            summary["source_fingerprint"] = fingerprint
    report["source_manifests"] = manifests
    write_json(args.out, report)


if __name__ == "__main__":
    main()
