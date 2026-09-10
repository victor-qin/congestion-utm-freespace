"""Replay one captured column-generation pricing sweep from any recorded round."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time


def encode(value):
    """Serialize exact float bits and unordered claim sets deterministically."""
    if is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((encode(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def digest(value) -> str:
    """Hash an exact canonical representation independent of pickle details."""
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True).encode()).hexdigest()


def write(path: Path, value) -> None:
    """Persist a readable artifact atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def signature(answer, search: dict) -> dict:
    """Separate column identity, claims, numbers, and search-work parity."""
    reduced_cost, column = answer
    counts = {
        key: search[key]
        for key in (
            "n_labels",
            "attempts",
            "bootstrap_labels",
            "bootstrap_goal_labels",
            "bootstrap_dp_labels",
            "status",
            "declined",
        )
        if key in search
    }
    identity = None if column is None else (
        column.flight_id,
        column.departure_step,
        column.level,
        column.origin_lane_idx,
        column.dest_lane_idx,
        column.cell_path,
    )
    return {
        "output_sha256": digest(answer),
        "identity_sha256": digest(identity),
        "claims_sha256": digest(None if column is None else column.claims),
        "cost_rc_sha256": digest((reduced_cost, None if column is None else column.delay_s)),
        "work_sha256": digest(counts),
        "work": counts,
        "rc": reduced_cost,
        "cost": None if column is None else column.delay_s,
        "returned_column": column is not None,
        "claim_count": 0 if column is None else len(column.claims),
    }


def _warm_optional_goal_kernel() -> dict:
    """Compile the candidate goal kernel before the measured sweep when present."""
    try:
        module = importlib.import_module("freespace_sim.planner.colgen.bootstrap_kernel")
    except ModuleNotFoundError as exc:
        if exc.name != "freespace_sim.planner.colgen.bootstrap_kernel":
            raise
        return {"available": False, "wall_s": 0.0, "signatures": 0}
    started = time.perf_counter()
    module.warm_kernel()
    return {
        "available": True,
        "wall_s": time.perf_counter() - started,
        "signatures": len(module._advance.signatures),
    }


def main() -> int:
    """Run one fresh-process sweep and optionally require baseline parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--budget", type=float, default=1800.0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a fresh output directory")
    args.out.mkdir(parents=True)
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    os.environ.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        COLGEN_GUROBI_THREADS="4",
    )
    from freespace_sim.planner.colgen import dp_kernel, pricing, pricing_pool
    from freespace_sim.planner.colgen.network import StaticTerminalCatalog
    from freespace_sim.planner.colgen.params import ColGenParams

    assert source in Path(dp_kernel.__file__).resolve().parents
    with args.capture.open("rb") as handle:
        captured = pickle.load(handle)  # Trusted local artifact from capture_full_run.py.
    old_params = captured["params"]
    params = ColGenParams(
        **{
            field.name: getattr(old_params, field.name)
            for field in fields(ColGenParams)
            if hasattr(old_params, field.name)
        }
    )
    if params.n_pricing_workers != 12:
        raise ValueError(f"Expected captured 12-worker policy, got {params.n_pricing_workers}")
    dp_warm_started = time.perf_counter()
    dp_kernel.warm_kernel()
    dp_warmup = {
        "wall_s": time.perf_counter() - dp_warm_started,
        "pricing_signatures": len(dp_kernel._price_dag.signatures),
        "feasible_signatures": len(dp_kernel._feasible_dag.signatures),
    }
    goal_warmup = _warm_optional_goal_kernel()
    harness_bytes = Path(__file__).read_bytes()
    capture_bytes = args.capture.read_bytes()
    manifest = {
        "source_root": str(source),
        "capture": str(args.capture.resolve()),
        "captured_round": captured.get("round"),
        "capture_sha256": hashlib.sha256(capture_bytes).hexdigest(),
        "harness_sha256": hashlib.sha256(harness_bytes).hexdigest(),
        "source_sha256": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((source / "freespace_sim").rglob("*.py"))
        },
        "params": encode(params),
        "workers": params.n_pricing_workers,
        "gurobi_threads": 4,
        "budget_s": args.budget,
        "dp_warmup": dp_warmup,
        "compiled_goal_warmup": goal_warmup,
        "scope": (
            "One captured pricing sweep at its recorded round, including fresh worker "
            "launch and worker graph/preparation setup; excludes master and IP solves."
        ),
    }
    write(args.out / "manifest.json", manifest)
    write(args.out / "status.json", {"state": "running"})
    catalog = StaticTerminalCatalog(captured["static_terminals"], captured["cfg"])
    started = time.perf_counter()
    try:
        result = pricing_pool.price_sweep(
            captured["pricing_order"],
            captured["requests"],
            {},
            captured["cfg"],
            params,
            catalog,
            captured["duals"],
            pricing.DualView(captured["duals"], captured["cfg"]),
            captured["flight_duals"],
            captured["known_columns"],
            deadline=time.monotonic() + args.budget,
            heuristic_only=captured.get("heuristic_only", False),
        )
        elapsed = time.perf_counter() - started
        by_flight = {record["flight_id"]: record for record in result.flight_records}
        records = [
            {
                "flight_id": flight_id,
                "signature": signature((reduced_cost, column), by_flight.get(flight_id, {})),
                "search": by_flight.get(flight_id, {}),
            }
            for flight_id, reduced_cost, column in zip(
                result.flight_ids, result.reduced_costs, result.columns, strict=True
            )
        ]
        write(args.out / "records.json", records)
        with gzip.open(args.out / "flight_records.jsonl.gz", "wt") as handle:
            for record in result.flight_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        with (args.out / "columns.pkl").open("wb") as handle:
            pickle.dump(dict(zip(result.flight_ids, result.columns, strict=True)), handle)
        stage_names = sorted(
            {
                key
                for record in result.flight_records
                for key, value in record.items()
                if key.endswith("_s") and isinstance(value, (int, float))
            }
        )
        stage_totals = {
            key: math.fsum(
                float(record[key]) for record in result.flight_records if key in record
            )
            for key in stage_names
        }
        summary = {
            "complete": result.complete,
            "flights": len(records),
            "wall_s": elapsed,
            "task_total_s": result.task_total_s,
            "pool_setup_s": result.pool_setup_s,
            "kernel_counters": dict(result.kernel_counters),
            "timeout_flight_id": result.timeout_flight_id,
            "positive_rc_sum": math.fsum(max(0.0, rc) for rc in result.reduced_costs),
            "task_s_per_flight": result.task_total_s / max(1, len(records)),
            "flight_stage_totals_s": stage_totals,
            "flight_stage_means_s": {
                key: total / max(1, len(records)) for key, total in stage_totals.items()
            },
            "bootstrap_goal_native": sum(
                bool(record.get("bootstrap_goal_native", False)) for record in result.flight_records
            ),
            "bootstrap_goal_calls": sum(
                float(record.get("bootstrap_goal_s", 0.0)) > 0.0
                for record in result.flight_records
            ),
            "bootstrap_goal_declines": sum(
                bool(record.get("bootstrap_goal_decline_reason"))
                for record in result.flight_records
            ),
            "bootstrap_goal_sinks_asked": sum(
                int(record.get("bootstrap_goal_sinks_asked", 0))
                for record in result.flight_records
            ),
            "bootstrap_goal_sinks_skipped": sum(
                int(record.get("bootstrap_goal_sinks_skipped", 0))
                for record in result.flight_records
            ),
        }
        write(args.out / "sweep.json", summary)
        if not result.complete or len(records) != len(captured["requests"]):
            raise RuntimeError("Captured sweep replay did not complete")
        if args.reference is not None:
            baseline = {
                record["flight_id"]: record
                for record in json.loads((args.reference / "records.json").read_text())
            }
            mismatches = []
            for record in records:
                expected = baseline[record["flight_id"]]["signature"]
                actual = record["signature"]
                changed = [
                    key
                    for key in (
                        "output_sha256",
                        "identity_sha256",
                        "claims_sha256",
                        "cost_rc_sha256",
                        "work_sha256",
                    )
                    if expected[key] != actual[key]
                ]
                if changed:
                    mismatches.append(
                        {
                            "flight_id": record["flight_id"],
                            "changed": changed,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
            parity = {"passed": not mismatches, "compared": len(records), "mismatches": mismatches}
            write(args.out / "parity.json", parity)
            if mismatches:
                raise AssertionError(f"{len(mismatches)} flights differ; inspect parity.json")
            if goal_warmup["available"]:
                reference_sweep = json.loads((args.reference / "sweep.json").read_text())
                native_usage = {
                    "expected_goal_calls": reference_sweep["bootstrap_goal_calls"],
                    "native_goal_calls": summary["bootstrap_goal_native"],
                    "native_declines": summary["bootstrap_goal_declines"],
                }
                native_usage["passed"] = (
                    native_usage["native_goal_calls"] == native_usage["expected_goal_calls"]
                    and native_usage["native_declines"] == 0
                )
                write(args.out / "native_usage.json", native_usage)
                if not native_usage["passed"]:
                    raise AssertionError("Compiled goal kernel did not cover every expected call")
        write(args.out / "status.json", {"state": "complete", "wall_s": elapsed})
        return 0
    except BaseException as exc:
        write(
            args.out / "status.json",
            {"state": "failed", "error": repr(exc), "wall_s": time.perf_counter() - started},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
