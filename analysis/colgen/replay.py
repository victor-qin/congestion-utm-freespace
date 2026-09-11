"""Replay one captured column-generation pricing sweep from any recorded round."""

from __future__ import annotations

import argparse
from dataclasses import fields
import gzip
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

from analysis.colgen.common import digest, encode, sha256_bytes, source_hashes, write_json


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
    parser.add_argument(
        "--expected-workers",
        type=int,
        default=12,
        help="Require this worker count in the capture (default: 12; 0 disables the check).",
    )
    parser.add_argument("--gurobi-threads", type=int, default=4)
    parser.add_argument("--require-independent-clean", action="store_true")
    parser.add_argument("--require-all-parent-certified", action="store_true")
    args = parser.parse_args()
    if args.gurobi_threads < 1 or args.expected_workers < 0:
        parser.error("--gurobi-threads must be positive and --expected-workers nonnegative")
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
        COLGEN_GUROBI_THREADS=str(args.gurobi_threads),
    )
    from freespace_sim.planner.colgen import dp_kernel, pricing, pricing_pool
    from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
    from freespace_sim.planner.colgen.objective import cost_model
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.planner.colgen.solver import _canonical_column

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
    if args.expected_workers and params.n_pricing_workers != args.expected_workers:
        raise ValueError(
            f"Expected captured {args.expected_workers}-worker policy, "
            f"got {params.n_pricing_workers}"
        )
    dp_warm_started = time.perf_counter()
    dp_kernel.warm_kernel()
    dp_warmup = {
        "wall_s": time.perf_counter() - dp_warm_started,
        "pricing_signatures": len(dp_kernel._price_dag.signatures),
        "feasible_signatures": len(dp_kernel._feasible_dag.signatures),
    }
    goal_warmup = _warm_optional_goal_kernel()
    catalog = StaticTerminalCatalog(captured["static_terminals"], captured["cfg"])
    graph_started = time.perf_counter()
    graphs = {
        request.flight_id: build_flight_graph(request, captured["cfg"], catalog, params)
        for request in captured["requests"]
    }
    graph_rebuild_s = time.perf_counter() - graph_started
    harness_bytes = Path(__file__).read_bytes()
    capture_bytes = args.capture.read_bytes()
    manifest = {
        "source_root": str(source),
        "capture": str(args.capture.resolve()),
        "captured_round": captured.get("round"),
        "capture_sha256": sha256_bytes(capture_bytes),
        "harness_sha256": sha256_bytes(harness_bytes),
        "tool_dependency_sha256": {
            "common.py": sha256_bytes(Path(__file__).with_name("common.py").read_bytes()),
        },
        "source_sha256": source_hashes(source),
        "params": encode(params),
        "workers": params.n_pricing_workers,
        "gurobi_threads": args.gurobi_threads,
        "budget_s": args.budget,
        "dp_warmup": dp_warmup,
        "compiled_goal_warmup": goal_warmup,
        "graph_rebuild_s_excluded": graph_rebuild_s,
        "scope": (
            "One captured pricing sweep at its recorded round, including fresh worker "
            "launch and worker graph/preparation setup; excludes master and IP solves."
        ),
    }
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "status.json", {"state": "running"})
    started = time.perf_counter()
    try:
        result = pricing_pool.price_sweep(
            captured["pricing_order"],
            captured["requests"],
            graphs,
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
        model = cost_model(captured["cfg"], params)
        certificate_started = time.perf_counter()
        certificate_helper = getattr(pricing_pool, "certified_sweep_columns", None)
        certified = (
            ()
            if certificate_helper is None
            else certificate_helper(result, graphs, captured["cfg"], params, catalog)
        )
        certified_ids = {id(column) for column in certified}
        accepted_columns = []
        certified_count = revalidated_count = 0
        for reduced_cost, column in zip(result.reduced_costs, result.columns, strict=True):
            if column is None or reduced_cost <= 1e-9:
                continue
            if id(column) in certified_ids:
                certified_count += 1
                accepted_columns.append(column)
            else:
                revalidated_count += 1
                accepted_columns.append(
                    _canonical_column(
                        column,
                        graphs[column.flight_id],
                        captured["cfg"],
                        **({"model": model} if certificate_helper is not None else {}),
                    )
                )
        for column in result.extra_columns:
            if id(column) in certified_ids:
                certified_count += 1
                accepted_columns.append(column)
            else:
                revalidated_count += 1
                accepted_columns.append(
                    _canonical_column(
                        column,
                        graphs[column.flight_id],
                        captured["cfg"],
                        **({"model": model} if certificate_helper is not None else {}),
                    )
                )
        certificate_s = time.perf_counter() - certificate_started

        # This second pass is deliberately outside the measured acceptance time. It
        # independently recomputes claims, cost, and reduced cost on fresh graphs.
        view = pricing.DualView(captured["duals"], captured["cfg"])
        audit_started = time.perf_counter()
        audit_graphs = {
            request.flight_id: build_flight_graph(
                request, captured["cfg"], catalog, params
            )
            for request in captured["requests"]
        }

        def cold_canonical(column):
            """Canonicalize without a receipt or claim cache from an earlier payload."""
            graph = audit_graphs[column.flight_id]
            search_cache = getattr(graph, "_search_cache", None)
            certified_claims = getattr(search_cache, "certified_claims", None)
            if certified_claims is not None:
                certified_claims.clear()
            return _canonical_column(column, graph, captured["cfg"], model=model)

        canonical_answers = []
        audit_changes = []
        for flight_id, reduced_cost, column in zip(
            result.flight_ids, result.reduced_costs, result.columns, strict=True
        ):
            if column is None:
                canonical_answers.append((reduced_cost, None))
                continue
            canonical = cold_canonical(column)
            canonical_rc = model.reduced_cost(
                benefit=params.M,
                cost=canonical.delay_s,
                dual_cost=view.claim_cost(canonical.claims),
                pi_f=captured["flight_duals"].get(flight_id, 0.0),
            )
            canonical_answers.append((canonical_rc, canonical))
            if canonical != column or canonical_rc != reduced_cost:
                audit_changes.append(
                    {
                        "flight_id": flight_id,
                        "raw_rc_hex": float(reduced_cost).hex(),
                        "canonical_rc_hex": float(canonical_rc).hex(),
                        "raw_cost_hex": float(column.delay_s).hex(),
                        "canonical_cost_hex": float(canonical.delay_s).hex(),
                        "claims_changed": column.claims != canonical.claims,
                    }
                )
        extra_audit = [cold_canonical(column) for column in result.extra_columns]
        extra_audit_changes = sum(
            canonical != column
            for canonical, column in zip(extra_audit, result.extra_columns, strict=True)
        )
        independent_audit_s = time.perf_counter() - audit_started
        by_flight = {record["flight_id"]: record for record in result.flight_records}
        records = [
            {
                "flight_id": flight_id,
                "signature": signature((reduced_cost, column), by_flight.get(flight_id, {})),
                "canonical_signature": signature(
                    canonical_answer, by_flight.get(flight_id, {})
                ),
                "search": by_flight.get(flight_id, {}),
            }
            for flight_id, reduced_cost, column, canonical_answer in zip(
                result.flight_ids,
                result.reduced_costs,
                result.columns,
                canonical_answers,
                strict=True,
            )
        ]
        write_json(args.out / "records.json", records)
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
            "parent_acceptance_policy": (
                "receipt_or_model_revalidation"
                if certificate_helper is not None
                else "legacy_claims_only_revalidation"
            ),
            "parent_acceptance_s": certificate_s,
            "parent_certified_columns": certified_count,
            "parent_revalidated_columns": revalidated_count,
            "parent_accepted_columns_sha256": digest(accepted_columns),
            "independent_cold_validation_s_excluded": independent_audit_s,
            "independent_validation_changes": len(audit_changes),
            "independent_validation_changes_detail": audit_changes,
            "independent_extra_columns_sha256": digest(extra_audit),
            "independent_extra_validation_changes": extra_audit_changes,
            "independent_validation_clean": not audit_changes and extra_audit_changes == 0,
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
        write_json(args.out / "sweep.json", summary)
        if not result.complete or len(records) != len(captured["requests"]):
            raise RuntimeError("Captured sweep replay did not complete")
        if args.require_independent_clean and not summary["independent_validation_clean"]:
            raise AssertionError("Independent cold validation changed a returned payload")
        if args.require_all_parent_certified and summary["parent_revalidated_columns"]:
            raise AssertionError("Parent receipt missed an accepted pricing column")
        if args.reference is not None:
            reference_sweep = json.loads((args.reference / "sweep.json").read_text())
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
            canonical_mismatches = []
            for record in records:
                expected = baseline[record["flight_id"]]["canonical_signature"]
                actual = record["canonical_signature"]
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
                    canonical_mismatches.append(
                        {"flight_id": record["flight_id"], "changed": changed}
                    )
            strict_mismatches = [
                mismatch
                for mismatch in mismatches
                if any(
                    key in mismatch["changed"]
                    for key in ("identity_sha256", "claims_sha256", "work_sha256")
                )
            ]
            parity = {
                "passed": (
                    not strict_mismatches
                    and not canonical_mismatches
                    and summary["independent_extra_columns_sha256"]
                    == reference_sweep["independent_extra_columns_sha256"]
                ),
                "compared": len(records),
                "raw_mismatches": mismatches,
                "strict_raw_mismatches": strict_mismatches,
                "canonical_mismatches": canonical_mismatches,
                "canonical_extra_columns_match": (
                    summary["independent_extra_columns_sha256"]
                    == reference_sweep["independent_extra_columns_sha256"]
                ),
                "raw_cost_or_rc_changes_allowed_only_when_canonical_matches": True,
            }
            write_json(args.out / "parity.json", parity)
            if not parity["passed"]:
                raise AssertionError("Replay differs after independent canonical validation")
            if goal_warmup["available"]:
                native_usage = {
                    "expected_goal_calls": reference_sweep["bootstrap_goal_calls"],
                    "native_goal_calls": summary["bootstrap_goal_native"],
                    "native_declines": summary["bootstrap_goal_declines"],
                }
                native_usage["passed"] = (
                    native_usage["native_goal_calls"] == native_usage["expected_goal_calls"]
                    and native_usage["native_declines"] == 0
                )
                write_json(args.out / "native_usage.json", native_usage)
                if not native_usage["passed"]:
                    raise AssertionError("Compiled goal kernel did not cover every expected call")
        write_json(args.out / "status.json", {"state": "complete", "wall_s": elapsed})
        return 0
    except BaseException as exc:
        write_json(
            args.out / "status.json",
            {"state": "failed", "error": repr(exc), "wall_s": time.perf_counter() - started},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
