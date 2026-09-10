"""Replay a trusted first-LP capture and compare its exact sweep with a measured arm."""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time


def arguments():
    """Select a source explicitly before importing any production modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-method", choices=["dp", "astar"], default="astar")
    parser.add_argument("--budget", type=float, default=1800)
    return parser.parse_args()


def identity(column):
    """Identify a canonical timed path independently of claim-object serialization."""
    return (column.flight_id, column.departure_step, column.level, column.origin_lane_idx,
            column.dest_lane_idx, tuple(column.cell_path))


def reference_columns(case, initial_count):
    """Read timed paths and distinguish preexisting columns from first-round additions."""
    with (case / "paths.jsonl").open() as handle:
        paths = {p["id"]: p for p in map(json.loads, handle)}
    initial, added = {}, {}
    with (case / "columns.jsonl").open() as handle:
        for column in map(json.loads, handle):
            if column["index"] >= initial_count and not (column["phase"] == "pricing" and column["sweep"] == 1):
                continue
            path = paths[column["path_id"]]
            key = (column["flight_id"], column["departure_step"], path["level"],
                   path["origin_lane_idx"], path["dest_lane_idx"],
                   tuple(tuple(cell) for cell in path["cells"]))
            (initial if column["index"] < initial_count else added)[key] = column["cost"]
    return initial, added


def main():
    """Run only the captured pricing sweep; preserve diagnostics and parity evidence."""
    args = arguments()
    args.source_root = args.source_root.resolve()
    script_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(script_root))
    sys.path.insert(0, str(args.source_root))
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      NUMBA_NUM_THREADS="1", COLGEN_GUROBI_THREADS="4")
    from benchmark_colgen_lns import dump, persist_flights
    from freespace_sim.planner.colgen import pricing_pool
    from freespace_sim.planner.colgen.network import StaticTerminalCatalog
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.planner.colgen.pricing import DualView

    assert args.source_root in Path(pricing_pool.__file__).resolve().parents
    if args.out.exists():
        raise FileExistsError("Use a fresh replay output directory")
    args.out.mkdir(parents=True)
    # The capture is generated locally by profile_colgen_pricing_current.py.
    with args.capture.open("rb") as handle:
        captured = pickle.load(handle)
    params = ColGenParams(**{f.name: getattr(captured["params"], f.name)
                             for f in fields(ColGenParams) if hasattr(captured["params"], f.name)})
    params = replace(params, bootstrap_method=args.bootstrap_method, columns_per_flight=1,
                     n_pricing_workers=12)
    cfg, requests = captured["cfg"], captured["requests"]
    order = captured["pricing_order"]
    catalog = StaticTerminalCatalog(captured["static_terminals"], cfg)
    reference = json.loads((args.reference / "iterations.json").read_text())[0]
    assert captured["first_lp"]["lp_column_count"] == reference["lp_column_count"]
    assert abs(captured["first_lp"]["lp_objective"] - reference["lp_objective"]) < 1e-6
    manifest = dict(source_root=str(args.source_root), params=params, workers=12,
                    gurobi_threads=4, budget_s=args.budget, capture=str(args.capture.resolve()),
                    capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                    reference=str(args.reference.resolve()),
                    source_sha256={str(p.relative_to(args.source_root)):
                        hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((args.source_root / "freespace_sim").rglob("*.py"))},
                    scope="Only first-LP pricing, including worker launch and graph preparation; no master/IP solve")
    dump(args.out / "manifest.json", manifest)
    dump(args.out / "status.json", dict(state="running"))
    started = time.perf_counter()
    try:
        result = pricing_pool.price_sweep(
            order, requests, {}, cfg, params, catalog, captured["duals"],
            DualView(captured["duals"], cfg), captured["flight_duals"], captured["known_columns"],
            deadline=time.monotonic() + args.budget,
        )
        elapsed = time.perf_counter() - started
        summary = persist_flights(args.out, 1, dict(iteration=1, sweep_s=elapsed,
                                                    sweep_flight_records=result.flight_records))
        dump(args.out / "sweep.json", dict(complete=result.complete, flight_ids=result.flight_ids,
             reduced_costs=result.reduced_costs, wall_s=elapsed, task_total_s=result.task_total_s,
             pool_setup_s=result.pool_setup_s, kernel_counters=dict(result.kernel_counters),
             timeout_flight_id=result.timeout_flight_id, diagnostics=summary))
        with (args.out / "columns.pkl").open("wb") as handle:
            pickle.dump(dict(zip(result.flight_ids, result.columns, strict=True)), handle)
        initial, expected = reference_columns(args.reference, reference["lp_column_count"])
        actual = {identity(c): c.delay_s for c, rc in zip(result.columns, result.reduced_costs, strict=True)
                  if c is not None and rc > 1e-9 and identity(c) not in initial}
        with gzip.open(args.reference / "flight_records/round_0001.jsonl.gz", "rt") as handle:
            records = [json.loads(line) for line in handle]
        expected_rc = {r["flight_id"]: r["final_rc"] for r in records if "final_rc" in r}
        actual_rc = dict(zip(result.flight_ids, result.reduced_costs, strict=True))
        rc_mismatches = {fid: dict(expected=rc, actual=actual_rc.get(fid))
                         for fid, rc in expected_rc.items()
                         if fid not in actual_rc or not math.isclose(rc, actual_rc[fid], rel_tol=0, abs_tol=1e-6)}
        positive_rc_sum = math.fsum(max(0.0, rc) for rc in result.reduced_costs)
        parity = dict(complete=result.complete, expected_columns=len(expected), actual_columns=len(actual),
                      missing_columns=[repr(k) for k in expected.keys() - actual.keys()],
                      unexpected_columns=[repr(k) for k in actual.keys() - expected.keys()],
                      cost_mismatches=[repr(k) for k in actual.keys() & expected.keys()
                                       if not math.isclose(actual[k], expected[k], rel_tol=0, abs_tol=1e-6)],
                      explicit_reference_rc_count=len(expected_rc), reference_flight_count=len(records),
                      rc_mismatches=rc_mismatches, positive_rc_sum=positive_rc_sum,
                      reference_positive_rc_sum=reference["rc_sum"],
                      positive_rc_sum_matches=math.isclose(positive_rc_sum, reference["rc_sum"], rel_tol=0, abs_tol=1e-5),
                      note="Compare canonical timed paths and costs; reference omits RC on early-exit flights, so those are covered only by aggregate positive RC and column parity.")
        parity["passed"] = bool(result.complete and not parity["missing_columns"] and
            not parity["unexpected_columns"] and not parity["cost_mismatches"] and
            not rc_mismatches and parity["positive_rc_sum_matches"])
        dump(args.out / "parity.json", parity)
        dump(args.out / "status.json", dict(state="complete", parity_passed=parity["passed"], elapsed_s=elapsed))
        print(json.dumps({key: (len(value) if key in {
            "missing_columns", "unexpected_columns", "cost_mismatches", "rc_mismatches"
        } else value) for key, value in parity.items()}), flush=True)
        return 0 if parity["passed"] else 1
    except BaseException as exc:
        dump(args.out / "status.json", dict(state="interrupted_or_failed", error=repr(exc),
                                            elapsed_s=time.perf_counter() - started))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
