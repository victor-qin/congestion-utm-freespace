"""Run one column-generation solve to natural termination and record every iteration.

Nothing in the repo currently answers "how many iterations does colgen actually need, and
what is each one buying" -- every production run so far has stopped on the clock or on the
iteration cap, so the tail of the curve has never been observed. This runs a solve with
both limits effectively removed and streams one JSON line per iteration to disk.

**Streaming and flushed on purpose.** The expected runtime is hours to days on the pure
Python pricing path, so a killed or interrupted run must still yield everything up to its
last completed iteration. Every line is written and flushed as the callback fires.

**The LP backend is a first-class variable, not a detail.** HiGHS and Gurobi terminate at
completely different iterations on the same instance -- Gurobi's duals can close the
default revenue gap at iteration 1 for a schedule HiGHS keeps improving for many more.
Run both and compare; a single-backend trace describes that backend, not the instance.

Non-serializable payload members (`master`, `lp_x`, `capacity_duals`) are summarized rather
than dropped: the top dual magnitudes and the LP's fractional support are the parts that
answer why an iteration added the columns it did.

Examples:

    uv run python analysis/colgen_reference_trace.py --flights 100 --solver highs
    uv run python analysis/colgen_reference_trace.py --flights 100 --solver gurobi \
        --scenario density_faa_wing_zipline --out runs/trace_gurobi
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

# ru_maxrss is bytes on macOS, kilobytes on Linux.
_MAXRSS_SCALE = 1 if sys.platform == "darwin" else 1024

# Payload members that are already JSON-safe scalars.
_SCALARS = (
    "iteration", "lp_objective", "upper_bound", "raw_upper_bound",
    "cost_upper_bound", "cost_lower_bound",
    "lp_gap", "lp_gap_revenue", "lp_gap_cost",
    "heuristic_gap", "heuristic_gap_revenue", "heuristic_gap_cost", "heuristic_cost",
    "sweep_s", "columns", "columns_added",
    "rc_sum", "rc_n_positive", "rc_max", "rc_p50", "rc_p90",
    "dual_l2", "dual_linf", "dual_nonzero",
    "max_column_cost", "n_uncovered", "n_rc_near_M", "n_overlap",
    "lazy_rows_added", "lazy_row_rounds", "elapsed_s",
)


def _finite(value):
    """JSON has no inf/nan; keep them as strings rather than silently emitting null."""

    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def _record(state: dict) -> dict:
    row = {name: _finite(state.get(name)) for name in _SCALARS}

    duals = state.get("capacity_duals") or {}
    magnitudes = sorted((abs(v) for v in duals.values()), reverse=True)
    row["dual_top10_share"] = (
        sum(magnitudes[:10]) / sum(magnitudes) if sum(magnitudes) else 0.0
    )
    row["dual_max"] = magnitudes[0] if magnitudes else 0.0

    x = state.get("lp_x")
    if x is not None:
        x = np.asarray(x, dtype=float)
        # How fractional the LP is -- the quantity that predicts the restricted IP's cost,
        # and the thing a growing column pool is supposed to reduce.
        row["x_support"] = int((x > 1e-9).sum())
        row["x_fractional"] = int(((x > 1e-9) & (x < 1.0 - 1e-9)).sum())

    master = state.get("master")
    if master is not None:
        row["n_materialized_rows"] = len(getattr(master, "rows", ()) or ())

    row["stage_s"] = {k: round(v, 4) for k, v in (state.get("stage_s") or {}).items()}
    row["stage_n"] = dict(state.get("stage_n") or {})
    row["rss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _MAXRSS_SCALE / 1e6, 1
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="density_faa_wing_zipline")
    parser.add_argument("--flights", type=int, default=100)
    parser.add_argument("--solver", default="highs", choices=("highs", "gurobi"))
    parser.add_argument(
        "--max-iterations", type=int, default=100_000,
        help="effectively unbounded; the solve is meant to stop on its own criterion",
    )
    parser.add_argument(
        "--time-limit-s", type=float, default=604_800.0,
        help="one week. Not infinity: the solver derives its pricing deadline and IP "
             "reserve from this, so it must be a finite number.",
    )
    parser.add_argument("--gap-metric", default="cost", choices=("cost", "revenue"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out or (
        REPO_ROOT / "runs" / f"colgen_trace_{args.scenario}_{args.flights}_{args.solver}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    iters_path = out_dir / "colgen_iters.jsonl"
    summary_path = out_dir / "summary.json"

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    if len(cfg.flight_levels_m) != 1:
        raise SystemExit(
            f"{args.scenario} has {len(cfg.flight_levels_m)} flight levels; "
            "colgen plans on a single level"
        )
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    static_terms = list(demand.terminals(cfg))
    params = ColGenParams(
        solver=args.solver,
        max_iterations=args.max_iterations,
        time_limit_s=args.time_limit_s,
        gap_metric=args.gap_metric,
    )

    header = {
        "tree": str(_loaded.parent.parent),
        "scenario": args.scenario,
        "flights": len(requests),
        "solver": args.solver,
        "gap_metric": args.gap_metric,
        "max_iterations": args.max_iterations,
        "time_limit_s": args.time_limit_s,
        "started_monotonic": time.monotonic(),
    }
    print(json.dumps(header, indent=2), flush=True)

    handle = iters_path.open("w", encoding="utf-8")
    handle.write(json.dumps({"kind": "header", **header}) + "\n")
    handle.flush()

    def on_iteration(state: dict) -> None:
        row = _record(state)
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        print(
            f"iter {row['iteration']:>5}  lp={row['lp_objective']:<14.6g} "
            f"gap_cost={row['lp_gap_cost']:<10.4g} gap_rev={row['lp_gap_revenue']:<10.4g} "
            f"cols={row['columns']:>6} (+{row['columns_added']:>4})  "
            f"rc_n+={row['rc_n_positive']:>4}  sweep={row['sweep_s']:>8.1f}s  "
            f"rss={row['rss_mb']:>7.1f}MB",
            flush=True,
        )

    started = time.perf_counter()
    try:
        result = ColGenSolver().solve(
            requests, cfg, static_terms, params, on_iteration=on_iteration
        )
    finally:
        handle.close()
    wall = time.perf_counter() - started

    stats = dict(result.stats)
    summary = {
        **header,
        "wall_s": wall,
        "termination_reason": stats.get("termination_reason"),
        "iterations": stats.get("iterations"),
        "objective": stats.get("objective"),
        "selected_flights": stats.get("selected_flights"),
        "denied_flight_ids": sorted(stats.get("denied_flight_ids", ())),
        "n_columns": stats.get("n_columns"),
        "n_materialized_rows": stats.get("n_materialized_rows"),
        "pricing_wall_s": stats.get("pricing_wall_s"),
        "ip_elapsed_s": stats.get("ip_elapsed_s"),
        "ip_objective": stats.get("ip_objective"),
        "final_lp_objective": stats.get("final_lp_objective"),
        "lp_gap": stats.get("lp_gap"),
        "peak_rss_gb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _MAXRSS_SCALE / 1e9, 2
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=repr), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, default=repr), flush=True)
    print(f"\niterations -> {iters_path}\nsummary    -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
