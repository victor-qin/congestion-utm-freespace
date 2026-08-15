"""One colgen solve, timed and broken down by stage.

The parity harness answers "did the answer move"; this answers "where did the time go".
Those became different questions once the pricing sweep was parallelised: at 8 workers the
sweep stops dominating, and whatever is still SERIAL -- graph build, seeding and the
departure ladder, the greedy's fixed wall-clock block, the master LP, the final IP --
becomes the thing worth attacking next.  A total wall figure cannot rank them.

Every number here already existed in `ColGenResult.stats` or in the `on_iteration`
payload's `stage_s`; nothing is re-instrumented for this script.

    uv run python analysis/run_colgen_timed.py --flights 500 --workers 8 --chunksize 8
    uv run python analysis/run_colgen_timed.py --flights 1000 --workers 8 --chunksize 8
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
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

from freespace_sim.planner.colgen import pricing as _pricing, solver as _solver  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.pricing import PricingTimeout  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

_RSS_SCALE = 2**20 if sys.platform == "darwin" else 1024

# How often the GREEDY's search actually reaches the compiled kernel.  It runs in the
# parent process, so counting here catches all of it -- and the count matters: if these
# searches mostly fall back to the pure-Python frontier, "the greedy is already compiled"
# is false and compiling it properly becomes a real lever rather than a dead end.  A
# fallback is silent by construction, which is exactly how a 5-7x regression hid for a
# whole issue once before.
_FEASIBLE = {"proved": 0, "fell_back": 0, "timed_out": 0}
_REAL_FEASIBLE_COMPILED = _pricing._feasible_compiled


def _counting_feasible_compiled(*args, **kwargs):
    try:
        out = _REAL_FEASIBLE_COMPILED(*args, **kwargs)
    except PricingTimeout:
        # A search cut off inside the kernel still REACHED the kernel.  Counting only
        # returns reported "2/2 reached the kernel" at 500 flights and "0/0" at 1000,
        # when in truth 170 of 202 greedy searches had entered it and 168 were cut off
        # by their per-flight slice -- i.e. the exact opposite of what it appeared to say.
        _FEASIBLE["timed_out"] += 1
        raise
    declined = isinstance(out, _pricing.Declined)
    _FEASIBLE["fell_back" if declined else "proved"] += 1
    if declined:
        _FEASIBLE[f"declined_{out.value}"] = _FEASIBLE.get(f"declined_{out.value}", 0) + 1
    return out


_pricing._feasible_compiled = _counting_feasible_compiled


def _budgeted_greedy(budget_s: float):
    """Replace the greedy's deadline, the one thing its wall-clock budget controls.

    The budget itself is now a parameter -- ``greedy_budget_s_per_flight``, which replaced
    the ``min(60.0, 0.55 * time_limit_s)`` literal that no parameter could lift -- but it is
    a RATE, and the question this seam answers is about a TOTAL: what the greedy finds when
    it is not starved, at a fixed number of seconds, independent of how many flights are in
    the batch.  Expressing that through the rate would mean dividing by the flight count at
    every call site that quotes a budget.
    """

    real = _solver._greedy_feasible_selection

    def _wrapped(*args, **kwargs):
        kwargs["deadline"] = time.monotonic() + budget_s
        return real(*args, **kwargs)

    _solver._greedy_feasible_selection = _wrapped


def _rss_mb(who) -> float:
    return resource.getrusage(who).ru_maxrss / _RSS_SCALE


def _worker_skew(records) -> dict:
    """Collapse per-flight rows to the one number a fixed worker assignment risks.

    Pinning each flight to a worker for the whole solve buys cache reuse and gives up
    `mp.Pool`'s rebalancing, so the question it owes an answer to is how uneven the split
    actually is. `sweep_flight_records` carries `worker` and `task_s` per flight, but it is
    emitted only through the transient iteration callback and is far too bulky to archive
    per flight -- so the ratio is computed here and the rows are dropped, which is what
    makes the claim checkable after the fact rather than only while a run is live.

    `max/mean` over per-worker task totals: 1.0 is a perfect split, and the sweep can never
    finish faster than its slowest worker, so this is the ceiling on what better balance
    could buy.
    """

    totals: dict[int, float] = collections.defaultdict(float)
    for row in records:
        worker = row.get("worker")
        if worker is not None:
            totals[worker] += float(row.get("task_s") or 0.0)
    if not totals:
        return {}
    busiest, mean = max(totals.values()), sum(totals.values()) / len(totals)
    return {
        "n_workers_seen": len(totals),
        "worker_task_s_max": round(busiest, 2),
        "worker_task_s_mean": round(mean, 2),
        "worker_skew": round(busiest / mean, 3) if mean > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flights", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunksize", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--solver", default="highs")
    ap.add_argument("--gap-metric", default="cost")
    ap.add_argument("--ladder", type=int, default=0)
    # ANSWER-AFFECTING, unlike every other knob on this harness. Both change which of two
    # equally-optimal columns comes back, so a run that moves either is not a clean speed
    # A/B against one that does not -- the objective may differ without anything being
    # wrong. Defaults track the shipped `ColGenParams` so the harness measures production
    # unless asked otherwise.
    ap.add_argument("--bootstrap-roots", type=int, default=ColGenParams().bootstrap_roots)
    ap.add_argument(
        "--bootstrap-ranking", default=ColGenParams().bootstrap_ranking,
        help="how the bootstrap orders roots before taking the top K; `bound` (g+h) is "
             "what makes K=1 viable at all.",
    )
    ap.add_argument(
        "--greedy-budget-s", type=float, default=None,
        help="Pin the greedy's wall clock to this many seconds TOTAL, in place of "
             "greedy_budget_s_per_flight * n_flights.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.greedy_budget_s is not None:
        _budgeted_greedy(args.greedy_budget_s)

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    static_terms = list(demand.terminals(cfg))
    params = ColGenParams(
        solver=args.solver,
        max_iterations=args.iterations,
        time_limit_s=86400.0,
        gap_metric=args.gap_metric,
        seed_ladder_steps=args.ladder,
        n_pricing_workers=args.workers,
        pricing_chunksize=args.chunksize,
        bootstrap_roots=args.bootstrap_roots,
        bootstrap_ranking=args.bootstrap_ranking,
    )

    header = {
        "scenario": args.scenario, "flights": len(requests),
        "iterations": args.iterations, "workers": args.workers,
        "chunksize": args.chunksize, "ladder": args.ladder, "solver": args.solver,
        "bootstrap_roots": args.bootstrap_roots,
        "bootstrap_ranking": args.bootstrap_ranking,
    }
    print(json.dumps(header), flush=True)

    per_iteration: list[dict] = []
    stages: dict[str, float] = collections.defaultdict(float)

    def on_iteration(state: dict) -> None:
        for key, value in (state.get("stage_s") or {}).items():
            stages[key] += float(value)
        sweep_s = float(state["sweep_s"])
        task_s = float(state.get("sweep_task_total_s") or 0.0)
        # Against ONE worker when sequential, so the sequential arm reads 100% rather than
        # 1/n -- it is the denominator, not a degenerate pool.
        lanes = max(1, args.workers)
        efficiency = task_s / (sweep_s * lanes) if sweep_s > 0 else 0.0
        per_iteration.append(
            {
                "iteration": state["iteration"],
                "sweep_s": round(sweep_s, 2),
                "sweep_task_total_s": round(task_s, 2),
                "worker_efficiency": round(efficiency, 4),
                "idle_worker_s": round(sweep_s * lanes - task_s, 2),
                "columns": state["columns"],
                "rc_n_positive": state["rc_n_positive"],
                "dual_nonzero": state["dual_nonzero"],
                **_worker_skew(state.get("sweep_flight_records") or ()),
            }
        )
        print(
            f"  it {state['iteration']:>3}  sweep={sweep_s:8.1f}s  "
            f"work={task_s:9.1f}s  eff={efficiency * 100:5.1f}%  "
            f"idle={sweep_s * lanes - task_s:8.1f}s  "
            f"cols={state['columns']:>6}  rc_n+={state['rc_n_positive']:>5}",
            flush=True,
        )

    started = time.perf_counter()
    result = ColGenSolver().solve(
        requests, cfg, static_terms, params, on_iteration=on_iteration
    )
    wall = time.perf_counter() - started
    stats = dict(result.stats)

    rows = [
        (
            int(c.flight_id), int(c.departure_step), repr(c.delay_s),
            tuple(tuple(cell) for cell in c.cell_path),
            tuple(sorted(tuple(r) for r in c.claims)),
        )
        for _, c in sorted(result.columns.items())
    ]
    sha = hashlib.sha256(repr(rows).encode()).hexdigest()[:16]

    graph_s = float(stats.get("graph_build_elapsed_s") or 0.0)
    seed_s = float(stats.get("seed_elapsed_s") or 0.0)
    greedy_s = float(stats.get("initial_greedy_elapsed_s") or 0.0)
    pricing_s = float(stats.get("pricing_wall_s") or 0.0)
    ip_s = float(stats.get("ip_elapsed_s") or 0.0)
    named = graph_s + seed_s + greedy_s + pricing_s + ip_s

    print(f"\n{'stage':32} {'seconds':>10} {'share':>8}")
    breakdown = [
        ("build_flight_graph", graph_s),
        ("seed + departure ladder", seed_s),
        ("greedy_feasible_selection", greedy_s),
        ("pricing sweeps (parallel)", pricing_s),
        ("final IP", ip_s),
        ("unattributed (LP, add_column, ...)", wall - named),
    ]
    for name, value in breakdown:
        print(f"{name:32} {value:10.2f} {value / wall * 100:7.1f}%")
    print(f"{'TOTAL':32} {wall:10.2f} {100.0:7.1f}%")
    total_feasible = (
        _FEASIBLE["proved"] + _FEASIBLE["fell_back"] + _FEASIBLE["timed_out"]
    )
    if total_feasible:
        print(
            f"\ngreedy feasible search: {total_feasible} reached the kernel -- "
            f"{_FEASIBLE['proved']} proved, {_FEASIBLE['fell_back']} fell back to Python "
            f"({_FEASIBLE['fell_back'] / total_feasible * 100:.1f}%), "
            f"{_FEASIBLE['timed_out']} cut off by their per-flight slice "
            f"({_FEASIBLE['timed_out'] / total_feasible * 100:.1f}%)"
        )

    if stages:
        print(f"\n{'master-side stage (summed)':32} {'seconds':>10} {'share':>8}")
        for name, value in sorted(stages.items(), key=lambda kv: -kv[1]):
            print(f"{name:32} {value:10.2f} {value / wall * 100:7.1f}%")

    summary = {
        **header,
        "wall_s": round(wall, 2),
        "column_sha": sha,
        "objective": repr(stats.get("objective")),
        "selected_flights": stats.get("selected_flights"),
        "n_columns": stats.get("n_columns"),
        "termination_reason": stats.get("termination_reason"),
        "iterations_run": stats.get("iterations"),
        "greedy_completed": stats.get("initial_greedy_completed"),
        "feasible_search_proved": _FEASIBLE["proved"],
        "feasible_search_fell_back": _FEASIBLE["fell_back"],
        "feasible_search_timed_out": _FEASIBLE["timed_out"],
        "greedy_budget_s": args.greedy_budget_s,
        "ip_status": stats.get("ip_status"),
        "ip_skipped": stats.get("ip_skipped"),
        "rss_self_mb": round(_rss_mb(resource.RUSAGE_SELF), 1),
        # LARGEST SINGLE CHILD, not the sum across the tree -- `getrusage` defines
        # `ru_maxrss` that way for RUSAGE_CHILDREN, so this reads flat however many
        # workers ran and says NOTHING about aggregate pool memory.  For that use
        # `analysis/sweep_pricing_workers.py`'s tree sampler.
        "rss_largest_child_mb": round(_rss_mb(resource.RUSAGE_CHILDREN), 1),
        "stage_s": {name: round(value, 2) for name, value in breakdown},
        "master_stage_s": {k: round(v, 2) for k, v in stages.items()},
        "per_iteration": per_iteration,
    }
    print("\n" + json.dumps(summary, indent=2, default=str), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
