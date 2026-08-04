"""Stage-by-stage wall-clock profile of a colgen solve, in execution order.

**Why not cProfile.**  This repo has now been burned three times by cProfile over-statement
on hot leaf functions (``column_clear`` "x2.44" that was really 4-13%; ``_visit_claims`` "44%"
that yielded 1.23x; ``RowKey.__new__`` "16.9s cumulative" whose removal bought 3.3s).  At
millions of calls, cProfile's own ~1-2 us per call dominates what it reports.  So this harness
wraps *named* functions with ``perf_counter`` instead, and prints the calibrated wrapper
overhead next to every row so an inflated line is visible rather than believed.

**What it shows.**  The solve in the order it actually executes, with each stage's children
attributed and the residual left explicit -- the residual is the stage's own loop body, which
is usually where the real time is once the callees are accounted for.

**Parallel note.**  Under ``--workers N`` the pricing sweep runs ``price_flight`` in worker
processes, where these parent-side wrappers do not exist.  The sweep is therefore reported
from the shipped telemetry (makespan / task-sum / straggler), and the per-flight *internals*
are profiled in a sequential run -- identical work per flight, only the dispatch differs.

Usage:
    uv run python analysis/prof_colgen_stages.py --flights 100
    uv run python analysis/prof_colgen_stages.py --flights 100 --workers 4
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import freespace_sim  # noqa: E402

assert Path(freespace_sim.__file__).resolve().is_relative_to(REPO_ROOT), (
    f"imported freespace_sim from {freespace_sim.__file__}, expected a tree under {REPO_ROOT}"
)

from freespace_sim.planner.colgen import (  # noqa: E402
    master as master_mod,
    network as network_mod,
    pricing as pricing_mod,
    solver as solver_mod,
)
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.pricing_pool import ParallelPricingConfig  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402


class _Ledger:
    """Accumulate (count, wall) per label, plus a calibrated per-call wrapper cost."""

    def __init__(self) -> None:
        self.wall: dict[str, float] = defaultdict(float)
        self.count: dict[str, int] = defaultdict(int)
        self._restore: list = []
        self.overhead_s = self._calibrate()

    @staticmethod
    def _calibrate(n: int = 200_000) -> float:
        """Cost of one wrapper's own bookkeeping, so inflated rows are identifiable."""
        wall: dict[str, float] = defaultdict(float)
        count: dict[str, int] = defaultdict(int)

        def victim():
            return None

        def wrapped():
            started = time.perf_counter()
            try:
                return victim()
            finally:
                wall["x"] += time.perf_counter() - started
                count["x"] += 1

        t0 = time.perf_counter()
        for _ in range(n):
            wrapped()
        measured = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(n):
            victim()
        bare = time.perf_counter() - t0
        return (measured - bare) / n

    def wrap(self, module, name: str, label: str | None = None) -> None:
        original = getattr(module, name)
        key = label or name

        def wrapper(*args, __orig=original, __key=key, **kwargs):
            started = time.perf_counter()
            try:
                return __orig(*args, **kwargs)
            finally:
                self.wall[__key] += time.perf_counter() - started
                self.count[__key] += 1

        setattr(module, name, wrapper)
        self._restore.append((module, name, original))

    def restore(self) -> None:
        for module, name, original in reversed(self._restore):
            setattr(module, name, original)
        self._restore.clear()

    def row(self, key: str) -> tuple[int, float, float]:
        n = self.count[key]
        return n, self.wall[key], n * self.overhead_s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--flights", type=int, default=100)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-iterations", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    args = ap.parse_args()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)),
        key=lambda r: (r.t_request, r.flight_id),
    )[: args.flights]
    static_terms = list(demand.terminals(cfg)) if cfg.terminal_airspace_always_active else []
    params = ColGenParams(max_iterations=args.max_iterations, time_limit_s=args.time_limit)
    pool_cfg = ParallelPricingConfig(n_workers=args.workers) if args.workers else None

    ledger = _Ledger()
    # Stage entry points (call counts in the ones-to-hundreds, so timing is exact).
    # solver.py binds these at import, so the patch must land on solver_mod, not on
    # the defining module -- patching network_mod here would silently record 0 calls.
    ledger.wrap(solver_mod, "build_flight_graph")
    ledger.wrap(solver_mod, "seed_column")
    ledger.wrap(solver_mod, "_canonical_column")
    ledger.wrap(solver_mod, "_initial_feasible_selection")
    ledger.wrap(solver_mod, "_greedy_feasible_selection")
    ledger.wrap(solver_mod, "find_feasible_column")
    ledger.wrap(solver_mod, "price_sweep")
    ledger.wrap(master_mod.RestrictedMaster, "solve_lp", "RestrictedMaster.solve_lp")
    ledger.wrap(master_mod.RestrictedMaster, "solve_ip", "RestrictedMaster.solve_ip")
    ledger.wrap(master_mod.RestrictedMaster, "add_column", "RestrictedMaster.add_column")
    if pool_cfg is None:
        ledger.wrap(solver_mod, "price_flight")
    # Children of find_feasible_column / price_flight.  These are what the stage residual
    # is measured against.
    for name in (
        "_shifted_seed_incumbent", "_path_claims", "_path_delay_s", "_canonical_candidate",
        "_endpoint_claims", "_visit_claims", "_rows_hit_forbidden", "_visit_hits_forbidden",
        "_best_column", "_topology_for", "seed_column",
    ):
        if hasattr(pricing_mod, name):
            ledger.wrap(pricing_mod, name, f"pricing.{name}")
    ledger.wrap(network_mod, "column_claims", "network.column_claims")

    started = time.perf_counter()
    try:
        result = ColGenSolver().solve(requests, cfg, static_terms, params, parallel=pool_cfg)
    finally:
        ledger.restore()
    wall = time.perf_counter() - started

    stats = result.stats
    print(f"scenario {args.scenario}  flights {len(requests)}  "
          f"workers {args.workers}  max_iterations {args.max_iterations}")
    print(f"objective {stats.get('objective')}  selected {stats.get('selected_flights')}"
          f"/{len(requests)}  wall {wall:.2f}s")
    print(f"wrapper overhead calibrated at {ledger.overhead_s * 1e9:.0f} ns/call "
          f"-- rows whose 'ovh' approaches their 'wall' are measurement, not work")
    print()

    def line(label: str, indent: int = 0) -> None:
        n, w, ovh = ledger.row(label)
        if not n:
            return
        pad = "  " * indent
        print(f"{pad}{label:<38} {n:>9,} {w:>9.2f}s {100 * w / wall:>6.1f}% "
              f"{1000 * w / n:>9.3f} ms/call  ovh {ovh:>6.2f}s")

    print(f"{'stage / function':<38} {'calls':>9} {'wall':>10} {'%':>7} {'per call':>12}")
    print("-" * 100)
    print("1. GRAPH BUILD (serial)")
    line("build_flight_graph", 1)
    print("2. SEEDING (serial)")
    line("seed_column", 1)
    line("_canonical_column", 1)
    line("RestrictedMaster.add_column", 1)
    print("3. INITIAL HEURISTIC / shifted seeds (serial)")
    line("_initial_feasible_selection", 1)
    print("4. MASTER LP/IP (serial)")
    line("RestrictedMaster.solve_lp", 1)
    line("RestrictedMaster.solve_ip", 1)
    print("5. GREEDY LOCAL SEARCH (serial)  <- the Amdahl tail")
    line("_greedy_feasible_selection", 1)
    line("find_feasible_column", 2)
    for label in ("pricing.seed_column", "pricing._shifted_seed_incumbent",
                  "pricing._canonical_candidate", "pricing._path_claims",
                  "pricing._path_delay_s", "pricing._endpoint_claims",
                  "pricing._visit_claims", "pricing._rows_hit_forbidden",
                  "pricing._visit_hits_forbidden", "network.column_claims"):
        line(label, 3)
    ffc_children = sum(
        ledger.wall[k] for k in (
            "pricing.seed_column", "pricing._shifted_seed_incumbent",
            "pricing._canonical_candidate", "pricing._path_claims", "pricing._path_delay_s",
            "pricing._endpoint_claims", "pricing._visit_claims",
            "pricing._rows_hit_forbidden", "pricing._visit_hits_forbidden",
        )
    )
    ffc = ledger.wall["find_feasible_column"]
    if ffc:
        print(f"      {'^ residual = best-first expansion loop':<36} "
              f"{'':>9} {ffc - ffc_children:>9.2f}s {100 * (ffc - ffc_children) / wall:>6.1f}%")
    print("6. PRICING SWEEP")
    if pool_cfg is None:
        line("price_flight", 1)
        line("pricing._best_column", 2)
        line("pricing._topology_for", 2)
    else:
        sweep = stats.get("parallel_sweep_wall_s", 0.0)
        total = stats.get("parallel_task_wall_total_s", 0.0)
        longest = stats.get("parallel_task_wall_max_s", 0.0)
        print(f"  price_sweep (parallel, {args.workers} workers)  makespan {sweep:.2f}s "
              f"= {100 * sweep / wall:.1f}% of wall")
        print(f"    task-sum {total:.2f}s across {len(requests)} flights "
              f"({1000 * total / max(1, len(requests)):.1f} ms/flight), "
              f"longest task {longest:.2f}s")
        print(f"    efficiency {100 * total / (sweep * args.workers):.1f}% of worker slots; "
              f"per-flight internals need a sequential run (workers lack these wrappers)")
    line("_canonical_column", 1)
    print("-" * 100)

    named = sum(
        ledger.wall[k] for k in (
            "build_flight_graph", "_initial_feasible_selection",
            "_greedy_feasible_selection", "RestrictedMaster.solve_lp", "RestrictedMaster.solve_ip",
            "RestrictedMaster.add_column", "price_sweep", "price_flight",
        )
    ) + ledger.wall["seed_column"]
    print(f"top-level stages account for {named:.2f}s of {wall:.2f}s "
          f"({100 * named / wall:.1f}%); the rest is solver.py's own bookkeeping "
          f"(row index, loads, column sorting)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
