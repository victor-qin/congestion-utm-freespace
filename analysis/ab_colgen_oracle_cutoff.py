"""The oracle-cutoff bound: what a *perfect* pricing incumbent would be worth.

Issue #90 argues that a flight which exhausts the compiled search's label pool does so
because it enters the search with a weak cutoff, and proposes a bootstrap round to
strengthen it.  A bootstrap is itself a search, so it can only ever recover *part* of what
a perfect cutoff would.  This measures the whole part -- the ceiling on that idea -- and
it is the number to read before writing any of it.

Two passes, deliberately in **separate processes**, because ``fg._search_cache.dag_budget``
memoizes the pool the last search on that graph settled on: a second solve in the same
process would start from the first's budget and the label counts would not be comparable.

* ``--capture PATH``  solve normally; pickle the column each flight's pricing returned.
* ``--inject PATH``   solve again, handing each flight *its own answer* as
  ``known_column``.  That is the same seam the master's heuristic column already uses
  (``pricing.py:2847``), so nothing new is exercised -- the incumbent is simply as good as
  an incumbent can be.

What to read:

* ``labels`` collapsing means the cutoff was the binding constraint, and a bootstrap that
  gets anywhere near the optimum is worth building.
* ``labels`` barely moving means the labels are being spent on something the cutoff does
  not gate -- the dominance key's width is the other candidate (``pricing.py:1291``) --
  and the bootstrap is the wrong lever.

**This is a bound, not a proposal.**  Injecting the answer makes ``price_flight`` return
``(rc, None)`` for every flight whose optimum it was handed, by the deliberate rule at
``pricing.py:2768-2770``, so the master receives no columns and the solve after iteration
1 is not the same solve.  Run it at ``--iterations 1`` and read the label counts only.

    uv run python analysis/ab_colgen_oracle_cutoff.py --capture /tmp/oracle.pkl \
        --scenario density_faa_wing_zipline --flights 12
    uv run python analysis/ab_colgen_oracle_cutoff.py --inject /tmp/oracle.pkl \
        --scenario density_faa_wing_zipline --flights 12
"""
from __future__ import annotations

import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen import pricing as pricing_mod  # noqa: E402
from freespace_sim.planner.colgen import pricing_pool as pricing_pool_mod  # noqa: E402
from freespace_sim.planner.colgen import solver as solver_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

try:
    from freespace_sim.planner.colgen import dp_kernel as dp_kernel_mod  # noqa: E402
except ImportError:  # pragma: no cover - depends on the install
    dp_kernel_mod = None

ROWS: list[dict] = []
CURRENT: dict = {}
CAPTURED: dict[int, object] = {}


def install(inject: dict[int, object] | None) -> None:
    price_flight = pricing_mod.price_flight

    def wrapped(fg, *args, **kwargs):
        flight_id = fg.request.flight_id
        if inject is not None and flight_id in inject and not kwargs.get("forbidden_rows"):
            kwargs["known_column"] = inject[flight_id]
        CURRENT.clear()
        CURRENT.update(flight=flight_id, entry_rc=None, labels=None, attempts=None,
                       status=None, search_s=0.0, fallback_s=0.0)
        started = time.perf_counter()
        try:
            reduced_cost, column = price_flight(fg, *args, **kwargs)
            CURRENT["final_rc"] = float(reduced_cost)
            if column is not None:
                CAPTURED[flight_id] = column
            return reduced_cost, column
        finally:
            CURRENT["total_s"] = time.perf_counter() - started
            ROWS.append(dict(CURRENT))

    best_compiled = pricing_mod._best_column_compiled

    def timed_best_compiled(*args, incumbent=None, **kwargs):
        CURRENT["entry_rc"] = -math.inf if incumbent is None else float(incumbent[0])
        started = time.perf_counter()
        try:
            return best_compiled(*args, incumbent=incumbent, **kwargs)
        finally:
            CURRENT["search_s"] = time.perf_counter() - started

    best_reference = pricing_mod._best_column

    def timed_best_reference(*args, **kwargs):
        fallback = not kwargs.get("seed", False)
        started = time.perf_counter()
        try:
            return best_reference(*args, **kwargs)
        finally:
            if fallback and CURRENT:
                CURRENT["fallback_s"] += time.perf_counter() - started

    pricing_mod.price_flight = wrapped
    pricing_pool_mod.price_flight = wrapped
    solver_mod.price_flight = wrapped
    pricing_mod._best_column_compiled = timed_best_compiled
    pricing_mod._best_column = timed_best_reference

    if dp_kernel_mod is not None:
        price_dag = dp_kernel_mod.price_dag

        def probed(*args, **kwargs):
            result = price_dag(*args, **kwargs)
            if CURRENT:
                CURRENT["labels"] = int(result.n_labels)
                CURRENT["attempts"] = int(result.attempts)
                CURRENT["status"] = dp_kernel_mod.STATUS_NAMES.get(
                    result.status, str(result.status)
                )
            return result

        dp_kernel_mod.price_dag = probed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--capture", type=Path)
    group.add_argument("--inject", type=Path)
    parser.add_argument("--scenario", default="density_faa_wing_zipline")
    parser.add_argument("--flights", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-label-log2", type=int, default=None)
    args = parser.parse_args()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    static_terms = list(demand.terminals(cfg))
    params = ColGenParams(
        solver="highs", max_iterations=args.iterations, time_limit_s=86400.0,
        gap_metric="cost",
        # SEQUENTIAL, and not optional here.  This probe monkeypatches module-level
        # functions in THIS process, and `n_pricing_workers` now DEFAULTS TO 4.  The pool
        # uses the `spawn` context, so a worker re-imports the module and binds the REAL
        # function: the patch would reach only the parent, which prices nothing, and the
        # report would come back empty with no error saying why.
        n_pricing_workers=0,
    )
    if args.max_label_log2 is not None and dp_kernel_mod is not None:
        dp_kernel_mod.MAX_LABEL_CAPACITY = 1 << args.max_label_log2

    inject = None
    if args.inject is not None:
        # Pickle rather than JSON because a `Column` carries a frozenset of `RowKey`, and a
        # JSON round trip would need a schema that drifts from the dataclass.  The only
        # producer is this script's own `--capture`, run by hand on the same machine
        # minutes earlier; nothing here reads a file it did not write.
        with args.inject.open("rb") as handle:
            inject = pickle.load(handle)
        print(f"injecting {len(inject)} captured optima as known_column")

    print(f"tree      {_loaded.parent.parent}")
    print(f"workload  {args.scenario} x{len(requests)} iters={args.iterations} "
          f"mode={'inject' if inject is not None else 'capture'}")
    if dp_kernel_mod is not None:
        print(f"ceilings  MAX_LABEL_CAPACITY={dp_kernel_mod.MAX_LABEL_CAPACITY:,}")

    install(inject)
    started = time.perf_counter()
    result = ColGenSolver().solve(requests, cfg, static_terms, params)
    wall = time.perf_counter() - started

    print("\n--- PER FLIGHT (first sweep, slowest first) ---")
    print(f"{'flight':>7} {'entry_rc':>14} {'final_rc':>14} {'gap':>12} {'labels':>12} "
          f"{'att':>4} {'status':>12} {'search_s':>9} {'fallb_s':>9} {'total_s':>9}")
    seen: set[int] = set()
    first_sweep = []
    for row in ROWS:
        if row["flight"] in seen:
            break
        seen.add(row["flight"])
        first_sweep.append(row)
    for row in sorted(first_sweep, key=lambda r: -r["total_s"]):
        entry = row["entry_rc"]
        final = row.get("final_rc")
        gap = None if final is None or entry is None or not math.isfinite(entry) else final - entry
        print(
            f"{row['flight']:>7} "
            f"{'-inf' if entry is None or not math.isfinite(entry) else format(entry, '14.4f')} "
            f"{'n/a' if final is None else format(final, '14.4f')} "
            f"{'n/a' if gap is None else format(gap, '12.4f')} "
            f"{'n/a' if row['labels'] is None else format(row['labels'], ',d'):>12} "
            f"{'n/a' if row['attempts'] is None else row['attempts']:>4} "
            f"{str(row['status']):>12} "
            f"{row['search_s']:>9.2f} {row['fallback_s']:>9.2f} {row['total_s']:>9.2f}"
        )
    print(f"\nfirst sweep total {sum(r['total_s'] for r in first_sweep):.1f}s")
    print(f"WALL {wall:.2f}s  obj={result.stats.get('objective')!r}")
    print(f"kernel {pricing_mod.kernel_stats()}")

    if args.capture is not None:
        with args.capture.open("wb") as handle:
            pickle.dump(CAPTURED, handle)
        print(f"captured {len(CAPTURED)} columns -> {args.capture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
