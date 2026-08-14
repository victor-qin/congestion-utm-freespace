"""What `materialize_rows` costs, and how the pool it reads is shaped.

**The scan this was written to measure is gone.** `materialize_rows` used to rebuild one row
of the column->rows transpose per row it materialized:

    indices = [i for i, column in enumerate(self._columns) if row in column.claims]  # O(n_columns)
    self._backend.add_row(row, rhs, indices)                                         # O(|indices|)

That line was 99.8% of the method and a third of colgen's serial tail (issue #92): at x500,
26,227 rows x ~11,000 columns = 2.9e8 frozenset probes to find 15 hits per row, against a
backend insert of 0.07 s. `RestrictedMaster._columns_by_row` replaced it with a lookup and it
now measures ~0.1 s.

So this script no longer decomposes anything: the `host-side` row below is a lookup plus
normalization, sorting and bookkeeping, NOT a scan, and it is not the ceiling of an
optimization still to come. Two things it is still good for:

* **Regression detection.** `materialize_rows` should stay ~0.1 s at x500. A figure in the
  tens of seconds means the index stopped being maintained -- most likely because something
  started trimming `self._columns`, which would silently invalidate every stored position.
* **Pool shape.** The row-degree distribution and step spans below are what decided the index
  design, and they are what any successor design has to be argued against.

To measure the old scan again, check out a tree from before the index and run this there;
nothing here reimplements the shipped body (it wraps `materialize_rows` and the backend's
`add_row`, so `total - backend` cannot drift from the real code).

Two iterations is enough. The departure ladder puts ~11k of the final 11.9k columns in the
master before the first LP, so `n_columns` is already at full scale in iteration 1, and only
the row count grows after that.

    uv run python analysis/probe_materialize_rows.py --flights 500 --iterations 2 --workers 16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim.planner.colgen import master as master_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

STATS = {
    "materialize_calls": 0,
    "materialize_s": 0.0,
    "backend_add_row_calls": 0,
    "backend_add_row_s": 0.0,
    "indices_total": 0,
    "columns_at_entry": [],
    "rows_per_call": [],
}


def install() -> None:
    cls = master_mod.RestrictedMaster
    materialize_rows = cls.materialize_rows

    def probed_materialize_rows(self, rows):
        rows = list(rows)
        before = STATS["backend_add_row_calls"]
        STATS["columns_at_entry"].append(len(self._columns))
        started = time.perf_counter()
        try:
            return materialize_rows(self, rows)
        finally:
            STATS["materialize_s"] += time.perf_counter() - started
            STATS["materialize_calls"] += 1
            STATS["rows_per_call"].append(STATS["backend_add_row_calls"] - before)

    cls.materialize_rows = probed_materialize_rows

    # Wrap every backend the solve might pick, on the class, so the timer survives whichever
    # `create_backend` returns.
    for backend_cls in (master_mod.HighsBackend, getattr(master_mod, "GurobiBackend", None)):
        if backend_cls is None:
            continue
        add_row = backend_cls.add_row

        def probed_add_row(self, row, rhs, column_indices, _add_row=add_row):
            STATS["indices_total"] += len(column_indices)
            started = time.perf_counter()
            try:
                return _add_row(self, row, rhs, column_indices)
            finally:
                STATS["backend_add_row_s"] += time.perf_counter() - started
                STATS["backend_add_row_calls"] += 1

        backend_cls.add_row = probed_add_row


def _index_shape(pool, master) -> None:
    """Which cheaper-than-full inverted index is worth building?

    Two competing refinements to a flat ``dict[RowKey, list[int]]``, and each is only worth
    its complexity if the pool has the right shape:

    * **promote-on-second-insert** (``int | list``) pays off in proportion to SINGLETON rows.
      A cell row has cap 1, so it needs load > 1 to be violated and a single column can only
      reach 1.0 -- singleton rows are not merely rare queries, they are *unviolatable*, and
      the list object wrapping them is pure waste.
    * **step bucketing** (``dict[int, list[int]]`` over each column's step span, then an exact
      membership check on the candidates) pays off in proportion to how much of the horizon a
      single column does NOT span.

    Also reports the degree of rows that were actually materialized against the degree of all
    rows, because the fix is queried only on the former.
    """

    from collections import Counter

    degree: Counter = Counter()
    span_entries = 0
    steps_seen: Counter = Counter()
    for column in pool:
        steps = [row.step for row in column.claims]
        if steps:
            low, high = min(steps), max(steps)
            span_entries += high - low + 1
            for step in range(low, high + 1):
                steps_seen[step] += 1
        for row in column.claims:
            degree[row] += 1

    buckets = [("1 (unviolatable)", 1, 1), ("2", 2, 2), ("3-5", 3, 5),
               ("6-15", 6, 15), ("16+", 16, 1 << 30)]
    total_rows = len(degree)
    total_pairs = sum(degree.values())
    print("\n--- row degree distribution (all rows in the pool) ---")
    for label, low, high in buckets:
        rows = sum(1 for d in degree.values() if low <= d <= high)
        pairs = sum(d for d in degree.values() if low <= d <= high)
        print(f"degree {label:18s} {rows:>9d} rows ({100 * rows / max(1, total_rows):5.1f}%)"
              f"   {pairs:>9d} pairs ({100 * pairs / max(1, total_pairs):5.1f}%)")

    materialized = master.materialized_rows
    if materialized:
        degrees = [degree.get(row, 0) for row in materialized]
        print(f"\nmaterialized rows  {len(materialized):>9d}   mean degree "
              f"{sum(degrees) / len(degrees):.1f}   min {min(degrees)}   max {max(degrees)}")
        singleton_materialized = sum(1 for d in degrees if d <= 1)
        print(f"  of which degree<=1 {singleton_materialized:>7d}   "
              "(should be ~0: a lone column cannot exceed cap 1)")

    if steps_seen:
        mean_active = sum(steps_seen.values()) / len(steps_seen)
        print("\n--- step bucketing ---")
        print(f"distinct steps     {len(steps_seen):>9d}")
        print(f"index entries      {span_entries:>9d}   (vs {total_pairs} for the full index, "
              f"{total_pairs / max(1, span_entries):.1f}x smaller)")
        print(f"mean columns/step  {mean_active:>9.0f}   of {len(pool)} "
              f"({len(pool) / max(1.0, mean_active):.1f}x pruning before the exact check)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="density_future_wing_zipline")
    parser.add_argument("--flights", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    params = ColGenParams(
        max_iterations=args.iterations,
        time_limit_s=1e9,
        n_pricing_workers=args.workers,
        bootstrap_roots=1,
        bootstrap_ranking="bound",
        gap_metric="revenue",
    )

    # The inverted index would hold one entry per (column, claim) pair, so its size is a
    # property of the final pool rather than of the rows anyone materialized. Capture the
    # pool itself: issue #85 has this planner RAM-bound already, and "just keep a
    # row -> [column_index] map" is only cheap if that number is small.
    masters: list[object] = []
    add_column = master_mod.RestrictedMaster.add_column

    def probed_add_column(self, column):
        if not masters or masters[-1] is not self:
            masters.append(self)
        return add_column(self, column)

    master_mod.RestrictedMaster.add_column = probed_add_column

    install()
    started = time.perf_counter()
    ColGenSolver().solve(requests, cfg, [], params)
    wall = time.perf_counter() - started

    rows = STATS["backend_add_row_calls"]
    total = STATS["materialize_s"]
    backend = STATS["backend_add_row_s"]
    host_side = total - backend
    columns = STATS["columns_at_entry"]
    mean_columns = sum(columns) / len(columns) if columns else 0.0
    print(f"\n--- materialize_rows, {args.scenario} x{args.flights}, {args.iterations} iters ---")
    print(f"WALL {wall:.2f}s")
    print(f"calls              {STATS['materialize_calls']:>10d}")
    print(f"rows materialized  {rows:>10d}   ({rows / max(1, STATS['materialize_calls']):.1f}/call)")
    print(f"n_columns at entry {mean_columns:>10.0f}   (mean; max {max(columns or [0])})")
    print(f"scan work AVOIDED  {rows * mean_columns:>10.3g}   rows x n_columns, the "
          "counterfactual the index removed")
    print()
    print(f"total              {total:>10.2f}s   100.0%")
    print(f"  host-side        {host_side:>10.2f}s   "
          f"{100 * host_side / total if total else 0:5.1f}%"
          "   <- lookup + normalize + sort + bookkeeping (NOT a scan)")
    print(f"  backend add_row  {backend:>10.2f}s   {100 * backend / total if total else 0:5.1f}%"
          "   <- proportional to |indices|")
    print()
    print(f"indices emitted    {STATS['indices_total']:>10d}   "
          f"({STATS['indices_total'] / max(1, rows):.1f} columns claim the average row)")
    # A THRESHOLD, not a ratio. The host/backend split is meaningless once both are ~0 -- at
    # four flights it reads 54% host-side off a total near the timer's own resolution. What
    # still carries information is the magnitude.
    verdict = (
        "OK -- the index is live"
        if total < 1.0
        else "SUSPECT -- expected ~0.1 s at x500; is `_columns_by_row` still maintained?"
    )
    print(f"\nmaterialize_rows total {total:.2f}s: {verdict}")

    if masters:
        pool = masters[-1].columns
        pairs = sum(len(column.claims) for column in pool)
        distinct = len({row for column in pool for row in column.claims})
        _index_shape(pool, masters[-1])
        # A naive `dict[RowKey, list[int]]` sizing -- one dict slot per distinct row plus one
        # list slot per pair, both over-allocating. An UPPER BOUND: it charges full price for
        # dict keys that are REFERENCES to `RowKey` objects already inside `column.claims`,
        # and for indices that are references to the ~11k shared `int` objects `add_column`
        # already made rather than one per pair.
        #
        # DO NOT MEASURE THE INDEX'S RSS FROM THIS SCRIPT. `_index_shape` above, and the
        # `distinct` set below, each build a ~1.28M-entry structure at x500 -- hundreds of MB
        # that dominate peak RSS in BOTH arms and mask the very delta you would be looking
        # for. Doing exactly that read +25 MB, which is wrong by ~8x.
        #
        # The honest figures come from `ab_colgen_parity.py`'s `rss_self_mb`, which is the
        # same `ru_maxrss` taken in a process that runs no analysis pass: at x500 / 2
        # iterations / ladder 20, parent peak 1,823 -> 2,029 MB on density_future (+206 MB)
        # and 1,772 -> 1,878 MB on density_faa (+106 MB). Nearer this estimate than not.
        estimate_mb = (distinct * 100 + pairs * 40) / 1e6
        print("\n--- inverted index sizing (final pool) ---")
        print(f"columns            {len(pool):>10d}")
        print(f"distinct rows      {distinct:>10d}")
        print(f"(column,row) pairs {pairs:>10d}   ({pairs / max(1, len(pool)):.0f} claims/column)")
        print(f"index size         {estimate_mb:>10.0f} MB UPPER BOUND (ab_colgen_parity measured "
              "+106..206 MB at x500; NOT measurable from this script -- see above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
