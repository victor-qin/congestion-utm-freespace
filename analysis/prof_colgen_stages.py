"""Where a column-generation solve spends its time, by stage and by function.

Two views, and the split between them is the point:

* **Stage timers** -- plain ``perf_counter`` wrappers on the dozen functions that carve a
  solve into phases. **These are authoritative for magnitude.** They add one timer call per
  invocation to functions called hundreds to thousands of times, which is negligible.
* **cProfile** -- ``tottime``/``cumtime`` ranking. **Ranking only; the magnitudes are not
  usable here.** Colgen's hot path is tens of millions of tiny calls, exactly the shape
  cProfile's per-call overhead distorts: the same 12-flight solve measures 38.85 s bare and
  152.9 s under cProfile, a 3.9x inflation that is not uniform across functions.

Both run **fixed work** -- ``max_iterations`` pinned, ``time_limit_s`` effectively infinite.
A time-limited solve prices as many flights as it can afford, so instrumentation changes
which subproblems are reached and the profile describes a different computation than the
one you meant to measure.

Note ``solver`` binds ``seed_column`` / ``find_feasible_column`` at import, and
``pricing_pool`` binds ``price_flight``, so patching only ``pricing`` instruments nothing
the solve actually calls. Every module object holding such a binding is patched below.

**Profile sequentially.** Neither these timers nor cProfile cross a process boundary, so
with ``n_pricing_workers`` set, everything inside the sweep happens in workers and is
invisible here --
the parent would show only the greedy, the LP and canonicalization. A sequential sweep is
exactly one worker's workload, which is the thing worth ranking anyway.

**Lazy row separation is split by CALL SITE, not summed.** ``add_violated_rows`` is the
largest serial item at 500 flights, and the rows nested under it -- ``fractional_loads`` and
``materialize_rows`` -- are what decides which fix to write. They are not comparable:
``fractional_loads`` walks only the LP support (sparse, ~1.5k of 11.9k columns), while
``materialize_rows`` walks the whole pool once per row (dense), which is why it measures ~12x
the other. ``materialize_rows`` also has a second caller, ``solve_ip``'s separation loop, and
that time appears as ``materialize_rows (from solve_ip)`` rather than inflating the parent.
``add_violated_rows`` MINUS its two nested rows is the ``violated`` comprehension itself.

**Read fallbacks off ``--- KERNEL ---``, never off the stderr warnings.**
``pricing._warn_budget`` guards on module globals, so it fires once per PROCESS per kind,
and the pool is rebuilt per sweep -- the warning count is therefore roughly
``n_workers x kinds x iterations`` and carries no information about how many flights
declined. ``kernel_fell_back`` is the only figure valid in both modes.

Examples:

    uv run python analysis/prof_colgen_stages.py --flights 12 --iterations 3
    uv run python analysis/prof_colgen_stages.py --scenario density_faa_wing_zipline \
        --flights 20 --iterations 2 --no-cprofile

    # how far past the shipped label ceiling does a workload actually reach?
    uv run python analysis/prof_colgen_stages.py --scenario density_faa_wing_zipline \
        --flights 12 --iterations 2 --no-cprofile --max-label-log2 26
"""
from __future__ import annotations

import argparse
import collections
import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim import volumes as volumes_mod  # noqa: E402
from freespace_sim.planner.colgen import dp_prepare as dp_prepare_mod  # noqa: E402
from freespace_sim.planner.colgen import master as master_mod  # noqa: E402
from freespace_sim.planner.colgen import network as network_mod  # noqa: E402
from freespace_sim.planner.colgen import pricing as pricing_mod  # noqa: E402
from freespace_sim.planner.colgen import pricing_pool as pricing_pool_mod  # noqa: E402
from freespace_sim.planner.colgen import solver as solver_mod  # noqa: E402
from freespace_sim.planner.colgen import translate as translate_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

# Optional: absent on a numba-less install, which is exactly the run where every pricing
# call falls back and the `price_dag` row would read 0 anyway.
try:
    from freespace_sim.planner.colgen import dp_kernel as dp_kernel_mod  # noqa: E402
except ImportError:  # pragma: no cover - depends on the install
    dp_kernel_mod = None

TOTAL_S: collections.Counter = collections.Counter()
CALLS: collections.Counter = collections.Counter()

# (label, [(module, attribute), ...]) -- a stage may be bound in more than one module.
STAGES: list[tuple[str, list[tuple[object, str]]]] = [
    # The top of the sweep, and the one binding that MOVED: `pricing_pool` imports
    # `price_flight` directly, so patching only `pricing` would instrument nothing the
    # sweep calls.  Both are wrapped, and only one of them fires per run -- the pool's
    # when `n_pricing_workers` is set (and then only in the parent, since cProfile and these
    # timers do not cross a process boundary), `pricing`'s otherwise.
    (
        "price_flight (sweep entry)",
        [(pricing_mod, "price_flight"), (pricing_pool_mod, "price_flight")],
    ),
    # `_best_column_compiled` is the entry to exact pricing since Phase 2c; `_best_column`
    # runs only when it cannot prove it completed, so a nonzero row there is a FALLBACK
    # count and worth reading as one.
    ("_best_column_compiled (Tier 1+2, entry)", [(pricing_mod, "_best_column_compiled")]),
    ("_best_column (reference FALLBACK)", [(pricing_mod, "_best_column")]),
    # THE COMPILED BORDER.  `price_dag` is the host wrapper, not the `@njit` function, so
    # this row spans the kernel PLUS every pause it makes -- `certify`, `envelopes.envelope`,
    # candidate-buffer drains -- all of which run in Python inside its resume loop.
    # Compiled time is therefore this row MINUS the `_canonical_candidate` and envelope time
    # attributed below; without it the whole kernel side landed in an unattributed remainder.
    ("  price_dag (COMPILED BORDER, incl. pauses)", [(dp_kernel_mod, "price_dag")]),
    ("  _dag_candidates (per-sink pricing)", [(pricing_mod, "_dag_candidates")]),
    ("  _certify_candidates (Tier 2 rank)", [(pricing_mod, "_certify_candidates")]),
    ("  prepared_for (cached packing)", [(dp_prepare_mod, "prepared_for")]),
    ("  prepare_duals (per sweep)", [(dp_prepare_mod, "prepare_duals")]),
    ("  prepare_variants (roots + gate)", [(dp_prepare_mod, "prepare_variants")]),
    ("  prepare_forbidden (per repair call)", [(dp_prepare_mod, "prepare_forbidden")]),
    ("_canonical_candidate (Tier 2 + incumbent)", [(pricing_mod, "_canonical_candidate")]),
    ("_path_claims", [(pricing_mod, "_path_claims")]),
    ("_path_delay_s", [(pricing_mod, "_path_delay_s")]),
    ("_endpoint_claims", [(pricing_mod, "_endpoint_claims")]),
    ("_endpoint_claims_uncached", [(pricing_mod, "_endpoint_claims_uncached")]),
    ("_canonical_column", [(solver_mod, "_canonical_column")]),
    ("seed_column", [(pricing_mod, "seed_column"), (solver_mod, "seed_column")]),
    (
        "find_feasible_column (greedy)",
        [(pricing_mod, "find_feasible_column"), (solver_mod, "find_feasible_column")],
    ),
    ("_shifted_seed_incumbent", [(pricing_mod, "_shifted_seed_incumbent")]),
    ("_greedy_feasible_selection", [(solver_mod, "_greedy_feasible_selection")]),
    ("column_claims", [(pricing_mod, "column_claims")]),
    ("column_to_intent", [(pricing_mod, "column_to_intent")]),
    # The `volumes.py` geometry both rulers bottom out in, and the answer to "would
    # compiling the ledger side help".  These are LEAVES of the rows above -- `_path_delay_s`
    # is essentially `enroute_flown_m`, and `column_to_intent` is essentially
    # `build_reservation_from_corners` -- so their sum is the ceiling on that idea, not an
    # addition to it.  Every module ON THE COLGEN PATH holding a direct-name binding is
    # patched, per the note in the module docstring -- `metrics`, `milp` and `shortcut` bind
    # these names too but no colgen solve reaches them, so a row here is this planner's cost
    # and not the ledger's in general.  `fold_corners_to_columns` is only ever reached as a
    # `volumes` module global, hence the single site.
    (
        "  enroute_flown_m (volumes)",
        [(volumes_mod, "enroute_flown_m"), (pricing_mod, "enroute_flown_m"),
         (translate_mod, "enroute_flown_m")],
    ),
    ("    fold_corners_to_columns (volumes)", [(volumes_mod, "fold_corners_to_columns")]),
    (
        "  build_reservation_from_corners (volumes)",
        [(volumes_mod, "build_reservation_from_corners"),
         (translate_mod, "build_reservation_from_corners")],
    ),
    (
        "  column_dwell_s (volumes)",
        [(volumes_mod, "column_dwell_s"), (pricing_mod, "column_dwell_s"),
         (translate_mod, "column_dwell_s"), (network_mod, "column_dwell_s"),
         (dp_prepare_mod, "column_dwell_s")],
    ),
]

# Bound methods, timed by wrapping the class attribute.
METHOD_STAGES: list[tuple[str, object, str]] = [
    ("master.solve_lp", master_mod.RestrictedMaster, "solve_lp"),
    ("master.round_heuristic", master_mod.RestrictedMaster, "round_heuristic"),
    ("master.add_column", master_mod.RestrictedMaster, "add_column"),
    ("master.solve_ip", master_mod.RestrictedMaster, "solve_ip"),
    # `add_violated_rows` is installed by `_install_separation_timers` instead, because it
    # has to set the call-site flag those timers read.
]

# Set while the stack is inside `add_violated_rows`, so `materialize_rows` can be charged to
# the caller that reached it.
_IN_ADD_VIOLATED = False


def _install_separation_timers() -> None:
    """Split lazy row separation into its two halves, attributed by call site.

    `add_violated_rows` is the largest single serial item at 500 flights, and it is two
    candidates rather than one: `fractional_loads` accumulates loads over every column in
    the LP support, then `materialize_rows` scans every column once per row it adds. They
    have different fixes -- an early materialized-row filter versus an inverted
    row->column index -- so a single outer timer cannot say which one to write.

    Attributing by call site rather than summing: `materialize_rows` has a SECOND caller,
    `solve_ip`'s separation loop (`master.py:896`), which reaches it with the *integral*
    violated set. That path shares the per-row column scan and so shares the inverted-index
    fix, but it is not part of `add_violated_rows` and folding the two together would
    over-state the outer method.
    """

    cls = master_mod.RestrictedMaster
    add_violated_rows = cls.add_violated_rows
    fractional_loads = cls.fractional_loads
    materialize_rows = cls.materialize_rows

    def timed_add_violated_rows(self, *args, **kwargs):
        global _IN_ADD_VIOLATED
        previous, _IN_ADD_VIOLATED = _IN_ADD_VIOLATED, True
        started = time.perf_counter()
        try:
            return add_violated_rows(self, *args, **kwargs)
        finally:
            _IN_ADD_VIOLATED = previous
            TOTAL_S["master.add_violated_rows"] += time.perf_counter() - started
            CALLS["master.add_violated_rows"] += 1

    def timed_fractional_loads(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return fractional_loads(self, *args, **kwargs)
        finally:
            TOTAL_S["  fractional_loads (nested)"] += time.perf_counter() - started
            CALLS["  fractional_loads (nested)"] += 1

    def timed_materialize_rows(self, *args, **kwargs):
        label = (
            "  materialize_rows (nested)" if _IN_ADD_VIOLATED
            else "  materialize_rows (from solve_ip)"
        )
        started = time.perf_counter()
        try:
            return materialize_rows(self, *args, **kwargs)
        finally:
            TOTAL_S[label] += time.perf_counter() - started
            CALLS[label] += 1

    cls.add_violated_rows = timed_add_violated_rows
    cls.fractional_loads = timed_fractional_loads
    cls.materialize_rows = timed_materialize_rows


def _wrap(label: str, target: object, name: str) -> None:
    original = getattr(target, name)

    def timed(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            TOTAL_S[label] += time.perf_counter() - started
            CALLS[label] += 1

    timed.__name__ = getattr(original, "__name__", name)
    setattr(target, name, timed)


def install_timers() -> None:
    for label, sites in STAGES:
        for module, name in sites:
            if hasattr(module, name):
                _wrap(label, module, name)
    for label, cls, name in METHOD_STAGES:
        if hasattr(cls, name):
            _wrap(label, cls, name)
    _install_separation_timers()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="colgen_test")
    parser.add_argument("--flights", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--solver", default="highs", choices=("highs", "gurobi"))
    parser.add_argument(
        "--gap-metric", default="cost", choices=("cost", "revenue"),
        help="`cost` by default, NOT the shipped `revenue`. The revenue gap is diluted by "
             "n*M and reads 2.67e-05 on density against a 1e-4 threshold, so the solve "
             "terminates at iteration 1 and the profile describes seeding rather than "
             "column generation.",
    )
    parser.add_argument(
        "--no-cprofile", action="store_true",
        help="skip the cProfile pass. Halves the run and loses only the function ranking; "
             "the stage table is unaffected either way.",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="fan the pricing sweep across N worker processes (0 = in-process). READ THE "
             "WARNING THIS PRINTS: neither the stage timers nor cProfile cross a process "
             "boundary, so above 0 the whole sweep becomes invisible to both and what is "
             "left is a profile of the SERIAL side only. Useful for exactly that -- what "
             "the parent still does while the pool works -- and misleading for anything "
             "else.",
    )
    parser.add_argument(
        "--chunksize", type=int, default=1,
        help="tasks handed to a worker at a time (`pool.imap`'s third argument). The "
             "default of 1 is almost certainly right here and raising it is a trap: "
             "chunking amortizes DISPATCH, and a pricing task ships one int in and ~14 KB "
             "out against tens of seconds of compute, so there is nothing to amortize. "
             "What it costs instead is load balance -- `mp.Pool` pre-partitions the "
             "iterable, per-flight cost varies several-fold, and n/chunksize chunks across "
             "n_workers lanes leaves nothing to rebalance a straggler against. Exposed so "
             "that claim can be MEASURED rather than asserted.",
    )
    parser.add_argument(
        "--max-label-log2", type=int, default=None,
        help="override `dp_kernel.MAX_LABEL_CAPACITY` to 2**N for this run (default: leave "
             "the shipped 1<<26 alone). Answer-neutral by the same argument the constant "
             "carries -- a budget bounds work, never the search -- so this changes how many "
             "flights DECLINE and fall back to the Python reference, not what any of them "
             "returns. Read as a knob for measuring how far past the shipped ceiling a "
             "workload actually reaches. Labels cost ~40 B each across the nine arrays, so "
             "N=26 is ~2.7 GB per worker and N=27 is ~5.4 GB.",
    )
    parser.add_argument(
        # CEILING, not the ladder's first rung.  `prof_colgen_cutoff.py --log2cap` is the
        # other end of the same ladder (it sets the STARTING rung, `INITIAL_LOG2CAP`), and
        # the two names are one letter apart -- check which tool you are in.
        "--max-log2cap", type=int, default=None,
        help="override `dp_kernel.MAX_LOG2CAP` (dominance table ceiling, shipped 26). "
             "Same answer-neutrality argument; ~32 B per slot.",
    )
    parser.add_argument(
        "--objective", default="total_cost", choices=("total_delay", "total_cost"),
        help="the shipped default. `total_delay` weights ground and air equally, which is "
             "a ~40x slower regime for pricing (issue #91) and profiles a different solve.",
    )
    # The bootstrap and its ranking change the SHAPE of what is profiled, not just the
    # clock: they cut pricing without touching the greedy or the LP, so a profile taken
    # without them over-states pricing's share and under-states everything serial. The
    # serial remainder is exactly what this tool is usually pointed at, so defaulting these
    # off made every such reading wrong by that ratio.
    parser.add_argument("--bootstrap-roots", type=int, default=0, metavar="K")
    parser.add_argument(
        "--bootstrap-ranking", default="score", choices=("score", "bound"),
    )
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    # Patched on the MODULE, which works because both ceilings are read as module globals
    # inside the host `price_dag`/`find_feasible_dag` retry loops -- not inside the `@njit`
    # kernels, which could not see a rebound global anyway.
    #
    # PARENT ONLY, AND THAT IS WHY THE COMBINATION IS REFUSED.  `_sweep_parallel` uses the
    # `spawn` context and its initargs carry no ceilings (`pricing_pool.py:355-366`), so a
    # worker imports `dp_kernel` fresh and gets the SHIPPED constant. Under `--workers N`
    # the override would reach only the parent -- which prices nothing -- so the arm would
    # silently run at `1 << 25` while its header claimed otherwise. Refuse rather than
    # measure the wrong thing quietly; overriding for a pooled sweep needs the value
    # threaded through the initializer, which is a change to shipped code, not to this
    # script.
    if args.workers and (args.max_label_log2 is not None or args.max_log2cap is not None):
        raise SystemExit(
            "--max-label-log2/--max-log2cap only reach the parent process, and a pooled "
            "sweep prices in spawned workers that re-import dp_kernel with the shipped "
            f"ceilings ({dp_kernel_mod.MAX_LABEL_CAPACITY:,} / "
            f"{dp_kernel_mod.MAX_LOG2CAP}). Drop --workers, or thread the override through "
            "pricing_pool's initializer first."
            if dp_kernel_mod is not None else
            "--max-label-log2/--max-log2cap require the compiled kernel"
        )
    if dp_kernel_mod is not None:
        if args.max_label_log2 is not None:
            dp_kernel_mod.MAX_LABEL_CAPACITY = 1 << args.max_label_log2
        if args.max_log2cap is not None:
            dp_kernel_mod.MAX_LOG2CAP = args.max_log2cap

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
        max_iterations=args.iterations,
        time_limit_s=86400.0,
        gap_metric=args.gap_metric,
        objective=args.objective,
        bootstrap_roots=args.bootstrap_roots,
        bootstrap_ranking=args.bootstrap_ranking,
        n_pricing_workers=args.workers,
        pricing_chunksize=args.chunksize,
    )

    print(f"tree      {_loaded.parent.parent}")
    print(f"workload  {args.scenario} x{len(requests)} iters={args.iterations} "
          f"obj={args.objective} K={args.bootstrap_roots}/{args.bootstrap_ranking} "
          f"{args.solver} gap={args.gap_metric} workers={args.workers} "
          f"chunksize={args.chunksize} (fixed work, no clock)")
    if dp_kernel_mod is not None:
        # Printed unconditionally, not only when overridden: an arm that declines is only
        # interpretable against the ceiling it was run at, and these are now runtime values.
        print(f"ceilings  MAX_LABEL_CAPACITY={dp_kernel_mod.MAX_LABEL_CAPACITY:,} "
              f"MAX_LOG2CAP={dp_kernel_mod.MAX_LOG2CAP}")
    if args.workers:
        print(
            "WARNING   the stage timers and cProfile below cover THIS PROCESS ONLY. With a\n"
            "          worker pool the entire pricing sweep runs elsewhere, so it will not\n"
            "          appear -- read the stage table as the serial tail, not as the solve.\n"
            "          The per-iteration table is unaffected and is the point of this mode."
        )

    # Per-iteration record.  The stage table says where time went; this says what each
    # iteration BOUGHT, which is a different question and the one a stage total cannot
    # answer -- an iteration that costs 300 s and closes the gap is not the same as one
    # that costs 300 s and does not.
    iteration_rows: list[dict] = []

    def _record(state: dict) -> None:
        sweep_s = float(state.get("sweep_s") or 0.0)
        task_s = float(state.get("sweep_task_total_s") or 0.0)
        lanes = max(1, args.workers)
        iteration_rows.append({
            "iteration": state.get("iteration"),
            "lp_objective": state.get("lp_objective"),
            "lp_gap_revenue": state.get("lp_gap_revenue"),
            "lp_gap_cost": state.get("lp_gap_cost"),
            "heuristic_gap_cost": state.get("heuristic_gap_cost"),
            "columns": state.get("columns"),
            "columns_added": state.get("columns_added"),
            "rc_n_positive": state.get("rc_n_positive"),
            "dual_nonzero": state.get("dual_nonzero"),
            "sweep_s": sweep_s,
            "sweep_task_total_s": task_s,
            "efficiency": task_s / (sweep_s * lanes) if sweep_s > 0 else None,
        })

    install_timers()
    started = time.perf_counter()
    result = ColGenSolver().solve(
        requests, cfg, static_terms, params, on_iteration=_record
    )
    wall = time.perf_counter() - started

    if iteration_rows:
        print("\n--- PER ITERATION ---")
        print(f"{'it':>3} {'lp_objective':>16} {'gap_revenue':>12} {'gap_cost':>12} "
              f"{'cols':>7} {'+add':>6} {'rc_n+':>6} {'duals':>7} {'sweep_s':>9} {'eff':>6}")
        for row in iteration_rows:
            efficiency = row["efficiency"]
            print(
                f"{row['iteration']:>3} {row['lp_objective']:>16.10g} "
                f"{row['lp_gap_revenue']:>12.4g} {row['lp_gap_cost']:>12.4g} "
                f"{row['columns']:>7} {row['columns_added']:>6} "
                f"{row['rc_n_positive']:>6} {row['dual_nonzero']:>7} "
                f"{row['sweep_s']:>9.1f} "
                f"{'n/a' if efficiency is None else format(efficiency * 100, '5.1f')}"
            )

    print(f"\nWALL {wall:.2f}s   pricing {result.stats['pricing_wall_s']:.2f}s "
          f"({100 * result.stats['pricing_wall_s'] / wall:.1f}%)   "
          f"iters={result.stats['iterations']} cols={result.stats['n_columns']} "
          f"obj={result.stats.get('objective')!r}")

    # THE FALLBACK LINE.  `kernel_fell_back` is the only number here that is authoritative in
    # BOTH modes: `_price_one` ships each task's delta, so a pooled sweep's declines are
    # summed back into the parent.  Do not read fallbacks off the stderr warnings instead --
    # `pricing._warn_budget` guards on module globals (`_kernel_restart_warned`,
    # `_kernel_budget_warned`), so it warns ONCE PER PROCESS PER KIND, and the pool is rebuilt
    # per sweep, so the warning COUNT is ~n_workers x kinds x iterations and says nothing
    # about how many flights were affected.
    #
    # `label_restarts` / `budget_declined` come from this PROCESS's `_KERNEL_STATS` and are
    # therefore parent-only: meaningful sequentially, structurally zero under a pool because
    # `SweepResult` does not carry them yet (issue #86's plumbing). Labelled as such rather
    # than printed as a flat zero that reads like good news.
    kernel_priced = result.stats.get("kernel_priced", 0)
    kernel_fell_back = result.stats.get("kernel_fell_back", 0)
    parent_stats = pricing_mod.kernel_stats()
    scope = "parent-only, N/A under a pool" if args.workers else "this process"
    print(
        f"\n--- KERNEL ---\n"
        f"priced {kernel_priced}  fell_back {kernel_fell_back} "
        f"({100.0 * kernel_fell_back / kernel_priced if kernel_priced else 0.0:.1f}%)"
        f"   [aggregated across workers]\n"
        f"label_restarts {parent_stats.get('label_restarts', 0)}  "
        f"budget_declined {parent_stats.get('budget_declined', 0)}   [{scope}]"
    )

    print("\n--- STAGE TIMERS (authoritative for magnitude) ---")
    print(f"{'stage':44s} {'calls':>9s} {'s':>9s} {'%wall':>7s}")
    for label in sorted(TOTAL_S, key=lambda k: -TOTAL_S[k]):
        print(f"{label:44s} {CALLS[label]:9d} {TOTAL_S[label]:9.2f} "
              f"{100 * TOTAL_S[label] / wall:6.1f}%")
    print("Stages nest -- _path_claims and _endpoint_claims sit inside _best_column, so the "
          "column does not sum to 100%.")

    if args.no_cprofile:
        return 0

    print("\n--- cProfile (RANKING ONLY -- magnitudes are inflated ~4x here) ---")
    TOTAL_S.clear()
    CALLS.clear()
    profiler = cProfile.Profile()
    profiler.enable()
    ColGenSolver().solve(requests, cfg, static_terms, params)
    profiler.disable()
    for sort_key in ("tottime", "cumulative"):
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats(sort_key).print_stats(args.top)
        print(f"\n[sorted by {sort_key}]")
        print(stream.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
