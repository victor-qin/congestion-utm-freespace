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
under ``parallel=`` everything inside the sweep happens in workers and is invisible here --
the parent would show only the greedy, the LP and canonicalization. A sequential sweep is
exactly one worker's workload, which is the thing worth ranking anyway.

Examples:

    uv run python analysis/prof_colgen_stages.py --flights 12 --iterations 3
    uv run python analysis/prof_colgen_stages.py --scenario density_faa_wing_zipline \
        --flights 20 --iterations 2 --no-cprofile
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
from freespace_sim.planner.colgen.pricing_pool import ParallelPricingConfig  # noqa: E402
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
    # when `parallel=` is set (and then only in the parent, since cProfile and these
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
    ("master.add_violated_rows", master_mod.RestrictedMaster, "add_violated_rows"),
]


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
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

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
    )

    parallel = (
        ParallelPricingConfig(n_workers=args.workers) if args.workers else None
    )

    print(f"tree      {_loaded.parent.parent}")
    print(f"workload  {args.scenario} x{len(requests)} iters={args.iterations} "
          f"{args.solver} gap={args.gap_metric} workers={args.workers} "
          f"(fixed work, no clock)")
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
        requests, cfg, static_terms, params, on_iteration=_record, parallel=parallel
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
