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

Note ``solver`` binds ``price_flight`` / ``seed_column`` / ``find_feasible_column`` at
import, so patching only ``pricing`` instruments nothing the solver actually calls. Both
module objects are patched below.

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

from freespace_sim.planner.colgen import dp_prepare as dp_prepare_mod  # noqa: E402
from freespace_sim.planner.colgen import master as master_mod  # noqa: E402
from freespace_sim.planner.colgen import pricing as pricing_mod  # noqa: E402
from freespace_sim.planner.colgen import solver as solver_mod  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.planner.colgen.solver import ColGenSolver  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402

TOTAL_S: collections.Counter = collections.Counter()
CALLS: collections.Counter = collections.Counter()

# (label, [(module, attribute), ...]) -- a stage may be bound in more than one module.
STAGES: list[tuple[str, list[tuple[object, str]]]] = [
    # `_best_column_compiled` is the entry to exact pricing since Phase 2c; `_best_column`
    # runs only when it cannot prove it completed, so a nonzero row there is a FALLBACK
    # count and worth reading as one.
    ("_best_column_compiled (Tier 1+2, entry)", [(pricing_mod, "_best_column_compiled")]),
    ("_best_column (reference FALLBACK)", [(pricing_mod, "_best_column")]),
    ("  _dag_candidates (per-sink pricing)", [(pricing_mod, "_dag_candidates")]),
    ("  _certify_candidates (Tier 2 rank)", [(pricing_mod, "_certify_candidates")]),
    ("  prepared_for (cached packing)", [(dp_prepare_mod, "prepared_for")]),
    ("  prepare_duals (per sweep)", [(dp_prepare_mod, "prepare_duals")]),
    ("  prepare_variants (roots + gate)", [(dp_prepare_mod, "prepare_variants")]),
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

    print(f"tree      {_loaded.parent.parent}")
    print(f"workload  {args.scenario} x{len(requests)} iters={args.iterations} "
          f"{args.solver} gap={args.gap_metric} (fixed work, no clock)")

    install_timers()
    started = time.perf_counter()
    result = ColGenSolver().solve(requests, cfg, static_terms, params)
    wall = time.perf_counter() - started

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
