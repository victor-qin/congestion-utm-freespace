"""Deterministic digest of one `colgen_test` solve, for before/after comparison.

Issue #78 collapsed the pricing search's two bounds into one knob. At the shipped default the
corridor radius and `max_air_hops` are unchanged by construction, so the change predicts a
BYTE-IDENTICAL schedule -- any difference is a bug, not an expected consequence. This emits
the JSON to prove or disprove that.

Runs the harness the `ColGenParams` measurement tables were taken on: the first N flights of
`colgen_test` through `ColGenSolver().solve` directly, HiGHS, `objective=total_delay`,
`gap_metric=cost`, `n_heuristic_tries=16`, and a clock long enough that the iteration cap is
what stops it.

Usage -- run once per tree and diff:

    uv run python analysis/colgen_bounds_digest.py --flights 50 --out after.json
    git worktree add /tmp/base origin/main
    cp analysis/colgen_bounds_digest.py /tmp/base/analysis/
    (cd /tmp/base && uv run python analysis/colgen_bounds_digest.py --flights 50 --out before.json)
    diff <(jq -S 'del(.wall_s)' before.json) <(jq -S 'del(.wall_s)' after.json)

Smoke gate (~1-2 min) before spending ~25 min on the pair:

    uv run python analysis/colgen_bounds_digest.py --flights 50 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


def _assert_tree() -> str:
    """Fail loudly if this loaded a different checkout than the one it sits in.

    A script run from outside the repo puts its own directory on ``sys.path[0]``, so a copy
    living in /tmp silently imports the workspace package and the A/B measures one tree
    against itself.  This is the guard that makes the comparison mean something.
    """

    import freespace_sim

    loaded = Path(freespace_sim.__file__).resolve().parent.parent
    expected = Path(__file__).resolve().parent.parent
    if loaded != expected:
        raise SystemExit(
            f"tree mismatch: this script lives under {expected} but imported freespace_sim "
            f"from {loaded}. Run it with that tree's own interpreter."
        )
    return str(loaded)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flights", type=int, default=50,
                   help="chronological prefix of colgen_test's demand (default 50)")
    p.add_argument("--scenario", default="colgen_test")
    p.add_argument("--out", type=Path, default=None, help="write JSON here as well as stdout")
    p.add_argument("--smoke", action="store_true",
                   help="3 iterations and a 300 s clock: a structural gate, not a measurement")
    p.add_argument("--label-counts", action="store_true",
                   help="instrument pricing._prefer to count labels considered (slows the run)")
    return p


def main() -> None:
    args = _parser().parse_args()
    tree = _assert_tree()

    from freespace_sim.planner.colgen import pricing
    from freespace_sim.planner.colgen.params import ColGenParams
    from freespace_sim.planner.colgen.solver import ColGenSolver
    from freespace_sim.scenarios import get_scenario

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    requests = spec.demand_model().generate(cfg, np.random.default_rng(cfg.seed))
    requests = sorted(requests, key=lambda r: (r.t_departure, r.flight_id))[: args.flights]

    params = ColGenParams(
        solver="highs",
        objective="total_delay",
        gap_metric="cost",
        n_heuristic_tries=16,
        max_iterations=3 if args.smoke else 30,
        time_limit_s=300.0 if args.smoke else 86_400.0,
    )

    labels = 0
    if args.label_counts:
        # No label counter is exported (`arc_expanded_nodes` is the ARC figure), and `_prefer`
        # is the one funnel every candidate label passes through.
        original = pricing._prefer

        def counting_prefer(*a, **kw):
            nonlocal labels
            labels += 1
            return original(*a, **kw)

        pricing._prefer = counting_prefer

    started = perf_counter()
    result = ColGenSolver().solve(requests, cfg, (), params)
    wall = perf_counter() - started

    stats = result.stats
    columns = sorted(
        (
            c.flight_id,
            c.departure_step,
            c.origin_lane_idx,
            c.dest_lane_idx,
            list(map(list, c.cell_path)),
        )
        for c in result.columns.values()
    )
    digest = {
        "tree": tree,
        "scenario": args.scenario,
        "flights": len(requests),
        "smoke": args.smoke,
        "objective": stats["objective"],
        "objective_name": stats["objective_name"],
        "iterations": stats["iterations"],
        "termination_reason": stats["termination_reason"],
        "selected_flights": stats["selected_flights"],
        "denied_flight_ids": sorted(stats["denied_flight_ids"]),
        "arc_expanded_nodes": stats["arc_expanded_nodes"],
        "columns": columns,
        "labels_considered": labels if args.label_counts else None,
        "wall_s": round(wall, 1),
    }

    text = json.dumps(digest, indent=2, sort_keys=True, default=str)
    if args.out is not None:
        args.out.write_text(text)
    # Everything except the schedule itself, so a terminal stays readable.
    summary = {k: v for k, v in digest.items() if k != "columns"}
    summary["n_columns"] = len(columns)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str), file=sys.stderr)
    if args.out is None:
        print(text)


if __name__ == "__main__":
    main()
