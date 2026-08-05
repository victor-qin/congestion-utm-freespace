"""Measure what the visit-window width costs the pricing DP's state space.

``derive_cell_window`` measures a W=4 footprint at defaults, and two separate
consequences follow from that single number:

* ``revisit_depth = hi - lo = 3`` -- the DP must ban re-entering any of the last
  three cells, because overlapping visit windows would double-charge rows that
  the column's claim *set* charges once.
* ``state_history_depth = max(2, revisit_depth) = 3`` -- the dominance key must
  therefore carry three recent cells, so several labels share each cell-step.

This probe separates them.  It re-runs one captured search with the two depths
overridden independently, so the label count attributable to each is measured
rather than inferred.

The overrides are DELIBERATELY UNSOUND as a solver: lowering ``revisit_depth``
admits self-overlapping columns whose kernel score is not their true reduced
cost.  Nothing here is a proposed configuration -- it exists to size a redesign
before anyone writes one.

Usage:
    uv run python analysis/probe_window_state_width.py /tmp/fail_3176.pkl
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
import time
from pathlib import Path

import freespace_sim
from freespace_sim.planner.colgen import dp_kernel, dp_prepare, pricing
from freespace_sim.planner.colgen.windows import derive_cell_window


def _load(path: Path):
    # The case file is a locally-produced capture of one in-process pricing call
    # (graph, duals, cfg), written by this repo's own straggler tooling.  It holds
    # live ``FlightGraph``/``DualView`` objects, so pickle is the only faithful
    # round-trip; it is never read from an untrusted or remote source.
    with path.open("rb") as handle:
        return pickle.load(handle)


def _prepare(case):
    fg, dual_view, cfg = case["graph"], case["duals"], case["cfg"]
    topology = dp_prepare.prepare_topology(fg, cfg)
    if topology.unsupported_reason is not None:
        raise SystemExit(f"topology unsupported: {topology.unsupported_reason}")
    duals = dp_prepare.prepare_duals(dual_view, topology)
    return fg, cfg, topology, duals


def _incumbent_cutoff(case, benefit):
    """Rebuild the cutoff ``price_flight`` would hand the kernel for this call.

    Passing ``None`` here is not a neutral simplification: the seed incumbent is
    what makes ``completion_can_compete`` bite, and without it every arm simply
    exhausts the label pool and the comparison measures nothing.  It also feeds
    ``prepare_variants``, whose ground-delay pre-filter otherwise keeps every
    departure step.
    """

    fg, view, cfg, pi_f = case["graph"], case["duals"], case["cfg"], case["pi_f"]
    seed = pricing.seed_column(fg, cfg)
    seed_rc = benefit - seed.delay_s - view.claim_cost(seed.claims) - pi_f
    incumbent = pricing._shifted_seed_incumbent(
        seed, fg, view, pi_f, cfg, benefit, frozenset(), (seed_rc, seed)
    )
    return incumbent[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, nargs="?", default=Path("/tmp/fail_3176.pkl"))
    # One fixed pool for every arm, so ``n_labels`` is a pure state count and no arm
    # pays a different number of discarded ladder rungs.  1<<25 is the shipped ceiling
    # and the captured straggler certifies just under it (32,274,881 labels).
    parser.add_argument("--label-limit", type=int, default=1 << 25)
    parser.add_argument(
        "--optimum-cutoff",
        type=float,
        default=144.0,
        help="certified optimum for the captured case; models a master-incumbent cutoff",
    )
    args = parser.parse_args()

    # The A/B harness lesson: prove which tree is imported before trusting a number.
    print(f"freespace_sim: {freespace_sim.__file__}")
    if not args.case.exists():
        raise SystemExit(f"no captured case at {args.case}")

    case = _load(args.case)
    fg, cfg, topology, duals = _prepare(case)
    benefit = case["params"].M
    pi_f = case["pi_f"]

    offsets = derive_cell_window(cfg)
    print(
        f"case={args.case.name}  cells={topology.n_cells}  "
        f"steps={topology.max_step - topology.min_step + 1}  "
        f"offsets={offsets}  W={offsets[1] - offsets[0] + 1}  "
        f"revisit_depth={topology.revisit_depth}  depth={topology.state_history_depth}"
    )

    cutoff = _incumbent_cutoff(case, benefit)

    # ``prepare_variants`` reads only duals and the clock, not the history depths,
    # so one build is shared by every arm and cannot skew the comparison.
    variants = dp_prepare.prepare_variants(
        fg, cfg, case["duals"], topology, seed=False,
        benefit=benefit, pi_f=pi_f, cost_cutoff=cutoff,
    )
    print(f"variants={variants.n_variants}  cost_cutoff={cutoff:.6f}\n")

    arms = [
        ("W=4 baseline      ", topology.revisit_depth, topology.state_history_depth),
        ("W=3 (buffer 2s)   ", 2, 2),
        ("W=2 (buffer 0s)   ", 1, 2),
        ("ban off, depth 2  ", 0, 2),
        ("ban off, depth 1  ", 0, 1),
    ]

    # The state width only matters if the search is large, and the search size is set
    # far more by the CUTOFF than by the key.  Sweep both so the two are not confused:
    # the seed cutoff is what pricing supplies today, the certified optimum is what a
    # master-incumbent cutoff (`price_flight`'s ``known_column``) approaches.
    for label, cut in (("seed cutoff", cutoff), ("optimum cutoff", args.optimum_cutoff)):
        if cut is None:
            continue
        print(f"\n=== {label}: cost_cutoff={cut:.6f} ===")
        nvar = dp_prepare.prepare_variants(
            fg, cfg, case["duals"], topology, seed=False,
            benefit=benefit, pi_f=pi_f, cost_cutoff=cut,
        )
        print(f"variants surviving the ground-delay prefilter: {nvar.n_variants}")
        header = f"{'arm':<20}{'rv':>3}{'dep':>4}{'labels':>13}{'wall':>9}{'cands':>8}  status"
        print(header)
        print("-" * len(header))
        for name, revisit, depth in arms:
            probe = dataclasses.replace(
                topology, revisit_depth=revisit, state_history_depth=depth
            )
            start = time.perf_counter()
            result = dp_kernel.search_dag(
                probe, duals, nvar,
                cfg=cfg, benefit=benefit, pi_f=pi_f, cost_cutoff=cut,
                seed=False, label_limit=args.label_limit, label_limit_max=args.label_limit,
            )
            wall = time.perf_counter() - start
            print(
                f"{name:<20}{revisit:>3}{depth:>4}{result.n_labels:>13,}"
                f"{wall:>8.2f}s{len(result.candidates):>8,}  {result.status_name}"
            )
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
