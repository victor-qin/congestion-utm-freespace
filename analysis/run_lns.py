"""Offline MAPF-LNS runner: FCFS A* baseline on a (truncated) scenario, then anytime LNS.

Truncation follows the spec contract (spec.py): BOTH --horizon and --demand-duration must be
set together — shrinking the horizon alone leaves the departure lead past it and the run
degenerates. E.g. a ~300-leg FAA cut:

    uv run python analysis/run_lns.py --scenario density_faa_wing_zipline \
        --demand-duration 120 --horizon 1500 --iterations 300 --out /tmp/lns_faa120.json

Writes a JSON with the baseline stats, the LNS config, the per-iteration trajectory (the
anytime curve) and the summary. Iterations are the reproducible budget; --time-limit is a
cluster safety net only.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings

import numpy as np

from freespace_sim import sim
from freespace_sim.planner.lns import LNSConfig
from freespace_sim.planner.lns.solver import run_lns_on_result
from freespace_sim.scenarios import get_scenario
from freespace_sim.scenarios.spec import with_overrides


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--demand-duration", type=float, default=None, help="seconds of demand (default: scenario's)")
    ap.add_argument("--horizon", type=float, default=None, help="sim horizon seconds (set WITH --demand-duration)")
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--neighborhood", type=int, default=8)
    ap.add_argument("--operators", default="agent,map,random")
    ap.add_argument("--gamma", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0, help="LNS seed (scenario seed comes from the spec)")
    ap.add_argument("--return-anchor", default="nominal", choices=["nominal", "realized"])
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--verify-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-incremental", action="store_true",
                    help="use the reset+reabsorb occupancy path instead of O(victims) removal (parity A/Bs)")
    ap.add_argument("--unimpeded-workers", type=int, default=None,
                    help="processes for the unimpeded delay ruler (default min(8, cpu-2); 1 = in-process)")
    ap.add_argument("--repair-order", default="premium", choices=["premium", "random"],
                    help="PP priority order: premium = most-delayed first (default), random = paper's")
    ap.add_argument("--out", default=None, help="JSON output path")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")  # surface lns progress

    spec = get_scenario(args.scenario)
    overrides = {}
    if args.demand_duration is not None:
        overrides["demand_duration_s"] = args.demand_duration
    if args.horizon is not None:
        overrides["horizon_s"] = args.horizon
    if overrides:
        if ("demand_duration_s" in overrides) != ("horizon_s" in overrides):
            ap.error("--demand-duration and --horizon must be set together (spec.py contract)")
        spec = with_overrides(spec, **overrides)
    cfg = spec.config()
    demand = spec.demand_model()

    t0 = time.time()
    res = sim.run(cfg, demand=demand, planner_name="astar", progress=False,
                  return_anchor=args.return_anchor)
    baseline_wall = time.time() - t0
    acc = res.accepted
    gd = np.array([i.ground_delay_s for i in acc]) if acc else np.zeros(1)
    baseline = {
        "n_legs": len(res.intents),
        "n_accepted": len(acc),
        "n_denied": len(res.intents) - len(acc),
        "cost": float(sum(i.cost for i in acc)),
        "ground_delay_mean_s": float(gd.mean()),
        "held_share": float((gd > 0).mean()),
        "verified": bool(res.verified),
        "wall_s": baseline_wall,
    }
    print(f"baseline: {baseline['n_legs']} legs, {baseline['n_accepted']} accepted "
          f"({baseline['n_denied']} denied), cost {baseline['cost']:.0f}, "
          f"mean hold {baseline['ground_delay_mean_s']:.1f}s ({baseline['held_share']:.0%} held), "
          f"verified={baseline['verified']}, {baseline_wall:.0f}s", flush=True)

    lns_cfg = LNSConfig(
        seed=args.seed,
        max_iterations=args.iterations,
        neighborhood_size=args.neighborhood,
        operators=tuple(s.strip() for s in args.operators.split(",") if s.strip()),
        gamma=args.gamma,
        incremental_release=not args.no_incremental,
        unimpeded_workers=args.unimpeded_workers,
        repair_order=args.repair_order,
        time_limit_s=args.time_limit,
        verify_every=args.verify_every,
        log_every=args.log_every,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="astar compiled kernel FALLBACK")
        out = run_lns_on_result(res, demand, lns_cfg, return_anchor=args.return_anchor)

    s = out.summary()
    print(f"lns: cost {s['cost_before']:.0f} -> {s['cost_after']:.0f} "
          f"(-{s['improvement']:.0f}, {s['improvement_pct']:.2f}%), "
          f"{s['n_accepted']}/{s['n_iterations']} accepted, verified={s['verified']}, "
          f"wall {s['wall_s']:.0f}s (init {s['init_wall_s']:.0f}s)", flush=True)
    print(f"weights: { {k: round(v, 3) for k, v in s['weights'].items()} }")

    # anytime readout: incumbent cost at fractions of the iteration budget
    if out.trajectory:
        marks = sorted({0, len(out.trajectory) // 4, len(out.trajectory) // 2,
                        3 * len(out.trajectory) // 4, len(out.trajectory) - 1})
        print("anytime curve (iter -> incumbent cost, wall):")
        for m in marks:
            row = out.trajectory[m]
            print(f"  {row['iter']:>6} -> {row['incumbent_cost']:.0f}  ({row['wall_s']:.0f}s)")

    if args.out:
        payload = {
            "scenario": args.scenario,
            "overrides": overrides,
            "return_anchor": args.return_anchor,
            "baseline": baseline,
            "lns_config": {k: (sorted(v) if isinstance(v, frozenset) else list(v) if isinstance(v, tuple) else v)
                           for k, v in vars(lns_cfg).items()},
            "summary": s,
            "trajectory": out.trajectory,
        }
        with open(args.out, "w") as fh:
            json.dump(payload, fh)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
