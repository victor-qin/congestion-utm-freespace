"""FCFS A* against a column-generation schedule, on the IDENTICAL flight set.

`colgen_iteration_study.py` reports `cost_upper_bound` -- the LP relaxation's schedule cost
-- and its incumbent, but nothing says what the *shipped alternative planner* achieves on
the same flights, so "the LP is at 240,000" has had no scale attached to it.  This runs
``sim.run`` with the A* planner over the same prefix of the same demand, with the same
config and static terminals, and reports total delay in the same seconds.

WHY THE REQUEST LIST IS BUILT HERE RATHER THAN VIA ``--demand-duration``.  The study takes
``sorted(demand.generate(...), key=flight_id)[:N]``.  Truncating by duration instead would
change the traffic MIX, not just the count, so the two runs would be scheduling different
problems and any difference would be uninterpretable.  ``sim.run(requests=...)`` accepts an
explicit list, so the two arms see byte-identical demand.

WHY BOTH ARMS GO THROUGH ``sim.run``.  Colgen's own ``cost_upper_bound`` is
``sum(Column.delay_s) + M * denied`` -- its internal ruler -- and reading it against A*'s
``metrics.total_delay_s`` is how you get the nonsense of A* apparently beating a *valid
lower bound*.  Running colgen through ``sim.run`` translates its columns to intents, so the
identical metric is applied to both and the comparison means something.

THE COMPONENT SPLIT is the other half.  ``total_delay_s`` is
``ground_delay_s + air_hold_s + (lattice_s + traffic_s) + climb_s``, and only ``lattice_s``
is not congestion: it is hex quantization, the cost of flying a lattice at all
(``metrics._detour_seconds`` splits it out exactly).

THE TWO RULERS DIFFER BY A FIXED LINEAR MAP, not by what they measure.  Both planners price
the air term against the STRAIGHT LINE -- ``pricing._path_delay_s`` and
``translate.column_to_intent`` both call ``volumes.enroute_reference_m``, so colgen charges
the full ``air_detour_m`` including hex quantization, exactly as ``total_delay_s`` does.  What
differs is the WEIGHTS: under the shipped ``objective="total_cost"``,

    Column.delay_s == cost_ground_delay_per_s * ground_delay_s
                    + cost_air_lateral_per_s  * (total_delay_s - ground_delay_s)
                 ==  1.0 * ground + 3.0 * (hold + lattice + deconfliction + climb)

verified to 2.8e-14 per column, and pinned by ``tests/test_colgen_performance.py``'s
``_weighted_cost``.  So reporting the split is what lets a reader convert between the two
rulers exactly rather than guessing -- e.g. A* at 2,000 flights reads 138,853 s of
``total_delay_s`` and 277,200 in colgen currency.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
_loaded = Path(freespace_sim.__file__).resolve()
if REPO_ROOT not in _loaded.parents:
    raise SystemExit(f"loaded the wrong tree: {_loaded} is not under {REPO_ROOT}")

from freespace_sim import sim  # noqa: E402
from freespace_sim.metrics import _detour_seconds, nominal_altitude_change_m, total_delay_s  # noqa: E402
from freespace_sim.planner.colgen.params import ColGenParams  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402
from freespace_sim.types import IntentStatus  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flights", type=int, default=2000)
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--planner", default="astar")
    ap.add_argument("--colgen-iterations", type=int, default=6)
    ap.add_argument("--colgen-workers", type=int, default=16)
    ap.add_argument("--colgen-solver", default="gurobi")
    ap.add_argument("--colgen-ladder", type=int, default=None)
    ap.add_argument("--colgen-ladder-stride", type=int, default=None)
    ap.add_argument("--colgen-lns", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]

    print(json.dumps({
        "scenario": args.scenario, "planner": args.planner, "flights": len(requests),
    }, indent=2), flush=True)

    planner_params = None
    if args.planner == "colgen":
        planner_params = ColGenParams(
            solver=args.colgen_solver,
            max_iterations=args.colgen_iterations,
            n_pricing_workers=args.colgen_workers,
            lns_destroy_flights=args.colgen_lns,
            **({} if args.colgen_ladder is None else {"seed_ladder_steps": args.colgen_ladder}),
            **({} if args.colgen_ladder_stride is None
               else {"seed_ladder_stride": args.colgen_ladder_stride}),
        )

    started = time.perf_counter()
    result = sim.run(cfg, requests=requests, demand=demand, planner_name=args.planner,
                     planner_params=planner_params, progress=True)
    wall_s = time.perf_counter() - started

    accepted = [i for i in result.intents if i.status is IntentStatus.ACCEPTED]
    denied = [i for i in result.intents if i.status is not IntentStatus.ACCEPTED]
    delays = [total_delay_s(intent, cfg) for intent in accepted]
    finite = [d for d in delays if math.isfinite(d)]

    # The four levers `total_delay_s` sums, reported separately so the number can be
    # translated into any other ruler rather than taken on faith.
    ground = air_hold = lattice = traffic = climb = 0.0
    for intent in accepted:
        lattice_s, traffic_s = _detour_seconds(intent, cfg)
        ground += intent.ground_delay_s
        air_hold += intent.air_hold_s
        lattice += lattice_s
        traffic += traffic_s
        climb += max(0.0, intent.altitude_change_m - nominal_altitude_change_m(cfg)) \
            / cfg.climb_rate_mps

    summary = {
        "wall_s": round(wall_s, 1),
        "n_requests": len(requests),
        "n_accepted": len(accepted),
        "n_denied": len(denied),
        "total_delay_s": round(math.fsum(finite), 1),
        "mean_delay_s": round(math.fsum(finite) / max(1, len(finite)), 2),
        "max_delay_s": round(max(finite, default=0.0), 1),
        "n_delay_nan": len(delays) - len(finite),
        # ground + hold + traffic + climb is congestion; lattice is hex quantization, which
        # colgen does not charge because it prices against the lattice geodesic.
        "ground_delay_s": round(ground, 1),
        "air_hold_s": round(air_hold, 1),
        "lattice_overhead_s": round(lattice, 1),
        "deconfliction_detour_s": round(traffic, 1),
        "climb_s": round(climb, 1),
        "congestion_only_s": round(ground + air_hold + traffic + climb, 1),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
