"""A/B per-plan wall-time: astar vs compiled sipp vs pure-Python sipp_ref on a warmed ledger.

Warms a non-terminal scenario to steady-state density with A*, then times planning (no commit) of the
next M requests against that fixed ledger with each planner — isolating per-plan cost on identical
obstacles. The go/no-go for the compiled-SIPP effort (issue #8).

    uv run --extra compiled python -m analysis.bench_sipp --scenario metro_uniform --lam 2000 --warm 800 --timed 100
"""
import argparse
import time

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.demand import UniformPoissonDemand
from freespace_sim.dss import DSS
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.uss import USS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="metro_uniform",
                    help="registry name, or 'uniform' for a bare big-region non-terminal config")
    ap.add_argument("--region", type=float, default=0.0, help="square region size m (uniform mode)")
    ap.add_argument("--lam", type=float, default=2000.0)
    ap.add_argument("--horizon", type=float, default=900.0)
    ap.add_argument("--warm", type=int, default=800)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.scenario == "uniform":                       # bare big-region non-terminal (long flights)
        w = args.region or 40000.0
        cfg = SimConfig(region_size_m=(w, w), lam_per_hour=args.lam, horizon_s=args.horizon,
                        seed=args.seed, planner="astar")
        demand = UniformPoissonDemand()
    else:
        spec = with_overrides(get_scenario(args.scenario), lam_per_hour=args.lam,
                              horizon_s=args.horizon, seed=args.seed)
        cfg = spec.config()
        demand = spec.demand_model() or UniformPoissonDemand()
    reqs = demand.generate(cfg, np.random.default_rng(cfg.seed))
    sc = scenario_from_requests(reqs)
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    usses = {u: USS(u, dss, cfg, get_planner("astar")) for u in sc.uss_ids}

    warm = min(args.warm, len(sc.events) - args.timed - 5)
    for ev in sc.events[:warm]:
        usses[ev.request.uss_id].handle_request(ev.request)
    batch = [ev.request for ev in sc.events[warm:warm + args.timed]]
    print(f"{args.scenario} lam={args.lam:g}: warmed {led.n_volumes} vols ({warm} flights); "
          f"per-plan over {len(batch)} (no commit):")

    res = {}
    for name in ("astar", "sipp_ref", "sipp"):
        p = get_planner(name)
        p.plan(batch[0], led, cfg)                       # warm (JIT compile for sipp)
        t = time.monotonic()
        for rq in batch:
            p.plan(rq, led, cfg)
        res[name] = (time.monotonic() - t) / len(batch) * 1000
        print(f"  {name:9s}: {res[name]:8.1f} ms/plan")
    print(f"  SPEEDUP  astar/sipp = {res['astar'] / res['sipp']:.2f}x   "
          f"sipp_ref/sipp = {res['sipp_ref'] / res['sipp']:.2f}x")


if __name__ == "__main__":
    main()
