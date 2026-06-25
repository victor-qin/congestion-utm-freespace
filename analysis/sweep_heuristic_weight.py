"""Weighted-A* sweep: how heuristic_weight trades optimality for search effort / wall time.

Warms one realistic ledger (the 60x45 km, 20+240-hub Dallas world), snapshots it, then replays the
SAME timed batch of flights under each ``heuristic_weight`` w. Because every w sees an identical
ledger + request stream, the columns are directly comparable. Reports, per w:

  * ms/flight (wall) and mean expansions/flight  -> the speedup (the 90%-of-runtime A* search)
  * mean accepted path cost and its ratio vs w=1  -> the price paid (bounded by w: cost <= w*optimal)
  * acc/den                                       -> weighting must not change feasibility, only cost

w=1.0 is the optimal baseline (bit-identical to pre-weighting A*). Safety is enforced by the search
gate + exact ledger check, NOT the cost, so a w-suboptimal path is still separation-safe.

Usage: uv run python analysis/sweep_heuristic_weight.py [lam] [warm] [timed] [w1,w2,...]
       uv run python analysis/sweep_heuristic_weight.py 4000 600 400 1,1.25,1.5,2,3
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand
from freespace_sim.dss import DSS
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.types import IntentStatus
from freespace_sim.uss import USS

lam = float(sys.argv[1]) if len(sys.argv) > 1 else 4000.0
warm_n = int(sys.argv[2]) if len(sys.argv) > 2 else 600
timed = int(sys.argv[3]) if len(sys.argv) > 3 else 400
weights = [float(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["1", "1.25", "1.5", "2", "3"])]

base_cfg = SimConfig(region_size_m=(60000.0, 45000.0), lam_per_hour=lam, horizon_s=900.0,
                     planner="astar", seed=0)
demand = HubRadiusDemand(
    n_hubs_per_uss={"walmart_uss": 20, "stripmall_uss": 240},
    radius_m={"walmart_uss": 8000.0, "stripmall_uss": 4000.0},
    terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
    pads_per_hub={"walmart_uss": 24, "stripmall_uss": 8}, return_flights=True,
)
reqs = demand.generate(base_cfg, np.random.default_rng(base_cfg.seed))
assert warm_n + timed <= len(reqs), f"need {warm_n + timed} flights, demand made {len(reqs)}"


def build_warm():
    """Warm an identical ledger at w=1 (same obstacle field for every trial) and return the ledger,
    DSS and its already-subscribed planner, so the timed phase reuses them (no orphan occupancy)."""
    ledger = ReservationLedger(base_cfg)
    dss = DSS(ledger=ledger, mechanism=FCFSMechanism())
    planner = get_planner("astar")
    usses = {uid: USS(uid, dss, base_cfg, planner) for uid in {r.uss_id for r in reqs}}
    for req in reqs[:warm_n]:
        usses[req.uss_id].handle_request(req)
    return ledger, dss, planner


print(f"lam={lam} warm={warm_n} timed={timed}  region=60x45km hubs=20+240 pads=24/8", flush=True)
print(f"{'w':>5} {'ms/flight':>10} {'exp/flight':>11} {'mean_cost':>11} {'cost/opt':>9} "
      f"{'acc':>5} {'den':>5} {'speedup':>8}", flush=True)

base_ms = base_cost = None
for w in weights:
    cfg = replace(base_cfg, heuristic_weight=w)      # trial w; warm-up stays w=1 (identical field)
    ledger, dss, planner = build_warm()              # reuse the warmed planner → no orphan svc
    usses = {uid: USS(uid, dss, cfg, planner) for uid in {r.uss_id for r in reqs}}

    acc = den = exp_sum = 0
    costs: list[float] = []
    t0 = time.monotonic()
    for req in reqs[warm_n:warm_n + timed]:
        intent = usses[req.uss_id].handle_request(req)
        exp_sum += planner.last_expansions
        if intent.accepted:
            acc += 1
            costs.append(intent.cost)
        else:
            den += 1
    elapsed = time.monotonic() - t0

    ms = elapsed / timed * 1000.0
    mean_cost = float(np.mean(costs)) if costs else float("nan")
    if base_ms is None:
        base_ms, base_cost = ms, mean_cost
    print(f"{w:>5.2f} {ms:>10.1f} {exp_sum / timed:>11.0f} {mean_cost:>11.1f} "
          f"{mean_cost / base_cost:>9.3f} {acc:>5} {den:>5} {base_ms / ms:>7.2f}x", flush=True)
