"""A/B timing harness: warm a realistic ledger, then time a fixed batch of A* plans.

Deterministic (seed + fixed warm/timed counts) so a BEFORE and AFTER run are directly comparable.
Usage: uv run python analysis/bench_ab.py [lam] [warm] [timed].
"""
from __future__ import annotations

import sys
import time

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
warm = int(sys.argv[2]) if len(sys.argv) > 2 else 500
timed = int(sys.argv[3]) if len(sys.argv) > 3 else 500

cfg = SimConfig(region_size_m=(60000.0, 45000.0), lam_per_hour=lam, horizon_s=900.0,
                planner="astar", seed=0)
demand = HubRadiusDemand(
    n_hubs_per_uss={"walmart_uss": 20, "stripmall_uss": 240},
    radius_m={"walmart_uss": 8000.0, "stripmall_uss": 4000.0},
    terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
    pads_per_hub=8, return_flights=True,
)
reqs = demand.generate(cfg, np.random.default_rng(cfg.seed))
ledger = ReservationLedger(cfg)
dss = DSS(ledger=ledger, mechanism=FCFSMechanism())
planner = get_planner("astar")
usses = {uid: USS(uid, dss, cfg, planner) for uid in {r.uss_id for r in reqs}}

assert warm + timed <= len(reqs), f"need {warm+timed} flights, demand made {len(reqs)}"
for req in reqs[:warm]:
    usses[req.uss_id].handle_request(req)

acc = den = 0
t0 = time.monotonic()
for req in reqs[warm:warm + timed]:
    intent = usses[req.uss_id].handle_request(req)
    acc += intent.accepted
    den += intent.status is IntentStatus.REJECTED
elapsed = time.monotonic() - t0
print(f"lam={lam} warm={warm} timed={timed} vols@start≈{ledger.n_volumes}  "
      f"-> {elapsed/timed*1000:.1f} ms/flight  ({elapsed:.1f}s for {timed}; acc={acc} den={den})",
      flush=True)
