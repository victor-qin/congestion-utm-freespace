"""Build a realistic ledger, then cProfile a batch of A* plans to find the hot spots.
Usage: uv run python analysis/prof_astar.py."""
from __future__ import annotations

import cProfile
import pstats

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand
from freespace_sim.dss import DSS
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.uss import USS

cfg = SimConfig(region_size_m=(60000.0, 45000.0), lam_per_hour=8000.0, horizon_s=900.0,
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

# warm the ledger to a steady-state density (first ~1200 flights), untimed
WARM = min(1200, len(reqs) - 300)
for req in reqs[:WARM]:
    usses[req.uss_id].handle_request(req)
print(f"warmed ledger to {ledger.n_volumes} volumes ({WARM} flights); profiling next 300 plans")

batch = reqs[WARM:WARM + 300]
pr = cProfile.Profile()
pr.enable()
for req in batch:
    usses[req.uss_id].handle_request(req)
pr.disable()
st = pstats.Stats(pr).sort_stats("cumulative")
st.print_stats(25)
print("=== by tottime ===")
pstats.Stats(pr).sort_stats("tottime").print_stats(20)
