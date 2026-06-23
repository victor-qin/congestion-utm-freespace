"""A* microbenchmark — windowed ms/flight on the real 60x45 geometry.

Prints per-window mean ms/flight + cumulative accepted/denied + denial-reason mix, flushing each
line, so we can watch how per-flight cost evolves as the ledger fills (the real driver of full-run
wall time). Useful for the calibration sweep. Usage: uv run python analysis/bench_astar.py [lam] [horizon] [window].
"""
from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand
from freespace_sim.dss import DSS
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.types import IntentStatus
from freespace_sim.uss import USS

lam = float(sys.argv[1]) if len(sys.argv) > 1 else 2000.0
horizon = float(sys.argv[2]) if len(sys.argv) > 2 else 900.0
window = int(sys.argv[3]) if len(sys.argv) > 3 else 100

cfg = SimConfig(region_size_m=(60000.0, 45000.0), lam_per_hour=lam, horizon_s=horizon,
                planner="astar", seed=0)
demand = HubRadiusDemand(
    n_hubs_per_uss={"walmart_uss": 20, "stripmall_uss": 240},
    radius_m={"walmart_uss": 8000.0, "stripmall_uss": 4000.0},
    terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
    pads_per_hub=8, return_flights=True,
)
reqs = demand.generate(cfg, np.random.default_rng(cfg.seed))
print(f"region=60x45km lam={lam}/h horizon={horizon}s -> {len(reqs)} flights "
      f"(walmart hubs=20, stripmall hubs=240, pads=8)", flush=True)

ledger = ReservationLedger(cfg)
dss = DSS(ledger=ledger, mechanism=FCFSMechanism())
planner = get_planner("astar")
usses = {uid: USS(uid, dss, cfg, planner) for uid in {r.uss_id for r in reqs}}

acc = den = 0
reasons: Counter = Counter()
t_win = time.monotonic()
t0 = t_win
for i, req in enumerate(reqs, 1):
    intent = usses[req.uss_id].handle_request(req)
    if intent.accepted:
        acc += 1
    elif intent.status is IntentStatus.REJECTED:
        den += 1
        reasons[intent.denial_reason.value] += 1
    if i % window == 0:
        now = time.monotonic()
        ms = (now - t_win) / window * 1000.0
        print(f"[{i:>5}/{len(reqs)}] {ms:7.1f} ms/flight (window)  acc={acc} den={den}  "
              f"reasons={dict(reasons)}  vols={ledger.n_volumes}", flush=True)
        t_win = now

total = time.monotonic() - t0
print(f"DONE {len(reqs)} flights in {total:.1f}s = {total/len(reqs)*1000:.1f} ms/flight mean  "
      f"acc={acc} den={den} ({den/max(1,acc+den)*100:.1f}%)  reasons={dict(reasons)}", flush=True)
# extrapolate to the full run (34,500 deliveries + 34,500 returns = 69,000 ops)
print(f"EXTRAPOLATION: at {total/len(reqs)*1000:.0f} ms/flight, 69,000 ops -> "
      f"{69000*total/len(reqs)/3600:.1f} h single-threaded", flush=True)
