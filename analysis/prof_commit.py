"""Phase E (issue #8): profile the coordinator's SERIAL commit floor, so the fix targets the real
hot spot instead of a remembered one.

`run_parallel` moves planning off-thread but every *commit* still runs serially on the coordinator:
`dss.commit` = `FCFSMechanism.commit` = `ledger.any_conflict` (re-check) + `ledger.commit`
(bucket-index insert + fire the occupancy `on_commit` hooks, which each rasterize the volume). This
floor caps the achievable speedup (Amdahl); the packing shrank plan time and made it the binding
constraint. Three candidate costs, three different fixes — measure which dominates:

* **recheck**  `any_conflict` over every committed volume near the flight. In EXACT mode this is
  PROVABLY redundant (the read-envelope already certifies byte-equality with the sequential run,
  whose own filed-corridor check passed — canary=0 across every run to date), so it can be skipped.
* **bucket**   the `steps × xy-cells` cross-product insert into `_buckets`. Feeds `any_conflict` and
  the final `verify`; if the recheck is skipped it is only needed at the very end.
* **hooks**    the occupancy `on_commit` rasterizations — TWO per commit (hex + compiled), each a
  `rasterize_volume_dual` sweep — maintained all run to serve only the ~3-8% of flights that replan
  serially. The worker that planned the flight ALREADY rasterized it once, into its replica.

The split says which lever pays: recheck→skip it, hooks→reuse the worker raster, bucket→defer it.

Usage:  uv run python analysis/prof_commit.py [lam] [warm] [timed]
"""
from __future__ import annotations

import sys
import time

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.types import IntentStatus

lam = float(sys.argv[1]) if len(sys.argv) > 1 else 8000.0
warm = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
timed = int(sys.argv[3]) if len(sys.argv) > 3 else 250


def _split_commit(ledger: ReservationLedger, fid: int, vols):
    """Commit ``vols`` while timing the bucket-insert and the observer hooks separately.

    Detaches the observers so `ledger.commit` does index work only, then fires them by hand — the
    resulting ledger + occupancy state is identical to a normal commit, just measured in two parts."""
    saved = ledger._observers
    ledger._observers = []
    t0 = time.perf_counter()
    ledger.commit(fid, vols)                       # bucket-index insert, no hooks
    t_bucket = time.perf_counter() - t0
    ledger._observers = saved
    t0 = time.perf_counter()
    for cb in saved:                               # hex on_commit + tcap + compiled on_commit
        cb(fid, vols)
    t_hooks = time.perf_counter() - t0
    return t_bucket, t_hooks


def main() -> None:
    cfg = SimConfig(region_size_m=(60000.0, 45000.0), lam_per_hour=lam, horizon_s=1800.0,
                    planner="astar", seed=0)
    demand = HubRadiusDemand(
        n_hubs_per_uss={"walmart_uss": 20, "stripmall_uss": 240},
        radius_m={"walmart_uss": 8000.0, "stripmall_uss": 4000.0},
        terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
        pads_per_hub=8, return_flights=True,
    )
    reqs = demand.generate(cfg, np.random.default_rng(cfg.seed))
    reqs.sort(key=lambda r: (r.t_request, r.flight_id))
    assert warm + timed <= len(reqs), f"need {warm + timed}, demand made {len(reqs)}"

    ledger = ReservationLedger(cfg)
    planner = get_planner("astar")
    # Subscribe the SAME occupancy images the coordinator's serial replan lane holds, so the hook
    # cost measured here is exactly the coordinator's. First plan triggers subscribe+absorb.
    for r in reqs[:warm]:
        intent = planner.plan(r, ledger, cfg)
        if intent.status == IntentStatus.ACCEPTED and intent.volumes:
            ledger.commit(r.request.flight_id if hasattr(r, "request") else r.flight_id, intent.volumes)

    n0 = ledger.n_volumes
    t_recheck = t_bucket = t_hooks = t_plan = 0.0
    n_acc = 0
    for r in reqs[warm:warm + timed]:
        t0 = time.perf_counter()
        intent = planner.plan(r, ledger, cfg)
        t_plan += time.perf_counter() - t0
        if intent.status != IntentStatus.ACCEPTED or not intent.volumes:
            continue
        n_acc += 1
        vols = intent.volumes
        t0 = time.perf_counter()
        ledger.any_conflict(vols)                  # the redundant-in-exact-mode re-check
        t_recheck += time.perf_counter() - t0
        tb, th = _split_commit(ledger, r.flight_id, vols)
        t_bucket += tb
        t_hooks += th

    ms = lambda t: 1000 * t / max(1, n_acc)
    commit = t_recheck + t_bucket + t_hooks
    print(f"\nλ={lam:.0f}  warm={warm}  timed={timed}  committed_vols={n0}→{ledger.n_volumes}  "
          f"accepted={n_acc}")
    print(f"  {'stage':<10} {'ms/flight':>10} {'% of commit':>12} {'vs plan':>9}")
    for name, t in [("plan", t_plan), ("recheck", t_recheck), ("bucket", t_bucket), ("hooks", t_hooks)]:
        share = "" if name == "plan" else f"{100 * t / max(1e-9, commit):>10.0f}%"
        print(f"  {name:<10} {ms(t):>10.2f} {share:>12} {t / max(1e-9, t_plan):>8.2f}x")
    print(f"  {'COMMIT':<10} {ms(commit):>10.2f} {'100%':>12} {commit / max(1e-9, t_plan):>8.2f}x")
    print(f"\n  parallel floor = commit ≈ {ms(commit):.1f} ms/flight → ceiling ≈ "
          f"{(t_plan + commit) / max(1e-9, commit):.1f}x  (serial plan+commit vs commit-only)")
    print(f"  exact-mode skip of recheck removes {100 * t_recheck / max(1e-9, commit):.0f}% of the floor; "
          f"reusing worker rasters targets the {100 * t_hooks / max(1e-9, commit):.0f}% in hooks.\n")


if __name__ == "__main__":
    main()
