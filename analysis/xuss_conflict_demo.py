"""Deterministic CROSS-USS demonstration of the hex-raster-vs-continuous-geometry CONFLICT_FILED (issue #22).

WHAT IT SHOWS
    Two flights of DIFFERENT USS operators collide at commit: a `uss_a` flight B, departing ~150 m from a
    FOREIGN hub, has its built 60 m corridor overlap a `uss_b` flight A's committed landing approach at that
    hub. The discrete hex search (`is_blocked`) reports B's path clear; the exact continuous `any_conflict`
    check then finds the overlap and files `CONFLICT_FILED`. BOTH the compiled numba kernel and the
    pure-Python reference file it — so it is the genuine raster slack, not a compiled-kernel divergence.

WHY IT IS BUILT THIS WAY (not a hand-made 2-flight fixture)
    The slack is EMERGENT and cannot be reproduced by two flights in isolation. Each corridor is rasterized
    inflated by `corridor_width/2 + R`, which is *designed* so that a discrete-clear path stays continuously
    clear for any isolated pair — so B always routes >= a corridor-width away from A unless dense traffic
    funnels it onto A's footprint boundary. (Empirically: A + B alone, or A + a constructed funnel, always
    route clear once the incremental-occupancy eviction watermark is handled correctly — plan candidates on
    a FRESH ledger, never reuse one planner across non-monotonic request times, or eviction silently drops
    committed volumes and fabricates false conflicts.)

    So the reproduction is the deterministic seed=0 `dallas_full` run, whose flight #763 is a real slack
    collision. The occupancy / capacity layer is USS-AGNOSTIC (it keys on hub `terminal_id` and geometry,
    never on `uss_id` — see `conflict.volumes_conflict`, `occupancy.is_blocked`, `TerminalCapacity`), so
    relabeling the two colliding flights to different operators leaves the geometry BYTE-IDENTICAL and the
    collision reproduces exactly — now cross-USS.

    Run: `uv run python analysis/xuss_conflict_demo.py`  (~70 s; deterministic).
"""
import math
import time
from dataclasses import replace

import numpy as np

from freespace_sim.conflict import volumes_conflict
from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand
from freespace_sim.dss import DSS
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.types import IntentStatus
from freespace_sim.uss import USS

B_FID, A_FID = 1903, 2133          # the real dallas_full @763 colliding pair (both stripmall at seed=0)


def main() -> None:
    cfg = SimConfig(region_size_m=(60000.0, 45000.0), lam_per_hour=12000.0, horizon_s=600.0,
                    planner="astar", seed=0)
    demand = HubRadiusDemand(
        n_hubs_per_uss={"walmart_uss": 20, "stripmall_uss": 240},
        radius_m={"walmart_uss": 8000.0, "stripmall_uss": 4000.0},
        terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
        pads_per_hub=8, return_flights=True)
    reqs = demand.generate(cfg, np.random.default_rng(cfg.seed))
    # Relabel ONLY the two colliding flights to different operators. Geometry is untouched (USS-agnostic).
    reqs = [replace(r, uss_id="uss_a") if r.flight_id == B_FID else
            replace(r, uss_id="uss_b") if r.flight_id == A_FID else r for r in reqs]
    by_fid = {r.flight_id: r for r in reqs}

    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    com = AStarPlanner(compiled=True)              # the production compiled kernel
    ref = AStarPlanner(compiled=False)             # the pure-Python oracle
    usses = {u: USS(u, dss, cfg, com) for u in {r.uss_id for r in reqs}}

    # Capture which committed flight the conflict is against (the ledger keys volumes by flight_id).
    last, _orig = {}, led.any_conflict
    def any_conflict(vols):
        r = _orig(vols)
        if r:
            last.clear()
            for q in vols:
                for f, cv in zip(led._fids, led._vols):
                    if volumes_conflict(q, cv):
                        last["p"] = f
                        break
                if "p" in last:
                    break
        return r
    led.any_conflict = any_conflict

    print("dallas_full seed=0, λ=12000 — replaying to the #763 slack collision (two flights relabeled to "
          "uss_a / uss_b)…", flush=True)
    t0 = time.monotonic()
    for req in reqs:
        last.clear()
        intent = usses[req.uss_id].handle_request(req)
        if req.flight_id != B_FID:
            continue
        cr = intent.denial_reason.value if intent.status is IntentStatus.REJECTED else "ACCEPTED"
        A = by_fid.get(last.get("p"))
        rp = ref.plan(req, led, cfg)               # reference on the SAME pre-commit ledger
        rr = rp.denial_reason.value if rp.status is IntentStatus.REJECTED else "ACCEPTED"
        gap = math.hypot(*(np.asarray(req.origin[:2]) - np.asarray(A.dest[:2]))) if A else float("nan")
        print(f"\nreached in {time.monotonic() - t0:.0f}s with {led.n_volumes} committed volumes\n"
              f"  B  fid{B_FID}  uss={req.uss_id!r}  O={tuple(round(x) for x in req.origin)} "
              f"→ D={tuple(round(x) for x in req.dest)}\n"
              f"  A  fid{A_FID}  uss={A.uss_id!r}  lands at foreign hub {A.dest_terminal.id!r} "
              f"({gap:.0f} m from B's origin)\n"
              f"  compiled kernel : {cr}\n"
              f"  python reference: {rr}", flush=True)
        ok = cr == rr == "conflict_filed" and A is not None and A.uss_id != req.uss_id
        verdict = ("CROSS-USS raster-slack CONFLICT_FILED — compiled == reference (genuine #22 slack)"
                   if ok else "did not reproduce")
        print(f"\n  VERDICT: {verdict}", flush=True)
        return
    print("B flight not found — demand indices changed?", flush=True)


if __name__ == "__main__":
    main()
