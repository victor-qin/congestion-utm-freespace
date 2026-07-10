"""CONFLICT_FILED denials at busy shared-terminal hubs — the three mechanisms surfaced by the
TerminalCapacity work, each reproduced *deterministically* from a tiny subset of the
``dallas_hub_2uss_large`` @ seed-0 demand (replaying only the few FCFS-ordered flights involved, so each
test is <1 s and needs no full ~100-flight run).

  1. **own-column vs foreign corridor** (the lazy-skip bug) — a same-hub sibling's own near-hub cruise
     corridor intruded a window its column "covered", so the unsound "already-deployed → skip the ledger"
     shortcut admitted a takeoff into it. **FIXED**: ``TerminalCapacity.column_clear`` always queries.
     Regression below: the {46,4,58,8} subset (which the pre-fix code denied) now fully admits.

  2. **cruise box vs sibling column** — a near-hub cruise box just past the exit lane reaches back into
     a same-hub sibling's column. **FIXED**: every box overlapping the flight's own column is now tagged
     (``volumes.segment_overlaps_column``, applied in ``astar._build`` + ``build_reservation_from_corners``),
     not just box[0]/box[-1], so the near-hub corridor is column-exempt instead of CONFLICT_FILED.

  3. **same-hub exit lanes collide** — two exit boxes overlap at the shared-column edge; box↔box is not
     exempt (``conflict.volumes_conflict`` L29-31). **FIXED**: the takeoff/landing edge is gated by
     ``TerminalCapacity.exit_clear`` — a precise FCL check of the exit lane vs committed sibling lanes —
     so the colliding flight ground-delays (the hex grid can't do this: its inflation would also
     serialize DIVERGENT launches).

All three are now FIXED; the tests below are **regressions** (they assert ``.accepted`` and pass). See
``tests/test_terminal_capacity.py::test_column_clear_always_queries_even_when_siblings_cover`` for the
unit-level guard on mechanism 1.
"""

import numpy as np
import pytest

from freespace_sim.dss import DSS
from freespace_sim.geometry import CylinderSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.sim import run, scenario_from_requests
from freespace_sim.uss import USS

# Flights extracted from dallas_hub_2uss_large @ seed 0, pads_per_hub=4. Replaying just these in FCFS
# order reproduces each denial exactly (the demand is deterministic for a fixed seed).
LAZY_SKIP = (46, 4, 58, 8)    # walmart#4 deliveries + fid 58's hub-crossing corridor; pre-fix denied 4 & 8
CRUISE_CLIP = (4, 8, 86)      # walmart#4 deliveries; fid 86's first cruise box clips fid 4/8 columns
EXIT_COLLISION = (44, 92)     # stripmall#5 deliveries; their exit lanes collide inside the shared column


def _replay(fids, fixed=False):
    """Plan just ``fids`` (FCFS-ordered) from the seed-0 dallas demand; return {flight_id: intent}."""
    # Pin the baseline the fixture flight-ids were extracted from: no always-active walls (else the
    # foreign-column filter drops fixture flights) and the default hover-footprint terminal radius (else
    # #27 reject-sampling places hubs differently). This is a fixed_exit_lanes regression test, not a taa one.
    spec = with_overrides(get_scenario("dallas_hub_2uss_large"),
                          demand_overrides={"pads_per_hub": 4, "terminal_radius_m": None}, planner="astar",
                          lam_per_hour=600.0, horizon_s=300.0, seed=0, fixed_exit_lanes=fixed,
                          terminal_airspace_always_active=False)
    cfg = spec.config()
    byid = {r.flight_id: r for r in spec.demand_model().generate(cfg, np.random.default_rng(cfg.seed))}
    sub = sorted((byid[i] for i in fids), key=lambda r: (r.t_request, r.flight_id))
    scen = scenario_from_requests(sub)
    dss = DSS(ledger=ReservationLedger(cfg), mechanism=FCFSMechanism())
    usses = {u: USS(u, dss, cfg, get_planner("astar")) for u in scen.uss_ids}
    default = next(iter(usses.values()))
    return {ev.request.flight_id: usses.get(ev.request.uss_id, default).handle_request(ev.request)
            for ev in scen.events}


def test_lazy_skip_column_admit_is_fixed():
    # Mechanism 1, REGRESSION (must pass). The pre-fix code denied fid 4 & 8 here (the lazy skip admitted
    # fid 8's column over fid 58's corridor, then commit rejected it). column_clear now always queries the
    # ledger, so fid 8 ground-delays past fid 58 and the whole subset is admitted, conflict-free.
    intents = _replay(LAZY_SKIP)
    assert all(intents[f].accepted for f in LAZY_SKIP), \
        {f: intents[f].denial_reason.value for f in LAZY_SKIP if not intents[f].accepted}


def test_cruise_box_does_not_clip_a_sibling_column():
    # Mechanism 2, FIXED. Every box reaching into its hub's own column is now tagged
    # (``volumes.segment_overlaps_column``, applied in ``astar._build`` + ``build_reservation_from_corners``),
    # not just box[0]/box[-1] — so fid 86's first cruise box is column-exempt instead of CONFLICT_FILED.
    intents = _replay(CRUISE_CLIP)
    assert intents[4].accepted and intents[8].accepted          # the two sibling columns commit
    assert intents[86].accepted                                  # column-exempt cruise box; admitted


def test_same_hub_exit_lanes_do_not_collide():
    # Mechanism 3, LEGACY path (fixed_exit_lanes off — _replay's default). The takeoff edge is gated by
    # ``TerminalCapacity.exit_clear`` — a precise (FCL) check of the exit lane toward the dest against
    # committed sibling lanes (box↔box, not column-exempt). fid 92's lane overlaps fid 44's, so it
    # ground-delays past it instead of being denied. (The default fixed-lane path instead serialises this
    # via exact cell occupancy in ``occupancy.is_blocked`` — see
    # ``test_fixed_exit_lanes_admit_all_three_mechanisms``.)
    intents = _replay(EXIT_COLLISION)
    assert intents[44].accepted                                  # the first lane commits
    assert intents[92].accepted                                  # ground-delays past fid 44's lane; admitted


@pytest.mark.parametrize("fids", [LAZY_SKIP, CRUISE_CLIP, EXIT_COLLISION])
def test_fixed_exit_lanes_admit_all_three_mechanisms(fids):
    # fixed_exit_lanes (issue #18): the structural fix, now the default. Each mechanism subset is admitted
    # conflict-free under fixed boundary-hex lanes — same-hub exit-lane contention serialises on exact
    # CELL occupancy (``occupancy.is_blocked`` sees a committed sibling exit corridor in the column
    # footprint) instead of filing a CONFLICT_FILED. All-accepted ⇒ all committed clean (the FCFS
    # mechanism re-checks the ledger at commit), so the run is conflict-free by construction.
    intents = _replay(fids, fixed=True)
    assert all(intents[f].accepted for f in fids), \
        {f: intents[f].denial_reason.value for f in fids if not intents[f].accepted}


def _max_concurrent(intervals):
    """Max number of half-open [t0, t1) windows overlapping at any instant."""
    evts = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
    cur = mx = 0
    for _, d in evts:
        cur += d
        mx = max(mx, cur)
    return mx


@pytest.mark.slow
@pytest.mark.parametrize("seed,pads", [(s, 2) for s in range(8)] + [(4, 4)])
def test_landing_capacity_accurate_to_arrival_and_not_oversubscribed(seed, pads):
    # Issue #15 regression (the landing-side capacity gate, across seeds, with returns). Two properties:
    #  (1) ACCURATE TO ARRIVAL — every return's committed LANDING column opens EXACTLY at the flown arrival
    #      (dest hover t_start == centerline[-1] time). The committed arrival is the tail-folded column
    #      edge, and the gate counts capacity there (``astar._committed_arrival``), so the window it counts
    #      IS the window the drone occupies the hub — not the goal-hex step time ``st[3]*dt``, ~2-7 s later.
    #  (2) NO OVER-SUBSCRIPTION — peak concurrent same-hub dwells (delivery-takeoffs + return-landings)
    #      never exceeds the pad count. Pre-fix, gating at the goal-hex step over-subscribed pads=2 on 7/8
    #      seeds (capacity-2 hubs reaching 3 concurrent dwells); capacity has no commit-time backstop
    #      (same-hub columns are conflict-exempt in volumes_conflict / verify), so ONLY this gate prevents
    #      it. (`slow`: one full ~100-flight hub run per case.)
    # Same pin as _replay: this is an issue-#15 landing-capacity regression, not a taa one — keep
    # dallas_hub_2uss_large's always-active walls + wide columns out of the pad-capacity signal.
    spec = with_overrides(get_scenario("dallas_hub_2uss_large"),
                          demand_overrides={"pads_per_hub": pads, "terminal_radius_m": None},
                          planner="astar", lam_per_hour=600.0, horizon_s=300.0, seed=seed,
                          terminal_airspace_always_active=False)
    res = run(spec.config(), demand=spec.demand_model())
    assert res.verified
    dwells: dict = {}
    for it in res.accepted:
        for v in it.volumes or []:
            if v.terminal_id is not None and isinstance(v.shape, CylinderSpec):
                dwells.setdefault(v.terminal_id, []).append((v.t_start, v.t_end))
        if it.request.dest_terminal is not None:                   # a return → lands at the hub
            # the committed LANDING column (_build emits it last) opens EXACTLY at the flown arrival —
            # so the capacity window the gate counted is the window the drone is actually in the hub
            landing_col = it.volumes[-1]
            assert landing_col.t_start == it.centerline[-1][1], (
                f"fid {it.request.flight_id}: landing column opens at {landing_col.t_start}, "
                f"flown arrival {it.centerline[-1][1]}")
    over = {h: _max_concurrent(iv) for h, iv in dwells.items() if _max_concurrent(iv) > pads}
    assert not over, f"pads over-subscribed (capacity {pads}): {over}"
