"""CONFLICT_FILED denials at busy shared-terminal hubs — the three mechanisms surfaced by the
TerminalCapacity work, each reproduced *deterministically* from a tiny subset of the
``dallas_hub_2uss_large`` @ seed-0 demand (replaying only the few FCFS-ordered flights involved, so each
test is <1 s and needs no full ~100-flight run).

  1. **own-column vs foreign corridor** (the lazy-skip bug) — a same-hub sibling's own near-hub cruise
     corridor intruded a window its column "covered", so the unsound "already-deployed → skip the ledger"
     shortcut admitted a takeoff into it. **FIXED**: ``TerminalCapacity.column_clear`` always queries.
     Regression below: the {46,4,58,8} subset (which the pre-fix code denied) now fully admits.

  2. **cruise box vs sibling column** — the first UNTAGGED cruise box just past the tagged exit lane
     reaches back into a same-hub sibling's column. Only box[0]/box[-1] are tagged
     (``astar._build`` ~L346, ``volumes.build_reservation_from_corners`` ~L137), and A* is blind to its
     own hub's columns (``occupancy.is_blocked`` own-hub exemption ~L139), so it files a clipping plan.
     **OPEN** (xfail).

  3. **same-hub exit lanes collide** — two tagged exit boxes overlap *inside* the shared column; box↔box
     is not exempt (``conflict.volumes_conflict`` L29-31), and A* can't see the sibling's in-column
     corridor (dropped by ``occupancy.add_volume`` own_cols ~L85) so it can't ground-delay around it.
     **OPEN** (xfail).

(See ``tests/test_terminal_capacity.py::test_column_clear_always_queries_even_when_siblings_cover`` for
the unit-level guard on mechanism 1.)
"""

import numpy as np

from freespace_sim.dss import DSS
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.sim import scenario_from_requests
from freespace_sim.uss import USS

# Flights extracted from dallas_hub_2uss_large @ seed 0, pads_per_hub=4. Replaying just these in FCFS
# order reproduces each denial exactly (the demand is deterministic for a fixed seed).
LAZY_SKIP = (46, 4, 58, 8)    # walmart#4 deliveries + fid 58's hub-crossing corridor; pre-fix denied 4 & 8
CRUISE_CLIP = (4, 8, 86)      # walmart#4 deliveries; fid 86's first cruise box clips fid 4/8 columns
EXIT_COLLISION = (44, 92)     # stripmall#5 deliveries; their exit lanes collide inside the shared column


def _replay(fids):
    """Plan just ``fids`` (FCFS-ordered) from the seed-0 dallas demand; return {flight_id: intent}."""
    spec = with_overrides(get_scenario("dallas_hub_2uss_large"),
                          demand_overrides={"pads_per_hub": 4}, planner="astar",
                          lam_per_hour=600.0, horizon_s=300.0, seed=0)
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
    # Mechanism 3, FIXED. The takeoff edge is now gated by ``TerminalCapacity.exit_clear`` — a precise
    # (FCL) check of the exit lane toward the dest against committed sibling lanes (box↔box, not
    # column-exempt). fid 92's lane overlaps fid 44's, so it ground-delays past it instead of being
    # denied. (The hex grid is too coarse for this: its inflation would also serialize DIVERGENT
    # launches; see ``exit_clear`` and ``test_terminal.test_divergent_same_hub_launches_are_concurrent``.)
    intents = _replay(EXIT_COLLISION)
    assert intents[44].accepted                                  # the first lane commits
    assert intents[92].accepted                                  # ground-delays past fid 44's lane; admitted
