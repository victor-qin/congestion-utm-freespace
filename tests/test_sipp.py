"""SIPP planner: a cost-aware safe-interval drop-in for A*.

The headline guarantee is **cost equivalence with A***: on the same committed ledger SIPP returns the
same accepted/denied outcome and the same optimal weighted cost as the A* planner — verified by
"replay" (plan every flight with both planners against the SAME, A*-committed, ledger, isolating
per-plan optimality from FCFS tie-cascade). Exact for non-terminal and legacy-terminal worlds; for the
default fixed-exit-lanes terminal path there is a rare marginal ground/hover tie (documented below),
where SIPP stays within a small tolerance, same accept set, still verified.
"""
import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.demand import UniformPoissonDemand
from freespace_sim.dss import DSS
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.planner import get_planner
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec
from freespace_sim.uss import USS
from freespace_sim.volumes import Volume4D

CFG = SimConfig()


def _req(fid=1, dx=2000.0):
    return FlightRequest(fid, vec(0, 0, 0), vec(dx, 0, 0), 0.0)


def _wall():
    return Volume4D(box_from_segment(vec(1000, -200, 150), vec(1000, 200, 150), 40, 400), 0.0, 1e6)


def _plan_both(req, committed=()):
    out = {}
    for name in ("astar", "sipp"):
        led = ReservationLedger(CFG)
        for fid, vols in committed:
            led.commit(fid, vols)
        out[name] = get_planner(name).plan(req, led, CFG)
    return out


# ---- isolated, exact ----

def test_sipp_empty_matches_astar():
    o = _plan_both(_req())
    assert o["sipp"].accepted and o["astar"].accepted
    assert not ReservationLedger(CFG).any_conflict(o["sipp"].volumes)   # self-consistent
    assert abs(o["sipp"].cost - o["astar"].cost) < 1e-6


def test_sipp_reroutes_around_wall_like_astar():
    o = _plan_both(_req(2), committed=[(99, [_wall()])])
    assert o["sipp"].accepted and o["astar"].accepted
    assert o["sipp"].air_detour_m > 0          # had to go around
    assert abs(o["sipp"].cost - o["astar"].cost) < 1e-6


def test_sipp_deterministic():
    a = get_planner("sipp").plan(_req(7), ReservationLedger(CFG), CFG)
    b = get_planner("sipp").plan(_req(7), ReservationLedger(CFG), CFG)
    assert abs(a.cost - b.cost) < 1e-12 and len(a.centerline) == len(b.centerline)


# ---- replay equivalence (the headline): both plan against the SAME A*-committed ledger ----

def _replay(scenario, lam, H, seed, demand_ov=None, fixed=None):
    ov = {} if fixed is None else {"fixed_exit_lanes": fixed}
    spec = with_overrides(get_scenario(scenario), lam_per_hour=lam, horizon_s=H, seed=seed,
                          demand_overrides=demand_ov, **ov)
    cfg = spec.config()
    reqs = (spec.demand_model() or UniformPoissonDemand()).generate(cfg, np.random.default_rng(cfg.seed))
    sc = scenario_from_requests(reqs)
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    astar = get_planner("astar")
    sipp = get_planner("sipp")
    usses = {u: USS(u, dss, cfg, astar) for u in sc.uss_ids}
    rows = []
    for ev in sc.events:
        rq = ev.request
        s = sipp.plan(rq, led, cfg)            # SIPP vs the current (A*-committed) ledger
        a = usses[rq.uss_id].handle_request(rq)  # A* plans + commits
        rows.append((a.accepted, s.accepted, a.cost if a.accepted else 0.0,
                     s.cost if s.accepted else 0.0))
    return rows


@pytest.mark.parametrize("seed", range(3))
def test_sipp_replay_exact_nonterminal(seed):
    for scenario, lam in (("metro_uniform", 120.0), ("dallas_hub_2uss", 240.0)):
        rows = _replay(scenario, lam, 600.0, seed)
        assert rows, f"{scenario} produced no flights"
        assert all(a == s for a, s, _, _ in rows), f"{scenario} accept-set mismatch"
        assert all(abs(ca - cs) < 1e-5 for a, s, ca, cs in rows if a), f"{scenario} cost mismatch"


def test_sipp_replay_exact_legacy_terminal():
    rows = _replay("dallas_hub_2uss_large", 150.0, 400.0, 0,
                   demand_ov={"pads_per_hub": 2, "radius_m": 2500.0}, fixed=False)
    assert any(c for _, _, c, _ in rows)       # terminal flights present
    assert all(a == s for a, s, _, _ in rows)
    assert all(abs(ca - cs) < 1e-5 for a, s, ca, cs in rows if a)


def test_sipp_replay_fixed_lanes_near_optimal():
    # Default fixed-exit-lanes terminal path: SIPP matches A* accept-for-accept and total cost within 1%.
    # KNOWN GAP: a rare marginal ground/hover tie at terminal takeoffs makes the fixed-lane SIPP and A*
    # optima differ by a few cost units per affected flight (both directions; ~0.03% aggregate here).
    # Not a safety issue — see test_sipp_fixed_lanes_full_run_verified. Tracked for an exact fix.
    rows = _replay("dallas_hub_2uss_large", 150.0, 400.0, 0,
                   demand_ov={"pads_per_hub": 2, "radius_m": 2500.0}, fixed=True)
    assert all(a == s for a, s, _, _ in rows)                       # identical accept set
    tot_a = sum(ca for a, s, ca, cs in rows if a)
    tot_s = sum(cs for a, s, ca, cs in rows if a)
    assert abs(tot_s - tot_a) <= tot_a * 0.01                       # within 1% of optimal, either way


def test_sipp_fixed_lanes_full_run_verified():
    # The non-negotiable property on the default terminal path: every committed set is conflict-free.
    spec = with_overrides(get_scenario("dallas_hub_2uss_large"), region_m=(8000.0, 8000.0),
                          lam_per_hour=150.0, horizon_s=400.0, seed=0,
                          demand_overrides={"pads_per_hub": 2, "radius_m": 2500.0})
    rs = run(spec.config(), demand=spec.demand_model(), planner_name="sipp")
    assert rs.verified
    assert rs.summary()["n_accepted"] > 0


# ---- full run + shortcut wrap ----

def test_sipp_full_run_verified_and_matches_astar():
    spec = with_overrides(get_scenario("dallas_hub_2uss"), lam_per_hour=240.0, horizon_s=600.0, seed=1)
    ra = run(spec.config(), demand=spec.demand_model(), planner_name="astar")
    rs = run(spec.config(), demand=spec.demand_model(), planner_name="sipp")
    assert rs.verified
    assert rs.summary()["n_accepted"] == ra.summary()["n_accepted"]


def test_sipp_shortcut_wraps_and_verifies():
    spec = with_overrides(get_scenario("dallas_hub_2uss"), lam_per_hour=240.0, horizon_s=600.0, seed=1)
    rsc = run(spec.config(), demand=spec.demand_model(), planner_name="sipp_shortcut")
    rs = run(spec.config(), demand=spec.demand_model(), planner_name="sipp")
    assert rsc.verified
    # shortcut only tightens: total accepted cost is <= the un-shortcut SIPP (within epsilon)
    csc = sum(i.cost for i in rsc.accepted)
    cs = sum(i.cost for i in rs.accepted)
    assert csc <= cs + 1e-6
