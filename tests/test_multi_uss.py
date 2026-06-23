"""Multi-USS: two operators planning through one shared FCFS ledger.

The simulator builds one independent planner per USS label (sim.run) sharing a single
ReservationLedger, but the default demand only ever emits one label. These tests exercise the
two-USS path end to end and confirm cross-USS strategic deconfliction holds — including the subtle
A* case where each USS planner keeps its *own* incremental hex-occupancy service, kept in sync only
by the ledger's commit-broadcast hook.
"""

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.demand import UniformPoissonDemand
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec


def _two_colliding_cross_uss():
    # identical O/D + filing time, but two different operators → second must yield to the first
    return [
        FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0, uss_id="walmart"),
        FlightRequest(1, vec(0, 0, 0), vec(2400, 0, 0), 0.0, uss_id="stripmall"),
    ]


def test_uniform_demand_emits_both_uss_labels():
    cfg = SimConfig(lam_per_hour=300.0, horizon_s=3600.0, seed=2)
    reqs = UniformPoissonDemand(uss_ids=("walmart", "stripmall")).generate(
        cfg, np.random.default_rng(cfg.seed)
    )
    labels = {r.uss_id for r in reqs}
    assert labels == {"walmart", "stripmall"}


def test_scenario_derives_uss_set():
    reqs = _two_colliding_cross_uss()
    scenario = scenario_from_requests(reqs)
    assert scenario.uss_ids == ["stripmall", "walmart"]   # sorted unique set


def test_two_uss_cross_conflict_yield_straight():
    cfg = SimConfig(planner="straight")
    res = run(cfg, requests=_two_colliding_cross_uss())
    assert res.verified                         # core ASTM invariant: no inter-flight conflict
    assert len(res.accepted) == 2
    delays = sorted(i.ground_delay_s for i in res.accepted)
    assert delays[0] == 0.0 and delays[1] > 0.0   # cross-USS FCFS: second operator waits


def test_two_uss_cross_conflict_astar():
    # the key test: two AStarPlanner instances, two occupancy services, one shared ledger.
    cfg = SimConfig(planner="astar")
    res = run(cfg, requests=_two_colliding_cross_uss())
    assert res.verified                         # plans do not overlap in 4D
    assert len(res.accepted) == 2
    by_id = {i.request.flight_id: i for i in res.accepted}
    # flight 1 is the FCFS newcomer; it must have paid *something* to deconflict from flight 0
    yielded = by_id[1].ground_delay_s + by_id[1].air_hold_s + by_id[1].air_detour_m
    assert yielded > 0.0


def test_multi_uss_astar_demand_run_verified():
    cfg = SimConfig(
        planner="astar", lam_per_hour=80.0, horizon_s=1800.0, seed=3,
        region_size_m=(5000.0, 5000.0),
    )
    res = run(cfg, demand=UniformPoissonDemand(uss_ids=("walmart", "stripmall")))
    assert res.verified
    s = res.summary()
    assert s["n_accepted"] + s["n_denied"] == s["n_requests"]
    accepted_usses = {i.request.uss_id for i in res.accepted}
    assert accepted_usses == {"walmart", "stripmall"}   # both operators get airspace
