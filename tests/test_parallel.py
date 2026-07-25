"""Track A Phase 2 (issue #8): speculative worker-pool sim — exact/relaxed modes.

The load-bearing gate is byte-parity: ``parallel=ParallelConfig(mode="exact")`` must reproduce the
sequential FCFS run intent-for-intent (status, reason, cost, delays, centerline, committed volume
geometry — everything but ``solve_time_s``). Relaxed mode must stay verified and deterministic.
Worker-pool tests spawn real processes (2 workers) — kept small so the suite stays fast.
"""
from __future__ import annotations

import pickle   # mirrors the worker-pool IPC wire (trusted, same-process-tree data built in-test)

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand
from freespace_sim.geometry import BoxSpec, CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.parallel import ParallelConfig
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, Terminal, vec
from freespace_sim.volumes import Volume4D


def _clkey(intent):
    return [(round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 3), round(float(t), 6))
            for p, t in (intent.centerline or [])]


def _volkey(intent):
    return [ReservationLedger._flat_aabb(v) + (float(v.t_start), float(v.t_end), v.terminal_id)
            for v in (intent.volumes or [])]


def _assert_byte_identical(seq, par):
    assert len(seq.intents) == len(par.intents)
    for k, (a, b) in enumerate(zip(seq.intents, par.intents)):
        assert a.request.flight_id == b.request.flight_id, f"flight order differs at {k}"
        assert a.status is b.status, f"flight {k}: {a.status} != {b.status}"
        assert a.denial_reason is b.denial_reason, f"flight {k}: denial reason differs"
        if a.accepted:
            assert abs(a.cost - b.cost) < 1e-9, f"flight {k}: cost differs"
            assert a.ground_delay_s == b.ground_delay_s and a.air_hold_s == b.air_hold_s
            assert a.air_detour_m == b.air_detour_m
            assert _clkey(a) == _clkey(b), f"flight {k}: centerline differs"
            assert _volkey(a) == _volkey(b), f"flight {k}: committed volumes differ"
    assert seq.ledger.n_volumes == par.ledger.n_volumes
    assert seq.verified and par.verified


# ---------------- exact mode: byte-identical to sequential ----------------

def test_exact_byte_identical_sparse():
    cfg = SimConfig(lam_per_hour=120.0, horizon_s=600.0, region_size_m=(4000.0, 4000.0), seed=3)
    seq = run(cfg)
    pc = ParallelConfig(n_workers=2, window=8)
    par = run(cfg, parallel=pc)
    _assert_byte_identical(seq, par)
    assert pc.stats["n_canary"] == 0


def test_exact_byte_identical_hub_dense_always_active():
    # multi-pad terminals + always-active walls: static walls registered on worker replicas,
    # TerminalCapacity replicated through the same delta stream, astar_shortcut envelopes via the
    # inner A* (the full production stack, miniaturized).
    cfg = SimConfig(region_size_m=(8000.0, 6000.0), lam_per_hour=500.0, horizon_s=240.0,
                    planner="astar_shortcut", seed=1, terminal_airspace_always_active=True)
    demand = HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 2, "stripmall_uss": 5},
                             radius_m={"walmart_uss": 2500.0, "stripmall_uss": 1500.0},
                             terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
                             pads_per_hub=4, return_flights=True)
    seq = run(cfg, demand=demand)
    assert len(seq.intents) > 10 and any(
        i.request.origin_terminal is not None or i.request.dest_terminal is not None
        for i in seq.intents)
    pc = ParallelConfig(n_workers=2, window=8)
    demand2 = HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 2, "stripmall_uss": 5},
                              radius_m={"walmart_uss": 2500.0, "stripmall_uss": 1500.0},
                              terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
                              pads_per_hub=4, return_flights=True)
    par = run(cfg, demand=demand2, parallel=pc)
    _assert_byte_identical(seq, par)
    assert pc.stats["n_canary"] == 0


def test_exact_forced_interference_replans():
    # two same-window flights whose corridors overlap (60 m apart = corridor width): flight 2's
    # speculation against the empty snapshot is NECESSARILY stale → recovered by eager re-spec or
    # a frontier serial replan; either way the outcome is byte-identical to sequential.
    cfg = SimConfig(region_size_m=(5000.0, 3000.0))
    reqs = [FlightRequest(1, vec(0, 0, 0), vec(3000, 0, 0), 0.0),
            FlightRequest(2, vec(0, 60, 0), vec(3000, 60, 0), 2.0)]
    seq = run(cfg, requests=list(reqs))
    pc = ParallelConfig(n_workers=2, window=8)
    par = run(cfg, requests=list(reqs), parallel=pc)
    _assert_byte_identical(seq, par)
    assert pc.stats["n_dirty"] + pc.stats["n_respec"] >= 1, \
        "overlapping corridors in one window must dirty the speculation"


def test_eager_respec_stats_and_parity():
    # eager off (max_respec=0) → every dirty result is a frontier serial replan; eager on may
    # convert some to re-specs. Results must be byte-identical in all three universes.
    cfg = SimConfig(region_size_m=(5000.0, 3000.0))
    reqs = [FlightRequest(i, vec(0, 60.0 * i, 0), vec(3000, 60.0 * i, 0), 2.0 * i)
            for i in range(6)]
    seq = run(cfg, requests=list(reqs))
    on, off = (ParallelConfig(n_workers=2, window=8, max_respec=2),
               ParallelConfig(n_workers=2, window=8, max_respec=0))
    par_on = run(cfg, requests=list(reqs), parallel=on)
    par_off = run(cfg, requests=list(reqs), parallel=off)
    _assert_byte_identical(seq, par_on)
    _assert_byte_identical(seq, par_off)
    assert off.stats["n_respec"] == 0
    assert off.stats["n_dirty"] >= 1                     # the interference really happened
    assert on.stats["n_canary"] == 0 and off.stats["n_canary"] == 0


# ---------------- eviction floor: out-of-order re-dispatch safety ----------------

def test_respec_out_of_order_eviction_safe():
    """A worker that already planned a LATER flight receives an earlier one (eager re-dispatch).
    With the frontier-clock evict_floor its occupancy AND TerminalCapacity keep the early state, so
    results byte-match fresh planners; uses a capacity-gated multi-pad hub so the tcap eviction path
    (seconds dimension) is exercised, not just the hex-occupancy steps dimension."""
    cfg = SimConfig()
    hub = Terminal("h#0", 2, 90.0)                       # capacity 2 → dwell contention matters
    seed_req = FlightRequest(1, vec(0, 0, 0), vec(3000, -400, 0), 0.0, origin_terminal=hub)
    early = FlightRequest(2, vec(0, 0, 0), vec(3000, 400, 0), 4.0, origin_terminal=hub)
    later = FlightRequest(3, vec(0, 0, 0), vec(3000, 0, 0), 600.0, origin_terminal=hub)

    led = ReservationLedger(cfg)
    seeder = AStarPlanner()
    s = seeder.plan(seed_req, led, cfg)
    assert s.accepted
    led.commit(1, s.volumes)

    # oracle: fresh planners, one per flight, against the same ledger state
    o_later = AStarPlanner().plan(later, led, cfg)
    o_early = AStarPlanner().plan(early, led, cfg)

    # worker: plans LATER first (advancing its request clock), then the earlier flight arrives —
    # the out-of-order case eager re-speculation creates. Floor pinned at the frontier (t=0).
    w = AStarPlanner()
    w.evict_floor = 0.0
    w_later = w.plan(later, led, cfg)
    w_early = w.plan(early, led, cfg)

    for o, got in ((o_later, w_later), (o_early, w_early)):
        assert o.status is got.status and o.denial_reason is got.denial_reason
        if o.accepted:
            assert abs(o.cost - got.cost) < 1e-12 and _clkey(o) == _clkey(got)
            assert _volkey(o) == _volkey(got)


# ---------------- relaxed mode ----------------

def test_relaxed_verified_and_deterministic():
    cfg = SimConfig(lam_per_hour=200.0, horizon_s=600.0, region_size_m=(4000.0, 4000.0), seed=7)
    runs = []
    for _ in range(2):
        pc = ParallelConfig(n_workers=2, window=8, mode="relaxed")
        runs.append(run(cfg, parallel=pc))
        assert runs[-1].verified
        assert pc.stats["n_respec"] == 0                 # eager disabled under relaxed+pinned
        assert pc.stats["predictive"] is False           # dispatch reordering disabled too (pinned
        #                                                  workers cannot shrink an absorbed prefix)
    a, b = runs
    assert len(a.intents) == len(b.intents)
    for x, y in zip(a.intents, b.intents):
        assert x.status is y.status and x.denial_reason is y.denial_reason
        if x.accepted:
            assert abs(x.cost - y.cost) < 1e-12 and _clkey(x) == _clkey(y)
            assert _volkey(x) == _volkey(y)


def test_relaxed_denials_kept():
    # Flight 2's origin sits inside a FOREIGN hub's always-active wall (registered from flight 1's
    # dest terminal at scenario setup, before any planning): takeoff is structurally impossible, so
    # BOTH the snapshot and the sequential search deny it (the wall is static — prefix-independent).
    # Relaxed mode must commit the snapshot denial as-is: no serial replan (obstacle monotonicity).
    from freespace_sim.types import DenialReason
    cfg = SimConfig(region_size_m=(5000.0, 4000.0), terminal_airspace_always_active=True)
    hub = Terminal("x#0", 4, 90.0)
    reqs = [FlightRequest(1, vec(2000, 1500, 0), vec(0, 0, 0), 0.0, dest_terminal=hub),
            FlightRequest(2, vec(0, 0, 0), vec(3000, 0, 0), 1.0)]
    seq = run(cfg, requests=list(reqs))
    seq_d = next(i for i in seq.intents if i.request.flight_id == 2)
    assert seq_d.denial_reason is DenialReason.BUDGET_EXCEEDED
    pc = ParallelConfig(n_workers=2, window=8, mode="relaxed")
    par = run(cfg, requests=list(reqs), parallel=pc)
    par_d = next(i for i in par.intents if i.request.flight_id == 2)
    assert par_d.denial_reason is DenialReason.BUDGET_EXCEEDED and par.verified
    assert pc.stats["n_serial_replans"] == 0             # the snapshot denial was kept, not replanned


# ---------------- replica / IPC fidelity ----------------

def test_delta_stream_rebuilds_identical_ledger():
    cfg = SimConfig()
    vols1 = [Volume4D(box_from_segment(vec(0, 0, 30), vec(500, 0, 30), 60, 25), 0.0, 40.0),
             Volume4D(CylinderSpec(cx=0.1 + 0.2, cy=1 / 3, radius=45.0, z_lo=0.0, z_hi=120.0),
                      0.0, 77.7)]
    vols2 = [Volume4D(box_from_segment(vec(100, 200, 70), vec(900, 800, 70), 60, 25), 10.0, 55.0,
                      terminal_id="h#0")]
    hub = Terminal("h#0", 4, 90.0)
    auth, replica = ReservationLedger(cfg), ReservationLedger(cfg)
    auth.register_static_terminal(vec(2000, 2000, 0), hub)
    replica.register_static_terminal(vec(2000, 2000, 0), hub)
    for fid, vols in ((1, vols1), (2, vols2)):
        auth.commit(fid, vols)
        wire_fid, wire_vols = pickle.loads(pickle.dumps(("delta", fid, vols),
                                                        protocol=pickle.HIGHEST_PROTOCOL))[1:]
        replica.commit(wire_fid, wire_vols)
    assert auth.n_volumes == replica.n_volumes
    assert auth._aabb == replica._aabb                   # exact float equality — bit-preserving wire
    assert auth._fids == replica._fids
    assert auth._buckets == replica._buckets
    assert auth._static_aabb == replica._static_aabb


def test_volume_pickle_bit_exact():
    ugly = 0.1 + 0.2                                     # not exactly representable in decimal
    box = BoxSpec(center=(ugly, 1 / 3, 2 / 7), rot=tuple(np.linalg.qr(
        np.arange(9.0).reshape(3, 3) + np.eye(3) * 5)[0].ravel()), extents=(11.1, 60.0, 25.0))
    cyl = CylinderSpec(cx=ugly * 1e5, cy=-1 / 7, radius=45.000000001, z_lo=0.3, z_hi=119.99999999)
    import dataclasses

    def _floats(spec):
        out = []
        for f in dataclasses.astuple(spec):
            out.extend(f if isinstance(f, tuple) else (f,))
        return out

    for spec in (box, cyl):
        v = Volume4D(spec, t_start=ugly * 100, t_end=1e6 / 3, terminal_id="hub#7")
        w = pickle.loads(pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL))
        assert type(w.shape) is type(v.shape) and w.terminal_id == v.terminal_id
        for a, b in ((v.t_start, w.t_start), (v.t_end, w.t_end)):
            assert np.float64(a).tobytes() == np.float64(b).tobytes()
        for fa, fb in zip(_floats(v.shape), _floats(w.shape), strict=True):
            assert np.float64(fa).tobytes() == np.float64(fb).tobytes()


# ---------------- guards + callback ordering ----------------

def test_parallel_guards():
    import pytest
    with pytest.raises(ValueError, match="envelope-recording planner"):
        run(SimConfig(planner="straight", lam_per_hour=30.0, horizon_s=120.0,
                      region_size_m=(3000.0, 3000.0), seed=2), parallel=2)
    with pytest.raises(ValueError, match="mode"):
        ParallelConfig(mode="bogus")


def test_progress_and_milestones_fire_in_commit_order():
    cfg = SimConfig(lam_per_hour=120.0, horizon_s=600.0, region_size_m=(4000.0, 4000.0), seed=3)
    calls = []

    def report(done, total, intent):
        calls.append((done, total, intent.request.flight_id))

    par = run(cfg, parallel=ParallelConfig(n_workers=2, window=8), progress=report)
    n = len(par.intents)
    assert [c[0] for c in calls] == list(range(1, n + 1))         # strictly in commit order
    assert all(c[1] == n for c in calls)
    assert [c[2] for c in calls] == [i.request.flight_id for i in par.intents]


# ---------------- Phase 3: spatial prediction + adaptation + telemetry ----------------

def test_predictive_dispatch_never_reorders_commits():
    # a huge tube margin makes EVERY pair overlap — maximal deferral pressure on the dispatcher.
    # Scoped to the ordering invariant: commits still land in scenario order (and, exact mode being
    # exact, results still byte-match sequential).
    cfg = SimConfig(lam_per_hour=120.0, horizon_s=600.0, region_size_m=(4000.0, 4000.0), seed=3)
    seq = run(cfg)
    calls = []
    pc = ParallelConfig(n_workers=2, window=8, tube_margin_m=1e6, predictive_dispatch=True)
    par = run(cfg, parallel=pc, progress=lambda d, t, i: calls.append(i.request.flight_id))
    assert calls == [i.request.flight_id for i in seq.intents]    # scenario order, no reordering
    _assert_byte_identical(seq, par)


def test_adaptive_window_clamps():
    from freespace_sim.parallel import AdaptiveWindow
    ctrl = AdaptiveWindow(lo=2, hi=16)
    assert ctrl.w == 16
    for _ in range(200):
        ctrl.observe(True)
        assert 2 <= ctrl.w <= 16
    assert ctrl.w == 2                                            # sustained dirt → floor, never below
    for _ in range(200):
        ctrl.observe(False)
        assert 2 <= ctrl.w <= 16
    assert ctrl.w == 16                                           # sustained clean → back to ceiling


def test_adaptive_window_run_parity():
    cfg = SimConfig(lam_per_hour=120.0, horizon_s=600.0, region_size_m=(4000.0, 4000.0), seed=3)
    seq = run(cfg)
    pc = ParallelConfig(n_workers=2, window=8, adaptive_window=True)
    par = run(cfg, parallel=pc)
    _assert_byte_identical(seq, par)                              # window is a pure throughput knob
    assert 2 <= pc.stats["final_window"] <= 8


def test_parallel_telemetry_merge_matches_sequential():
    # flight 2 must detour around flight 1's always-active hub wall, and the tight detour budget
    # denies the go-around → a guaranteed _file_deny (BUDGET_EXCEEDED with filed volumes) in BOTH
    # runs. Exact mode ⇒ identical plans ⇒ identical telemetry rows, merged in commit order.
    cfg = SimConfig(region_size_m=(5000.0, 4000.0), terminal_airspace_always_active=True,
                    max_detour_factor=1.005)
    hub = Terminal("x#0", 4, 90.0)
    reqs = [FlightRequest(1, vec(1500, 1500, 0), vec(1500, 0, 0), 0.0, dest_terminal=hub),
            FlightRequest(2, vec(0, 0, 0), vec(3000, 0, 0), 1.0)]
    seq = run(cfg, requests=list(reqs), telemetry=True)
    assert seq.telemetry is not None and len(seq.telemetry.filed_volumes) > 0, \
        "scenario produced no filed-denial telemetry — the merge test would be vacuous"
    pc = ParallelConfig(n_workers=2, window=8)
    par = run(cfg, requests=list(reqs), telemetry=True, parallel=pc)
    assert par.telemetry is not None

    def _norm(rows):
        # NaN loses its singleton identity across the worker pickle boundary (nan == nan is False,
        # dict equality only shortcuts on identity) — normalize to None for comparison.
        return [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
                for r in rows]

    assert _norm(par.telemetry.filed_volumes) == _norm(seq.telemetry.filed_volumes)
    assert _norm(par.telemetry.conflict_events) == _norm(seq.telemetry.conflict_events)
    assert par.telemetry.terminals == seq.telemetry.terminals
