"""MAPF-LNS over a committed schedule: ledger release_many semantics, destroy operators,
and the anytime solver's invariants (feasible + monotone incumbent, exact revert,
determinism, frozen flights, paired-return guard)."""

import hashlib
import pickle

import numpy as np
import pytest

from analysis.ab_column_clear import _intent_digest, _value_sig
from freespace_sim.config import SimConfig
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.lns import LNSConfig, run_lns
from freespace_sim.planner.lns.neighborhood import (
    AdaptiveSelector,
    agent_based_neighborhood,
    map_based_neighborhood,
    random_neighborhood,
)
from freespace_sim.planner.lns import state as lns_state
from freespace_sim.planner.lns.state import LNSState
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()


def _req(fid=1, y=0.0):
    return FlightRequest(fid, vec(0, y, 0), vec(2000, y, 0), 0.0)


def _wall(x=1000.0):
    return Volume4D(box_from_segment(vec(x, -200, 150), vec(x, 200, 150), 40, 400), 0.0, 1e6)


def _ledger_multiset(ledger) -> str:
    """Order-insensitive digest of the live committed (fid, volume) multiset — a revert
    re-commits at the tail, so commit ORDER legitimately changes while content must not."""
    rows = sorted(
        hashlib.sha256(pickle.dumps((fid, _value_sig(v)), protocol=5)).hexdigest()
        for fid, v in ledger.iter_committed()
    )
    return hashlib.sha256("".join(rows).encode()).hexdigest()


# --------------------------------------------------------------------- ledger.release_many
def test_release_many_tombstones_conflicts_and_iteration():
    led = ReservationLedger(CFG)
    led.commit(1, [_wall(1000.0)])
    led.commit(2, [_wall(3000.0)])
    probe = [_wall(1000.0)]
    assert {f for f, _ in led.conflicts(probe)} == {1}
    assert led.n_volumes == 2

    assert led.release_many([1]) == 1
    assert led.conflicts(probe) == []
    assert led.n_volumes == 1
    assert [f for f, _ in led.iter_committed()] == [2]

    led.commit(1, [_wall(1000.0)])          # re-commit after release works
    assert {f for f, _ in led.conflicts(probe)} == {1}
    assert led.n_volumes == 2


def test_release_many_compaction_preserves_content():
    led = ReservationLedger(CFG)
    for fid in range(6):
        led.commit(fid, [_wall(1000.0 + 500 * fid)])
    before_live = _ledger_multiset(led)
    led.release_many([0, 1, 2, 3])           # 4 dead > 2 live -> compaction fires
    assert led._n_dead == 0                  # compacted
    assert led.n_volumes == 2
    assert sorted(f for f, _ in led.iter_committed()) == [4, 5]
    led.commit(0, [_wall(1000.0)])
    led.commit(1, [_wall(1500.0)])
    led.commit(2, [_wall(2000.0)])
    led.commit(3, [_wall(2500.0)])
    assert _ledger_multiset(led) == before_live


def test_incremental_release_reference_service_refcounts():
    """Two flights covering the same cells: removing one must NOT free the cells (refcounts),
    removing both must."""
    from freespace_sim.planner.occupancy import HexOccupancyService

    svc = HexOccupancyService(CFG, track_removal=True)
    svc.on_commit(1, [_wall()])
    svc.on_commit(2, [_wall()])
    step, cells = next(iter(svc.blocked.items()))
    cell = next(iter(cells))
    assert svc.n_added == 2

    svc.on_release(1, [_wall()])
    assert cell in svc.blocked.get(step, {})           # flight 2 still holds the cell
    assert svc.n_added == 1
    svc.on_release(2, [_wall()])
    assert not svc.blocked and not svc.pad             # now genuinely free
    assert svc.n_added == 0


def test_incremental_release_compiled_matches_fresh_absorb():
    """After removing one flight, every pool query must match a fresh instance that only ever
    saw the surviving flight."""
    from freespace_sim.planner.compiled_hex_occupancy import CompiledHexOccupancy

    keep, drop = _wall(3000.0), _wall(1000.0)
    occ = CompiledHexOccupancy(CFG, track_removal=True)
    occ.on_commit(1, [drop])
    occ.on_commit(2, [keep])
    occ.on_release(1, [drop])

    fresh = CompiledHexOccupancy(CFG)
    fresh.on_commit(2, [keep])

    from freespace_sim.planner import hexgrid as hg
    R = hg.circumradius(CFG)
    for x in (1000.0, 3000.0):
        q, r = hg.enu_to_axial(x, 0.0, R)
        for dq in (-1, 0, 1):
            for s in (0, 5, 50, 900):
                assert occ.blocked_py(q + dq, r, 0, s) == fresh.blocked_py(q + dq, r, 0, s)
    assert occ.n_added == fresh.n_added == 1


def test_release_many_incremental_heals_planner_without_rebuild():
    """With incremental_release, release_many is absorbed via on_release: the next plan must be
    byte-identical to a fresh planner AND the shrink tripwire must never fire (no rebuild)."""
    import warnings as _w

    led = ReservationLedger(CFG)
    led.commit(99, [_wall()])
    planner = AStarPlanner(incremental_release=True)
    blocked = planner.plan(_req(1), led, CFG)
    assert blocked.accepted and blocked.air_detour_m > 0.0

    led.release_many([99])
    with _w.catch_warnings():
        _w.filterwarnings("error", message="ReservationLedger shrank")  # a rebuild would fail loudly
        healed = planner.plan(_req(2), led, CFG)
    fresh = AStarPlanner().plan(_req(2), ReservationLedger(CFG), CFG)
    assert _intent_digest(healed) == _intent_digest(fresh)


def test_legacy_release_refuses_release_subscribers():
    led = ReservationLedger(CFG)
    led.commit(1, [_wall()])
    led.subscribe_release(lambda fid, vols: None)
    with pytest.raises(RuntimeError, match="release_many"):
        led.release(1)


def test_release_many_heals_planner_occupancy():
    """After release_many (no observer re-feed), the planner's next plan must rebuild its
    occupancy via the shrink tripwire and produce the same intent as a fresh planner on a
    ledger that never contained the released flight."""
    led = ReservationLedger(CFG)
    led.commit(99, [_wall()])
    planner = AStarPlanner()
    blocked = planner.plan(_req(1), led, CFG)
    assert blocked.accepted and blocked.air_detour_m > 0.0    # wall absorbed, routed around

    led.release_many([99])
    healed = planner.plan(_req(2), led, CFG)

    fresh = AStarPlanner().plan(_req(2), ReservationLedger(CFG), CFG)
    assert _intent_digest(healed) == _intent_digest(fresh)


# ------------------------------------------------------------------------ destroy operators
class _StubCtx:
    """Minimal DestroyContext over hand-built schedules."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n_levels = 1
        self.delays = {}
        self.paths = {}          # fid -> list[(step, cell)]
        self.claims = {}         # cell -> list[(s_lo, s_hi, fid)]
        self.frozen = set()

    def movable_ids(self):
        return sorted(self.delays)

    def is_movable(self, fid):
        return fid in self.delays and fid not in self.frozen

    def delay(self, fid):
        return self.delays[fid]

    def visits(self, fid):
        return self.paths[fid]

    def unimpeded_launch_step(self, fid):
        return self.paths[fid][0][0]

    def owners_over(self, cell, s_lo, s_hi):
        return {f for a, b, f in self.claims.get(cell, ()) if a <= s_hi and b >= s_lo}

    def claim_span(self, cell):
        entries = self.claims[cell]
        return min(e[0] for e in entries), max(e[1] for e in entries)

    def contention_cells(self):
        return sorted(c for c in self.claims if len({e[2] for e in self.claims[c]}) >= 2)


def _delayed_ctx():
    """Flight 1 is delayed (arrival step 12 for a 6-hop route), flight 2 claims every cell
    flight 1's random walk can step onto, flight 3 is undelayed and far away."""
    ctx = _StubCtx(seed=3)
    path = [(s, (s, 0, 0)) for s in range(6)] + [(12, (6, 0, 0))]
    ctx.paths[1] = path
    ctx.delays[1] = 40.0
    ctx.delays[2] = 5.0
    ctx.paths[2] = [(0, (0, 1, 0)), (1, (1, 1, 0))]   # collected agents can become walkers
    ctx.delays[3] = 0.0
    ctx.paths[3] = [(0, (50, 50, 0)), (1, (51, 50, 0))]
    for q in range(-2, 10):
        for r in range(-2, 3):
            ctx.claims[(q, r, 0)] = [(0, 40, 2)]
    return ctx


def test_agent_based_walk_collects_blocker_and_updates_tabu():
    ctx = _delayed_ctx()
    tabu = set()
    nb = agent_based_neighborhood(ctx, n=3, tabu=tabu)
    assert 1 in nb                     # the delayed seed
    assert 2 in nb                     # the blocker its walk ran into
    assert tabu == {1}


def test_agent_based_tabu_resets_when_only_zero_delay_left():
    ctx = _delayed_ctx()
    tabu = {1, 2}                      # every delayed flight already tabued
    nb = agent_based_neighborhood(ctx, n=2, tabu=tabu)
    assert 1 in nb                     # reset re-selects the most delayed overall
    assert 1 in tabu


def test_map_based_collects_claimants_of_contended_cell():
    ctx = _StubCtx(seed=0)
    ctx.delays = {1: 1.0, 2: 1.0, 3: 1.0}
    ctx.claims[(4, 4, 0)] = [(0, 5, 1), (3, 9, 2)]
    ctx.claims[(9, 9, 0)] = [(2, 4, 3)]          # single owner: not contended
    nb = map_based_neighborhood(ctx, n=4, max_cells=64)
    assert nb == {1, 2}


def test_random_neighborhood_size_and_membership():
    ctx = _StubCtx(seed=1)
    ctx.delays = {f: 0.0 for f in range(20)}
    nb = random_neighborhood(ctx, n=8)
    assert len(nb) == 8 and nb <= set(range(20))


def test_adaptive_selector_rewards_shift_probability():
    sel = AdaptiveSelector(("a", "b"), gamma=0.5)
    for _ in range(5):
        sel.update("a", 100.0)
        sel.update("b", 0.0)
    assert sel.weights["a"] > 10 * sel.weights["b"]
    picks = {sel.pick(np.random.default_rng(k)) for k in range(20)}
    assert "a" in picks


# ------------------------------------------------------------------------- solver invariants
def _congested(lam=700.0, horizon=360.0, seed=1):
    """Delay-dominated regime: every flight admitted, most held — the world LNS is for.
    (A saturated world with a binding max_ground_delay_s cap makes PP repair fail wholesale:
    releasing k flights and greedily replanning the first can squeeze the rest past the
    admission cap. See context/lns_plan.md §5.)"""
    return SimConfig(
        planner="astar", flight_levels_m=(75.0,), airspace_ceiling_m=125.0,
        lam_per_hour=lam, horizon_s=horizon,
        region_size_m=(3000.0, 3000.0), seed=seed, max_ground_delay_s=300.0,
    )


def _lns(res, **kw):
    lns_cfg = LNSConfig(**{"seed": 7, "neighborhood_size": 4, "log_every": 0, **kw})
    return run_lns(res.config, res.ledger, res.intents, lns_cfg)


@pytest.mark.slow
def test_lns_zero_iterations_is_identity():
    res = run(_congested(lam=400.0, horizon=240.0))
    before = _ledger_multiset(res.ledger)
    digests = [_intent_digest(i) for i in res.intents]
    out = _lns(res, max_iterations=0)
    assert out.cost_before == out.cost_after
    assert [_intent_digest(i) for i in out.intents] == digests
    assert _ledger_multiset(res.ledger) == before
    assert out.verified


@pytest.mark.slow
def test_lns_all_rejected_reverts_ledger_exactly():
    res = run(_congested(lam=400.0, horizon=240.0))
    before = _ledger_multiset(res.ledger)
    digests = [_intent_digest(i) for i in res.intents]
    out = _lns(res, max_iterations=25, accept_epsilon=float("inf"))
    assert out.n_accepted == 0 and out.cost_after == out.cost_before
    assert [_intent_digest(i) for i in out.intents] == digests
    assert _ledger_multiset(res.ledger) == before
    assert out.verified


@pytest.mark.slow
def test_lns_improves_and_stays_feasible_and_monotone():
    res = run(_congested())
    baseline = sum(i.cost for i in res.accepted)
    out = _lns(res, max_iterations=60, verify_every=2)
    assert out.verified
    assert out.cost_after <= out.cost_before <= baseline + 1e-6
    assert out.n_accepted > 0 and out.cost_after < out.cost_before  # congested: must find something
    costs = [row["incumbent_cost"] for row in out.trajectory]
    assert all(b <= a + 1e-6 for a, b in zip(costs, costs[1:]))
    # accepted rows improve the neighborhood strictly; the incumbent drops by exactly that much
    for prev, row in zip(out.trajectory, out.trajectory[1:]):
        if row["accepted"]:
            assert row["cost_new"] < row["cost_old"]
            assert row["incumbent_cost"] == pytest.approx(
                prev["incumbent_cost"] - (row["cost_old"] - row["cost_new"]))


@pytest.mark.slow
def test_lns_incremental_release_matches_rebuild():
    """Byte-parity of the O(victims) removal path against the reset+reabsorb reference: same
    seed, identical trajectory and identical final schedule."""
    def key(out):
        return [(r["op"], r["n"], tuple(r["victims"]), r["accepted"], r["reason"],
                 round(r["cost_old"], 6), None if r["cost_new"] is None else round(r["cost_new"], 6))
                for r in out.trajectory]

    fast = _lns(run(_congested()), max_iterations=40, incremental_release=True)
    slow = _lns(run(_congested()), max_iterations=40, incremental_release=False)
    assert key(fast) == key(slow)
    assert [_intent_digest(i) for i in fast.intents] == [_intent_digest(i) for i in slow.intents]
    assert fast.verified and slow.verified
    assert fast.cost_after == pytest.approx(slow.cost_after)


@pytest.mark.slow
def test_lns_is_deterministic_per_seed():
    def key(out):
        return [(r["op"], r["n"], r["accepted"], round(r["incumbent_cost"], 6))
                for r in out.trajectory]

    runs = [_lns(run(_congested()), max_iterations=40) for _ in range(2)]
    assert key(runs[0]) == key(runs[1])
    assert runs[0].cost_after == pytest.approx(runs[1].cost_after)


@pytest.mark.slow
def test_lns_never_touches_frozen_flights():
    res = run(_congested())
    frozen = frozenset(i.request.flight_id for i in res.accepted[:5])
    baseline = {f: _intent_digest(i) for f, i in
                ((i.request.flight_id, i) for i in res.intents) if f in frozen}
    out = _lns(res, max_iterations=60, frozen_flight_ids=frozen)
    for row in out.trajectory:
        assert not (set(row["victims"]) & frozen)
    for it in out.intents:
        fid = it.request.flight_id
        if fid in frozen:
            assert _intent_digest(it) == baseline[fid]
    assert out.verified


@pytest.mark.slow
def test_repair_order_modes(monkeypatch):
    """premium mode plans victims in non-increasing delay order; random mode permutes them."""
    res = run(_congested())
    state = LNSState(res.config, res.ledger, res.intents)
    victims = sorted(state.movable_ids(), key=lambda f: -state.delay(f))[:4]
    recorded = []

    def echo_plan(req, ledger, cfg):
        recorded.append(req.flight_id)
        return state.incumbent[req.flight_id]  # accepted, same volumes/cost -> no_improvement

    monkeypatch.setattr(state.repair_planner, "plan", echo_plan)

    out = state.try_repair(victims, np.random.default_rng(3), order_mode="premium")
    assert not out.accepted and out.reason == "no_improvement"
    delays = [state.delay(f) for f in recorded]
    assert sorted(recorded) == sorted(victims) and delays == sorted(delays, reverse=True)

    recorded.clear()
    state.try_repair(victims, np.random.default_rng(3), order_mode="random")
    assert sorted(recorded) == sorted(victims)

    with pytest.raises(ValueError, match="order_mode"):
        state.try_repair(victims, np.random.default_rng(3), order_mode="bogus")


@pytest.mark.slow
def test_paired_return_anchor_guard_rejects_and_reverts(monkeypatch):
    res = run(_congested(lam=400.0, horizon=240.0))
    state = LNSState(res.config, res.ledger, res.intents, turnaround_s=60.0)
    victim = state.movable_ids()[0]
    state._return_anchor[victim] = 0.0                      # committed return departs at t=0
    monkeypatch.setattr(lns_state, "realized_release_s", lambda intent: 1e9)
    before = _ledger_multiset(res.ledger)
    out = state.try_repair([victim], np.random.default_rng(0))
    assert not out.accepted and out.reason == "anchor"
    assert _ledger_multiset(res.ledger) == before
