"""MAPF-LNS over a committed schedule: ledger release_many semantics, destroy operators,
and the anytime solver's invariants (feasible + monotone incumbent, exact revert,
determinism, frozen flights, paired-return guard)."""

import hashlib
import logging
import pickle
from dataclasses import replace

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


def test_release_many_handles_a_flight_committed_in_several_calls():
    """`_runs` records a flight's slot runs and coalesces contiguous appends, so a flight committed
    in ONE call costs one run and a flight committed in several (interleaved with another's) costs
    several. Releasing must take all of them — a single-run index would silently strand volumes in
    the ledger with nobody owning them."""
    led = ReservationLedger(CFG)
    led.commit(1, [_wall(1000.0), _wall(1500.0)])
    led.commit(2, [_wall(3000.0)])
    led.commit(1, [_wall(2000.0)])                   # non-contiguous second run for flight 1
    assert led._runs[1] == [[0, 2], [3, 4]] and led._runs[2] == [[2, 3]]
    assert led.release_many([1]) == 3                # both runs, not just the first
    assert [f for f, _ in led.iter_committed()] == [2]
    assert led.n_volumes == 1
    assert 1 not in led._runs


def test_compact_renumbers_runs_and_buckets():
    """Compaction moves every live slot, so the own-slot index and the (step, cell) buckets have to
    move with it. A stale run would tombstone SOMEONE ELSE's volume on the next release."""
    led = ReservationLedger(CFG)
    for fid in range(6):
        led.commit(fid, [_wall(1000.0 + 500 * fid)])
    led.release_many([0, 1, 2, 3])                   # 4 dead > 2 live -> compaction fires
    assert led._n_dead == 0
    assert sorted(led._runs) == [4, 5]
    assert sorted(r for runs in led._runs.values() for r in runs) == [[0, 1], [1, 2]]
    probe = [_wall(1000.0 + 500 * 4)]                # flight 4's wall is still found after renumber
    assert {f for f, _ in led.conflicts(probe)} == {4}
    assert led.release_many([4]) == 1 and led.conflicts(probe) == []


def test_compact_coalesces_runs_that_become_adjacent():
    led = ReservationLedger(CFG)
    led.commit(1, [_wall(1000.0)])
    led.commit(2, [_wall(1500.0)])
    led.commit(1, [_wall(2000.0)])
    led.commit(3, [_wall(2500.0)])
    led.commit(4, [_wall(3000.0)])

    led.release_many([2, 3, 4])                    # compaction removes every gap around flight 1
    assert led._runs == {1: [[0, 2]]}
    assert led.release_many([1]) == 2
    assert led.n_volumes == 0


def test_incremental_release_reference_service_refcounts():
    """Two flights covering the same cells: removing one must NOT free the cells (refcounts),
    removing both must."""
    from freespace_sim.planner.astar.occupancy import HexOccupancyService

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
    from freespace_sim.planner.astar.compiled_hex_occupancy import CompiledHexOccupancy

    keep, drop = _wall(3000.0), _wall(1000.0)
    occ = CompiledHexOccupancy(CFG, track_removal=True)
    occ.on_commit(1, [drop])
    occ.on_commit(2, [keep])
    occ.on_release(1, [drop])

    fresh = CompiledHexOccupancy(CFG)
    fresh.on_commit(2, [keep])

    from freespace_sim.planner import hexgrid as hg
    R = hg.circumradius(CFG)
    # every LEVEL, not just 0: the walls span the whole column, so a removal that healed only the
    # bottom level would still answer every L=0 query correctly.
    for x in (1000.0, 3000.0):
        q, r = hg.enu_to_axial(x, 0.0, R)
        for dq in (-1, 0, 1):
            for level in range(CFG.n_levels):
                for s in (0, 5, 50, 900):
                    assert (occ.blocked_py(q + dq, r, level, s)
                            == fresh.blocked_py(q + dq, r, level, s)), (x, dq, level, s)
    assert occ.n_added == fresh.n_added == 1


def test_incremental_release_compiled_treats_identical_spans_as_a_multiset():
    """The claim need not repeat its owner: equal spans are fungible, but their multiplicity is not."""
    from freespace_sim.planner import hexgrid as hg
    from freespace_sim.planner.astar.compiled_hex_occupancy import CompiledHexOccupancy

    wall = _wall()
    occ = CompiledHexOccupancy(CFG, track_removal=True)
    occ.on_commit(1, [wall])
    occ.on_commit(2, [wall])
    q, r, level, s_lo, _s_hi, _in_blk = next(
        row for row in hg.rasterize_ranges(
            wall, CFG, occ.R, occ.infl_blocked, occ.infl_pad
        ) if row[-1]
    )
    step = max(0, s_lo)                                  # pools seed their query horizon at step 0
    assert occ.blocked_py(q, r, level, step)

    occ.on_release(1, [wall])
    assert occ.blocked_py(q, r, level, step)          # the equal claim from flight 2 survives
    occ.on_release(2, [wall])
    assert not occ.blocked_py(q, r, level, step)
    assert occ.n_added == 0


def test_pool_reset_cell_reclaims_overflow_slots():
    """`reset_cell` is called on the same hot cells every LNS iteration. Abandoning the old chain's
    slots is harmless per call but unbounded per run: `_alloc` only bumps, so the pool would grow
    (and `_grow` would double the array) for a working set that never grows."""
    from freespace_sim.planner.astar.compiled_hex_occupancy import _Pool

    pool = _Pool(8, 1000)
    base = pool.nslots
    for _ in range(500):
        pool.block_range(3, 100, 200)          # splits [0,1000] -> allocates an overflow slot
        pool.block_range(3, 400, 500)          # splits the tail -> a second one
        pool.reset_cell(3)
        pool.block_range(3, 100, 200)          # what on_release does: re-seed, re-apply survivors
        pool.reset_cell(3)
    assert pool.nslots <= base + 4             # reused, not leaked (was base + 1500)
    pool.block_range(3, 100, 200)              # and recycled slots still answer correctly
    assert pool.blocked_at(3, 150)
    assert not pool.blocked_at(3, 99) and not pool.blocked_at(3, 300)


def _tcap(track_removal=True):
    from freespace_sim.planner.terminal_capacity import TerminalCapacity

    return TerminalCapacity(CFG, ReservationLedger(CFG), track_removal=track_removal)


def _col(t0=100.0, tid="hub#0"):
    from freespace_sim.volumes import hover_reservation

    return hover_reservation(vec(0, 0, 0), t0, CFG, terminal_id=tid, radius=180.0)


def test_terminal_capacity_release_subtracts_exactly_one_dwell():
    """Dwells are a MULTISET: releasing one flight must remove one instance, not every instance at
    the hub. Two same-hub legs sharing a window is ordinary, and a same-hub column conflicts with
    nothing (conflict.volumes_conflict exempts them), so over-subscribing a pad is invisible to
    `verify` — this counter is the only thing standing between a released flight and a hub that
    silently admits past its capacity."""
    tcap = _tcap()
    same, other = _col(100.0), _col(500.0)
    tcap.on_commit(1, [same])
    tcap.on_commit(2, [same])                       # identical window, different flight
    tcap.on_commit(3, [other])                      # same hub, different window
    span, other_span = (same.t_start, same.t_end), (other.t_start, other.t_end)
    assert sorted(tcap.dwells["hub#0"]) == sorted([span, span, other_span])

    tcap.on_release(1, [same])
    assert sorted(tcap.dwells["hub#0"]) == sorted([span, other_span])   # the twin SURVIVES
    tcap.on_release(3, [other])
    assert tcap.dwells["hub#0"] == [span]
    tcap.on_release(2, [same])
    assert not tcap.dwells and tcap._n_observed_volumes == 0


def test_terminal_capacity_eviction_and_removal_stay_symmetric():
    """`evict_before` drops intervals without touching the per-owner rows, so removal has to know
    which rows eviction already took. The floor test answers that EXACTLY — but only because
    `on_commit` applies the same clamp, so a dwell behind the watermark is never recorded at all.

    Removing by value instead would look safer and be worse: after an eviction the rows and the
    intervals have diverged, so `remove((t0, t1))` can consume a DIFFERENT, still-committed flight's
    identical dwell and delete a live reservation with no error."""
    tcap = _tcap()
    col = _col(100.0)
    span = (col.t_start, col.t_end)

    tcap.on_commit(1, [col])
    assert tcap.dwells["hub#0"] == [span]
    tcap.evict_before(col.t_end + 1.0)              # drops the interval, keeps flight 1's row
    assert not tcap.dwells

    tcap.on_commit(2, [col])                        # behind the watermark: refused, so nothing to steal
    assert not tcap.dwells
    tcap.on_release(1, [col])                       # the floor explains the absence; no raise, no steal
    assert not tcap.dwells
    tcap.on_release(2, [col])
    assert not tcap.dwells and tcap._n_observed_volumes == 0

    # a dwell AHEAD of the watermark is still recorded and still removed exactly
    ahead = _col(col.t_end + 100.0)
    tcap.on_commit(3, [ahead])
    assert tcap.dwells["hub#0"] == [(ahead.t_start, ahead.t_end)]
    tcap.on_release(3, [ahead])
    assert not tcap.dwells


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


def test_legacy_release_delegates_when_release_subscribers_exist():
    """`release`'s rebuild re-commits every survivor, re-feeding commit observers volume by volume —
    which desyncs a service that tracks per-owner rows. It must delegate to `release_many` (whose
    removal publish is exact for those services) rather than lock the caller out of `release` for
    the ledger's whole remaining life: the subscription is a solver's, the ledger outlives it."""
    led = ReservationLedger(CFG)
    led.commit(1, [_wall(1000.0)])
    led.commit(2, [_wall(3000.0)])
    seen = []
    led.subscribe_release(lambda fid, vols: seen.append((fid, len(vols))))

    led.release(1)
    assert seen == [(1, 1)]                              # one exact removal publish, not a re-feed
    assert [f for f, _ in led.iter_committed()] == [2]
    assert led.n_volumes == 1
    assert led.conflicts([_wall(1000.0)]) == []


def test_detach_subscribers_forces_a_stale_planner_to_rebind():
    """Taking a ledger over (LNS) clears its subscribers, leaving the previous planner bound but
    DEAF: it neither re-subscribes nor rebuilds, and its shrink tripwire cannot help — a release
    plus a re-commit nets to the same `n_volumes`. The epoch bump is what makes it rebind."""
    led = ReservationLedger(CFG)
    planner = AStarPlanner()
    assert planner.plan(_req(1), led, CFG).accepted        # binds its services to this ledger

    led.detach_subscribers()                               # ownership transfer
    led.commit(99, [_wall()])                              # a commit the old planner never observed

    rebound = planner.plan(_req(2), led, CFG)
    walled_led = ReservationLedger(CFG)
    walled_led.commit(99, [_wall()])
    walled = AStarPlanner().plan(_req(2), walled_led, CFG)
    blind = AStarPlanner().plan(_req(2), ReservationLedger(CFG), CFG)
    assert _intent_digest(walled) != _intent_digest(blind)  # non-vacuous: the wall changes this plan
    assert _intent_digest(rebound) == _intent_digest(walled)


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
    # The wheel must actually BE weighted, not merely hold weights: here p(b) ≈ 0.03/97 ≈ 3e-4, so
    # 200 draws expect ~0 of them. A uniform pick — the mutation this pins — would take "b" ~100
    # times, while "is 'a' ever picked?" passes under either.
    rng = np.random.default_rng(0)
    picks = [sel.pick(rng) for _ in range(200)]
    assert picks.count("b") <= 2
    assert picks.count("a") >= 198


# ------------------------------------------------------------------------- solver invariants
def _congested(lam=700.0, horizon=360.0, seed=1, levels=(75.0,)):
    """Delay-dominated regime: every flight admitted, most held — the world LNS is for.
    (A saturated world with a binding max_ground_delay_s cap makes PP repair fail wholesale:
    releasing k flights and greedily replanning the first can squeeze the rest past the
    admission cap. See context/lns_plan.md §5.)"""
    return SimConfig(
        planner="astar", flight_levels_m=levels, airspace_ceiling_m=125.0,
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
def test_lns_improves_a_multi_level_schedule():
    """Every other solver test flies ONE flight level, where every level-indexed structure — the
    claim index's (q, r, level) key, the per-cell occupancy removal, `_extract_visits`' level pick —
    collapses to the same index for every flight, so a level bug cancels out. Fly two."""
    res = run(_congested(levels=(45.0, 85.0)))
    cruise = {round(max(float(p[2]) for p, _t in i.centerline), 1) for i in res.accepted}
    assert len(cruise) > 1, f"single-level schedule ({cruise}): this test would pass vacuously"

    out = _lns(res, max_iterations=40, verify_every=2)
    assert out.verified                     # verify_every replays the WHOLE schedule, all levels
    assert out.cost_after < out.cost_before
    assert len(out.intents) == len(res.intents)


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
def test_repair_restores_the_ledger_on_denial_and_on_exception(monkeypatch):
    """Every exit from the destroyed state must put the incumbent back. The ledger is the run's only
    copy of the schedule: a transaction that unwinds without restoring loses flights outright, and
    nothing downstream can tell (the next iteration replans against a world short k flights)."""
    from freespace_sim.planner.astar.planner import _deny
    from freespace_sim.types import DenialReason

    res = run(_congested(lam=400.0, horizon=240.0))
    state = LNSState(res.config, res.ledger, res.intents)
    victims = sorted(state.movable_ids())[:3]
    before = _ledger_multiset(res.ledger)

    # 1. denial part-way through: `new` is a STRICT subset of `victims`, so the restore must also
    #    re-commit the victims the repair never reached.
    calls = []

    def deny_after_one(req, ledger, cfg):
        calls.append(req.flight_id)
        if len(calls) == 1:
            return state.incumbent[req.flight_id]
        return _deny(req, DenialReason.CONFLICT_FILED)

    monkeypatch.setattr(state.repair_planner, "plan", deny_after_one)
    out = state.try_repair(victims, np.random.default_rng(0))
    assert not out.accepted and out.reason == "denied"
    assert 0 < out.n_planned < len(victims)
    assert _ledger_multiset(res.ledger) == before

    # 2. an exception mid-repair, after some victims are already re-committed
    calls.clear()

    def boom(req, ledger, cfg):
        calls.append(req.flight_id)
        if len(calls) == len(victims):
            raise RuntimeError("planner exploded")
        return state.incumbent[req.flight_id]

    monkeypatch.setattr(state.repair_planner, "plan", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        state.try_repair(victims, np.random.default_rng(0))
    assert _ledger_multiset(res.ledger) == before

    # 3. a bad order_mode must be caught BEFORE the destroy, not after it
    with pytest.raises(ValueError, match="order_mode"):
        state.try_repair(victims, np.random.default_rng(0), order_mode="bogus")
    assert _ledger_multiset(res.ledger) == before


def test_claim_index_excludes_the_flights_own_terminal_interior():
    """The claim index must mirror what A* actually deconflicts against. Corridor cells inside the
    flight's OWN terminal column are the vertiport's unreserved tactical interior — the occupancy
    services drop them — so indexing them would make every pair of same-hub flights look mutually
    contended, and the map-based operator picks its neighborhoods BY contention."""
    from freespace_sim.geometry import CylinderSpec
    from freespace_sim.planner import hexgrid as hg
    from freespace_sim.planner.astar.occupancy import HexOccupancyService
    from freespace_sim.volumes import corridor_segment_volume, hover_reservation

    cfg = SimConfig(flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
                    region_size_m=(20_000.0, 20_000.0), terminal_radius_m=400.0)
    # a big-box hub: the column spans several hexes, and the flight's own lane crosses them
    vols = [
        hover_reservation(vec(0, 0, 0), 0.0, cfg, terminal_id="hub#0", radius=400.0),
        corridor_segment_volume(vec(0, 0, 100), 0.0, vec(3000, 0, 100), 200.0, cfg,
                                terminal_id="hub#0"),
    ]
    state = LNSState(cfg, ReservationLedger(cfg), [])
    state._index_add(7, vols)
    svc = HexOccupancyService(cfg)          # the authority the repair planner consults
    svc.on_commit(7, vols)

    blocked = {cell for cells in svc.blocked.values() for cell in cells}
    assert set(state._claims) == blocked

    naive = set()                           # what the index would hold without the own-column drop
    for v in vols:
        if v.terminal_id is not None and isinstance(v.shape, CylinderSpec):
            continue
        for q, r, level, _lo, _hi, in_blk in hg.rasterize_ranges(
            v, cfg, state._R, state._infl_b, state._infl_p
        ):
            if in_blk:
                naive.add((q, r, level))
    assert naive - blocked, "no own-column cells here: this test would pass vacuously"


def test_borrowed_repair_planner_keeps_its_own_eviction_floor():
    """`evict_floor = 0.0` is a requirement on the repair planner, not something to silently write
    into a caller's object — that mutation outlives the state and changes how their planner behaves
    everywhere else."""
    led = ReservationLedger(CFG)
    led.subscribe(lambda fid, vols: None)
    with pytest.raises(ValueError, match="evict_floor"):
        LNSState(CFG, led, [], repair_planner=AStarPlanner())
    assert led._observers and led.epoch == 0   # a construction that raises leaves the ledger alone

    ready = AStarPlanner()
    ready.evict_floor = 0.0
    state = LNSState(CFG, ReservationLedger(CFG), [], repair_planner=ready)
    assert state.repair_planner is ready and ready.evict_floor == 0.0


# ------------------------------------------------------------------------------ solver entry points
def test_empty_operators_is_a_clear_error():
    with pytest.raises(ValueError, match="operators is empty"):
        run_lns(CFG, ReservationLedger(CFG), [], LNSConfig(operators=(), max_iterations=1))


def test_run_lns_does_not_leak_a_global_warnings_filter():
    """The per-shrink warning is noise INSIDE the loop only. Installed globally it would stay in
    force for everything the caller does afterwards, including a later run that wants to see it."""
    import warnings as _w

    before = list(_w.filters)
    run_lns(CFG, ReservationLedger(CFG), [], LNSConfig(max_iterations=0, log_every=0))
    assert list(_w.filters) == before


def _result(ledger=None, return_anchor="nominal"):
    """A REAL SimResult, not a hand-rolled stand-in. A stub with the same four attributes would let
    these tests keep passing after the field is renamed or dropped from the dataclass, while every
    real run silently lost the anchor mode — the exact hole `run_lns_on_result` reads it to close."""
    from freespace_sim.sim import SimResult

    return SimResult(config=CFG, intents=[], ledger=ledger or ReservationLedger(CFG),
                     verified=True, return_anchor=return_anchor)


class _WallInventingDemand:
    turnaround_s = 900.0

    def terminals(self, cfg):
        raise AssertionError("walls must come from the ledger, not be re-derived from the demand")


def test_run_lns_on_result_takes_the_walls_from_the_ledger(monkeypatch):
    """`terminal_airspace_always_active` and the demand model's own hub set are independent: a run
    with the flag off files NO walls, and a demand model without `terminals` makes sim.run fall back
    to the scenario's hubs. Re-deriving them here invents an obstacle field the schedule was never
    planned against (over-charging every unimpeded cost, so delay premiums — which choose the
    victims — are wrong) or crashes. Only the ledger knows what was filed."""
    from freespace_sim.planner.lns import solver as lns_solver
    from freespace_sim.types import Terminal

    captured = {}
    monkeypatch.setattr(lns_solver, "run_lns", lambda *a, **kw: captured.update(kw))

    walled = ReservationLedger(CFG)
    walled.register_static_terminal(vec(0, 0, 0), Terminal("hub#0", 8, 180.0))
    lns_solver.run_lns_on_result(_result(walled), _WallInventingDemand(), LNSConfig())
    assert [t.id for _c, t in captured["static_terms"]] == ["hub#0"]

    captured.clear()
    lns_solver.run_lns_on_result(_result(), _WallInventingDemand(), LNSConfig())
    assert captured["static_terms"] == ()          # flag off ⇒ no walls, whatever the demand says


def test_run_lns_on_result_takes_the_anchor_mode_from_the_baseline(monkeypatch):
    """The paired-return guard is only correct if it runs whenever the baseline anchored returns to
    realized arrivals. Defaulting the mode here disabled it silently for exactly those runs."""
    from freespace_sim.planner.lns import solver as lns_solver

    captured = {}
    monkeypatch.setattr(lns_solver, "run_lns", lambda *a, **kw: captured.update(kw))
    demand = _WallInventingDemand()

    lns_solver.run_lns_on_result(_result(return_anchor="realized"), demand, LNSConfig())
    assert captured["turnaround_s"] == 900.0       # guard armed without being asked

    captured.clear()
    lns_solver.run_lns_on_result(_result(), demand, LNSConfig())
    assert captured["turnaround_s"] is None

    with pytest.raises(ValueError, match="contradicts"):
        lns_solver.run_lns_on_result(_result(return_anchor="realized"), demand,
                                     LNSConfig(), return_anchor="nominal")


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


# --------------------------------------------------------------- transaction atomicity
@pytest.mark.slow
def test_repair_restores_when_the_commit_itself_raises(monkeypatch):
    """`ledger.commit` appends the volumes and only THEN fires observers, so an observer that raises
    leaves the repair's volumes live while `try_repair`'s `new[fid] = it` never ran. The rewind must
    release the VICTIMS — a superset of what it recorded — or that flight ends up double-booked: its
    abandoned repair and its restored incumbent both live, mutually conflicting, and invisible to
    `verify` (which replays final_intents(), not the ledger)."""
    res = run(_congested(lam=400.0, horizon=240.0))
    state = LNSState(res.config, res.ledger, res.intents)
    victims = sorted(state.movable_ids())[:3]
    before = _ledger_multiset(res.ledger)
    live_before = res.ledger.n_volumes

    seen = {"n": 0}

    def explode_on_second_commit(fid, vols):
        seen["n"] += 1
        if seen["n"] == 2:
            raise RuntimeError("observer exploded")

    res.ledger.subscribe(explode_on_second_commit)
    monkeypatch.setattr(state.repair_planner, "plan",
                        lambda req, ledger, cfg: state.incumbent[req.flight_id])
    try:
        with pytest.raises(RuntimeError, match="observer exploded"):
            state.try_repair(victims, np.random.default_rng(0))
    finally:
        res.ledger._observers.remove(explode_on_second_commit)
    assert _ledger_multiset(res.ledger) == before
    assert res.ledger.n_volumes == live_before          # nothing double-booked


@pytest.mark.slow
def test_repair_restores_when_the_destroy_itself_raises():
    """`release_many` tombstones every victim volume BEFORE notifying removal subscribers, so a
    raising hook leaves the ledger stripped. The destroy therefore has to sit inside the same try as
    the repair — outside it, the rewind never runs and k flights are simply gone."""
    res = run(_congested(lam=400.0, horizon=240.0))
    state = LNSState(res.config, res.ledger, res.intents)
    victims = sorted(state.movable_ids())[:3]
    before = _ledger_multiset(res.ledger)

    def explode(fid, vols):
        raise RuntimeError("release hook exploded")

    res.ledger.subscribe_release(explode)
    try:
        with pytest.raises(RuntimeError, match="release hook exploded"):
            state.try_repair(victims, np.random.default_rng(0))
    finally:
        res.ledger._release_subs.remove(explode)
    assert _ledger_multiset(res.ledger) == before


@pytest.mark.slow
@pytest.mark.parametrize("method_name", ["_index_remove", "_index_add"])
def test_repair_restores_when_accept_index_mutation_raises(monkeypatch, method_name):
    """An accept-side index failure is caught by ``try_repair``, so its rollback must restore every
    in-memory view as well as the ledger before re-raising."""
    res = run(CFG, requests=[_req(1)])
    state = LNSState(res.config, res.ledger, res.intents)
    fid = state.movable_ids()[0]
    old = state.incumbent[fid]
    candidate = replace(old, cost=old.cost - 1.0)

    ledger_before = _ledger_multiset(res.ledger)
    cost_before = state.total_cost
    claims_before = {cell: list(rows) for cell, rows in state._claims.items()}
    cells_before = {owner: set(rows) for owner, rows in state._cells_of.items()}
    contention_before = list(state.contention_cells())

    monkeypatch.setattr(
        state.repair_planner, "plan", lambda request, ledger, cfg: candidate
    )
    original = getattr(state, method_name)
    calls = 0

    def explode_after_mutation(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 1:
            raise RuntimeError("index mutation exploded")
        return result

    monkeypatch.setattr(state, method_name, explode_after_mutation)
    with pytest.raises(RuntimeError, match="index mutation exploded"):
        state.try_repair([fid], np.random.default_rng(0))

    assert state.incumbent[fid] is old
    assert state.total_cost == cost_before
    assert _ledger_multiset(res.ledger) == ledger_before
    assert state._claims == claims_before
    assert {owner: set(rows) for owner, rows in state._cells_of.items()} == cells_before
    assert state.contention_cells() == contention_before
    assert lns_state._same_committed_schedule(res.ledger, state.final_intents())


# ------------------------------------------------------------------ unimpeded delay ruler
def test_unimpeded_costs_parallel_matches_sequential():
    """The ruler shards across processes because its ledger is never committed to — plan i cannot
    observe plan j. Force the pool (probe prefix of 0, no projection floor) and require the SAME
    costs in the SAME request order: this is the only thing standing between a throughput knob and
    a silently different delay premium, which would re-rank every victim and every repair."""
    from freespace_sim.planner.lns import unimpeded as U

    reqs = [_req(fid, y=200.0 * fid) for fid in range(1, 13)]
    seq = U.unimpeded_costs(CFG, (), reqs, n_workers=1)
    par = _forced_parallel(U, CFG, reqs, workers=4)
    assert [r[0] for r in seq] == [r.flight_id for r in reqs]     # request order, not shard order
    assert par == seq
    assert all(c is not None for _f, c, _d in seq)                # a walls-free world places them all


def test_unimpeded_costs_survives_a_dead_worker(monkeypatch):
    """A worker that dies must cost throughput, not a flight: its shard is replanned in-process.
    Losing one silently would leave `delay()` reading a KeyError-free but WRONG premium."""
    from freespace_sim.planner.lns import unimpeded as U

    reqs = [_req(fid, y=200.0 * fid) for fid in range(1, 9)]
    expected = U.unimpeded_costs(CFG, (), reqs, n_workers=1)

    # module level: `spawn` pickles the Process target BY NAME, so a closure cannot be one
    monkeypatch.setattr(U, "_worker_main", _suicide_worker)
    assert _forced_parallel(U, CFG, reqs, workers=2) == expected


def test_unimpeded_costs_cleans_up_after_partial_spawn_failure(monkeypatch):
    """If worker N fails to start, workers 0..N-1 must be reaped and every request replanned."""
    from freespace_sim.planner.lns import unimpeded as U

    reqs = [_req(fid, y=200.0 * fid) for fid in range(1, 3)]
    expected = U.unimpeded_costs(CFG, (), reqs, n_workers=1)

    class FakeConn:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self, fail):
            self.fail = fail
            self.pid = None
            self.exitcode = None
            self.alive = False
            self.terminated = False

        def start(self):
            if self.fail:
                raise OSError("spawn limit")
            self.pid = 123
            self.alive = True

        def join(self, timeout=None):
            if self.terminated:
                self.alive = False
                self.exitcode = -15

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.alive = False

    class FakeContext:
        def __init__(self):
            self.processes = []
            self.connections = []

        def Pipe(self, duplex=False):
            assert duplex is False
            pair = FakeConn(), FakeConn()
            self.connections.extend(pair)
            return pair

        def Process(self, **_kwargs):
            proc = FakeProcess(fail=bool(self.processes))
            self.processes.append(proc)
            return proc

    ctx = FakeContext()
    monkeypatch.setattr(U.mp, "get_context", lambda _method: ctx)

    assert _forced_parallel(U, CFG, reqs, workers=2) == expected
    assert ctx.processes[0].terminated
    assert all(conn.closed for conn in ctx.connections)


def _suicide_worker(conn, cfg, static_terms, requests):
    """A worker that exits without answering (see test_unimpeded_costs_survives_a_dead_worker)."""
    conn.close()


def _forced_parallel(U, cfg, reqs, workers):
    """Run `unimpeded_costs` with the pool forced on regardless of how fast the probe ran."""
    probe, floor = U._PROBE_N, U._MIN_PARALLEL_S
    U._PROBE_N, U._MIN_PARALLEL_S = 0, -1.0
    try:
        return U.unimpeded_costs(cfg, (), reqs, n_workers=workers)
    finally:
        U._PROBE_N, U._MIN_PARALLEL_S = probe, floor


@pytest.mark.slow
def test_lns_seed_changes_the_search():
    """LNSConfig.seed is the reproducibility knob and nothing else in the suite varies it, so a
    solver that ignored it entirely would still look deterministic."""
    def traj(seed):
        res = run(_congested())
        out = run_lns(res.config, res.ledger, res.intents,
                      LNSConfig(seed=seed, max_iterations=12, neighborhood_size=4, log_every=0))
        return [tuple(r["victims"]) for r in out.trajectory]

    assert traj(1) != traj(2)


# ------------------------------------------------------------- ownership transfer and teardown
def test_detach_subscribers_releases_the_static_hooks_too():
    """The static hooks are BOUND METHODS of the detached services, so keeping them pins the very
    objects the transfer exists to release, and every rebind appends another pair. Nothing is lost:
    subscribe_static REPLAYS every registered hub to each new subscriber."""
    from freespace_sim.types import Terminal

    led = ReservationLedger(CFG)
    led.register_static_terminal(vec(0, 0, 0), Terminal("hub#0", 8, 180.0))
    planner = AStarPlanner()
    planner.plan(_req(1), led, CFG)
    assert led._static_subs                       # the planner derived its routing walls

    led.detach_subscribers()
    assert not led._observers and not led._release_subs and not led._static_subs

    seen = []                                     # a new owner still gets the walls, via the replay
    led.subscribe_static(lambda c, t: seen.append(t.id))
    assert seen == ["hub#0"]


@pytest.mark.slow
def test_run_lns_hands_the_ledger_back_clean():
    """Without a symmetric teardown the repair planner's services stay wired to a ledger whose owner
    is gone, and any later commit by the caller silently feeds them — the takeover's own failure mode
    in the other direction. Needs real iterations: the repair planner subscribes lazily on its first
    plan(), so a zero-iteration run would pass this vacuously."""
    res = run(_congested(lam=400.0, horizon=240.0))
    out = _lns(res, max_iterations=6)
    assert any(row["n"] for row in out.trajectory)      # the repair planner really did bind
    assert not res.ledger._observers
    assert not res.ledger._release_subs
    assert not res.ledger._static_subs


@pytest.mark.slow
def test_run_lns_logs_and_detaches_when_an_iteration_raises(monkeypatch, caplog):
    from freespace_sim.planner.lns import solver as lns_solver

    res = run(CFG, requests=[_req(1)])

    def explode(self, victims, rng, accept_epsilon=0.0, order_mode="premium"):
        fid = next(iter(victims))
        request = self.incumbent[fid].request
        self.repair_planner._occupancy(request, self.ledger, self.cfg)
        assert self.ledger._observers
        assert self.ledger._release_subs
        assert self.ledger._static_subs
        raise RuntimeError("repair iteration exploded")

    monkeypatch.setattr(lns_solver.LNSState, "try_repair", explode)
    config = LNSConfig(
        max_iterations=1,
        neighborhood_size=1,
        operators=("random",),
        adaptive=False,
        log_every=0,
    )
    with caplog.at_level(logging.ERROR, logger="freespace_sim.lns"):
        with pytest.raises(RuntimeError, match="repair iteration exploded"):
            lns_solver.run_lns(res.config, res.ledger, res.intents, config)

    assert not res.ledger._observers
    assert not res.ledger._release_subs
    assert not res.ledger._static_subs
    records = [record for record in caplog.records if "lns aborted" in record.message]
    assert len(records) == 1 and records[0].exc_info is not None


def test_milp_capacity_rebinds_after_a_takeover():
    """The epoch contract is ledger-wide, not an A* detail: MILPOptPlanner keeps its own pad-capacity
    index on the shared ledger, and its count tripwire cannot see a takeover (LNS restores every
    flight it releases, so n_volumes ends at or above the frozen count)."""
    from freespace_sim.planner.milp import MILPOptPlanner

    led = ReservationLedger(CFG)
    planner = MILPOptPlanner()
    first = planner._capacity(led, CFG, 0.0)
    assert led._observers

    led.detach_subscribers()
    second = planner._capacity(led, CFG, 0.0)
    assert second is not first                    # rebound, not merely reused
    assert led._observers                         # and re-subscribed, so it stays in sync


# -------------------------------------------------------------------- run_lns argument contract
def test_run_lns_defaults_unimpeded_ruler_to_in_process(monkeypatch):
    """A public API caller must opt into spawn; its top-level module may not have a main guard."""
    seen = []

    def capture(_cfg, _static_terms, _requests, *, n_workers, log_every=1000):
        seen.append(n_workers)
        return []

    monkeypatch.setattr(lns_state, "unimpeded_costs", capture)
    assert LNSConfig().unimpeded_workers == 1

    run_lns(CFG, ReservationLedger(CFG), [],
            LNSConfig(max_iterations=0, log_every=0))
    LNSState(CFG, ReservationLedger(CFG), [])       # direct construction is safe by default too
    assert seen == [1, 1]


def test_run_lns_defaults_its_walls_to_the_ledgers(monkeypatch):
    """Left at (), the unimpeded baseline is wall-free — inflating every delay premium, which is the
    ranking that picks victims and orders the repair — and the closing verify replays a world the
    schedule was never planned against, so `verified` can come back True for an infeasible one."""
    from freespace_sim.planner.lns import solver as lns_solver
    from freespace_sim.types import Terminal

    led = ReservationLedger(CFG)
    led.register_static_terminal(vec(0, 0, 0), Terminal("hub#0", 8, 180.0))
    seen = {}
    real = lns_solver.LNSState

    def spy(*a, **kw):
        seen["static_terms"] = kw["static_terms"]
        return real(*a, **kw)

    monkeypatch.setattr(lns_solver, "LNSState", spy)
    run_lns(CFG, led, [], LNSConfig(max_iterations=0, log_every=0))
    assert [t.id for _c, t in seen["static_terms"]] == ["hub#0"]


def test_duplicate_operators_are_rejected():
    """`ops` collapses the duplicate while AdaptiveSelector.names keeps it, so the operator would get
    a double share of the wheel while the reported weights dict is shorter than the configuration."""
    with pytest.raises(ValueError, match="duplicate"):
        run_lns(CFG, ReservationLedger(CFG), [], LNSConfig(operators=("agent", "agent", "map")))


def test_operator_validation_precedes_the_ledger_takeover():
    """The guards are pure functions of the config, while constructing LNSState detaches the caller's
    subscribers irrecoverably and spends one A* plan per movable flight."""
    led = ReservationLedger(CFG)
    led.subscribe(lambda fid, vols: None)
    for bad in (LNSConfig(operators=()), LNSConfig(operators=("agnt",)),
                LNSConfig(operators=("map", "map"))):
        with pytest.raises(ValueError):
            run_lns(CFG, led, [], bad)
        assert led._observers and led.epoch == 0      # ledger untouched by the rejected call


def test_run_lns_on_result_refuses_a_result_without_an_anchor_mode():
    """'nominal' is the value that DISARMS the paired-return guard, so defaulting to it on a result
    type that does not carry the field is the unsafe direction."""
    from freespace_sim.planner.lns import solver as lns_solver

    class _NoAnchor:
        config, ledger, intents = CFG, ReservationLedger(CFG), []

    with pytest.raises(TypeError, match="return_anchor"):
        lns_solver.run_lns_on_result(_NoAnchor(), None, LNSConfig())


def test_lns_state_refuses_intents_that_are_not_this_ledgers():
    """run_lns mutates the ledger in place and never writes back to the caller's intent list, so
    reusing one SimResult for a second pass measures a stale baseline against an improved ledger and
    returns a genuinely conflicting schedule."""
    led = ReservationLedger(CFG)
    led.commit(1, [_wall()])
    with pytest.raises(ValueError, match="not the same schedule"):
        LNSState(CFG, led, [])


def test_lns_state_refuses_a_same_count_different_schedule():
    """A count-only guard misses a stale schedule when both versions contain the same number of
    volumes; the exact run-pair check must reject it before taking over the ledger."""
    res = run(CFG, requests=[_req(1)])
    intent = res.intents[0]
    different = [replace(v, t_start=v.t_start + 1000.0, t_end=v.t_end + 1000.0)
                 for v in intent.volumes]
    ledger = ReservationLedger(CFG)
    ledger.commit(intent.request.flight_id, different)
    assert ledger.n_volumes == len(intent.volumes)

    with pytest.raises(ValueError, match="not the same schedule"):
        LNSState(CFG, ledger, [intent])


def test_lns_refuses_a_baseline_it_cannot_measure():
    """The unimpeded ruler and the repair planner are plain A*; measuring a shortcut/MILP baseline
    against them subtracts one planner's cost from another's, and delay()'s max(0.0, ...) hides the
    result by clamping the negative premiums to zero."""
    cfg = SimConfig(planner="astar_shortcut")
    with pytest.raises(ValueError, match="cannot measure"):
        LNSState(cfg, ReservationLedger(cfg), [])


def test_sim_run_records_the_anchor_mode_it_flew():
    """The read side is pinned by the run_lns_on_result tests; this is the write. Without it the
    field silently defaults to 'nominal' and the paired-return guard never arms."""
    cfg = _congested(lam=200.0, horizon=120.0)
    assert run(cfg).return_anchor == "nominal"
    assert run(cfg, return_anchor="realized").return_anchor == "realized"


def test_pool_reset_clears_the_free_list():
    """`reset()` restarts the bump allocator at NC, invalidating every previously-freed slot id.
    Keeping them hands the same slot out twice — once from the free list, again as nslots climbs past
    it — aliasing two cells' interval chains and silently corrupting blocked_at."""
    from freespace_sim.planner.astar._packed import P_NXT
    from freespace_sim.planner.astar.compiled_hex_occupancy import _Pool

    pool = _Pool(8, 1000)
    pool.block_range(3, 100, 200)
    pool.reset_cell(3)
    assert pool._free                              # a slot really was reclaimed

    pool.reset()
    assert not pool._free and pool.nslots == pool.NC

    for c in range(pool.NC):                       # each split allocates one overflow slot
        pool.block_range(c, 100, 200)
    seen = set()
    for c in range(pool.NC):
        slot = int(pool.iv[c, P_NXT])
        while slot != -1:
            assert slot not in seen, f"slot {slot} aliased between two cell chains"
            seen.add(slot)
            slot = int(pool.iv[slot, P_NXT])


# ---------------------------------------------------------------- reject-path undo journal

def _occ_answers(state, n=4000, seed=3):
    """A sample of the compiled occupancy's ANSWERS, not its bytes.

    The journal's restore is deliberately CANONICAL rather than byte-identical — it drops the empty
    `lo > hi` slots `block_range` leaves behind, so a restored chain can be shorter than a rebuilt
    one while describing the same free-step set. `_Pool.reset_cell` already states the underlying
    contract ("which slot holds an interval never affects an answer"), so the thing to pin is
    `blocked_py`, which is exactly what the kernel reads."""
    cocc = state.repair_planner._cocc
    rng = np.random.default_rng(seed)
    qs = rng.integers(cocc.qmin, cocc.qmin + cocc.qspan, n)
    rs = rng.integers(cocc.rmin, cocc.rmin + cocc.rspan, n)
    ss = rng.integers(0, cocc.MAXS, n)
    return [cocc.blocked_py(int(q), int(r), 0, int(s)) for q, r, s in zip(qs, rs, ss)]


def _drive(res, journal, iters=25, n=4):
    """Run the same LNS tasks with the journal on or off, returning what must not differ."""
    led = ReservationLedger(res.config)
    for c, t in res.ledger.static_terminals():
        led.register_static_terminal(c, t)
    for it in res.intents:
        if it.accepted and it.volumes:
            led.commit(it.request.flight_id, it.volumes)
    st = LNSState(res.config, led, list(res.intents),
                  static_terms=res.ledger.static_terminals(), undo_journal=journal)
    rows = []
    for i in range(iters):
        rng = np.random.default_rng(np.random.SeedSequence([11, i]))
        st.rng = rng
        victims = random_neighborhood(st, n)
        if not victims:
            continue
        out = st.try_repair(victims, rng, 0.0, "premium")
        rows.append((sorted(victims), out.accepted, out.reason,
                     round(float(out.cost_old), 9), round(float(out.cost_new), 9)))
    return rows, _occ_answers(st), _ledger_multiset(led), st


@pytest.mark.slow
def test_undo_journal_is_answer_identical_to_release_and_recommit():
    """The reject path's whole contract: rolling the occupancy back to a snapshot must leave it
    answering exactly as releasing and re-committing does.

    Both arms run the SAME victim sequence, so any divergence is the journal and nothing else. Three
    things are compared, because they fail differently: the per-task outcomes (a wrong occupancy
    makes a repair see a different world and take a different decision), the sampled occupancy
    answers (a drift too small to move a decision yet), and the final committed schedule."""
    res = run(_congested(lam=700.0, horizon=360.0))
    a_rows, a_occ, a_dig, a_st = _drive(res, journal=False)
    b_rows, b_occ, b_dig, b_st = _drive(res, journal=True)
    assert a_rows == b_rows, "the journal changed a repair decision"
    assert a_occ == b_occ, "the journal left the occupancy answering differently"
    assert a_dig == b_dig, "the journal changed the committed schedule"
    # Non-vacuous on both counts: some task must have REJECTED (that is the path under test), and
    # the sample must contain both answers or comparing constants would pass.
    assert any(not r[1] for r in b_rows), "no task rejected — the journal was never exercised"
    assert 0 < sum(b_occ) < len(b_occ), "occupancy sample is all-free or all-blocked"
    assert b_st.repair_planner.undo_journals(), "no journalling service — the arm was vacuous"


@pytest.mark.slow
def test_undo_journal_survives_a_repair_that_raises():
    """`_rewind` runs on EVERY exit including an exception mid-repair, so the rollback has to too —
    and `resume_undo` must fire even when `_rewind` itself raises, or the service would silently
    ignore every later commit. Force a raise from `plan` and assert the schedule survives and the
    service is neither suspended nor holding an open journal."""
    res = run(_congested(lam=700.0, horizon=360.0))
    led = ReservationLedger(res.config)
    for it in res.intents:
        if it.accepted and it.volumes:
            led.commit(it.request.flight_id, it.volumes)
    st = LNSState(res.config, led, list(res.intents),
                  static_terms=res.ledger.static_terminals(), undo_journal=True)
    st.repair_planner.plan(res.intents[0].request, led, res.config)   # bind the service
    before = _ledger_multiset(led)
    occ_before = _occ_answers(st)
    rng = np.random.default_rng(5)
    st.rng = rng
    victims = random_neighborhood(st, 4)
    boom = RuntimeError("planner exploded mid-repair")

    def explode(*a, **kw):
        raise boom

    st.repair_planner.plan = explode
    with pytest.raises(RuntimeError) as ei:
        st.try_repair(victims, rng, 0.0, "premium")
    assert ei.value is boom
    cocc = st.repair_planner._cocc
    assert cocc._undo is None and not cocc._suspended, "journal left open or service left suspended"
    assert _ledger_multiset(led) == before
    assert _occ_answers(st) == occ_before, "occupancy not restored after a mid-repair raise"
