"""Flat claim arena (``planner.astar.claim_arena``) — the storage the pool-less occupancy needs.

The arena replaced both the claim dict and the free-interval pools: a claim is a blocked span, so
removing a flight removes its own claims instead of rebuilding every cell it touched from that cell's
survivors. Checked at three levels, because they fail differently:

* the arena's own operations (growth, swap-remove, compaction, the atomic-add guarantee) — unit
  tests over hand-built batches, where a bug is visible directly rather than as a wrong plan;
* the arena's behaviour on a real schedule — a release must return it to exactly its pre-commit
  state, which is the internal invariant now that no second structure exists to diff against;
* ``blocked_py`` over a partly-released schedule. The EXTERNAL check lives in
  ``test_compiled_occupancy_matches_is_blocked``, which compares that same ``blocked_py`` against
  ``HexOccupancyService.is_blocked`` — an independent implementation that never used the pools.

Multisets, not sequences: a swap-remove reorders a slab and a growth re-homes it, both deliberately,
so insertion order carries no meaning and pinning it would pin something untrue.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.astar.compiled_hex_occupancy import (
    _FIELD_MASK,
    _S0_SHIFT,
    _SPAN_BITS,
    CompiledHexOccupancy,
)

pytest.importorskip("numba", reason="the claim arena is a numba structure")

from freespace_sim.planner.astar.claim_arena import ClaimArena  # noqa: E402


def _arena(n_keys=64, capacity=8):
    return ClaimArena(n_keys, _S0_SHIFT, _SPAN_BITS, _FIELD_MASK, capacity=capacity)


def _pack(s0, s1, fid=0):
    return (s0 << _S0_SHIFT) | (s1 << _SPAN_BITS) | fid


# ------------------------------------------------------------------ the arena's own operations

def test_add_grows_and_keeps_every_claim():
    """Cells that outgrow their slabs repeatedly must still hold every claim, and must not bleed into
    each other's extents when they are re-homed.

    Three interleaved keys, one claim at a time, so every add is a candidate for a growth that moves
    a slab past its neighbours. The assertions are on CONTENT and on capacity having grown past the
    minimum — not on leftover garbage or on a particular offset, both of which are artefacts of when
    compaction last ran rather than of the behaviour under test."""
    a = _arena(capacity=8)
    want = {3: [], 5: [], 8: []}
    for i in range(50):
        k = (3, 5, 8)[i % 3]
        v = _pack(i, i + 1, i)
        want[k].append(v)
        a.add(np.array([k], np.int64), np.array([v], np.int64))
    for k, vs in want.items():
        assert sorted(int(x) for x in a.slab(k)) == sorted(vs), f"key {k} lost or gained claims"
    assert a.n_claims == 50
    assert max(int(a.cap[k]) for k in want) > 4, "no slab ever grew past the minimum"


def test_add_is_atomic_when_the_arena_is_too_small():
    """``add_many`` computes the tail it needs BEFORE writing, so a caller can grow and retry without
    tracking what a partial batch already applied. Call the kernel directly at a capacity that cannot
    fit the batch and assert nothing moved."""
    from freespace_sim.planner.astar.claim_arena import add_many

    a = _arena(capacity=4)
    keys = np.array([1, 1, 1, 1, 1, 1], np.int64)
    vals = np.array([_pack(i, i, i) for i in range(6)], np.int64)
    short = add_many(keys, vals, 6, a.arena, a.start, a.length, a.cap, a.tail, a.garbage)
    assert short > 0, "expected a shortfall at this capacity"
    assert a.n_claims == 0 and int(a.tail[0]) == 0, "a failed add wrote something"
    a.add(keys, vals)                                 # the wrapper grows and retries
    assert sorted(int(x) for x in a.slab(1)) == sorted(int(v) for v in vals)


def test_remove_is_a_swap_and_reports_drift():
    a = _arena()
    vals = [_pack(i, i + 2, i) for i in range(6)]
    a.add(np.full(6, 9, np.int64), np.array(vals, np.int64))
    a.remove(np.array([9, 9], np.int64), np.array([vals[0], vals[3]], np.int64))
    assert sorted(int(x) for x in a.slab(9)) == sorted(vals[1:3] + vals[4:])
    with pytest.raises(ValueError, match="drifted"):
        a.remove(np.array([9], np.int64), np.array([vals[0]], np.int64))


def test_compact_reclaims_growth_garbage_without_changing_answers():
    a = _arena(capacity=8)
    for k in (2, 5, 11):
        for i in range(20):
            a.add(np.array([k], np.int64), np.array([_pack(i, i, k)], np.int64))
    before = {k: sorted(int(x) for x in a.slab(k)) for k in (2, 5, 11)}
    tail_before, garbage_before = int(a.tail[0]), int(a.garbage[0])
    assert garbage_before > 0
    a.compact()
    assert int(a.garbage[0]) == 0
    assert int(a.tail[0]) < tail_before
    assert {k: sorted(int(x) for x in a.slab(k)) for k in (2, 5, 11)} == before


def test_blocked_membership_matches_a_plain_scan():
    a = _arena()
    spans = [(3, 9), (14, 17), (30, 33)]
    a.add(np.full(len(spans), 7, np.int64),
          np.array([_pack(s0, s1, 1) for s0, s1 in spans], np.int64))
    for s in range(0, 40):
        want = any(s0 <= s <= s1 for s0, s1 in spans)
        assert a.blocked(7, s) is want, f"step {s}"


# ------------------------------------------------------------------ against the live structures

def _committed_occupancy():
    """A real, fragmented schedule with an arena, a claim dict and interval pools all live."""
    from freespace_sim.demand import HubRadiusDemand
    from freespace_sim.dss import DSS
    from freespace_sim.mechanism import FCFSMechanism
    from freespace_sim.uss import USS

    cfg = SimConfig(region_size_m=(20000.0, 15000.0), lam_per_hour=3000.0, horizon_s=300.0,
                    planner="astar", seed=0)
    reqs = HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 6, "stripmall_uss": 30},
                           radius_m={"walmart_uss": 6000.0, "stripmall_uss": 3000.0},
                           terminal_radius_m={"walmart_uss": 125.0, "stripmall_uss": 90.0},
                           pads_per_hub=8, return_flights=True).generate(
        cfg, np.random.default_rng(cfg.seed))
    led = ReservationLedger(cfg)
    dss = DSS(ledger=led, mechanism=FCFSMechanism())
    astar = get_planner("astar_ref")
    usses = {u: USS(u, dss, cfg, astar) for u in {r.uss_id for r in reqs}}
    for ev in reqs[:400]:
        usses[ev.uss_id].handle_request(ev)
    cocc = CompiledHexOccupancy(cfg, track_removal=True)
    led.subscribe(cocc.on_commit)
    led.subscribe_release(cocc.on_release)
    fids = []
    for fid, grp in itertools.groupby(led.iter_committed(), key=lambda fv: fv[0]):
        cocc.on_commit(fid, [v for _, v in grp])
        fids.append(fid)
    return cfg, led, cocc, fids


def test_release_returns_the_arena_to_its_pre_commit_state():
    """A commit followed by its release must leave the arena exactly as it was.

    With the claim dict and the interval pools gone the arena IS the occupancy, so there is no second
    structure left to compare it against — this is the internal invariant, and
    ``test_compiled_occupancy_matches_is_blocked`` supplies the external one by checking
    ``blocked_py`` (now an arena scan) against ``HexOccupancyService.is_blocked``, an entirely
    independent implementation that never used the pools."""
    cfg, led, cocc, fids = _committed_occupancy()
    assert cocc._arena.n_claims > 50_000
    before = {int(k): sorted(int(x) for x in cocc._arena.slab(int(k)))
              for k in np.nonzero(cocc._arena.length)[0]}
    victims = fids[: max(1, len(fids) // 4)]
    removed = [(fid, [v for f, v in led.iter_committed() if f == fid]) for fid in victims]
    led.release_many(victims)
    assert cocc._arena.n_claims < sum(len(v) for v in before.values())
    for fid, vols in removed:
        cocc.on_commit(fid, vols)
    after = {int(k): sorted(int(x) for x in cocc._arena.slab(int(k)))
             for k in np.nonzero(cocc._arena.length)[0]}
    assert after == before, "release + re-commit did not restore the arena"


def test_arena_answers_blocked_over_a_partly_released_schedule():
    """``blocked_py`` is the compiled path's oracle and is now an arena scan. Exercise it over a
    schedule that has been partly released, so the slabs carry swap-removed holes, and assert the
    sample is neither all-free nor all-blocked — a constant would otherwise pass."""
    import random

    cfg, led, cocc, fids = _committed_occupancy()
    led.release_many(fids[: max(1, len(fids) // 4)])
    keys = np.nonzero(cocc._arena.length)[0]
    cells = {int(k) >> 1 for k in keys}
    assert len(cells) > 500
    own = set(sorted(cells)[::11])
    random.seed(0)
    checked = blocked = 0
    for c in random.sample(sorted(cells), min(300, len(cells))):
        iq, rem = divmod(c, cocc.rspan * cocc.n_levels)
        ir, L = divmod(rem, cocc.n_levels)
        q, r = iq + cocc.qmin, ir + cocc.rmin
        for s in range(0, cocc.MAXS, 9):
            checked += 1
            blocked += cocc.blocked_py(q, r, L, s, own_cells=own)
    assert checked > 10_000
    assert 0 < blocked < checked


def test_arena_survives_the_lns_reject_path():
    """A rejected LNS task releases its victims, commits repairs, then restores the incumbent. With
    the arena as the occupancy that is release-then-re-commit all the way down — no journal, no
    snapshot — so the invariant is simply that the claim count comes back to where it started."""
    from freespace_sim.planner.lns.neighborhood import random_neighborhood
    from freespace_sim.planner.lns.state import LNSState
    from freespace_sim.sim import run

    res = run(SimConfig(planner="astar", flight_levels_m=(75.0,), airspace_ceiling_m=125.0,
                        lam_per_hour=700.0, horizon_s=360.0, region_size_m=(3000.0, 3000.0),
                        seed=1, max_ground_delay_s=300.0))
    led = ReservationLedger(res.config)
    for it in res.intents:
        if it.accepted and it.volumes:
            led.commit(it.request.flight_id, it.volumes)
    st = LNSState(res.config, led, list(res.intents),
                  static_terms=res.ledger.static_terminals())
    rng = np.random.default_rng(4)
    st.rng = rng
    st.repair_planner.plan(res.intents[0].request, led, res.config)     # bind the service
    cocc = st.repair_planner._cocc
    assert cocc._arena is not None
    rejects = 0
    for _ in range(12):
        victims = random_neighborhood(st, 4)
        if not victims:
            continue
        before = cocc._arena.n_claims
        out = st.try_repair(victims, rng, 0.0, "premium")
        if out.accepted:
            continue
        rejects += 1
        assert cocc._arena.n_claims == before, "a rejected task left claims behind"
    assert rejects, "no task rejected — the revert path was never exercised"
