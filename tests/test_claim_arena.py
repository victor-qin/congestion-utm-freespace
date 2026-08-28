"""Flat claim arena (``planner.astar.claim_arena``) — the storage the pool-less occupancy needs.

Step 1 of that work maintains the arena ALONGSIDE the existing claim dict and interval pools, so its
one contract is that all three answer the same question the same way. That is checked at three
levels, because they fail differently:

* the arena's own operations (growth, swap-remove, compaction, the atomic-add guarantee) — unit
  tests over hand-built batches, where a bug is visible directly rather than as a wrong plan;
* the arena against ``_claims`` on a real schedule, through commits AND releases;
* the arena against the interval pools through ``blocked_py``, which is what the kernel actually
  reads and therefore the only comparison that says the substitution would be safe.

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


def test_arena_tracks_the_claim_dict_through_commit_and_release():
    cfg, led, cocc, fids = _committed_occupancy()
    assert cocc._arena is not None and cocc._arena.n_claims > 50_000
    assert cocc.arena_matches_claims(), "arena and claim dict differ after the commits"
    led.release_many(fids[: max(1, len(fids) // 4)])
    assert cocc._arena.n_claims > 0
    assert cocc.arena_matches_claims(), "arena and claim dict differ after a release"


def test_arena_pools_and_claim_dict_answer_blocked_identically():
    """The comparison that says the substitution is safe: three independent implementations of the
    same question — free-interval pools, the claim dict, and the flat arena — over a schedule that
    has been partly released, so the pools carry split intervals and empty ``lo > hi`` slots and the
    slabs carry swap-removed holes."""
    import random

    cfg, led, cocc, fids = _committed_occupancy()
    led.release_many(fids[: max(1, len(fids) // 4)])
    cells = {key >> 1 for key in cocc._claims}
    assert len(cells) > 500
    own = set(sorted(cells)[::11])
    random.seed(0)
    checked = blocked = 0
    for c in random.sample(sorted(cells), min(300, len(cells))):
        iq, rem = divmod(c, cocc.rspan * cocc.n_levels)
        ir, L = divmod(rem, cocc.n_levels)
        q, r = iq + cocc.qmin, ir + cocc.rmin
        for s in range(0, cocc.MAXS, 9):
            pool = cocc.blocked_py(q, r, L, s, own_cells=own)
            claims = cocc.blocked_py_claims(q, r, L, s, own_cells=own)
            arena = cocc.blocked_py_arena(q, r, L, s, own_cells=own)
            assert pool == claims == arena, \
                f"pool={pool} claims={claims} arena={arena} at (q={q},r={r},L={L},s={s})"
            checked += 1
            blocked += pool
    assert checked > 10_000
    assert 0 < blocked < checked, "sample is all-free or all-blocked — constants would pass this"


def test_arena_survives_the_lns_reject_path():
    """A rejected LNS task rolls the pools and the claim dict back from the #120 undo journal. The
    arena has no snapshot, so it re-derives — and if that re-derivation were wrong or skipped, the
    arena would drift from the dict on the first reject and never recover."""
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
        rejects += int(not st.try_repair(victims, rng, 0.0, "premium").accepted)
        assert cocc.arena_matches_claims(), "arena drifted from the claim dict during the loop"
    assert rejects, "no task rejected — the rollback path was never exercised"
