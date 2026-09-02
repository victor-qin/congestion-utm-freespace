"""SIPP as an LNS repair planner: incremental release for SIPP's two ledger structures, the
static-wall trap that removal introduces, and the A*/SIPP cost-currency gates.

Design record: `context/sipp_lns_plan.md`. Structure of the file mirrors the phases there —
§4.1/§4.2 removal gates first (they fail legibly), then the wiring, then the currency tests.
"""

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.compiled_occupancy import CompiledOccupancy
from freespace_sim.planner.sipp import SIPPPlanner, SafeIntervalIndex
from freespace_sim.types import FlightRequest, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()


def _wall(x=1000.0):
    return Volume4D(box_from_segment(vec(x, -200, 150), vec(x, 200, 150), 40, 400), 0.0, 1e6)


def _req(fid=1, y=0.0):
    return FlightRequest(fid, vec(0, y, 0), vec(2000, y, 0), 0.0)


def _hub_req(fid=1, y=0.0):
    """A hub departure. SIPP binds its SafeIntervalIndex only for terminal flights (the own-column
    transparency overlay), so a plain point-to-point request exercises three structures, not four."""
    from freespace_sim.types import Terminal

    return FlightRequest(fid, vec(0, y, 0), vec(2000, y, 0), 0.0,
                         origin_terminal=Terminal("hub#0", radius=180.0))


def _rows_of(struct, vol):
    """The in-block (q, r, L, s_lo, s_hi) rows one volume rasterizes to — the same producer both
    structures consume, so a test can name a cell they definitely touched."""
    return [row[:5] for row in hg.rasterize_ranges(
        vol, CFG, struct.R, struct.infl_blocked, struct.infl_pad) if row[-1]]


# ------------------------------------------------------------------ §4.1 SafeIntervalIndex

def test_sidx_release_refcounts_shared_cells():
    """Two flights covering the same cells: removing one must NOT free them. The port of
    `test_incremental_release_reference_service_refcounts` onto SIPP's step-keyed index."""
    sidx = SafeIntervalIndex(CFG, track_removal=True)
    sidx.on_commit(1, [_wall()])
    sidx.on_commit(2, [_wall()])
    assert sidx.n_added == 2

    cell = next(iter(sidx.corr))
    step = next(iter(sidx.corr[cell]))
    assert sidx.corr[cell][step] == 2                  # a refcount, not a set membership

    sidx.on_release(1, [_wall()])
    assert step in sidx.corr[cell]                     # flight 2 still holds it
    assert sidx.n_added == 1
    sidx.on_release(2, [_wall()])
    assert not sidx.corr                               # now genuinely free, and the cell key is gone
    assert sidx.n_added == 0


def test_sidx_release_matches_fresh_absorb():
    """After removing one flight, every query must match a fresh index that only saw the survivor —
    including `free_intervals`, which is what the kernel overlay is built from."""
    keep, drop = _wall(3000.0), _wall(1000.0)
    sidx = SafeIntervalIndex(CFG, track_removal=True)
    sidx.on_commit(1, [drop])
    sidx.on_commit(2, [keep])
    sidx.on_release(1, [drop])

    fresh = SafeIntervalIndex(CFG)
    fresh.on_commit(2, [keep])

    own = frozenset()
    R = hg.circumradius(CFG)
    for x in (1000.0, 3000.0):
        q, r = hg.enu_to_axial(x, 0.0, R)
        for dq in (-1, 0, 1):
            for level in range(CFG.n_levels):
                for s in (0, 5, 50, 900):
                    assert (sidx.cell_blocked(q + dq, r, level, s, own, True)
                            == fresh.cell_blocked(q + dq, r, level, s, own, True)), (x, dq, level, s)
                assert (sidx.free_intervals(q + dq, r, level, own, 0, 1000, True)
                        == fresh.free_intervals(q + dq, r, level, own, 0, 1000, True))
    assert sidx.n_added == fresh.n_added == 1


def test_sidx_release_reverses_the_full_span_under_a_raised_evict_floor():
    """`SafeIntervalIndex.evict_before` reclaims nothing, so a release must decrement the FULL
    recorded span. Clamping it (as the A* twin does, where `evict_before` really deletes and the
    insert clamps too) would strand every step below the floor as a phantom block."""
    sidx = SafeIntervalIndex(CFG, track_removal=True)
    sidx.on_commit(1, [_wall()])
    cell = next(iter(sidx.corr))
    assert min(sidx.corr[cell]) < 400                 # there is something below the floor to strand

    sidx.evict_before(400)
    sidx.on_release(1, [_wall()])
    assert not sidx.corr and not sidx.cols            # no residue below the floor
    assert sidx.n_added == 0


def test_sidx_reset_drops_the_journal():
    """`reset()` (the shrink-rebuild path) discards the structures the journal describes, so the
    journal must go with them — otherwise the next release decrements counts a fresh absorb just
    rebuilt. Defensive: under `incremental_release=True` this path is unreachable, because
    `on_release` keeps `n_added` in lockstep and `ledger.release` delegates to `release_many`
    whenever a release subscriber exists."""
    keep, drop = _wall(3000.0), _wall(1000.0)
    sidx = SafeIntervalIndex(CFG, track_removal=True)
    sidx.on_commit(1, [drop])
    sidx.reset()
    assert not sidx._rows
    sidx.on_commit(1, [drop])                          # rebuild, as `_absorb` would
    sidx.on_commit(2, [keep])
    sidx.on_release(1, [drop])

    fresh = SafeIntervalIndex(CFG)
    fresh.on_commit(2, [keep])
    assert sidx.corr.keys() == fresh.corr.keys()
    assert all(sidx.corr[c] == dict.fromkeys(fresh.corr[c], 1) for c in fresh.corr)


def test_sidx_track_removal_off_is_the_original_set_shape():
    """Flag off ⇒ byte-for-byte the old structures, so the non-LNS path pays nothing."""
    sidx = SafeIntervalIndex(CFG)
    sidx.on_commit(1, [_wall()])
    assert all(isinstance(v, set) for v in sidx.corr.values())
    assert not sidx._rows


# ------------------------------------------------------------------ §4.2 CompiledOccupancy

def test_cocc_release_matches_fresh_absorb():
    """The pool cannot be un-split in place, so removal resets each touched cell and re-applies the
    survivors. Every query must match a fresh pool that only saw the survivor."""
    keep, drop = _wall(3000.0), _wall(1000.0)
    cocc = CompiledOccupancy(CFG, track_removal=True)
    cocc.on_commit(1, [drop])
    cocc.on_commit(2, [keep])
    cocc.on_release(1, [drop])

    fresh = CompiledOccupancy(CFG)
    fresh.on_commit(2, [keep])

    R = hg.circumradius(CFG)
    for x in (1000.0, 3000.0):
        q, r = hg.enu_to_axial(x, 0.0, R)
        for dq in (-1, 0, 1):
            for level in range(CFG.n_levels):
                assert (cocc.free_intervals_py(q + dq, r, level, 0, cocc.MAXS)
                        == fresh.free_intervals_py(q + dq, r, level, 0, cocc.MAXS)), (x, dq, level)
    assert cocc.n_added == fresh.n_added == 1


def test_cocc_release_treats_identical_spans_as_a_multiset():
    """Equal spans are fungible; their multiplicity is not."""
    wall = _wall()
    cocc = CompiledOccupancy(CFG, track_removal=True)
    cocc.on_commit(1, [wall])
    cocc.on_commit(2, [wall])
    q, r, level, s_lo, _s_hi = _rows_of(cocc, wall)[0]
    c = cocc.cell_id(q, r, level)
    step = max(0, s_lo)
    assert not any(lo <= step <= hi for lo, hi in cocc.free_intervals_py(q, r, level, 0, cocc.MAXS))

    cocc.on_release(1, [wall])
    assert not any(lo <= step <= hi                     # the equal claim from flight 2 survives
                   for lo, hi in cocc.free_intervals_py(q, r, level, 0, cocc.MAXS))
    cocc.on_release(2, [wall])
    assert any(lo <= step <= hi for lo, hi in cocc.free_intervals_py(q, r, level, 0, cocc.MAXS))
    assert cocc.n_added == 0
    assert c not in cocc._claims                        # empty claim lists are cleaned up


def test_cocc_release_does_not_unwall_a_static_terminal():
    """THE trap. `reset_cell`'s blank slate is fully FREE, but a walled cell's is fully BLOCKED, and
    the claim journal only describes commit-derived blocks. The bind order makes it unavoidable:
    `_absorb` records claims BEFORE `subscribe_static` replays the hubs, so walled cells carry
    claims no "skip walled cells" guard can prevent."""
    from freespace_sim.types import Terminal

    center, term = vec(1000.0, 0.0, 0.0), Terminal("hub#0", radius=180.0)
    wall = _wall(1000.0)
    cocc = CompiledOccupancy(CFG, track_removal=True)
    cocc.on_commit(1, [wall])                     # claims recorded first — as `_absorb` does
    cocc.register_static_terminal(center, term)   # ...and the hub walled afterwards
    walled = [c for c in (cocc.cell_id(q, r, L) for (q, r) in hg.terminal_cells(center, term, CFG)
                          for L in range(cocc.nlevels)) if c >= 0]
    assert walled
    touched = [c for c in walled if c in cocc._claims]
    assert touched, "the fixture must overlap the hub, or it is not exercising the trap"

    cocc.on_commit(2, [wall])                     # a SECOND owner of the same walled cells
    cocc.on_release(1, [wall])
    for c in touched:
        assert cocc.iv_lo[c] > cocc.iv_hi[c], f"cell {c} was silently unwalled by the rebuild"

    # And the re-wall must not discard the OTHER owner's claims: flight 2 still holds journal rows
    # pointing at these cells, so dropping them here KeyErrors when flight 2 is released. (This is
    # not hypothetical — it is what the first implementation did, and it died on the first
    # density_faa iteration.)
    cocc.on_release(2, [wall])
    for c in touched:
        assert cocc.iv_lo[c] > cocc.iv_hi[c]
    assert cocc.n_added == 0


def test_cocc_reset_cell_reclaims_overflow_slots():
    """`_alloc` is otherwise a pure bump allocator, so under LNS — which resets and re-applies the
    same hot cells every iteration — the pool would grow without bound for a static working set."""
    cocc = CompiledOccupancy(CFG, track_removal=True)
    base = cocc.nslots
    for _ in range(500):
        cocc.block_range(3, 100, 200)
        cocc.block_range(3, 400, 500)
        cocc.reset_cell(3)
        cocc.block_range(3, 100, 200)
        cocc.reset_cell(3)
    assert cocc.nslots <= base + 4                 # reused, not leaked
    cocc.block_range(3, 100, 200)                  # and recycled slots still answer correctly
    assert not any(lo <= 150 <= hi for lo, hi in _chain(cocc, 3))
    assert any(lo <= 99 <= hi for lo, hi in _chain(cocc, 3))


def _chain(cocc, c):
    """Cell ``c``'s live (lo, hi) free intervals, walked straight off the pool. `free_intervals_py`
    takes (q, r, L) and this test blocks a raw slot id, so the chain walk is the honest reader."""
    out, slot = [], c
    while slot != -1:
        lo, hi = int(cocc.iv_lo[slot]), int(cocc.iv_hi[slot])
        if lo <= hi:
            out.append((lo, hi))
        slot = int(cocc.iv_nxt[slot])
    return out


def test_cocc_reset_drops_the_journal():
    """Same defensive gate as the SafeIntervalIndex one: a journal surviving `reset()` would
    re-block spans the fresh absorb already applied."""
    keep, drop = _wall(3000.0), _wall(1000.0)
    cocc = CompiledOccupancy(CFG, track_removal=True)
    cocc.on_commit(1, [drop])
    cocc.reset()
    assert not cocc._rows and not cocc._claims
    cocc.on_commit(1, [drop])
    cocc.on_commit(2, [keep])
    cocc.on_release(1, [drop])

    fresh = CompiledOccupancy(CFG)
    fresh.on_commit(2, [keep])
    R = hg.circumradius(CFG)
    q, r = hg.enu_to_axial(1000.0, 0.0, R)
    assert (cocc.free_intervals_py(q, r, 0, 0, cocc.MAXS)
            == fresh.free_intervals_py(q, r, 0, 0, cocc.MAXS))


def test_cocc_records_the_clamped_span_so_release_is_exact():
    """`_record` journals what `block_range` will ACTUALLY apply, not the raw rasterized span. A
    volume can outlive the pool horizon; recording it raw would pack a too-large `s_hi` and replay
    every SURVIVING claim in that cell at a garbage span."""
    cocc = CompiledOccupancy(CFG, track_removal=True)
    cocc._claims.clear()
    cocc._record(7, -5, cocc.MAXS + 1000, None)
    packed = cocc._claims[7][0]
    from freespace_sim.planner.compiled_occupancy import _FIELD_MASK, _SPAN_BITS
    assert (packed >> _SPAN_BITS, packed & _FIELD_MASK) == (0, cocc.MAXS)


# ------------------------------------------------------------------ §4.3 wiring

def test_sipp_subscribes_release_hooks_for_every_structure_it_binds():
    """The count §0 of the plan turns on — and it is FLIGHT-DEPENDENT, which the plan did not say.

    Compiled A* always binds three release-hooked structures. SIPP binds three for a point-to-point
    flight and a fourth (`_sidx`) only for a TERMINAL one, because the SafeIntervalIndex exists to
    build the own-column transparency overlay (`sipp.py`'s `if fixed and (o_term or d_term)`). So
    SIPP's commit-side headwind is 4:3 on a hub scenario like density_faa, where every leg has a
    terminal, and 3:3 — i.e. none — on point-to-point traffic.

    Before this branch every one of these was 0 for SIPP's own two, and each destroy rebuilt them
    from the whole ledger."""
    astar = AStarPlanner(incremental_release=True)
    led_a = ReservationLedger(CFG)
    astar.plan(_req(), led_a, CFG)
    assert len(led_a._release_subs) == 3           # _svc, _tcap, _cocc

    led_p2p = ReservationLedger(CFG)
    SIPPPlanner(incremental_release=True).plan(_req(), led_p2p, CFG)
    assert len(led_p2p._release_subs) == 3         # _svc, _tcap, _scocc — no overlay needed

    led_hub = ReservationLedger(CFG)
    SIPPPlanner(incremental_release=True).plan(_hub_req(), led_hub, CFG)
    assert len(led_hub._release_subs) == 4         # ...plus _sidx


def test_sipp_release_keeps_n_added_in_lockstep_so_no_rebuild_fires():
    """The whole point of Phase 1: a destroy must not trip the shrink tripwire in
    `_sipp_index`/`_scompiled_occ`, whose healing path is a full `_absorb` of the ledger."""
    led = ReservationLedger(CFG)
    sipp = SIPPPlanner(incremental_release=True)
    sipp.evict_floor = 0.0
    for fid in (1, 2, 3):
        it = sipp.plan(_hub_req(fid, y=400.0 * fid), led, CFG)   # hub legs, so `_sidx` binds too
        assert it.accepted
        led.commit(fid, it.volumes)
    assert sipp._sidx.n_added == sipp._scocc.n_added == led.n_volumes

    led.release_many([2])
    assert sipp._sidx.n_added == sipp._scocc.n_added == led.n_volumes, "tripwire would fire"


def test_sipp_plan_never_reports_a_previous_flights_read_set():
    """`_fallback` runs `AStarPlanner.plan`, which SETS `last_envelope`. Without the reset at the top
    of `SIPPPlanner.plan`, the NEXT plan would file the fallback flight's read set under its own name
    — and a DROP coordinator would test the wrong region, read clean, and merge a genuinely stale
    repair. `verify` cannot catch that: the symptom is a worse cost, not a conflict.

    Now that SIPP records its own envelope this is no longer "None vs not-None"; the envelope must
    describe THIS flight. Two far-apart corridors make that checkable by geometry."""
    led = ReservationLedger(CFG)
    sipp = SIPPPlanner()
    sipp.record_envelope = True

    # Stand in for the fallback path: A*'s search, A*'s envelope, for a flight way off to one side.
    AStarPlanner.plan(sipp, FlightRequest(1, vec(0, 4000, 0), vec(2000, 4000, 0), 0.0), led, CFG)
    stale = sipp.last_envelope
    assert stale is not None and stale.xy is not None

    fresh_intent = sipp.plan(_req(2, y=0.0), led, CFG)
    fresh = sipp.last_envelope
    assert fresh_intent.accepted
    assert fresh is not None and fresh is not stale, "the fallback's envelope object survived"
    # Disjoint in y: flight 1 flew at y=4000, flight 2 at y=0. A leak would report the former.
    assert fresh.xy[3] < stale.xy[1], f"envelope still describes the previous flight: {fresh.xy}"


def test_sipp_record_envelope_off_leaves_none():
    """The reset must also hold when nothing is recorded, or an LNS run that toggles the flag mid-life
    would keep serving the last recorded envelope forever."""
    led = ReservationLedger(CFG)
    sipp = SIPPPlanner()
    sipp.record_envelope = True
    sipp.plan(_req(1), led, CFG)
    assert sipp.last_envelope is not None
    sipp.record_envelope = False
    sipp.plan(_req(2, y=600.0), led, CFG)
    assert sipp.last_envelope is None


def test_sipp_honors_evict_floor_for_its_own_structures():
    """Out-of-order LNS repair needs the full-horizon occupancy; the floor must reach SIPP's two."""
    led = ReservationLedger(CFG)
    sipp = SIPPPlanner()
    sipp.evict_floor = 0.0
    from dataclasses import replace

    sipp.plan(replace(_hub_req(1), t_request=900.0, t_departure=900.0), led, CFG)
    assert sipp._sidx.evicted_before == 0
    assert sipp._scocc.evicted_before == 0


# ------------------------------------------------------------------ §6 currency

def _replay_costs(n_flights, seed, lam):
    """Plan each request with A* and SIPP against the SAME A*-committed ledger."""
    from freespace_sim.demand import UniformPoissonDemand

    cfg = SimConfig(region_size_m=(6000.0, 6000.0), lam_per_hour=lam, horizon_s=600.0, seed=seed)
    reqs = UniformPoissonDemand().generate(cfg, np.random.default_rng(cfg.seed))[:n_flights]
    led = ReservationLedger(cfg)
    astar, sipp = AStarPlanner(), SIPPPlanner()
    rows = []
    for rq in reqs:
        a = astar.plan(rq, led, cfg)
        s = sipp.plan(rq, led, cfg)
        rows.append((a.accepted, s.accepted, a.cost, s.cost))
        if a.accepted:
            led.commit(rq.flight_id, a.volumes)
    return rows


def test_sipp_and_astar_agree_on_unimpeded_cost():
    """Licenses keeping the A* unimpeded ruler for a SIPP repair: the ruler world is EMPTY (walls
    only, nothing ever committed), and both planners are exact optimizers of the same weighted cost,
    so they must return the same number. This is the whole argument for not paying SIPP prices to
    compute a number A* already computes faster in that regime."""
    from freespace_sim.demand import UniformPoissonDemand

    cfg = SimConfig(region_size_m=(6000.0, 6000.0), lam_per_hour=400.0, horizon_s=600.0, seed=0)
    reqs = UniformPoissonDemand().generate(cfg, np.random.default_rng(cfg.seed))[:15]
    assert reqs
    free = ReservationLedger(cfg)                 # nothing is ever committed: the ruler world
    astar, sipp = AStarPlanner(), SIPPPlanner()
    for rq in reqs:
        a, s = astar.plan(rq, free, cfg), sipp.plan(rq, free, cfg)
        assert a.accepted == s.accepted
        if a.accepted:
            assert a.cost == pytest.approx(s.cost, abs=1e-9), rq.flight_id


def test_sipp_and_astar_agree_on_a_congested_ledger():
    """The gap G4 named: no test compared them with traffic committed. Routes may differ (ties break
    differently); costs may not — LNS accepts on any strict improvement."""
    rows = _replay_costs(60, seed=1, lam=400.0)
    assert sum(a for a, _, _, _ in rows) > 20, "fixture is not congested enough to be a test"
    for acc_a, acc_s, ca, cs in rows:
        assert acc_a == acc_s
        if acc_a:
            assert ca == pytest.approx(cs, abs=1e-6)


# ------------------------------------------------------------------ §9 end-to-end gates

def _congested(lam=700.0, horizon=360.0, seed=1):
    """The delay-dominated regime `tests/test_lns.py` uses: every flight admitted, most held.
    A saturated world (binding `max_ground_delay_s`) makes PP repair fail wholesale instead."""
    return SimConfig(
        planner="astar", flight_levels_m=(75.0,), airspace_ceiling_m=125.0,
        lam_per_hour=lam, horizon_s=horizon,
        region_size_m=(3000.0, 3000.0), seed=seed, max_ground_delay_s=300.0,
    )


def _sipp_lns(res, **kw):
    from freespace_sim.planner.lns import LNSConfig, run_lns

    return run_lns(res.config, res.ledger, res.intents,
                   LNSConfig(**{"seed": 7, "neighborhood_size": 4, "log_every": 0,
                                "repair_planner": "sipp", **kw}))


def _traj_key(out):
    return [(r["op"], r["n"], tuple(r["victims"]), r["accepted"], r["reason"],
             round(r["cost_old"], 6),
             None if r["cost_new"] is None else round(r["cost_new"], 6))
            for r in out.trajectory]


@pytest.mark.parametrize("repair_planner", ("astar", "astar_ref", "sipp", "sipp_ref"))
def test_result_preserves_the_repair_planner_registry_name(repair_planner):
    """Compiled/reference arms share implementation classes, so the result must retain the
    validated registry key rather than reverse-mapping the planner object's type."""
    from freespace_sim.planner.lns import LNSConfig, run_lns

    out = run_lns(
        CFG, ReservationLedger(CFG), [],
        LNSConfig(max_iterations=0, log_every=0, repair_planner=repair_planner),
    )
    assert out.repair_planner == repair_planner
    assert out.summary()["repair_planner"] == repair_planner


@pytest.mark.slow
def test_sipp_incremental_release_matches_rebuild():
    """THE Phase 1 gate, and the direct analogue of `test_lns_incremental_release_matches_rebuild`.

    There is no byte-parity gate against the A* LNS run — SIPP breaks cost ties differently, so the
    trajectory diverges from iteration 1. This is the gate that actually pins the new removal
    machinery: the O(victims) `on_release` path against the reset+reabsorb reference, same seed,
    identical trajectory and identical final schedule."""
    from analysis.ab_column_clear import _intent_digest
    from freespace_sim.sim import run

    fast = _sipp_lns(run(_congested()), max_iterations=40, incremental_release=True)
    slow = _sipp_lns(run(_congested()), max_iterations=40, incremental_release=False)
    assert _traj_key(fast) == _traj_key(slow)
    assert [_intent_digest(i) for i in fast.intents] == [_intent_digest(i) for i in slow.intents]
    assert fast.verified and slow.verified
    assert fast.cost_after == pytest.approx(slow.cost_after)
    assert fast.n_accepted > 0, "a vacuous run would make this gate meaningless"


@pytest.mark.slow
def test_sipp_repair_is_conflict_free_under_continuous_verification():
    """Phase 2: every accepted iteration replayed independently. `verify_every=1` is the strongest
    form — a repair that files a conflicting schedule fails on the iteration that produced it."""
    from freespace_sim.sim import run

    out = _sipp_lns(run(_congested()), max_iterations=30, verify_every=1)
    assert out.verified and out.n_accepted > 0


def test_the_repair_planner_reaches_a_parallel_worker():
    """Pins the config -> WorkerSpec -> LNSState.replica chain, because a worker that builds A* while
    the coordinator believes it is running SIPP differs SILENTLY: `verify` checks 4D conflicts only,
    so such a run still reports itself verified.

    Deliberately a chain test rather than an end-to-end parity gate, and the reason is worth writing
    down. The obvious gate — run parallel and sequential with the same seed and compare — only works
    where the parallel engine is deterministic, i.e. at `search_workers=1`. But the merged
    `run_lns_parallel` DELEGATES to the sequential engine below an effective width of two ("a private
    replica cannot add concurrency in that case"), so at m=1 it compares the sequential loop with
    itself and never spawns a worker at all. At m>=2 there is no byte-parity to assert against. An
    earlier version of this test papered over that by comparing the parallel run to a sequential A*
    run — which differ for engine reasons whatever planner the worker used, so it passed against a
    build with the forward deleted entirely. Each link is checked directly instead."""
    from freespace_sim.planner.lns import LNSConfig
    from freespace_sim.planner.lns.parallel import WorkerSpec
    from freespace_sim.planner.lns.solver import _validate_lns_config
    from freespace_sim.planner.lns.state import LNSState
    from freespace_sim.sim import run

    # link 1: the knob survives normalization
    assert _validate_lns_config(LNSConfig(repair_planner="sipp")).repair_planner == "sipp"

    # link 2: WorkerSpec carries it, and is picklable with it (a planner OBJECT would not be)
    import pickle

    spec = WorkerSpec(
        neighborhood_size=4, accept_epsilon=0.0, repair_order="premium", max_walks=10,
        map_max_cells=4096, turnaround_s=None, frozen_flight_ids=frozenset(),
        movable_uss_ids=None, incremental_release=True, kernel_log2_min=None,
        repair_planner="sipp",
    )
    assert pickle.loads(pickle.dumps(spec)).repair_planner == "sipp"

    # link 3: replica builds what the spec names — the link whose absence is invisible at runtime
    res = run(_congested(lam=200.0, horizon=120.0))
    movable = [i.request.flight_id for i in res.intents if i.accepted]
    replica = LNSState.replica(
        res.config, res.intents,
        static_terms=res.ledger.static_terminals(),
        unimpeded_cost=dict.fromkeys(movable, 0.0),
        repair_planner_name=spec.repair_planner,
    )
    assert type(replica.repair_planner).__name__ == "SIPPPlanner"
    assert replica.repair_planner.evict_floor == 0.0
    # and the default still builds A*, so the forward is what selects it — not a global flip
    default = LNSState.replica(
        res.config, res.intents,
        static_terms=res.ledger.static_terminals(),
        unimpeded_cost=dict.fromkeys(movable, 0.0),
    )
    assert type(default.repair_planner).__name__ == "AStarPlanner"

    # link 4: `_worker_main` actually PASSES the spec's name through. Links 1-3 all hold with this
    # forward deleted — the test would be calling `replica` itself and never notice — so spy on the
    # real call site rather than trust it.
    from freespace_sim.planner.lns import parallel as lns_parallel

    seen = {}
    orig = LNSState.replica

    class _StopHere(RuntimeError):
        pass

    def _spy(cls_cfg, cls_intents, **kw):
        seen.update(kw)
        raise _StopHere

    class _Conn:
        def send(self, _msg):
            pass

    lns_parallel.LNSState.replica = staticmethod(_spy)
    try:
        lns_parallel._worker_main(_Conn(), res.config, res.intents,
                                  res.ledger.static_terminals(),
                                  dict.fromkeys(movable, 0.0), spec, 0)
    except _StopHere:
        pass
    finally:
        lns_parallel.LNSState.replica = orig
    assert seen.get("repair_planner_name") == "sipp", \
        f"_worker_main did not forward the spec's repair planner (saw {seen.get('repair_planner_name')!r})"


def test_the_coordinator_puts_the_repair_planner_in_the_worker_spec():
    """Link 5, and the one the chain test above cannot reach: `run_lns_parallel` builds the
    `WorkerSpec` itself, so deleting `repair_planner=lns.repair_planner` there leaves every other
    link intact and the whole fleet silently runs A*.

    Intercepted at the POOL boundary rather than by spying on `WorkerSpec` — the spec is pickled
    into each spawned child, and a test double in that graph breaks spawn serialization. Stopping at
    the pool also means no processes are created, so this stays a fast test."""
    from freespace_sim.planner.lns import LNSConfig, parallel as lns_parallel
    from freespace_sim.planner.lns.parallel import run_lns_parallel
    from freespace_sim.sim import run

    seen = []

    class _StopHere(RuntimeError):
        pass

    def _capture(cfg, intents, static_terms, unimpeded_cost, spec, n_workers):
        seen.append(spec)
        raise _StopHere

    res = run(_congested(lam=200.0, horizon=120.0))
    orig = lns_parallel.LNSWorkerPool
    lns_parallel.LNSWorkerPool = _capture
    try:
        run_lns_parallel(res.config, res.ledger, res.intents,
                         LNSConfig(seed=7, neighborhood_size=4, log_every=0, max_iterations=4,
                                   search_workers=2, parallel_mode="sync",
                                   repair_planner="sipp"))
    except _StopHere:
        pass
    finally:
        lns_parallel.LNSWorkerPool = orig
    assert seen, "the coordinator never reached the pool — the run did not go parallel"
    assert seen[0].repair_planner == "sipp", \
        f"coordinator shipped {seen[0].repair_planner!r} instead of 'sipp'"


def test_bad_repair_planner_name_leaves_the_ledger_intact():
    """`_new_repair_planner` runs AFTER `ledger.detach_subscribers()`, whose epoch bump is not
    reversible — so a bad name must be rejected by `_validate_lns_config` while the caller's ledger
    is still whole. Without that ordering the typo silently costs the caller every subscriber."""
    from freespace_sim.planner.lns import LNSConfig
    from freespace_sim.planner.lns.solver import _validate_lns_config, run_lns
    from freespace_sim.sim import run

    with pytest.raises(ValueError, match="not a supported LNS repair planner"):
        _validate_lns_config(LNSConfig(repair_planner="sipp2"))

    res = run(_congested(lam=200.0, horizon=120.0))
    subs = (len(res.ledger._observers), len(res.ledger._release_subs), res.ledger.epoch)
    assert subs[0] > 0, "the baseline run must leave subscribers to lose"
    with pytest.raises(ValueError, match="not a supported LNS repair planner"):
        run_lns(res.config, res.ledger, res.intents,
                LNSConfig(repair_planner="sipp2", max_iterations=5, log_every=0))
    assert (len(res.ledger._observers), len(res.ledger._release_subs),
            res.ledger.epoch) == subs, "a rejected name stripped the caller's ledger"


def test_a_sipp_baseline_is_accepted_and_ruled_by_astar():
    """§6's follow-up, now that the two cost-parity gates above are evidence rather than hope.

    `_REPRODUCIBLE_PLANNERS` gates the BASELINE's planner (`cfg.planner`), not the repair planner —
    so before this it refused a `planner="sipp"` run outright while silently permitting the opposite
    mixing. Both directions are legitimate for the same reason: A* and SIPP are exact optimizers of
    the same weighted cost, so the A* unimpeded ruler measures a SIPP incumbent's delay correctly."""
    from freespace_sim.planner.lns import LNSConfig, run_lns
    from freespace_sim.sim import run

    cfg = _congested(lam=400.0, horizon=240.0)
    res = run(cfg, planner_name="sipp")
    assert res.config.planner in ("sipp", "astar")
    from dataclasses import replace as dc_replace
    sipp_cfg = dc_replace(res.config, planner="sipp")
    out = run_lns(sipp_cfg, res.ledger, res.intents,
                  LNSConfig(seed=7, neighborhood_size=4, log_every=0, max_iterations=20,
                            repair_planner="sipp"))
    assert out.verified
    # delay() must stay non-negative for every movable flight: a currency mismatch shows up here
    # first, as an incumbent "cheaper than unimpeded".
    assert out.cost_after <= out.cost_before
