"""Seams that a parallel (DROP-LNS) worker stands on.

A worker holds a PRIVATE replica of the incumbent and is told "the incumbent moved" by delta,
so three things have to be exact before any pool exists:

* ``LNSState.replica`` must reproduce the state it copies — same movable set, same delay ruler,
  same claim index, same ledger content — and must forward every keyword that changes what a
  repair is ALLOWED to do (the anchor guard, the USS-restriction hooks, the occupancy path).
* ``LNSState.apply_delta`` must move the ledger AND the in-memory views together, reversibly.
* ``RepairOutcome`` must carry the repaired intents and their read sets back out, since the
  coordinator — not the worker — owns the incumbent.

See ``context/lns_plan.md`` and the DROP-LNS design record for why each of these is load-bearing.
"""

import hashlib
import multiprocessing as mp

import numpy as np
import pytest

from analysis.ab_column_clear import _intent_digest
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.lns.state import LNSState
from freespace_sim.sim import run
from tests.test_lns import _congested, _ledger_multiset


def _claim_digest(state) -> str:
    """Order-insensitive digest of the destroy heuristics' claim index."""
    rows = sorted(
        f"{cell}:{sorted(entries)}" for cell, entries in state._claims.items() if entries
    )
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def _state_digest(state) -> tuple:
    """Everything a replica must reproduce about the state it copied."""
    return (
        round(state.total_cost, 6),
        tuple(state.movable_ids()),
        tuple(round(state._unimp_cost[f], 6) for f in state.movable_ids()),
        _ledger_multiset(state.ledger),
        _claim_digest(state),
        tuple(_intent_digest(i) for i in state.final_intents()),
    )


# ------------------------------------------------------------------ replica fidelity
@pytest.mark.slow
def test_replica_reproduces_the_state_it_copies():
    res = run(_congested(lam=400.0, horizon=240.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    rep = LNSState.replica(res.config, base.final_intents(),
                           static_terms=base.static_terms,
                           unimpeded_cost=dict(base._unimp_cost))
    assert _state_digest(rep) == _state_digest(base)
    # The replica owns a DIFFERENT ledger — otherwise the two would share a mutation surface.
    assert rep.ledger is not base.ledger


@pytest.mark.slow
def test_replica_forwards_the_movable_filters():
    """frozen/uss filters derive _movable; a replica that drops them would let the destroy
    operators select flights the USS-restriction hook exists to protect."""
    res = run(_congested(lam=400.0, horizon=240.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    intents = base.final_intents()
    frozen = frozenset(i.request.flight_id for i in intents if i.accepted)
    frozen = frozenset(sorted(frozen)[:3])
    unimp = {f: c for f, c in base._unimp_cost.items()}

    rep = LNSState.replica(res.config, intents, static_terms=base.static_terms,
                           unimpeded_cost=unimp, frozen_flight_ids=frozen)
    assert not (set(rep.movable_ids()) & frozen)
    for fid in frozen:
        assert not rep.is_movable(fid)


@pytest.mark.slow
def test_replica_forwards_the_anchor_guard():
    """turnaround_s builds _return_anchor. Dropping it disarms try_repair's anchor rejection
    SILENTLY: verify.find_interflight_conflict checks 4D conflicts only, so a schedule that
    re-times an outbound past its return's departure still reports verified."""
    res = run(_congested(lam=400.0, horizon=240.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    intents = base.final_intents()
    unimp = dict(base._unimp_cost)

    armed = LNSState.replica(res.config, intents, static_terms=base.static_terms,
                             unimpeded_cost=unimp, turnaround_s=60.0)
    disarmed = LNSState.replica(res.config, intents, static_terms=base.static_terms,
                                unimpeded_cost=unimp)
    assert armed._turnaround_s == 60.0
    assert disarmed._turnaround_s is None


def test_replica_spawns_no_child_processes():
    """Coexistence rule: a replica is constructed INSIDE a worker, so it must never stand up a
    pool of its own — m search workers x m ruler workers is a fork bomb, not a speedup."""
    res = run(_congested(lam=200.0, horizon=120.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    before = len(mp.active_children())
    LNSState.replica(res.config, base.final_intents(), static_terms=base.static_terms,
                     unimpeded_cost=dict(base._unimp_cost))
    assert len(mp.active_children()) == before


# ------------------------------------------------------------------ the delay ruler seam
@pytest.mark.slow
def test_injected_unimpeded_cost_matches_the_in_process_ruler():
    res = run(_congested(lam=400.0, horizon=240.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    rep = LNSState.replica(res.config, base.final_intents(), static_terms=base.static_terms,
                           unimpeded_cost=dict(base._unimp_cost))
    for fid in base.movable_ids():
        assert rep.delay(fid) == pytest.approx(base.delay(fid))


@pytest.mark.slow
def test_injected_denial_resolves_exactly_like_the_in_process_ruler(monkeypatch):
    """`None` means the ruler denied the flight. __init__ is the ONE owner of the fallback
    (incumbent cost -> zero premium, "never seed a walk"); any other value would make delay()
    a large positive premium and pin every agent-based neighborhood on an unimprovable flight."""
    res = run(_congested(lam=400.0, horizon=240.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    intents = base.final_intents()
    victim = base.movable_ids()[0]

    raw = {f: (None if f == victim else c) for f, c in base._unimp_cost.items()}
    rep = LNSState.replica(res.config, intents, static_terms=base.static_terms,
                           unimpeded_cost=raw)
    assert rep._unimp_cost[victim] == pytest.approx(float(rep.incumbent[victim].cost))
    assert rep.delay(victim) == 0.0          # exactly the "treat as undelayed" contract


@pytest.mark.slow
def test_injected_unimpeded_cost_must_cover_every_movable_flight():
    """A short dict is silent otherwise: delay() KeyErrors mid-walk, potentially an hour in."""
    res = run(_congested(lam=400.0, horizon=240.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    short = dict(base._unimp_cost)
    short.pop(base.movable_ids()[0])
    with pytest.raises(ValueError, match="missing"):
        LNSState.replica(res.config, base.final_intents(), static_terms=base.static_terms,
                         unimpeded_cost=short)


# ------------------------------------------------------------------ apply_delta
@pytest.mark.slow
def test_apply_delta_moves_a_replica_onto_an_accepted_repair_and_back():
    """The worker-sync round trip: a replica told "the incumbent moved" must land byte-exactly
    where the state that did the repair is, and its inverse must restore it exactly."""
    res = run(_congested(lam=400.0, horizon=240.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    intents = base.final_intents()
    unimp = dict(base._unimp_cost)
    rep = LNSState.replica(res.config, intents, static_terms=base.static_terms,
                           unimpeded_cost=unimp)
    at_start = _state_digest(rep)
    assert at_start == _state_digest(base)

    # Find an accepted repair on `base`; its new_intents are what a worker would report.
    accepted = None
    for i in range(60):
        rng = np.random.default_rng(np.random.SeedSequence([7, i]))
        base.rng = rng
        victims = sorted(base.movable_ids())[: 4 + (i % 3)]
        out = base.try_repair(victims, rng)
        if out.accepted:
            accepted = out
            break
    assert accepted is not None, "no accepted repair in 60 tries — pick a denser world"
    assert accepted.new_intents, "the accept return must carry the repaired schedule"

    old = {f: rep.incumbent[f] for f in accepted.new_intents}
    rep.apply_delta(accepted.new_intents)
    assert _state_digest(rep) == _state_digest(base)      # caught up, exactly

    rep.apply_delta(old)
    assert _state_digest(rep) == at_start                 # and the inverse is exact


@pytest.mark.slow
def test_apply_delta_is_a_noop_on_an_empty_change_set():
    res = run(_congested(lam=200.0, horizon=120.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    before = _state_digest(base)
    base.apply_delta({})
    assert _state_digest(base) == before


# ------------------------------------------------------------------ RepairOutcome payload
@pytest.mark.slow
def test_reject_path_carries_no_payload():
    """79% of iterations reject; building the payload for them would be pure waste."""
    res = run(_congested(lam=400.0, horizon=240.0))
    state = LNSState(res.config, res.ledger, res.intents,
                     static_terms=res.ledger.static_terminals())
    rng = np.random.default_rng(0)
    state.rng = rng
    rejected = None
    for i in range(60):
        r = np.random.default_rng(np.random.SeedSequence([11, i]))
        out = state.try_repair(sorted(state.movable_ids())[: 3], r)
        if not out.accepted:
            rejected = out
            break
    assert rejected is not None
    assert rejected.new_intents == {}
    assert rejected.envelopes == ()


@pytest.mark.slow
def test_envelopes_are_recorded_only_when_the_planner_records_them():
    res = run(_congested(lam=400.0, horizon=240.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    intents = base.final_intents()
    unimp = dict(base._unimp_cost)

    # replica() turns record_envelope on; the plain constructor leaves it off.
    rep = LNSState.replica(res.config, intents, static_terms=base.static_terms,
                           unimpeded_cost=unimp)
    assert rep.repair_planner.record_envelope is True
    assert base.repair_planner.record_envelope is False

    rng = np.random.default_rng(np.random.SeedSequence([7, 0]))
    rep.rng = rng
    victims = sorted(rep.movable_ids())[:4]
    out = rep.try_repair(victims, rng)
    if out.accepted:
        assert len(out.envelopes) == len(out.new_intents)
        # A None entry is legal (host-side early denial) and means "always dirty".
        assert all(e is None or hasattr(e, "unbounded") for e in out.envelopes)


def test_replica_planner_is_configured_for_out_of_order_repair():
    """evict_floor = 0.0 is what lets a worker replan victims in ANY priority order; the
    constructor refuses a borrowed planner without it, so replica() must set it."""
    res = run(_congested(lam=200.0, horizon=120.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    rep = LNSState.replica(res.config, base.final_intents(), static_terms=base.static_terms,
                           unimpeded_cost=dict(base._unimp_cost))
    assert rep.repair_planner.evict_floor == 0.0
    assert isinstance(rep.repair_planner, AStarPlanner)


# ==================================================================== the pool + SYNC mode
def _trajectory_key(out):
    """Row-for-row identity of an anytime trajectory, EXCLUDING wall_s.

    Every row carries `wall_s = monotonic() - t0`, so a digest over raw rows can never match
    across two runs — both existing parity helpers in test_lns.py project it out for that reason.
    TODO(rebase): hoist this into test_lns.py and have
    `test_lns_incremental_release_matches_rebuild` and `test_lns_is_deterministic_per_seed` use it
    too; three copies that can drift is how a parity test silently stops testing parity. Kept
    local for now because `victor-qin/lns-efficiency-fixes` owns test_lns.py this week.
    """
    return [(r["iter"], r["op"], r["n"], tuple(r["victims"]), r["accepted"], r["reason"],
             round(r["cost_old"], 6),
             None if r["cost_new"] is None else round(r["cost_new"], 6),
             round(r["incumbent_cost"], 6))
            for r in out.trajectory]


@pytest.mark.slow
def test_one_worker_sync_matches_sequential():
    """THE gate. Not `cost_after` — a cost tie would hide a divergent victim set.

    The subtle failure this pins: `AdaptiveSelector.pick` consumes one draw from the SAME
    generator the destroy operators then read, so a worker that re-seeded from (seed, i) would
    start a draw earlier and pick different victims from iteration 0. The coordinator ships
    `rng.bit_generator.state` instead of the seed precisely to avoid that.
    """
    from freespace_sim.planner.lns import LNSConfig, run_lns
    from freespace_sim.planner.lns.parallel import run_lns_parallel

    cfg = _congested(lam=400.0, horizon=240.0)
    kw = dict(seed=7, neighborhood_size=4, log_every=0, max_iterations=40)

    a = run(cfg)
    seq = run_lns(a.config, a.ledger, a.intents, LNSConfig(**kw))
    b = run(cfg)
    par = run_lns_parallel(b.config, b.ledger, b.intents, LNSConfig(search_workers=1, **kw))

    assert _trajectory_key(par) == _trajectory_key(seq)
    assert [_intent_digest(i) for i in par.intents] == [_intent_digest(i) for i in seq.intents]
    assert par.weights == seq.weights
    assert par.n_accepted == seq.n_accepted and par.verified
    assert seq.n_accepted > 0, "a vacuous run would make this gate meaningless"


@pytest.mark.slow
@pytest.mark.parametrize("m", [2, 4])
def test_multi_worker_stays_feasible_and_writes_the_ledger_back(m):
    """`run_lns`'s contract is that the CALLER's ledger is mutated in place and the returned
    intents supersede it. The coordinator holds its state on that ledger, so apply_delta is the
    write-back — this pins that it actually happened, which a zero-accept run would not."""
    from freespace_sim import verify
    from freespace_sim.planner.lns import LNSConfig
    from freespace_sim.planner.lns.parallel import run_lns_parallel

    res = run(_congested(lam=400.0, horizon=240.0))
    static_terms = res.ledger.static_terminals()
    out = run_lns_parallel(res.config, res.ledger, res.intents,
                           LNSConfig(seed=7, neighborhood_size=4, log_every=0,
                                     max_iterations=40, search_workers=m))
    assert out.n_accepted > 0, "nothing accepted — the write-back assertion would be vacuous"
    assert out.verified
    assert verify.find_interflight_conflict(out.intents, res.config,
                                            static_terminals=static_terms) is None

    # The ledger holds the IMPROVED schedule, not the FCFS one it came in with.
    live = sum(len(i.volumes) for i in out.intents if i.accepted)
    assert res.ledger.n_volumes == live
    assert _ledger_multiset(res.ledger) == _ledger_multiset(
        _replay_ledger(res.config, out.intents, static_terms))
    # and it is handed back clean
    assert res.ledger._observers == [] and res.ledger._release_subs == []


def _replay_ledger(cfg, intents, static_terms):
    from freespace_sim.ledger import ReservationLedger
    led = ReservationLedger(cfg)
    for center, term in static_terms:
        led.register_static_terminal(center, term)
    for it in intents:
        if it.accepted and it.volumes:
            led.commit(it.request.flight_id, it.volumes)
    return led


@pytest.mark.slow
def test_sync_is_deterministic_at_every_worker_count():
    """SYNC is the replication vehicle: same seed, same m, same trajectory, run to run."""
    from freespace_sim.planner.lns import LNSConfig
    from freespace_sim.planner.lns.parallel import run_lns_parallel

    cfg = _congested(lam=400.0, horizon=240.0)
    kw = dict(seed=3, neighborhood_size=4, log_every=0, max_iterations=24, search_workers=3)
    outs = []
    for _ in range(2):
        res = run(cfg)
        outs.append(run_lns_parallel(res.config, res.ledger, res.intents, LNSConfig(**kw)))
    assert _trajectory_key(outs[0]) == _trajectory_key(outs[1])
    assert [_intent_digest(i) for i in outs[0].intents] == \
           [_intent_digest(i) for i in outs[1].intents]


# ==================================================================== pool robustness / config
def test_a_dead_worker_fails_loudly_rather_than_hanging():
    """The documented failure mode of every pool in this repo: an OOM-killed worker that is
    merely waited on HANGS the run. Waiting on proc.sentinel alongside the pipe is what turns
    that into an exception."""
    from freespace_sim.planner.lns.parallel import LNSWorkerPool, WorkerLost, WorkerSpec

    res = run(_congested(lam=200.0, horizon=120.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    spec = WorkerSpec(neighborhood_size=4, accept_epsilon=0.0, repair_order="premium",
                      max_walks=10, map_max_cells=4096, turnaround_s=None,
                      frozen_flight_ids=frozenset(), movable_uss_ids=None,
                      incremental_release=True, kernel_log2_min=None)
    pool = LNSWorkerPool(res.config, base.final_intents(), base.static_terms,
                         dict(base._unimp_cost), spec, 2).start()
    try:
        pool._procs[0].kill()
        with pytest.raises(WorkerLost):
            pool.collect(1, timeout=30.0)
    finally:
        pool.close()


def test_search_workers_and_parallel_mode_are_validated():
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import LNSConfig, run_lns
    from freespace_sim.config import SimConfig

    cfg = SimConfig()
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="search_workers"):
            run_lns(cfg, ReservationLedger(cfg), [],
                    LNSConfig(max_iterations=0, log_every=0, search_workers=bad))
    with pytest.raises(ValueError, match="search_workers"):     # the 4x-cores ceiling
        run_lns(cfg, ReservationLedger(cfg), [],
                LNSConfig(max_iterations=0, log_every=0, search_workers=10_000))
    with pytest.raises(ValueError, match="parallel_mode"):
        run_lns(cfg, ReservationLedger(cfg), [],
                LNSConfig(max_iterations=0, log_every=0, parallel_mode="deta"))


def test_default_config_is_the_sequential_path():
    """search_workers=1 by default: a worker holds a full private replica, so memory is linear
    in workers — the same reason colgen's pricing pool is still defaulted off."""
    from freespace_sim.planner.lns import LNSConfig

    c = LNSConfig()
    assert c.search_workers == 1
    assert c.parallel_mode == "sync"


# ==================================================================== DROP mode
@pytest.mark.slow
def test_drop_at_one_worker_matches_sequential():
    """With one worker nothing can interleave, so every result lands with base_version == the
    current version and DROP's accept rule collapses to the sequential one. A divergence here
    means the staleness branches are firing when they cannot possibly apply."""
    from freespace_sim.planner.lns import LNSConfig, run_lns
    from freespace_sim.planner.lns.parallel import run_lns_parallel

    cfg = _congested(lam=400.0, horizon=240.0)
    kw = dict(seed=7, neighborhood_size=4, log_every=0, max_iterations=40)

    a = run(cfg)
    seq = run_lns(a.config, a.ledger, a.intents, LNSConfig(**kw))
    b = run(cfg)
    drop = run_lns_parallel(b.config, b.ledger, b.intents,
                            LNSConfig(search_workers=1, parallel_mode="drop", **kw))

    assert _trajectory_key(drop) == _trajectory_key(seq)
    assert [_intent_digest(i) for i in drop.intents] == [_intent_digest(i) for i in seq.intents]
    st = drop.parallel_stats
    assert st["n_stale_victims"] == st["n_stale_cost"] == st["n_dirty"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("m", [2, 4])
def test_drop_stays_feasible_and_writes_the_ledger_back(m):
    from freespace_sim import verify
    from freespace_sim.planner.lns import LNSConfig
    from freespace_sim.planner.lns.parallel import run_lns_parallel

    res = run(_congested(lam=400.0, horizon=240.0))
    static_terms = res.ledger.static_terminals()
    out = run_lns_parallel(res.config, res.ledger, res.intents,
                           LNSConfig(seed=7, neighborhood_size=4, log_every=0,
                                     max_iterations=40, search_workers=m,
                                     parallel_mode="drop", verify_every=1))
    assert out.verified
    assert verify.find_interflight_conflict(out.intents, res.config,
                                            static_terminals=static_terms) is None
    assert out.cost_after <= out.cost_before        # monotone: every accept strictly improves
    live = sum(len(i.volumes) for i in out.intents if i.accepted)
    assert res.ledger.n_volumes == live
    # DROP rows stay auditable even though the run is not reproducible.
    assert all("base_version" in r and "worker" in r for r in out.trajectory)


@pytest.mark.slow
def test_drop_rows_cover_every_dispatched_task():
    from freespace_sim.planner.lns import LNSConfig
    from freespace_sim.planner.lns.parallel import run_lns_parallel

    res = run(_congested(lam=400.0, horizon=240.0))
    out = run_lns_parallel(res.config, res.ledger, res.intents,
                           LNSConfig(seed=5, neighborhood_size=4, log_every=0,
                                     max_iterations=24, search_workers=3, parallel_mode="drop"))
    assert len(out.trajectory) == out.n_iterations == 24
    assert sorted(r["iter"] for r in out.trajectory) == list(range(24))


# ==================================================================== _Changelog
@pytest.mark.slow
def test_changelog_diff_revert_and_touched():
    """The wholesale-overwrite path (paper Alg. 2 line 23) is rare end-to-end, so its core — the
    changelog's ability to walk the incumbent back to an arbitrary base — is pinned directly.

    The load-bearing subtlety is that ``revert_to`` must keep the FIRST recorded old per fid.
    Keeping the last would walk a flight back only to an intermediate value, and the "solution"
    handed to the ledger would be one no worker ever planned against.
    """
    from freespace_sim.planner.lns.parallel import _Changelog

    res = run(_congested(lam=200.0, horizon=120.0))
    ints = [i for i in res.intents if i.accepted][:3]
    a, b, c = (i.request.flight_id for i in ints)
    oldA, oldB, midA = ints[0], ints[1], ints[2]

    cl = _Changelog()
    assert cl.version == 0 and cl.diff_since(0) == {}
    cl.record({a: midA}, {a: oldA})            # v1
    cl.record({b: oldB}, {b: oldB})            # v2
    cl.record({a: oldA}, {a: midA})            # v3: a moves AGAIN

    assert cl.version == 3
    assert cl.diff_since(0) == {a: oldA, b: oldB}      # last write wins
    assert cl.diff_since(2) == {a: oldA}
    assert cl.touched_since(0) == {a, b}
    assert cl.touched_since(2) == {a}
    # FIRST old per fid — oldA, not the intermediate midA that v3 recorded.
    assert cl.revert_to(0) == {a: oldA, b: oldB}
    assert cl.revert_to(2) == {a: midA}
    assert cl.boxes_since(0), "removed+added volumes must be reported for the read-set test"
    cl.trim(2)
    assert cl.diff_since(0) == {a: oldA}               # trimmed entries are gone


def test_read_set_is_clean_treats_none_as_dirty():
    """`AStarPlanner.plan` resets last_envelope per plan and only _mk_envelope sets it, so a
    host-side early denial leaves it None. None means "read set unknown", NOT "read nothing" —
    treating it as clean would merge a plan whose reads were never recorded."""
    from freespace_sim.planner.lns.parallel import _read_set_is_clean

    box = ((0.0, 0.0, 0.0, 10.0, 10.0, 10.0), 0.0, 100.0)
    assert _read_set_is_clean((None,), []) is True         # nothing committed: vacuously clean
    assert _read_set_is_clean((None,), [box]) is False     # something committed + unknown reads
