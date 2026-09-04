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
import time
from types import SimpleNamespace

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


def _exit_before_ready(*_args):
    """Spawn-safe worker target used to exercise the startup EOF/sentinel path."""


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


@pytest.mark.slow
def test_index_free_coordinator_does_not_build_or_maintain_claims(monkeypatch):
    res = run(_congested(lam=200.0, horizon=120.0))
    state = LNSState(
        res.config, res.ledger, res.intents,
        static_terms=res.ledger.static_terminals(), maintain_claim_index=False,
    )
    assert state._claims == state._cells_of == {}
    assert state._contended == set()

    def must_not_index(*_args, **_kwargs):
        pytest.fail("index-free state attempted claim maintenance")

    monkeypatch.setattr(state, "_index_remove", must_not_index)
    monkeypatch.setattr(state, "_index_add", must_not_index)
    fid = state.movable_ids()[0]
    state.apply_delta({fid: state.incumbent[fid]})
    assert state._claims == state._cells_of == {}


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
def test_report_only_repair_returns_candidate_without_adopting_or_indexing_it(monkeypatch):
    """Workers report successes but retain only coordinator-blessed state between tasks."""
    res = run(_congested(lam=400.0, horizon=240.0))
    base = LNSState(res.config, res.ledger, res.intents,
                    static_terms=res.ledger.static_terminals())
    rep = LNSState.replica(
        res.config, base.final_intents(), static_terms=base.static_terms,
        unimpeded_cost=dict(base._unimp_cost),
    )
    at_start = _state_digest(rep)

    def must_not_adopt(*_args):
        pytest.fail("report-only repair adopted and indexed its candidate")

    monkeypatch.setattr(rep, "_apply_in_memory", must_not_adopt)
    accepted = None
    for i in range(60):
        rng = np.random.default_rng(np.random.SeedSequence([7, i]))
        victims = sorted(rep.movable_ids())[: 4 + (i % 3)]
        out = rep.try_repair(victims, rng, report_only=True)
        assert _state_digest(rep) == at_start
        if out.accepted:
            accepted = out
            break
    assert accepted is not None, "no accepted report-only repair in 60 tries"
    assert accepted.new_intents


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
    quiet = LNSState.replica(res.config, intents, static_terms=base.static_terms,
                             unimpeded_cost=unimp, record_envelope=False)
    assert rep.repair_planner.record_envelope is True
    assert quiet.repair_planner.record_envelope is False
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

    Effective widths below two deliberately use the in-process engine: a private replica cannot
    add concurrency, and this pins that routing preserves the complete schedule trajectory.
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
    assert par.parallel_mode == "sequential" and par.parallel_stats == {}
    assert par.npo == par.n_iterations == 40
    assert seq.npo == seq.n_iterations == 40
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
    kw = dict(
        seed=3, neighborhood_size=4, log_every=0, max_iterations=24,
        search_workers=3, parallel_mode="sync",
    )
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
                      incremental_release=True, kernel_log2_min=None, window_bytes=None)
    pool = LNSWorkerPool(res.config, base.final_intents(), base.static_terms,
                         dict(base._unimp_cost), spec, 2).start()
    try:
        pool._procs[0].kill()
        with pytest.raises(WorkerLost):
            pool.collect(1, timeout=30.0)
    finally:
        pool.close()


def test_worker_exit_during_startup_is_cleaned_up(monkeypatch):
    """A child that exits before its ready message must neither hang nor leak sibling workers."""
    from freespace_sim.config import SimConfig
    from freespace_sim.planner.lns import parallel

    monkeypatch.setattr(parallel, "_worker_main", _exit_before_ready)
    spec = parallel.WorkerSpec(
        neighborhood_size=4, accept_epsilon=0.0, repair_order="premium",
        max_walks=10, map_max_cells=4096, turnaround_s=None,
        frozen_flight_ids=frozenset(), movable_uss_ids=None,
        incremental_release=True, kernel_log2_min=None,
    )
    pool = parallel.LNSWorkerPool(SimConfig(), [], (), {}, spec, 2)
    with pytest.raises(parallel.WorkerLost, match="before reporting ready"):
        pool.start(deadline=time.monotonic() + 10.0)
    assert pool._procs == []
    assert pool._conns == []


def test_pool_start_preserves_optional_kernel_fallback(monkeypatch):
    """Pool startup must not import the optional numba module ahead of AStarPlanner's guard."""
    import builtins

    from freespace_sim.config import SimConfig
    from freespace_sim.planner.lns.parallel import LNSWorkerPool, WorkerSpec

    real_import = builtins.__import__

    def numba_free_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "freespace_sim.planner.astar" and "kernel" in (fromlist or ()):
            raise ImportError("optional numba kernel unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", numba_free_import)
    spec = WorkerSpec(
        neighborhood_size=4, accept_epsilon=0.0, repair_order="premium",
        max_walks=10, map_max_cells=4096, turnaround_s=None,
        frozen_flight_ids=frozenset(), movable_uss_ids=None,
        incremental_release=True, kernel_log2_min=None,
    )
    pool = LNSWorkerPool(SimConfig(), [], (), {}, spec, 1)
    try:
        pool.start(deadline=time.monotonic() + 30.0)
    finally:
        pool.close()


def test_search_workers_and_parallel_mode_are_validated():
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import LNSConfig, run_lns
    from freespace_sim.config import SimConfig

    cfg = SimConfig()
    for bad in (0, -1, True, 1.5, "1"):
        with pytest.raises(ValueError, match="search_workers"):
            run_lns(cfg, ReservationLedger(cfg), [],
                    LNSConfig(max_iterations=0, log_every=0, search_workers=bad))
    with pytest.raises(ValueError, match="search_workers"):     # the 4x-cores ceiling
        run_lns(cfg, ReservationLedger(cfg), [],
                LNSConfig(max_iterations=0, log_every=0, search_workers=10_000))
    with pytest.raises(ValueError, match="parallel_mode"):
        run_lns(cfg, ReservationLedger(cfg), [],
                LNSConfig(max_iterations=0, log_every=0, parallel_mode="deta"))
    with pytest.raises(ValueError, match="operators"):
        run_lns(cfg, ReservationLedger(cfg), [],
                LNSConfig(max_iterations=0, log_every=0, operators=(1,)))


def test_execution_config_is_validated_before_ledger_takeover():
    from freespace_sim.config import SimConfig
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import LNSConfig, run_lns

    cfg = SimConfig()
    invalid = {
        "seed": (-1, True, 1.5, "1"),
        "max_iterations": (-1, True, 1.5, "1"),
        "neighborhood_size": (0, -1, True, 1.5, "1"),
        "adaptive": (0, 1, "true"),
        "gamma": (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"),
        "accept_epsilon": (-1.0, -float("inf"), float("nan"), True, "1"),
        "repair_order": ("bogus", "Premium", 1),
        "max_walks": (-1, True, 1.5, "1"),
        "map_max_cells": (-1, True, 1.5, "1"),
        "frozen_flight_ids": (None, "1", frozenset({True}), frozenset({1.5})),
        "movable_uss_ids": ("uss", frozenset({1}), 1),
        "incremental_release": (0, 1, "true"),
        "unimpeded_workers": (0, -1, True, 1.5, "1"),
        "worker_kernel_log2": (-1, True, 1.5, "1"),
        "window_bytes": (0, -1, True, 1.5, "1"),
        "time_limit_s": (-1.0, -float("inf"), float("nan"), True, "1"),
        "verify_every": (-1, True, 1.5, "1"),
        "log_every": (-1, True, 1.5, "1"),
    }
    for name, values in invalid.items():
        for value in values:
            kwargs = {"max_iterations": 0, "log_every": 0}
            kwargs[name] = value
            ledger = ReservationLedger(cfg)
            ledger.subscribe(lambda _fid, _volumes: None)
            with pytest.raises(ValueError, match=name):
                run_lns(cfg, ledger, [], LNSConfig(**kwargs))
            assert ledger._observers and ledger.epoch == 0


def test_lns_config_normalizes_compatible_numpy_scalars():
    from freespace_sim.planner.lns import LNSConfig
    from freespace_sim.planner.lns.solver import _validate_lns_config

    normalized = _validate_lns_config(LNSConfig(
        seed=np.int64(3),
        max_iterations=np.int64(1),
        neighborhood_size=np.int64(4),
        operators=["agent", "random"],
        adaptive=np.bool_(True),
        gamma=np.float32(0.25),
        search_workers=np.int64(2),
        accept_epsilon=np.float32(0.25),
        max_walks=np.int64(2),
        map_max_cells=np.int64(32),
        frozen_flight_ids={np.int64(1), np.int64(2)},
        movable_uss_ids={"uss-a", "uss-b"},
        incremental_release=np.bool_(False),
        unimpeded_workers=np.int64(2),
        worker_kernel_log2=np.int64(0),
        window_bytes=np.int64(2048),
        time_limit_s=np.float32(10.0),
        verify_every=np.int64(1),
        log_every=np.int64(0),
    ))
    for name in (
        "seed", "max_iterations", "neighborhood_size", "search_workers", "max_walks",
        "map_max_cells", "unimpeded_workers", "worker_kernel_log2", "window_bytes",
        "verify_every", "log_every",
    ):
        assert type(getattr(normalized, name)) is int
    for name in ("gamma", "accept_epsilon", "time_limit_s"):
        assert type(getattr(normalized, name)) is float
    assert normalized.operators == ("agent", "random")
    assert normalized.frozen_flight_ids == frozenset({1, 2})
    assert normalized.movable_uss_ids == frozenset({"uss-a", "uss-b"})
    assert type(normalized.adaptive) is bool
    assert type(normalized.incremental_release) is bool

    unlimited = _validate_lns_config(LNSConfig(
        accept_epsilon=float("inf"), time_limit_s=float("inf")))
    assert unlimited.accept_epsilon == unlimited.time_limit_s == float("inf")


def test_lns_config_is_keyword_only():
    from freespace_sim.planner.lns import LNSConfig

    with pytest.raises(TypeError, match="positional"):
        LNSConfig(7)


def test_auc_integrates_the_full_step_trajectory():
    """AUC includes the initial segment and final tail; incumbent cost changes at completion."""
    from freespace_sim.planner.lns.solver import _trajectory_auc

    trajectory = [
        {"wall_s": 2.0, "incumbent_cost": 80.0},
        {"wall_s": 4.0, "incumbent_cost": 60.0},
    ]
    assert _trajectory_auc(trajectory, cost_before=100.0, horizon_s=4.0) == 360.0
    assert _trajectory_auc(trajectory, cost_before=100.0, horizon_s=6.0) == 480.0
    assert _trajectory_auc([], cost_before=100.0, horizon_s=3.0) == 300.0


def test_shared_trajectory_factory_preserves_mode_specific_schema():
    from freespace_sim.planner.lns.solver import _trajectory_row

    sequential = _trajectory_row(
        0, "random", (2, 1), False, "denied", 10.0, float("inf"), 100.0, 0.5,
    )
    assert sequential == {
        "iter": 0, "op": "random", "n": 2, "victims": [2, 1],
        "accepted": False, "reason": "denied", "cost_old": 10.0,
        "cost_new": None, "incumbent_cost": 100.0, "wall_s": 0.5,
    }
    parallel = _trajectory_row(
        1, "agent", (3,), True, "overwrite", 20.0, 15.0, 95.0, 0.75,
        incumbent_before=100.0,
        audit={"dispatch_iter": 4, "base_version": 2, "worker": 1},
    )
    assert parallel["realized_improvement"] == 5.0
    assert (parallel["dispatch_iter"], parallel["base_version"], parallel["worker"]) == (4, 2, 1)


def test_parallel_state_build_honors_unimpeded_workers(monkeypatch):
    """The parallel coordinator must forward the parent's one-off ruler pool setting."""
    from freespace_sim.config import SimConfig
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import LNSConfig
    from freespace_sim.planner.lns import state as state_module
    from freespace_sim.planner.lns.parallel import run_lns_parallel

    seen = []

    def fake_unimpeded_costs(_cfg, _static_terms, requests, *, n_workers, shortcut=False):
        seen.append((list(requests), n_workers, shortcut))
        return []

    monkeypatch.setattr(state_module, "unimpeded_costs", fake_unimpeded_costs)
    cfg = SimConfig()
    out = run_lns_parallel(
        cfg, ReservationLedger(cfg), [],
        LNSConfig(max_iterations=0, log_every=0, search_workers=2, unimpeded_workers=3),
    )
    # shortcut=False is not incidental: a shortcut arm is refused on the parallel path (the DROP
    # workers build their own LNSState and would repair with bare A*), so the coordinator's ruler
    # must stay bare too or its premiums would not match what the workers actually produce.
    assert seen == [([], 3, False)]
    assert out.n_iterations == out.npo == 0
    assert out.pool_spawn_s == 0.0


def test_parallel_caps_pool_skips_coordinator_index_and_closes_before_verify(monkeypatch):
    import json

    from freespace_sim.config import SimConfig
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import LNSConfig, parallel, solver

    observed = {"closed": False}

    class Pool:
        spawn_s = 0.25

        def __init__(self, _cfg, _intents, _static_terms, _unimpeded, spec, n_workers):
            observed["workers"] = self.n_workers = n_workers
            observed["record_envelope"] = spec.record_envelope

        def start(self, *, deadline=None):
            return self

        def close(self):
            observed["closed"] = True

    def loop(state, pool, *_args):
        assert state._maintain_claim_index is False
        assert state._claims == state._cells_of == {}
        return {"n_iter": 0, "n_accepted": 0}

    def verify_after_close(*_args, **_kwargs):
        assert observed["closed"] is True
        return None

    monkeypatch.setattr(parallel, "LNSWorkerPool", Pool)
    monkeypatch.setattr(parallel, "_loop_drop", loop)
    monkeypatch.setattr(solver.verify, "find_interflight_conflict", verify_after_close)

    cfg = SimConfig()
    out = parallel.run_lns_parallel(
        cfg, ReservationLedger(cfg), [],
        LNSConfig(max_iterations=np.int64(2), log_every=0,
                  search_workers=np.int64(4), parallel_mode="drop",
                  repair_planner="astar_ref"),
    )
    assert observed == {"closed": True, "workers": 2, "record_envelope": True}
    assert type(out.search_workers) is int
    assert out.search_workers == 2
    assert out.pool_spawn_s == 0.25
    assert out.repair_planner == "astar_ref"
    assert out.n_release_subs is None
    assert out.n_commit_subs is None
    json.dumps(out.summary())


def test_infinite_time_limit_starts_the_pool_without_a_deadline(monkeypatch):
    from freespace_sim.config import SimConfig
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import LNSConfig, parallel

    deadlines = []

    class Pool:
        spawn_s = 0.0

        def __init__(self, _cfg, _intents, _static_terms, _unimpeded, _spec, n_workers):
            self.n_workers = n_workers

        def start(self, *, deadline=None):
            deadlines.append(deadline)
            return self

        def close(self):
            pass

    monkeypatch.setattr(parallel, "LNSWorkerPool", Pool)
    monkeypatch.setattr(
        parallel, "_loop_sync",
        lambda *_args: {"n_iter": 0, "n_accepted": 0, "n_not_selected": 0},
    )

    cfg = SimConfig()
    parallel.run_lns_parallel(
        cfg, ReservationLedger(cfg), [],
        LNSConfig(
            max_iterations=2, search_workers=2, parallel_mode="sync",
            time_limit_s=float("inf"), log_every=0,
        ),
    )
    assert deadlines == [None]


def test_sync_logging_fires_when_an_interval_is_crossed(monkeypatch):
    from freespace_sim.planner.lns import parallel

    calls = []
    monkeypatch.setattr(parallel.log, "info", lambda *args: calls.append(args))
    lns = SimpleNamespace(log_every=200, max_iterations=500)
    state = SimpleNamespace(total_cost=90.0)
    selector = SimpleNamespace(weights={"random": 1.0})

    parallel._maybe_log(
        lns, "sync", 3, 201, 1, state, selector, 100.0, previous_iter=198,
    )
    parallel._maybe_log(
        lns, "sync", 3, 204, 1, state, selector, 100.0, previous_iter=201,
    )
    parallel._maybe_log(
        lns, "sync", 3, 402, 2, state, selector, 100.0, previous_iter=399,
    )
    assert [args[3] for args in calls] == [201, 402]


@pytest.mark.parametrize("iterations", [0, 1])
def test_effective_width_below_two_uses_the_sequential_engine(monkeypatch, iterations):
    from freespace_sim.config import SimConfig
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import LNSConfig, parallel, run_lns

    class UnexpectedPool:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("a zero/one-task budget must not build a worker replica")

    monkeypatch.setattr(parallel, "LNSWorkerPool", UnexpectedPool)
    cfg = SimConfig()
    config = LNSConfig(
        max_iterations=iterations, operators=("random",), log_every=0,
        search_workers=4, parallel_mode="drop",
    )
    for runner in (run_lns, parallel.run_lns_parallel):
        out = runner(cfg, ReservationLedger(cfg), [], config)
        assert out.parallel_mode == "sequential"
        assert out.search_workers == 1
        assert out.pool_spawn_s == 0.0
        assert out.n_iterations == iterations


def test_budget_exhaustion_before_pool_start_reports_zero_workers(monkeypatch):
    from freespace_sim.config import SimConfig
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner.lns import LNSConfig, parallel

    class UnexpectedPool:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("an exhausted budget must not construct the pool")

    monkeypatch.setattr(parallel, "LNSWorkerPool", UnexpectedPool)
    monkeypatch.setattr(parallel, "_out_of_budget", lambda *_args: True)
    cfg = SimConfig()
    out = parallel.run_lns_parallel(
        cfg, ReservationLedger(cfg), [],
        LNSConfig(max_iterations=2, search_workers=2, parallel_mode="drop", log_every=0),
    )
    assert out.parallel_mode == "drop"
    assert out.search_workers == 0
    assert out.n_iterations == 0
    assert out.pool_spawn_s == 0.0


def test_pool_start_timeout_is_closed_and_reported_as_zero_workers():
    from freespace_sim.planner.lns.parallel import WorkerStartTimeout, _run_and_close_pool

    class Pool:
        spawn_s = 0.25
        n_workers = 2
        closed = False

        def start(self, *, deadline=None):
            raise WorkerStartTimeout("budget")

        def close(self):
            self.closed = True

    pool = Pool()
    completed, spawn_s, started = _run_and_close_pool(
        pool, deadline=1.0, execute=lambda _pool: pytest.fail("loop must not run"),
    )
    assert (completed, spawn_s, started) == (None, 0.25, 0)
    assert pool.closed


def test_zero_sequential_rate_has_no_relative_comparison():
    from analysis.sweep_lns_workers import _relative_rate

    assert _relative_rate(0.0, 0.0) is None
    assert _relative_rate(2.0, 0.0) is None
    assert _relative_rate(2.0, 1.0) == 2.0


def test_replica_profiler_warms_lazy_planner_state_before_ready(monkeypatch):
    from analysis import prof_lns_replica_memory as profiler

    events = []
    rng = object()

    class State:
        ledger = SimpleNamespace(n_volumes=17)

        def __init__(self):
            self.rng = rng

        def movable_ids(self):
            return list(range(12))

        def try_repair(self, victims, actual_rng, epsilon, **kwargs):
            events.append(("repair", victims, actual_rng, epsilon, kwargs))

    class Conn:
        def send(self, message):
            events.append(("send", message[0]))

        def recv(self):
            return ("stop",)

    monkeypatch.setattr(profiler.LNSState, "replica", lambda *_args, **_kwargs: State())
    profiler._worker_main(Conn(), None, None, None, None, None)

    kind, victims, actual_rng, epsilon, kwargs = events[0]
    assert kind == "repair" and victims == list(range(8)) and actual_rng is rng
    assert epsilon == float("inf")
    assert kwargs == {"order_mode": "premium", "report_only": True}
    assert events[1] == ("send", "ready")


def test_sweep_reports_effective_workers_without_mislabeling_the_baseline():
    from analysis.sweep_lns_workers import _sequential_baseline, _worker_metadata

    capped = _worker_metadata(
        4, SimpleNamespace(search_workers=np.int64(1), parallel_mode="sequential")
    )
    sequential = _worker_metadata(
        1, SimpleNamespace(search_workers=1, parallel_mode="sequential")
    )
    assert capped == {"requested_workers": 4, "workers": 1, "mode": "sequential"}
    assert _sequential_baseline([capped, sequential]) is sequential


def test_default_config_is_the_sequential_path():
    """search_workers=1 stays in-process; enabling real workers adds full private replicas, so
    memory is linear in workers — the same reason colgen's pricing pool is defaulted off."""
    from freespace_sim.planner.lns import LNSConfig

    c = LNSConfig()
    assert c.search_workers == 1
    assert c.parallel_mode == "drop"


# ==================================================================== DROP mode
@pytest.mark.slow
def test_drop_at_one_worker_matches_sequential():
    """A requested DROP run with no possible concurrency uses the sequential engine unchanged."""
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
    assert drop.parallel_mode == "sequential"
    assert drop.parallel_stats == {}


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
    assert [r["iter"] for r in out.trajectory] == list(range(24))
    assert sorted(r["dispatch_iter"] for r in out.trajectory) == list(range(24))

    incumbent = out.cost_before
    for row in out.trajectory:
        assert row["realized_improvement"] == pytest.approx(
            incumbent - row["incumbent_cost"])
        incumbent = row["incumbent_cost"]


def test_drop_rows_keep_completion_order_and_auditable_dispatch_ids():
    from freespace_sim.planner.lns.neighborhood import AdaptiveSelector
    from freespace_sim.planner.lns.parallel import TaskResult, _Changelog, _loop_drop

    results = [
        TaskResult(i, 0, i, 0, "random", (), {}, 0.0, 0.0, (), "empty")
        for i in (2, 0, 1)
    ]

    class Pool:
        n_workers = 3
        worker_version = [0, 0, 0]

        def sync(self, *_args):
            pass

        def dispatch(self, *_args):
            pass

        def collect(self, _n):
            return [results.pop(0)]

    lns = SimpleNamespace(
        max_iterations=3, time_limit_s=None, seed=0, adaptive=False,
        operators=("random",), accept_epsilon=0.0, verify_every=0, log_every=0,
    )
    trajectory = []
    _loop_drop(
        SimpleNamespace(total_cost=10.0), Pool(), lns,
        AdaptiveSelector(("random",)), set(), _Changelog(),
        time.monotonic(), trajectory, 10.0,
    )
    assert [row["iter"] for row in trajectory] == [0, 1, 2]
    assert [row["dispatch_iter"] for row in trajectory] == [2, 0, 1]


def test_sync_attaches_the_incumbent_change_to_the_winner_row():
    from freespace_sim.planner.lns.neighborhood import AdaptiveSelector
    from freespace_sim.planner.lns.parallel import TaskResult, _Changelog, _loop_sync

    old = SimpleNamespace(cost=50.0, volumes=())
    runner_up = SimpleNamespace(cost=45.0, volumes=())
    winner = SimpleNamespace(cost=40.0, volumes=())
    results = [
        TaskResult(0, 2, 2, 0, "random", (1,), {1: winner}, 50.0, 40.0, (), "improved"),
        TaskResult(0, 0, 0, 0, "random", (), {}, 0.0, 0.0, (), "empty"),
        TaskResult(0, 1, 1, 0, "random", (1,), {1: runner_up}, 50.0, 45.0, (), "improved"),
    ]

    class State:
        total_cost = 100.0
        incumbent = {1: old}

        def apply_delta(self, changes):
            for fid, intent in changes.items():
                self.total_cost += intent.cost - self.incumbent[fid].cost
                self.incumbent[fid] = intent

    class Pool:
        n_workers = 3
        worker_version = [0, 0, 0]

        def sync_all(self, *_args):
            pass

        def dispatch(self, *_args):
            pass

        def collect(self, _n):
            return results

    lns = SimpleNamespace(
        max_iterations=3, time_limit_s=None, seed=0, adaptive=False,
        operators=("random",), verify_every=0, log_every=0,
    )
    trajectory = []
    _loop_sync(
        State(), Pool(), lns, AdaptiveSelector(("random",)), set(), _Changelog(),
        time.monotonic(), trajectory, 100.0,
    )
    assert [row["incumbent_cost"] for row in trajectory] == [100.0, 100.0, 90.0]
    assert [row["realized_improvement"] for row in trajectory] == [0.0, 0.0, 10.0]
    assert [row["accepted"] for row in trajectory] == [False, False, True]


def test_sync_trims_history_and_does_not_adapt_in_uniform_mode():
    from freespace_sim.planner.lns.neighborhood import AdaptiveSelector
    from freespace_sim.planner.lns.parallel import (
        TaskResult,
        _Changelog,
        _loop_sync,
    )

    intent = SimpleNamespace(cost=10.0, volumes=())
    changelog = _Changelog()
    changelog.record({1: intent}, {1: intent})
    result = TaskResult(0, 0, 0, 1, "random", (), {}, 0.0, 0.0, (), "empty")

    class Pool:
        n_workers = 1
        worker_version = [0]

        def sync_all(self, log):
            self.worker_version = [log.version]

        def dispatch(self, *_args):
            pass

        def collect(self, _n):
            return [result]

    lns = SimpleNamespace(
        max_iterations=1, time_limit_s=None, seed=0, adaptive=False,
        operators=("random",), verify_every=0, log_every=0,
    )
    selector = AdaptiveSelector(("random",), gamma=0.25)
    trajectory = []
    stats = _loop_sync(
        SimpleNamespace(total_cost=10.0), Pool(), lns, selector, set(), changelog,
        time.monotonic(), trajectory, 10.0,
    )

    assert changelog._entries == []
    assert selector.weights == {"random": 1.0}
    assert stats["n_iter"] == 1


def test_drop_does_not_adapt_in_uniform_mode():
    from freespace_sim.planner.lns.neighborhood import AdaptiveSelector
    from freespace_sim.planner.lns.parallel import TaskResult, _Changelog, _loop_drop

    result = TaskResult(0, 0, 0, 0, "random", (), {}, 0.0, 0.0, (), "empty")

    class Pool:
        n_workers = 1
        worker_version = [0]

        def sync(self, *_args):
            pass

        def dispatch(self, *_args):
            pass

        def collect(self, _n):
            return [result]

    lns = SimpleNamespace(
        max_iterations=1, time_limit_s=None, seed=0, adaptive=False,
        operators=("random",), accept_epsilon=0.0, verify_every=0, log_every=0,
    )
    selector = AdaptiveSelector(("random",), gamma=0.25)
    _loop_drop(
        SimpleNamespace(total_cost=10.0), Pool(), lns, selector, set(), _Changelog(),
        time.monotonic(), [], 10.0,
    )
    assert selector.weights == {"random": 1.0}


@pytest.mark.parametrize(
    ("accept_epsilon", "accepted", "victims_changed"),
    [(2.0, False, False), (1.0, True, False), (1.0, True, True)],
)
def test_drop_overwrite_is_one_atomic_net_improvement(
    accept_epsilon, accepted, victims_changed, monkeypatch,
):
    """A stale overwrite applies once and is judged/rewarded by its realized net gain."""
    from freespace_sim.planner.lns import parallel
    from freespace_sim.planner.lns.neighborhood import AdaptiveSelector

    base_a = SimpleNamespace(cost=50.0, volumes=())
    current_a = SimpleNamespace(cost=41.0, volumes=())
    old_b = SimpleNamespace(cost=50.0, volumes=())
    new_b = SimpleNamespace(cost=39.0, volumes=())

    class State:
        def __init__(self):
            self.incumbent = (
                {1: base_a, 2: current_a} if victims_changed
                else {1: current_a, 2: old_b}
            )
            self.total_cost = 91.0
            self.apply_calls = []

        def apply_delta(self, changes):
            self.apply_calls.append(tuple(changes))
            for fid, intent in changes.items():
                self.total_cost += intent.cost - self.incumbent[fid].cost
                self.incumbent[fid] = intent

    result = parallel.TaskResult(
        0, 0, 0, 0, "random", (2,), {2: new_b}, 50.0, 39.0, (), "improved",
    )

    class Pool:
        n_workers = 1
        worker_version = [0]

        def sync(self, *_args):
            pass

        def dispatch(self, *_args):
            pass

        def collect(self, _n):
            return [result]

    changelog = parallel._Changelog()
    if victims_changed:
        # The first in-flight repair already changed this candidate's victim. It still deserves
        # the same whole-solution comparison, but no geometric clean merge is possible.
        changelog.record({2: current_a}, {2: old_b})
        monkeypatch.setattr(
            parallel, "_read_set_is_clean",
            lambda *_args: pytest.fail("overlapping victims cannot take the clean-merge path"),
        )
    else:
        changelog.record({1: current_a}, {1: base_a})
        # Empty fake volumes otherwise make the geometric read set vacuously clean.
        monkeypatch.setattr(parallel, "_read_set_is_clean", lambda *_args: False)
    state = State()
    selector = AdaptiveSelector(("random",), gamma=1.0)
    lns = SimpleNamespace(
        max_iterations=1, time_limit_s=None, seed=0, adaptive=True,
        operators=("random",), accept_epsilon=accept_epsilon,
        verify_every=0, log_every=0,
    )
    trajectory = []
    stats = parallel._loop_drop(
        state, Pool(), lns, selector, set(), changelog,
        time.monotonic(), trajectory, 100.0,
    )

    assert trajectory[0]["accepted"] is accepted
    if accepted:
        # Local gain 11 minus the reverted gain 9 = a realized gain of 2.
        assert state.apply_calls == ([(2,)] if victims_changed else [(1, 2)])
        assert state.total_cost == 89.0
        assert changelog.version == 2
        assert selector.weights["random"] == 2.0
        assert stats["n_overwrite"] == 1
        assert stats["n_stale_victims"] == 0
        assert trajectory[0]["reason"] == "overwrite"
        assert trajectory[0]["realized_improvement"] == 2.0
    else:
        # Equality to epsilon is not a strict improvement, and nothing is partially reverted.
        assert state.apply_calls == []
        assert state.total_cost == 91.0
        assert changelog.version == 1


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
    assert list(cl.diff_since(0)) == [b, a]             # last TOUCH wins replay order
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
