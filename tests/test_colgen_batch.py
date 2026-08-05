"""Focused contracts for whole-schedule column-generation integration."""

from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.colgen import batch
from freespace_sim.planner.colgen.batch import ColumnGenerationPlanner, run_batch
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing_pool import ParallelPricingConfig
from freespace_sim.planner.colgen.solver import ColGenResult
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.types import (
    DenialReason,
    FlightRequest,
    IntentStatus,
    OperationalIntent,
    Terminal,
    vec,
)
from freespace_sim.volumes import Volume4D


def _requests() -> list[FlightRequest]:
    return [
        FlightRequest(2, vec(0, 200, 0), vec(1200, 200, 0), 8.0),
        FlightRequest(1, vec(0, 0, 0), vec(1200, 0, 0), 4.0),
    ]


class _RecordingDSS:
    def __init__(self, *, mutate_accepted: bool = False) -> None:
        self.mutate_accepted = mutate_accepted
        self.intents: list[OperationalIntent] = []

    def commit(self, intent: OperationalIntent) -> bool:
        self.intents.append(intent)
        if self.mutate_accepted and intent.accepted:
            intent.status = IntentStatus.REJECTED
            intent.denial_reason = DenialReason.CONFLICT_AT_COMMIT
        return intent.accepted


def test_phase3_params_validate_objective_and_shortcut():
    params = ColGenParams(objective="total_delay", shortcut=True)
    assert params.objective == "total_delay"
    assert params.shortcut is True

    with pytest.raises(ValueError, match="objective"):
        ColGenParams(objective="throughput")
    with pytest.raises(TypeError, match="objective"):
        ColGenParams(objective=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="shortcut"):
        ColGenParams(shortcut=1)  # type: ignore[arg-type]


def test_factory_returns_batch_only_wall_aware_planner():
    planner = get_planner("colgen")

    assert isinstance(planner, ColumnGenerationPlanner)
    assert planner.plans_whole_schedule is True
    assert planner.plans_terminal_airspace is True
    request = _requests()[0]
    cfg = SimConfig(planner="colgen")
    with pytest.raises(RuntimeError, match="whole-schedule.*batch mode"):
        planner.plan(request, ReservationLedger(cfg), cfg)


def test_run_batch_maps_missing_columns_and_fires_callbacks_in_event_order(
    monkeypatch,
    caplog,
):
    cfg = SimConfig(planner="colgen")
    scenario = scenario_from_requests(_requests())
    ledger = ReservationLedger(cfg)
    dss = _RecordingDSS()
    params = ColGenParams(max_iterations=1)
    captured = {}
    warnings = []

    def fake_solve(self, requests, solve_cfg, static_terms, solve_params, parallel=None,
                   on_iteration=None):
        captured.update(
            requests=requests,
            cfg=solve_cfg,
            static_terms=static_terms,
            params=solve_params,
        )
        return ColGenResult(
            columns={},
            stats={
                "backend": "gurobi",
                "termination_reason": "lp_gap",
                "iterations": 2,
                "search_exhausted_flight_ids": (1,),
            },
        )

    monkeypatch.setattr(batch.ColGenSolver, "solve", fake_solve)
    monkeypatch.setattr(
        batch,
        "_warn_if_terminal_dropped",
        lambda request, intent: warnings.append((request.flight_id, intent)),
    )
    status_calls = []
    report_calls = []
    caplog.set_level(logging.INFO, logger="freespace_sim.planner.colgen.batch")

    intents = run_batch(
        scenario,
        cfg,
        ledger,
        dss,  # type: ignore[arg-type]
        (),
        lambda done, request, intent: status_calls.append((done, request.flight_id, intent)),
        lambda done, total, intent: report_calls.append((done, total, intent)),
        collector=object(),
        params=params,
    )

    assert captured["requests"] == [event.request for event in scenario.events]
    assert captured["cfg"] is cfg
    assert captured["static_terms"] == ()
    assert captured["params"] is params
    assert [intent.request.flight_id for intent in intents] == [1, 2]
    assert all(intent.status is IntentStatus.REJECTED for intent in intents)
    assert [intent.denial_reason for intent in intents] == [
        DenialReason.SEARCH_EXHAUSTED,
        DenialReason.BUDGET_EXCEEDED,
    ]
    assert all(intent.planner == "colgen" for intent in intents)
    assert all(intent.solve_time_s >= 0.0 for intent in intents)
    assert dss.intents == intents
    assert [flight_id for flight_id, _intent in warnings] == [1, 2]
    assert [(done, flight_id) for done, flight_id, _intent in status_calls] == [(1, 1), (2, 2)]
    assert [(done, total) for done, total, _intent in report_calls] == [(1, 2), (2, 2)]
    assert "backend=gurobi termination=lp_gap iterations=2" in caplog.text
    assert "covering bug" not in caplog.text


def test_run_batch_uses_precommit_acceptance_for_covering_bug_log(monkeypatch, caplog):
    cfg = SimConfig(planner="colgen")
    request = _requests()[0]
    scenario = scenario_from_requests([request])
    dss = _RecordingDSS(mutate_accepted=True)

    monkeypatch.setattr(
        batch.ColGenSolver,
        "solve",
        lambda self, requests, solve_cfg, static_terms, params, parallel=None,
        on_iteration=None: ColGenResult(
            columns={request.flight_id: object()},
            stats={},
        ),
    )
    monkeypatch.setattr(
        batch,
        "column_to_intent",
        lambda column, selected_request, solve_cfg, solve_share_s=0.0: OperationalIntent(
            request=selected_request,
            status=IntentStatus.ACCEPTED,
            volumes=[object()],  # type: ignore[list-item]
            planner="colgen",
            solve_time_s=solve_share_s,
        ),
    )
    caplog.set_level(logging.ERROR, logger="freespace_sim.planner.colgen.batch")

    intents = run_batch(
        scenario,
        cfg,
        ReservationLedger(cfg),
        dss,  # type: ignore[arg-type]
        (),
        lambda *_args: None,
        None,
        None,
    )

    assert intents[0].status is IntentStatus.REJECTED
    assert intents[0].denial_reason is DenialReason.CONFLICT_AT_COMMIT
    assert "covering bug: flight_id=2" in caplog.text


def test_run_batch_rejects_terminal_requests_without_permanent_walls(monkeypatch):
    cfg = SimConfig(planner="colgen", terminal_airspace_always_active=False)
    terminal = Terminal("hub", capacity=2)
    request = FlightRequest(
        1,
        vec(0, 0, 0),
        vec(1200, 0, 0),
        0.0,
        origin_terminal=terminal,
    )
    monkeypatch.setattr(
        batch.ColGenSolver,
        "solve",
        lambda *_args, **_kwargs: pytest.fail("guard must run before the solver"),
    )

    with pytest.raises(NotImplementedError, match="terminal_airspace_always_active=True"):
        run_batch(
            scenario_from_requests([request]),
            cfg,
            ReservationLedger(cfg),
            _RecordingDSS(),  # type: ignore[arg-type]
            (),
            lambda *_args: None,
            None,
            None,
        )


def test_run_batch_rejects_a_prepopulated_dynamic_ledger(monkeypatch):
    cfg = SimConfig(planner="colgen")
    ledger = ReservationLedger(cfg)
    ledger.commit(
        99,
        [Volume4D(CylinderSpec(0.0, 0.0, 1.0, 0.0, 1.0), 0.0, 1.0)],
    )
    monkeypatch.setattr(
        batch.ColGenSolver,
        "solve",
        lambda *_args, **_kwargs: pytest.fail("guard must run before the solver"),
    )

    with pytest.raises(ValueError, match="empty dynamic ledger"):
        run_batch(
            scenario_from_requests(_requests()),
            cfg,
            ledger,
            _RecordingDSS(),  # type: ignore[arg-type]
            (),
            lambda *_args: None,
            None,
            None,
        )


def test_sim_batch_branch_forwards_planner_params(monkeypatch):
    sim_module = importlib.import_module("freespace_sim.sim")
    cfg = SimConfig(planner="colgen")
    params = ColGenParams(max_iterations=3, shortcut=True)
    calls = []

    monkeypatch.setattr(
        sim_module,
        "get_planner",
        lambda name: ColumnGenerationPlanner(params),
    )

    def fake_run_batch(
        scenario,
        solve_cfg,
        ledger,
        dss,
        static_terms,
        status,
        report,
        collector,
        *,
        params,
    ):
        calls.append(
            SimpleNamespace(
                scenario=scenario,
                cfg=solve_cfg,
                ledger=ledger,
                dss=dss,
                static_terms=static_terms,
                status=status,
                report=report,
                collector=collector,
                params=params,
            )
        )
        return [
            OperationalIntent(
                event.request,
                IntentStatus.REJECTED,
                denial_reason=DenialReason.BUDGET_EXCEEDED,
                planner="colgen",
            )
            for event in scenario.events
        ]

    monkeypatch.setattr(batch, "run_batch", fake_run_batch)
    result = sim_module.run(cfg, requests=_requests())

    assert len(calls) == 1
    assert calls[0].cfg is cfg
    assert calls[0].params is params
    assert calls[0].report is None
    assert [intent.request.flight_id for intent in result.intents] == [1, 2]
    assert result.verified


def test_run_batch_defaults_pricing_onto_the_process_pool(monkeypatch):
    """The production route must be able to reach the pricing pool, and does by default.

    ``sim.run`` -> ``run_batch`` is the only production entry point for whole-schedule
    planning: ``sim.run(parallel=...)`` selects the A* speculative runner and rejects
    batch planners outright.  Before this, ``run_batch`` called ``solve`` with no
    ``parallel`` argument and had no parameter to pass one, so every measured speedup
    from :mod:`~.pricing_pool` was unreachable in production.
    """

    from freespace_sim.planner.colgen.batch import (
        _PARALLEL_MIN_FLIGHTS,
        _default_pricing_pool,
    )

    assert _default_pricing_pool(_PARALLEL_MIN_FLIGHTS - 1) is None, "tiny batches stay serial"
    pool = _default_pricing_pool(_PARALLEL_MIN_FLIGHTS)
    assert pool is not None
    assert pool.n_workers >= 2
    assert pool.chunksize > 1, "chunked dispatch amortises per-task overhead"

    cfg = SimConfig(planner="colgen")
    seen: list = []

    def _capture(self, requests, solve_cfg, static_terms, params, parallel=None,
                 on_iteration=None):
        seen.append(parallel)
        return ColGenResult(columns={}, stats={})

    monkeypatch.setattr(batch.ColGenSolver, "solve", _capture)

    def _run(requests, **kwargs):
        return run_batch(
            scenario_from_requests(requests),
            cfg,
            ReservationLedger(cfg),
            _RecordingDSS(),  # type: ignore[arg-type]
            (),
            lambda *_args: None,
            None,
            None,
            **kwargs,
        )

    small = [FlightRequest(i, vec(0, 0, 0), vec(1200, 0, 0), float(i)) for i in range(4)]
    big = [
        FlightRequest(i, vec(0, 0, 0), vec(1200, 0, 0), float(i))
        for i in range(_PARALLEL_MIN_FLIGHTS + 1)
    ]
    explicit = ParallelPricingConfig(n_workers=3, chunksize=7)

    _run(small)
    _run(big)
    _run(big, parallel=explicit)

    assert seen[0] is None, "below the threshold the pool costs more than it saves"
    assert seen[1] is not None and seen[1].n_workers >= 2, "a real batch fans out by default"
    assert seen[2] is explicit, "an explicit config is forwarded verbatim"


def test_run_batch_forwards_the_per_iteration_callback(monkeypatch):
    """``on_iteration`` has to survive the production entry point to be worth anything.

    The solver grew per-iteration telemetry, but ``run_batch`` neither accepted nor
    forwarded it, so it was reachable only by calling ``ColGenSolver.solve`` directly.
    Every production run -- including a full 4,636-flight scenario solve -- was blind to
    it while appearing to have it.
    """

    cfg = SimConfig(planner="colgen")
    seen: list = []

    def _capture(self, requests, solve_cfg, static_terms, params, parallel=None,
                 on_iteration=None):
        seen.append(on_iteration)
        return ColGenResult(columns={}, stats={})

    monkeypatch.setattr(batch.ColGenSolver, "solve", _capture)

    def _sentinel(payload):  # never called; identity is the whole assertion
        raise AssertionError("stub solver must not invoke the callback")

    requests = [FlightRequest(1, vec(0, 0, 0), vec(1200, 0, 0), 0.0)]
    for kwargs in ({}, {"on_iteration": _sentinel}):
        run_batch(
            scenario_from_requests(requests), cfg, ReservationLedger(cfg),
            _RecordingDSS(),  # type: ignore[arg-type]
            (), lambda *_args: None, None, None, **kwargs,
        )

    assert seen[0] is None, "absent by default, so the solver skips the payload entirely"
    assert seen[1] is _sentinel, "and forwarded by identity when supplied"


def test_worker_recycle_budget_is_counted_in_flights_not_chunks():
    """``max_tasks_per_child`` must not silently multiply by ``chunksize``.

    ``mp.Pool`` increments its per-worker completion counter once per item pulled off the
    inqueue, and under ``imap_unordered(chunksize=k)`` that item is a chunk of k flights.
    Passing the flight budget straight through therefore recycles k times too late: the
    shipped 4-flight budget at chunksize 4 gave workers a 16-flight life, four times the
    documented residue bound.  Verified against real pools (64 items, 2 workers,
    maxtasksperchild=4): chunksize=1 -> 16 distinct pids, chunksize=4 -> 4.
    """

    for chunksize, expected_chunks in ((1, 16), (2, 8), (4, 4), (16, 1), (32, 1)):
        cfg = ParallelPricingConfig(
            n_workers=2, max_tasks_per_child=16, chunksize=chunksize
        )
        assert cfg.pool_maxtasksperchild == expected_chunks
        # The invariant that matters: flights per worker lifetime stays ~the budget,
        # never a multiple of it.
        assert cfg.pool_maxtasksperchild * chunksize >= 16
        assert cfg.pool_maxtasksperchild * chunksize < 16 + chunksize

    assert ParallelPricingConfig(n_workers=2, max_tasks_per_child=None).pool_maxtasksperchild is None
    # A budget finer than one chunk cannot be expressed; round up rather than to zero,
    # which mp.Pool would reject.
    assert ParallelPricingConfig(
        n_workers=2, max_tasks_per_child=1, chunksize=8
    ).pool_maxtasksperchild == 1
    with pytest.raises(ValueError, match="chunksize"):
        ParallelPricingConfig(n_workers=2, chunksize=0)
