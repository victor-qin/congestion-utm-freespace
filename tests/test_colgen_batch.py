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


def _cfg(**overrides) -> SimConfig:
    """A config colgen can actually plan on: one flight level (v1 has no level choice)."""
    return SimConfig(planner="colgen", flight_levels_m=(100.0,), **overrides)


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


def test_params_validate_the_objective():
    params = ColGenParams(objective="total_delay")
    assert params.objective == "total_delay"

    with pytest.raises(ValueError, match="objective"):
        ColGenParams(objective="throughput")
    with pytest.raises(TypeError, match="objective"):
        ColGenParams(objective=1)  # type: ignore[arg-type]


def test_factory_returns_batch_only_wall_aware_planner():
    planner = get_planner("colgen")

    assert isinstance(planner, ColumnGenerationPlanner)
    assert planner.plans_whole_schedule is True
    assert planner.plans_terminal_airspace is True
    request = _requests()[0]
    cfg = _cfg()
    with pytest.raises(RuntimeError, match="whole-schedule.*batch mode"):
        planner.plan(request, ReservationLedger(cfg), cfg)


def test_run_batch_maps_missing_columns_and_fires_callbacks_in_event_order(
    monkeypatch,
    caplog,
):
    cfg = _cfg()
    scenario = scenario_from_requests(_requests())
    ledger = ReservationLedger(cfg)
    dss = _RecordingDSS()
    params = ColGenParams(max_iterations=1)
    captured = {}
    warnings = []

    def fake_solve(self, requests, solve_cfg, static_terms, solve_params,
                   on_iteration=None, **_kw):
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

    intents, _stats = run_batch(
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
    cfg = _cfg()
    request = _requests()[0]
    scenario = scenario_from_requests([request])
    dss = _RecordingDSS(mutate_accepted=True)

    monkeypatch.setattr(
        batch.ColGenSolver,
        "solve",
        lambda self, requests, solve_cfg, static_terms, params,
        on_iteration=None, **_kw: ColGenResult(
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

    intents, _stats = run_batch(
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
    cfg = _cfg(terminal_airspace_always_active=False)
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


def test_run_batch_rejects_a_multi_level_run_up_front():
    """The shipped ladder is three levels, so this is the first wall a new colgen run hits.

    ``build_flight_graph`` guards it too, but per flight and four frames down; refusing at the
    entry point is what lets the message name ``flight_levels_m``.
    """

    cfg = SimConfig(planner="colgen")  # default (30, 70, 110) ladder
    assert cfg.n_levels == 3
    with pytest.raises(NotImplementedError, match="flight_levels_m"):
        run_batch(
            scenario_from_requests(_requests()), cfg, ReservationLedger(cfg),
            _RecordingDSS(),  # type: ignore[arg-type]
            (), lambda *_args: None, None, None,
        )


def test_run_batch_rejects_a_prepopulated_dynamic_ledger(monkeypatch):
    cfg = _cfg()
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
    cfg = _cfg()
    params = ColGenParams(max_iterations=3, n_heuristic_tries=8)
    calls = []

    monkeypatch.setattr(
        sim_module,
        "get_planner",
        lambda name, planner_params=None: ColumnGenerationPlanner(params),
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
        ], {"termination_reason": "lp_gap", "iterations": 4}

    monkeypatch.setattr(batch, "run_batch", fake_run_batch)
    result = sim_module.run(cfg, requests=_requests())

    assert len(calls) == 1
    assert calls[0].cfg is cfg
    assert calls[0].params is params
    assert calls[0].report is None
    assert [intent.request.flight_id for intent in result.intents] == [1, 2]
    assert result.verified
    # The solve's diagnostics have to survive onto the result, or the run folder cannot
    # say whether this schedule came from a converged solve or a truncated one.
    assert result.planner_stats == {"termination_reason": "lp_gap", "iterations": 4}


def test_run_batch_forwards_the_per_iteration_callback(monkeypatch):
    """``on_iteration`` has to survive the production entry point to be worth anything.

    The solver grew per-iteration telemetry, but ``run_batch`` neither accepted nor
    forwarded it, so it was reachable only by calling ``ColGenSolver.solve`` directly.
    Every production run -- including a full 4,636-flight scenario solve -- was blind to
    it while appearing to have it.
    """

    cfg = _cfg()
    seen: list = []

    def _capture(self, requests, solve_cfg, static_terms, params, on_iteration=None, **_kw):
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

    assert seen[0] is batch._log_iteration, (
        "a caller that supplies nothing still gets progress; the entry point must not go "
        "silent for the length of a solve"
    )
    assert seen[1] is _sentinel, "and an explicit callback is forwarded by identity"


def test_factory_carries_colgen_params_and_refuses_them_elsewhere():
    """A params object must reach the planner, and never be dropped on the floor.

    ``get_planner`` is the only route from a run's configuration to the solver's budget, and a
    silently ignored budget produces a run that looks converged at the default limit.
    """

    params = ColGenParams(max_iterations=7, time_limit_s=900.0)

    assert get_planner("colgen", params).params is params
    assert get_planner("colgen").params == ColGenParams(), "no params -> the shipped defaults"
    with pytest.raises(ValueError, match="takes no params"):
        get_planner("astar", params)


def test_sim_run_forwards_planner_params_to_the_factory(monkeypatch):
    """``sim.run(planner_params=...)`` is the production route from the CLI to the solver."""

    sim_module = importlib.import_module("freespace_sim.sim")
    params = ColGenParams(max_iterations=3)
    seen: list = []

    def _capture(name, planner_params=None):
        seen.append((name, planner_params))
        return ColumnGenerationPlanner(planner_params)

    monkeypatch.setattr(sim_module, "get_planner", _capture)
    monkeypatch.setattr(
        batch, "run_batch",
        lambda *args, **kwargs: ([
            OperationalIntent(event.request, IntentStatus.REJECTED,
                              denial_reason=DenialReason.BUDGET_EXCEEDED, planner="colgen")
            for event in args[0].events
        ], {}),
    )
    sim_module.run(_cfg(), requests=_requests(), planner_params=params)

    assert seen == [("colgen", params)]


@pytest.mark.parametrize(
    ("stats", "make_params", "expected", "forbidden"),
    [
        # A time-limited solve returns a full, ordinary-looking schedule; the only place the
        # truncation can surface is the log, at the moment the run could still be relaunched.
        (
            {"termination_reason": "time_limit", "iterations": 1},
            lambda: ColGenParams(time_limit_s=45.0),
            ("stopped on its time limit (45s)",),
            None,
        ),
        # The gate is the revenue scale; the honest magnitude is the cost scale, and they
        # disagree. On ``colgen_test`` Gurobi's duals close the revenue gap at iteration 1
        # where HiGHS's leave it open, so which backend is installed decides the early stop and
        # nothing in the results says so -- hence the warning quotes the cost-scale value.
        (
            {
                "termination_reason": "lp_gap", "iterations": 1,
                "lp_gap_revenue": 4.65e-05, "lp_gap_cost": 1.166,
            },
            lambda: ColGenParams(gap_metric="revenue"),
            ("stopped on the revenue-scale lp_gap", "still 1.17"),
            None,
        ),
        # Both scales agreeing is the case the warning must stay quiet for.
        (
            {
                "termination_reason": "lp_gap", "iterations": 40,
                "lp_gap_revenue": 1e-06, "lp_gap_cost": 1e-05,
            },
            lambda: ColGenParams(gap_metric="revenue"),
            (),
            None,
        ),
        # ``ip_not_proven`` is not a budget timeout: a run whose loop converged on ``lp_gap``
        # and whose final MILP merely failed to certify has not exhausted its wall clock, so
        # the message must name the columns, not the (absent) time limit.
        (
            {
                "termination_reason": "ip_not_proven", "iterations": 12,
                "ip_status": "iteration_limit", "n_columns": 431,
            },
            lambda: ColGenParams(time_limit_s=45.0),
            ("without proving optimality over its 431 columns",),
            "stopped on its time limit",
        ),
        # The iteration cap is a truncated exit like ``time_limit`` -- every denial becomes
        # SEARCH_EXHAUSTED and the schedule is uncertified -- yet it was the one that said
        # nothing, which is the failure the other warnings exist to prevent.
        (
            {"termination_reason": "iteration_limit", "iterations": 30},
            lambda: ColGenParams(max_iterations=30),
            ("iteration cap (30)",),
            None,
        ),
        # ``lp_gap`` and ``heuristic_gap`` are different quantities against different
        # thresholds. The LP's cost gap has closed; the HEURISTIC's has not. Keying on the
        # former would stay silent on exactly the run worth warning about.
        (
            {
                "termination_reason": "heuristic_gap", "iterations": 3,
                "lp_gap_cost": 1e-9, "heuristic_gap_cost": 0.42,
            },
            lambda: ColGenParams(gap_metric="revenue", ip_gap=1e-3),
            ("stopped on the revenue-scale heuristic_gap", "still 0.42", "threshold 0.001"),
            None,
        ),
    ],
    ids=[
        "budget", "revenue_gap", "converged",
        "uncertified_ip", "iteration_cap", "heuristic_gap",
    ],
)
def test_solve_termination_is_reported_faithfully(
    monkeypatch, caplog, stats, make_params, expected, forbidden
):
    """Every truncated or early exit is announced against the right scale, a genuinely
    converged solve stays quiet, and no warning names a cause the run does not have.

    A finished run folder looks the same whichever way a solve stopped; the only place the
    difference surfaces is the log, at the moment the run could still be relaunched.
    ``expected`` is the set of substrings that must co-occur in one record; an empty
    ``expected`` asserts silence; ``forbidden`` names a cause the message must not claim.
    """

    cfg = _cfg()
    monkeypatch.setattr(
        batch.ColGenSolver, "solve",
        lambda self, requests, solve_cfg, static_terms, params, on_iteration=None, **_kw: ColGenResult(
            columns={}, stats=stats,
        ),
    )

    with caplog.at_level(logging.WARNING, logger=batch.__name__):
        run_batch(
            scenario_from_requests(_requests()), cfg, ReservationLedger(cfg),
            _RecordingDSS(),  # type: ignore[arg-type]
            (), lambda *_args: None, None, None,
            params=make_params(),
        )

    if expected:
        assert any(
            all(substring in record.message for substring in expected)
            for record in caplog.records
        )
    else:
        assert not caplog.records, "a converged solve must not cry wolf"

    if forbidden is not None:
        assert not any(forbidden in record.message for record in caplog.records), (
            "the budget message names a cause this run does not have"
        )
