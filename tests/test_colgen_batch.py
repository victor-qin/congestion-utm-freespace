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

    def fake_solve(self, requests, solve_cfg, static_terms, solve_params):
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
        lambda self, requests, solve_cfg, static_terms, params: ColGenResult(
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
