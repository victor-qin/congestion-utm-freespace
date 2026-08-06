"""End-to-end contracts for batch column generation and continuous filing."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from freespace_sim import verify
from freespace_sim.config import SimConfig
from freespace_sim.dss import DSS
from freespace_sim.geometry import CylinderSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.mechanism import FCFSMechanism
from freespace_sim.metrics import total_delay_s
from freespace_sim.planner import get_planner, hexgrid as hg
from freespace_sim.planner.colgen.batch import run_batch
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.scenario import scenario_from_requests
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.scenarios.colgen import COLGEN_USS
from freespace_sim.sim import _wall_aware, run
from freespace_sim.types import DenialReason, FlightRequest, Terminal, vec


def _point(cell: tuple[int, int], cfg: SimConfig):
    x, y = hg.hex_center(*cell, hg.circumradius(cfg))
    return vec(x, y, cfg.ground_level_m)


def _run_direct_batch(
    requests: list[FlightRequest],
    cfg: SimConfig,
    params: ColGenParams,
    static_terms=(),
):
    ledger = ReservationLedger(cfg)
    terms = tuple(static_terms)
    for center, terminal in terms:
        ledger.register_static_terminal(center, terminal)
    dss = DSS(ledger, FCFSMechanism())
    intents, _stats = run_batch(
        scenario_from_requests(requests),
        cfg,
        ledger,
        dss,
        terms,
        lambda *_args: None,
        None,
        None,
        params=params,
    )
    return intents, ledger


def _max_concurrent(intervals: list[tuple[float, float]]) -> int:
    events = sorted([(start, 1) for start, _end in intervals] + [(end, -1) for _start, end in intervals])
    current = maximum = 0
    for _time, delta in events:
        current += delta
        maximum = max(maximum, current)
    return maximum


def _radial_deliveries(
    hub: tuple[float, float],
    terminal: Terminal,
    count: int,
) -> list[FlightRequest]:
    requests = []
    for flight_id in range(count):
        angle = 2.0 * np.pi * flight_id / count
        destination = vec(
            hub[0] + 2_000.0 * np.cos(angle),
            hub[1] + 2_000.0 * np.sin(angle),
            0.0,
        )
        requests.append(
            FlightRequest(
                flight_id,
                vec(*hub, 0.0),
                destination,
                0.0,
                0.0,
                origin_terminal=terminal,
            )
        )
    return requests


def test_real_crossing_batch_files_every_selected_column():
    cfg = SimConfig(
        planner="colgen",
        flight_levels_m=(100.0,),
        airspace_ceiling_m=125.0,
        region_size_m=(20_000.0, 20_000.0),
        max_ground_delay_s=32.0,
    )
    requests = [
        FlightRequest(1, _point((-4, 0), cfg), _point((4, 0), cfg), 0.0, 0.0),
        FlightRequest(2, _point((0, -4), cfg), _point((0, 4), cfg), 0.0, 0.0),
    ]
    params = ColGenParams(solver="highs", detour_slack_hops=0, time_limit_s=30.0)

    intents, _ledger = _run_direct_batch(requests, cfg, params)

    assert len(intents) == len(requests)
    assert all(intent.accepted for intent in intents)
    assert all(intent.denial_reason is DenialReason.NONE for intent in intents)
    assert sorted(total_delay_s(intent, cfg) for intent in intents) == pytest.approx([0.0, 16.0])
    assert verify.find_interflight_conflict(intents, cfg) is None


def test_colgen_terminal_rows_enforce_pad_capacity():
    cfg = SimConfig(
        planner="colgen",
        region_size_m=(6_000.0, 6_000.0),
        flight_levels_m=(100.0,),
        airspace_ceiling_m=125.0,
        terminal_airspace_always_active=True,
        max_ground_delay_s=64.0,
    )
    hub = (3_000.0, 3_000.0)
    terminal = Terminal("H", capacity=2, radius=180.0)
    requests = _radial_deliveries(hub, terminal, 3)
    params = ColGenParams(solver="highs", detour_slack_hops=0, time_limit_s=30.0)

    intents, _ledger = _run_direct_batch(
        requests,
        cfg,
        params,
        ((vec(*hub, 0.0), terminal),),
    )

    assert len(intents) == len(requests)
    assert all(intent.accepted for intent in intents)
    assert sum(intent.ground_delay_s == 0.0 for intent in intents) == terminal.capacity
    dwells = [
        (volume.t_start, volume.t_end)
        for intent in intents
        for volume in intent.volumes or ()
        if volume.terminal_id == terminal.id and isinstance(volume.shape, CylinderSpec)
    ]
    assert dwells
    assert _max_concurrent(dwells) <= terminal.capacity
    assert verify.find_interflight_conflict(
        intents,
        cfg,
        static_terminals=((vec(*hub, 0.0), terminal),),
    ) is None


def test_budget_denial_is_not_logged_as_a_covering_bug(caplog):
    cfg = SimConfig(
        planner="colgen",
        region_size_m=(6_000.0, 6_000.0),
        flight_levels_m=(100.0,),
        airspace_ceiling_m=125.0,
        terminal_airspace_always_active=True,
        max_ground_delay_s=0.0,
    )
    hub = (3_000.0, 3_000.0)
    terminal = Terminal("H", capacity=1, radius=180.0)
    requests = _radial_deliveries(hub, terminal, 2)
    params = ColGenParams(solver="highs", detour_slack_hops=0, time_limit_s=30.0)
    caplog.set_level(logging.ERROR, logger="freespace_sim.planner.colgen.batch")

    intents, _ledger = _run_direct_batch(
        requests,
        cfg,
        params,
        ((vec(*hub, 0.0), terminal),),
    )

    reasons = Counter(intent.denial_reason for intent in intents)
    assert reasons == Counter({DenialReason.NONE: 1, DenialReason.BUDGET_EXCEEDED: 1})
    assert "covering bug" not in caplog.text


def test_colgen_is_wall_aware_and_customer_batch_needs_no_terminal_walls():
    assert _wall_aware(get_planner("colgen"))
    cfg = SimConfig(
        planner="colgen",
        flight_levels_m=(100.0,),
        airspace_ceiling_m=125.0,
        region_size_m=(4_000.0, 4_000.0),
        terminal_airspace_always_active=False,
        max_ground_delay_s=0.0,
    )
    request = FlightRequest(1, vec(500.0, 500.0, 0.0), vec(2_500.0, 500.0, 0.0), 0.0, 0.0)

    intents, _ledger = _run_direct_batch(
        [request],
        cfg,
        ColGenParams(solver="highs", detour_slack_hops=0),
    )

    assert len(intents) == 1 and intents[0].accepted


def test_colgen_test_fast_real_batch_smoke():
    spec = with_overrides(
        get_scenario("colgen_test"),
        planner="colgen",
        horizon_s=600.0,
        demand_duration_s=4.0,
    )
    cfg = replace(spec.config(), max_ground_delay_s=64.0)
    demand = spec.demand_model()
    requests = demand.generate(cfg, np.random.default_rng(cfg.seed))
    assert 0 < len(requests) <= 20
    params = ColGenParams(
        solver="highs",
        detour_slack_hops=0,
        max_iterations=10,
        time_limit_s=30.0,
    )

    result = run(cfg, requests=requests, demand=demand, progress=False, planner_params=params)

    assert result.verified
    assert len(result.intents) == len(requests)
    assert all(intent.accepted for intent in result.intents)
    assert any(intent.ground_delay_s > 0.0 for intent in result.intents)
    assert not {
        DenialReason.CONFLICT_FILED,
        DenialReason.CONFLICT_AT_COMMIT,
    }.intersection(intent.denial_reason for intent in result.intents)


@pytest.fixture(scope="module")
def full_colgen_test_results():
    """Run the acceptance world once per module and share it across slow assertions."""

    spec = with_overrides(get_scenario("colgen_test"), planner="colgen")
    cfg = spec.config()
    demand = spec.demand_model()
    requests = demand.generate(cfg, np.random.default_rng(cfg.seed))

    solver_time_limit_s = 100.0
    started = time.monotonic()
    colgen = run(
        cfg, requests=requests, demand=demand, progress=False,
        planner_params=ColGenParams(time_limit_s=solver_time_limit_s),
    )
    colgen_wall_s = time.monotonic() - started

    astar_cfg = replace(cfg, planner="astar")
    started = time.monotonic()
    astar = run(astar_cfg, requests=requests, demand=demand, progress=False)
    astar_wall_s = time.monotonic() - started
    return SimpleNamespace(
        spec=spec,
        requests=requests,
        colgen=colgen,
        astar=astar,
        colgen_wall_s=colgen_wall_s,
        astar_wall_s=astar_wall_s,
        solver_time_limit_s=solver_time_limit_s,
    )


@pytest.mark.slow
def test_colgen_runs_full_density_miniature_without_filing_denials(full_colgen_test_results):
    data = full_colgen_test_results
    result = data.colgen

    assert len(data.requests) >= 80
    assert len(
        {
            terminal.id
            for request in data.requests
            for terminal in (request.origin_terminal, request.dest_terminal)
            if terminal is not None
        }
    ) >= 2
    assert result.verified
    assert len(result.accepted) == len(data.requests)
    assert not result.denied
    assert data.colgen_wall_s < 120.0, (
        f"colgen_test took {data.colgen_wall_s:.3f}s "
        f"with a {data.solver_time_limit_s:.3f}s solver cap (wall budget 120.000s)"
    )
    assert any(intent.ground_delay_s > 0.0 for intent in result.accepted)
    assert not {
        DenialReason.CONFLICT_FILED,
        DenialReason.CONFLICT_AT_COMMIT,
    }.intersection(intent.denial_reason for intent in result.intents)

    terminal_capacity = data.spec.demand.pads_per_hub[COLGEN_USS]
    by_terminal: dict[object, list[tuple[float, float]]] = {}
    for intent in result.accepted:
        for volume in intent.volumes or ():
            if volume.terminal_id is not None and isinstance(volume.shape, CylinderSpec):
                by_terminal.setdefault(volume.terminal_id, []).append(
                    (volume.t_start, volume.t_end)
                )
    assert by_terminal
    assert all(
        _max_concurrent(intervals) <= terminal_capacity for intervals in by_terminal.values()
    )


@pytest.mark.slow
def test_colgen_not_worse_than_astar_on_full_colgen_test(full_colgen_test_results):
    data = full_colgen_test_results
    assert data.colgen.verified
    assert len(data.colgen.accepted) == len(data.requests)
    assert not data.colgen.denied
    assert data.astar.verified
    assert len(data.astar.accepted) == len(data.requests)
    astar_delay = sum(total_delay_s(intent, data.astar.config) for intent in data.astar.accepted)
    colgen_delay = sum(total_delay_s(intent, data.colgen.config) for intent in data.colgen.accepted)

    assert astar_delay > 0.0
    assert colgen_delay <= astar_delay * 1.05


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(5))
def test_colgen_small_hub_seed_sweep_has_no_filing_denials(seed):
    spec = with_overrides(
        get_scenario("colgen_test"),
        planner="colgen",
        seed=seed,
        horizon_s=600.0,
        demand_duration_s=12.0,
        demand_overrides={"radius_m": {COLGEN_USS: 600.0}},
    )
    cfg = replace(spec.config(), max_ground_delay_s=64.0)
    demand = spec.demand_model()
    requests = demand.generate(cfg, np.random.default_rng(seed))
    assert requests
    params = ColGenParams(
        solver="highs",
        detour_slack_hops=0,
        max_iterations=10,
        time_limit_s=30.0,
    )

    result = run(cfg, requests=requests, demand=demand, progress=False, planner_params=params)

    assert result.verified
    assert len(result.accepted) == len(requests)
    assert not result.denied
