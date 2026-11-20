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
from freespace_sim.planner.turn_rule import path_turns_valid
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
    params = ColGenParams(solver="highs", max_air_overrun_hops=0, time_limit_s=30.0)

    intents, _ledger = _run_direct_batch(requests, cfg, params)

    assert len(intents) == len(requests)
    assert all(intent.accepted for intent in intents)
    assert all(intent.denial_reason is DenialReason.NONE for intent in intents)
    assert sorted(total_delay_s(intent, cfg) for intent in intents) == pytest.approx([0.0, 16.0])
    assert verify.find_interflight_conflict(intents, cfg) is None


def test_dp_turn_rule_batch_is_safe_and_worker_count_independent():
    """Production DP with the opt-in strict-turn domain: every pooled column obeys it.

    On this instance the free solve pools and selects a route turning 90 degrees or more,
    so the rule binds. With it, serial and two-worker solves return the same schedule and
    pool, and the schedule remains accepted and collision-free.
    """
    from freespace_sim.planner.colgen.solver import ColGenSolver
    from freespace_sim.planner.colgen.translate import column_to_intent

    cfg = SimConfig(
        planner="colgen",
        flight_levels_m=(100.0,),
        airspace_ceiling_m=125.0,
        region_size_m=(20_000.0, 20_000.0),
        max_ground_delay_s=8.0,
    )
    requests = [
        FlightRequest(index + 1, _point(origin, cfg), _point(dest, cfg), 0.0, 0.0)
        for index, (origin, dest) in enumerate((((-3, 0), (3, 0)), ((0, -3), (0, 3)),
                                                ((-3, 3), (3, -3)), ((3, 0), (-3, 0))))
    ]
    params = ColGenParams(solver="highs", max_air_overrun_hops=3, time_limit_s=30.0)
    pools = []
    free = ColGenSolver().solve(
        requests, cfg, (), params,
        on_iteration=lambda state: pools.append(state["master"]) if not pools else None,
    )
    assert any(not path_turns_valid(c.cell_path) for c in pools[0].columns)
    outcomes = []
    for workers in (0, 2):
        masters = []
        result = ColGenSolver().solve(
            requests, cfg, (), replace(params, turn_below_90=True, n_pricing_workers=workers),
            on_iteration=lambda state: masters.append(state["master"]) if not masters else None,
        )
        assert masters and all(path_turns_valid(c.cell_path) for c in masters[0].columns)
        assert result.columns
        by_id = {request.flight_id: request for request in requests}
        intents = [column_to_intent(column, by_id[fid], cfg)
                   for fid, column in result.columns.items()]
        assert all(intent.accepted for intent in intents)
        assert verify.find_interflight_conflict(intents, cfg) is None
        outcomes.append((result.columns, tuple(masters[0].columns)))
    assert outcomes[0] == outcomes[1]
    assert free.columns


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
    params = ColGenParams(solver="highs", max_air_overrun_hops=0, time_limit_s=30.0)

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


@pytest.mark.parametrize("shared_hub", [False, True], ids=["crossing", "terminal-pads"])
@pytest.mark.parametrize("backend,lazy,workers,prune_snap,early_return,completion_bounds", [
    ("reference", False, 0, False, False, ()), ("reference", True, 0, False, False, ()),
    ("compiled", False, 0, False, False, ()), ("compiled", True, 0, False, False, ()),
    ("compiled", True, 2, False, False, ()), ("compiled", True, 2, True, False, ()),
    ("reference", True, 0, False, True, ()), ("compiled", True, 0, False, True, ()),
    ("compiled", True, 2, True, True, ()),
    ("compiled", True, 2, True, True, ("arrival", "resource", "selected", "crossing")),
    ("reference", True, 0, True, True, ("arrival", "resource", "selected", "crossing")),
    ("compiled-long24", True, 2, True, True, ()),
    ("compiled-merge12", True, 2, True, True, ()),
    ("compiled-merge240", True, 2, True, True, ()),
    ("compiled-merge12-forward", True, 2, True, True, ()),
    ("compiled-merge12-forward-dp", True, 2, True, True, ()),
    ("compiled-merge240-forward", True, 2, True, True, ()),
    ("compiled-forward", True, 2, True, True, ()),
    ("compiled-merge12-lattice", True, 2, True, True, ()),
    ("compiled-merge240-lattice", True, 2, True, True, ()),
    ("compiled-indexed", True, 2, True, True, ()),
    ("compiled-merge240-indexed", True, 2, True, True, ()),
    ("compiled-merge12-forward-dp-indexed", True, 2, True, True, ()),
    ("compiled-merge12-forward-dp-indexed-turn", True, 2, True, True, ()),
    ("compiled-merge240-indexed-turn", True, 2, True, True, ()),
    ("reference-turn", True, 0, True, True, ()),
], ids=["reference-eager", "reference-lazy", "compiled-eager", "compiled-lazy", "parallel",
        "parallel-pruned-snapped", "early-reference", "early-compiled", "early-parallel", "all-bounds-parallel", "all-bounds-reference",
        "long24-parallel", "merged-12-parallel", "merged-240-parallel",
        "forward-12-parallel", "forward-dp-parallel", "forward-240-parallel", "forward-full-parallel",
        "lattice-12-parallel", "lattice-240-parallel", "indexed-full-parallel",
        "indexed-radius-parallel", "indexed-forward-dp-parallel",
        "turn-indexed-forward-dp-parallel", "turn-indexed-radius-parallel", "turn-reference"])
def test_zeta_prototype_files_full_volume_claims_and_respects_terminal_capacity(
    shared_hub, backend, lazy, workers, prune_snap, early_return, completion_bounds,
):
    """Any-angle selected routes preserve physical safety, objective and terminal capacity."""
    from freespace_sim.planner.colgen.objective import cost_model
    from freespace_sim.planner.colgen.zeta import ZetaOptions
    from freespace_sim.planner.zeta.geometry import reservation_claims
    from freespace_sim.planner.colgen.zeta_solver import solve_zeta

    turn = backend.endswith("-turn")
    backend = backend.removesuffix("-turn")
    native_state_index = backend.endswith("-indexed")
    backend = backend.removesuffix("-indexed")
    forward_dp = backend.endswith("-dp")
    backend = backend.removesuffix("-dp")
    lattice = backend.endswith("-lattice")
    backend = backend.removesuffix("-lattice")
    forward = backend.endswith("-forward")
    backend = backend.removesuffix("-forward")
    restricted = backend.endswith("-long24")
    merge_equivalent = backend in {"compiled-merge12", "compiled-merge240"}
    merged12 = backend.endswith("-merge12")
    radius = 240.0 if backend.endswith("-merge240") else 480.0
    if merge_equivalent:
        backend = "compiled"
    backend = backend.removesuffix("-long24")
    if backend == "compiled":
        pytest.importorskip("numba")
    if native_state_index:
        pytest.importorskip("Cython")
    cfg = SimConfig(
        planner="colgen", flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
        region_size_m=(6000.0, 6000.0), max_ground_delay_s=64.0,
        terminal_airspace_always_active=True, fixed_exit_lanes=True,
    )
    terms = ()
    if shared_hub:
        hub = (3000.0, 3000.0)
        terminal = Terminal("zeta-H", capacity=2, radius=180.0)
        requests = [
            replace(request, dest=vec(
                hub[0] + 720.0 * np.cos(2.0 * np.pi * index / 3),
                hub[1] + 720.0 * np.sin(2.0 * np.pi * index / 3), 0.0,
            ))
            for index, request in enumerate(_radial_deliveries(hub, terminal, 3))
        ]
        terms = ((vec(*hub, 0.0), terminal),)
    else:
        requests = [
            FlightRequest(1, _point((-2, 0), cfg), _point((2, 0), cfg), 0.0, 0.0),
            FlightRequest(2, _point((0, -2), cfg), _point((0, 2), cfg), 0.0, 0.0),
        ]
    params = ColGenParams(
        solver="highs", max_air_overrun_hops=0, time_limit_s=25.0,
        ip_time_limit_s=3.0, ip_reserve_s=4.0, iteration_ip_time_limit_s=0.0,
        n_heuristic_tries=4, n_pricing_workers=workers, turn_below_90=turn,
    )
    options = ZetaOptions(connection_radius_m=radius, max_expansions=1000, max_labels=10000,
                          backend=backend, lazy_successors=lazy, expire_history=prune_snap,
                          history_dominance=prune_snap, arrival_snap_s=float(prune_snap),
                          early_return=early_return, completion_bounds=completion_bounds,
                          connection_mode="hex12" if merged12 else "hex12_long" if restricted else "full",
                          strong_history_dominance=merge_equivalent,
                          merge_equivalent_histories=merge_equivalent,
                          forward_cruise=forward,
                          traversal="forward_dp" if forward_dp else "best_first",
                          lattice_distance_bound=lattice,
                          compiled_loop=merge_equivalent or native_state_index,
                          compiled_history=merge_equivalent,
                          native_state_index=native_state_index, turn_below_90=turn)
    result = solve_zeta(
        requests, cfg, terms, params, options=options,
    )
    assert result.stats["turn_below_90"] is turn
    with pytest.raises(ValueError, match="must agree"):
        solve_zeta(requests, cfg, terms, replace(params, turn_below_90=not turn), options=options)
    if turn:
        assert "turns strictly below 90 degrees" in result.stats["certificate_scope"]
        assert all(path_turns_valid(column.cell_path) for column in result.master.columns)
    if forward:
        assert result.stats["forward_cruise"]
        assert result.stats["certificate_scope"].startswith("forward-only cruise;")
        by_id = {request.flight_id: request for request in requests}
        for fid, column in result.columns.items():
            direction = by_id[fid].dest[:2] - by_id[fid].origin[:2]
            direction = direction / np.linalg.norm(direction)
            for first, last in zip(column.cell_path, column.cell_path[1:]):
                assert np.dot((_point(last, cfg) - _point(first, cfg))[:2], direction) > 1e-8
    if workers:
        sequential = solve_zeta(
            requests, cfg, terms, replace(params, n_pricing_workers=0),
            options=replace(options, native_state_index=False),
        )
        assert result.master.columns == sequential.master.columns
        assert result.columns == sequential.columns
        for key in ("objective_cost", "global_cost_lower_bound", "termination_reason",
                    "pricing_search_complete", "pricing_bound_certified"):
            assert result.stats[key] == sequential.stats[key]
        assert result.stats["pricing_search_complete"]
    if prune_snap:
        unpruned = solve_zeta(
            requests, cfg, terms, replace(params, n_pricing_workers=0),
            options=replace(options, expire_history=False, history_dominance=False,
                            strong_history_dominance=False, merge_equivalent_histories=False),
        )
        for key in ("objective_cost", "global_cost_lower_bound"):
            assert result.stats[key] == pytest.approx(unpruned.stats[key], abs=1e-6)
        assert unpruned.stats["pricing_search_complete"]
        assert result.stats["pricing_bound_certified"] == unpruned.stats["pricing_bound_certified"]
    assert len(result.columns) == len(requests)
    if early_return:
        assert result.stats["pricing_search_complete"]
        assert result.stats["history"][-1]["pricing_complete"]
        assert result.stats["history"][-1]["new_columns"] == 0
    assert result.stats["violated_rows"] == 0
    assert verify.find_interflight_conflict(list(result.intents.values()), cfg, terms) is None
    ledger = ReservationLedger(cfg)
    for center, terminal in terms:
        ledger.register_static_terminal(center, terminal)
    dss = DSS(ledger, FCFSMechanism())
    for intent in result.intents.values():
        assert dss.commit(intent)
    protected = frozenset(terminal.id for _point_, terminal in terms)
    for flight_id, column in result.columns.items():
        intent = result.intents[flight_id]
        assert intent.accepted
        assert column.claims == reservation_claims(
            intent.volumes, cfg, row_index=result.master.row_index,
            protected_terminal_ids=protected,
        )
        assert column.delay_s == pytest.approx(cost_model(cfg, params).intent_cost(intent, cfg))
        for (start, before), (end, after) in zip(intent.centerline, intent.centerline[1:]):
            horizontal = float(np.linalg.norm(end[:2] - start[:2]))
            if horizontal > 1e-8 and abs(float(end[2] - start[2])) < 1e-8:
                speed = horizontal / (after - before)
                assert speed >= cfg.nominal_speed_mps * (1.0 - options.speed_fraction) - 1e-8
                assert speed <= cfg.nominal_speed_mps * (1.0 + options.speed_fraction) + 1e-8
    assert result.stats["global_cost_lower_bound"] <= result.stats["objective_cost"] + 1e-6
    if shared_hub:
        dwells = [
            (volume.t_start, volume.t_end)
            for intent in result.intents.values()
            for volume in intent.volumes
            if volume.terminal_id == terminal.id and isinstance(volume.shape, CylinderSpec)
        ]
        assert _max_concurrent(dwells) <= terminal.capacity


@pytest.mark.parametrize("continue_after_improvement", [False, True])
@pytest.mark.parametrize("early_return", [False, True])
@pytest.mark.parametrize("backend,lazy", [
    ("reference", False), ("reference", True), ("compiled", False), ("compiled", True),
], ids=["reference-eager", "reference-lazy", "compiled-eager", "compiled-lazy"])
def test_zeta_prototype_keeps_incumbent_without_claiming_capped_convergence(
    monkeypatch, continue_after_improvement, early_return, backend, lazy,
):
    """A capped pricing sweep may improve the pool but cannot claim exhaustive convergence."""
    from freespace_sim.planner.colgen import zeta_solver
    from freespace_sim.planner.colgen.zeta import ZetaOptions
    from freespace_sim.planner.colgen.translate import Column

    if backend == "compiled":
        pytest.importorskip("numba")
    cfg = SimConfig(
        planner="colgen", flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
        max_ground_delay_s=8.0,
    )
    requests = [FlightRequest(1, _point((0, 0), cfg), _point((2, 0), cfg), 0.0, 0.0)]
    params = ColGenParams(
        solver="highs", max_air_overrun_hops=0, time_limit_s=10.0,
        ip_time_limit_s=1.0, ip_reserve_s=2.0, iteration_ip_time_limit_s=0.0,
    )
    real_price = zeta_solver.price_zeta
    calls = 0

    def capped(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_price(*args, **kwargs)
        if continue_after_improvement and calls > 1:
            return result
        return replace(result, complete=False, reason="test_work_cap", upper_bound=result.upper_bound + 1.0)

    monkeypatch.setattr(zeta_solver, "price_zeta", capped)
    supplied = None
    if continue_after_improvement:
        params = replace(params, seed_nominal_routes=False, seed_ladder_steps=0)
        supplied = {1: [Column(1, 1, 0, None, None, ((0, 0), (2, 0)), 4.0)]}
    result = zeta_solver.solve_zeta(
        requests, cfg, (), params,
        options=ZetaOptions(backend=backend, lazy_successors=lazy, early_return=early_return),
        seed_columns=supplied,
    )
    assert len(result.columns) == 1
    assert result.stats["termination_reason"] == ("converged" if continue_after_improvement else "pricing_limit")
    assert result.stats["pricing_search_complete"] is continue_after_improvement
    assert result.stats["pricing_bound_certified"] is True
    assert result.stats["capped_searches"] == 1
    assert result.stats["global_cost_lower_bound"] <= result.stats["objective_cost"] + 1e-6
    if continue_after_improvement:
        assert calls >= 2 and result.columns[1].departure_step == 0


@pytest.mark.parametrize("failure", ["worker_lost", "deadline"])
@pytest.mark.parametrize("phase", ["seed", "price"])
def test_zeta_solver_does_not_certify_a_partial_parallel_sweep(monkeypatch, failure, phase):
    """Partial seeds or prices retain feasible routes without certifying missing flights."""
    from freespace_sim.planner.colgen import zeta_solver
    from freespace_sim.planner.colgen.zeta import ZetaPricingResult
    from freespace_sim.planner.colgen.zeta_pool import (
        ZetaSeedSweepResult, ZetaSweepResult, build_seed_columns,
    )

    cfg = SimConfig(planner="colgen", flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
                    max_ground_delay_s=8.0)
    requests = [
        FlightRequest(1, _point((0, 0), cfg), _point((2, 0), cfg), 0.0, 0.0),
        FlightRequest(2, _point((5, 0), cfg), _point((7, 0), cfg), 0.0, 0.0),
    ]
    params = ColGenParams(solver="highs", n_pricing_workers=2, max_air_overrun_hops=0,
                          time_limit_s=10.0, ip_time_limit_s=1.0, ip_reserve_s=2.0,
                          iteration_ip_time_limit_s=0.0)
    received = ZetaPricingResult(None, 0.0, 0.0, True, "optimal", 0, 0, 0.0)
    sweep = ZetaSweepResult(((1, received),), False, failure, 0.0, 0.0, 0.0)
    closed = []

    def make_pool(graphs, worker_params):
        """Return one seed or pricing prefix without relying on clock-sensitive scheduling."""
        seeds = tuple((fid, build_seed_columns(graph, worker_params))
                      for fid, graph in sorted(graphs.items()))
        seed_sweep = ZetaSeedSweepResult(
            seeds[:1] if phase == "seed" else seeds,
            phase != "seed", failure if phase == "seed" else None, 0.0, 0.0, 0.0,
        )
        return SimpleNamespace(
            run_seed_sweep=lambda *_args, **_kwargs: seed_sweep,
            run_sweep=lambda *_args, **_kwargs: sweep,
            close=lambda: closed.append(True), setup_s=0.0, close_s=0.0,
            worker_peak_rss_bytes={},
        )

    monkeypatch.setattr(zeta_solver, "ZetaPricingPool", make_pool)
    result = zeta_solver.solve_zeta(requests, cfg, (), params)
    assert closed
    assert len(result.columns) == (1 if phase == "seed" else len(requests))
    assert result.stats["termination_reason"] == ("seed_" if phase == "seed" else "pricing_") + failure
    assert not result.stats["pricing_search_complete"]
    assert not result.stats["pricing_bound_certified"]
    assert result.stats["global_cost_lower_bound"] == 0.0
    if phase == "seed":
        assert result.stats["history"] == []
    else:
        assert result.stats["history"][0]["search_statuses"] == {1: "optimal"}


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
    params = ColGenParams(solver="highs", max_air_overrun_hops=0, time_limit_s=30.0)
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
        ColGenParams(solver="highs", max_air_overrun_hops=0),
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
        max_air_overrun_hops=0,
        max_iterations=10,
        time_limit_s=30.0,
    )

    result = run(cfg, requests=requests, demand=demand, progress=False, planner_params=params)

    assert result.verified
    assert len(result.intents) == len(requests)
    assert all(intent.accepted for intent in result.intents)
    # No ground-delay assertion: colgen_test is one-way now, halving the load its lam_per_uss was
    # calibrated for, so at the old λ nothing is held. Restoring it needs the scenario re-tuned.
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

    # `solver="highs"` rather than the "auto" default: the two backends return different
    # optimal dual vertices of a degenerate master, and on this world Gurobi's close the
    # revenue gap at iteration 1 where HiGHS's run to the budget. Left on auto, this
    # acceptance fixture measures a 1-iteration solve on a machine with Gurobi installed
    # and a ~17-iteration one everywhere else. HiGHS is always available (SciPy), so
    # pinning it is what makes the assertions below mean the same thing for everyone;
    # `test_backend_parity` is where Gurobi is exercised.
    solver_time_limit_s = 100.0
    started = time.monotonic()
    colgen = run(
        cfg, requests=requests, demand=demand, progress=False,
        planner_params=ColGenParams(solver="highs", time_limit_s=solver_time_limit_s),
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
@pytest.mark.xfail(strict=True, reason=(
    "colgen_test is one-way (run_batch refuses a round-trip itinerary), halving the load its "
    "lam_per_uss was calibrated for: 49 requests where this wants >=80. Re-tuning the scenario is a "
    "research decision, not a threshold to lower. STRICT so it fails once someone recalibrates."))
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
    # What is contractual is that the solver honours the budget it was given, which it
    # reports itself. A wall-clock ceiling here would instead assert that the machine is
    # fast -- flaky under load, and silent about the thing that could actually regress.
    stats = data.colgen.planner_stats
    assert stats["termination_reason"] in {"time_limit", "lp_gap", "heuristic_gap",
                                           "no_improving_columns", "no_new_columns",
                                           "iteration_limit", "ip_not_proven"}
    assert stats["time_limit_overrun_s"] < 0.25 * data.solver_time_limit_s, (
        f"solve ran {stats['elapsed_s']:.1f}s against a {data.solver_time_limit_s:.1f}s cap; "
        "a single synchronous geometry/backend call cannot be preempted, so a small "
        "overrun is expected and a large one means a stage stopped checking its deadline"
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
        max_air_overrun_hops=0,
        max_iterations=10,
        time_limit_s=30.0,
    )

    result = run(cfg, requests=requests, demand=demand, progress=False, planner_params=params)

    assert result.verified
    assert len(result.accepted) == len(requests)
    assert not result.denied
