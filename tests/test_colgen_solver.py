"""Phase-2 contracts for pricing, the restricted master, and orchestration.

The end-to-end fixtures use customer endpoints on exact hex centres.  That removes
lane and snap constants from the arithmetic, so every asserted delay is simply an
integer number of four-second lattice/ground steps.  The continuous ledger remains
the independent referee for selected columns.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

import freespace_sim.planner.colgen.master as master_module
import freespace_sim.planner.colgen.solver as solver_module
from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.metrics import total_delay_s
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.colgen.master import (
    BackendIpResult,
    HighsBackend,
    RestrictedMaster,
    create_backend,
)
from freespace_sim.planner.colgen.network import RowIndex, RowKey, build_flight_graph, column_claims
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import (
    DualView,
    PricingTimeout,
    _visit_claims,
    price_flight,
    seed_column,
)
from freespace_sim.planner.colgen.solver import (
    ColGenSolver,
    _fixed_loads,
    _greedy_feasible_selection,
    _initial_feasible_selection,
    _shift_claims,
)
from freespace_sim.planner.colgen.translate import Column, column_to_intent
from freespace_sim.planner.colgen.windows import (
    derive_cell_window,
    endpoint_claim_cells,
    visit_rows,
)
from freespace_sim.types import FlightRequest, IntentStatus, Terminal, vec


def _cfg(**overrides) -> SimConfig:
    values = {
        "planner": "colgen",
        "flight_levels_m": (100.0,),
        "airspace_ceiling_m": 125.0,
        "region_size_m": (20_000.0, 20_000.0),
        "terminal_airspace_always_active": True,
        "max_ground_delay_s": 48.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _point(cell: tuple[int, int], cfg: SimConfig):
    x, y = hg.hex_center(*cell, hg.circumradius(cfg))
    return vec(x, y, cfg.ground_level_m)


def _request(
    flight_id: int,
    origin: tuple[int, int],
    destination: tuple[int, int],
    cfg: SimConfig,
    *,
    departure: float = 0.0,
) -> FlightRequest:
    return FlightRequest(
        flight_id,
        _point(origin, cfg),
        _point(destination, cfg),
        departure,
        departure,
    )


def _params(**overrides) -> ColGenParams:
    values = {
        "solver": "highs",
        "detour_slack_hops": 0,
        "max_iterations": 30,
        "time_limit_s": 30.0,
        "n_heuristic_tries": 16,
    }
    values.update(overrides)
    return ColGenParams(**values)


def _synthetic_column(
    flight_id: int,
    delay_s: float,
    claims: frozenset[RowKey],
    *,
    departure_step: int = 0,
) -> Column:
    return Column(
        flight_id,
        departure_step,
        0,
        None,
        None,
        ((0, 0), (1, 0)),
        delay_s,
        claims,
    )


def _assert_claim_feasible(columns: dict[int, Column], rows: RowIndex | None = None) -> None:
    rows = RowIndex() if rows is None else rows
    loads = Counter(row for column in columns.values() for row in column.claims)
    assert all(load <= rows.cap(row) for row, load in loads.items())


def _assert_files_cleanly(
    requests: list[FlightRequest], columns: dict[int, Column], cfg: SimConfig
) -> None:
    by_id = {request.flight_id: request for request in requests}
    ledger = ReservationLedger(cfg)
    for flight_id, column in sorted(columns.items()):
        intent = column_to_intent(column, by_id[flight_id], cfg)
        assert intent.status is IntentStatus.ACCEPTED
        assert column.delay_s == pytest.approx(total_delay_s(intent, cfg), abs=1e-9)
        assert not ledger.any_conflict(intent.volumes)
        ledger.commit(flight_id, intent.volumes)


def _assert_endpoint_cells_pairwise_disjoint(requests: list[FlightRequest], cfg: SimConfig) -> None:
    endpoint_sets = [
        set(endpoint_claim_cells(point, cfg.effective_hover_radius_m, cfg))
        for request in requests
        for point in (request.origin, request.dest)
    ]
    assert all(
        first.isdisjoint(second)
        for index, first in enumerate(endpoint_sets)
        for second in endpoint_sets[index + 1 :]
    )


def _geodesics(
    start: tuple[int, int], destination: tuple[int, int]
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Enumerate shortest paths for the tiny hand-checked fixtures."""

    if start == destination:
        return ((start,),)
    remaining = hg.hex_distance(start, destination)
    paths = []
    for dq, dr in hg.AXIAL_NEIGHBORS:
        neighbour = start[0] + dq, start[1] + dr
        if hg.hex_distance(neighbour, destination) != remaining - 1:
            continue
        paths.extend((start, *tail) for tail in _geodesics(neighbour, destination))
    return tuple(paths)


def test_solver_params_validate_phase2_controls():
    params = ColGenParams(solver="HiGhS", max_iterations=np.int64(4))
    assert params.solver == "highs"
    assert params.max_iterations == 4

    with pytest.raises(ValueError, match="solver"):
        ColGenParams(solver="cbc")
    with pytest.raises(ValueError, match="max_iterations"):
        ColGenParams(max_iterations=0)
    with pytest.raises(TypeError, match="max_iterations"):
        ColGenParams(max_iterations=True)
    with pytest.raises(ValueError, match="lp_gap"):
        ColGenParams(lp_gap=1.0)
    with pytest.raises(ValueError, match="epsilon"):
        ColGenParams(epsilon=0.5)


def test_seed_and_zero_dual_pricing_match_exact_delay_contract():
    cfg = _cfg()
    request = _request(1, (-4, 0), (4, 0), cfg, departure=101.3)
    graph = build_flight_graph(request, cfg, (), _params())

    seed = seed_column(graph, cfg)
    reduced_cost, priced = price_flight(graph, {}, 0.0, cfg, _params())

    assert priced is not None
    assert seed.departure_step == graph.base_step
    assert len(seed.cell_path) == graph.shortest_hops + 1
    assert seed.delay_s == pytest.approx(0.0, abs=1e-9)
    assert priced.delay_s == pytest.approx(seed.delay_s, abs=1e-9)
    assert reduced_cost == pytest.approx(_params().M - priced.delay_s, abs=1e-7)
    assert priced.claims


def test_hub_fold_legs_and_terminal_rows_are_priced_exactly():
    cfg = _cfg(max_ground_delay_s=16.0)
    origin = _point((-8, 0), cfg)
    destination = _point((8, 0), cfg)
    origin_terminal = Terminal("A", 2, radius=90.0)
    destination_terminal = Terminal("B", 2, radius=90.0)
    request = FlightRequest(
        30,
        origin,
        destination,
        0.0,
        0.0,
        origin_terminal=origin_terminal,
        dest_terminal=destination_terminal,
    )
    params = _params(detour_slack_hops=4)
    graph = build_flight_graph(
        request,
        cfg,
        ((origin, origin_terminal), (destination, destination_terminal)),
        params,
    )

    reduced_cost, column = price_flight(graph, {}, 0.0, cfg, params)

    assert column is not None
    assert column.origin_lane_idx is not None and column.dest_lane_idx is not None
    assert {row.terminal_id for row in column.claims if row.kind == "term"} == {"A", "B"}
    intent = column_to_intent(column, request, cfg)
    assert column.delay_s == pytest.approx(total_delay_s(intent, cfg), abs=1e-9)
    assert reduced_cost == pytest.approx(params.M - column.delay_s, abs=1e-7)


def test_hub_pruning_does_not_treat_fold_replacement_as_unavoidable_delay():
    """An extra hop may replace terminal folding distance without adding delay."""

    cfg = _cfg(max_ground_delay_s=0.0, max_detour_factor=10.0)
    origin = _point((1, 39), cfg)
    destination = _point((-2, 42), cfg)
    origin_terminal = Terminal("A", 1, radius=90.0)
    destination_terminal = Terminal("B", 1, radius=90.0)
    request = FlightRequest(
        31,
        origin,
        destination,
        0.0,
        0.0,
        origin_terminal=origin_terminal,
        dest_terminal=destination_terminal,
    )
    params = _params(detour_slack_hops=2)
    graph = build_flight_graph(
        request,
        cfg,
        ((origin, origin_terminal), (destination, destination_terminal)),
        params,
    )
    duals = {
        RowKey.cell((1, 38), 0, 9): 1.0,
        RowKey.cell((1, 40), 0, 10): 1.0,
        RowKey.term("B", 10): 10.0,
        RowKey.cell((-1, 41), 0, 9): 30.0,
    }

    reduced_cost, column = price_flight(graph, duals, 0.0, cfg, params)

    assert column is not None
    assert column.origin_lane_idx == 2
    assert column.dest_lane_idx == 4
    assert column.cell_path == (
        (2, 39),
        (1, 40),
        (0, 41),
        (-1, 42),
        (-2, 43),
        (-3, 43),
    )
    assert column.delay_s == pytest.approx(8.0)
    assert column.claims.isdisjoint(duals)
    assert reduced_cost == pytest.approx(params.M - 8.0)


def test_pricing_recomputes_reduced_cost_from_deduplicated_claims():
    cfg = _cfg()
    request = _request(2, (-3, 0), (3, 0), cfg)
    graph = build_flight_graph(request, cfg, (), _params())
    seed = seed_column(graph, cfg)
    # Penalise all rows equally.  Endpoint and visit construction overlap on
    # several keys, so summing arc-local lists instead of the canonical set
    # would make this equality fail.
    duals = {row: 0.25 for row in seed.claims}
    reduced_cost, priced = price_flight(graph, duals, 7.0, cfg, _params())

    assert priced is not None
    expected = (
        _params().M - priced.delay_s - 7.0 - sum(duals.get(row, 0.0) for row in priced.claims)
    )
    assert reduced_cost == pytest.approx(expected, abs=1e-7)
    assert len(priced.claims) == len(set(priced.claims))


@pytest.mark.parametrize("time_buffer_s", [4.0, 0.0])
def test_visit_cost_prefix_sums_equal_the_explicit_claim_sum(time_buffer_s):
    """``DualView`` exposes two ways to price one cell visit; pin them together.

    ``visit_cost`` answers from per-resource prefix sums in O(1); ``claim_cost``
    builds the explicit ``RowKey`` window and sums it.  Pricing has only ever used
    the second, so the first had no caller and no coverage — yet it is the cheap
    form any bulk pricing path wants.  Both offset regimes are exercised because
    ``derive_cell_window`` is asymmetric: ``(-2, 1)`` by default, ``(-1, 0)`` at
    ``time_buffer_s=0``.
    """

    cfg = _cfg(time_buffer_s=time_buffer_s)
    offsets = derive_cell_window(cfg)
    assert offsets == ((-2, 1) if time_buffer_s else (-1, 0))
    rng = np.random.default_rng(20260803)

    for _trial in range(60):
        # A sparse, duplicated, mixed-sign dual map: duplicates exercise the
        # normalizing accumulation, negatives the signed prefix arithmetic.
        duals: dict[RowKey, float] = {}
        for _entry in range(int(rng.integers(1, 14))):
            cell = (int(rng.integers(-3, 4)), int(rng.integers(-3, 4)))
            step = int(rng.integers(-4, 12))
            key = RowKey.cell(cell, 0, step)
            duals[key] = duals.get(key, 0.0) + float(rng.normal(0.0, 5.0))
        view = DualView(duals, cfg)

        for _probe in range(12):
            cell = (int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
            visit_step = int(rng.integers(-6, 14))
            claims = _visit_claims(cell, 0, visit_step, offsets)
            assert len(claims) == offsets[1] - offsets[0] + 1

            assert view.visit_cost(cell, 0, visit_step) == pytest.approx(
                view.claim_cost(claims), abs=1e-12
            )

            # The DP charges a visit net of rows an endpoint already paid, so the
            # prefix form must also support subtracting an arbitrary paid subset.
            paid = frozenset(
                row for row in claims if rng.random() < 0.5
            ) | _visit_claims((9, 9), 0, visit_step, offsets)
            expected = view.claim_cost(claims - paid)
            corrected = view.visit_cost(cell, 0, visit_step) - math.fsum(
                view.row_cost(row) for row in claims & paid
            )
            assert corrected == pytest.approx(expected, abs=1e-12)


def test_zero_buffer_pricing_keeps_predecessor_dependent_endpoint_duals():
    """W=2 labels at one cell cannot merge after paying different endpoint rows."""

    cfg = _cfg(time_buffer_s=0.0, max_ground_delay_s=0.0)
    params = _params()
    request = _request(6, (0, 0), (2, 1), cfg)
    graph = build_flight_graph(request, cfg, (), params)
    duals = {
        RowKey.cell((2, 0), 0, 7): 10.0,
        RowKey.cell((1, 1), 0, 6): 5.0,
    }

    reduced_cost, column = price_flight(graph, duals, 0.0, cfg, params)

    assert column is not None
    assert column.cell_path == ((0, 0), (1, 0), (2, 0), (2, 1))
    assert sum(duals.get(row, 0.0) for row in column.claims) == pytest.approx(10.0)
    assert reduced_cost == pytest.approx(params.M - column.delay_s - 10.0)


def test_pricing_allows_wide_loops_but_no_tight_revisits():
    """Slack sizes the ellipse, not path length; a dual-optimal loop may revisit at W."""

    cfg = _cfg(max_ground_delay_s=16.0)
    params = _params(detour_slack_hops=2)
    request = _request(3, (0, 0), (2, 0), cfg)
    graph = build_flight_graph(request, cfg, (), params)
    duals = {
        RowKey.cell((3, 0), 0, 9): 100.0,
        RowKey.cell((-1, 0), 0, 12): 100.0,
    }

    reduced_cost, column = price_flight(graph, duals, 0.0, cfg, params)

    assert column is not None
    assert column.cell_path == (
        (0, 0),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (0, 0),
        (1, 0),
        (2, 0),
    )
    assert len(column.cell_path) - 1 > graph.shortest_hops + graph.detour_slack_hops
    assert column.delay_s == pytest.approx(16.0)
    assert reduced_cost == pytest.approx(params.M - 16.0)
    assert column.claims.isdisjoint(duals)

    _lo, hi = derive_cell_window(cfg)
    min_repeat = hi - _lo + 1
    positions: dict[tuple[int, int], int] = {}
    for index, cell in enumerate(column.cell_path):
        if cell in positions:
            assert index - positions[cell] >= min_repeat
        positions[cell] = index


def test_master_lazy_lp_and_ip_rows_are_claim_feasible():
    params = _params(M=1000.0)
    row = RowKey.cell((4, -2), 0, 9)
    rows = RowIndex()
    master = RestrictedMaster((1, 2), rows, params, seed=3)
    shared_one = _synthetic_column(1, 0.0, frozenset({row}))
    private_one = _synthetic_column(
        1, 4.0, frozenset({RowKey.cell((10, 0), 0, 9)}), departure_step=1
    )
    shared_two = _synthetic_column(2, 0.0, frozenset({row}))
    private_two = _synthetic_column(
        2, 4.0, frozenset({RowKey.cell((-10, 0), 0, 9)}), departure_step=1
    )
    for column in (shared_one, private_one, shared_two, private_two):
        master.add_column(column)

    objective, _duals, x = master.solve_lp()
    assert objective == pytest.approx(2 * params.M)
    assert master.add_violated_rows(x, 1e-8) == 1

    objective, duals, x = master.solve_lp()
    assert master.add_violated_rows(x, 1e-8) == 0
    assert objective == pytest.approx(2 * params.M - 4.0)
    assert duals[row] >= -1e-9
    assert all(value >= -1e-9 for value in master.flight_duals.values())

    rounded = master.round_heuristic(x, np.random.default_rng(3), 12)
    _assert_claim_feasible(rounded, rows)
    selected = master.solve_ip()
    _assert_claim_feasible(selected, rows)
    assert sum(column.delay_s for column in selected.values()) == pytest.approx(4.0)
    assert master.last_ip_objective == pytest.approx(2 * params.M - 4.0)
    assert master.last_ip_bound == pytest.approx(master.last_ip_objective)
    assert master.last_ip_status == "optimal"
    assert master.last_ip_optimal is True


def test_highs_flight_row_owns_its_maximize_dual():
    """A redundant variable upper bound must not absorb the pricing dual pi_f."""

    params = _params(M=123.0)
    master = RestrictedMaster((9,), RowIndex(), params)
    master.add_column(_synthetic_column(9, 0.0, frozenset()))

    objective, row_duals, x = master.solve_lp()

    assert objective == pytest.approx(123.0)
    assert x == pytest.approx([1.0])
    assert row_duals == {}
    assert master.flight_duals[9] == pytest.approx(123.0)


def test_backend_native_mip_gap_is_scaled_out_of_big_m_revenue():
    params = _params(M=1000.0, ip_gap=0.1)
    backend = create_backend((1, 2), params)

    assert isinstance(backend, HighsBackend)
    assert backend.ip_gap == pytest.approx(params.ip_gap / (2 * params.M))


def test_ip_materializes_a_row_never_added_by_the_caller():
    """An integer winner is rechecked against all claims, not only LP rows."""

    params = _params(M=1000.0)
    shared = RowKey.cell((0, 0), 0, 5)
    rows = RowIndex()
    master = RestrictedMaster((1, 2), rows, params, seed=0)
    for flight_id in (1, 2):
        master.add_column(_synthetic_column(flight_id, 0.0, frozenset({shared})))
        master.add_column(
            _synthetic_column(
                flight_id,
                8.0,
                frozenset({RowKey.cell((flight_id, 9), 0, 5)}),
                departure_step=2,
            )
        )

    assert not master.materialized_rows
    selected = master.solve_ip()
    assert shared in master.materialized_rows
    _assert_claim_feasible(selected, rows)
    assert sum(column.delay_s for column in selected.values()) == pytest.approx(8.0)


def test_head_on_path_rows_require_an_eight_step_hold():
    """Path-row arithmetic for five opposing visits gives min |D|=8 (32 s).

    The production customer-cylinder fixture cannot isolate these rows: on a
    collinear overlap each flight necessarily traverses the other's inner
    destination, whose dwell remains active much longer than W.  This master
    fixture pins the paper/path calculation without deleting canonical endpoint
    claims from full solver tests.
    """

    params = _params(M=1000.0)
    offsets = (-2, 1)
    cells = tuple((index, 0) for index in range(5))

    def path_rows(reverse: bool, shift: int) -> frozenset[RowKey]:
        ordered = tuple(reversed(cells)) if reverse else cells
        return frozenset(
            RowKey.cell(cell, 0, row_step)
            for visit_step, cell in enumerate(ordered, start=shift)
            for row_step in visit_rows(visit_step, offsets)
        )

    rows = RowIndex()
    master = RestrictedMaster((1, 2), rows, params, seed=0)
    for shift in range(9):
        master.add_column(
            _synthetic_column(1, shift * 4.0, path_rows(False, shift), departure_step=shift)
        )
        master.add_column(
            _synthetic_column(2, shift * 4.0, path_rows(True, shift), departure_step=shift)
        )
    selected = master.solve_ip()

    _assert_claim_feasible(selected, rows)
    assert sorted(column.delay_s for column in selected.values()) == [0.0, 32.0]


def test_rounding_honours_fixed_and_unmaterialized_claims():
    params = _params(M=1000.0)
    saturated = RowKey.cell((0, 0), 0, 0)
    rows = RowIndex()
    master = RestrictedMaster(
        (1, 2),
        rows,
        params,
        seed=11,
        fixed_loads={saturated: 1},
    )
    for flight_id in (1, 2):
        master.add_column(_synthetic_column(flight_id, 0.0, frozenset({saturated})))
        master.add_column(
            _synthetic_column(
                flight_id,
                4.0,
                frozenset({RowKey.cell((flight_id, 3), 0, 0)}),
                departure_step=1,
            )
        )
    rounded = master.round_heuristic(np.full(4, 0.5), np.random.default_rng(11), 32)

    assert rounded
    assert all(saturated not in column.claims for column in rounded.values())
    _assert_claim_feasible(rounded, rows)


def test_hand_checked_60deg_crossing():
    """Two unique geodesics share one cell; W=4 makes one hold 4 steps (16 s)."""

    cfg = _cfg(max_ground_delay_s=32.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]
    _assert_endpoint_cells_pairwise_disjoint(requests, cfg)
    for request in requests:
        start = hg.enu_to_axial(*request.origin[:2], hg.circumradius(cfg))
        destination = hg.enu_to_axial(*request.dest[:2], hg.circumradius(cfg))
        assert hg.hex_distance(start, destination) == 8
        assert len(_geodesics(start, destination)) == 1

    result = ColGenSolver().solve(requests, cfg, (), _params())

    assert len(result.columns) == 2
    assert result.stats["objective"] == pytest.approx(16.0, abs=1e-8)
    assert result.stats["initial_heuristic_flights"] == 2
    assert result.stats["initial_heuristic_delay_s"] == pytest.approx(16.0, abs=1e-8)
    assert sorted(column.delay_s for column in result.columns.values()) == pytest.approx(
        [0.0, 16.0], abs=1e-8
    )
    _assert_claim_feasible(result.columns)
    _assert_files_cleanly(requests, result.columns, cfg)


def test_two_crossing_flights_bruteforce_optimal():
    """Production objective equals exhaustive delay-pair enumeration on unique routes."""

    cfg = _cfg(max_ground_delay_s=32.0)
    params = _params()
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]
    candidate_sets: list[list[Column]] = []
    for request in requests:
        graph = build_flight_graph(request, cfg, (), params)
        path = seed_column(graph, cfg).cell_path
        candidates = []
        for departure_step in range(graph.base_step, graph.latest_departure_step + 1):
            raw = Column(
                request.flight_id,
                departure_step,
                0,
                None,
                None,
                path,
                (departure_step - graph.base_step) * cfg.dt_s,
            )
            candidates.append(replace(raw, claims=column_claims(raw, graph, cfg)))
        candidate_sets.append(candidates)

    brute = min(
        first.delay_s + second.delay_s
        for first in candidate_sets[0]
        for second in candidate_sets[1]
        if first.claims.isdisjoint(second.claims)
    )
    result = ColGenSolver().solve(requests, cfg, (), params)

    assert brute == pytest.approx(16.0)
    assert result.stats["objective"] == pytest.approx(brute, abs=1e-8)


def test_hand_checked_merge_three_flights():
    """Three unique geodesics share one cell; holds 0, 4, 8 steps cost 48 s."""

    cfg = _cfg(max_ground_delay_s=48.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
        _request(3, (-4, 4), (4, -4), cfg),
    ]
    _assert_endpoint_cells_pairwise_disjoint(requests, cfg)
    for request in requests:
        start = hg.enu_to_axial(*request.origin[:2], hg.circumradius(cfg))
        destination = hg.enu_to_axial(*request.dest[:2], hg.circumradius(cfg))
        assert len(_geodesics(start, destination)) == 1

    result = ColGenSolver().solve(requests, cfg, (), _params())

    assert len(result.columns) == 3
    assert result.stats["objective"] == pytest.approx(48.0, abs=1e-8)
    assert sorted(column.delay_s for column in result.columns.values()) == pytest.approx(
        [0.0, 16.0, 32.0], abs=1e-8
    )
    _assert_claim_feasible(result.columns)
    _assert_files_cleanly(requests, result.columns, cfg)


def test_hand_checked_detour_beats_hold():
    """One extra hop (4 s) resolves a crossing that otherwise needs a W-step hold."""

    cfg = _cfg(flight_levels_m=(30.0,), max_ground_delay_s=20.0)
    requests = [
        _request(1, (-2, -5), (-2, -11), cfg),
        _request(2, (-8, -4), (0, -4), cfg),
    ]
    _assert_endpoint_cells_pairwise_disjoint(requests, cfg)

    # Pinned to the cost-scale gap because the bound assertions below are cost-scale
    # claims: that the LP itself *proves* 4.0, not merely that the IP happens to find it.
    # Under the default revenue metric this instance stops after one iteration -- with
    # M=1e6 and two flights the denominator is ~2e6, so the 12-unit slack reads as 6e-6
    # and clears the 1e-4 threshold before the LP is re-solved with the detour column.
    # The returned solution is optimal either way; only `cost_upper_bound` stays loose.
    no_detour = ColGenSolver().solve(
        requests,
        cfg,
        (),
        _params(detour_slack_hops=0, gap_metric="cost"),
    )
    with_detour = ColGenSolver().solve(
        requests,
        cfg,
        (),
        _params(detour_slack_hops=1, gap_metric="cost"),
    )

    assert no_detour.stats["objective"] == pytest.approx(16.0, abs=1e-8)
    assert with_detour.stats["objective"] == pytest.approx(4.0, abs=1e-8)
    assert with_detour.stats["cost_lower_bound"] == pytest.approx(4.0, abs=1e-7)
    assert with_detour.stats["cost_upper_bound"] == pytest.approx(4.0, abs=1e-7)
    assert sorted(column.delay_s for column in with_detour.columns.values()) == [0.0, 4.0]
    assert any(
        len(column.cell_path) - 1 == hg.hex_distance(column.cell_path[0], column.cell_path[-1]) + 1
        for column in with_detour.columns.values()
    )
    _assert_files_cleanly(requests, with_detour.columns, cfg)


def test_revenue_gap_stops_early_but_still_returns_the_optimum():
    """The paper's revenue-scale gap is loose here because ``M`` is an artificial big-M.

    Equation (10) normalizes by the maximize objective, which the paper can do because
    its ``M`` is real per-flight revenue.  Ours is a constant chosen only to make
    cancellation unattractive, so the denominator is ~``n * M`` whatever the delays are
    and the ratio mostly measures how large ``M`` was set.  On this two-flight instance
    that means "converged" fires an iteration before the cost-scale gap has moved at all.

    Pinned deliberately: the solution is still optimal, because pricing's columns are
    banked in the master before the loop breaks and the final IP sees them.  What the
    early stop costs is bound *tightness* in the reported stats, not solution quality --
    and that distinction is the thing worth not rediscovering.
    """

    cfg = _cfg(flight_levels_m=(30.0,), max_ground_delay_s=20.0)
    requests = [
        _request(1, (-2, -5), (-2, -11), cfg),
        _request(2, (-8, -4), (0, -4), cfg),
    ]

    revenue = ColGenSolver().solve(
        requests, cfg, (), _params(detour_slack_hops=1, gap_metric="revenue")
    )

    assert revenue.stats["termination_reason"] == "lp_gap"
    assert revenue.stats["iterations"] == 1
    # Optimal solution ...
    assert revenue.stats["objective"] == pytest.approx(4.0, abs=1e-8)
    assert revenue.stats["cost_lower_bound"] == pytest.approx(4.0, abs=1e-7)
    # ... reached while the two scales disagree by five orders of magnitude.
    assert revenue.stats["lp_gap_revenue"] < 1e-4
    assert revenue.stats["lp_gap_cost"] > 0.5


def test_colgen_beats_fcfs_on_constructed_congestion():
    """One global hold clears two crossings that FCFS handles independently."""

    cfg = _cfg(max_ground_delay_s=64.0)
    requests = [
        _request(1, (-6, 0), (6, 0), cfg),
        _request(2, (-2, -4), (-2, 4), cfg),
        _request(3, (2, -4), (2, 4), cfg, departure=16.0),
    ]
    _assert_endpoint_cells_pairwise_disjoint(requests, cfg)
    for request in requests:
        start = hg.enu_to_axial(*request.origin[:2], hg.circumradius(cfg))
        destination = hg.enu_to_axial(*request.dest[:2], hg.circumradius(cfg))
        assert len(_geodesics(start, destination)) == 1

    colgen = ColGenSolver().solve(requests, cfg, (), _params())

    astar_cfg = replace(cfg, planner="astar")
    ledger = ReservationLedger(astar_cfg)
    astar = AStarPlanner()
    fcfs_delays = []
    for request in requests:
        intent = astar.plan(request, ledger, astar_cfg)
        assert intent.status is IntentStatus.ACCEPTED
        assert not ledger.any_conflict(intent.volumes)
        ledger.commit(request.flight_id, intent.volumes)
        fcfs_delays.append(total_delay_s(intent, astar_cfg))

    assert sorted(column.delay_s for column in colgen.columns.values()) == [0.0, 0.0, 16.0]
    assert fcfs_delays == pytest.approx([0.0, 24.0, 24.0])
    assert colgen.stats["total_delay_s"] == pytest.approx(16.0)
    assert colgen.stats["total_delay_s"] < sum(fcfs_delays)
    _assert_files_cleanly(requests, colgen.columns, cfg)


@pytest.mark.parametrize("departure", [100.0, 101.3])
def test_single_flight_empty_world_matches_astar_cost(departure: float):
    cfg = _cfg(max_ground_delay_s=16.0)
    request = _request(20, (-4, 0), (4, 0), cfg, departure=departure)

    result = ColGenSolver().solve([request], cfg, (), _params())
    column = result.columns[request.flight_id]
    intent = column_to_intent(column, request, cfg)
    astar = AStarPlanner().plan(request, ReservationLedger(cfg), replace(cfg, planner="astar"))

    assert intent.ground_delay_s == pytest.approx(astar.ground_delay_s, abs=1e-9)
    assert intent.cost == pytest.approx(astar.cost, abs=1e-9)
    assert len(intent.centerline) > 1


def test_fixed_claim_displaces_a_flight_by_the_measured_window():
    cfg = _cfg(max_ground_delay_s=32.0)
    request = _request(7, (-4, 0), (4, 0), cfg)
    params = _params()
    graph = build_flight_graph(request, cfg, (), params)
    centre_step = graph.base_step + graph.takeoff_steps[0] + 4
    fixed = frozenset({RowKey.cell((0, 0), 0, centre_step + 1)})

    result = ColGenSolver().solve([request], cfg, (), params, fixed_claims=(fixed,))

    assert len(result.columns) == 1
    selected = result.columns[request.flight_id]
    assert selected.delay_s == pytest.approx(16.0, abs=1e-8)
    assert selected.claims.isdisjoint(fixed)


def test_budget_limited_crossing_denies_exactly_one_in_repair():
    cfg = _cfg(max_ground_delay_s=12.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]

    result = ColGenSolver().solve(requests, cfg, (), _params())

    assert len(result.columns) == 1
    assert len(result.stats["denied_flight_ids"]) == 1
    _assert_claim_feasible(result.columns)


def test_repair_covers_all_feasible_flights(monkeypatch):
    """Force an empty RMP incumbent; sequential pricing must rebuild a full schedule."""

    cfg = _cfg(max_ground_delay_s=32.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]
    monkeypatch.setattr(
        "freespace_sim.planner.colgen.solver._initial_feasible_selection",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "freespace_sim.planner.colgen.solver._greedy_feasible_selection",
        lambda *args, **kwargs: ({}, True),
    )
    monkeypatch.setattr(RestrictedMaster, "round_heuristic", lambda *args, **kwargs: {})
    monkeypatch.setattr(RestrictedMaster, "solve_ip", lambda *args, **kwargs: {})

    result = ColGenSolver().solve(requests, cfg, (), _params())

    assert len(result.columns) == len(requests)
    assert result.stats["repair_added"] == len(requests)
    assert result.stats["denied_flight_ids"] == ()
    _assert_claim_feasible(result.columns)
    _assert_files_cleanly(requests, result.columns, cfg)


def test_first_master_has_only_bounded_shortest_path_initialization(monkeypatch):
    """Route alternatives are generated after, never before, the first LP."""

    cfg = _cfg(max_ground_delay_s=32.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]
    events: list[str] = []
    first_lp_counts: Counter[int] = Counter()
    first_lp_routes: dict[int, set] = {}
    original_solve_lp = RestrictedMaster.solve_lp

    def capture_first_lp(master):
        events.append("lp")
        if not first_lp_counts:
            first_lp_counts.update(column.flight_id for column in master.columns)
            for column in master.columns:
                first_lp_routes.setdefault(column.flight_id, set()).add(
                    tuple(column.cell_path)
                )
        return original_solve_lp(master)

    def capture_greedy(*_args, **kwargs):
        events.append("greedy")
        return dict(kwargs["initial"]), True

    monkeypatch.setattr(RestrictedMaster, "solve_lp", capture_first_lp)
    monkeypatch.setattr(solver_module, "_greedy_feasible_selection", capture_greedy)

    result = ColGenSolver().solve(requests, cfg, (), _params())

    assert result.columns
    assert first_lp_counts
    assert set(first_lp_counts) == {1, 2}
    # The contract is "no ROUTE alternatives before the first LP", and a departure
    # ladder is not one: every pre-LP column for a flight is the same cell path at a
    # different clock.  Asserting that directly is stronger than the old <=2 column
    # count, which was a proxy that the ladder (seed_ladder_steps) invalidated without
    # touching the invariant.
    assert max(first_lp_counts.values()) <= 1 + _params().seed_ladder_steps
    assert all(len(routes) == 1 for routes in first_lp_routes.values()), (
        f"pre-LP pool holds route alternatives: {first_lp_routes}"
    )
    assert events.index("lp") < events.index("greedy")


def test_greedy_timeout_is_local_to_one_flight(monkeypatch):
    cfg = _cfg(max_ground_delay_s=32.0)
    params = _params()
    requests = [
        _request(1, (-4, -8), (4, -8), cfg),
        _request(2, (-4, 8), (4, 8), cfg),
    ]
    graphs = {
        request.flight_id: build_flight_graph(request, cfg, (), params)
        for request in requests
    }
    initial = {
        flight_id: solver_module._shift_column(
            seed_column(graph, cfg),
            graph.base_step + 1,
            cfg,
        )
        for flight_id, graph in graphs.items()
    }
    calls: list[int] = []

    def fake_search(graph, *_args, **_kwargs):
        calls.append(graph.request.flight_id)
        if graph.request.flight_id == 1:
            raise PricingTimeout
        return None

    monkeypatch.setattr(solver_module, "find_feasible_column", fake_search)

    selected, completed = _greedy_feasible_selection(
        graphs,
        {},
        RowIndex(),
        cfg,
        params,
        deadline=time.monotonic() + 10.0,
        initial=initial,
    )

    assert calls == [1, 2]
    assert selected == initial
    assert completed is False


def test_initial_seed_shifts_stop_at_the_path_specific_arrival_horizon():
    """A detoured seed can be longer than the graph's nominal hop allowance."""

    cfg = _cfg(max_ground_delay_s=48.0)
    request = _request(1, (-10, 0), (10, 0), cfg)
    static_terms = tuple(
        (_point(cell, cfg), Terminal(f"X{index}", 1, radius=0.0))
        for index, cell in enumerate(((3, 1), (1, -1)))
    )
    params = _params(detour_slack_hops=1)
    graph = build_flight_graph(request, cfg, static_terms, params)
    seed = seed_column(graph, cfg)
    assert len(seed.cell_path) - 1 > graph.shortest_hops + graph.detour_slack_hops

    invalid_latest_claims = _shift_claims(
        seed.claims,
        graph.latest_departure_step - seed.departure_step,
    )
    fixed_claims: list[frozenset[RowKey]] = []
    used: set[RowKey] = set()
    for departure_step in range(seed.departure_step, graph.latest_departure_step):
        shifted = _shift_claims(seed.claims, departure_step - seed.departure_step)
        row = sorted(shifted - invalid_latest_claims - used)[0]
        fixed_claims.append(frozenset({row}))
        used.add(row)

    selected = _initial_feasible_selection(
        {request.flight_id: seed},
        {request.flight_id: graph},
        _fixed_loads(fixed_claims),
        RowIndex(),
        cfg,
    )

    assert selected == {}
    forbidden_rows = frozenset().union(*fixed_claims)
    _reduced_cost, priced = price_flight(
        graph,
        {},
        0.0,
        cfg,
        params,
        forbidden_rows=forbidden_rows,
        require_improving=False,
    )
    if priced is not None:
        assert column_claims(priced, graph, cfg) == priced.claims


def test_repair_finds_feasible_column_even_when_delay_exceeds_m():
    """Repair is a feasibility pass, so reduced-cost sign cannot suppress coverage."""

    cfg = _cfg(max_ground_delay_s=32.0)
    request = _request(7, (-4, 0), (4, 0), cfg)
    fixed = frozenset({RowKey.cell((0, 0), 0, 10)})

    result = ColGenSolver().solve([request], cfg, (), _params(M=1.0), fixed_claims=(fixed,))

    assert result.stats["denied_flight_ids"] == ()
    assert result.stats["ip_objective"] == pytest.approx(0.0)
    assert result.stats["ip_upper_bound"] == pytest.approx(0.0)
    assert result.stats["ip_gap"] == pytest.approx(0.0)
    assert result.stats["ip_gap_met"] is True
    assert result.columns[7].delay_s == pytest.approx(16.0)
    assert result.columns[7].claims.isdisjoint(fixed)


def test_disconnected_static_wall_denies_only_that_flight():
    cfg = _cfg(max_ground_delay_s=32.0)
    request = _request(8, (-4, 0), (4, 0), cfg)
    wall_center = _point((0, 0), cfg)
    static_terms = ((wall_center, Terminal("X", 1, radius=90.0)),)

    result = ColGenSolver().solve([request], cfg, static_terms, _params())

    assert result.columns == {}
    assert result.stats["denied_flight_ids"] == (8,)
    assert result.stats["seedless_flight_ids"] == (8,)


def test_swap_never_selected_and_head_on_files_cleanly():
    """Canonical endpoint claims make this full-ledger head-on optimum 64 s, not 32 s."""

    cfg = _cfg(flight_levels_m=(30.0,), max_ground_delay_s=80.0)
    params = _params()
    requests = [
        _request(1, (-10, 0), (2, 0), cfg),
        _request(2, (10, 0), (-2, 0), cfg),
    ]
    result = ColGenSolver().solve(requests, cfg, (), params)
    assert result.stats["objective"] == pytest.approx(64.0, abs=1e-8)

    graph_by_id = {
        request.flight_id: build_flight_graph(request, cfg, (), params) for request in requests
    }
    edge_times: dict[int, list[tuple[tuple[int, int], tuple[int, int], int]]] = {}
    for flight_id, column in result.columns.items():
        graph = graph_by_id[flight_id]
        start = column.departure_step + graph.takeoff_steps[column.level]
        edge_times[flight_id] = [
            (first, second, start + index)
            for index, (first, second) in enumerate(zip(column.cell_path, column.cell_path[1:]))
        ]

    window = derive_cell_window(cfg)
    width = window[1] - window[0] + 1
    for first, second, first_step in edge_times[1]:
        for other_first, other_second, second_step in edge_times[2]:
            if first == other_second and second == other_first:
                assert abs(first_step - second_step) >= width
    _assert_files_cleanly(requests, result.columns, cfg)


def test_seed_columns_warm_start_the_pool_without_changing_the_answer():
    """Seeding is a knob on the PATH to the optimum, not on the optimum.

    solver.py keeps initialization deliberately small on the bet that route alternatives
    are cheaper to discover by reduced-cost pricing than to enumerate up front.  This
    parameter exists so that bet can be measured instead of assumed -- forensics on a
    100-flight solve found ~95% of early additions were time shifts of a route already
    in the pool, which `_shift_column` produces arithmetically while a pricing sweep
    costs 15-17s.

    Whatever seeding does to iteration count, it must not move the objective: every
    seeded column goes through the same canonical claim gate, so it cannot introduce a
    trajectory pricing could not have produced.
    """

    cfg = _cfg(max_ground_delay_s=32.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]
    params = _params()

    plain = ColGenSolver().solve(requests, cfg, (), params)

    # Time translations of each flight's own shortest seed -- the cheapest possible
    # alternatives, and exactly what pricing spends its early iterations rediscovering.
    graphs = {r.flight_id: build_flight_graph(r, cfg, (), params) for r in requests}
    seeds = {}
    for flight_id, graph in graphs.items():
        seed = seed_column(graph, cfg)
        seeds[flight_id] = [
            solver_module._shift_column(seed, seed.departure_step + step, cfg)
            for step in (1, 2)
            if seed.departure_step + step <= graph.latest_departure_step
        ]

    seeded = ColGenSolver().solve(requests, cfg, (), params, seed_columns=seeds)

    n_seeded = sum(len(v) for v in seeds.values())
    assert n_seeded > 0, "the fixture must actually seed something"
    assert seeded.stats["seeded_columns"] == n_seeded
    assert seeded.stats["n_columns"] >= plain.stats["n_columns"]
    # The answer is the contract; the route there is not.
    assert seeded.stats["objective"] == pytest.approx(plain.stats["objective"], abs=1e-7)
    _assert_claim_feasible(seeded.columns)
    _assert_files_cleanly(requests, seeded.columns, cfg)

    with pytest.raises(KeyError, match="not in this batch"):
        ColGenSolver().solve(requests, cfg, (), params, seed_columns={999: []})


def test_ip_solve_is_timed_separately_from_the_rest_of_the_solve():
    """The final IP was the last unattributed block in the solve.

    It matters because the intuition is wrong by orders of magnitude: on a 1,138-column
    100-flight pool the whole solve took 643s and the IP was under a second of it, the
    rest being pricing.  Without a number, "the IP is slow" is an unfalsifiable
    explanation for any slow run.
    """

    cfg = _cfg(max_ground_delay_s=32.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]

    # ladder off so the heuristic cannot prove the gap and skip the MILP outright.
    result = ColGenSolver().solve(requests, cfg, (), _params(seed_ladder_steps=0))

    assert result.stats["ip_status"] != "skipped", "this fixture must reach the IP"
    elapsed = result.stats["ip_elapsed_s"]
    assert isinstance(elapsed, float)
    assert 0.0 <= elapsed <= result.stats["elapsed_s"] + 1e-6, (
        "the IP is one block inside the solve, so its time cannot exceed the whole"
    )


def test_iteration_payload_carries_both_gap_scales_and_the_master():
    """Per-iteration telemetry has to answer questions the final stats cannot.

    Both scales, because with an artificial big-M the revenue gap reads as converged
    while the cost gap is still enormous.  The master itself, because the rounding
    heuristic and the restricted IP over the same columns are different numbers --
    measured 13,266.8 vs 13,099.3 on one pool -- so only holding the master lets an
    analysis run ask what the IP would have said at that iteration.
    """

    cfg = _cfg(max_ground_delay_s=32.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]
    seen: list[dict] = []

    ColGenSolver().solve(
        requests, cfg, (), _params(), on_iteration=seen.append
    )

    assert seen, "at least one iteration must report"
    payload = seen[0]
    for key in (
        "lp_gap_cost", "lp_gap_revenue",
        "heuristic_gap_cost", "heuristic_gap_revenue",
        "cost_upper_bound", "cost_lower_bound", "rc_sum", "dual_l2",
    ):
        assert key in payload, key
    assert payload["master"] is not None
    assert hasattr(payload["master"], "solve_ip"), "analysis needs the live master"
    # The two scales share a numerator and differ only by denominator, so neither is
    # derivable from the other without also knowing M and the flight count.
    assert payload["lp_gap_cost"] >= payload["lp_gap_revenue"]


def test_bounds_are_monotone_and_solver_is_deterministic():
    cfg = _cfg(max_ground_delay_s=48.0, seed=29)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
        _request(3, (-4, 4), (4, -4), cfg),
    ]
    params = _params()

    first = ColGenSolver().solve(requests, cfg, (), params)
    second = ColGenSolver().solve(requests, cfg, (), params)

    bounds = first.stats["upper_bounds"]
    assert bounds
    assert all(next_ <= current + 1e-7 for current, next_ in zip(bounds, bounds[1:]))
    assert first.stats["lp_gap"] <= params.lp_gap + 1e-8
    assert first.columns == second.columns
    # Runtime is intentionally excluded from the reproducibility contract.
    runtime_stats = {
        "elapsed_s",
        "graph_build_elapsed_s",
        "initial_greedy_elapsed_s",
        "ip_elapsed_s",
        "seed_elapsed_s",
        "time_to_master_s",
    }
    for key in set(first.stats) - runtime_stats:
        assert first.stats[key] == second.stats[key]


def test_single_level_guard():
    cfg = replace(_cfg(), flight_levels_m=(30.0, 70.0, 110.0))
    request = _request(1, (-2, 0), (2, 0), cfg)
    with pytest.raises(NotImplementedError, match="single flight level"):
        ColGenSolver().solve([request], cfg, (), _params())


def test_backend_parity():
    pytest.importorskip("gurobipy")
    cfg = _cfg(max_ground_delay_s=32.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]

    highs = ColGenSolver().solve(requests, cfg, (), _params(solver="highs"))
    gurobi = ColGenSolver().solve(requests, cfg, (), _params(solver="gurobi"))

    assert highs.stats["objective"] == pytest.approx(gurobi.stats["objective"], abs=1e-6)
    assert highs.stats["backend"] == "highs"
    assert gurobi.stats["backend"] == "gurobi"
    assert sorted(column.delay_s for column in highs.columns.values()) == pytest.approx(
        sorted(column.delay_s for column in gurobi.columns.values())
    )
    _assert_claim_feasible(highs.columns)
    _assert_claim_feasible(gurobi.columns)
    _assert_files_cleanly(requests, highs.columns, cfg)
    _assert_files_cleanly(requests, gurobi.columns, cfg)


def test_backend_parity_when_final_integer_master_runs():
    """HiGHS and Gurobi return the same answer when the final IP cannot be skipped."""

    pytest.importorskip("gurobipy")
    cfg = _cfg(max_ground_delay_s=48.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
        _request(3, (-4, 4), (4, -4), cfg),
    ]
    controls = {
        "max_iterations": 1,
        "lp_gap": 0.0,
        "ip_gap": 0.0,
    }

    # See note on the ladder below: this test must reach the final integer master.
    controls = {**controls, "seed_ladder_steps": 0}
    highs = ColGenSolver().solve(requests, cfg, (), _params(solver="highs", **controls))
    gurobi = ColGenSolver().solve(requests, cfg, (), _params(solver="gurobi", **controls))

    assert highs.stats["ip_skipped"] is False
    assert gurobi.stats["ip_skipped"] is False
    assert highs.stats["objective"] == pytest.approx(gurobi.stats["objective"], abs=1e-6)
    assert sorted(column.delay_s for column in highs.columns.values()) == pytest.approx(
        sorted(column.delay_s for column in gurobi.columns.values())
    )
    assert highs.stats["denied_flight_ids"] == gurobi.stats["denied_flight_ids"] == ()
    _assert_claim_feasible(highs.columns)
    _assert_claim_feasible(gurobi.columns)
    _assert_files_cleanly(requests, highs.columns, cfg)
    _assert_files_cleanly(requests, gurobi.columns, cfg)


def test_final_ip_time_limit_without_native_incumbent_keeps_heuristic(monkeypatch):
    """A native timeout cannot discard the RMP's feasible incumbent."""

    master = RestrictedMaster((1, 2), RowIndex(), _params())
    heuristic = {
        flight_id: _synthetic_column(
            flight_id,
            float(flight_id),
            frozenset({RowKey.cell((flight_id, 0), 0, 0)}),
        )
        for flight_id in (1, 2)
    }
    for column in heuristic.values():
        master.add_column(column)
    master.set_heuristic(heuristic)
    monkeypatch.setattr(
        master.backend,
        "solve_ip",
        lambda warm_start: BackendIpResult(
            0.0,
            np.zeros(len(master.columns), dtype=float),
            math.inf,
            "time_limit_no_incumbent",
            False,
        ),
    )

    selected = master.solve_ip()

    assert selected == heuristic
    assert master.last_ip_status == "time_limit_no_incumbent"
    assert master.last_ip_objective == pytest.approx(master.objective_of(heuristic))
    assert master.last_ip_bound == math.inf


def test_incomplete_pricing_sweep_never_publishes_a_global_bound(monkeypatch):
    """A mid-sweep deadline may add columns, but cannot certify missing RCs."""

    cfg = _cfg(max_ground_delay_s=32.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]
    calls: list[int] = []

    def timeout_on_second(graph, duals, pi_f, solve_cfg, params, **kwargs):
        del duals, pi_f, kwargs
        calls.append(graph.request.flight_id)
        if len(calls) == 2:
            raise PricingTimeout("test deadline")
        column = seed_column(graph, solve_cfg)
        return params.M - column.delay_s, column

    monkeypatch.setattr(solver_module, "price_flight", timeout_on_second)
    monkeypatch.setattr(
        solver_module,
        "_greedy_feasible_selection",
        lambda *args, **kwargs: ({}, True),
    )

    result = ColGenSolver().solve(
        requests,
        cfg,
        (),
        _params(max_iterations=3, lp_gap=0.0, ip_gap=0.0),
    )

    assert result.stats["termination_reason"] == "time_limit"
    assert result.stats["pricing_flights_completed"] == 1
    assert result.stats["pricing_sweeps_completed"] == 0
    assert result.stats["pricing_timeout_flight_id"] == calls[-1]
    assert result.stats["upper_bounds"] == ()
    assert math.isinf(result.stats["upper_bound"])
    assert result.stats["denied_flight_ids"] == ()
    _assert_claim_feasible(result.columns)


def test_ip_deadline_stops_lazy_separation_and_keeps_heuristic(monkeypatch):
    shared = RowKey.cell((0, 0), 0, 0)
    master = RestrictedMaster((1, 2), RowIndex(), _params())
    shared_columns = {
        flight_id: _synthetic_column(flight_id, 0.0, frozenset({shared}))
        for flight_id in (1, 2)
    }
    heuristic = {
        flight_id: _synthetic_column(
            flight_id,
            4.0,
            frozenset({RowKey.cell((flight_id, 1), 0, 0)}),
            departure_step=1,
        )
        for flight_id in (1, 2)
    }
    for flight_id in (1, 2):
        master.add_column(shared_columns[flight_id])
        master.add_column(heuristic[flight_id])
    master.set_heuristic(heuristic)
    calls = []

    def overloaded_ip(_warm_start):
        calls.append(master.backend.time_limit_s)
        return BackendIpResult(
            2 * master.params.M,
            np.array([1.0, 0.0, 1.0, 0.0]),
            2 * master.params.M,
            "optimal",
            True,
        )

    monkeypatch.setattr(master.backend, "solve_ip", overloaded_ip)
    clock = iter((0.0, 2.0))
    monkeypatch.setattr(master_module.time, "monotonic", lambda: next(clock))

    selected = master.solve_ip(deadline=1.0)

    assert selected == heuristic
    assert calls == [pytest.approx(1.0)]
    assert master.backend.time_limit_s == master.params.time_limit_s
    assert shared in master.materialized_rows
    assert master.last_ip_status == "time_limit_separation"
    assert master.last_ip_bound == math.inf


def test_preprocessing_deadline_returns_search_exhausted_without_building_graphs(monkeypatch):
    cfg = _cfg()
    request = _request(1, (-2, 0), (2, 0), cfg)
    clock = iter((0.0, 2.0, 2.0, 2.0, 2.0))
    monkeypatch.setattr(solver_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        solver_module,
        "build_flight_graph",
        lambda *_args, **_kwargs: pytest.fail("expired solve must not build another graph"),
    )

    result = ColGenSolver().solve(
        [request],
        cfg,
        (),
        _params(time_limit_s=1.0),
    )

    assert result.columns == {}
    assert result.stats["termination_reason"] == "time_limit"
    assert result.stats["search_exhausted_flight_ids"] == (request.flight_id,)
    assert result.stats["budget_denied_flight_ids"] == ()


def test_seeding_timeout_reports_partial_master_progress(monkeypatch):
    cfg = _cfg()
    requests = [
        _request(1, (-4, -4), (4, -4), cfg),
        _request(2, (-4, 4), (4, 4), cfg),
    ]
    original_seed = solver_module.seed_column
    calls = 0

    def timeout_on_second(graph, solve_cfg, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PricingTimeout("test seeding timeout")
        return original_seed(graph, solve_cfg, **kwargs)

    monkeypatch.setattr(solver_module, "seed_column", timeout_on_second)

    # ladder off: this test counts the partial seed prefix a timeout leaves behind,
    # and the ladder would add seed_ladder_steps columns per seeded flight on top.
    result = ColGenSolver().solve(requests, cfg, (), _params(seed_ladder_steps=0))

    assert result.columns == {}
    assert result.stats["backend"] == "highs"
    assert result.stats["preprocessing_stage"] == "seeding"
    assert result.stats["graphs_built"] == 2
    assert result.stats["seed_flights_processed"] == 1
    assert result.stats["seeds_completed"] == 1
    assert result.stats["initial_seed_columns"] == 1
    assert result.stats["n_columns"] == 1
    assert result.stats["time_to_master_s"] > 0.0


def test_nonoptimal_final_ip_cannot_certify_a_budget_denial(monkeypatch):
    cfg = _cfg(max_ground_delay_s=32.0)
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]
    monkeypatch.setattr(
        solver_module,
        "_initial_feasible_selection",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        solver_module,
        "_greedy_feasible_selection",
        lambda *args, **kwargs: ({}, True),
    )
    monkeypatch.setattr(
        solver_module,
        "price_flight",
        lambda *args, **kwargs: (1.0, None),
    )

    def partial_ip(master, *args, **kwargs):
        del args, kwargs
        column = next(column for column in master.columns if column.flight_id == 1)
        master.last_ip_objective = master.objective_of((column,))
        master.last_ip_bound = math.inf
        master.last_ip_status = "status_1"
        master.last_ip_optimal = False
        return {1: column}

    monkeypatch.setattr(RestrictedMaster, "solve_ip", partial_ip)

    result = ColGenSolver().solve(
        requests,
        cfg,
        (),
        # ladder off: this test needs the final MILP to actually run.
        _params(lp_gap=0.0, ip_gap=0.0, seed_ladder_steps=0),
    )

    assert result.stats["termination_reason"] == "time_limit"
    assert result.stats["ip_status"] == "status_1"
    assert len(result.stats["denied_flight_ids"]) == 1
    assert result.stats["budget_denied_flight_ids"] == ()
    assert result.stats["search_exhausted_flight_ids"] == result.stats["denied_flight_ids"]
