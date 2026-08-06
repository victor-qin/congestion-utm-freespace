"""Structural performance regressions for column-generation pricing.

These tests avoid wall-clock thresholds.  They pin the branch that prevents a
geographically unrelated master row from expanding a flight's full time DAG,
while retaining the exact signed-dual fallback needed for pricing correctness.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.metrics import total_delay_s
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph, column_claims
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.objective import CostModel
from freespace_sim.planner.colgen.pricing import DualView, price_flight, seed_column
from freespace_sim.planner.colgen.translate import Column, column_to_intent
from freespace_sim.planner.colgen.windows import derive_cell_window, endpoint_claim_cells
from freespace_sim.types import FlightRequest, IntentStatus, Terminal, vec
from freespace_sim.volumes import exit_radius


def _cfg() -> SimConfig:
    return SimConfig(
        planner="colgen",
        flight_levels_m=(100.0,),
        airspace_ceiling_m=125.0,
        region_size_m=(20_000.0, 20_000.0),
        terminal_airspace_always_active=True,
        max_ground_delay_s=48.0,
        max_detour_factor=10.0,
    )


def _point(cell: tuple[int, int], cfg: SimConfig):
    x, y = hg.hex_center(*cell, hg.circumradius(cfg))
    return vec(x, y, cfg.ground_level_m)


def _graph(cfg: SimConfig, *, slack: int = 4):
    request = FlightRequest(1, _point((0, 0), cfg), _point((2, 0), cfg), 0.0, 0.0)
    params = ColGenParams(
        solver="highs",
        detour_slack_hops=slack,
        max_iterations=3,
        time_limit_s=30.0,
        n_heuristic_tries=2,
    )
    return build_flight_graph(request, cfg, (), params), params


def test_seed_astar_expands_only_a_small_lazy_subset():
    """A long seed follows a promising geodesic without materializing its ellipse."""

    cfg = _cfg()
    request = FlightRequest(101, _point((0, 0), cfg), _point((80, 0), cfg), 0.0, 0.0)
    params = ColGenParams(solver="highs", detour_slack_hops=12)
    graph = build_flight_graph(request, cfg, (), params)

    assert not graph.corridor_cells.is_materialized
    seed = seed_column(graph, cfg)
    stats = dict(graph.arc_cache_stats)

    assert len(seed.cell_path) - 1 == 80
    assert stats["expanded_nodes"] <= 80
    assert not graph.corridor_cells.is_materialized
    assert stats["expanded_nodes"] < len(graph.corridor_cells) // 10


def test_repeated_pricing_reuses_lazy_arcs_and_cached_seed():
    """A second pricing call must not re-derive any arc geometry.

    ``arc_checks``/``expanded_nodes`` holding still is the invariant: no wall geometry is
    recomputed and no new source cell is expanded.  The search walks the lazy oracle again,
    so ``cache_hits`` is what grows -- that split is the evidence the reuse is real rather
    than the second call quietly skipping work.
    """

    cfg = _cfg()
    graph, params = _graph(cfg, slack=2)
    credited = RowKey.cell((-2, 0), 0, 5)

    first = price_flight(graph, {credited: -100.0}, 0.0, cfg, params)
    first_stats = dict(graph.arc_cache_stats)
    second = price_flight(graph, {credited: -100.0}, 0.0, cfg, params)
    second_stats = dict(graph.arc_cache_stats)

    assert second == first
    assert second_stats["arc_checks"] == first_stats["arc_checks"]
    assert second_stats["expanded_nodes"] == first_stats["expanded_nodes"]
    assert second_stats["cache_hits"] > first_stats["cache_hits"]


def test_canonical_claim_cache_is_strictly_bounded():
    cfg = _cfg()
    graph, _params = _graph(cfg, slack=2)
    seed = seed_column(graph, cfg)

    for delta in range(6):
        shifted = replace(
            seed,
            departure_step=seed.departure_step + delta,
            claims=frozenset(),
        )
        column_claims(shifted, graph, cfg)

    assert len(graph._search_cache.certified_claims) == 2


def test_uncertified_wall_seed_cannot_skip_exact_pricing(monkeypatch):
    """A valid fallback seed is an incumbent, not a global zero-dual proof."""

    cfg = _cfg()
    graph, params = _graph(cfg, slack=1)
    seed_column(graph, cfg)
    graph._search_cache.seed_delay_certified = False
    original = pricing._best_column
    calls = 0

    def capture_exact_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pricing, "_best_column", capture_exact_search)

    _reduced_cost, column = price_flight(graph, {}, 0.0, cfg, params)

    assert calls == 1
    assert column is not None


def test_role_specific_wall_arcs_survive_pricing_dominance():
    """A wall-invalid prefix cannot dominate the only valid final-role path."""

    cfg = SimConfig(
        planner="colgen",
        flight_levels_m=(100.0,),
        airspace_ceiling_m=125.0,
        region_size_m=(20_000.0, 20_000.0),
        terminal_airspace_always_active=True,
        max_ground_delay_s=0.0,
        max_detour_factor=10.0,
    )
    params = ColGenParams(
        solver="highs",
        detour_slack_hops=6,
        M=1_000_000.0,
    )
    origin = vec(34.27069337311164, -48.529334145873435, 0.0)
    destination = vec(6.242554805910697, -463.9202302821416, 0.0)
    origin_terminal = Terminal("A", 1, radius=120.0)
    destination_terminal = Terminal("B", 1, radius=120.0)
    request = FlightRequest(
        1,
        origin,
        destination,
        0.0,
        0.0,
        origin_terminal=origin_terminal,
        dest_terminal=destination_terminal,
    )
    graph = build_flight_graph(
        request,
        cfg,
        ((origin, origin_terminal), (destination, destination_terminal)),
        params,
    )
    duals = {
        RowKey.cell((-1, -1), 0, 7): -1000.0,
        RowKey.cell((0, -2), 0, 8): -1000.0,
        RowKey.cell((2, -3), 0, 10): -1000.0,
        RowKey.cell((3, -3), 0, 11): -1000.0,
        RowKey.cell((4, -4), 0, 12): -1000.0,
        RowKey.cell((1, -3), 0, 9): 1.0,
    }
    expected = Column(
        request.flight_id,
        graph.base_step,
        0,
        0,
        5,
        (
            (-1, -1),
            (0, -2),
            (1, -3),
            (2, -3),
            (3, -3),
            (4, -4),
        ),
        0.0,
    )
    expected = replace(
        expected,
        delay_s=total_delay_s(column_to_intent(expected, request, cfg), cfg),
        claims=column_claims(expected, graph, cfg),
    )

    invalid = replace(
        expected,
        cell_path=(
            (-1, -1),
            (0, -2),
            (1, -2),
            (2, -3),
            (3, -3),
            (4, -4),
        ),
        delay_s=0.0,
        claims=frozenset(),
    )
    with pytest.raises(ValueError, match="permanent static terminal"):
        column_claims(invalid, graph, cfg)

    reduced_cost, column = price_flight(
        graph,
        duals,
        0.0,
        cfg,
        params,
        require_improving=False,
    )
    expected_reduced_cost = params.M - expected.delay_s - sum(
        duals.get(row, 0.0) for row in expected.claims
    )

    assert expected.delay_s == pytest.approx(21.45534857624976)
    assert expected_reduced_cost == pytest.approx(1_004_977.5446514237)
    assert reduced_cost == pytest.approx(expected_reduced_cost)
    assert column == expected


def test_unrelated_rows_keep_the_canonical_seed_fast_path(monkeypatch):
    """Positive prices and exclusions elsewhere cannot make another route better."""

    cfg = _cfg()
    graph, params = _graph(cfg)
    seed = seed_column(graph, cfg)
    unrelated = RowKey.cell((999, 999), 0, 10)

    def fail_if_expanded(*_args, **_kwargs):
        pytest.fail("an unrelated row expanded the full pricing DAG")

    monkeypatch.setattr(pricing, "_best_column", fail_if_expanded)
    reduced_cost, column = price_flight(
        graph,
        {unrelated: 7.0},
        3.0,
        cfg,
        params,
        forbidden_rows=frozenset({unrelated}),
        require_improving=False,
    )

    assert column == seed
    assert reduced_cost == pytest.approx(params.M - seed.delay_s - 3.0)


def test_negative_dual_disables_the_seed_locality_shortcut():
    """Even an off-seed negative row can make a longer route price best."""

    cfg = _cfg()
    graph, params = _graph(cfg)
    credited = RowKey.cell((-2, 0), 0, 5)
    assert credited not in seed_column(graph, cfg).claims

    reduced_cost, column = price_flight(
        graph,
        {credited: -100.0},
        0.0,
        cfg,
        params,
    )

    assert column is not None
    assert credited in column.claims
    assert column.delay_s == pytest.approx(20.0)
    assert reduced_cost == pytest.approx(
        params.M - column.delay_s + 100.0,
        abs=1e-7,
    )


@pytest.mark.parametrize("terminal_origin", [False, True])
def test_shifted_seed_claims_are_canonical_time_translations(terminal_origin):
    """Every canonical row moves by exactly the integer departure shift."""

    cfg = _cfg()
    params = ColGenParams(solver="highs", detour_slack_hops=4)
    origin = _point((0, 0), cfg)
    destination = _point((4, 0), cfg)
    terminal = Terminal("hub", 2, radius=180.0) if terminal_origin else None
    request = FlightRequest(
        2,
        origin,
        destination,
        0.0,
        0.0,
        origin_terminal=terminal,
    )
    static_terms = () if terminal is None else ((origin, terminal),)
    graph = build_flight_graph(request, cfg, static_terms, params)
    seed = seed_column(graph, cfg)

    for delta in (0, 1, 4, 12):
        shifted = replace(
            seed,
            departure_step=seed.departure_step + delta,
            delay_s=seed.delay_s + delta * cfg.dt_s,
            claims=frozenset(),
        )
        assert pricing._shift_claims(seed.claims, delta) == column_claims(shifted, graph, cfg)


def test_shifted_seed_is_only_an_incumbent_for_exact_pricing(monkeypatch):
    """A cheap held seed tightens pruning but never bypasses the exact DAG."""

    cfg = _cfg()
    graph, params = _graph(cfg)
    seed = seed_column(graph, cfg)
    priced_at_nominal = min(seed.claims, key=lambda row: row.step)
    captured = {}

    def capture_exact_search(*args, **kwargs):
        captured["incumbent"] = kwargs["incumbent"]
        return kwargs["incumbent"]

    monkeypatch.setattr(pricing, "_best_column", capture_exact_search)
    reduced_cost, column = price_flight(
        graph,
        {priced_at_nominal: 100.0},
        0.0,
        cfg,
        params,
        require_improving=False,
    )

    assert captured["incumbent"] is not None
    assert column is not None
    assert column.departure_step == seed.departure_step + 1
    assert column.delay_s == pytest.approx(seed.delay_s + cfg.dt_s)
    assert column.claims == pricing._shift_claims(seed.claims, 1)
    assert reduced_cost == pytest.approx(params.M - column.delay_s)


def test_arc_bound_matches_ground_only_pricing_sweep(monkeypatch):
    """The guarded arc lower bound preserves the exact DP optimum across row signs."""

    cfg = _cfg()
    graph, params = _graph(cfg, slack=2)
    seed = seed_column(graph, cfg)
    dual_sweep = (
        {min(seed.claims, key=lambda row: row.step): 100.0},
        {RowKey.cell((-1, 1), 0, 6): 40.0},
        {RowKey.cell((-2, 0), 0, 5): -100.0},
        {
            min(seed.claims, key=lambda row: row.step): 20.0,
            RowKey.cell((-1, 1), 0, 6): -15.0,
        },
    )
    guarded = [price_flight(graph, duals, 3.0, cfg, params) for duals in dual_sweep]

    monkeypatch.setattr(
        pricing,
        "_arc_delay_lower_bound_s",
        lambda *, ground_delay_s, **_kwargs: ground_delay_s,
    )
    ground_only = [price_flight(graph, duals, 3.0, cfg, params) for duals in dual_sweep]

    for (guarded_rc, guarded_column), (baseline_rc, baseline_column) in zip(guarded, ground_only):
        assert guarded_rc == pytest.approx(baseline_rc, abs=1e-8)
        assert guarded_column == baseline_column


def test_terminal_fold_retention_uses_the_canonical_sqrt_rounding():
    """``hypot`` differs by one ulp here and would unsafely retain the lane."""

    cfg = _cfg()
    dx = 105.50984759064562
    dy = -97.97238970423132
    hub = vec(-dx, -dy, 0.0)
    edge = math.hypot(dx, dy)
    terminal = Terminal(
        "rounding",
        1,
        radius=edge - cfg.corridor_width_m / 2.0,
    )

    assert (0, 0) in {lane.cell for lane in hg.terminal_lanes(hub, terminal, cfg)}
    assert math.sqrt(dx * dx + dy * dy) == 143.98234990080914
    assert exit_radius(terminal, cfg) == 143.98234990080917
    assert pricing._terminal_fold_leg_s(hub, terminal, (0, 0), cfg) == (0.0, False)


def _exhaustive_columns(graph, cfg):
    """Enumerate the complete W-valid column universe of one tiny customer graph."""

    _lo, hi = derive_cell_window(cfg)
    revisit_depth = hi - _lo
    columns = []
    for departure_step in range(graph.base_step, graph.latest_departure_step + 1):
        start_step = departure_step + graph.takeoff_steps[0]
        max_hops = graph.max_step - start_step

        def visit(path):
            hops = len(path) - 1
            if hops >= 1 and path[-1] == graph.dest_cell:
                raw = Column(
                    graph.request.flight_id,
                    departure_step,
                    0,
                    None,
                    None,
                    tuple(path),
                    0.0,
                )
                try:
                    claims = column_claims(raw, graph, cfg)
                except (ValueError, NotImplementedError):
                    pass
                else:
                    intent = column_to_intent(raw, graph.request, cfg)
                    if intent.status is IntentStatus.ACCEPTED:
                        columns.append(
                            replace(
                                raw,
                                delay_s=total_delay_s(intent, cfg),
                                claims=claims,
                            )
                        )
            if hops >= max_hops:
                return
            for neighbour in sorted(hg.hex_neighbors(*path[-1])):
                if neighbour not in graph.corridor_cells:
                    continue
                if neighbour in path[-revisit_depth:]:
                    continue
                if (path[-1], neighbour) in graph.forbidden_hops:
                    continue
                visit((*path, neighbour))

        visit((graph.origin_cell,))
    return tuple(columns)


def test_endpoint_union_bound_matches_exhaustive_pricing_oracle():
    """Positive, negative, and exact-tie endpoint prices preserve the global optimum."""

    cfg = replace(_cfg(), max_ground_delay_s=8.0)
    params = ColGenParams(solver="highs", detour_slack_hops=0)
    request = FlightRequest(7, _point((0, 0), cfg), _point((1, 1), cfg), 0.0, 0.0)
    graph = build_flight_graph(request, cfg, (), params)
    columns = _exhaustive_columns(graph, cfg)
    assert len(columns) == 10

    endpoint_only_cell = (0, 2)
    assert endpoint_only_cell not in graph.corridor_cells
    assert endpoint_only_cell in endpoint_claim_cells(
        request.dest,
        cfg.effective_hover_radius_m,
        cfg,
    )
    early = RowKey.cell(endpoint_only_cell, 0, 6)
    common = RowKey.cell(endpoint_only_cell, 0, 18)
    late = RowKey.cell(endpoint_only_cell, 0, 20)
    dual_sweep = (
        {common: 5.0},
        {common: 5.0, late: 20.0},
        {common: 5.0, early: 4.0},
        {common: 5.0, early: 9.0},
        {common: 5.0, late: -12.0},
    )

    def tie_key(column):
        return (
            len(column.cell_path) - 1,
            column.departure_step,
            -1,
            -1,
            column.cell_path,
        )

    for duals in dual_sweep:
        ranked = sorted(
            (
                (
                    params.M - column.delay_s - sum(duals.get(row, 0.0) for row in column.claims),
                    column,
                )
                for column in columns
            ),
            key=lambda item: (-item[0], tie_key(item[1])),
        )
        oracle_rc, oracle_column = ranked[0]
        reduced_cost, column = price_flight(graph, duals, 0.0, cfg, params)

        assert reduced_cost == pytest.approx(oracle_rc, abs=1e-8)
        assert column == oracle_column


def _weighted_cost(column, graph, cfg, model: CostModel) -> float:
    """Cost one enumerated column WITHOUT reusing pricing's own arithmetic.

    Built from the filed intent's ground/detour split, so the oracle and the thing it
    checks share nothing but the geometry.
    """

    intent = column_to_intent(column, graph.request, cfg)
    ground_s = intent.ground_delay_s
    return model.evaluate(
        ground_s=ground_s, air_detour_s=total_delay_s(intent, cfg) - ground_s
    )


@pytest.mark.parametrize(
    "ground_weight,air_weight", [(1.0, 1.0), (1.0, 3.0), (3.0, 1.0)]
)
def test_pricing_returns_the_exhaustive_optimum_under_any_weighting(
    ground_weight, air_weight
):
    """The search's answer must be the objective's argmin, whichever way it leans.

    The label score is the dominance currency: score two labels in the wrong units and
    the survivor is whichever the tie-break reached first, not the one the objective
    prefers. This fixture has the substitution in it -- 567 columns over 2..9 hops and 7
    departure steps, so ground and air genuinely trade -- and the oracle re-costs every
    one of them from its filed intent, sharing nothing with pricing but the geometry.

    The third case is the discriminating one, and the duals are the seed that exposes it:
    scoring in raw seconds returns a column 8.0 short of the optimum here. With ground
    dearer than air the objective wants an early departure and a longer route, which is
    the opposite of what the ``(hops, departure_step)`` tie-break picks on a raw-seconds
    tie -- so the two only agree while air is the expensive one.
    """

    cfg = replace(
        _cfg(),
        max_ground_delay_s=24.0,
        cost_ground_delay_per_s=ground_weight,
        cost_air_lateral_per_s=air_weight,
        cost_air_hold_per_s=air_weight,
    )
    params = ColGenParams(
        solver="highs", detour_slack_hops=1, objective="total_cost", time_limit_s=60.0
    )
    request = FlightRequest(7, _point((0, 0), cfg), _point((2, 0), cfg), 0.0, 0.0)
    graph = build_flight_graph(request, cfg, (), params)
    columns = _exhaustive_columns(graph, cfg)
    assert len(columns) == 567, "a shrunken universe would hide a miss rather than fail"

    cells = sorted(graph.corridor_cells)
    rng = np.random.default_rng(17)
    duals: dict[RowKey, float] = {}
    for _ in range(int(rng.integers(3, 12))):
        cell = cells[int(rng.integers(0, len(cells)))]
        step = int(rng.integers(graph.min_step, graph.max_step + 1))
        duals[RowKey.cell(cell, 0, step)] = float(rng.uniform(-15.0, 70.0))
    view = DualView(duals, cfg)
    pi_f = float(rng.uniform(0.0, 30.0))

    model = CostModel(ground_weight, air_weight)
    best = max(
        model.reduced_cost(
            benefit=params.M,
            cost=_weighted_cost(column, graph, cfg, model),
            dual_cost=view.claim_cost(column.claims),
            pi_f=pi_f,
        )
        for column in columns
    )

    reduced_cost, priced = price_flight(
        graph, view, pi_f, cfg, params, require_improving=False
    )

    assert priced is not None
    assert reduced_cost == pytest.approx(best, abs=1e-9)
