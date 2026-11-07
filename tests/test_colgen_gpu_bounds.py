"""Canonical geometry and mandatory-row requirements for certified GPU pruning."""

from dataclasses import replace
import math
import random

import numpy as np
import pytest

from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.dp_prepare import PreparedVariants, prepare_variants, prepared_for
from freespace_sim.planner.colgen.gpu_bounds import CompletionBounds
from freespace_sim.planner.colgen.gpu_pricing import _Batch
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph, column_claims
from freespace_sim.planner.colgen.objective import cost_model
from freespace_sim.planner.colgen.pricing_platform import PricingProblem
from freespace_sim.planner.colgen.translate import Column, column_to_intent
from freespace_sim.types import Terminal, vec
from tests.test_colgen_pricing_platform import _assert_canonical, _platform
from tests.test_colgen_solver import _cfg, _params, _request


def _bounds(problem: PricingProblem) -> tuple[CompletionBounds, PreparedVariants]:
    """Prepare one helper without running either CPU or GPU route pricing."""
    topology, rows = prepared_for(problem.graph, problem.cfg)
    view = pricing.DualView(problem.duals, problem.cfg)
    variants = prepare_variants(
        problem.graph,
        problem.cfg,
        view,
        topology,
        rows,
        benefit=problem.params.M,
        pi_f=problem.pi_f,
        model=cost_model(problem.cfg, problem.params),
        forbidden_rows=problem.forbidden_rows,
    )
    return CompletionBounds(problem, topology, variants, view), variants


def _select_variants(variants: PreparedVariants, indices: np.ndarray) -> PreparedVariants:
    """Select/reorder roots while retaining the paid-class table they reference."""
    names = (
        "departure_step", "lane_idx", "cell", "start_step", "score", "ground_delay_s",
        "origin_leg_w_s", "start_dual_cost", "paid_class",
    )
    return replace(variants, **{name: getattr(variants, name)[indices] for name in names})


def _assert_same_bounds(actual: CompletionBounds, expected: CompletionBounds) -> None:
    """Compare geometry and price-dependent bounds for every current root.

    Parameters
    ------------
    - actual (CompletionBounds): result whose geometry may have been reused.
    - expected (CompletionBounds): independent fresh geometry and current price preparation.

    Return
    --------
    - output (None): requires byte-exact coefficients and equal certified-bound calculations.
    """
    for name in ("air_per_hop", "_edge_min", "_reference", "_cost_scale", "_absolute_prices"):
        assert getattr(actual, name) == getattr(expected, name)
    for name in ("air_offset", "_length_offset"):
        assert getattr(actual, name).tobytes() == getattr(expected, name).tobytes()
    p = actual.problem
    from freespace_sim.planner.colgen.gpu_integer_bounds import cost_operands

    for variant in range(actual.variants.n_variants):
        operands = cost_operands(actual, variant)
        for hops, (cost, affine) in enumerate(operands):
            assert cost == actual.cost_lower(variant, hops)
            first = math.nextafter(float(actual.variants.ground_delay_s[variant])
                                   + float(actual.air_offset[variant]), -math.inf)
            assert affine == math.nextafter(first + math.nextafter(hops * actual.air_per_hop, -math.inf),
                                            -math.inf)
        assert actual.root_positive(variant) == expected.root_positive(variant)
        for hops in (0, actual.topology.shortest_hops, actual.topology.air_hop_limit):
            assert actual.cost_lower(variant, hops) == expected.cost_lower(variant, hops)
            arguments = (variant, hops, 0.0, 0.0, p.params.M, p.pi_f, 0.0)
            assert actual.future_upper(*arguments) == expected.future_upper(*arguments)


@pytest.mark.parametrize("translation", [(0, 0), (10000, -9000), (-14000, 11000)])
@pytest.mark.parametrize("objective", ["total_delay", "total_cost"])
def test_geometry_bound_covers_translated_resampled_routes(
    translation: tuple[int, int],
    objective: str,
) -> None:
    """A useful air-cost floor must never exceed independent canonical geometry cost.

    Parameters
    ------------
    - translation (tuple[int, int]): exercises coordinate cancellation and both signs.
    - objective (str): changes air and ground weights without changing route geometry.

    Return
    --------
    - output (None): checks sampled route orders, extra hops and both legal departures.
    """
    cfg = _cfg(
        max_ground_delay_s=4.0,
        max_detour_factor=5.0,
        cost_ground_delay_per_s=7.0,
        cost_air_lateral_per_s=3.0,
    )
    params = _params(max_air_overrun_hops=2, objective=objective)
    q, r = translation
    request = _request(1, (q, r), (q + 12, r + 8), cfg)
    request = replace(
        request,
        origin=request.origin + np.array([7.0, -11.0, 0.0]),
        dest=request.dest + np.array([-8.0, 12.0, 0.0]),
    )
    graph = build_flight_graph(request, cfg, (), params)
    problem = PricingProblem(graph, cfg, params, {}, 0.0)
    fresh, _ = _bounds(problem)
    bounds, variants = _bounds(problem)
    _assert_same_bounds(bounds, fresh)
    model = cost_model(cfg, params)
    rng = random.Random(41)
    for sample in range(12):
        moves = [(1, 0)] * 12 + [(0, 1)] * 8
        if sample % 2:
            moves.remove((1, 0))
            moves.extend(((1, -1), (0, 1)))
        rng.shuffle(moves)
        path = [(q, r)]
        for dq, dr in moves:
            path.append((path[-1][0] + dq, path[-1][1] + dr))
        for variant in range(variants.n_variants):
            column = Column(
                1, int(variants.departure_step[variant]), 0, None, None, tuple(path), 0.0
            )
            actual = model.intent_cost(column_to_intent(column, request, cfg), cfg)
            lower = bounds.cost_lower(variant, len(path) - 1)
            assert lower <= actual
            assert lower > float(variants.ground_delay_s[variant])
            for hops in (0, len(path) // 2, len(path) - 1):
                future = math.nextafter((len(path) - 1 - hops) * bounds.air_per_hop, -math.inf)
                upper = bounds.future_upper(variant, hops, 0.0, future, params.M, 0.0, 0.0)
                assert math.isfinite(upper) and upper >= params.M - actual


def test_reused_geometry_refreshes_prices_and_filtered_reordered_roots() -> None:
    """A later LP snapshot can add rows and change the roots without stale bound data.

    Parameters
    ------------
    - none: builds several terminal lanes and departures, then forbids a late endpoint row.

    Return
    --------
    - output (None): reused geometry matches fresh preparation for new prices and root order.
    """
    cfg = _cfg(
        max_ground_delay_s=12.0, max_detour_factor=5.0,
        cost_ground_delay_per_s=7.0, cost_air_lateral_per_s=3.0,
    )
    params = _params(max_air_overrun_hops=2)
    request = _request(1, (0, 0), (8, 4), cfg)
    request = replace(
        request, origin=request.origin + np.array([17.0, 13.0, 0.0]),
        origin_terminal=Terminal("hub", 1, radius=110.0),
    )
    graph = build_flight_graph(request, cfg, ((request.origin, request.origin_terminal),), params)
    problem = PricingProblem(graph, cfg, params, {RowKey.cell((999, 999), 0, 0): 3.0}, 0.0)
    topology, rows = prepared_for(graph, cfg)
    initial_view = pricing.DualView(problem.duals, cfg)
    variants = prepare_variants(graph, cfg, initial_view, topology, rows, model=cost_model(cfg, params))
    initial_variants = _select_variants(variants, np.flatnonzero(variants.cell == variants.cell[0]))
    initial = CompletionBounds(problem, topology, initial_variants, initial_view)
    cached = graph._search_cache.gpu_geometry
    assert all(initial.root_positive(i) == 0.0 for i in range(initial_variants.n_variants))

    late_claims = pricing._endpoint_claims(
        graph, cfg, origin=True, step=graph.latest_departure_step, timing_steps=0,
    )
    forbidden = frozenset({max(late_claims, key=lambda row: row.step)})
    early_claims = pricing._endpoint_claims(
        graph, cfg, origin=True, step=graph.base_step, timing_steps=0,
    )
    prices = {row: (11.0 if row.kind == "term" else -3.0) for row in early_claims}
    prices[RowKey.cell((999, 999), 0, 1)] = 1e8
    current = replace(problem, duals=prices, pi_f=17.0, forbidden_rows=forbidden)
    view = pricing.DualView(prices, cfg)
    current_variants = prepare_variants(
        graph, cfg, view, topology, rows, model=cost_model(cfg, params), forbidden_rows=forbidden,
    )
    order = np.flatnonzero(current_variants.cell != variants.cell[0])[::-1]
    current_variants = _select_variants(current_variants, order)
    assert current_variants.n_variants > 1
    assert current_variants.ground_delay_s.max() < initial_variants.ground_delay_s.max()
    reused = CompletionBounds(current, topology, current_variants, view)
    assert graph._search_cache.gpu_geometry is cached
    fresh = CompletionBounds(replace(current, graph=replace(graph)), topology, current_variants, view)
    _assert_same_bounds(reused, fresh)
    assert all(
        reused.origin_claims(i) == fresh.origin_claims(i) for i in range(current_variants.n_variants)
    )
    assert reused._cost_scale < initial._cost_scale
    assert reused._absolute_prices > initial._absolute_prices
    assert any(reused.root_positive(i) > 0.0 for i in range(current_variants.n_variants))
    assert len(np.unique(reused.air_offset)) > 1

    # An empty root pass must not erase the bounds for a later unrestricted pass.
    empty = _select_variants(variants, np.array([], dtype=np.intp))
    CompletionBounds(current, topology, empty, view)
    restored = CompletionBounds(current, topology, variants, view)
    restored_fresh = CompletionBounds(replace(current, graph=replace(graph)), topology, variants, view)
    _assert_same_bounds(restored, restored_fresh)


@pytest.mark.parametrize("change", ["objective", "configuration", "topology"])
def test_geometry_rebuilds_for_changed_geometry_or_objective(change: str) -> None:
    """Cache reuse must never retain coefficients from a different geometry or objective.

    Parameters
    ------------
    - change (str): independently changes the objective, configuration or topology's hop cap.

    Return
    --------
    - output (None): changed inputs replace the cache and match a fresh bound calculation.
    """
    cfg = _cfg(max_ground_delay_s=8.0, cost_ground_delay_per_s=7.0)
    params = _params(max_air_overrun_hops=2, objective="total_cost")
    graph = build_flight_graph(_request(1, (0, 0), (8, 4), cfg), cfg, (), params)
    problem = PricingProblem(graph, cfg, params, {}, 0.0)
    initial, variants = _bounds(problem)
    cached = graph._search_cache.gpu_geometry
    topology = initial.topology
    if change == "objective":
        problem = replace(problem, params=replace(params, objective="total_delay"))
        variants = replace(variants, ground_delay_s=variants.ground_delay_s / 7.0)
    elif change == "configuration":
        problem = replace(problem, cfg=replace(cfg, nominal_speed_mps=cfg.nominal_speed_mps / 2.0))
    else:
        topology = replace(topology, air_hop_limit=topology.air_hop_limit + 7)
    view = pricing.DualView({}, problem.cfg)
    changed = CompletionBounds(problem, topology, variants, view)
    assert graph._search_cache.gpu_geometry is not cached
    fresh = CompletionBounds(replace(problem, graph=replace(graph)), topology, variants, view)
    _assert_same_bounds(changed, fresh)
    assert changed._cost_scale != initial._cost_scale


def test_boundary_fold_falls_back_and_origin_prices_include_terminal_rows() -> None:
    """An uncertain fold cannot overprice air, and terminal rows remain mandatory.

    Parameters
    ------------
    - none: uses a retained-lane rounding counterexample and a customer destination.

    Return
    --------
    - output (None): checks fallback cost and positive-price union against canonical claims.
    """
    cfg = _cfg(max_ground_delay_s=0.0, max_detour_factor=5.0)
    dx, dy = 105.50984759064562, -97.97238970423132
    hub = vec(-dx, -dy, 0.0)
    terminal = Terminal("rounding", 1, radius=math.hypot(dx, dy) - cfg.corridor_width_m / 2.0)
    request = replace(_request(1, (0, 0), (8, 0), cfg), origin=hub, origin_terminal=terminal)
    params = _params(max_air_overrun_hops=2)
    graph = build_flight_graph(request, cfg, ((hub, terminal),), params)
    problem = PricingProblem(graph, cfg, params, {}, 0.0)
    initial, variants = _bounds(problem)
    topology = initial.topology
    variant = next(
        i
        for i, cell in enumerate(variants.cell)
        if (int(topology.cell_q[cell]), int(topology.cell_r[cell])) == (0, 0)
    )
    claims = initial.origin_claims(variant)
    prices = {row: (11.0 if row.kind == "term" else -7.0) for row in claims}
    bounds, _ = _bounds(replace(problem, duals=prices))
    assert bounds.cost_lower(variant, graph.max_air_hops) == 0.0
    assert any(row.kind == "term" for row in claims)
    expected = math.fsum(max(0.0, price) for price in prices.values())
    assert 0.0 < bounds.root_positive(variant) <= expected
    assert bounds.root_positive(variant) == pytest.approx(expected, abs=1e-10)
    path = pricing._shortest_cell_path(graph, (0, 0), (8, 0))
    assert path is not None
    column = Column(
        1,
        int(variants.departure_step[variant]),
        0,
        int(variants.lane_idx[variant]),
        None,
        path,
        0.0,
    )
    assert claims.issubset(column_claims(column, graph, cfg))


@pytest.mark.parametrize("reachable_destination", [True, False])
def test_geometry_bound_uses_only_reachable_destination_lanes(
    monkeypatch: pytest.MonkeyPatch, reachable_destination: bool,
) -> None:
    """Disconnected terminal lanes cannot crash pricing or invalidate its cost bound.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): forbids original CPU pricing during the Metal check.
    - reachable_destination (bool): keeps the reachable lanes or removes every stop
      from the bound's prepared domain to exercise its conservative fallback.

    Return
    --------
    - output (None): verifies a canonical Metal result or a ground-only empty-set bound.
    """
    cfg = _cfg(max_ground_delay_s=0.0, max_detour_factor=5.0)
    params = _params(max_air_overrun_hops=0)
    destination = Terminal("dest", 1, radius=70.0)
    request = replace(_request(1, (0, 0), (8, 0), cfg), dest_terminal=destination)
    wall = Terminal("wall", 1, radius=30.0)
    wall_position = request.dest + vec(120.0 * math.cos(math.pi / 6.0), 60.0, 0.0)
    graph = build_flight_graph(
        request, cfg, ((request.dest, destination), (wall_position, wall)), params,
    )
    topology, rows = prepared_for(graph, cfg)
    cells = set(zip(map(int, topology.cell_q), map(int, topology.cell_r)))
    assert set(pricing._destination_options(graph)) - cells
    assert topology.dest_mask.any()
    model = cost_model(cfg, params)
    seed = pricing._shortest_seed(graph, cfg, model=model)
    assert seed is not None
    view = pricing.DualView({}, cfg)
    variants = prepare_variants(
        graph, cfg, view, topology, rows, benefit=params.M, pi_f=0.0,
        model=model, forbidden_rows=frozenset(),
    )
    assert variants.n_variants > 0
    if not reachable_destination:
        topology = replace(topology, dest_mask=np.zeros_like(topology.dest_mask))
    problem = PricingProblem(graph, cfg, params, view, 0.0)
    bounds = CompletionBounds(problem, topology, variants, view)
    if not reachable_destination:
        assert bounds.air_per_hop == 0.0
        assert np.all(np.isneginf(bounds.air_offset))
        for variant in range(variants.n_variants):
            assert bounds.cost_lower(variant, topology.air_hop_limit) == 0.0
        return

    def forbid_cpu(*args: object, **kwargs: object) -> None:
        """Reject substitution of the original CPU search for GPU pricing."""
        raise AssertionError("Original CPU pricing is forbidden")

    for name in ("price_flight", "_best_column", "_best_column_compiled"):
        monkeypatch.setattr(pricing, name, forbid_cpu)
    (result,) = _platform("metal").price_batch([problem])
    assert result.certified and result.search_complete
    assert result.stats["gpu_dispatches"] > 0
    assert result.upper_bound >= params.M - seed.delay_s
    _assert_canonical(problem, result)


@pytest.mark.parametrize("magnitude", [1.0, 1e8])
def test_future_bound_covers_canonical_signed_price_and_score_rounding(magnitude: float) -> None:
    """Coupled prefix/suffix arithmetic covers canonical rounding despite cancellation.

    Parameters
    ------------
    - magnitude (float): scales cancelling positive and negative original prices.

    Return
    --------
    - output (None): bound encloses canonical RC for every split of the same complete route.
    """
    cfg = _cfg(flight_levels_m=(30.0,), max_ground_delay_s=0.0, max_detour_factor=5.0)
    params = _params(max_air_overrun_hops=2, M=1e12)
    request = _request(1, (0, 0), (3, 2), cfg)
    graph = build_flight_graph(request, cfg, (), params)
    path = ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2))
    column = Column(1, graph.base_step, 0, None, None, path, 0.0)
    offsets = pricing.derive_cell_window(cfg)
    start = graph.base_step + graph.takeoff_steps[0]
    visits = [pricing._visit_claims(cell, 0, start + hop, offsets) for hop, cell in enumerate(path)]
    selected = [min(visits[index], key=lambda row: row.step) for index in (0, 2, 4)]
    prices = dict(zip(selected, (magnitude, -magnitude + 0.125, 0.25), strict=True))
    problem = PricingProblem(graph, cfg, params, prices, params.M - 13.0)
    bounds, _variants = _bounds(problem)
    view = pricing.DualView(prices, cfg)
    cost = cost_model(cfg, params).intent_cost(column_to_intent(column, request, cfg), cfg)
    actual = cost_model(cfg, params).reduced_cost(
        benefit=params.M,
        cost=cost,
        dual_cost=view.claim_cost(column_claims(column, graph, cfg)),
        pi_f=problem.pi_f,
    )
    negative = math.nextafter(math.fsum(value for value in prices.values() if value < 0), -math.inf)
    for hops in range(len(path)):
        prefix = 0.0
        for row in set().union(*visits[: hops + 1]):
            prefix = math.nextafter(prefix + max(0.0, view.row_cost(row)), -math.inf)
        suffix = 0.0
        for visit in visits[hops + 1 :]:
            suffix = math.nextafter(suffix + bounds.air_per_hop, -math.inf)
            for row in visit:
                suffix = math.nextafter(suffix + max(0.0, view.row_cost(row)), -math.inf)
        upper = bounds.future_upper(0, hops, prefix, suffix, params.M, problem.pi_f, negative)
        assert actual <= upper <= actual + 0.02


@pytest.mark.parametrize("buffer", [0.0, 4.0])
def test_incremental_positive_prices_match_every_generated_prefix_union(
    monkeypatch: pytest.MonkeyPatch,
    buffer: float,
) -> None:
    """GPU prefixes charge origin overlaps and legal revisits once across reused searches.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): shadows the optimized price bound with a full union.
    - buffer (float): exercises narrow and wider disjoint-visit windows.

    Return
    --------
    - output (None): every checked bound covers the same positive rows as path reconstruction.
    """
    platform = _platform("metal")
    cfg = _cfg(
        flight_levels_m=(30.0,),
        max_ground_delay_s=4.0,
        time_buffer_s=buffer,
        hover_radius_m=180.0,
        max_detour_factor=50.0,
    )
    params = _params(max_air_overrun_hops=4)
    graph = build_flight_graph(_request(1, (0, 0), (1, 0), cfg), cfg, (), params)
    prices = (-17.0, 0.0, 23.0, 41.0, 0.0)
    duals = {
        RowKey.cell(cell, 0, step): prices[(cell[0] * 7 + cell[1] * 11 + step) % len(prices)]
        for cell in graph.corridor_cells
        for step in range(graph.max_step + 8)
    }
    # This unavailable negative row keeps the safe global credit deliberately
    # loose, so the full search must exercise longer prefixes and legal revisits.
    duals[RowKey.cell((1000, 1000), 0, 0)] = -1e6
    problem = PricingProblem(graph, cfg, params, duals, 0.0)
    original = _Batch.positive_lower
    seen = {"checks": 0, "overlap": False, "revisit": False}
    departures: set[int] = set()

    def checked(batch: _Batch, root: int, parent: int, cell: int, hops: int) -> float:
        """Compare each optimized bound against independent reconstructed claim membership."""
        value = original(batch, root, parent, cell, hops)
        i, variant = batch.root_flights[root], batch.root_variants[root]
        p, v, view = batch.problems[i], batch.variants[i], batch.views[i]
        departure = int(v.departure_step[variant])
        start = int(v.start_step[variant])
        path = batch.path(parent, cell)
        origin = pricing._endpoint_claims(
            p.graph, p.cfg, origin=True, step=departure, timing_steps=0
        )
        claims = set(origin)
        for hop, position in enumerate(path):
            visit = pricing._visit_claims(position, 0, start + hop, view.offsets)
            if hop and visit & origin:
                seen["overlap"] = True
            claims.update(visit)
        expected = math.fsum(max(0.0, view.row_cost(row)) for row in claims)
        endpoint = batch.endpoints[i]
        assert endpoint is not None
        assert batch.paid_mask(root, parent, cell, hops) == endpoint.mask(claims)
        assert value <= expected
        assert value == pytest.approx(expected, abs=1e-9)
        departures.add(departure)
        seen["checks"] += 1
        seen["revisit"] |= len(path) > len(set(path))
        return value

    monkeypatch.setattr(_Batch, "positive_lower", checked)
    for multiplier in (1.0, 1.5):
        current = replace(problem, duals={row: price * multiplier for row, price in duals.items()})
        (result,) = platform.price_batch((current,))
        assert result.search_complete and result.certified
        assert result.stats["bootstrap_gpu_dispatches"] > 0
    assert seen["checks"] > 0 and seen["overlap"] and seen["revisit"]
    assert len(departures) > 1
