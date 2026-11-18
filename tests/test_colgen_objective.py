"""The colgen objective has exactly one definition, and pricing obeys it.

``CostModel`` is the source of truth for every reduced cost, bound and master coefficient
in column generation.  These tests pin two things: that a weighted objective actually
reaches every one of those sites, and that the (1.0, 1.0) default is a genuine no-op.
Several assertions below are bit-identity rather than approximate agreement, because the
longhand sums this replaced are only safely expressible as ``evaluate`` calls if term
order and association were preserved exactly.
"""

import math
import time
from dataclasses import fields, replace
from fractions import Fraction

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.conflict import volumes_conflict
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen import network as network_mod
from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph
from freespace_sim.planner.colgen.objective import DELAY_MODEL, CostModel, cost_model
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import DualView, price_flight, seed_column
from freespace_sim.planner.colgen.translate import Column
from freespace_sim.planner.colgen.zeta import ZetaGraph, ZetaOptions, ZetaPricingResult, price_zeta
from freespace_sim.planner.colgen.zeta_geometry import anyangle_to_intent, reservation_claims
from freespace_sim.planner.colgen.zeta_history import PriceHistory
from freespace_sim.planner.colgen.zeta_history_guard import HistoryGuard
from freespace_sim.types import FlightRequest, Terminal, vec
from freespace_sim.volumes import enroute_flown_m, exit_radius
from tests._colgen_support import with_air_hops


def _cfg(**overrides) -> SimConfig:
    values = {
        "planner": "colgen",
        "flight_levels_m": (100.0,),
        "airspace_ceiling_m": 125.0,
        "region_size_m": (20_000.0, 20_000.0),
        "terminal_airspace_always_active": True,
        "max_ground_delay_s": 48.0,
        "max_detour_factor": 10.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _point(cell, cfg):
    x, y = hg.hex_center(*cell, hg.circumradius(cfg))
    return vec(x, y, cfg.ground_level_m)


def _graph(cfg, origin=(0, 0), dest=(4, -1), overrun=4, flight_id=1, **kw):
    params = ColGenParams(solver="highs", max_air_overrun_hops=overrun, **kw)
    request = FlightRequest(flight_id, _point(origin, cfg), _point(dest, cfg), 0.0, 0.0)
    return build_flight_graph(request, cfg, (), params), params


def _random_duals(rng, cells, base_step, count):
    duals: dict[RowKey, float] = {}
    for _ in range(count):
        cell = cells[int(rng.integers(0, len(cells)))]
        step = int(rng.integers(base_step - 3, base_step + 30))
        key = RowKey.cell(cell, 0, step)
        duals[key] = duals.get(key, 0.0) + float(rng.gamma(2.0, 3.0))
    return duals


# ----------------------------------------------------------------- the function itself


def test_evaluate_at_unit_weights_is_bit_identical_to_the_longhand_sum():
    """The (1.0, 1.0) default must be a no-op down to the last bit.

    Every call site previously wrote ``ground + hold + detour`` inline.  Multiplying by
    1.0 is exact in IEEE-754, so the only thing that could break bit-identity is a change
    of association -- which is why ``evaluate`` sums in that same left-to-right order.
    """

    rng = np.random.default_rng(20260805)
    for _ in range(2000):
        ground, hold, detour = (float(v) for v in rng.gamma(2.0, 30.0, size=3))
        assert DELAY_MODEL.evaluate(
            ground_s=ground, air_hold_s=hold, air_detour_s=detour
        ) == ground + hold + detour


def test_evaluate_applies_ground_and_air_weights_separately():
    model = CostModel(ground_weight=1.0, air_weight=3.0)
    assert model.evaluate(ground_s=10.0, air_detour_s=10.0) == 40.0
    assert model.evaluate(ground_s=10.0, air_hold_s=10.0, air_detour_s=10.0) == 70.0


def test_cost_model_defaults_to_the_configs_weights_not_to_equal_ones():
    """A default-constructed ``ColGenParams`` prices what the config says, 1:3.

    This is the assertion that changed when the default moved off ``total_delay``, and it
    is worth stating as a ratio rather than as an identity: equal weights make
    ``ground + flown`` invariant under a ground-for-air swap, so whole families of columns
    tie EXACTLY and dominance keeps whichever the tie-break reached first.  A default that
    silently returned ``DELAY_MODEL`` would put every un-flagged run in that regime.
    """

    cfg = _cfg()
    model = cost_model(cfg, ColGenParams())
    assert model is not DELAY_MODEL
    assert model.ground_weight == cfg.cost_ground_delay_per_s
    assert model.air_weight == cfg.cost_air_lateral_per_s
    assert model.air_weight == 3.0 * model.ground_weight

    # No params at all still means the historical model: `cost_model` reads
    # `getattr(params, "objective", "total_delay")`, and a caller with no params object is
    # not a caller who opted out of the config's weights -- it is one that has no config
    # opinion to read.  Kept explicit so the two paths cannot drift silently.
    assert cost_model(cfg, None) is DELAY_MODEL
    assert cost_model(cfg, ColGenParams(objective="total_delay")) is DELAY_MODEL


def test_total_cost_refuses_a_config_the_cost_model_cannot_express():
    """One air scalar has to cover cruise and loiter, so divergence must raise.

    ``CostModel`` carries a single air weight and pricing charges it per step.  Were hold
    priced differently from lateral, it would silently bill loiter as cruise -- exactly
    the class of silent mismatch this module exists to prevent.
    """

    cfg = _cfg(cost_air_hold_per_s=5.0, cost_air_lateral_per_s=3.0)
    with pytest.raises(ValueError, match="cost_air_hold_per_s"):
        cost_model(cfg, ColGenParams(objective="total_cost"))


# --------------------------------------------------------- the model reaches every site


def test_seed_cache_is_keyed_on_the_cost_model():
    """A cached seed must not answer for an objective it was not costed under.

    ``_shortest_seed_columns`` memoises on ``fg._search_cache``.  That cache used to be
    consulted without regard to ``model``, so pricing one graph under two objectives --
    the natural way to write a comparison -- returned the FIRST model's seed both times,
    silently.  Nothing raised; the second answer was simply wrong.

    This is the failure mode the rest of the objective refactor was built to eliminate:
    a weight that fails to reach a computation and reports a plausible number anyway.
    """

    cfg = _cfg()
    weighted = CostModel(ground_weight=1.0, air_weight=3.0)
    assert weighted.air_weight != weighted.ground_weight, "fixture needs unequal weights"

    # One graph, both models, delay-first -- the order that used to poison the cache.
    graph, _params = _graph(cfg)
    delay_seed = seed_column(graph, cfg)
    cost_seed = seed_column(graph, cfg, model=weighted)

    # Independently: a graph that has only ever seen the weighted model.
    reference = seed_column(_graph(cfg)[0], cfg, model=weighted)

    assert cost_seed.delay_s == pytest.approx(reference.delay_s, abs=1e-9), (
        "the cached unweighted seed leaked into the weighted query"
    )
    assert cost_seed.delay_s != pytest.approx(delay_seed.delay_s, abs=1e-9), (
        "this fixture must have air time, or the two models cannot differ"
    )
    # And the reverse order, so the fix is not an artifact of which model asked first.
    graph2, _ = _graph(cfg)
    assert seed_column(graph2, cfg, model=weighted).delay_s == pytest.approx(
        cost_seed.delay_s, abs=1e-9
    )
    assert seed_column(graph2, cfg).delay_s == pytest.approx(delay_seed.delay_s, abs=1e-9)


# ------------------------------------------------------------------- the default is a no-op


def _weighted_probe(ground_weight: float, air_weight: float):
    """Price one fixed congested instance under a given ground:air weighting.

    Identical in every respect except the weights, so the ground/air split of the answer
    is attributable to the objective and nothing else.
    """

    cfg = _cfg(
        cost_ground_delay_per_s=ground_weight,
        cost_air_lateral_per_s=air_weight,
        cost_air_hold_per_s=air_weight,
    )
    # The air-time ceiling is lifted: the ground-heavy arm's whole point is absorbing delay
    # in the AIR by flying further, which is exactly what that ceiling bounds. At the
    # shipped budget it would decide this test, not the objective -- and the objective is
    # what is under test. Lifted at the GRAPH, not through `ColGenParams`: one knob sizes the
    # corridor and the budget together (issue #78), so lifting it there would widen the
    # corridor with it and this fixture would stop being the one below is derived from.
    corridor = 1
    graph, params = _graph(cfg, dest=(3, 0), overrun=corridor, objective="total_cost")
    graph = with_air_hops(graph, graph.shortest_hops + 64)
    rng = np.random.default_rng(7)
    cells = sorted(graph.corridor_cells)
    # Duals have to land on steps a route actually occupies. Drawing them across `max_step`
    # made this fixture a function of the air-time ceiling: `max_step` is denominated in
    # `max_air_hops`, so lifting the ceiling widens the horizon, the same nine draws scatter
    # over a longer axis, and the instance stops being the one these assertions were derived
    # from -- at overrun=64 both arms collapsed to the same answer and the test measured
    # nothing. The departure window plus a geodesic route is what the search reaches whatever
    # ceiling is left over, so anchoring there makes the instance depend on the weighting
    # alone, which is the variable under test.
    dual_span = (
        (graph.latest_departure_step - graph.base_step)
        + graph.shortest_hops
        + corridor
        + 4
    )
    duals: dict[RowKey, float] = {}
    for _ in range(9):
        cell = cells[int(rng.integers(0, len(cells)))]
        step = int(rng.integers(graph.min_step, graph.min_step + dual_span + 1))
        duals[RowKey.cell(cell, 0, step)] = float(rng.uniform(-20.0, 60.0))
    _rc, column = price_flight(
        graph, DualView(duals, cfg), float(rng.uniform(0.0, 40.0)), cfg, params,
        require_improving=False,
    )
    assert column is not None
    return column


def test_the_objective_steers_the_ground_air_trade_in_the_search_itself():
    """The label score is the DOMINANCE currency, so it has to be the objective.

    Within one time layer `ground + flown` is invariant -- a step of ground delay buys
    exactly one hop of air -- so two labels reaching the same cell at the same step with
    different splits are tied in raw seconds and NOT tied under any other weighting. The
    search used to score them in raw seconds, which made dominance keep whichever the
    `(hops, departure_step)` tie-break reached first: fewest hops. That coincides with the
    objective whenever air costs at least as much as ground, which is why the shipped
    1:3 config never showed it -- and inverts the moment it does not.

    Measured on this fixture with ground priced above air, the raw-seconds score returned
    a column costing 48.0 where 32.0 was available.
    """

    air_heavy = _weighted_probe(ground_weight=1.0, air_weight=3.0)
    ground_heavy = _weighted_probe(ground_weight=3.0, air_weight=1.0)

    # Air expensive: hold on the pad and fly the short route.
    assert air_heavy.departure_step > 0
    assert len(air_heavy.cell_path) - 1 == 3

    # Ground expensive: leave immediately and absorb the same congestion in the air.
    assert ground_heavy.departure_step == 0
    assert len(ground_heavy.cell_path) - 1 > 3
    assert ground_heavy.delay_s == pytest.approx(32.0, abs=1e-9)


def test_the_shipped_weighting_is_unchanged_by_that_currency():
    """...and the fix is a no-op where the tie-break already agreed.

    The rest of this suite runs at unit weights, so bit-identity there is covered by it
    passing. This pins the other regime the repo actually ships: 1:3, where the answer
    must be the same one the raw-seconds score produced.
    """

    column = _weighted_probe(ground_weight=1.0, air_weight=3.0)

    assert column.departure_step == 8
    assert len(column.cell_path) - 1 == 3
    assert column.delay_s == pytest.approx(32.0, abs=1e-9)


def test_reduced_cost_is_the_same_arithmetic_it_replaced():
    rng = np.random.default_rng(5)
    for _ in range(1000):
        benefit, cost, dual, pi = (float(v) for v in rng.normal(50.0, 200.0, size=4))
        assert DELAY_MODEL.reduced_cost(
            benefit=benefit, cost=cost, dual_cost=dual, pi_f=pi
        ) == benefit - cost - dual - pi


def test_arc_delay_lower_bound_scales_with_the_model():
    """The admissible bound has to move with the objective or pruning goes wrong.

    Too small and it prunes nothing; too large and it discards the optimum. Under a pure
    air reweighting the excess term scales and the ground term does not.
    """

    kw = dict(
        ground_delay_s=8.0,
        origin_fold_s=1.0,
        hops=3,
        remaining_hops=4,
        destination_fold_s=2.0,
        reference_time_s=5.0,
        dt_s=4.0,
        folding_exact=True,
    )
    plain = pricing._arc_delay_lower_bound_s(**kw, model=DELAY_MODEL)
    scaled = pricing._arc_delay_lower_bound_s(**kw, model=CostModel(1.0, 3.0))
    excess = plain - 8.0
    assert excess > 0.0, "the fixture must exercise the air term"
    assert scaled == pytest.approx(8.0 + 3.0 * excess)

    # The conservative branch is ground only, so an air weight must not touch it.
    ground_only = dict(kw, folding_exact=False)
    assert pricing._arc_delay_lower_bound_s(
        **ground_only, model=CostModel(1.0, 3.0)
    ) == pytest.approx(8.0)
    assert pricing._arc_delay_lower_bound_s(
        **ground_only, model=CostModel(2.0, 3.0)
    ) == pytest.approx(16.0)


def test_shifting_a_column_later_charges_ground_weight_only():
    """A clock translation moves ground delay; the route, and so the air term, is fixed."""

    from freespace_sim.planner.colgen import solver as solver_mod

    cfg = _cfg()
    graph, params = _graph(cfg)
    seed = pricing.seed_column(graph, cfg)
    dt = float(cfg.dt_s)

    plain = solver_mod._shift_column(seed, seed.departure_step + 3, cfg, DELAY_MODEL)
    assert plain.delay_s == pytest.approx(seed.delay_s + 3 * dt)

    scaled = solver_mod._shift_column(
        seed, seed.departure_step + 3, cfg, CostModel(ground_weight=2.0, air_weight=3.0)
    )
    assert scaled.delay_s == pytest.approx(seed.delay_s + 2.0 * (3 * dt))
    assert math.isfinite(scaled.delay_s)


# ------------------------------------------------- the endpoint claim cache


def _terminal_graph(cfg):
    """A graph whose endpoints are both terminals, so the *term* row branch is reached."""

    origin, dest = _point((0, 0), cfg), _point((4, -1), cfg)
    origin_terminal = Terminal("cache-A", 1, radius=90.0)
    dest_terminal = Terminal("cache-B", 1, radius=90.0)
    request = FlightRequest(
        7, origin, dest, 0.0, 0.0,
        origin_terminal=origin_terminal,
        dest_terminal=dest_terminal,
    )
    params = ColGenParams(solver="highs", max_air_overrun_hops=4)
    static = [(origin, origin_terminal), (dest, dest_terminal)]
    return build_flight_graph(request, cfg, static, params), params


@pytest.mark.parametrize("terminals", [False, True])
def test_endpoint_claims_cache_matches_the_uncached_oracle(terminals):
    """The memoized endpoint rows are the uncached ones, over the whole reachable key space.

    Both endpoint shapes are exercised because they do not share a step rule: a terminal
    endpoint claims *term* rows over an unpadded interval that ignores ``timing_steps``,
    while a bare point claims *cell* rows over a range rounded outward by a drift that
    scales WITH ``timing_steps``.  A cache that collapsed the two would still pass on
    ``colgen_test``-shaped graphs and fail on every terminal one.
    """

    cfg = _cfg()
    graph, params = _terminal_graph(cfg) if terminals else _graph(cfg)
    assert (graph.origin_terminal is not None) is terminals

    # Drive a real pricing call first, so the keys under test are the ones the search
    # actually reaches rather than a hand-picked range.  Price the SEED's own rows rather
    # than sampling the corridor: a seed that pays nothing is provably optimal, and
    # `price_flight` then returns it without ever running the DAG search that fills this
    # cache.  On the terminal graph the seed touches 3 of 46 corridor cells, so random
    # duals miss it almost every time.
    seed = seed_column(graph, cfg)
    duals = {row: 4.0 for row in seed.claims if row.kind == "cell"}
    assert duals, "seed claims no cell rows, so it cannot be priced off its shortcut"
    price_flight(graph, DualView(duals, cfg), 0.0, cfg, params, require_improving=False)
    reached = dict(graph._search_cache.endpoint_claims)
    assert reached, "pricing reached no endpoint claim sets"

    for (origin, step, timing_steps), cached in reached.items():
        expected = pricing._endpoint_claims_uncached(
            graph, cfg, origin=origin, step=step, timing_steps=timing_steps
        )
        assert cached == expected, (origin, step, timing_steps)
        # A terminal endpoint claims `term` rows; a bare point claims `cell` rows.  Pinned
        # so a future unification of the two span rules cannot pass this test silently.
        assert all(row.kind == ("term" if terminals else "cell") for row in cached)

    # Sweep beyond what pricing happened to reach, including keys it never asked for.
    for origin in (True, False):
        for step in range(graph.min_step, min(graph.min_step + 12, graph.max_step + 1)):
            for timing_steps in (0, 1, 5, 17):
                assert pricing._endpoint_claims(
                    graph, cfg, origin=origin, step=step, timing_steps=timing_steps
                ) == pricing._endpoint_claims_uncached(
                    graph, cfg, origin=origin, step=step, timing_steps=timing_steps
                )


def test_endpoint_claims_cache_is_bounded():
    """The LRU cap holds under more distinct keys than it can store."""

    cfg = _cfg()
    graph, _ = _graph(cfg)
    for step in range(network_mod._MAX_ENDPOINT_CLAIMS + 64):
        pricing._endpoint_claims(graph, cfg, origin=True, step=step, timing_steps=0)
        assert len(graph._search_cache.endpoint_claims) <= network_mod._MAX_ENDPOINT_CLAIMS

    # Evicting is answer-neutral: the coldest keys are simply recomputed on demand.
    assert pricing._endpoint_claims(
        graph, cfg, origin=True, step=0, timing_steps=0
    ) == pricing._endpoint_claims_uncached(graph, cfg, origin=True, step=0, timing_steps=0)


def _exhaustive_anyangle_columns(
    graph: network_mod.FlightGraph,
    cfg: SimConfig,
    params: ColGenParams,
    row_index: network_mod.RowIndex,
    radius: float,
    options: ZetaOptions = ZetaOptions(),
) -> list[Column]:
    """Enumerate every feasible walk independently of pricing's labels and bounds.

    Parameters
    ------------
    - graph (FlightGraph): declared waypoint, departure and terminal choices.
    - cfg (SimConfig): simulator rules used to rebuild every complete reservation.
    - params (ColGenParams): unchanged physical objective weights.
    - row_index (RowIndex): terminal capacity authority.
    - radius (float): maximum permitted waypoint connection length.
    - options (ZetaOptions): declared arrival timing, speed variation and connection domain.

    Return
    --------
    - output (list[Column]): all accepted complete walks with canonical costs and claims.
    """
    cells = tuple(graph.corridor_cells)
    points = {c: _point(c, cfg) for c in cells}
    model = cost_model(cfg, params)
    columns = []
    limit = graph.max_air_hops * cfg.nominal_speed_mps * cfg.dt_s
    restricted = {(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                  (2, -1), (1, -2), (-1, -1), (-2, 1), (-1, 2), (1, 1)}
    if options.connection_mode == "hex12_long":
        restricted |= {(-3, 1), (-3, 2), (-2, -1), (-2, 3), (-1, -2), (-1, 3),
                       (1, -3), (1, 2), (2, -3), (2, 1), (3, -2), (3, -1)}
    forward_graph = (ZetaGraph(graph, cfg, row_index, replace(options, connection_radius_m=radius))
                     if options.forward_cruise else None)

    protected = frozenset(w.terminal_id for w in graph.static_walls)
    origins = [(i, lane.cell) for i, lane in enumerate(graph.origin_lanes)
               if lane.cell in cells] if graph.origin_terminal else [(None, graph.origin_cell)]
    destinations = [(i, lane.cell) for i, lane in enumerate(graph.dest_lanes)
                    if lane.cell in cells] if graph.dest_terminal else [(None, graph.dest_cell)]

    def walk(path, length, origin_lane, dest_lane, goal):
        if path[-1] == goal and len(path) > 1:
            for dep in range(graph.base_step, graph.latest_departure_step + 1):
                column = Column(graph.request.flight_id, dep, 0, origin_lane, dest_lane, path, 0.0)
                intent = anyangle_to_intent(
                    column, graph.request, cfg, arrival_snap_s=options.arrival_snap_s,
                    speed_fraction=options.speed_fraction,
                )
                if intent.accepted and not any(volumes_conflict(v, w)
                                               for v in intent.volumes for w in graph.static_walls):
                    columns.append(replace(
                        column, delay_s=model.intent_cost(intent, cfg),
                        claims=reservation_claims(intent.volumes, cfg, row_index=row_index,
                                                  protected_terminal_ids=protected),
                    ))
        for target in (forward_graph.neighbors(path[-1]) if forward_graph is not None else cells):
            offset = target[0] - path[-1][0], target[1] - path[-1][1]
            if options.connection_mode != "full" and offset not in restricted:
                continue
            distance = math.dist(points[path[-1]], points[target])
            remaining = math.dist(points[target], points[goal])
            if 0 < distance <= radius + 1e-8 and length + distance + remaining <= limit + 1e-7:
                walk(path + (target,), length + distance, origin_lane, dest_lane, goal)

    for origin_lane, start in origins:
        for dest_lane, goal in destinations:
            walk((start,), 0.0, origin_lane, dest_lane, goal)
    return columns


def _check_anyangle_future_price_bound(
    graph: ZetaGraph,
    columns: list[Column],
    prices: dict[RowKey, float],
    params: ColGenParams,
    *, require_buffer_gain: bool = False,
) -> float:
    """Compare continuation bounds against every canonical customer-route suffix.

    Parameters
    ------------
    - graph (ZetaGraph): declared customer waypoint and timing choices.
    - columns (list[Column]): exhaustive feasible complete routes in that domain.
    - prices (dict[RowKey, float]): frozen signed reservation prices.
    - params (ColGenParams): canonical objective weights for the joint continuation floor.
    - require_buffer_gain (bool): also prove a buffered fixture improves on point sampling.

    Return
    --------
    - output (float): largest valid bound observed across complete paid prefixes.
    """
    from freespace_sim.planner.colgen.zeta_bound import FuturePriceBound

    cfg = graph.cfg
    model = cost_model(cfg, params)
    table = FuturePriceBound.build(graph, prices, deadline=None)
    assert table is not None
    joint = FuturePriceBound.build(graph, prices, deadline=None, air_weight=model.air_weight)
    assert joint is not None
    sampled = sampled_joint = None
    if require_buffer_gain:
        sampled = FuturePriceBound.build(graph, prices, deadline=None, buffered_periods=False)
        sampled_joint = FuturePriceBound.build(
            graph, prices, deadline=None, air_weight=model.air_weight, buffered_periods=False,
        )
        assert sampled is not None and sampled_joint is not None
    negative = math.fsum(price for price in prices.values() if price < 0.0)
    remaining_by_state = {}
    completion_by_state = {}
    paid_overlap = False
    for column in columns:
        intent = graph.intent(column)
        cursor = 0
        length = positive_paid = 0.0
        previous_paid = frozenset()
        ground = model.ground_weight * ((column.departure_step - graph.fg.base_step) * cfg.dt_s)
        completion = column.delay_s + math.fsum(prices.get(row, 0.0) for row in column.claims)
        for index, cell in enumerate(column.cell_path):
            previous_cursor = cursor
            if index:
                cursor += 1
                point = _point(cell, cfg)
                while not np.allclose(intent.centerline[cursor][0][:2], point[:2],
                                      atol=1e-8, rtol=0.0):
                    cursor += 1
            for (start, _), (end, _) in zip(intent.centerline[previous_cursor:cursor],
                                           intent.centerline[previous_cursor + 1:cursor + 1]):
                dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
                length += math.sqrt(dx * dx + dy * dy)
            stamp = intent.centerline[cursor][1]
            paid = reservation_claims(intent.volumes[:cursor + 1], cfg, row_index=graph.row_index)
            positive_paid += math.fsum(prices.get(row, 0.0) for row in paid - previous_paid
                                       if prices.get(row, 0.0) > 0.0)
            previous_paid = paid
            future = reservation_claims(intent.volumes[cursor + 1:], cfg, row_index=graph.row_index)
            paid_overlap |= any(prices.get(row, 0.0) > 0.0 for row in paid & future)
            remaining = math.fsum(max(0.0, prices.get(row, 0.0)) for row in future - paid)
            key = cell, stamp, paid
            remaining_by_state[key] = min(remaining_by_state.get(key, math.inf), remaining)
            key = cell, stamp, length, paid, positive_paid, ground
            completion_by_state[key] = min(completion_by_state.get(key, math.inf), completion)
    assert paid_overlap
    values = []
    buffer_gain = 0.0
    for (cell, stamp, paid), remaining in remaining_by_state.items():
        value = table.value(cell, stamp, paid)
        assert 0.0 <= value <= remaining + 1e-9
        live = frozenset(row for row in paid
                         if (row.step + 1) * cfg.dt_s > stamp - max(0.0, cfg.time_buffer_s))
        assert 0.0 <= table.value(cell, stamp, live) <= remaining + 1e-9
        if sampled is not None:
            for retained in (paid, live):
                improvement = table.value(cell, stamp, retained) - sampled.value(cell, stamp, retained)
                assert improvement >= -1e-9
                buffer_gain = max(buffer_gain, improvement)
        values.append(value)
    for (cell, stamp, length, paid, positive_paid, ground), completion in completion_by_state.items():
        live = frozenset(row for row in paid
                         if (row.step + 1) * cfg.dt_s > stamp - max(0.0, cfg.time_buffer_s))
        for retained in (paid, live):
            units = sum(joint.row_units.get(row, 0) for row in retained
                        if joint.row_samples.get(row, -math.inf) > stamp)
            floor = joint.floor(cell, stamp, length, units, positive_paid, ground, negative)
            assert floor <= completion + 1e-9
            if sampled_joint is not None:
                sampled_units = sum(sampled_joint.row_units.get(row, 0) for row in retained
                                    if sampled_joint.row_samples.get(row, -math.inf) > stamp)
                old_floor = sampled_joint.floor(
                    cell, stamp, length, sampled_units, positive_paid, ground, negative,
                )
                assert floor >= old_floor - 1e-9
                buffer_gain = max(buffer_gain, floor - old_floor)
    if require_buffer_gain:
        assert buffer_gain > 1e-6
    for row, stamp in table.row_samples.items():
        for cell in graph.cells:
            value = table.value(cell, stamp, ())
            if value > 0.0:
                assert table.value(cell, stamp, (row,)) == value
                after = math.nextafter(stamp, math.inf)
                assert table.value(cell, after, (row,)) == table.value(cell, after, ())
                return max(values)
    return max(values)


def _check_anyangle_completion_bounds(
    graph: ZetaGraph, columns: list[Column], prices: dict[RowKey, float], params: ColGenParams,
) -> dict[str, float]:
    """Check every component's floor against actual complete routes at every prefix.

    Parameters
    ------------
    - graph (ZetaGraph): finite route universe and enabled completion relaxations.
    - columns (list[Column]): independently enumerated canonical feasible routes.
    - prices (dict[RowKey, float]): current signed prices, including overlapping rows.
    - params (ColGenParams): unchanged objective and pricing tolerance.

    Return
    --------
    - output (dict[str, float]): strongest additional component floor beyond sunk prices.
    """
    from freespace_sim.planner.colgen.zeta_completion import CompletionBounds
    from freespace_sim.planner.colgen.zeta_completion_geometry import geometry
    from freespace_sim.planner.colgen.zeta_geometry import volume_spatial_cells
    from freespace_sim.planner.colgen.zeta_timing import edge_end_times
    from freespace_sim.volumes import Volume4D

    cfg, model = graph.cfg, cost_model(graph.cfg, params)
    from freespace_sim.planner.colgen.zeta_lattice_bound import LatticeDistanceBound
    lattice = (LatticeDistanceBound.build(graph, model, prices)
               if graph.options.lattice_distance_bound else None)
    bounds = CompletionBounds.build(graph, prices, model)
    assert set(bounds.names) == set(graph.options.completion_bounds)
    relaxed_cells, outgoing = geometry(graph, prices)
    relaxed_ids = {cell: index for index, cell in enumerate(relaxed_cells)}
    relaxed_edges = {(source, edge.target): edge for source, group in enumerate(outgoing)
                     for edge in group}
    negative = math.fsum(price for price in prices.values() if price < 0.0)
    strongest = dict.fromkeys(bounds.names, -math.inf)
    for column in columns:
        lane = next(steps for index, _cell, steps in graph.origins
                    if index == column.origin_lane_idx)
        stamp = (column.departure_step + graph.fg.takeoff_steps[0] + lane) * cfg.dt_s
        ground = model.ground_weight * ((column.departure_step - graph.fg.base_step) * cfg.dt_s)
        paid = {row for row in graph.rows(graph.endpoint(True, column.departure_step * cfg.dt_s))
                if row in prices}
        positive = math.fsum(prices[row] for row in paid if prices[row] > 0.0)
        length = 0.0
        goal = column.cell_path[-1]
        fold = graph.fold_length(column.origin_lane_idx, column.dest_lane_idx)
        actual = column.delay_s + math.fsum(prices.get(row, 0.0) for row in column.claims)
        cruise = math.fsum(part.length
                          for i, (first, last) in enumerate(zip(column.cell_path, column.cell_path[1:]))
                          for part in graph.parts(first, last, i == 0, i + 2 == len(column.cell_path)))
        for index, cell in enumerate(column.cell_path):
            if index:
                parts = graph.parts(column.cell_path[index - 1], cell, index == 1,
                                    index == len(column.cell_path) - 1)
                assert parts is not None
                relaxed = relaxed_edges[relaxed_ids[column.cell_path[index - 1]], relaxed_ids[cell]]
                assert relaxed.lengths == tuple(part.length for part in parts)
                assert relaxed.durations == tuple(part.duration for part in parts)
                assert all(covered <= volume_spatial_cells(part.shape, cfg)
                           for covered, part in zip(relaxed.covered, parts, strict=True))
                ends = edge_end_times(stamp, tuple(part.duration for part in parts),
                                      graph.options.arrival_snap_s, graph.options.speed_fraction)
                rows = set()
                for part, end in zip(parts, ends, strict=True):
                    rows.update(graph.rows(Volume4D(part.shape, stamp - cfg.time_buffer_s,
                                                    end + cfg.time_buffer_s, part.terminal_id)))
                    stamp = end
                    length += part.length
                fresh = {row for row in rows if row in prices} - paid
                positive += math.fsum(prices[row] for row in fresh if prices[row] > 0.0)
                paid |= fresh
            live = {row for row in paid
                    if (row.step + 1) * cfg.dt_s > stamp - max(0.0, cfg.time_buffer_s)}
            if lattice is not None:
                remaining = lattice.remaining(cell, goal)
                assert length + remaining <= cruise + 1e-7
                physical = ground
                if fold is not None and graph.reference > 1e-9:
                    physical += model.air_weight * max(0.0, length + remaining + fold
                                                       - graph.reference) / cfg.nominal_speed_mps
                assert physical <= column.delay_s + 1e-9
                assert physical + positive + negative <= actual + 1e-9
            for retained in (paid, live):
                for name, bound in zip(bounds.names, bounds.components, strict=True):
                    value = bound.floor(cell, goal, stamp, length, positive, ground,
                                        negative, fold, retained,
                                        terminal=index == len(column.cell_path) - 1)
                    assert value <= actual + 1e-9, (name, column, index, value, actual)
                    strongest[name] = max(strongest[name], value - ground - positive - negative)
    expired = CompletionBounds.build(graph, prices, model, deadline=time.perf_counter() - 1.0)
    assert not expired.components
    return strongest


def _without_anyangle_cache_metrics(result: ZetaPricingResult) -> ZetaPricingResult:
    """Compare answers and search work while allowing reuse and elapsed-time counters to differ."""
    counters = {field.name: 0 for field in fields(result)
                if field.name in {"wall_s", "bound_build_s", "incumbent_setup_s", "topology_setup_s"}
                or field.name.startswith(("shared_leg_cache_", "shared_footprint_cache_",
                                          "persistent_timing_cache_"))}
    return replace(result, **counters)


def _anyangle_history_type(compiled_history: bool) -> type:
    """Select the real helper used by interpreted or native history comparisons."""
    if not compiled_history:
        return PriceHistory
    pytest.importorskip("Cython")
    import sys

    from freespace_sim.planner.colgen.zeta_native import prepare_compiled_loop

    name, _binary = prepare_compiled_loop()
    return sys.modules[name].PriceHistory


@pytest.mark.parametrize("objective", ["total_delay", "total_cost"])
@pytest.mark.parametrize("forbid", [False, True])
@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("terminal", ["customer", "origin", "both"])
@pytest.mark.parametrize("backend,lazy,prune_snap,compiled_loop,completion_bounds", [
    ("reference", False, False, False, ()), ("reference", True, False, False, ()),
    ("compiled", False, False, False, ()), ("compiled", True, False, False, ()),
    ("reference", True, True, False, ()), ("compiled", True, True, False, ()),
    ("compiled", False, False, True, ()), ("compiled", True, True, True, ()),
    ("reference", True, True, False, ("arrival",)),
    ("reference", True, True, False, ("resource",)),
    ("reference", True, True, False, ("selected",)),
    ("reference", True, True, False, ("crossing",)),
    ("reference", True, True, False, ("arrival", "resource", "selected", "crossing")),
    ("compiled", True, True, False, ("arrival", "resource", "selected", "crossing")),
    ("compiled-shared", True, True, True, ("arrival", "resource", "selected", "crossing")),
    ("reference-shared", True, True, False, ()),
    ("compiled-shared", True, True, True, ()),
    ("reference-strong", True, True, False, ()),
    ("compiled-strong", True, True, False, ()),
    ("compiled-strong", True, True, True, ()),
    ("compiled-strong-native-history", True, True, True, ()),
    ("reference-hex12", True, True, False, ()),
    ("compiled-strong-native-history-hex12_long", True, True, True, ()),
    ("reference-hex12-forward", True, True, False, ()),
    ("compiled-hex12-forward", True, True, False, ()),
    ("compiled-strong-native-history-hex12_long-forward", True, True, True, ()),
    ("reference-hex12-lattice", True, False, False, ()),
    ("compiled-radius240-lattice", True, True, False, ()),
    ("compiled-strong-native-history-hex12-forward-lattice", True, True, True, ()),
], ids=["reference-eager", "reference-lazy", "compiled-eager", "compiled-lazy",
        "reference-pruned-snapped", "compiled-pruned-snapped",
        "compiled-loop-eager", "compiled-loop-pruned-snapped", "arrival-bound",
        "resource-bound", "selected-bound", "crossing-bound", "all-bounds-reference",
        "all-bounds-compact", "all-bounds-native-shared", "reference-shared",
        "compiled-loop-shared", "reference-strong-history", "compact-strong-history",
        "native-strong-history", "compiled-history-helper", "hex12-reference",
        "hex12-long-native-history", "forward-hex12-reference", "forward-hex12-compact",
        "forward-hex12-long-native-history", "lattice-hex12-reference",
        "lattice-radius240-compact", "lattice-forward-hex12-native"])
def test_anyangle_exact_pricing_matches_exhaustive_physical_walks(
    objective, forbid, merge, terminal, backend, lazy, prune_snap, compiled_loop, completion_bounds,
):
    """Finite-domain certification must match full physical union pricing, including credits."""
    lattice_distance_bound = backend.endswith("-lattice")
    backend = backend.removesuffix("-lattice")
    radius = 240.0 if backend.endswith("-radius240") else 480.0
    backend = backend.removesuffix("-radius240")
    forward_cruise = backend.endswith("-forward")
    backend = backend.removesuffix("-forward")
    connection_mode = next((mode for mode in ("hex12", "hex12_long")
                            if backend.endswith("-" + mode)), "full")
    if connection_mode != "full":
        backend = backend.removesuffix("-" + connection_mode)
    compiled_history = backend.endswith("-native-history")
    backend = backend.removesuffix("-native-history")
    strong = backend.endswith("-strong")
    backend = backend.removesuffix("-strong")
    shared = backend.endswith("-shared")
    backend = backend.removesuffix("-shared")
    if backend == "compiled":
        pytest.importorskip("numba")
    if compiled_loop:
        pytest.importorskip("Cython")
    cfg = _cfg(max_ground_delay_s=8.0, max_detour_factor=2.0)
    fg, params = _graph(cfg, dest=(2, 0), overrun=1, objective=objective)
    row_index = network_mod.RowIndex()
    if terminal != "customer":
        hub_radius = 80.0 if strong or lattice_distance_bound else 90.0
        hub = Terminal("oracle-hub", capacity=2, radius=hub_radius)
        dest_hub = (Terminal("oracle-destination", capacity=1, radius=hub_radius)
                    if terminal == "both" else None)
        request = FlightRequest(1, _point((0, 0), cfg), _point((5 if dest_hub else 4, 0), cfg),
                                0.3, 0.3, origin_terminal=hub, dest_terminal=dest_hub)
        terms = [(request.origin, hub)] + ([(request.dest, dest_hub)] if dest_hub else [])
        fg = build_flight_graph(request, cfg, terms, params)
        # A declared tiny waypoint domain makes exhaustive comparison practical.
        fg = replace(fg, corridor_cells=frozenset({(1, 0), (2, 0), (3, 0), (3, -1), (4, 0)}),
                     max_air_hops=4)
        row_index.register_terminal(hub)
        if dest_hub:
            row_index.register_terminal(dest_hub)
    options = ZetaOptions(connection_radius_m=radius, merge_identical_states=merge,
                          backend=backend, lazy_successors=lazy, expire_history=prune_snap,
                          history_dominance=prune_snap, arrival_snap_s=float(prune_snap),
                          future_price_bound=prune_snap and not completion_bounds and not strong,
                          strong_history_dominance=strong,
                          compiled_loop=compiled_loop, compiled_history=compiled_history,
                          completion_bounds=completion_bounds,
                          leg_cache=shared, persistent_timing_cache=shared,
                          connection_mode=connection_mode, forward_cruise=forward_cruise,
                          lattice_distance_bound=lattice_distance_bound)
    graph = ZetaGraph(fg, cfg, row_index, options)
    columns = _exhaustive_anyangle_columns(fg, cfg, params, row_index, radius, options)
    rows = sorted({r for c in columns for r in c.claims})
    rng = np.random.default_rng(54)
    prices = {r: float(rng.choice([-0.25, 0.0, 0.5, 3.0, 8.0])) for r in rows}
    if strong and backend == "reference" and merge and not forbid:
        model = cost_model(cfg, params)
        guard = HistoryGuard(graph, model, prices)
        guarded = 0
        for column in columns:
            margin = guard.margin(column.origin_lane_idx, column.dest_lane_idx)
            if margin is None:
                continue
            guarded += 1
            intent = graph.intent(column)
            flown = enroute_flown_m([point for point, _stamp in intent.centerline],
                                   fg.request.origin, fg.request.dest,
                                   fg.origin_terminal, fg.dest_terminal, cfg)
            accumulated = 0.0
            for index, (source, target) in enumerate(zip(column.cell_path, column.cell_path[1:])):
                for part in graph.parts(source, target, index == 0,
                                        index == len(column.cell_path) - 2):
                    accumulated += part.length
            estimate = accumulated + graph.fold_length(column.origin_lane_idx, column.dest_lane_idx)
            assert len(intent.centerline) - 1 <= guard.max_parts
            assert abs(flown - estimate) <= guard.length_error
            estimated_cost = model.evaluate(
                ground_s=intent.ground_delay_s,
                air_detour_s=max(0.0, estimate - graph.reference) / cfg.nominal_speed_mps,
            )
            assert abs(column.delay_s - estimated_cost) <= margin
        assert guarded > 0
    if (completion_bounds or lattice_distance_bound) and merge and not forbid and backend == "reference":
        _check_anyangle_completion_bounds(graph, columns, prices, params)
    seed = graph.seed_columns(params)[0]
    blockable = sorted(r for r in seed.claims if terminal == "customer" or r.kind == "term")
    forbidden = frozenset([blockable[0]]) if forbid else frozenset()
    legal = [c for c in columns if c.claims.isdisjoint(forbidden)]
    pi_f = 17.5
    scores = [params.M - pi_f - c.delay_s - math.fsum(prices[r] for r in c.claims)
              for c in legal]
    expected = max(scores, default=-math.inf)
    result = price_zeta(graph, prices, pi_f, params, forbidden_rows=forbidden)
    if lattice_distance_bound and terminal == "customer":
        assert result.lattice_bound_calls == 1
    assert result.complete
    assert result.reduced_cost == pytest.approx(expected, abs=1e-6)
    assert result.upper_bound >= expected - 1e-6
    assert result.upper_bound <= expected + 1e-6
    assert result.column is not None and result.column in legal
    assert len(columns) > 3
    assert any(hg.hex_distance(a, b) > 1 for c in columns
               for a, b in zip(c.cell_path, c.cell_path[1:]))
    if (forward_cruise or lattice_distance_bound) and terminal == "customer" and merge and not forbid:
        changed_prices = {row: -price for row, price in prices.items()}
        changed_expected = max(params.M - pi_f - c.delay_s
                               - math.fsum(changed_prices[row] for row in c.claims) for c in legal)
        changed = price_zeta(graph, changed_prices, pi_f, params)
        assert changed.complete and changed.column in legal
        assert changed.reduced_cost == pytest.approx(changed_expected, abs=1e-6)
        assert changed.upper_bound == pytest.approx(changed_expected, abs=1e-6)

    early_options = replace(options, early_return=True)
    early_graph = ZetaGraph(fg, cfg, row_index, early_options)
    early = price_zeta(early_graph, prices, pi_f, params, forbidden_rows=forbidden)
    assert early.column in legal
    assert early.upper_bound >= expected - 1e-6
    assert early.reduced_cost <= expected + 1e-6
    if early.complete:
        assert early.reduced_cost == pytest.approx(expected, abs=1e-6)
    else:
        assert early.reason == "improving_column"
        assert early.reduced_cost > params.epsilon

    # Raw pool entries must be canonicalized before checking whether a route is new.
    raw_known = replace(result.column, delay_s=result.column.delay_s + 123.0,
                        claims=frozenset())
    duplicate = price_zeta(early_graph, prices, pi_f, params, known_columns=(raw_known,),
                           forbidden_rows=forbidden)
    assert duplicate.complete and duplicate.reason == "optimal"
    assert duplicate.reduced_cost == pytest.approx(expected, abs=1e-6)

    # A threshold certificate may stop before identifying the exact best route.
    priced_best = result.column.delay_s + math.fsum(prices[r] for r in result.column.claims)
    boundary_pi = params.M - priced_best - 0.25
    threshold = params.M - boundary_pi - priced_best
    assert 0.0 < threshold < 0.5
    boundary = price_zeta(early_graph, prices, boundary_pi, replace(params, epsilon=threshold),
                          forbidden_rows=forbidden)
    assert boundary.complete and boundary.reason in {"optimal", "no_improvement"}
    assert boundary.reduced_cost <= threshold + 1e-6
    assert boundary.upper_bound >= threshold - 1e-6
    if boundary.reason == "no_improvement":
        assert boundary.upper_bound <= threshold
    else:
        assert boundary.reduced_cost == pytest.approx(threshold, abs=1e-6)
    no_improvement = price_zeta(early_graph, prices, boundary_pi + 1.0, params,
                                forbidden_rows=forbidden)
    assert no_improvement.complete and no_improvement.reason in {"optimal", "no_improvement"}
    assert no_improvement.column in legal
    assert no_improvement.reduced_cost <= params.epsilon
    assert no_improvement.upper_bound >= threshold - 1.0 - 1e-6
    assert no_improvement.upper_bound <= params.epsilon
    if no_improvement.reason == "optimal":
        assert no_improvement.upper_bound == pytest.approx(threshold - 1.0, abs=1e-6)

    if objective == "total_delay" and terminal == "customer" and merge and not forbid:
        worst = max(legal, key=lambda column: column.delay_s
                    + math.fsum(prices[row] for row in column.claims))
        worst_value = worst.delay_s + math.fsum(prices[row] for row in worst.claims)
        assert worst_value > priced_best + 1e-6
        far_pi = params.M + 1_000_000.0
        proof_graph = ZetaGraph(fg, cfg, row_index, replace(early_options, max_expansions=1))
        proof = price_zeta(proof_graph, prices, far_pi, params, known_columns=(worst,))
        far_optimum = max(params.M - far_pi - column.delay_s
                          - math.fsum(prices[row] for row in column.claims) for column in legal)
        assert proof.complete and proof.reason == "no_improvement"
        assert proof.column == worst
        assert proof.reduced_cost < far_optimum - 1e-6
        assert far_optimum - 1e-6 <= proof.upper_bound <= params.epsilon
        assert proof.upper_bound < 0.0  # Preserve the actual certificate, not a zero placeholder.
        exact_capped = price_zeta(
            ZetaGraph(fg, cfg, row_index, replace(options, max_expansions=1)),
            prices, far_pi, params, known_columns=(worst,),
        )
        assert not exact_capped.complete and exact_capped.reason == "search_cap"

        for multiple in (0.5, 4.0):
            near_pi = params.M - priced_best - multiple * params.epsilon
            near_optimum = max(params.M - near_pi - column.delay_s
                               - math.fsum(prices[row] for row in column.claims) for column in legal)
            near = price_zeta(early_graph, prices, near_pi, params, known_columns=(worst,))
            assert near.column in legal
            assert near.upper_bound >= near_optimum - 1e-9
            if multiple < 1.0:
                assert 0.0 < near_optimum < params.epsilon
                assert near.complete and near.reason in {"optimal", "no_improvement"}
                assert near.upper_bound <= params.epsilon
            else:
                assert near_optimum > params.epsilon
                assert near.reason in {"optimal", "improving_column"}
                assert near.reduced_cost > params.epsilon

        one_departure = replace(fg, latest_departure_step=fg.base_step)
        threshold_graph = ZetaGraph(one_departure, cfg, row_index, early_options)
        threshold_seed = threshold_graph.seed_columns(params)[0]
        assert threshold_seed.delay_s == 0.0
        origin_rows = threshold_graph.rows(threshold_graph.endpoint(True, fg.base_step * cfg.dt_s))
        required_row = min(origin_rows)
        threshold_prices = {required_row: 9899.699999}
        threshold_pi = 100.3
        search_score = params.M - threshold_pi - threshold_prices[required_row]
        accepted_score = cost_model(cfg, params).reduced_cost(
            benefit=params.M, cost=0.0, dual_cost=threshold_prices[required_row], pi_f=threshold_pi,
        )
        assert accepted_score <= params.epsilon < search_score
        borderline = price_zeta(threshold_graph, threshold_prices, threshold_pi, params)
        assert borderline.complete and borderline.reason == "optimal"
        assert borderline.reduced_cost == search_score

    # Limits within a sweep must include all unexamined continuations in the certificate.
    capped_runs = [(replace(options, max_expansions=cap), None) for cap in (1, 4, 16)]
    capped_runs += [(replace(options, max_labels=cap), None) for cap in (1, 4, 16)]
    capped_runs.append((options, time.perf_counter() - 1.0))
    for capped, deadline in capped_runs:
        limited = price_zeta(ZetaGraph(fg, cfg, row_index, capped), prices, pi_f, params,
                             forbidden_rows=forbidden, deadline=deadline)
        assert limited.upper_bound >= expected - 1e-6
        assert limited.reduced_cost <= expected + 1e-6
        if limited.complete:
            assert limited.reduced_cost == pytest.approx(expected, abs=1e-6)
            assert limited.upper_bound <= expected + 1e-6
        if capped.max_labels == 1 or deadline is not None:
            assert not limited.complete
        if limited.column is not None:
            assert limited.column in legal


def test_anyangle_shared_leg_cache_preserves_geometry_clocks_and_bounded_reuse(monkeypatch):
    """Shared entries preserve exact physical claims, request-specific walls and bounded retention."""
    from freespace_sim.planner.colgen import zeta_leg_cache as cache
    from freespace_sim.planner.colgen.windows import terminal_claim_steps
    from freespace_sim.planner.colgen.zeta_geometry import volume_spatial_cells
    from freespace_sim.planner.colgen.zeta_timing import edge_end_times

    cache.clear_caches()
    cfg = _cfg(corridor_width_m=10.0, hover_radius_m=5.0, max_ground_delay_s=0.0)
    fg, params = _graph(cfg, dest=(4, 0), overrun=2)
    graph = ZetaGraph(fg, cfg, network_mod.RowIndex(), ZetaOptions())
    source, target = (0, 0), (2, 1)
    a, b = graph.point(source), graph.point(target)
    expected = graph.parts(source, target, False, False)
    assert expected is not None
    saved = cache.get_leg(a, b, cfg)
    assert [(part.shape, part.duration, part.length) for part in saved] == [
        (part.shape, part.duration, part.length) for part in expected
    ]
    for part in saved:
        assert cache.get_footprint(part.shape, cfg) == volume_spatial_cells.__wrapped__(part.shape, cfg)
    assert cache.get_leg(a, b, cfg) == saved
    assert cache.stats()["leg_hits"] > 0
    wider = replace(cfg, corridor_width_m=cfg.corridor_width_m + 13.0)
    wider_fg, _ = _graph(wider, dest=(4, 0), overrun=2)
    wider_graph = ZetaGraph(wider_fg, wider, network_mod.RowIndex(), ZetaOptions())
    wider_expected = wider_graph.parts(source, target, False, False)
    wider_saved = cache.get_leg(a, b, wider)
    assert [part.shape for part in wider_saved] == [part.shape for part in wider_expected]
    assert wider_saved[0].shape != saved[0].shape
    slower = replace(cfg, nominal_speed_mps=cfg.nominal_speed_mps * .75)
    slow_parts = cache.get_leg(a, b, slower)
    assert all(part.duration == max(part.length / slower.nominal_speed_mps, 1e-3)
               for part in slow_parts)
    assert math.fsum(part.duration for part in slow_parts) > math.fsum(part.duration for part in saved)

    # Wall decisions belong to a flight even when the exact physical leg is shared.
    options = ZetaOptions(leg_cache=True)
    open_graph = ZetaGraph(fg, cfg, network_mod.RowIndex(), options)
    assert open_graph.parts((0, 0), (4, 0), False, False) is not None
    wall = Terminal("shared-cache-foreign-wall", capacity=1, radius=20.0)
    closed_fg = build_flight_graph(fg.request, cfg, ((_point((2, 0), cfg), wall),), params)
    closed_graph = ZetaGraph(closed_fg, cfg, network_mod.RowIndex(), options)
    assert closed_graph.parts((0, 0), (4, 0), False, False) is None
    assert ZetaGraph(fg, cfg, network_mod.RowIndex(), options).parts(
        (0, 0), (3, 0), False, False,
    ) is not None

    durations = tuple(part.duration for part in saved)
    for dt in (4.0, .7):
        for buffer in (0.0, math.nextafter(dt, math.inf)):
            clock_cfg = replace(cfg, dt_s=dt, time_buffer_s=buffer)
            for snap in (0.0, 1.0):
                for start in (0.0, math.nextafter(dt, -math.inf), dt,
                              math.nextafter(dt, math.inf), 10000.3):
                    ends, firsts, stops = cache.get_timing(start, durations, snap, .05, dt, buffer)
                    assert tuple(ends) == edge_end_times(start, durations, snap, .05)
                    stamp = start
                    for end, first, stop in zip(ends, firsts, stops, strict=True):
                        periods = terminal_claim_steps(stamp - buffer, float(end) + buffer, clock_cfg)
                        assert (first, stop) == (periods.start, periods.stop)
                        stamp = float(end)
                    assert all(not array.flags.writeable for array in (ends, firsts, stops))
                    repeated = cache.get_timing(start, durations, snap, .05, dt, buffer)
                    assert all(np.array_equal(first, second) for first, second in zip(
                        (ends, firsts, stops), repeated, strict=True,
                    ))
    assert cache.stats()["timing_hits"] > 0

    try:
        with monkeypatch.context() as small:
            small.setattr(cache, "GEOMETRY_CACHE_BYTES", 16384)
            small.setattr(cache, "TIMING_CACHE_BYTES", 16384)
            cache.clear_caches()
            first_leg = cache.get_leg(a, b, cfg)
            first_times = cache.get_timing(0.0, durations, 1.0, .05, cfg.dt_s, cfg.time_buffer_s)
            assert 0 < cache.stats()["geometry_bytes"] <= 16384
            assert 0 < cache.stats()["timing_bytes"] <= 16384
            for index in range(1, 65):
                shift = index * 1000.0
                shifted_a = (a[0] + shift, a[1], a[2])
                shifted_b = (b[0] + shift, b[1], b[2])
                cache.get_leg(shifted_a, shifted_b, cfg)
                cache.get_timing(float(index), durations, 1.0, .05, cfg.dt_s, cfg.time_buffer_s)
                assert cache.stats()["geometry_bytes"] <= 16384
                assert cache.stats()["timing_bytes"] <= 16384
            before = cache.stats()
            assert cache.get_leg(a, b, cfg) == first_leg
            rebuilt = cache.get_timing(0.0, durations, 1.0, .05, cfg.dt_s, cfg.time_buffer_s)
            assert all(np.array_equal(first, second) for first, second in zip(first_times, rebuilt, strict=True))
            assert cache.stats()["leg_misses"] == before["leg_misses"] + 1
            assert cache.stats()["timing_misses"] == before["timing_misses"] + 1
            small.setattr(cache, "GEOMETRY_CACHE_BYTES", 1)
            small.setattr(cache, "TIMING_CACHE_BYTES", 1)
            cache.clear_caches()
            assert cache.get_leg(a, b, cfg) == first_leg
            bypassed = cache.get_timing(0.0, durations, 1.0, .05, cfg.dt_s, cfg.time_buffer_s)
            assert all(np.array_equal(first, second) for first, second in zip(first_times, bypassed, strict=True))
            assert cache.stats()["geometry_bytes"] == cache.stats()["timing_bytes"] == 0
    finally:
        cache.clear_caches()


@pytest.mark.parametrize("backend,compiled_search,cache_bytes,compiled_loop", [
    ("reference", False, 0, False), ("compiled", False, 0, False),
    ("compiled", True, 0, False), ("compiled", True, 1, False),
    ("compiled", True, 2048, False), ("compiled", False, 16 * 1024 * 1024, False),
    ("compiled", True, 16 * 1024 * 1024, False),
    ("compiled", True, 16 * 1024 * 1024, True),
    ("compiled", True, 0, True),
    ("compiled", False, 16 * 1024 * 1024, True),
    ("reference-shared", False, 0, False),
    ("compiled-shared", True, 0, False),
    ("compiled-shared", True, 16 * 1024 * 1024, True),
    ("compiled-coverage", True, 16 * 1024 * 1024, True),
], ids=["reference", "legacy-compiled", "compiled-no-cache", "cache-bypass",
        "small-cache", "cache-without-compiled-search", "cached", "compiled-loop-cached",
        "compiled-loop-no-cache", "compiled-loop-python-scans", "reference-shared",
        "shared-without-local-cache", "compiled-loop-shared", "compiled-loop-coverage"])
@pytest.mark.parametrize("buffer", [0.0, math.nextafter(0.7, math.inf)])
@pytest.mark.parametrize("expire,dominance,snap,uncapped,future_bound,completion_bounds", [
    (False, False, 0.0, False, False, ()), (True, False, 0.0, False, False, ()),
    (True, True, 1.0, False, False, ()), (True, True, 1.0, True, False, ()),
    (True, True, 0.0, True, True, ()), (True, True, 1.0, True, True, ()),
    (True, True, 0.0, True, False, ("arrival", "resource", "selected", "crossing")),
    (True, True, 1.0, True, False, ("arrival", "resource", "selected", "crossing")),
], ids=["full-history", "expiry", "expiry-dominance-snapped", "uncapped-pruned-snapped",
        "future-bound-nominal", "future-bound-snapped", "all-bounds-nominal", "all-bounds-snapped"])
def test_anyangle_pricing_reuses_geometry_with_new_prices_and_exact_clock_windows(
    backend, compiled_search, cache_bytes, compiled_loop, buffer, expire, dominance, snap, uncapped,
    future_bound, completion_bounds, monkeypatch,
):
    """Search compilation and bounded caches must preserve new duals, blocks and exact clocks."""
    compiled_route_coverage = backend.endswith("-coverage")
    backend = backend.removesuffix("-coverage")
    shared = backend.endswith("-shared")
    backend = backend.removesuffix("-shared")
    if backend == "compiled":
        pytest.importorskip("numba")
    if compiled_loop:
        pytest.importorskip("Cython")
    cfg = _cfg(dt_s=0.7, time_buffer_s=buffer, corridor_width_m=10.0,
               hover_radius_m=5.0, max_ground_delay_s=2.1, max_detour_factor=2.0)
    fg, params = _graph(cfg, dest=(2, 0), overrun=1)
    row_index = network_mod.RowIndex()
    options = ZetaOptions(
        backend=backend, lazy_successors=True, connection_radius_m=4 * cfg.corridor_segment_len_m,
        expire_history=expire, history_dominance=dominance, arrival_snap_s=snap,
        future_price_bound=future_bound, compiled_search=compiled_search,
        timed_cache_bytes=cache_bytes, compiled_loop=compiled_loop, completion_bounds=completion_bounds,
        leg_cache=shared, persistent_timing_cache=shared,
        route_coverage_cache=compiled_route_coverage,
        compiled_route_coverage=compiled_route_coverage,
    )
    if uncapped:
        options = replace(options, max_expansions=None, max_labels=None)
    graph = ZetaGraph(fg, cfg, row_index, options)
    columns = _exhaustive_anyangle_columns(
        fg, cfg, params, row_index, graph.options.connection_radius_m, graph.options,
    )
    rows = sorted({row for column in columns for row in column.claims})
    varied = [row for row in rows if 0 < sum(row in c.claims for c in columns) < len(columns)]
    assert varied
    priced = varied[len(varied) // 2]
    newly_priced = next(row for row in varied if row != priced)
    previous = ()
    for prices, forbidden in [
        ({}, frozenset()),
        ({priced: 25.0}, frozenset()),
        ({priced: 25.0, newly_priced: -7.0}, frozenset()),
        ({priced: -25.0}, frozenset()),
        ({priced: 0.0}, frozenset({priced})),
        ({priced: -25.0}, frozenset(rows)),
        ({priced: -25.0}, frozenset()),
    ]:
        legal = [c for c in columns if c.claims.isdisjoint(forbidden)]
        expected = max((params.M - c.delay_s - math.fsum(prices.get(r, 0.0) for r in c.claims)
                        for c in legal), default=-math.inf)
        known_before = previous
        result = price_zeta(graph, prices, 0.0, params, known_columns=previous,
                            forbidden_rows=forbidden)
        assert 0 <= result.timed_cache_peak_bytes <= cache_bytes
        assert result.complete
        assert result.reduced_cost == pytest.approx(expected, abs=1e-6)
        assert result.upper_bound == pytest.approx(expected, abs=1e-6)
        if legal:
            assert result.column in legal
            assert result.column.claims == next(c.claims for c in legal if c == result.column)
            previous = (result.column,)
        else:
            assert result.column is None
        if shared:
            control = price_zeta(
                ZetaGraph(fg, cfg, row_index, replace(options, leg_cache=False,
                                                     persistent_timing_cache=False)),
                prices, 0.0, params, known_columns=known_before,
                forbidden_rows=forbidden,
            )
            assert _without_anyangle_cache_metrics(control) == _without_anyangle_cache_metrics(result)
            if control.column is not None:
                assert control.column.claims == result.column.claims
            other_id = fg.request.flight_id + 1000
            other_fg = replace(fg, request=replace(fg.request, flight_id=other_id))
            other = price_zeta(ZetaGraph(other_fg, cfg, row_index, options), prices, 0.0, params,
                               known_columns=tuple(replace(column, flight_id=other_id)
                                                   for column in known_before),
                               forbidden_rows=forbidden)
            if other.column is not None:
                other = replace(other, column=replace(other.column, flight_id=fg.request.flight_id))
            assert _without_anyangle_cache_metrics(other) == _without_anyangle_cache_metrics(result)
            if other.column is not None:
                assert other.column in legal and other.column.claims == result.column.claims

    prices = {row: float(index % 5) + 0.25 for index, row in enumerate(rows)}
    expected = max(params.M - c.delay_s - math.fsum(prices[r] for r in c.claims)
                   for c in columns)
    if compiled_loop:
        from freespace_sim.planner.colgen import zeta_native

        def unexpected_build(*args, **kwargs):
            """An expired pricing call must not start or wait for a native build."""
            raise AssertionError("expired pricing attempted compilation")

        with monkeypatch.context() as cold:
            cold.setattr(zeta_native, "_LOADED", None)
            cold.setattr(zeta_native, "_run", unexpected_build)
            expired = price_zeta(graph, prices, 0.0, params, deadline=time.perf_counter() - 1.0)
        assert not expired.complete and expired.reason == "deadline"
        assert expired.upper_bound >= expected - 1e-6
    interrupted_after_work = False
    budgets = (2, 8, 32, *range(128, 1025, 16)) if completion_bounds else (2, 8, 32, 128)
    for budget in budgets:
        ticks = iter(range(100, 10000))
        with monkeypatch.context() as clock:
            clock.setattr(time, "perf_counter", lambda: float(next(ticks)))
            result = price_zeta(graph, prices, 0.0, params, deadline=100.0 + budget)
        assert 0 <= result.timed_cache_peak_bytes <= cache_bytes
        assert result.upper_bound >= expected - 1e-6
        assert result.reduced_cost <= expected + 1e-6
        if result.complete:
            assert result.reduced_cost == pytest.approx(expected, abs=1e-6)
        else:
            assert result.reason == "deadline"
            interrupted_after_work |= result.expanded > 0
        if result.column is not None:
            assert result.column in columns
        if interrupted_after_work and result.complete:
            break
    assert interrupted_after_work

    if compiled_loop and buffer == 0.0 and not any((expire, dominance, snap, uncapped, future_bound)):
        import gc
        import weakref

        enabled = gc.isenabled()
        gc.disable()
        try:
            transient = ZetaGraph(fg, cfg, row_index, options)
            reference = weakref.ref(transient)
            price_zeta(transient, prices, 0.0, params)
            del transient
            # Completed calls must release their graph and search arenas between LP rounds.
            assert reference() is None
        finally:
            if enabled:
                gc.enable()


@pytest.mark.parametrize("backend,compiled_loop,compiled_history", [
    ("reference", False, False), ("compiled", False, False),
    ("compiled", True, False), ("compiled", True, True),
    ("compiled-lattice", True, True), ("compiled-indexed", True, True),
    ("compiled-indexed", True, False),
], ids=["reference", "compiled", "compiled-loop", "compiled-history-helper", "lattice-native",
        "native-state-index", "native-index-python-history"])
@pytest.mark.parametrize("snap", [0.0, 1.0], ids=["nominal", "snapped"])
@pytest.mark.parametrize("history_mode", ["exact", "strong", "strong-skips", "merge"])
def test_anyangle_history_pruning_preserves_signed_union_optimum_and_frontier(
    backend: str, compiled_loop: bool, compiled_history: bool, snap: float, history_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discarding distinct paid histories must preserve exact answers and interrupted bounds."""
    native_state_index = backend.endswith("-indexed")
    lattice_distance_bound = backend.endswith("-lattice")
    backend = backend.removesuffix("-lattice").removesuffix("-indexed")
    strong = history_mode != "exact"
    group_skips = history_mode in {"strong-skips", "merge"}
    merge_equivalent = history_mode == "merge"
    if backend == "compiled":
        pytest.importorskip("numba")
    if compiled_loop:
        pytest.importorskip("Cython")
    cfg = _cfg(max_ground_delay_s=0.0, max_detour_factor=4.0,
               corridor_width_m=10.0, hover_radius_m=5.0)
    fg, params = _graph(cfg, dest=(2, 0), overrun=3)
    fg = replace(fg, corridor_cells=frozenset({(0, 0), (0, 1), (1, -1), (1, 0), (2, 0)}),
                 max_air_hops=5)
    row_index = network_mod.RowIndex()
    options = ZetaOptions(connection_radius_m=240.0, backend=backend, lazy_successors=True,
                          arrival_snap_s=snap, compiled_loop=compiled_loop,
                          compiled_history=compiled_history,
                          lattice_distance_bound=lattice_distance_bound,
                          native_state_index=native_state_index)
    columns = _exhaustive_anyangle_columns(fg, cfg, params, row_index, 240.0, options)
    assert any(len(set(c.cell_path)) < len(c.cell_path) for c in columns)
    assert any(hg.hex_distance(a, b) > 1 for c in columns
               for a, b in zip(c.cell_path, c.cell_path[1:]))
    rows = sorted({row for column in columns for row in column.claims})
    rng = np.random.default_rng(3)
    prices = {row: float(rng.choice([-20.0, -2.5, 5.0, 30.0])) for row in rows}
    history_type = _anyangle_history_type(compiled_history)
    large_retired_charges = []
    if compiled_loop:
        # An unreachable tiny price expands the exact common scale without changing any route.
        unused_row = RowKey.cell(10_000, 10_000, 0, 0)
        assert unused_row not in rows
        prices[unused_row] = 1e-200
        accept_with_exact_charge = history_type.accept

        def observe_exact_charge(history, key, paid, retired, identifier, **kwargs):
            """Check that compiled enqueue passes arbitrary-size signed charges unchanged."""
            if abs(retired).bit_length() > 64:
                large_retired_charges.append(retired)
            return accept_with_exact_charge(history, key, paid, retired, identifier, **kwargs)

        monkeypatch.setattr(history_type, "accept", observe_exact_charge)
    expected = max(params.M - c.delay_s - math.fsum(prices[r] for r in c.claims)
                   for c in columns)
    early = price_zeta(
        ZetaGraph(fg, cfg, row_index, replace(options, early_return=True)), prices, 0.0, params,
    )
    assert early.reason == "improving_column" and not early.complete
    assert early.column in columns
    assert params.epsilon < early.reduced_cost < expected - 1e-6
    assert early.upper_bound >= expected - 1e-6
    if backend == "reference":
        _check_anyangle_future_price_bound(
            ZetaGraph(fg, cfg, row_index, options), columns, prices, params,
        )

    # These prefixes share a continuation clock but have different charged histories.
    for join in (2, 4):
        retired, live = [], []
        for middle in ((0, 1), (1, -1)):
            path = ((0, 0), middle, (1, 0), (2, 0), (1, 0), (2, 0))
            column = next(c for c in columns if c.cell_path == path)
            intent = anyangle_to_intent(column, fg.request, cfg, arrival_snap_s=snap)
            prefix = reservation_claims(intent.volumes[:join + 1], cfg, row_index=row_index)
            cutoff = intent.centerline[join][1] - cfg.time_buffer_s
            retired.append(frozenset(r for r in prefix if (r.step + 1) * cfg.dt_s <= cutoff))
            live.append(prefix - retired[-1])
        assert (live[0] == live[1]) is (join == 4)
        assert math.fsum(prices[r] for r in retired[0]) != math.fsum(prices[r] for r in retired[1])

    results = {}
    for expire, dominance in ((False, False), (True, False), (False, True), (True, True)):
        selected = replace(options, expire_history=expire, history_dominance=dominance,
                           strong_history_dominance=strong and dominance,
                           history_group_skips=group_skips and dominance,
                           merge_equivalent_histories=merge_equivalent and dominance)
        result = price_zeta(ZetaGraph(fg, cfg, row_index, selected), prices, 0.0, params)
        allowance = result.merge_error_bound
        assert expected - allowance - 1e-6 <= result.reduced_cost <= expected + 1e-6
        assert expected - 1e-6 <= result.upper_bound <= expected + allowance + 1e-6
        if allowance:
            assert not result.complete or result.upper_bound <= params.epsilon
        else:
            assert result.complete
        assert result.column in columns
        assert result.column.claims == next(c.claims for c in columns if c == result.column)
        assert len(set(result.column.cell_path)) < len(result.column.cell_path)
        if native_state_index:
            control = price_zeta(
                ZetaGraph(fg, cfg, row_index, replace(selected, native_state_index=False)),
                prices, 0.0, params,
            )
            ignored = dict(wall_s=0.0, bound_build_s=0.0, incumbent_setup_s=0.0,
                           topology_setup_s=0.0)
            assert replace(result, **ignored) == replace(control, **ignored)
        if backend == "compiled":
            uncached = replace(selected, compiled_search=False, timed_cache_bytes=0,
                               compiled_loop=False, compiled_history=False,
                               native_state_index=False)
            control = price_zeta(ZetaGraph(fg, cfg, row_index, uncached), prices, 0.0, params)
            assert (result.column, result.reduced_cost, result.upper_bound,
                    result.complete, result.reason) == (
                        control.column, control.reduced_cost, control.upper_bound,
                        control.complete, control.reason,
                    )
            assert result.column.claims == control.column.claims
        if selected.history_group_skips:
            control = price_zeta(
                ZetaGraph(fg, cfg, row_index, replace(selected, history_group_skips=False)),
                prices, 0.0, params,
            )
            ignored = dict(wall_s=0.0, bound_build_s=0.0, incumbent_setup_s=0.0,
                           topology_setup_s=0.0, history_comparisons=0,
                           history_group_reject_skips=0, history_group_replace_skips=0)
            assert replace(result, **ignored) == replace(control, **ignored)
            assert result.column.claims == control.column.claims
        results[expire, dominance] = result
    assert results[True, False].history_expired_rows > 0
    assert results[True, True].history_expired_rows > 0
    assert results[True, True].history_pruned > 0
    assert results[True, True].history_comparisons > 0
    assert results[True, True].generated < results[False, False].generated
    if strong:
        margin = HistoryGuard(ZetaGraph(fg, cfg, row_index, selected),
                              cost_model(cfg, params), prices).margin(None, None)
        assert margin is not None and math.isfinite(margin) and margin > 0.0
    if compiled_loop:
        assert large_retired_charges
        assert any(charge < 0 for charge in large_retired_charges)

    if merge_equivalent:
        # A new dual vector must rebuild equivalence and exact retired charges.
        reused = ZetaGraph(fg, cfg, row_index, selected)
        for current_prices in (prices, {row: -price for row, price in prices.items()}):
            current_optimum = max(
                params.M - column.delay_s
                - math.fsum(current_prices[row] for row in column.claims)
                for column in columns
            )
            current = price_zeta(reused, current_prices, 0.0, params)
            assert current.upper_bound >= current_optimum - 1e-6
            assert current.reduced_cost >= current_optimum - current.merge_error_bound - 1e-6
            assert current.column in columns

        # A valid but deliberately wide rounding envelope must prevent a false proof
        # at zero reduced cost, while still allowing a proof with sufficient slack.
        tie_fg, tie_params = _graph(cfg, dest=(3, 1), overrun=3)
        tie_fg = replace(tie_fg, max_air_hops=6,
                         corridor_cells=frozenset({(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)}))
        tie_columns = _exhaustive_anyangle_columns(tie_fg, cfg, tie_params, row_index, 240.0, selected)
        tie_prices = {row: 100.0 for column in tie_columns for row in column.claims
                      if row.kind == "cell" and row[1:3] == (3, 1)}
        tie_optimum = max(tie_params.M - column.delay_s
                          - math.fsum(tie_prices.get(row, 0.0) for row in column.claims)
                          for column in tie_columns)
        guard = HistoryGuard(ZetaGraph(tie_fg, cfg, row_index, selected),
                             cost_model(cfg, tie_params), tie_prices)
        assert 0.0 < guard.margin(None, None) < 1.0
        with monkeypatch.context() as wide_guard:
            wide_guard.setattr(HistoryGuard, "margin", lambda _self, _oi, _di: 1.0)
            for slack in (0.0, 2.0):
                checked = price_zeta(
                    ZetaGraph(tie_fg, cfg, row_index,
                              replace(selected, early_return=True,
                                      max_expansions=None, max_labels=None)),
                    tie_prices, tie_optimum + slack, tie_params, known_columns=tie_columns,
                )
                assert checked.equivalent_histories_merged > 0
                assert checked.merge_error_bound == 1.0
                assert checked.upper_bound >= -slack - 1e-6
                assert checked.complete is (slack == 2.0)
                if checked.complete:
                    assert checked.upper_bound <= params.epsilon

    real_accept = history_type.accept
    replaced_existing = []

    def observe_replacements(history, *args, **kwargs):
        """Record that bounded searches actually invalidate previously retained work."""
        accepted = real_accept(history, *args, **kwargs)
        if history.dead:
            replaced_existing.append(True)
        return accepted

    monkeypatch.setattr(history_type, "accept", observe_replacements)
    pruned_before_stop = False
    for cap in (16, 32, 64):
        for capped in (replace(selected, max_expansions=cap), replace(selected, max_labels=cap)):
            replaced_existing.clear()
            limited = price_zeta(ZetaGraph(fg, cfg, row_index, capped), prices, 0.0, params)
            assert limited.upper_bound >= expected - 1e-6
            assert limited.reduced_cost <= expected + 1e-6
            if limited.complete:
                assert (limited.merge_error_bound == 0.0
                        or limited.upper_bound <= params.epsilon)
                assert limited.reduced_cost >= expected - limited.merge_error_bound - 1e-6
            elif limited.history_pruned and replaced_existing:
                pruned_before_stop = True
            if limited.column is not None:
                assert limited.column in columns
            if capped.history_group_skips:
                control = price_zeta(
                    ZetaGraph(fg, cfg, row_index, replace(capped, history_group_skips=False)),
                    prices, 0.0, params,
                )
                assert replace(limited, **ignored) == replace(control, **ignored)
    assert pruned_before_stop


@pytest.mark.parametrize("backend,compiled_history", [
    ("reference", False), ("compiled", False), ("compiled", True),
], ids=["reference", "compact", "native-history"])
def test_anyangle_shorter_histories_preserve_full_and_interrupted_pricing(
    backend: str, compiled_history: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Price real unequal-length joins against exhaustive walks across search exits.

    Parameters
    ------------
    - backend (str): reference or compact pricing loop.
    - compiled_history (bool): compile the compact loop and its history helper.
    - monkeypatch (pytest.MonkeyPatch): interrupt the clock after a guarded merge.

    Return
    --------
    - output (None): checks feasible columns and conservative reduced-cost bounds.
    """
    if backend == "compiled":
        pytest.importorskip("numba")
    history_type = _anyangle_history_type(compiled_history)
    cfg = _cfg(max_ground_delay_s=0.0, max_detour_factor=4.0,
               corridor_width_m=10.0, hover_radius_m=5.0)
    fg, params = _graph(cfg, dest=(2, 0), overrun=3)
    fg = replace(fg, corridor_cells=frozenset({(0, 0), (0, 1), (1, 0), (2, 0)}),
                 max_air_hops=5)
    rows = network_mod.RowIndex()
    options = ZetaOptions(connection_radius_m=240.0, backend=backend, lazy_successors=True,
                          history_dominance=True, expire_history=True,
                          strong_history_dominance=True, history_group_skips=True,
                          compiled_loop=compiled_history, compiled_history=compiled_history,
                          arrival_snap_s=8.0, speed_fraction=0.75)
    columns = _exhaustive_anyangle_columns(fg, cfg, params, rows, 240.0, options)
    control_graph = ZetaGraph(fg, cfg, rows, options)
    short, long = [next(c for c in columns if c.cell_path == ((0, 0), middle, (2, 0)))
                   for middle in ((1, 0), (0, 1))]
    assert short.delay_s < long.delay_s
    assert control_graph.intent(short).centerline[-1][1] == control_graph.intent(long).centerline[-1][1]

    # An unreachable credit keeps the detours in the frontier without changing any route price.
    unreachable = RowKey.cell(10_000, 10_000, 0, 0)
    assert all(unreachable not in c.claims for c in columns)
    prices = {unreachable: -100.0}
    options = replace(options, merge_shorter_histories=True)
    graph = ZetaGraph(fg, cfg, rows, options)
    optimum = max(params.M - c.delay_s for c in columns)
    control = price_zeta(control_graph, prices, 0.0, params)
    result = price_zeta(graph, prices, 0.0, params)
    assert control.complete and control.shorter_histories_merged == 0
    assert 0 < result.shorter_histories_merged == result.equivalent_histories_merged
    assert result.merge_error_bound > 0.0
    assert result.reduced_cost == pytest.approx(optimum, abs=1e-6)
    assert result.column in columns
    assert optimum <= result.upper_bound <= optimum + result.merge_error_bound + 1e-6
    if backend == "reference":
        from freespace_sim.planner.colgen import zeta

        negative_air = CostModel(air_weight=-1.0)
        assert HistoryGuard(graph, negative_air, prices).margin(None, None) is not None
        with monkeypatch.context() as reversed_objective:
            reversed_objective.setattr(zeta, "cost_model", lambda _cfg, _params: negative_air)
            negative_result = price_zeta(graph, prices, 0.0, params)
        assert negative_result.shorter_histories_merged == 0
        assert negative_result.merge_error_bound == 0.0
    for capped in (replace(options, max_labels=result.generated - 1),
                   replace(options, max_expansions=result.expanded - 1)):
        limited = price_zeta(ZetaGraph(fg, cfg, rows, capped), prices, 0.0, params)
        assert limited.reason == "search_cap" and not limited.complete
        assert limited.shorter_histories_merged > 0
        assert limited.upper_bound >= optimum - 1e-6
        assert limited.column in columns

    real_accept = history_type.accept
    now = 100.0

    def expire_after_merge(history, *args, **kwargs) -> bool:
        """Advance the mock deadline only after replacing or rejecting a history."""
        nonlocal now
        accepted = real_accept(history, *args, **kwargs)
        if history.shorter_merged:
            now = 102.0
        return accepted

    with monkeypatch.context() as clock:
        clock.setattr(history_type, "accept", expire_after_merge)
        clock.setattr(time, "perf_counter", lambda: now)
        interrupted = price_zeta(graph, prices, 0.0, params, deadline=101.0)
    assert interrupted.reason == "deadline" and not interrupted.complete
    assert interrupted.shorter_histories_merged > 0
    assert interrupted.upper_bound >= optimum - 1e-6
    assert interrupted.column in columns

    route_rows = sorted(set().union(*(c.claims for c in columns)))
    changed = {row: (-20.0, 2.0, 8.0)[idx % 3] for idx, row in enumerate(route_rows)}
    for current_prices in (changed, {row: -value for row, value in changed.items()}, prices):
        current_optimum = max(params.M - c.delay_s
                              - math.fsum(current_prices.get(row, 0.0) for row in c.claims)
                              for c in columns)
        current = price_zeta(graph, current_prices, 0.0, params)
        assert current.column in columns
        assert current.upper_bound >= current_optimum - 1e-6
        assert current.reduced_cost >= current_optimum - current.merge_error_bound - 1e-6
        if current_prices is prices:
            assert _without_anyangle_cache_metrics(current) == _without_anyangle_cache_metrics(result)

    early_graph = ZetaGraph(fg, cfg, rows, replace(options, early_return=True))
    early = price_zeta(early_graph, prices, 0.0, params)
    assert early.reason == "improving_column" and not early.complete
    assert early.column in columns and early.reduced_cost > params.epsilon
    assert early.upper_bound >= optimum - 1e-6
    with monkeypatch.context() as wide_guard:
        wide_guard.setattr(HistoryGuard, "margin", lambda _self, _oi, _di: 1.0)
        for slack in (0.0, 2.0):
            checked = price_zeta(early_graph, prices, optimum + slack, params,
                                 known_columns=columns)
            assert checked.shorter_histories_merged > 0
            assert checked.merge_error_bound == 1.0
            assert checked.upper_bound >= -slack - 1e-6
            assert checked.complete is (slack == 2.0)
            assert checked.reason == ("no_improvement" if checked.complete else "merge_uncertainty")


@pytest.mark.parametrize("backend,compiled_loop", [
    ("reference", False), ("compiled", False), ("compiled", True), ("forward_dp", True),
    ("compiled-indexed", True), ("forward_dp-indexed", True),
], ids=["reference", "compact", "native-history", "forward-dp", "indexed", "forward-indexed"])
def test_anyangle_shorter_histories_preserve_canonical_length_budget(
    backend: str, compiled_loop: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a feasible route when the sequential comparison ties at a graph budget.

    Parameters
    ------------
    - backend (str): reference or compact pricing loop.
    - compiled_loop (bool): compile the compact loop and its history helper.
    - monkeypatch (pytest.MonkeyPatch): arrange rounding at the graph length threshold.

    Return
    --------
    - output (None): verifies that the feasible terminal label survives the join.
    """
    native_state_index = backend.endswith("-indexed")
    backend = backend.removesuffix("-indexed")
    forward_dp = backend == "forward_dp"
    backend = "compiled" if forward_dp else backend
    if backend == "compiled":
        pytest.importorskip("numba")
    if compiled_loop:
        pytest.importorskip("Cython")
    from freespace_sim.planner.colgen.zeta import _Part

    cfg = _cfg(max_ground_delay_s=0.0, max_detour_factor=4.0)
    fg, params = _graph(cfg, dest=(2, 0), overrun=3)
    origin, goal = fg.origin_cell, fg.dest_cell
    over_budget, feasible = (0, 1), (1, -1)
    fg = replace(fg, corridor_cells=frozenset({origin, goal, over_budget, feasible}))
    options = ZetaOptions(connection_radius_m=240.0, backend=backend,
                          history_dominance=True, strong_history_dominance=True,
                          merge_shorter_histories=True, compiled_loop=compiled_loop,
                          compiled_history=compiled_loop, native_state_index=native_state_index,
                          traversal="forward_dp" if forward_dp else "best_first",
                          forward_cruise=forward_dp,
                          connection_mode="hex12" if forward_dp else "full")
    graph = ZetaGraph(fg, cfg, network_mod.RowIndex(), options)
    # Perturb only the reported lengths of two mirrored paths, by less than 3e-13 m.
    # Their same clock and sequential length mask a difference in the edge-by-edge sum.
    tail_lengths = {over_budget: (103.92304845413265, 103.9230484541329),
                    feasible: (103.92304845413264, 103.92304845413284)}
    canonical_limit = 120.0 + sum(tail_lengths[feasible])
    graph.max_length = canonical_limit - 1e-7
    assert graph.max_length + 1e-7 == canonical_limit
    for first, second in tail_lengths.values():
        assert 120.0 + first + second == canonical_limit
    assert 120.0 + sum(tail_lengths[over_budget]) > canonical_limit
    real_parts = graph.parts

    def budget_parts(
        source: tuple[int, int], target: tuple[int, int], first: bool, last: bool,
    ) -> tuple[_Part, ...] | None:
        """Preserve resolved geometry and timing while isolating the two rounding paths."""
        if first and source == origin and target in tail_lengths:
            parts = real_parts(source, target, first, last)
            assert parts and len(parts) in (1, 2)
            lengths = (120.0 / len(parts),) * len(parts)
        elif last and source in tail_lengths and target == goal:
            parts = real_parts(source, target, first, last)
            lengths = tail_lengths[source]
        else:
            return None
        assert parts and len(parts) == len(lengths)
        assert all(abs(part.length - length) < 3e-13 for part, length in zip(parts, lengths))
        return tuple(replace(part, length=length) for part, length in zip(parts, lengths))

    monkeypatch.setattr(graph, "parts", budget_parts)
    monkeypatch.setattr(graph, "seed_columns", lambda *_args: ())
    model = cost_model(cfg, params)
    assert HistoryGuard(graph, model, {}).margin(None, None) is not None
    bad = Column(fg.request.flight_id, fg.base_step, 0, None, None, (origin, over_budget, goal), 0.0)
    good = replace(bad, cell_path=(origin, feasible, goal))
    assert graph.intent(bad).centerline[-1][1] == graph.intent(good).centerline[-1][1]
    assert graph.canonical_column(bad, model) is None
    expected = graph.canonical_column(good, model)
    assert expected is not None
    result = price_zeta(graph, {}, 0.0, params)
    assert result.shorter_histories_merged > 0
    assert result.column == expected
    assert result.reduced_cost == params.M - expected.delay_s <= result.upper_bound


@pytest.mark.parametrize("shift,off_center", [(0, False), (0, True), (1 << 24, True)],
                         ids=["centered", "customer-attachments", "large-coordinates"])
def test_anyangle_strong_history_guard_preserves_ruler_and_budget_boundaries(
    shift: int, off_center: bool,
) -> None:
    """The broader comparison must cover measured ruler error and fall back at tight gates."""
    with pytest.raises(ValueError, match="merge_shorter_histories requires strong_history_dominance"):
        ZetaOptions(merge_shorter_histories=True)
    cfg = _cfg(max_ground_delay_s=0.0, max_detour_factor=100.0,
               corridor_width_m=10.0, hover_radius_m=5.0)
    fg, params = _graph(cfg, origin=(shift, -shift), dest=(shift + 3, 1 - shift), overrun=2)
    if off_center:
        request = replace(fg.request, origin=fg.request.origin + vec(29.0, 11.0),
                          dest=fg.request.dest + vec(-27.0, 18.0))
        fg = build_flight_graph(request, cfg, (), params)
    fg = replace(fg, corridor_cells=frozenset((q + shift, r - shift)
                                             for q, r in {(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)}),
                 max_air_hops=5)
    row_index = network_mod.RowIndex()
    options = ZetaOptions(connection_radius_m=240.0, history_dominance=True,
                          strong_history_dominance=True)
    graph = ZetaGraph(fg, cfg, row_index, options)
    prices = {RowKey.cell(1, -1, 0, 9): -2.0, RowKey.cell(1, 0, 0, 8): 0.5}
    guard = HistoryGuard(graph, cost_model(cfg, params), prices)
    assert guard.margin(None, None) is not None
    from freespace_sim.planner.colgen.zeta_lattice_bound import LatticeDistanceBound

    enabled = replace(options, lattice_distance_bound=True)
    lattice_graph = ZetaGraph(fg, cfg, row_index, enabled)
    lattice = LatticeDistanceBound.build(lattice_graph, cost_model(cfg, params), prices)
    assert lattice is not None
    assert lattice.remaining(fg.origin_cell, fg.dest_cell) > math.dist(
        lattice_graph.point(fg.origin_cell), lattice_graph.point(fg.dest_cell)) + 1.0
    if off_center:
        assert lattice_graph.fold_length(None, None) > 50.0
    columns = _exhaustive_anyangle_columns(fg, cfg, params, row_index, 240.0, options)
    assert columns
    _check_anyangle_completion_bounds(lattice_graph, columns, prices, params)
    for column in columns:
        intent = graph.intent(column)
        flown = enroute_flown_m([point for point, _stamp in intent.centerline],
                               fg.request.origin, fg.request.dest, None, None, cfg)
        accumulated = 0.0
        for index, (source, target) in enumerate(zip(column.cell_path, column.cell_path[1:])):
            for part in graph.parts(source, target, index == 0,
                                    index == len(column.cell_path) - 2):
                accumulated += part.length
        assert len(intent.centerline) - 1 <= guard.max_parts
        assert abs(flown - (accumulated + graph.fold_length(None, None))) <= guard.length_error

    tight_cfg = replace(cfg, max_detour_factor=1.0)
    tight_fg = replace(fg, _cfg=tight_cfg)
    tight_graph = ZetaGraph(tight_fg, tight_cfg, row_index, options)
    assert HistoryGuard(tight_graph, cost_model(tight_cfg, params), prices).margin(None, None) is None
    assert LatticeDistanceBound.build(tight_graph, cost_model(tight_cfg, params), prices) is None
    control = price_zeta(ZetaGraph(tight_fg, tight_cfg, row_index,
                                   replace(options, strong_history_dominance=False)),
                         prices, 0.0, params)
    candidate = price_zeta(tight_graph, prices, 0.0, params)
    assert candidate.complete and candidate.strong_history_pruned == 0
    assert _without_anyangle_cache_metrics(candidate) == _without_anyangle_cache_metrics(control)
    merged = price_zeta(
        ZetaGraph(tight_fg, tight_cfg, row_index,
                  replace(options, merge_equivalent_histories=True, merge_shorter_histories=True)),
        prices, 0.0, params,
    )
    assert merged.equivalent_histories_merged == merged.shorter_histories_merged == 0
    assert merged.merge_error_bound == 0.0
    assert _without_anyangle_cache_metrics(merged) == _without_anyangle_cache_metrics(control)

    # A boundary lane can change the final fold after endpoint interpolation rounds inward.
    terminal = Terminal("boundary-destination", capacity=1,
                        radius=cfg.corridor_segment_len_m - cfg.corridor_width_m / 2.0)
    request = FlightRequest(2, _point((0, 0), cfg), _point((6, 0), cfg), 0.0, 0.0,
                            dest_terminal=terminal)
    terminal_fg = build_flight_graph(request, cfg, [(request.dest, terminal)], params)
    terminal_rows = network_mod.RowIndex()
    terminal_rows.register_terminal(terminal)
    terminal_graph = ZetaGraph(terminal_fg, cfg, terminal_rows, options)
    lane_index = next(index for index, lane in enumerate(terminal_fg.dest_lanes)
                      if lane.cell == (5, 0))
    lane_point = terminal_graph.point((5, 0))
    distance = math.hypot(lane_point[0] - request.dest[0], lane_point[1] - request.dest[1])
    assert abs(distance - exit_radius(terminal, cfg)) < 1e-9
    assert HistoryGuard(terminal_graph, cost_model(cfg, params), {}).margin(None, lane_index) is None
    assert LatticeDistanceBound.build(terminal_graph, cost_model(cfg, params), {}) is None
    for unsupported in (replace(options, connection_radius_m=480.0),
                        replace(options, connection_mode="hex12_long")):
        control = price_zeta(ZetaGraph(fg, cfg, row_index, unsupported), prices, 0.0, params)
        candidate = price_zeta(ZetaGraph(fg, cfg, row_index,
                                       replace(unsupported, lattice_distance_bound=True)),
                               prices, 0.0, params)
        assert candidate.lattice_bound_calls == 0
        assert _without_anyangle_cache_metrics(candidate) == _without_anyangle_cache_metrics(control)


@pytest.mark.parametrize("dt,snap,off_center", [
    (4.0, 0.0, False), (4.0, 1.0, False), (0.7, 1.0, False),
    (4.0, 0.0, True), (0.7, 1.0, True),
], ids=["nominal", "snapped", "nonbinary-fallback", "attachments-nominal",
        "attachments-snapped"])
@pytest.mark.parametrize("backend", ["reference", "compiled"])
@pytest.mark.parametrize("completion_bounds", [(), ("arrival",), ("resource",),
                                               ("selected",), ("crossing",),
                                               ("arrival", "resource", "selected", "crossing")])
def test_anyangle_future_price_bound_is_positive_and_respects_physical_suffix_unions(
    dt, snap, off_center, backend, completion_bounds,
):
    """Mandatory future reservations yield a positive bound without double-charging paid rows."""
    cfg = _cfg(dt_s=dt, time_buffer_s=math.nextafter(dt, math.inf),
               corridor_width_m=5.0, hover_radius_m=2.0,
               max_ground_delay_s=0.0, max_detour_factor=4.0,
               cost_ground_delay_per_s=1.3, cost_air_lateral_per_s=2.7, cost_air_hold_per_s=2.7)
    fg, params = _graph(cfg, dest=(3, 0), overrun=2)
    if off_center:
        pitch = cfg.corridor_segment_len_m
        request = replace(fg.request, origin=fg.request.origin + vec(0.29 * pitch, 0.11 * pitch),
                          dest=fg.request.dest + vec(-0.27 * pitch, 0.18 * pitch))
        fg = build_flight_graph(request, cfg, (), params)
    fg = replace(fg, corridor_cells=frozenset((q, 0) for q in range(4)), max_air_hops=5)
    row_index = network_mod.RowIndex()
    options = ZetaOptions(connection_radius_m=3 * cfg.corridor_segment_len_m,
                          arrival_snap_s=snap, completion_bounds=completion_bounds)
    graph = ZetaGraph(fg, cfg, row_index, options)
    if off_center:
        assert graph.fold_length(None, None) > 0.5 * cfg.corridor_segment_len_m
    columns = _exhaustive_anyangle_columns(
        fg, cfg, params, row_index, options.connection_radius_m, options,
    )
    rows = sorted({row for column in columns for row in column.claims})
    prices = {row: 1.0 + index % 5 for index, row in enumerate(rows)
              if row.cell_coord != fg.origin_cell}
    if completion_bounds:
        gains = _check_anyangle_completion_bounds(graph, columns, prices, params)
        assert all(value > 1e-6 for value in gains.values())
    else:
        assert _check_anyangle_future_price_bound(graph, columns, prices, params) > 0.0
    if not completion_bounds and dt == 4.0 and snap == 0.0 and not off_center:
        # One long connection forces the relaxation to retain its buffered subpart coverage.
        direct_fg = replace(fg, corridor_cells=frozenset({fg.origin_cell, fg.dest_cell}))
        direct_graph = ZetaGraph(direct_fg, cfg, row_index, options)
        direct_columns = _exhaustive_anyangle_columns(
            direct_fg, cfg, params, row_index, options.connection_radius_m, options,
        )
        assert _check_anyangle_future_price_bound(
            direct_graph, direct_columns, prices, params, require_buffer_gain=True,
        ) > 0.0
    expected = max(params.M - column.delay_s
                   - math.fsum(prices.get(row, 0.0) for row in column.claims)
                   for column in columns)
    enabled = replace(options, backend=backend, lazy_successors=True,
                      future_price_bound=not completion_bounds,
                      expire_history=True, history_dominance=True)
    for capped in (enabled, replace(enabled, max_expansions=1), replace(enabled, max_labels=2)):
        result = price_zeta(ZetaGraph(fg, cfg, row_index, capped), prices, 0.0, params)
        assert result.bound_table_states > 0
        assert result.bound_lookups > 0
        assert result.upper_bound >= expected - 1e-6
        assert result.reduced_cost <= expected + 1e-6
        if result.complete:
            assert result.reduced_cost == pytest.approx(expected, abs=1e-6)
            assert result.upper_bound == pytest.approx(expected, abs=1e-6)
        if capped is enabled:
            assert result.complete
        if result.column is not None:
            assert result.column in columns


@pytest.mark.parametrize("retired_parts", [
    ("retired_large", "retired_unit", "retired_credit"),
    ("retired_large", "retired_tiny", "retired_credit"),
], ids=["cancellation", "separated-exponents"])
@pytest.mark.parametrize("margin", [None, 0.0, 0.1, 1.0, math.nextafter(1.0, math.inf), 1e-200],
                         ids=["exact-key", "strict-zero", "fraction", "equality", "above", "tiny"])
@pytest.mark.parametrize("compiled_history", [False, True], ids=["python-helper", "compiled-helper"])
@pytest.mark.parametrize("group_skips", [False, True], ids=["all-groups", "skip-groups"])
@pytest.mark.parametrize("merge_equivalent", [False, True], ids=["retain-ties", "merge-ties"])
def test_anyangle_signed_history_dominance_preserves_every_future_union(
    retired_parts, margin, compiled_history, group_skips, merge_equivalent,
):
    """Exact comparisons must retain a cheaper future even when float addition loses its cost."""
    prices = {0: 1e16, 1: -1e16, 2: 0.1,
              "retired_large": 1e16, "retired_credit": -1e16,
              "retired_unit": 1.0, "retired_tiny": 1e-200}
    exact = {row: Fraction.from_float(price) for row, price in prices.items()}
    retired_cost = sum(exact[row] for row in retired_parts)
    history_type = _anyangle_history_type(compiled_history)
    tails = [frozenset(row for row in range(3) if mask & (1 << row)) for mask in range(8)]
    for first in tails:
        for second in tails:
            first_costs = [retired_cost + sum(exact[row] for row in first | future)
                           for future in tails]
            second_costs = [sum(exact[row] for row in second | future) for future in tails]
            if margin is None:
                first_dominates = all(a <= b for a, b in zip(first_costs, second_costs))
                second_dominates = all(b <= a for a, b in zip(first_costs, second_costs))
                keys, kwargs = ((), ()), {}
            else:
                allowance = Fraction.from_float(margin)
                first_dominates = all(a + allowance < b for a, b in zip(first_costs, second_costs))
                second_dominates = all(b + allowance < a for a, b in zip(first_costs, second_costs))
                keys, kwargs = (("first-sequence",), ("second-sequence",)), {"broad_key": (), "margin": margin}
            history = history_type(prices, group_skips=group_skips,
                                   merge_equivalent=merge_equivalent)
            assert history.accept(keys[0], first, history.charge(retired_parts), 0, **kwargs)
            accepted = history.accept(keys[1], second, 0, 1, **kwargs)
            equivalent = (merge_equivalent and margin is not None
                          and first == second and retired_cost == 0)
            assert accepted is not (first_dominates or equivalent)
            if accepted:
                assert (0 in history.dead) is second_dominates

    if margin == 0.1:
        # Merged or separately retained ties must keep a representative until replacement.
        history = history_type({"charge": 1.0}, group_skips=group_skips,
                               merge_equivalent=merge_equivalent)
        kwargs = {"broad_key": (), "margin": margin}
        assert history.accept((0,), (), history.charge(("charge",)), 0, **kwargs)
        assert history.accept((1,), (), history.charge(("charge",)), 1, **kwargs) is not merge_equivalent
        assert history.merged == int(merge_equivalent)
        assert history.merge_error_bound == (margin if merge_equivalent else 0.0)
        assert history.accept((2,), (), 0, 2, **kwargs)
        assert ({0} if merge_equivalent else {0, 1}).issubset(history.dead)
        assert not history.accept((3,), (), history.charge(("charge",)), 3, **kwargs)
        assert history.accept((4,), (), -history.charge(("charge",)), 3, **kwargs)
        assert 2 in history.dead
        if group_skips:
            assert history.group_reject_skips > 0
            if not merge_equivalent:
                assert history.group_replace_skips > 0

        if merge_equivalent:
            # Different live reservations can reverse their ranking after a suffix.
            distinct = history_type({"X": 5.0, "Y": 4.0}, merge_equivalent=True)
            assert distinct.accept((0,), ("X",), 0, 0, **kwargs)
            assert distinct.accept((1,), ("Y",), 0, 1, **kwargs)
            assert distinct.merged == 0 and distinct.merge_error_bound == 0.0
            assert distinct.accept((2,), ("X",), 0, 2,
                                   broad_key=("other-context",), margin=margin)
            assert distinct.merged == 0
            assert not distinct.accept((3,), ("X",), 0, 3, **kwargs)
            assert distinct.merged == 1 and distinct.merge_error_bound == margin

            # No finite numerical guard means retaining sequence-sensitive comparisons.
            unguarded = history_type({"X": 5.0}, merge_equivalent=True)
            assert unguarded.accept((0,), ("X",), 0, 0)
            assert unguarded.accept((1,), ("X",), 0, 1)
            assert unguarded.merged == 0 and unguarded.merge_error_bound == 0.0
            assert not unguarded.accept((1,), ("X",), 0, 2)
            assert unguarded.merged == 0 and unguarded.merge_error_bound == 0.0

            # Numerically equal float totals are not equal exact retired charges.
            exact_charges = history_type({"large": 1e16, "unit": 1.0}, merge_equivalent=True)
            first = exact_charges.charge(("large",))
            second = exact_charges.charge(("large", "unit"))
            assert first != second and math.fsum((1e16, 1.0)) == 1e16
            assert exact_charges.accept((0,), (), first, 0, broad_key=(), margin=2.0)
            assert exact_charges.accept((1,), (), second, 1, broad_key=(), margin=2.0)
            assert exact_charges.merged == 0 and exact_charges.merge_error_bound == 0.0

    if margin == 1.0:
        # A live-set bucket may contain close charges that must remain queued separately.
        history = history_type({"quarter": 0.25}, group_skips=group_skips,
                               merge_equivalent=merge_equivalent)
        unit = history.charge(("quarter",))
        kwargs = {"broad_key": (), "margin": margin}
        for identifier in range(3):
            assert history.accept((identifier,), (), identifier * unit, identifier, **kwargs)
        assert history.accept((3,), (), -3 * unit, 3, **kwargs)
        assert history.dead == {2}
        assert history.accept((4,), (), -5 * unit, 4, **kwargs)
        assert history.dead == {0, 1, 2}

        # Interleave live sets and reused exact keys so removing an extreme bucket
        # cannot change subsequent rejection or replacement decisions.
        varied = {"positive": 2.0, "negative": -3.0, "tiny": 1e-200, "quarter": 0.25}
        baseline = history_type(varied, merge_equivalent=merge_equivalent)
        candidate = history_type(varied, group_skips=group_skips,
                                 merge_equivalent=merge_equivalent)
        unit = baseline.charge(("quarter",))
        rng = np.random.default_rng(11)
        next_identifier = 0
        for step in range(120):
            tail = tuple(row for row in varied if rng.integers(2))
            charge = int(rng.integers(-20, 21)) * unit
            key = (step % 7,)
            kwargs = {"broad_key": (), "margin": margin}
            accepted = baseline.accept(key, tail, charge, next_identifier, **kwargs)
            assert candidate.accept(key, tail, charge, next_identifier, **kwargs) is accepted
            assert candidate.dead == baseline.dead
            assert candidate.pruned == baseline.pruned
            assert candidate.strong_pruned == baseline.strong_pruned
            assert candidate.merged == baseline.merged
            assert candidate.merge_error_bound == baseline.merge_error_bound
            next_identifier += accepted


@pytest.mark.parametrize("compiled_history", [False, True], ids=["python-helper", "compiled-helper"])
def test_anyangle_shorter_histories_preserve_distinct_future_states(compiled_history: bool) -> None:
    """Keep every incomparable length pair and carry one allowance through replacements."""
    history_type = _anyangle_history_type(compiled_history)
    prices = {"X": 5.0, "Y": -2.0, "large": 1e16, "unit": 1.0, "tiny": 1e-200}
    margin = 2.0
    for lengths in ((9.0, 10.0), (10.0, 11.0)):
        history = history_type(prices, merge_shorter=True)
        other = (10.0, 11.0) if lengths == (9.0, 10.0) else (9.0, 10.0)
        assert history.accept((*lengths, 0), ("X",), 0, 0, broad_key=lengths,
                              margin=margin, length_key=(), lengths=lengths)
        assert history.accept((*other, 1), ("X",), 0, 1, broad_key=other,
                              margin=margin, length_key=(), lengths=other) is (other < lengths)
        assert history.dead == ({0} if other < lengths else set())
        assert history.merged == history.shorter_merged == 1
        assert history.merge_error_bound == margin

    history = history_type(prices, merge_equivalent=True, merge_shorter=True, group_skips=True)
    retired = history.charge(("large",))

    def offer(
        identifier: int, lengths: tuple[float, float], *, paid: tuple[str, ...] = ("X",),
        charge: int = retired, context: str = "join", guarded: bool = True,
    ) -> bool:
        """Offer a prefix with distinct segment order to the shared continuation."""
        broad_key = (context, *lengths) if guarded else None
        return history.accept((context, *lengths, identifier), paid, charge, identifier,
                              broad_key=broad_key, margin=margin, length_key=(context,),
                              lengths=lengths)

    assert offer(0, (9.0, 11.0)) and offer(1, (10.0, 10.0))
    rounded_tie = history.charge(("large", "unit"))
    assert retired != rounded_tie and math.fsum((1e16, 1.0)) == 1e16
    assert offer(2, (8.0, 9.0), charge=rounded_tie)
    assert offer(3, (8.0, 9.0), paid=("Y",))
    assert offer(4, (8.0, 9.0), context="other-join")
    assert offer(5, (7.0, 8.0), guarded=False)
    assert not history.dead and history.merged == 0
    assert offer(6, (8.0, 9.0))
    assert history.dead == {0, 1}
    assert not offer(7, (8.0, 9.0))
    cheaper = retired - 4 * history.charge(("unit",))
    assert offer(7, (8.0, 9.0), charge=cheaper)
    assert history.dead == {0, 1, 2, 6}
    assert offer(8, (7.0, 8.0), charge=cheaper)
    assert history.dead == {0, 1, 2, 6, 7}
    assert history.strong_pruned == 2
    assert history.merged == 4 and history.shorter_merged == 3
    assert history.merge_error_bound == margin


def test_anyangle_numeric_state_groups_preserve_exact_continuations() -> None:
    """Native group IDs must preserve every key field, signed zero, and identity on growth."""
    with pytest.raises(ValueError, match="native_state_index requires compiled_loop"):
        ZetaOptions(native_state_index=True)
    pytest.importorskip("Cython")
    import sys

    from freespace_sim.planner.colgen.zeta_native import prepare_compiled_loop

    name, _binary = prepare_compiled_loop()
    index = sys.modules[name]._StateIndex()
    expected = ({}, {}, {})
    rng = np.random.default_rng(78)
    states = []
    for i in range(3000):
        stamp = float(rng.integers(8)) if i % 3 else math.nextafter(4.0, math.inf)
        state = (int(rng.integers(20)), int(rng.integers(4)), stamp,
                 float(rng.integers(12)), float(rng.integers(12)), bool(i % 2),
                 bool(i % 3), int(rng.integers(8)))
        states.append(state)
    states.extend([(0, 0, zero, zero, zero, False, False, 0)
                   for zero in (0.0, -0.0)])
    for state in states + states[::-1]:
        keys = (state, state[:-1], state[:3] + state[5:7])
        actual = index.keys(*state, True, True)
        for mapping, key, identifier in zip(expected, keys, actual):
            if key in mapping:
                assert identifier == mapping[key]
            else:
                assert identifier not in mapping.values()
                mapping[key] = identifier
    assert index.keys(*states[0], False, False)[1:] == (None, None)
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            index.keys(0, 0, invalid, 1.0, 1.0, False, False, 0, True, True)

    # Repeated offers reuse paid IDs, but a new signed price vector owns fresh partitions.
    history_type = sys.modules[name].PriceHistory
    for sign in (1.0, -1.0):
        prices = {0: sign*1e16, 1: -sign*1e16, 2: sign*0.1, 3: sign*1e-200}
        control = history_type(prices, group_skips=True, merge_equivalent=True, merge_shorter=True)
        indexed = history_type(prices, group_skips=True, merge_equivalent=True, merge_shorter=True)
        paid_ids = {}
        identifier = 0
        for step in range(500):
            state = (step % 4, step % 2, float(step % 3), 10.0+step % 2,
                     11.0-step % 2, bool(step % 2), False, step % 7)
            exact_key, broad_key, length_key = index.keys(*state, True, True)
            paid = tuple(row for row in prices if rng.integers(2))
            paid_id = paid_ids.setdefault(paid, len(paid_ids))
            retired = int(rng.integers(-4, 5)) * control.weights[2]
            kwargs = dict(margin=0.25, lengths=state[3:5])
            accepted = control.accept(state, paid, retired, identifier, broad_key=state[:-1],
                                      length_key=state[:3]+state[5:7], **kwargs)
            assert indexed.accept(exact_key, np.array(paid, dtype=np.int64), retired, identifier,
                                  broad_key=broad_key, length_key=length_key,
                                  paid_id=paid_id, **kwargs) is accepted
            for field in ("dead", "pruned", "comparisons", "strong_pruned", "merged",
                          "shorter_merged", "merge_error_bound", "group_reject_skips",
                          "group_replace_skips"):
                assert getattr(indexed, field) == getattr(control, field)
            identifier += accepted


@pytest.mark.parametrize("max_routes", [0, 1, 4], ids=["disabled", "eviction", "reuse"])
@pytest.mark.parametrize("compiled_route_coverage", [False, True], ids=["python-cache", "compiled-cache"])
def test_anyangle_route_coverage_cache_reprices_complete_unions(
    max_routes: int, compiled_route_coverage: bool,
) -> None:
    """Cached coverage must preserve union prices across clock, altitude and dual changes."""
    from freespace_sim.planner.colgen.zeta_route_cache import RouteCoverageCache

    cache_type = RouteCoverageCache
    if compiled_route_coverage:
        pytest.importorskip("Cython")
        import sys

        from freespace_sim.planner.colgen.zeta_native import prepare_compiled_loop

        name, _binary = prepare_compiled_loop()
        cache_type = sys.modules[name].RouteCoverageCache
        if max_routes == 0:
            with pytest.raises(ValueError, match="compiled_route_coverage"):
                ZetaOptions(backend="compiled", route_coverage_cache=True,
                            compiled_route_coverage=True)
            with pytest.raises(ValueError, match="compiled_route_coverage"):
                ZetaOptions(backend="compiled", compiled_loop=True,
                            compiled_route_coverage=True)

    first = RowKey.cell(0, 0, 0, 1)
    later = RowKey.cell(0, 0, 0, 2)
    higher = RowKey.cell(0, 0, 1, 1)
    terminal = RowKey.term("hub", 1)
    formerly_free = RowKey.cell(1, 0, 0, 2)
    unrelated = RowKey.term("other", 1)
    leg_one = frozenset((first, later, terminal))
    leg_two = frozenset((later, higher, terminal, formerly_free))
    claims = leg_one | leg_two
    prices = {first: 1e16, later: -1e16, higher: 0.25, terminal: 0.1,
              formerly_free: 0.0, unrelated: 123.0}
    expected = math.fsum(prices.get(row, 0.0) for row in claims)
    cache = cache_type(max_routes=max_routes)
    original = cache.bind(prices)
    assert original.price(claims) == expected
    assert original.price(claims) == expected
    assert (original.hits > 0) is bool(max_routes)
    assert expected != math.fsum(prices.get(row, 0.0) for row in leg_one) + math.fsum(
        prices.get(row, 0.0) for row in leg_two
    )

    changed_prices = {row: -2.0 * price for row, price in prices.items()}
    changed = cache.bind(changed_prices)
    assert changed.price(claims) == math.fsum(changed_prices.get(row, 0.0) for row in claims)
    assert (changed.hits > 0) is bool(max_routes)
    assert original.price(claims) == expected

    changed_support = {first: -3.0, higher: 0.0, terminal: 1.25, formerly_free: 7.0}
    new_rows = cache.bind(changed_support)
    assert new_rows.price(claims) == math.fsum(changed_support.get(row, 0.0) for row in claims)
    assert (new_rows.hits > 0) is bool(max_routes)
    assert original.price(claims) == expected
    assert changed.price(claims) == math.fsum(changed_prices.get(row, 0.0) for row in claims)
    assert cache.bind({}).price(claims) == 0.0

    fresh = cache_type(max_routes=max_routes).bind(prices)
    other_claims = frozenset((first, unrelated))
    assert fresh.price(claims) == expected
    assert fresh.price(other_claims) == math.fsum(prices[row] for row in other_claims)
    misses = fresh.misses
    assert fresh.price(claims) == expected
    assert (fresh.misses > misses) is (max_routes < 2)

    # Price insertion order differs from claims order. Preserve fsum's finite-input
    # overflow behavior as well as its ordinary rounded result on a cached route.
    ordered_claims = tuple(claims)
    for values in (
        (1e308, -1e308, -1e308, 1e308, 1e308),
        (1e308, 1e308, -1e308, -1e308, 1e308),
        (1e308, -1e308, 0.0, 1e-200, 0.0),
    ):
        extreme_prices = dict(zip(reversed(ordered_claims), reversed(values)))
        view = cache.bind(extreme_prices)
        try:
            expected_extreme = math.fsum(extreme_prices.get(row, 0.0) for row in claims)
        except OverflowError:
            with pytest.raises(OverflowError):
                view.price(claims)
        else:
            assert view.price(claims) == expected_extreme


@pytest.mark.parametrize("native", [False, True, "indexed"])
@pytest.mark.parametrize("snap", [0.0, 1.0])
def test_forward_dp_matches_exhaustive_union_prices_and_interrupted_bounds(native, snap, monkeypatch):
    """A waypoint sweep retains asynchronous histories, signed unions and all pending bounds."""
    cfg = _cfg(max_ground_delay_s=4.0, max_detour_factor=2.0)
    fg, params = _graph(cfg, dest=(3, -1), overrun=1)
    rows = network_mod.RowIndex()
    options = ZetaOptions(
        backend="compiled", compiled_loop=bool(native), traversal="forward_dp",
        connection_mode="hex12", forward_cruise=True, arrival_snap_s=snap,
        expire_history=True, history_dominance=True, strong_history_dominance=True,
        compiled_history=bool(native), native_state_index=native == "indexed",
        merge_equivalent_histories=True, merge_shorter_histories=True,
        lattice_distance_bound=True, max_expansions=None, max_labels=None,
        leg_cache=True, persistent_timing_cache=True,
    )
    columns = _exhaustive_anyangle_columns(fg, cfg, params, rows, 480.0, options)
    assert len(columns) > 2
    all_rows = sorted({row for column in columns for row in column.claims})
    prices = {row: (-2.0 if i % 3 else 8.0) for i, row in enumerate(all_rows)}
    graph = ZetaGraph(fg, cfg, rows, options)
    for current, forbidden in ((prices, frozenset()),
                               ({r: -p for r, p in prices.items()}, frozenset()),
                               (prices, frozenset({all_rows[len(all_rows)//2]}))):
        legal = [c for c in columns if c.claims.isdisjoint(forbidden)]
        optimum = max((params.M - c.delay_s - math.fsum(current.get(r, 0.0) for r in c.claims)
                       for c in legal), default=-math.inf)
        result = price_zeta(graph, current, 0.0, params, forbidden_rows=forbidden)
        control = price_zeta(ZetaGraph(fg, cfg, rows, replace(options, traversal="best_first")),
                             current, 0.0, params, forbidden_rows=forbidden)
        assert result.reduced_cost == pytest.approx(optimum, abs=1e-6)
        assert control.reduced_cost == pytest.approx(optimum, abs=1e-6)
        assert optimum - 1e-6 <= result.upper_bound <= optimum + 1e-5
        assert result.column is None or result.column in legal
        for cap in (1, 3, 12):
            capped = ZetaGraph(fg, cfg, rows, replace(options, max_expansions=cap))
            limited = price_zeta(capped, current, 0.0, params, forbidden_rows=forbidden)
            assert limited.upper_bound >= optimum - 1e-6
            assert limited.expanded <= cap
            assert limited.complete or limited.reason == "search_cap"
            capped = ZetaGraph(fg, cfg, rows, replace(options, max_labels=cap))
            limited = price_zeta(capped, current, 0.0, params, forbidden_rows=forbidden)
            assert limited.upper_bound >= optimum - 1e-6
            assert limited.generated <= cap
    early = ZetaGraph(fg, cfg, rows, replace(options, early_return=True))
    found = price_zeta(early, {}, 0.0, params)
    assert found.column is not None and found.reduced_cost > params.epsilon
    assert found.reason == "improving_column" and not found.complete
    discovered = ZetaGraph(fg, cfg, rows, replace(options, early_return=True))
    monkeypatch.setattr(discovered, "seed_columns", lambda *_args: ())
    import heapq
    with monkeypatch.context() as no_queue:
        def reject_heap(*_args):
            """The primary forward sweep must never use a priority queue."""
            raise AssertionError("priority queue used by forward DP")
        no_queue.setattr(heapq, "heappush", reject_heap)
        searched = price_zeta(discovered, {}, 0.0, params)
    assert searched.expanded > 0 and searched.reason == "improving_column"
    assert searched.reduced_cost > params.epsilon and searched.column in columns
    proof = price_zeta(early, {}, params.M + 1, params, known_columns=columns)
    assert proof.complete and proof.upper_bound <= params.epsilon
    duplicate = price_zeta(early, prices, 0.0, params, known_columns=columns)
    assert duplicate.reason != "improving_column"
    expired = price_zeta(early, prices, 0.0, params, deadline=time.perf_counter()-1)
    assert not expired.complete and expired.reason == "deadline"
    interrupted_graph = ZetaGraph(fg, cfg, rows, options)
    monkeypatch.setattr(interrupted_graph, "seed_columns", lambda *_args: ())
    real_parts = interrupted_graph.parts
    clock = [0.0]
    def expire_after_edge(*args):
        """Expire the call while a parent still has ungenerated successors."""
        result = real_parts(*args)
        clock[0] = 2.0
        return result
    monkeypatch.setattr(interrupted_graph, "parts", expire_after_edge)
    with monkeypatch.context() as during_batch:
        during_batch.setattr(time, "perf_counter", lambda: clock[0])
        interrupted = price_zeta(interrupted_graph, prices, 0.0, params, deadline=1.0)
    optimum = max(params.M - c.delay_s - math.fsum(prices.get(r, 0.0) for r in c.claims)
                  for c in columns)
    assert interrupted.reason == "deadline" and not interrupted.complete
    assert interrupted.expanded > 0 and interrupted.upper_bound >= optimum - 1e-6


@pytest.mark.parametrize("backend,compiled_loop", [("reference", False), ("compiled", False),
                                                   ("compiled", True), ("compiled-indexed", True)])
@pytest.mark.parametrize("merge_equivalent", [False, True])
@pytest.mark.parametrize("forward_cruise", [False, True])
def test_anyangle_hybrid_preserves_full_domain_and_combined_budget(
    backend, compiled_loop, merge_equivalent, forward_cruise,
):
    """Restricted discovery cannot hide omitted improvements, duplicates, credits or caps."""
    native_state_index = backend.endswith("-indexed")
    backend = backend.removesuffix("-indexed")
    if backend == "compiled":
        pytest.importorskip("numba")
    if compiled_loop:
        pytest.importorskip("Cython")
    cfg = _cfg(max_ground_delay_s=4.0, max_detour_factor=2.0)
    fg, params = _graph(cfg, dest=(2, 0), overrun=0)
    fg = replace(fg, corridor_cells=frozenset({(0, 0), (2, 0)}), max_air_hops=2)
    rows = network_mod.RowIndex()
    options = ZetaOptions(
        backend=backend, compiled_loop=compiled_loop, native_state_index=native_state_index,
        lazy_successors=True,
        early_return=True, expire_history=True, history_dominance=True,
        strong_history_dominance=True, compiled_history=compiled_loop,
        history_group_skips=True, route_coverage_cache=True,
        compiled_route_coverage=compiled_loop, arrival_snap_s=1.0,
        cheap_connection_mode="hex12_long", max_expansions=100, max_labels=1000,
        cheap_max_expansions=20, cheap_max_labels=100,
        merge_equivalent_histories=merge_equivalent,
        forward_cruise=forward_cruise,
    )
    graph = ZetaGraph(fg, cfg, rows, options)
    cheap_options = replace(options, cheap_connection_mode=None, connection_mode="hex12_long")
    cheap = price_zeta(ZetaGraph(fg, cfg, rows, cheap_options), {}, 0.0, params)
    assert cheap.complete and cheap.column is None
    found = price_zeta(graph, {}, 0.0, params)
    assert found.full_calls == 1 and found.column is not None
    assert found.reduced_cost > params.epsilon
    assert found.expanded <= options.max_expansions and found.generated <= options.max_labels
    assert graph.options == options
    all_columns = _exhaustive_anyangle_columns(fg, cfg, params, rows, 480.0, options)
    prices = {row: (-0.25 if index % 2 else 2.0)
              for index, row in enumerate(sorted(found.column.claims))}
    for current in (prices, {row: -value for row, value in prices.items()}):
        costs = [column.delay_s + math.fsum(current.get(row, 0.0) for row in column.claims)
                 for column in all_columns]
        pi_f = params.M - min(costs) + 1.0
        proof = price_zeta(graph, current, pi_f, params, known_columns=all_columns)
        assert proof.full_calls == 1 and proof.complete
        assert proof.upper_bound <= params.epsilon
        assert proof.reduced_cost == pytest.approx(params.M - pi_f - min(costs))
    # An adjacent path is available to both stages, so the first column may defer full pricing.
    connected = replace(fg, corridor_cells=frozenset({(0, 0), (1, 0), (2, 0)}))
    shared = ZetaGraph(connected, cfg, rows, options)
    discovery = price_zeta(shared, {}, 0.0, params)
    assert discovery.cheap_columns_returned == 1 and discovery.full_calls == 0
    assert not discovery.complete and discovery.upper_bound == math.inf
    raw_duplicate = replace(discovery.column, delay_s=-100.0, claims=frozenset())
    duplicate = price_zeta(shared, {}, 0.0, params, known_columns=(raw_duplicate,) * 2)
    assert duplicate.full_calls == 1 and duplicate.cheap_columns_returned == 0
    assert duplicate.column == discovery.column
    proof = price_zeta(shared, {}, params.M + 1.0, params,
                       known_columns=(discovery.column,))
    assert proof.complete and proof.full_calls == 1 and proof.upper_bound <= params.epsilon
    expired = price_zeta(shared, prices, 0.0, params, deadline=time.perf_counter() - 1.0)
    assert not expired.complete and expired.reason == "deadline"
    assert expired.expanded == expired.generated == 0
    capped_options = replace(options, max_expansions=1, max_labels=1)
    limited = price_zeta(ZetaGraph(fg, cfg, rows, capped_options), prices, 0.0, params)
    assert not limited.complete and limited.full_calls == 0
    assert limited.upper_bound == math.inf
    assert limited.expanded <= 1 and limited.generated <= 1
    threshold_graph = ZetaGraph(replace(connected, latest_departure_step=fg.base_step),
                                cfg, rows, options)
    required_row = min(threshold_graph.rows(threshold_graph.endpoint(True, fg.base_step * cfg.dt_s)))
    threshold_prices = {required_row: 9899.699999}
    borderline = price_zeta(threshold_graph, threshold_prices, 100.3, params)
    assert borderline.full_calls == 1 and borderline.cheap_columns_returned == 0
    assert borderline.complete and math.isfinite(borderline.upper_bound)
    assert cost_model(cfg, params).reduced_cost(
        benefit=params.M, cost=borderline.column.delay_s,
        dual_cost=math.fsum(threshold_prices.get(row, 0.0) for row in borderline.column.claims),
        pi_f=100.3,
    ) <= params.epsilon
    wider = replace(connected, corridor_cells=connected.corridor_cells | {(1, -1)}, max_air_hops=4)
    universe = _exhaustive_anyangle_columns(wider, cfg, params, rows, 480.0, options)
    signed = {row: (-2.0 if index % 3 else 8.0)
              for index, row in enumerate(sorted({row for col in universe for row in col.claims}))}
    optimum = max(params.M - col.delay_s - math.fsum(signed.get(row, 0.0) for row in col.claims)
                  for col in universe)
    for cap in (2, 3, 5, 8):
        capped = replace(options, max_labels=cap, lazy_successors=False)
        result = price_zeta(ZetaGraph(wider, cfg, rows, capped), signed, 0.0, params,
                            known_columns=universe)
        assert result.upper_bound >= optimum - 1e-6
        assert result.reduced_cost <= optimum + 1e-6
