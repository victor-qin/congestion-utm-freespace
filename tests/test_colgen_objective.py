"""The colgen objective has exactly one definition, and pricing obeys it.

``CostModel`` is the source of truth for every reduced cost, bound and master coefficient
in column generation.  These tests pin two things: that a weighted objective actually
reaches every one of those sites, and that the (1.0, 1.0) default is a genuine no-op.
Several assertions below are bit-identity rather than approximate agreement, because the
longhand sums this replaced are only safely expressible as ``evaluate`` calls if term
order and association were preserved exactly.
"""

import math

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen import network as network_mod
from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph
from freespace_sim.planner.colgen.objective import DELAY_MODEL, CostModel, cost_model
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import DualView, price_flight, seed_column
from freespace_sim.types import FlightRequest, Terminal, vec
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


def test_cost_model_defaults_to_the_delay_objective():
    cfg = _cfg()
    assert cost_model(cfg, ColGenParams()) is DELAY_MODEL
    assert cost_model(cfg, None) is DELAY_MODEL


def test_total_cost_objective_picks_up_the_config_weights():
    cfg = _cfg()
    model = cost_model(cfg, ColGenParams(objective="total_cost"))
    assert model.ground_weight == cfg.cost_ground_delay_per_s
    assert model.air_weight == cfg.cost_air_lateral_per_s
    # The documented 1:3 ratio the A* planner uses; if config changes, this should fail
    # loudly rather than let colgen quietly price something else.
    assert (model.ground_weight, model.air_weight) == (1.0, 3.0)


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
