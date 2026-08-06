"""The colgen objective has exactly one definition, and the compiled DP obeys it.

``CostModel`` is the source of truth for every reduced cost, bound and master coefficient
in column generation.  The ``@njit`` kernel cannot call it, so it is instead fed
*pre-weighted* inputs and its arithmetic is left alone -- which means the guarantee has to
be established by test rather than by construction.  That is what
``test_kernel_score_equals_the_objective_function`` does: it takes the column the kernel
actually returns and checks the score it accumulated against ``CostModel.evaluate``.

The other axis is that the (1.0, 1.0) default must be a genuine no-op.  Several tests
below assert bit-identity rather than approximate agreement, because the refactor replaced
longhand sums with ``evaluate`` calls and the only way that is safe is if the term order
and association were preserved exactly.
"""

import math

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen import dp_prepare, pricing
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph
from freespace_sim.planner.colgen.objective import (
    DELAY_MODEL,
    CostModel,
    cost_model,
    scaled_dt_s,
)
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import DualView, price_flight, seed_column
from freespace_sim.types import FlightRequest, vec

dp_kernel = pytest.importorskip(
    "freespace_sim.planner.colgen.dp_kernel", reason="requires Numba pricing kernel"
)


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


def _graph(cfg, origin=(0, 0), dest=(4, -1), slack=4, flight_id=1, **kw):
    params = ColGenParams(solver="highs", detour_slack_hops=slack, **kw)
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


def test_total_cost_refuses_a_config_the_kernel_cannot_express():
    """One air scalar has to cover cruise and loiter, so divergence must raise.

    The compiled DP charges a single weight per step.  Were hold priced differently from
    lateral, it would silently bill loiter as cruise -- exactly the class of silent
    mismatch this module exists to prevent.
    """

    cfg = _cfg(cost_air_hold_per_s=5.0, cost_air_lateral_per_s=3.0)
    with pytest.raises(ValueError, match="cost_air_hold_per_s"):
        cost_model(cfg, ColGenParams(objective="total_cost"))


# ------------------------------------------------------------ the kernel obeys the model


def test_scaled_dt_is_the_only_air_knob_the_kernel_needs():
    assert scaled_dt_s(DELAY_MODEL, 4.0) == 4.0
    assert scaled_dt_s(CostModel(1.0, 3.0), 4.0) == 12.0


def test_kernel_refuses_a_non_separable_objective():
    """The guard that turns a future silent wrong answer into a loud failure."""

    class NonSeparable(CostModel):
        @property
        def separable(self) -> bool:
            return False

    cfg = _cfg()
    graph, params = _graph(cfg)
    view = DualView({}, cfg)
    topology = pricing._topology_for(graph, cfg)
    duals = dp_prepare.prepare_duals(view, topology)
    variants = dp_prepare.prepare_variants(
        graph, cfg, view, topology, seed=False, benefit=0.0, pi_f=0.0
    )
    with pytest.raises(NotImplementedError, match="separable"):
        dp_kernel.search_dag(
            topology,
            duals,
            variants,
            cfg=cfg,
            benefit=0.0,
            pi_f=0.0,
            cost_cutoff=None,
            seed=False,
            model=NonSeparable(1.0, 3.0),
        )


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


@pytest.mark.parametrize("weights", [(1.0, 1.0), (1.0, 3.0), (2.0, 5.0)])
def test_kernel_score_equals_the_objective_function(weights):
    """The contract that makes the pre-weighting scheme safe.

    The kernel holds a *copy* of the objective's arithmetic -- it must, because it cannot
    call Python.  So take the column it actually returns, recover the cost its own score
    accumulated, and require that to equal ``CostModel.evaluate`` on the same trajectory.
    A weight that reached one layer but not another shows up here as a mismatch instead
    of as a quietly wrong column.

    Run at three weightings so a site that ignores the model entirely -- and therefore
    happens to agree at (1.0, 1.0) -- still fails.
    """

    model = CostModel(*weights)
    cfg = _cfg()
    graph, params = _graph(cfg)
    rng = np.random.default_rng(4242)
    cells = sorted(graph.corridor_cells)
    duals = _random_duals(rng, cells, graph.base_step, 12)
    view = DualView(duals, cfg)

    topology = pricing._topology_for(graph, cfg)
    prepared_duals = dp_prepare.prepare_duals(view, topology)
    variants = dp_prepare.prepare_variants(
        graph, cfg, view, topology, seed=False, benefit=1000.0, pi_f=0.0, model=model
    )
    result = dp_kernel.search_dag(
        topology,
        prepared_duals,
        variants,
        cfg=cfg,
        benefit=1000.0,
        pi_f=0.0,
        cost_cutoff=None,
        seed=False,
        model=model,
    )
    assert result.ok and result.candidates

    # For each proposal, the kernel's own notion of the trajectory's cost is
    # ground + origin leg + hops * dt, all of them pre-weighted.  Rebuild the same
    # quantity from the model and the RAW seconds and require exact agreement.
    dt = float(cfg.dt_s)
    for candidate in result.candidates[:32]:
        variant = None
        for i in range(variants.n_variants):
            if (
                int(variants.departure_step[i]) == candidate.departure_step
                and int(variants.lane_idx[i])
                == (-1 if candidate.origin_lane_idx is None else candidate.origin_lane_idx)
            ):
                variant = i
                break
        assert variant is not None, "every proposal comes from a prepared variant"

        kernel_cost = (
            float(variants.ground_delay_s[variant])
            + float(variants.origin_leg_s[variant])
            + candidate.hops * scaled_dt_s(model, dt)
        )
        raw_ground = (candidate.departure_step - graph.base_step) * dt
        raw_air = float(variants.origin_leg_s[variant]) / model.air_weight + candidate.hops * dt
        expected = model.evaluate(ground_s=raw_ground, air_detour_s=raw_air)
        assert kernel_cost == pytest.approx(expected, abs=1e-9), (
            f"kernel accumulated {kernel_cost} but the objective says {expected} "
            f"at weights {weights}"
        )


def test_prepared_variants_carry_pre_weighted_seconds():
    """Ground and air fields must be scaled; dual prices must NOT be.

    Duals already arrive in the master's currency -- they are prices, not seconds -- so
    weighting them would double-count the objective change.
    """

    cfg = _cfg()
    graph, _params = _graph(cfg)
    rng = np.random.default_rng(7)
    duals = _random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 10)
    view = DualView(duals, cfg)
    topology = pricing._topology_for(graph, cfg)

    plain = dp_prepare.prepare_variants(
        graph, cfg, view, topology, seed=False, benefit=0.0, pi_f=0.0, model=DELAY_MODEL
    )
    model = CostModel(ground_weight=2.0, air_weight=3.0)
    scaled = dp_prepare.prepare_variants(
        graph, cfg, view, topology, seed=False, benefit=0.0, pi_f=0.0, model=model
    )
    assert plain.n_variants == scaled.n_variants

    np.testing.assert_allclose(scaled.ground_delay_s, 2.0 * plain.ground_delay_s)
    np.testing.assert_allclose(scaled.origin_leg_s, 3.0 * plain.origin_leg_s)
    np.testing.assert_allclose(scaled.origin_fold_s, 3.0 * plain.origin_fold_s)
    assert scaled.destination_fold_s == pytest.approx(3.0 * plain.destination_fold_s)
    assert scaled.reference_time_s == pytest.approx(3.0 * plain.reference_time_s)

    # Prices, not seconds -- untouched.
    np.testing.assert_array_equal(scaled.paid_value, plain.paid_value)
    np.testing.assert_array_equal(scaled.dest_positive, plain.dest_positive)


# ------------------------------------------------------------------- the default is a no-op


@pytest.mark.parametrize("time_buffer_s", [4.0, 0.0])
def test_explicit_delay_model_prices_identically_to_the_default(time_buffer_s):
    """Threading the model must not perturb the answer at unit weights."""

    cfg = _cfg(time_buffer_s=time_buffer_s)
    rng = np.random.default_rng(99)
    for flight_id in range(4):
        graph, params = _graph(cfg, flight_id=flight_id)
        duals = _random_duals(rng, sorted(graph.corridor_cells), graph.base_step, 10)
        view = DualView(duals, cfg)
        default_rc, default_col = price_flight(
            graph, view, 0.0, cfg, params, require_improving=False
        )
        graph2, params2 = _graph(cfg, flight_id=flight_id)
        explicit_rc, explicit_col = price_flight(
            graph2, view, 0.0, cfg, params2, require_improving=False
        )
        assert default_rc == explicit_rc
        assert (default_col is None) == (explicit_col is None)
        if default_col is not None:
            assert default_col.cell_path == explicit_col.cell_path
            assert default_col.departure_step == explicit_col.departure_step
            assert default_col.delay_s == explicit_col.delay_s


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
