"""Restricted pricing may add columns, but only exact sweeps certify a bound."""

import math
from dataclasses import replace

import pytest

from freespace_sim.planner.colgen import pricing, solver
from freespace_sim.planner.colgen.master import RestrictedMaster
from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph, column_claims
from freespace_sim.planner.colgen.objective import cost_model
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing_pool import PricingPool, SweepResult, price_sweep
from tests.test_colgen_solver import _cfg, _params, _request


def test_defaults_choose_eager_ip_and_point_one_percent_lp_gap():
    params = ColGenParams()
    assert params.iteration_ip_time_limit_s == 30
    assert params.iteration_ip_eager is True
    assert params.lp_gap == 0.001
    assert params.cheap_pricing is False


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_exact_interval_rejects_invalid_values(value):
    with pytest.raises((ValueError, TypeError)):
        ColGenParams(exact_pricing_interval=value)


def test_cheap_column_is_canonical_and_does_not_call_unrestricted_search(monkeypatch):
    cfg = _cfg(max_ground_delay_s=48)
    params = _params(max_air_overrun_hops=1, bootstrap_method="dp")
    graph = build_flight_graph(_request(1, (-3, 0), (3, 0), cfg), cfg, (), params)
    seed = pricing.seed_column(graph, cfg, model=cost_model(cfg, params))
    duals = pricing.DualView({row: 10.0 for row in seed.claims}, cfg)
    pi = 0.0
    original = pricing._best_column_compiled
    seen = []

    def restricted(*args, **kwargs):
        assert kwargs.get("keep_roots"), "cheap mode reached unrestricted search"
        seen.append(kwargs["keep_roots"])
        return original(*args, **kwargs)

    monkeypatch.setattr(pricing, "_best_column_compiled", restricted)
    rc, column = pricing.price_flight(graph, duals, pi, cfg, params, heuristic_only=True)
    assert seen
    assert all(len(roots) <= params.bootstrap_roots for roots in seen)
    assert column is not None
    assert column.claims == column_claims(column, graph, cfg)
    assert rc == pytest.approx(params.M - column.delay_s - duals.claim_cost(column.claims) - pi)
    monkeypatch.setattr(pricing, "_best_column_compiled", original)
    exact_rc, _ = pricing.price_flight(graph, duals, pi, cfg, params)
    assert exact_rc >= rc - 1e-7


def test_stalled_cheap_search_falls_back_before_certifying(monkeypatch):
    cfg = _cfg()
    requests = [_request(1, (-3, 0), (3, 0), cfg),
                _request(2, (0, -3), (0, 3), cfg)]
    params = _params(cheap_pricing=True, exact_pricing_interval=5, max_iterations=3,
                     iteration_ip_time_limit_s=1, ip_gap=0)
    original = solver.price_sweep
    modes = []

    def stalled(order, *args, **kwargs):
        modes.append(kwargs["heuristic_only"])
        if kwargs["heuristic_only"]:
            # An empty restricted search is NOT evidence of zero exact RC.
            return SweepResult(tuple(order), (0.0,) * len(order), (None,) * len(order), None)
        return original(order, *args, **kwargs)

    monkeypatch.setattr(solver, "price_sweep", stalled)
    snapshots = []
    result = solver.ColGenSolver().solve(requests, cfg, (), params, on_iteration=snapshots.append)
    assert modes[:2] == [True, False]
    assert all(p["pricing_exact"] for p in snapshots)
    assert result.stats["exact_pricing_sweeps"] > 0
    assert result.stats["cheap_pricing_sweeps"] > 0


def test_cheap_round_retains_bound_and_last_round_is_exact(monkeypatch):
    cfg = _cfg(max_ground_delay_s=480)
    requests = [_request(i, (-4, 0), (4, 0), cfg) for i in range(6)]
    params = _params(cheap_pricing=True, max_iterations=3, exact_pricing_interval=5,
                     seed_ladder_steps=0, iteration_ip_time_limit_s=1,
                     time_limit_s=120, ip_time_limit_s=5, ip_gap=0, lp_gap=0)
    original_bound = RestrictedMaster.upper_bound
    exact_bounds = []

    def bound(master, *args):
        exact_bounds.append(args)
        return original_bound(*args)

    monkeypatch.setattr(RestrictedMaster, "upper_bound", bound)
    snapshots = []
    result = solver.ColGenSolver().solve(requests, cfg, (), params, on_iteration=snapshots.append)
    cheap = [p for p in snapshots if not p["pricing_exact"]]
    assert cheap, "exercise a productive cheap round"
    assert math.isinf(cheap[0]["upper_bound"])
    assert snapshots[-1]["pricing_exact"]
    assert len(exact_bounds) == result.stats["exact_pricing_sweeps"]
    assert result.stats["cost_lower_bound"] <= result.stats["objective"] + 1e-6


def test_persistent_pool_switches_between_cheap_and_exact():
    cfg = _cfg()
    params = _params(n_pricing_workers=2, max_air_overrun_hops=1)
    requests = [_request(i, (-3, i), (3, i), cfg) for i in range(2)]
    catalog = StaticTerminalCatalog((), cfg)
    graphs = {r.flight_id: build_flight_graph(r, cfg, catalog, params) for r in requests}
    seeds = {f: pricing.seed_column(g, cfg, model=cost_model(cfg, params)) for f, g in graphs.items()}
    duals = {row: 10.0 for c in seeds.values() for row in c.claims}
    view = pricing.DualView(duals, cfg)
    flight_duals = {f: params.M - c.delay_s for f, c in seeds.items()}
    with PricingPool(requests, cfg, params, catalog) as pool:
        for cheap in (True, False, True):
            expected = price_sweep(list(graphs), requests, graphs, cfg,
                                   replace(params, n_pricing_workers=0), catalog,
                                   duals, view, flight_duals, seeds, heuristic_only=cheap)
            actual = price_sweep(list(graphs), requests, graphs, cfg, params, catalog,
                                 duals, view, flight_duals, seeds, pool=pool, heuristic_only=cheap)
            assert actual.complete
            assert actual.reduced_costs == expected.reduced_costs
            assert actual.columns == expected.columns
