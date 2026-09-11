"""Independent regressions for pricing resources and solver result contracts."""

import math
import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from freespace_sim.planner.colgen import batch, pricing
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph, column_claims
from freespace_sim.planner.colgen.objective import cost_model
from freespace_sim.planner.colgen.solver import ColGenSolver
from freespace_sim.planner.colgen.translate import Column, column_to_intent
from freespace_sim.planner.colgen.windows import derive_cell_window
from freespace_sim.types import DenialReason, Terminal
from tests.test_colgen_e2e import _run_direct_batch
from tests.test_colgen_solver import _cfg, _params, _point, _request


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("bootstrap", [0, 1])
def test_pricing_preserves_labels_with_more_remaining_hops(monkeypatch, compiled, bootstrap):
    if compiled:
        pytest.importorskip("numba")
    cfg = _cfg(
        flight_levels_m=(30.0,), max_ground_delay_s=32.0, time_buffer_s=0.0,
        cost_ground_delay_per_s=20.0, cost_air_hold_per_s=3.0,
        cost_air_lateral_per_s=3.0, max_detour_factor=5.0,
    )
    params = _params(max_air_overrun_hops=4, bootstrap_roots=bootstrap)
    request = _request(1, (-4, 0), (4, 0), cfg)
    graph = build_flight_graph(request, cfg, (), params)
    view = pricing.DualView({
        RowKey.cell(5, -1, 0, 18): 200.0,
        RowKey.cell(5, -1, 0, 16): 500.0,
    }, cfg)
    model = cost_model(cfg, params)
    # A concrete feasible witness, independent of either search implementation.
    witness = Column(1, 6, 0, None, None, (
        (-4, 0), (-5, 0), (-6, 1), (-5, 1), (-4, 1), (-3, 1), (-2, 1),
        (-1, 1), (0, 1), (1, 1), (2, 0), (3, 0), (4, 0),
    ), 0.0)
    claims = column_claims(witness, graph, cfg)
    intent = column_to_intent(witness, request, cfg)
    witness_rc = params.M - model.intent_cost(intent, cfg) - view.claim_cost(claims) - 9432.0
    assert len(witness.cell_path) - 1 == graph.max_air_hops
    assert witness_rc == pytest.approx(40.0)

    if compiled:
        real = pricing._best_column_compiled

        def require_compiled(*args, **kwargs):
            outcome = real(*args, **kwargs)
            assert not isinstance(outcome, pricing.Declined), outcome
            return outcome

        monkeypatch.setattr(pricing, "_best_column_compiled", require_compiled)
    else:
        monkeypatch.setattr(
            pricing, "_best_column_compiled", lambda *a, **k: pricing.Declined.NO_NUMBA
        )
    # Before the fix both searches declared no improving column at this dual.
    rc, column = pricing.price_flight(graph, view, 9432.0, cfg, params)
    assert column is not None
    assert rc >= witness_rc - 1e-8
    assert column_claims(column, graph, cfg) == column.claims


def _enumerated_best_rc(graph, cfg, model, duals, benefit):
    """Enumerate every small path/departure, with no label dominance or bound pruning."""
    lo, hi = derive_cell_window(cfg)
    revisit_depth = hi - lo
    best = -math.inf
    stack = [(graph.origin_cell,)]
    while stack:
        path = stack.pop()
        if len(path) > 1 and path[-1] == graph.dest_cell:
            for departure in range(graph.base_step, graph.latest_departure_step + 1):
                column = Column(graph.request.flight_id, departure, 0, None, None, path, 0.0)
                try:
                    claims = column_claims(column, graph, cfg)
                except ValueError:
                    continue
                intent = column_to_intent(column, graph.request, cfg)
                rc = benefit - model.intent_cost(intent, cfg) - math.fsum(
                    duals.get(row, 0.0) for row in claims
                )
                best = max(best, rc)
        if len(path) - 1 < graph.max_air_hops:
            recent = path[-revisit_depth:] if revisit_depth else ()
            stack.extend(
                (*path, neighbour) for neighbour in graph.outgoing_neighbors(path[-1])
                if neighbour not in recent
            )
    return best


@pytest.mark.parametrize("ground_weight", [1.0, 20.0])
@pytest.mark.parametrize("buffer", [0.0, 4.0])
@pytest.mark.parametrize("compiled", [False, True])
def test_pricing_matches_exhaustive_small_path_enumeration(ground_weight, buffer, compiled):
    if compiled:
        pytest.importorskip("numba")
    cfg = _cfg(
        flight_levels_m=(30.0,), max_ground_delay_s=8.0, time_buffer_s=buffer,
        cost_ground_delay_per_s=ground_weight, cost_air_hold_per_s=3.0,
        cost_air_lateral_per_s=3.0, max_detour_factor=5.0,
    )
    params = _params(max_air_overrun_hops=2)
    graph = build_flight_graph(_request(1, (0, 0), (3, 0), cfg), cfg, (), params)
    model = cost_model(cfg, params)
    rng = random.Random(19)
    duals = {
        RowKey.cell(cell, 0, step): rng.choice((0.0, 0.0, 12.0, 50.0, 200.0))
        for cell in sorted(graph.corridor_cells)
        for step in range(graph.max_step + 6)
    }
    expected = _enumerated_best_rc(graph, cfg, model, duals, params.M)
    view = pricing.DualView(duals, cfg)
    args = (graph, view, 0.0, cfg, params.M, frozenset())
    outcome = (
        pricing._best_column_compiled(*args, model=model) if compiled
        else pricing._best_column(*args, seed=False, model=model)
    )
    assert not isinstance(outcome, pricing.Declined), outcome
    assert outcome[1] is not None
    assert outcome[0] == pytest.approx(expected)


@pytest.mark.parametrize("objective", ["total_cost", "total_delay"])
@pytest.mark.parametrize("stale_cost", [None, 0.0])
def test_imported_seeds_are_repriced_for_the_active_objective(objective, stale_cost):
    cfg = _cfg(flight_levels_m=(30.0,), max_ground_delay_s=16.0)
    params = _params(max_air_overrun_hops=1, objective=objective)
    request = _request(1, (-3, 0), (0, 3), cfg)
    graph = build_flight_graph(request, cfg, (), params)
    seed = pricing.seed_column(graph, cfg)
    baseline = ColGenSolver().solve([request], cfg, (), params)
    # Also reject stale coefficients and claims from an external seed producer.
    imported = replace(
        seed, delay_s=seed.delay_s if stale_cost is None else stale_cost, claims=frozenset(),
    )
    result = ColGenSolver().solve([request], cfg, (), params, seed_columns={1: [imported]})
    selected = result.columns[1]
    actual = cost_model(cfg, params).intent_cost(column_to_intent(selected, request, cfg), cfg)
    assert actual > 0.0
    assert selected.delay_s == pytest.approx(actual)
    assert result.stats["objective"] == pytest.approx(baseline.stats["objective"])
    assert selected.claims == column_claims(selected, graph, cfg)


@pytest.mark.parametrize("metric", ["cost", "revenue"])
def test_global_integer_gap_does_not_use_the_restricted_ip_bound(metric):
    cfg = _cfg(flight_levels_m=(30.0,), max_ground_delay_s=16.0)
    data = [((-2, -2), (3, 2), 4), ((-5, -5), (-3, -5), 0),
            ((-5, 2), (1, 3), 8), ((0, -5), (-5, -1), 4),
            ((-1, -5), (-4, 5), 8), ((1, -5), (-2, 5), 4)]
    requests = [_request(i, o, d, cfg, departure=t) for i, (o, d, t) in enumerate(data)]
    result = ColGenSolver().solve(requests, cfg, (), _params(
        max_air_overrun_hops=1, max_iterations=1, gap_metric=metric,
    ))
    stats = result.stats
    revenue = stats["master_objective"]
    cost = len(requests) * 10000.0 - revenue
    cost_gap = (cost - stats["cost_lower_bound"]) / max(1.0, abs(cost))
    revenue_gap = (stats["upper_bound"] - revenue) / abs(revenue)
    assert stats["ip_optimal"] is True
    assert stats["restricted_ip_gap"] == pytest.approx(0.0)
    assert stats["global_integer_gap_cost"] == pytest.approx(cost_gap)
    assert stats["global_integer_gap_revenue"] == pytest.approx(revenue_gap)
    assert stats["ip_gap"] == pytest.approx(cost_gap if metric == "cost" else revenue_gap)
    assert stats["ip_gap"] > 0.1
    assert stats["ip_gap_met"] is False


def test_iteration_callback_has_an_aligned_immutable_lp_snapshot():
    cfg = _cfg(flight_levels_m=(30.0,), max_ground_delay_s=16.0)
    requests = [_request(1, (-2, -5), (-2, -11), cfg),
                _request(2, (-8, -4), (0, -4), cfg)]
    snapshots = []

    def observe(state):
        assert len(state["lp_x"]) == len(state["master"].columns)
        assert state["lp_columns"] == state["master"].columns
        state["master"].fractional_loads(state["lp_x"])
        assert not state["lp_x"].flags.writeable
        assert np.all(state["lp_x"][state["lp_column_count"]:] == 0.0)
        snapshots.append(state)

    ColGenSolver().solve(requests, cfg, (), _params(
        max_air_overrun_hops=1, max_iterations=3, lp_gap=0.0, ip_gap=0.0,
    ), on_iteration=observe)
    assert snapshots
    assert any(len(s["lp_columns"]) > s["lp_column_count"] for s in snapshots)
    for state in snapshots:
        assert len(state["lp_x"]) == len(state["lp_columns"])
    for previous, current in zip(snapshots, snapshots[1:]):
        assert not np.shares_memory(previous["lp_x"], current["lp_x"])


@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize("failure", ["same_hex", "foreign_wall"])
def test_unrepresentable_request_is_denied_without_aborting_others(workers, failure):
    cfg = _cfg(flight_levels_m=(30.0,), max_ground_delay_s=16.0)
    valid = _request(1, (-3, 0), (0, 3), cfg)
    bad = _request(99, (20, 20), (20, 20) if failure == "same_hex" else (24, 20), cfg)
    bad = replace(bad, dest=bad.dest + [1.0, 0.0, 0.0])
    walls = () if failure == "same_hex" else ((_point((20, 20), cfg), Terminal("X", 1)),)
    result = ColGenSolver().solve([bad, valid], cfg, walls, _params(
        n_pricing_workers=workers, max_iterations=1,
    ))
    assert set(result.columns) == {1}
    assert result.stats["denied_flight_ids"] == (99,)
    assert result.stats["seedless_flight_ids"] == (99,)
    assert 99 not in result.stats["search_exhausted_flight_ids"]
    assert result.stats["heuristic_cost"] == pytest.approx(10000.0 + result.columns[1].delay_s)


def test_all_unrepresentable_requests_return_denials():
    cfg = _cfg()
    bad = _request(99, (0, 0), (0, 0), cfg)
    result = ColGenSolver().solve([bad], cfg, (), _params(n_pricing_workers=1))
    assert result.columns == {}
    assert result.stats["denied_flight_ids"] == (99,)
    assert result.stats["heuristic_cost"] == pytest.approx(10000.0)


def test_batch_files_valid_flights_and_denies_same_hex_request():
    cfg = _cfg(flight_levels_m=(30.0,))
    valid = _request(1, (-4, 0), (4, 0), cfg)
    bad = _request(99, (20, 20), (20, 20), cfg)
    bad = replace(bad, dest=bad.dest + [1.0, 0.0, 0.0])
    intents, _ledger = _run_direct_batch([valid, bad], cfg, _params())
    by_id = {intent.request.flight_id: intent for intent in intents}
    assert by_id[1].accepted
    assert by_id[99].denial_reason is DenialReason.BUDGET_EXCEEDED


def test_seeding_timeout_preserves_known_graph_denials(monkeypatch):
    from freespace_sim.planner.colgen import solver

    cfg = _cfg()
    bad = _request(99, (20, 20), (20, 20), cfg)
    valid = _request(1, (-4, 0), (4, 0), cfg)

    def timeout(*args, **kwargs):
        raise pricing.PricingTimeout("test seeding deadline")

    monkeypatch.setattr(solver, "seed_column", timeout)
    result = ColGenSolver().solve([bad, valid], cfg, (), _params())
    assert result.stats["budget_denied_flight_ids"] == (99,)
    assert result.stats["search_exhausted_flight_ids"] == (1,)
    assert math.isinf(result.stats["global_integer_gap"])
    assert result.stats["ip_gap_met"] is False


def test_graph_configuration_errors_are_not_converted_to_denials():
    cfg = _cfg(fixed_exit_lanes=False)
    request = replace(_request(1, (0, 0), (0, 0), cfg), origin_terminal=Terminal("A", 1))
    with pytest.raises(NotImplementedError, match="fixed_exit_lanes"):
        ColGenSolver().solve([request], cfg, (), _params())


@pytest.mark.parametrize("include_valid", [False, True])
def test_warm_start_skips_graph_incompatible_accepted_requests(monkeypatch, include_valid):
    import freespace_sim.sim as sim

    cfg = _cfg(flight_levels_m=(30.0,))
    params = _params(warm_start_planner="astar")
    bad = _request(99, (20, 20), (20, 20), cfg)
    requests = [bad]
    intents = [SimpleNamespace(request=bad, accepted=True)]
    if include_valid:
        valid = _request(1, (-4, 0), (4, 0), cfg)
        graph = build_flight_graph(valid, cfg, (), params)
        requests.append(valid)
        intents.append(column_to_intent(pricing.seed_column(graph, cfg), valid, cfg))
    monkeypatch.setattr(sim, "run", lambda *a, **k: SimpleNamespace(intents=intents))
    seeds = batch._build_warm_start(requests, cfg, (), params)
    assert (set(seeds) == {1}) if include_valid else (seeds is None)
