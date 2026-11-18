"""Provided-only initialization must remain feasible and recover missing flights."""

from dataclasses import replace

import pytest

import freespace_sim.planner.colgen.solver as solver_module
from experiments.run import colgen_params_from_args, parse_args
from freespace_sim.planner.colgen.network import RowIndex, build_flight_graph
from freespace_sim.planner.colgen.objective import cost_model
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import seed_column
from freespace_sim.planner.colgen.solver import ColGenSolver, _shift_column
from freespace_sim.planner.colgen.translate import column_to_intent
from freespace_sim.planner.colgen.warm_start import build, intent_to_column
from tests.test_colgen_solver import _cfg, _params, _request


def test_seed_policy_defaults_and_cli():
    assert ColGenParams().seed_nominal_routes is True
    assert ColGenParams().worker_seed_routes is False
    assert ColGenParams().warm_start_max_shift_steps == 8
    args = parse_args([
        "--scenario", "metro_uniform", "--planner", "colgen",
        "--colgen-warm-start", "astar", "--colgen-no-nominal-seeds",
        "--colgen-warm-start-max-shift", "0",
        "--colgen-provided-seed-ladder", "10", "--colgen-worker-seeds",
    ])
    params = colgen_params_from_args(args, "colgen")
    assert params.worker_seed_routes is True
    assert params.warm_start_planner == "astar"
    assert params.seed_nominal_routes is False
    assert params.warm_start_max_shift_steps == 0
    assert params.provided_seed_ladder_steps == 10


@pytest.mark.parametrize("value", [1, "false", None])
@pytest.mark.parametrize("field", ["seed_nominal_routes", "worker_seed_routes"])
def test_seed_policy_requires_boolean(value, field):
    with pytest.raises(TypeError, match=field):
        ColGenParams(**{field: value})


@pytest.mark.parametrize("value, error", [(True, TypeError), (1.5, TypeError), (-1, ValueError)])
def test_import_shift_requires_nonnegative_integer(value, error):
    with pytest.raises(error, match="warm_start_max_shift_steps"):
        ColGenParams(warm_start_max_shift_steps=value)


def test_provided_only_requires_columns():
    cfg = _cfg()
    with pytest.raises(ValueError, match="requires non-empty seed_columns"):
        ColGenSolver().solve(
            [_request(1, (-3, 0), (3, 0), cfg)], cfg, (),
            _params(seed_nominal_routes=False),
        )


@pytest.mark.parametrize("import_count", [1, 2])
def test_provided_only_skips_nominal_ladder_and_greedy_but_prices_missing_flights(
    monkeypatch, import_count,
):
    cfg = _cfg()
    params = _params(
        seed_nominal_routes=False, seed_ladder_steps=20, worker_seed_routes=True,
        n_pricing_workers=2,
        iteration_ip_time_limit_s=3, ip_time_limit_s=5, time_limit_s=60,
    )
    requests = [
        _request(1, (-3, 0), (3, 0), cfg),
        _request(2, (-3, 10), (3, 10), cfg),
    ]
    model = cost_model(cfg, params)
    seeds = {}
    for request in requests[:import_count]:
        graph = build_flight_graph(request, cfg, (), params)
        nominal = seed_column(graph, cfg, model=model)
        seeds[request.flight_id] = [_shift_column(nominal, nominal.departure_step + 8, cfg, model)]

    def forbidden(*args, **kwargs):
        pytest.fail("provided-only initialization called a nominal/greedy seed builder")

    monkeypatch.setattr(solver_module.PricingPool, "seed_routes", forbidden)
    for name in ("seed_column", "_add_departure_ladder", "_initial_feasible_selection",
                 "_greedy_feasible_selection"):
        monkeypatch.setattr(solver_module, name, forbidden)
    rounds = []
    result = ColGenSolver().solve(
        requests, cfg, (), params, seed_columns=seeds,
        on_iteration=lambda record: rounds.append(record),
    )
    assert rounds[0]["lp_column_count"] == import_count
    assert result.stats["initial_seed_columns"] == 0
    assert result.stats["seeded_columns"] == import_count
    assert result.stats["initial_heuristic_flights"] == import_count
    assert len(result.columns) == 2
    assert all(column.departure_step == 0 for column in result.columns.values())


def test_air_hold_is_rejected_before_cell_deduplication_erases_it():
    cfg, params = _cfg(), _params()
    request = _request(1, (-3, 0), (3, 0), cfg)
    graph = build_flight_graph(request, cfg, (), params)
    model = cost_model(cfg, params)
    nominal = seed_column(graph, cfg, model=model)
    intent = replace(column_to_intent(nominal, request, cfg), air_hold_s=cfg.dt_s)
    column, reason = intent_to_column(intent, graph, cfg, model)
    assert column is None
    assert reason == "air hold is not representable"


def test_import_collision_priority_is_request_time_before_flight_id():
    cfg, params = _cfg(), _params()
    model = cost_model(cfg, params)
    requests = [
        replace(_request(1, (-3, 0), (3, 0), cfg, departure=20), t_request=10),
        replace(_request(2, (-3, 0), (3, 0), cfg, departure=20), t_request=0),
    ]
    graphs = {r.flight_id: build_flight_graph(r, cfg, (), params) for r in requests}
    accepted = {
        r.flight_id: column_to_intent(seed_column(graphs[r.flight_id], cfg, model=model), r, cfg)
        for r in requests
    }
    seeds, stats = build(accepted, graphs, cfg, model, RowIndex(), max_shift=0)
    assert set(seeds) == {2}
    assert stats["dropped (no feasible shift)"] == 1
    assert stats["total held steps"] == 0


def test_import_does_not_expand_the_pricing_hop_budget():
    cfg, params = _cfg(), _params(max_air_overrun_hops=0)
    request = _request(1, (-1, 0), (1, 0), cfg)
    graph = build_flight_graph(request, cfg, (), params)
    model = cost_model(cfg, params)
    nominal = seed_column(graph, cfg, model=model)
    detour = replace(nominal, cell_path=((-1, 0), (-1, 1), (0, 1), (1, 0)))
    intent = column_to_intent(detour, request, cfg)
    column, reason = intent_to_column(intent, graph, cfg, model)
    assert column is None
    assert reason == "path exceeds pricing hop budget"


@pytest.mark.parametrize(
    "departure_step,cfg_overrides,exact_earlier_costs",
    [
        (0, {}, False),
        (3, {}, False),
        (15, {}, False),
        (31, {}, False),
        pytest.param(
            7,
            {
                "dt_s": 0.7,
                "nominal_speed_mps": 120.0 / 0.7,
                "max_ground_delay_s": 14.0,
                "cost_ground_delay_per_s": 0.3,
            },
            True,
            id="fractional-weighted",
        ),
    ],
)
def test_centered_ladder_is_clipped_to_both_legal_departure_bounds(
    departure_step, cfg_overrides, exact_earlier_costs,
):
    from freespace_sim.planner.colgen.network import column_claims

    cfg_values = {"max_ground_delay_s": 128, **cfg_overrides}
    cfg = _cfg(**cfg_values)
    params = _params(seed_nominal_routes=False, provided_seed_ladder_steps=10,
                     max_iterations=1, iteration_ip_time_limit_s=3,
                     ip_time_limit_s=5, time_limit_s=60)
    request = _request(1, (-3, 0), (3, 0), cfg)
    graph = build_flight_graph(request, cfg, (), params)
    model = cost_model(cfg, params)
    nominal = seed_column(graph, cfg, model=model)
    seed = _shift_column(nominal, departure_step, cfg, model)
    snapshots = []
    ColGenSolver().solve([request], cfg, (), params, seed_columns={1: [seed]},
                        on_iteration=lambda row: snapshots.append(row["lp_columns"][:row["lp_column_count"]]))
    initial = snapshots[0]
    assert {c.departure_step for c in initial} == set(range(
        max(graph.base_step, departure_step - 10),
        min(graph.latest_departure_step, departure_step + 10) + 1,
    ))
    assert len(initial) == len({c.departure_step for c in initial})
    for column in initial:
        assert column.cell_path == seed.cell_path
        assert column.claims == column_claims(column, graph, cfg)
        expected = model.intent_cost(column_to_intent(column, request, cfg), cfg)
        if exact_earlier_costs and column.departure_step <= departure_step:
            assert column.delay_s == expected
        else:
            assert column.delay_s == pytest.approx(expected)


def test_unrepaired_import_keeps_conflicting_centers_for_the_ip():
    cfg, params = _cfg(), _params()
    model = cost_model(cfg, params)
    requests = [_request(1, (-3, 0), (3, 0), cfg), _request(2, (0, -3), (0, 3), cfg)]
    graphs = {r.flight_id: build_flight_graph(r, cfg, (), params) for r in requests}
    accepted = {
        r.flight_id: column_to_intent(seed_column(graphs[r.flight_id], cfg, model=model), r, cfg)
        for r in requests
    }
    seeds, stats = build(accepted, graphs, cfg, model, RowIndex(), max_shift=0,
                         require_joint_feasibility=False)
    assert set(seeds) == {1, 2}
    assert all(cols[0].departure_step == 0 for cols in seeds.values())
    assert stats["rows over cap"] > 0
    assert stats["total held steps"] == 0

    result = ColGenSolver().solve(requests, cfg, (), _params(
        seed_nominal_routes=False, provided_seed_ladder_steps=10,
        iteration_ip_time_limit_s=3, ip_time_limit_s=5, time_limit_s=60,
    ), seed_columns=seeds)
    assert result.stats["initial_heuristic_flights"] == 0
    assert len(result.columns) == 2
    assert result.columns[1].departure_step != result.columns[2].departure_step


@pytest.mark.parametrize("value, error", [(True, TypeError), (1.5, TypeError), (-1, ValueError)])
def test_provided_ladder_requires_nonnegative_integer(value, error):
    with pytest.raises(error, match="provided_seed_ladder_steps"):
        ColGenParams(provided_seed_ladder_steps=value)


def test_departure_ladder_reuses_validation_with_cold_claim_and_cost_parity(monkeypatch):
    """Every ladder rung has independently rebuilt claims and cost after one route validation."""
    from types import SimpleNamespace

    import freespace_sim.planner.colgen.translate as translate_module
    from freespace_sim.planner.colgen.network import column_claims

    cfg = _cfg(dt_s=0.7, nominal_speed_mps=120.0 / 0.7, max_ground_delay_s=14.0)
    params = _params()
    request = _request(603, (-3, 0), (3, 0), cfg, departure=1e12)
    graph = build_flight_graph(request, cfg, (), params)
    model = cost_model(cfg, params)
    nominal = seed_column(graph, cfg, model=model)
    # A fresh graph ensures this test measures the ladder's first certification too.
    graph = build_flight_graph(request, cfg, (), params)
    calls = []
    original = translate_module.column_to_intent

    def observed(*args, **kwargs):
        calls.append(args[0].departure_step)
        return original(*args, **kwargs)

    monkeypatch.setattr(translate_module, "column_to_intent", observed)
    canonical = solver_module._canonical_column(nominal, graph, cfg)
    columns = []
    added = solver_module._add_departure_ladder(
        SimpleNamespace(add_column=columns.append), canonical, graph, cfg, model, steps=4, stride=2
    )
    assert added == len(columns) == 4
    assert calls == [nominal.departure_step]
    monkeypatch.setattr(translate_module, "column_to_intent", original)
    for index, column in enumerate(columns, start=1):
        assert column.departure_step == nominal.departure_step + 2 * index
        assert column.claims == column_claims(
            column, build_flight_graph(request, cfg, (), params), cfg
        )
        assert column.delay_s == pytest.approx(
            model.intent_cost(original(column, request, cfg), cfg)
        )


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("objective", ["total_delay", "total_cost"])
def test_worker_initial_routes_preserve_ladders_and_first_pricing(monkeypatch, workers, objective):
    """The opt-in preserves initial insertion order, claims, costs and completed solver decisions."""
    from freespace_sim.planner.colgen.pricing_pool import PricingPool
    from tests.test_colgen_pricing_pool import _fingerprint

    cfg = _cfg()
    requests = [_request(3, (-3, 0), (3, 0), cfg), _request(1, (0, -3), (0, 3), cfg)]
    params = _params(n_pricing_workers=workers, seed_ladder_steps=3, seed_ladder_stride=2,
                     max_iterations=3, time_limit_s=120, objective=objective)
    before, after = [], []
    baseline = ColGenSolver().solve(requests, cfg, (), params, on_iteration=before.append)
    if not workers:
        def forbidden(*args, **kwargs):
            pytest.fail("serial solve started worker seeds")
        monkeypatch.setattr(PricingPool, "seed_routes", forbidden)
    else:
        def forbidden(*args, **kwargs):
            pytest.fail("parallel seeding called the parent seed builder")
        monkeypatch.setattr(solver_module, "seed_column", forbidden)
    candidate = ColGenSolver().solve(
        requests, cfg, (), replace(params, worker_seed_routes=True), on_iteration=after.append)
    assert _fingerprint(candidate) == _fingerprint(baseline)
    assert candidate.stats["ladder_columns"] == baseline.stats["ladder_columns"] > 0
    assert [row["lp_columns"] for row in before] == [row["lp_columns"] for row in after]
    assert [row["capacity_duals"] for row in before] == [row["capacity_duals"] for row in after]


@pytest.mark.parametrize("exit_kind", ["timeout", "worker_lost", "canonical", "warm_start", "parent_deadline"])
def test_early_worker_seed_exit_closes_the_live_pool(monkeypatch, exit_kind):
    """Startup before the first LP must be cleaned up on partial seeds and insertion errors."""
    from freespace_sim.planner.colgen.pricing_pool import PricingPool

    cfg = _cfg()
    requests = [_request(1, (-3, 0), (3, 0), cfg), _request(2, (0, -3), (0, 3), cfg)]
    original = PricingPool.seed_routes
    processes = []

    def seed(pool, flight_ids, deadline):
        result = original(pool, flight_ids, deadline)
        processes.extend(channel.proc for channel in pool._channels)
        if exit_kind in {"timeout", "worker_lost"}:
            return replace(result, flight_ids=(1,), columns=result.columns[:1],
                           reduced_costs=(0.0,), timeout_flight_id=2,
                           kernel_counters={"pool_worker_lost": int(exit_kind == "worker_lost")})
        return result

    monkeypatch.setattr(PricingPool, "seed_routes", seed)
    if exit_kind == "parent_deadline":
        original_canonical = solver_module._canonical_column
        def expire_after_insertion(*args, **kwargs):
            column = original_canonical(*args, **kwargs)
            monkeypatch.setattr(solver_module.time, "monotonic", lambda: 1e20)
            return column
        monkeypatch.setattr(solver_module, "_canonical_column", expire_after_insertion)
    if exit_kind == "canonical":
        def reject(*args, **kwargs):
            raise ValueError("canonical rejection")
        monkeypatch.setattr(solver_module, "_canonical_column", reject)
    params = _params(n_pricing_workers=2, worker_seed_routes=True, seed_ladder_steps=0)
    extras = {999: [None]} if exit_kind == "warm_start" else None
    if exit_kind in {"timeout", "parent_deadline"}:
        result = ColGenSolver().solve(requests, cfg, (), params)
        assert result.stats["preprocessing_stage"] == "seeding"
        assert result.stats["seed_flights_processed"] == result.stats["n_columns"] == 1
    else:
        with pytest.raises((ValueError, KeyError, RuntimeError)):
            ColGenSolver().solve(requests, cfg, (), params, seed_columns=extras)
    assert len(processes) == 2
    assert all(not process.is_alive() for process in processes)
