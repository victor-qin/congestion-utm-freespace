"""Column-generation initialization policies and their feasible starting pools."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import freespace_sim.planner.colgen.solver as solver_module
import freespace_sim.planner.colgen.warm_start as warm_start_module
from experiments.run import colgen_params_from_args, parse_args
from freespace_sim import sim as sim_module
from freespace_sim.planner.colgen.network import RowIndex, build_flight_graph, column_claims
from freespace_sim.planner.colgen.objective import cost_model
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import seed_column
from freespace_sim.planner.colgen.solver import ColGenSolver, _shift_column
from freespace_sim.planner.colgen.translate import column_to_intent
from freespace_sim.planner.colgen.warm_start import build, intent_to_column
from freespace_sim.types import Terminal
from freespace_sim.verify import find_interflight_conflict
from tests.test_colgen_solver import _cfg, _params, _point, _request


def test_seed_policy_defaults_and_cli():
    assert ColGenParams().initial_routes == "nominal"
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
    assert params.initial_routes == "nominal"
    assert params.worker_seed_routes is True
    assert params.warm_start_planner == "astar"
    assert params.seed_nominal_routes is False
    assert params.warm_start_max_shift_steps == 0
    assert params.provided_seed_ladder_steps == 10
    args = parse_args([
        "--scenario", "metro_uniform", "--planner", "colgen",
        "--colgen-initial-routes", "astar", "--colgen-worker-seeds",
    ])
    assert colgen_params_from_args(args, "colgen").initial_routes == "astar"


@pytest.mark.parametrize("value", [1, "false", None])
@pytest.mark.parametrize("field", ["seed_nominal_routes", "worker_seed_routes"])
def test_seed_policy_requires_boolean(value, field):
    with pytest.raises(TypeError, match=field):
        ColGenParams(**{field: value})


@pytest.mark.parametrize("value,error", [(True, TypeError), (None, TypeError), ("dp", ValueError)])
def test_initial_routes_requires_a_supported_name(value, error):
    with pytest.raises(error, match="initial_routes"):
        ColGenParams(initial_routes=value)


@pytest.mark.parametrize(
    "option,value", [("warm_start_planner", "astar"), ("seed_nominal_routes", False),
                     ("provided_seed_ladder_steps", 1)],
)
def test_astar_initial_routes_conflicts_with_other_seed_policies(option, value):
    with pytest.raises(ValueError, match=option):
        ColGenParams(initial_routes="astar", **{option: value})


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


def test_astar_initial_routes_rejects_explicit_seed_columns():
    cfg, params = _cfg(), _params(initial_routes="astar")
    request = _request(1, (-3, 0), (3, 0), cfg)
    graph = build_flight_graph(request, cfg, (), params)
    seed = seed_column(graph, cfg, model=cost_model(cfg, params))

    with pytest.raises(ValueError, match="seed_columns"):
        ColGenSolver().solve([request], cfg, (), params, seed_columns={1: [seed]})


def test_astar_initial_routes_supply_a_repaired_weighted_pool_with_or_without_workers(monkeypatch):
    """Only the rejected route falls back, and later retimes and repair retain valid claims."""
    cfg = _cfg(max_ground_delay_s=128.0, cost_ground_delay_per_s=0.8,
               cost_air_lateral_per_s=1.5, cost_air_hold_per_s=1.5)
    params = _params(initial_routes="astar", objective="total_cost", worker_seed_routes=True,
                     seed_ladder_steps=3, seed_ladder_stride=2, max_iterations=1, time_limit_s=60)
    requests = [_request(1, (-3, 0), (3, 0), cfg), _request(2, (0, -3), (0, 3), cfg)]
    static_terms = [(_point((6, 6), cfg), Terminal("unused-hub", 1, radius=90.0))]
    model = cost_model(cfg, params)
    original_run = sim_module.run
    original_import = warm_start_module.intent_to_column
    original_seed = solver_module.seed_column
    original_select = solver_module.RestrictedMaster.set_heuristic
    sources, seeded, heuristics = [], [], []

    def observe_fcfs(source_cfg, **kwargs):
        assert kwargs.get("parallel") is None
        assert kwargs["planner_name"] == "astar"
        assert [terminal.id for _, terminal in kwargs["static_terminals"]] == ["unused-hub"]
        assert source_cfg.astar_allow_airborne_holds is False
        assert source_cfg.cost_ground_delay_per_s == model.ground_weight
        assert source_cfg.cost_air_lateral_per_s == model.air_weight
        result = original_run(source_cfg, **kwargs)
        sources.append(result.intents)
        return result

    def reject_later_flight(intent, *args, **kwargs):
        if intent.request.flight_id == 2:
            return None, "test import rejection"
        return original_import(intent, *args, **kwargs)

    def observe_fallback(graph, *args, **kwargs):
        seeded.append(graph.request.flight_id)
        return original_seed(graph, *args, **kwargs)

    def observe_selection(master, selection):
        heuristics.append(dict(selection))
        return original_select(master, selection)

    def forbid_worker_seeds(*args, **kwargs):
        pytest.fail("A* initialization started a nominal seed sweep")

    monkeypatch.setattr(sim_module, "run", observe_fcfs)
    monkeypatch.setattr(warm_start_module, "intent_to_column", reject_later_flight)
    monkeypatch.setattr(solver_module, "seed_column", observe_fallback)
    monkeypatch.setattr(solver_module.RestrictedMaster, "set_heuristic", observe_selection)
    monkeypatch.setattr(solver_module.PricingPool, "seed_routes", forbid_worker_seeds)

    snapshots = []
    for workers in (0, 2):
        sources.clear()
        seeded.clear()
        heuristics.clear()
        rounds = []
        result = ColGenSolver().solve(
            requests, cfg, iter(static_terms), replace(params, n_pricing_workers=workers),
            on_iteration=rounds.append,
        )
        assert seeded == [2]
        (source,) = sources
        assert all(intent.accepted and intent.air_hold_s == 0.0 for intent in source)
        assert cfg.astar_allow_airborne_holds is True
        assert result.stats["initial_routes"] == "astar"
        assert result.stats["astar_accepted_routes"] == 2
        assert result.stats["astar_imported_routes"] == 1
        assert sum(result.stats["astar_rejected_routes"].values()) == 1
        assert result.stats["astar_nominal_fallback_routes"] == 1
        assert result.stats["initial_heuristic_strategy"] == "astar_routes"
        assert result.stats["initial_heuristic_flights"] == 2

        first = rounds[0]
        pool = first["lp_columns"][:first["lp_column_count"]]
        selection = heuristics[0]
        assert first["master"].is_claim_feasible(selection)
        intents = [column_to_intent(selection[r.flight_id], r, cfg) for r in requests]
        assert find_interflight_conflict(intents, cfg, static_terminals=static_terms) is None
        assert result.stats["initial_heuristic_delay_s"] == pytest.approx(
            sum(model.intent_cost(intent, cfg) for intent in intents)
        )
        for request in requests:
            fid = request.flight_id
            graph = build_flight_graph(request, cfg, static_terms, params)
            columns = [column for column in pool if column.flight_id == fid]
            base = min(columns, key=lambda column: column.departure_step)
            assert selection[fid] in columns
            assert {column.departure_step for column in columns} == {
                base.departure_step, base.departure_step + 2, base.departure_step + 4,
                base.departure_step + 6, selection[fid].departure_step,
            }
            assert all(column.cell_path == base.cell_path for column in columns)
            for column in columns:
                assert column.claims == column_claims(column, graph, cfg)
                intent = column_to_intent(column, request, cfg)
                assert column.delay_s == pytest.approx(model.intent_cost(intent, cfg))
            if fid == 1:
                imported, reason = original_import(source[0], graph, cfg, model)
                assert reason is None
                assert base.cell_path == imported.cell_path
                assert base.departure_step == imported.departure_step
            else:
                assert base.departure_step == graph.base_step < selection[fid].departure_step
        snapshots.append((pool, selection))
    assert snapshots[0] == snapshots[1]


@pytest.mark.parametrize("phase", ["fcfs", "import"])
def test_astar_initial_routes_stop_at_pricing_deadline_before_starting_workers(monkeypatch, phase):
    """FCFS shares the solve's clock and preserves the final-IP time reservation."""
    cfg = _cfg()
    params = _params(initial_routes="astar", objective="total_delay", time_limit_s=10.0,
                     ip_reserve_s=2.0, n_pricing_workers=2, worker_seed_routes=True)
    request = _request(1, (-3, 0), (3, 0), cfg)
    graph = build_flight_graph(request, cfg, (), params)
    nominal = seed_column(graph, cfg, model=cost_model(cfg, params))
    filed = column_to_intent(nominal, request, cfg)
    now = 100.0
    original_column_at = warm_start_module._column_at

    def expire_after_import(*args, **kwargs):
        nonlocal now
        result = original_column_at(*args, **kwargs)
        now = 108.0
        return result

    def expire_after_filing(source_cfg, **kwargs):
        nonlocal now
        assert source_cfg.cost_ground_delay_per_s == source_cfg.cost_air_lateral_per_s == 1.0
        if phase == "fcfs":
            now = 108.0
        kwargs["progress"](1, 1, filed)
        if phase == "import":
            from freespace_sim.ledger import ReservationLedger
            return SimpleNamespace(verified=True, intents=[filed], ledger=ReservationLedger(cfg))
        pytest.fail("FCFS continued after the solver's initialization deadline")

    def forbid_solver_work(*args, **kwargs):
        pytest.fail("an expired FCFS initializer started nominal or pricing work")

    monkeypatch.setattr(sim_module, "run", expire_after_filing)
    if phase == "import":
        monkeypatch.setattr(warm_start_module, "_column_at", expire_after_import)
    monkeypatch.setattr(solver_module.time, "monotonic", lambda: now)
    monkeypatch.setattr(solver_module, "seed_column", forbid_solver_work)
    monkeypatch.setattr(solver_module.PricingPool, "start", forbid_solver_work)
    result = ColGenSolver().solve([request], cfg, (), params, on_iteration=forbid_solver_work)

    assert now == 108.0
    assert result.columns == {}
    assert result.stats["termination_reason"] == "time_limit"
    assert result.stats["preprocessing_stage"] == "astar_initialization"
    assert result.stats["astar_accepted_routes"] == 1
    assert result.stats["astar_imported_routes"] == (1 if phase == "import" else 0)
    assert result.stats["astar_initialization_complete"] is False
    assert result.stats["search_exhausted_flight_ids"] == (request.flight_id,)
    assert result.stats["iterations"] == 0


def test_astar_initial_routes_stop_on_failed_physical_audit(monkeypatch):
    """An invalid FCFS source cannot become a pool or silently fall back to nominal."""
    cfg, params = _cfg(), _params(initial_routes="astar")
    request = _request(1, (-3, 0), (3, 0), cfg)

    def fail_audit(*args, **kwargs):
        from freespace_sim.ledger import ReservationLedger
        return SimpleNamespace(verified=False, intents=(), ledger=ReservationLedger(cfg))

    def forbid_fallback(*args, **kwargs):
        pytest.fail("a failed A* physical audit should stop the solve")

    monkeypatch.setattr(sim_module, "run", fail_audit)
    monkeypatch.setattr(solver_module, "seed_column", forbid_fallback)
    with pytest.raises(RuntimeError, match="physical audit"):
        ColGenSolver().solve([request], cfg, (), params)


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
