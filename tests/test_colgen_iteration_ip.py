"""Per-round integer solves must preserve feasibility, bounds, and final-IP time."""

import math
import time

import pytest

from freespace_sim.planner.colgen.master import GurobiBackend, HighsBackend, RestrictedMaster
from freespace_sim.planner.colgen.solver import ColGenSolver
from tests.test_colgen_solver import _cfg, _params, _request


@pytest.mark.parametrize("search", [True, False], ids=["search", "rows_only"])
@pytest.mark.parametrize("backend", ["highs", "gurobi"])
def test_round_ip_feeds_feasible_incumbents_back_into_cg(monkeypatch, backend, search):
    if backend == "gurobi":
        pytest.importorskip("gurobipy")
    cfg = _cfg(max_ground_delay_s=480.0, seed=29)
    requests = [_request(i, (-4, 0), (4, 0), cfg) for i in range(1, 7)]
    params = _params(solver=backend, max_iterations=3, time_limit_s=120,
                     ip_reserve_s=10, ip_time_limit_s=10, ip_gap=0, lp_gap=0,
                     iteration_ip_time_limit_s=5, seed_ladder_steps=0,
                     iteration_ip_search=search)
    snapshots = []
    original_ip = RestrictedMaster.solve_ip
    deadlines = []
    solver_ip_calls = []

    def no_rounding(*args, **kwargs):
        pytest.fail("per-round IP mode must replace rounding and LNS")

    def observe_ip(master, *args, **kwargs):
        if "eager" in kwargs:
            deadlines.append(kwargs["deadline"])
        return original_ip(master, *args, **kwargs)

    def count_solver_ip(original):
        def counted(backend_self, *args, **kwargs):
            solver_ip_calls.append(1)
            return original(backend_self, *args, **kwargs)
        return counted

    def observe_round(payload):
        master = payload["master"]
        incumbent = master._heuristic_selection
        assert set(incumbent) == {r.flight_id for r in requests}
        assert master.is_claim_feasible(incumbent)
        # The new incumbent and every freshly priced column are visible before
        # the callback and will therefore be available to the next round.
        info = payload["iteration_ip"]
        assert info["columns"] == len(master.columns)
        assert info["cost"] == pytest.approx(payload["heuristic_cost"])
        assert info["cost"] == pytest.approx(sum(c.delay_s for c in incumbent.values()))
        assert payload["cost_lower_bound"] <= info["cost"] + 1e-6
        # Every row that could bind reaches the next LP, with or without the search.
        assert master.bindable_rows <= master.materialized_rows
        if not search:
            assert info["status"] == "rows_only"
            assert not solver_ip_calls, "rows-only rounds must not run a solver search"
        snapshots.append(payload)

    monkeypatch.setattr(RestrictedMaster, "round_heuristic", no_rounding)
    monkeypatch.setattr(RestrictedMaster, "lns_heuristic", no_rounding)
    monkeypatch.setattr(RestrictedMaster, "solve_ip", observe_ip)
    for backend_class in (HighsBackend, GurobiBackend):
        monkeypatch.setattr(backend_class, "solve_ip", count_solver_ip(backend_class.solve_ip))
    started = time.monotonic()
    result = ColGenSolver().solve(requests, cfg, (), params, on_iteration=observe_round)
    assert len(snapshots) >= 2, "exercise an LP solve after changing the master to binary"
    assert len(deadlines) == result.stats["pricing_sweeps_completed"]
    assert result.stats["iteration_ip_calls"] == len(deadlines)
    assert all(deadline <= started + params.time_limit_s - 9.9 for deadline in deadlines)
    costs = [p["heuristic_cost"] for p in snapshots]
    assert costs == sorted(costs, reverse=True)
    # The final IP still searches and returns a complete schedule.
    assert solver_ip_calls
    assert len(result.columns) == len(requests)


@pytest.mark.parametrize(("overrides", "message"), [
    ({"iteration_ip_time_limit_s": -1}, "iteration_ip_time_limit_s"),
    ({"iteration_ip_time_limit_s": math.inf}, "iteration_ip_time_limit_s"),
    ({"iteration_ip_time_limit_s": math.nan}, "iteration_ip_time_limit_s"),
    ({"iteration_ip_search": False, "iteration_ip_eager": False}, "iteration_ip_search"),
])
def test_invalid_iteration_ip_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _params(**overrides)


def test_expired_ip_does_not_reuse_previous_incumbent_trajectory():
    from freespace_sim.planner.colgen.network import RowIndex

    master = RestrictedMaster([1], RowIndex(), _params())
    master.last_ip_trajectory = ((1.0, 5.0, 6.0),)
    assert master.solve_ip(deadline=0, eager=False) == {}
    assert master.last_ip_status == "time_limit_separation"
    assert master.last_ip_trajectory == ()
