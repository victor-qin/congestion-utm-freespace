"""Full column-generation contracts for certified and approximate GPU pricing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import pytest

from freespace_sim.planner.colgen import pricing_pool, solver
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph
from freespace_sim.planner.colgen.pricing import seed_column
from freespace_sim.planner.colgen.pricing_platform import (
    PricingPlatform,
    PricingProblem,
    PricingResult,
)
from freespace_sim.planner.colgen.solver import ColGenSolver
from tests.test_colgen_pricing_platform import _platform
from tests.test_colgen_solver import (
    _assert_claim_feasible,
    _assert_files_cleanly,
    _cfg,
    _params,
    _request,
)


def _no_cpu_pricing(*args: object, **kwargs: object) -> None:
    """Fail if a selected GPU solve falls back to either full CPU pricing entry point."""
    raise AssertionError("GPU solve must use GPU pricing, including the final repair pass")


@pytest.mark.parametrize("status", ["label_limit", "deadline", "unsupported"])
def test_incomplete_gpu_sweep_emits_diagnostics_before_final_ip(
    monkeypatch: pytest.MonkeyPatch, status: str,
) -> None:
    """An interrupted sweep reports its work and retains routes without certifying a bound.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): substitutes GPU results without requiring a device.
    - status (str): resource failure reported by the pricing backend.

    Return
    --------
    - output (None): checks callback diagnostics, column retention and incomplete termination.
    """
    cfg = _cfg(max_ground_delay_s=0.0)
    params = _params(
        pricing_device="gpu", pricing_gpu_backend="metal", pricing_gpu_batch_size=2,
        seed_nominal_routes=False, seed_ladder_steps=0, max_iterations=2,
        iteration_ip_time_limit_s=1.0, ip_gap=0.0, lp_gap=0.0,
    )
    requests = [_request(i, (0, i * 10), (2, i * 10), cfg) for i in (1, 2)]
    seed = seed_column(build_flight_graph(requests[0], cfg, (), params), cfg)
    observations = []

    def batch(
        platform: PricingPlatform, problems: Sequence[PricingProblem],
        deadline: float | None = None,
    ) -> tuple[PricingResult, ...]:
        """Return valid partial routes and shared batch diagnostics for both flights."""
        assert all(problem.require_improving for problem in problems)
        return tuple(
            PricingResult(
                problem.graph.request.flight_id, params.M, seed_column(problem.graph, cfg),
                None, False, False, "metal", True, status,
                {"gpu_dispatches": 7, "labels": 123, "stop_reason": status},
            )
            for problem in problems
        )

    monkeypatch.setattr(PricingPlatform, "price_batch", batch)
    monkeypatch.setattr(solver, "price_flight", _no_cpu_pricing)
    monkeypatch.setattr(pricing_pool, "price_flight", _no_cpu_pricing)
    result = ColGenSolver().solve(
        requests, cfg, (), params, seed_columns={1: [seed]}, on_iteration=observations.append,
    )
    assert len(observations) == 1
    observation, = observations
    assert not observation["pricing_search_complete"]
    assert not observation["pricing_bound_certified"]
    assert observation["last_pricing_status"] == status
    assert observation["pricing_failure_status"] == status
    assert observation["iteration_ip"] == {}
    assert math.isinf(observation["upper_bound"])
    records = observation["sweep_flight_records"]
    assert len(records) == 2
    assert all(record["pricing_status"] == status for record in records)
    assert records[0]["gpu_batch_stats"]["stop_reason"] == status
    assert result.stats["gpu_dispatches"] == 7
    assert result.stats["pricing_sweeps_completed"] == 0
    assert result.stats["iteration_ip_calls"] == 0
    assert result.stats["termination_reason"] == (
        "time_limit" if status == "deadline" else f"pricing_{status}"
    )
    assert not result.stats["ip_gap_met"]
    assert set(result.columns) == {1, 2}
    _assert_files_cleanly(requests, result.columns, cfg, params)


@pytest.mark.parametrize("bound", [0.0, 1e-7])
def test_gpu_residual_bound_is_separate_from_column_score(
    monkeypatch: pytest.MonkeyPatch, bound: float,
) -> None:
    """A positive residual bound preserves the global gap without inventing a useful route.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): replaces device execution with a bounded result.
    - bound (float): certified residual pricing upper bound above a non-improving route.

    Return
    --------
    - output (None): verifies bound propagation, insertion gating and termination semantics.
    """
    cfg = _cfg(max_ground_delay_s=0.0)
    request = _request(1, (0, 0), (2, 0), cfg)
    params = _params(
        M=100.0, pricing_device="gpu", pricing_gpu_backend="metal",
        max_iterations=2, seed_ladder_steps=0, ip_gap=0.0, lp_gap=0.0,
    )
    observations = []

    def batch(
        platform: PricingPlatform, problems: Sequence[PricingProblem],
        deadline: float | None = None,
    ) -> tuple[PricingResult, ...]:
        """Supply a certified bound without an improving incumbent score."""
        return tuple(
            PricingResult(
                problem.graph.request.flight_id, -1e-8, seed_column(problem.graph, cfg),
                bound, True, True, "metal", True, "complete",
            )
            for problem in problems
        )

    monkeypatch.setattr(PricingPlatform, "price_batch", batch)
    result = ColGenSolver().solve([request], cfg, (), params, on_iteration=observations.append)
    observation, = observations
    assert observation["upper_bound"] == observation["lp_objective"] + bound
    assert observation["pricing_bound_certified"]
    assert observation["columns_added"] == 0
    assert observation["rc_n_positive"] == 0
    assert result.stats["pricing_certified_columns"] == 0
    assert math.isfinite(result.stats["upper_bound"])
    assert result.stats["termination_reason"] == ("pricing_bound_gap" if bound else "lp_gap")
    assert result.stats["ip_gap_met"] is (bound == 0.0)


@pytest.mark.parametrize("backend", ["highs", "gurobi"])
@pytest.mark.parametrize("cpu_recompute,integer_bounds", [(True, False), (False, False), (True, True)])
def test_full_solver_uses_metal_and_preserves_certificate_meaning(
    monkeypatch: pytest.MonkeyPatch, backend: str, cpu_recompute: bool, integer_bounds: bool,
) -> None:
    """Real GPU sweeps feed a feasible master/IP solve without giving FP32 a global proof.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): disables full CPU pricing after building the oracle.
    - backend (str): LP/IP implementation exercised with actual Metal pricing.
    - cpu_recompute (bool): selects certified or unchecked FP32 GPU search.
    - integer_bounds (bool): use checked GPU deletion bounds with CPU canonical completions.

    Return
    --------
    - output (None): asserts feasibility, GPU execution and correct certificate semantics.
    """
    _platform("metal")
    if backend == "gurobi":
        pytest.importorskip("gurobipy")
        monkeypatch.setenv("COLGEN_GUROBI_THREADS", "1")
    cfg = _cfg(
        flight_levels_m=(30.0,),
        max_ground_delay_s=48.0,
        time_buffer_s=0.0,
        max_detour_factor=5.0,
    )
    requests = [_request(i, (0, 0), (3, 0), cfg) for i in (1, 2)]
    params = _params(
        solver=backend,
        max_air_overrun_hops=1,
        max_iterations=8,
        seed_ladder_steps=0,
        iteration_ip_time_limit_s=1.0,
        ip_time_limit_s=2.0,
        ip_reserve_s=3.0,
        ip_gap=0.0,
        lp_gap=0.0,
    )
    expected = ColGenSolver().solve(requests, cfg, (), params) if cpu_recompute else None
    monkeypatch.setattr(solver, "price_flight", _no_cpu_pricing)
    monkeypatch.setattr(pricing_pool, "price_flight", _no_cpu_pricing)
    observations = []
    gpu_params = replace(
        params,
        pricing_device="gpu",
        pricing_gpu_backend="metal",
        pricing_cpu_recompute=cpu_recompute,
        pricing_gpu_integer_bounds=integer_bounds,
        pricing_gpu_batch_size=2,
        n_pricing_workers=2,
    )
    if integer_bounds:
        from freespace_sim.planner.colgen.gpu_pricing import _Batch

        monkeypatch.setattr(_Batch, "bound", _no_cpu_pricing)
    result = ColGenSolver().solve(requests, cfg, (), gpu_params, on_iteration=observations.append)
    stats = result.stats
    assert stats["backend"] == backend
    assert stats["pricing_device"] == "gpu" and stats["pricing_gpu_backend"] == "metal"
    assert stats["pricing_cpu_recompute"] is cpu_recompute
    assert stats["gpu_dispatches"] > 0 and stats["gpu_batches"] > 0
    assert stats["pricing_search_complete"]
    assert stats["n_pricing_workers"] == 0
    assert stats["iteration_ip_calls"] > 0
    assert set(result.columns) == {1, 2}
    _assert_claim_feasible(result.columns)
    _assert_files_cleanly(requests, result.columns, cfg, gpu_params)
    assert observations
    if cpu_recompute:
        assert stats["pricing_bound_certified"]
        assert stats["exact_pricing_sweeps"] > 0
        assert stats["objective"] == pytest.approx(expected.stats["objective"], abs=1e-7)
        assert all(p["pricing_exact"] for p in observations)
    else:
        assert not stats["pricing_bound_certified"]
        assert stats["exact_pricing_sweeps"] == 0
        assert stats["approximate_pricing_sweeps"] > 0
        assert math.isinf(stats["upper_bound"])
        assert math.isinf(stats["lp_gap"])
        assert not stats["ip_gap_met"]
        assert stats["termination_reason"] in {"approximate_no_new_columns", "iteration_limit"}
        assert all(not p["pricing_exact"] for p in observations)


def test_gpu_exhaustion_preserves_earlier_bound_and_valid_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete GPU sweep preserves an earlier valid bound without tightening it.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): interrupts the second GPU sweep and forbids CPU fallback.

    Return
    --------
    - output (None): verifies incumbent retention and conservative denial classification.
    """
    _platform("metal")
    cfg = _cfg(
        flight_levels_m=(30.0,),
        max_ground_delay_s=0.0,
        time_buffer_s=0.0,
        max_detour_factor=5.0,
    )
    requests = [_request(i, (0, 0), (3, 0), cfg) for i in (1, 2)]
    params = _params(
        max_air_overrun_hops=1,
        max_iterations=2,
        seed_ladder_steps=0,
        seed_nominal_routes=False,
        ip_gap=0.0,
        lp_gap=0.0,
        pricing_device="gpu",
        pricing_gpu_backend="metal",
        pricing_gpu_batch_size=2,
    )
    graph = build_flight_graph(requests[0], cfg, (), params)
    seed = seed_column(graph, cfg)
    actual_price_batch = PricingPlatform.price_batch
    improving_calls = 0

    def interrupt_second_sweep(
        platform: PricingPlatform, problems: Sequence[PricingProblem],
        deadline: float | None = None,
    ) -> tuple[PricingResult, ...]:
        """Keep real GPU results but mark the second improving sweep incomplete."""
        nonlocal improving_calls
        results = actual_price_batch(platform, problems, deadline=deadline)
        if all(problem.require_improving for problem in problems):
            improving_calls += 1
            if improving_calls == 2:
                return tuple(replace(result, search_complete=False, certified=False,
                                     upper_bound=None, status="label_limit") for result in results)
        return results

    monkeypatch.setattr(PricingPlatform, "price_batch", interrupt_second_sweep)
    monkeypatch.setattr(solver, "price_flight", _no_cpu_pricing)
    monkeypatch.setattr(pricing_pool, "price_flight", _no_cpu_pricing)
    observations = []
    result = ColGenSolver().solve(requests, cfg, (), params, seed_columns={1: [seed]},
                                  on_iteration=observations.append)
    stats = result.stats
    assert improving_calls == 2
    assert stats["termination_reason"] == "pricing_label_limit"
    assert not stats["pricing_search_complete"] and stats["pricing_bound_certified"]
    assert observations[0]["pricing_exact"] and observations[0]["pricing_search_complete"]
    assert not observations[-1]["pricing_exact"] and not observations[-1]["pricing_search_complete"]
    assert stats["upper_bound"] == observations[0]["upper_bound"] == observations[-1]["upper_bound"]
    assert math.isfinite(stats["upper_bound"]) and not stats["ip_gap_met"]
    # Only flight 1 entered in the supplied pool; flight 2's valid first-sweep result remains.
    assert stats["seeded_columns"] == 1 and stats["n_columns"] >= 2
    assert stats["pricing_certified_columns"] + stats["pricing_revalidated_columns"] > 0
    assert len(stats["denied_flight_ids"]) == 1
    assert stats["search_exhausted_flight_ids"] == stats["denied_flight_ids"]
    assert stats["budget_denied_flight_ids"] == ()
    _assert_claim_feasible(result.columns)
    _assert_files_cleanly(requests, result.columns, cfg, params)


@pytest.mark.parametrize("repair_status", ["complete", "deadline"])
def test_gpu_repair_can_select_a_feasible_negative_reduced_cost_route(
    monkeypatch: pytest.MonkeyPatch, repair_status: str
) -> None:
    """GPU repair honors feasibility requests even when their route cost exceeds benefit.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): observes GPU repair and forbids CPU pricing fallback.
    - repair_status (str): retains the completed result or simulates an incomplete search.

    Return
    --------
    - output (None): verifies that negative reduced cost or a deadline cannot lose a repair.
    """
    _platform("metal")
    cfg = _cfg(max_ground_delay_s=32.0)
    request = _request(7, (-4, 0), (4, 0), cfg)
    fixed = frozenset({RowKey.cell((0, 0), 0, 10)})
    params = _params(M=1.0, pricing_device="gpu", pricing_gpu_backend="metal")
    actual_price_batch = PricingPlatform.price_batch
    repair_scores = []

    def observe_batch(
        platform: PricingPlatform,
        problems: Sequence[PricingProblem],
        deadline: float | None = None,
    ) -> tuple[PricingResult, ...]:
        """Record real returned repair routes while preserving physical GPU execution."""
        results = actual_price_batch(platform, problems, deadline=deadline)
        repair_scores.extend(
            result.reduced_cost
            for problem, result in zip(problems, results, strict=True)
            if not problem.require_improving and result.column is not None
        )
        if repair_status == "deadline":
            results = tuple(
                replace(
                    result,
                    status="deadline",
                    search_complete=False,
                    certified=False,
                    upper_bound=None,
                )
                if not problem.require_improving
                else result
                for problem, result in zip(problems, results, strict=True)
            )
        return results

    monkeypatch.setattr(PricingPlatform, "price_batch", observe_batch)
    monkeypatch.setattr(solver, "price_flight", _no_cpu_pricing)
    monkeypatch.setattr(pricing_pool, "price_flight", _no_cpu_pricing)
    result = ColGenSolver().solve([request], cfg, (), params, fixed_claims=(fixed,))
    assert result.stats["denied_flight_ids"] == ()
    assert result.stats["repair_added"] == 1
    assert result.stats["gpu_dispatches"] > 0
    assert repair_scores and max(repair_scores) < 0.0
    assert result.columns[7].delay_s == pytest.approx(16.0)
    assert result.columns[7].claims.isdisjoint(fixed)
    if repair_status == "deadline":
        assert result.stats["termination_reason"] == "time_limit"
        assert result.stats["pricing_failure_status"] == "deadline"
    _assert_files_cleanly([request], result.columns, cfg, params)
