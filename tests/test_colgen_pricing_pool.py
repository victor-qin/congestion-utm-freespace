"""Parallel pricing must be a pure performance knob.

The whole justification for fanning the pricing sweep across processes is that within one CG
iteration the subproblems are independent: ``solve`` builds the duals once before the sweep and
never mutates them inside it.  If that were ever untrue the failure would be silent -- a
slightly different column set, a slightly different objective -- so these tests pin equality of
the *result*, not just that the pool runs.
"""
from __future__ import annotations

import pytest

from freespace_sim.planner.colgen.pricing_pool import ParallelPricingConfig
from freespace_sim.planner.colgen.solver import ColGenSolver

from .test_colgen_solver import _cfg, _params, _request


def _crossing_requests(cfg):
    """Flights that genuinely contend, so pricing has to resolve a conflict.

    A non-interacting set would price identically under any dispatch order and the parity
    assertion below would pass vacuously.
    """
    return [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
        _request(3, (-4, 4), (4, -4), cfg),
    ]


def test_parallel_pricing_reproduces_the_sequential_result():
    cfg = _cfg(max_ground_delay_s=32.0)
    requests = _crossing_requests(cfg)

    sequential = ColGenSolver().solve(requests, cfg, (), _params())
    parallel = ColGenSolver().solve(
        requests, cfg, (), _params(), parallel=ParallelPricingConfig(n_workers=2)
    )

    # Anti-vacuity: the pool must actually have run, and the flights must actually contend.
    assert parallel.stats["parallel_worker_processes"] >= 1
    assert parallel.stats["parallel_workers"] == 2
    assert sequential.stats["objective"] > 0.0

    assert parallel.stats["objective"] == pytest.approx(
        sequential.stats["objective"], abs=1e-12
    )
    assert set(parallel.columns) == set(sequential.columns)
    for flight_id, column in sequential.columns.items():
        other = parallel.columns[flight_id]
        assert other.cell_path == column.cell_path
        assert other.departure_step == column.departure_step
        assert other.delay_s == pytest.approx(column.delay_s, abs=1e-12)
        assert other.claims == column.claims


def test_worker_recycling_replaces_processes_and_still_matches():
    """``max_tasks_per_child`` is what returns a worker's arena to the OS.

    It also re-runs the initializer, re-ships the duals, and re-warms the kernel, so it is a
    genuinely different code path from the never-recycle case and needs its own parity check.
    """
    cfg = _cfg(max_ground_delay_s=32.0)
    requests = _crossing_requests(cfg)

    baseline = ColGenSolver().solve(requests, cfg, (), _params())
    recycled = ColGenSolver().solve(
        requests, cfg, (), _params(),
        parallel=ParallelPricingConfig(n_workers=2, max_tasks_per_child=1),
    )

    assert recycled.stats["objective"] == pytest.approx(baseline.stats["objective"], abs=1e-12)
    assert set(recycled.columns) == set(baseline.columns)
    # One task per child over several flights and iterations must retire more processes than
    # the pool is wide -- otherwise recycling silently is not happening.
    assert recycled.stats["parallel_worker_processes"] > 2


def test_zero_workers_keeps_the_sequential_path():
    cfg = _cfg(max_ground_delay_s=32.0)
    requests = _crossing_requests(cfg)

    result = ColGenSolver().solve(
        requests, cfg, (), _params(), parallel=ParallelPricingConfig(n_workers=0)
    )

    assert not ParallelPricingConfig(n_workers=0).enabled
    assert result.stats["parallel_worker_processes"] == 0
    assert result.stats["objective"] == pytest.approx(
        ColGenSolver().solve(requests, cfg, (), _params()).stats["objective"], abs=1e-12
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"n_workers": -1}, {"n_workers": 2, "max_tasks_per_child": 0}],
)
def test_invalid_pool_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ParallelPricingConfig(**kwargs)
