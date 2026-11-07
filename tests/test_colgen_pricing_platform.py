"""Functional contracts for selectable CPU and certified Metal pricing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import math
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.gpu_runtime import GPUUnavailable
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph, column_claims
from freespace_sim.planner.colgen.objective import cost_model
from freespace_sim.planner.colgen.pricing_platform import (
    PricingOptions,
    PricingPlatform,
    PricingProblem,
    PricingResult,
)
from freespace_sim.planner.colgen.translate import column_to_intent
from tests.test_colgen_review_fixes import _enumerated_best_rc
from tests.test_colgen_solver import _cfg, _params, _request


def _problem(
    *, buffer: float = 4.0, objective: str = "total_cost", signed: bool = False
) -> PricingProblem:
    """
    Construct a small pricing domain whose complete paths can be enumerated independently.

    Parameters
    ------------
    - buffer (float): occupancy time buffer in seconds.
    - objective (str): weighted cost or unweighted delay objective.
    - signed (bool): include negative capacity prices as well as positive prices.

    Return
    --------
    - output (PricingProblem): a three-hop trip with two extra hops and eight seconds of delay.
    """
    cfg = _cfg(
        flight_levels_m=(30.0,),
        max_ground_delay_s=8.0,
        time_buffer_s=buffer,
        cost_ground_delay_per_s=20.0,
        cost_air_hold_per_s=3.0,
        cost_air_lateral_per_s=3.0,
        max_detour_factor=5.0,
    )
    params = _params(max_air_overrun_hops=2, objective=objective)
    graph = build_flight_graph(_request(1, (0, 0), (3, 0), cfg), cfg, (), params)
    rng = random.Random(19)
    prices = (-50.0, -12.0, 0.0, 12.0, 50.0) if signed else (0.0, 0.0, 12.0, 50.0, 200.0)
    duals = {
        RowKey.cell(cell, 0, step): rng.choice(prices)
        for cell in sorted(graph.corridor_cells)
        for step in range(graph.max_step + 6)
    }
    return PricingProblem(graph=graph, cfg=cfg, params=params, duals=duals, pi_f=0.0)


def _platform(
    backend: str,
    *,
    cpu_recompute: bool = True,
    max_labels: int = 250_000,
    chunk_size: int = 2048,
) -> PricingPlatform:
    """
    Select a backend, skipping Metal tests only when its platform or hardware is absent.

    Parameters
    ------------
    - backend (str): cpu or metal.
    - cpu_recompute (bool): enable certified numerical recomputation for GPU search.
    - max_labels (int): storage budget used to test incomplete-search handling.
    - chunk_size (int): expansion chunk size used by GPU search.

    Return
    --------
    - output (PricingPlatform): actual pricing backend; compiler failures remain test failures.
    """
    if backend == "metal":
        if sys.platform != "darwin":
            pytest.skip("Metal requires macOS")
        metal = pytest.importorskip(
            "Metal", reason="Metal tests require the optional PyObjC bridge"
        )
        if metal.MTLCreateSystemDefaultDevice() is None:
            pytest.skip("No Metal GPU is available")
    return PricingPlatform(
        PricingOptions(
            device="cpu" if backend == "cpu" else "gpu",
            gpu_backend=None if backend == "cpu" else backend,
            cpu_recompute=cpu_recompute,
            max_labels=max_labels,
            chunk_size=chunk_size,
        )
    )


def _oracle(problem: PricingProblem) -> float:
    """Return the best reduced cost by independent enumeration of this small bare graph."""
    return (
        _enumerated_best_rc(
            problem.graph,
            problem.cfg,
            cost_model(problem.cfg, problem.params),
            problem.duals,
            problem.params.M,
        )
        - problem.pi_f
    )


def _assert_canonical(problem: PricingProblem, result: PricingResult) -> None:
    """
    Check returned geometry, capacity claims, objective cost, and reduced cost independently.

    Parameters
    ------------
    - problem (PricingProblem): source graph, prices, and objective.
    - result (PricingResult): result required to contain a returned route.

    Return
    --------
    - output (None): assertions fail if the returned route or its coefficients are inconsistent.
    """
    column = result.column
    assert column is not None
    assert result.flight_id == column.flight_id == problem.graph.request.flight_id
    assert column.claims == column_claims(column, problem.graph, problem.cfg)
    intent = column_to_intent(column, problem.graph.request, problem.cfg)
    model = cost_model(problem.cfg, problem.params)
    actual_cost = model.intent_cost(intent, problem.cfg)
    assert column.delay_s == pytest.approx(actual_cost, rel=0.0, abs=1e-9)
    view = (problem.duals if isinstance(problem.duals, pricing.DualView)
            else pricing.DualView(problem.duals, problem.cfg))
    actual_rc = model.reduced_cost(
        benefit=problem.params.M,
        cost=actual_cost,
        dual_cost=view.claim_cost(column.claims),
        pi_f=problem.pi_f,
    )
    assert result.reduced_cost == pytest.approx(actual_rc, rel=0.0, abs=1e-9)


def test_cpu_adapter_preserves_existing_pricing_result() -> None:
    """Selecting CPU retains the existing solver's route and reduced cost."""
    problem = _problem()
    expected = pricing.price_flight(
        problem.graph,
        problem.duals,
        problem.pi_f,
        problem.cfg,
        problem.params,
        known_column=problem.known_column,
        forbidden_rows=problem.forbidden_rows,
    )
    (result,) = _platform("cpu").price_batch([problem])
    assert (result.reduced_cost, result.column) == expected
    assert result.search_complete and result.certified
    assert result.backend == "cpu"
    _assert_canonical(problem, result)


@pytest.mark.parametrize("backend,first_limit", [("cpu", None), ("metal", None), ("metal", 4)])
@pytest.mark.parametrize("buffer", [0.0, 4.0])
@pytest.mark.parametrize("objective", ["total_cost", "total_delay"])
@pytest.mark.parametrize("signed", [False, True])
def test_certified_pricing_matches_complete_path_enumeration(
    backend: str, first_limit: int | None, buffer: float, objective: str, signed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete pricing, including an early future retry, covers every canonical route."""
    if first_limit is not None:
        from freespace_sim.planner.colgen import gpu_pricing

        monkeypatch.setattr(gpu_pricing, "_INITIAL_FULL_LABEL_LIMIT", first_limit)
    platform = _platform(backend)
    original = _problem(buffer=buffer, objective=objective, signed=signed)
    expected = _oracle(original)
    problem = replace(original, pi_f=expected - 0.125)
    (result,) = platform.price_batch([problem])
    assert result.search_complete and result.certified
    assert result.backend == backend
    if backend == "metal":
        assert result.stats["gpu_dispatches"] > 0
    assert result.reduced_cost == pytest.approx(0.125, rel=0.0, abs=1e-8)
    assert result.upper_bound is not None
    assert result.upper_bound == pytest.approx(0.125, rel=0.0, abs=1e-8)
    _assert_canonical(problem, result)


@pytest.mark.parametrize("backend", ["cpu", "metal"])
def test_certificate_can_establish_no_improving_route(backend: str) -> None:
    """A complete search can certify absence of improvement at a new assignment price."""
    platform = _platform(backend)
    original = _problem(signed=True)
    problem = replace(original, pi_f=_oracle(original) + 0.125)
    (result,) = platform.price_batch([problem])
    assert result.search_complete and result.certified
    if backend == "metal":
        assert result.stats["gpu_dispatches"] > 0
    assert result.column is None
    assert result.upper_bound is not None
    assert -0.125 - 1e-8 <= result.upper_bound <= 1e-8


def test_gpu_fp32_mode_keeps_validation_but_never_certifies() -> None:
    """Unchecked GPU search improves a priced seed before future preparation without certifying."""
    platform = _platform("metal", cpu_recompute=False)
    original = _problem(buffer=0.0)
    model = cost_model(original.cfg, original.params)
    seed = pricing.seed_column(original.graph, original.cfg, model=model)
    priced_row = next(row for row in sorted(seed.claims) if row[1:3] == (2, 0))
    problem = replace(original, duals={priced_row: 1000.0}, known_column=seed)
    seed_rc = model.reduced_cost(
        benefit=problem.params.M, cost=seed.delay_s, dual_cost=1000.0, pi_f=problem.pi_f,
    )
    (result,) = platform.price_batch([problem])
    assert result.search_complete
    assert result.stats["gpu_dispatches"] > 0
    assert result.stats.get("future_gpu_dispatches", 0) == 0
    assert result.reduced_cost > seed_rc + 100.0
    assert result.column != seed
    assert priced_row not in result.column.claims
    assert not result.certified and not result.cpu_recompute
    assert result.upper_bound is None
    _assert_canonical(problem, result)


def test_cpu_recomputation_recovers_improvement_hidden_by_fp32(monkeypatch) -> None:
    """A rounded-away improving route survives refinement without rerunning CPU pricing."""
    exact = _platform("metal")
    approximate = _platform("metal", cpu_recompute=False)
    cfg = _cfg(
        flight_levels_m=(30.0,),
        max_ground_delay_s=0.0,
        time_buffer_s=0.0,
        cost_ground_delay_per_s=1e-10,
        cost_air_hold_per_s=1e-10,
        cost_air_lateral_per_s=1e-10,
        max_detour_factor=5.0,
    )
    params = _params(max_air_overrun_hops=2, objective="total_cost")
    graph = build_flight_graph(_request(1, (0, 0), (3, 0), cfg), cfg, (), params)
    problem = PricingProblem(graph, cfg, params, {RowKey.cell((2, 0), 0, 3): 1e-5}, 9999.99998)
    expected = _oracle(problem)

    def no_cpu_search(*args, **kwargs):
        raise AssertionError("GPU certification must not rerun CPU pricing")

    monkeypatch.setattr(pricing, "price_flight", no_cpu_search)
    monkeypatch.setattr(pricing, "_best_column", no_cpu_search)
    monkeypatch.setattr(pricing, "_best_column_compiled", no_cpu_search)
    (certified,) = exact.price_batch([problem])
    (fp32,) = approximate.price_batch([problem])
    assert certified.certified and certified.stats["cpu_recomputations"] > 0
    assert certified.stats["gpu_dispatches"] > 0
    assert certified.reduced_cost == pytest.approx(expected, rel=0, abs=1e-9)
    assert certified.reduced_cost > fp32.reduced_cost + 5e-6
    assert not fp32.certified and fp32.upper_bound is None
    _assert_canonical(problem, certified)
    _assert_canonical(problem, fp32)


@pytest.mark.parametrize("chunk_size", [1, 2048])
@pytest.mark.parametrize("prepare_future", [False, True])
@pytest.mark.parametrize("share_prices", [False, True])
def test_gpu_batches_and_reuses_runtime_without_mixing_prices(
    monkeypatch: pytest.MonkeyPatch, chunk_size: int, prepare_future: bool, share_prices: bool,
) -> None:
    """Flights with different time windows preserve independent optima across batch reuse.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): optionally prepares the future table before search.
    - chunk_size (int): expansion chunk size, including split single-label layers.
    - prepare_future (bool): exercises stronger bounds even when this small case fits in memory.
    - share_prices (bool): reuse one immutable LP snapshot across flights and separate batches.

    Return
    --------
    - output (None): compares independent enumerated scores and canonical routes in both orders.
    """
    from freespace_sim.planner.colgen.gpu_pricing import _Batch

    if prepare_future:
        actual_prepare = _Batch.prepare

        def prepare_with_future(batch: _Batch) -> None:
            """Force the optional future table without depending on an incidental label cap."""
            actual_prepare(batch)
            batch.prepare_future()

        monkeypatch.setattr(_Batch, "prepare", prepare_with_future)
    platform = _platform("metal", chunk_size=chunk_size)
    first = _problem(signed=True)
    second = _problem(signed=False)
    graph = build_flight_graph(
        _request(2, (0, 0), (3, 0), second.cfg, departure=32.0), second.cfg, (), second.params
    )
    second = replace(second, graph=graph, duals={
        RowKey((*row[:-1], row[-1] + graph.base_step)): price
        for row, price in second.duals.items()
    })
    assert first.graph.base_step != second.graph.base_step
    assert first.graph.max_step != second.graph.max_step
    if share_prices:
        duals = first.duals | second.duals
        first, second = replace(first, duals=duals), replace(second, duals=duals)
    expected = [_oracle(p) for p in (first, second)]
    if share_prices:
        view = pricing.DualView(duals, first.cfg)
        first, second = replace(first, duals=view), replace(second, duals=view)
    for problems in ((first, second), (second, first), (first,), (second,)):
        results = platform.price_batch(problems)
        for problem, result in zip(problems, results, strict=True):
            assert result.certified and result.search_complete
            assert result.reduced_cost == pytest.approx(expected[result.flight_id - 1], abs=1e-8)
            assert result.upper_bound >= expected[result.flight_id - 1] - 1e-9
            if prepare_future:
                assert result.stats["future_gpu_dispatches"] > 0
            _assert_canonical(problem, result)

    if share_prices:
        # Reordering flights and starting another batch must retain the same device table.
        assert platform._runtime.counters["price_uploads"] == 1
        assert platform._runtime.counters["price_cache_hits"] == 3

    original_prices = (
        first.duals._duals if isinstance(first.duals, pricing.DualView) else first.duals
    )
    changed_prices = {
        row: -price * 0.5 for index, (row, price) in enumerate(original_prices.items())
        if index % 3
    }
    # Endpoint reservations include rows outside the original cruise-only snapshot.
    new_rows = pricing.seed_column(first.graph, first.cfg).claims - original_prices.keys()
    assert new_rows
    changed_prices[next(iter(sorted(new_rows)))] = 37.0
    changed = replace(first, duals=changed_prices, pi_f=17.0)
    changed_expected = _oracle(changed)
    changed = replace(changed, duals=pricing.DualView(changed_prices, first.cfg))
    for problem, optimum in ((changed, changed_expected), (first, expected[0])):
        (result,) = platform.price_batch([problem])
        assert result.certified and result.search_complete
        assert result.reduced_cost == pytest.approx(optimum, abs=1e-8)
        assert result.upper_bound >= optimum - 1e-9
        _assert_canonical(problem, result)

    if share_prices:
        assert platform._runtime.counters["price_uploads"] == 2
        assert len(platform._runtime.prices.stores) == 2


@pytest.mark.parametrize("buffer", [0.0, 4.0, 12.0])
def test_shared_visit_cache_preserves_prices_restrictions_and_new_snapshots(
    monkeypatch: pytest.MonkeyPatch, buffer: float,
) -> None:
    """One cache serves different graphs without sharing restrictions or stale LP prices.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): counts complete shared-cache constructions.
    - buffer (float): changes the inclusive occupancy window.

    Return
    --------
    - output (None): checks exact visit inputs, one build per snapshot and live future bounds.
    """
    from freespace_sim.planner.colgen import gpu_pricing
    from freespace_sim.planner.colgen.gpu_prices import SparseVisitPrices

    cfg = _cfg(time_buffer_s=buffer)
    prices = {
        RowKey.cell((0, 0), 0, -1): -0.0,
        RowKey.cell((0, 0), 0, 0): 1e20,
        RowKey.cell((0, 0), 0, 1): 3.25,
        RowKey.cell((0, 0), 0, 2): -1e20,
        RowKey.cell((0, 0), 0, 3): -0.125,
        RowKey.cell((1, 0), 0, 1): 7.0,
        RowKey.cell((0, 0), 1, 1): 900.0,
        RowKey.term("terminal", 1): 800.0,
        RowKey.cell((99, 99), 0, 1): 23.0,
    }
    view = pricing.DualView(prices, cfg)
    builds = []

    def counted(snapshot: pricing.DualView, check: Callable[[], None]) -> SparseVisitPrices:
        """Observe the required single construction without changing price arithmetic."""
        builds.append(snapshot)
        return SparseVisitPrices(snapshot, check)

    monkeypatch.setattr(gpu_pricing, "SparseVisitPrices", counted)

    def packed(snapshot: pricing.DualView, cells: list[tuple[int, int]]) -> gpu_pricing._Batch:
        """Supply a tiny prepared batch so price gathering can be tested without a GPU."""
        batch = object.__new__(gpu_pricing._Batch)
        batch.deadline = None
        batch.cells, batch.cell_offsets = cells, list(range(len(cells)))
        batch.views = [snapshot] * len(cells)
        batch.offsets = snapshot.offsets
        batch.topologies = [SimpleNamespace(n_cells=1) for _ in cells]
        batch.problems = [SimpleNamespace(forbidden_rows=frozenset([
            RowKey.cell(cells[0], 0, 1), RowKey.term("terminal", 0),
            RowKey.cell(cells[0], 1, 0),
        ]))] + [SimpleNamespace(forbidden_rows=frozenset()) for _ in cells[1:]]
        batch.future = [None] * len(cells)
        batch._visit_layers = None
        batch._forbidden_layers = {}
        return batch

    def verify(batch: gpu_pricing._Batch, step: int) -> None:
        """Compare packed GPU inputs against direct canonical visit-row enumeration."""
        expected = np.zeros((len(batch.cells), 3), np.float32)
        expected[:, 2] = math.inf
        blocked = []
        for i, cell in enumerate(batch.cells):
            claims = pricing._visit_claims(cell, 0, step, batch.offsets)
            values = [batch.views[i].row_cost(row) for row in claims]
            expected[i, :2] = math.fsum(values), math.fsum(max(0.0, p) for p in values)
            blocked.append(not claims.isdisjoint(batch.problems[i].forbidden_rows))
            future = batch.future[i]
            if future is not None and future.first_step <= step <= future.last_step:
                expected[i, 2] = future.row(step)[0]
        actual, actual_blocked = batch.visits(step)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(actual_blocked, blocked)
        actual[:] = 999.0
        actual_blocked[:] = 1
        again, again_blocked = batch.visits(step)
        np.testing.assert_array_equal(again.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(again_blocked, blocked)

    batch = packed(view, [(0, 0), (0, 0), (1, 0)])
    for step in range(-10, 13):
        verify(batch, step)
    batch.future[1] = SimpleNamespace(first_step=0, last_step=4,
                                     row=lambda step: np.array([step + 0.25]))
    for step in range(-1, 6):
        verify(batch, step)
    verify(packed(view, [(99, 99)]), 1)
    assert builds == [view]
    prices[RowKey.cell((99, 99), 0, 1)] = -41.0
    verify(packed(view, [(99, 99)]), 1)
    new_view = pricing.DualView(prices, cfg)
    verify(packed(new_view, [(99, 99)]), 1)
    assert builds == [view, new_view]


def test_raw_price_mutations_and_interrupted_cache_builds_do_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw maps refresh on each call, and a deadline cannot publish a partial cache.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): interrupts shared-cache construction deterministically.

    Return
    --------
    - output (None): preserves snapshot isolation and retries incomplete preparation safely.
    """
    from freespace_sim.planner.colgen.gpu_pricing import _Batch, _Stopped

    problem = _problem()
    runtime = SimpleNamespace(dispatches=0)
    batch = _Batch([problem, problem], PricingOptions(), runtime, None)
    assert batch.views[0] is batch.views[1]
    row = next(iter(problem.duals))
    previous = batch.views[0].row_cost(row)
    problem.duals[row] = previous + 13.0
    next_batch = _Batch([problem], PricingOptions(), runtime, None)
    assert next_batch.views[0].row_cost(row) == previous + 13.0
    assert batch.views[0].row_cost(row) == previous
    batch.cells = [(0, 0)]
    batch.cell_offsets = [0, 1]
    batch.topologies = [SimpleNamespace(n_cells=1), SimpleNamespace(n_cells=0)]
    checks = 0

    def expire() -> None:
        """Expire inside the cache builder after initial batch preparation checks."""
        nonlocal checks
        checks += 1
        if checks == 3:
            raise _Stopped("deadline")

    with monkeypatch.context() as scoped:
        scoped.setattr(batch, "check", expire)
        with pytest.raises(_Stopped, match="deadline"):
            batch.visits(1)
    assert batch.views[0]._gpu_visit_prices is None
    assert batch._visit_layers is None
    batch.visits(1)
    assert batch.views[0]._gpu_visit_prices is not None


def test_gpu_rejects_dual_view_with_mismatched_occupancy_window() -> None:
    """Reusing prices prepared for different visit rows cannot produce a false certificate."""
    platform = _platform("metal")
    problem = _problem(buffer=0.0)
    wrong = pricing.DualView(problem.duals, replace(problem.cfg, time_buffer_s=12.0))
    with pytest.raises(ValueError, match="offsets"):
        platform.price_batch([replace(problem, duals=wrong)])


@pytest.mark.parametrize("exclude_arrival", [False, True])
def test_gpu_terminal_lanes_walls_and_forbidden_arrivals(
    monkeypatch: pytest.MonkeyPatch, exclude_arrival: bool,
) -> None:
    """Terminal lane geometry, wall roles and arrival exclusions preserve the optimum.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): installs future bounds before the small GPU search.
    - exclude_arrival (bool): forbids the arrival's mandatory terminal resource.

    Return
    --------
    - output (None): GPU completion and its upper bound agree with the reference domain.
    """
    from tests.test_colgen_dp_kernel import _cfg as terminal_cfg, _terminal_graph
    from freespace_sim.planner.colgen.gpu_pricing import _Batch

    bootstrap = _Batch.bootstrap

    def with_future(batch: _Batch) -> None:
        """Exercise hop-aware verification with actual terminal roles and obstacles."""
        batch.prepare_future()
        bootstrap(batch)

    monkeypatch.setattr(_Batch, "bootstrap", with_future)

    platform = _platform("metal")
    cfg = terminal_cfg(max_ground_delay_s=4.0)
    graph, params = _terminal_graph(cfg, overrun=0)
    arrival = RowKey.term("kern-B", 15)
    forbidden = frozenset((arrival,)) if exclude_arrival else frozenset()
    problem = PricingProblem(
        graph,
        cfg,
        params,
        {arrival: -12.0, RowKey.cell((2, 0), 0, 7): 100.0},
        0.0,
        forbidden_rows=forbidden,
    )
    (expected,) = _platform("cpu").price_batch([problem])
    (result,) = platform.price_batch([problem])
    assert result.certified and result.stats["gpu_dispatches"] > 0
    assert result.reduced_cost == pytest.approx(expected.reduced_cost, abs=1e-8)
    if exclude_arrival:
        assert result.column is None and result.upper_bound == -float("inf")
    else:
        _assert_canonical(problem, result)


@pytest.mark.parametrize("backend", ["cpu", "metal"])
def test_expired_deadline_never_certifies(backend: str) -> None:
    """No backend can report a complete certificate after receiving an expired deadline."""
    platform = _platform(backend)
    problem = _problem()
    (result,) = platform.price_batch([problem], deadline=time.monotonic() - 1.0)
    assert not result.search_complete and not result.certified
    assert result.upper_bound is None or result.upper_bound >= _oracle(problem) - 1e-8


def test_gpu_label_limit_preserves_incomplete_status() -> None:
    """Exhausting label storage does not become a no-improvement certificate."""
    platform = _platform("metal", max_labels=1, chunk_size=1)
    problem = _problem(signed=True)
    (result,) = platform.price_batch([problem])
    assert not result.search_complete and not result.certified
    assert result.upper_bound is None or result.upper_bound >= _oracle(problem) - 1e-8
    if result.column is not None:
        _assert_canonical(problem, result)


def test_long_diagonal_flight_closes_with_a_certified_cost_bound(monkeypatch) -> None:
    """A many-path long flight must finish within a small storage budget using safe pruning."""
    cfg = _cfg(flight_levels_m=(30.0,), max_ground_delay_s=120.0,
               time_buffer_s=4.0, max_detour_factor=5.0, cost_ground_delay_per_s=10.0)
    params = _params(max_air_overrun_hops=6, objective="total_cost")
    graph = build_flight_graph(_request(9, (85, -30), (165, 10), cfg), cfg, (), params)
    problem = PricingProblem(graph, cfg, params, {}, 0.0)
    seed = pricing._shortest_seed(graph, cfg, model=cost_model(cfg, params))

    def no_cpu_search(*args, **kwargs):
        raise AssertionError("A long GPU search must not rerun CPU pricing")

    monkeypatch.setattr(pricing, "price_flight", no_cpu_search)
    monkeypatch.setattr(pricing, "_best_column", no_cpu_search)
    monkeypatch.setattr(pricing, "_best_column_compiled", no_cpu_search)
    (result,) = _platform("metal", max_labels=1000).price_batch([problem])
    assert result.search_complete and result.certified
    assert result.stats["gpu_dispatches"] > 0 and result.stats["root_prunes"] > 0
    assert result.stats["screened_departures"] > 0
    expected = params.M - seed.delay_s
    assert result.reduced_cost == pytest.approx(expected, rel=0.0, abs=1e-7)
    assert result.upper_bound >= result.reduced_cost
    assert result.upper_bound - result.reduced_cost <= 1.01e-7
    _assert_canonical(problem, result)


@pytest.mark.parametrize("improvement", [5e-8, 5e-7])
def test_near_tie_certificate_covers_improvement_even_when_no_column_is_returned(
    monkeypatch: pytest.MonkeyPatch, improvement: float,
) -> None:
    """Closing tiny branches reports their upper bound instead of claiming an exact optimum.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): installs remaining-hop bounds before route discovery.
    - improvement (float): canonical gain deliberately smaller than the closure tolerance.

    Return
    --------
    - output (None): the certificate covers the improvement even when no column is returned.
    """
    from freespace_sim.planner.colgen.gpu_pricing import _Batch

    bootstrap = _Batch.bootstrap

    def with_future(batch: _Batch) -> None:
        """Keep the same near-tie requirement when remaining-hop bounds are installed."""
        batch.prepare_future()
        bootstrap(batch)

    monkeypatch.setattr(_Batch, "bootstrap", with_future)
    cfg = _cfg(flight_levels_m=(30.0,), max_ground_delay_s=0.0, time_buffer_s=0.0,
               cost_ground_delay_per_s=1e-12, cost_air_hold_per_s=1e-12,
               cost_air_lateral_per_s=1e-12, max_detour_factor=5.0)
    params = _params(max_air_overrun_hops=2, objective="total_cost")
    graph = build_flight_graph(_request(1, (0, 0), (3, 0), cfg), cfg, (), params)
    problem = PricingProblem(graph, cfg, params, {RowKey.cell((2, 0), 0, 3): improvement},
                             params.M - improvement)
    expected = _oracle(problem)
    (result,) = _platform("metal").price_batch([problem])
    assert result.search_complete and result.certified
    assert result.column is None and result.reduced_cost < expected
    assert result.upper_bound >= expected > pricing._IMPROVING_RC_TOL
    assert result.upper_bound - result.reduced_cost <= improvement + 1e-8


@pytest.mark.parametrize(
    "device,backend",
    [
        ("invalid", None),
        ("cpu", "metal"),
        ("gpu", "invalid"),
        ("gpu", None),
    ],
)
def test_invalid_backend_selection_fails_explicitly(device: str, backend: str | None) -> None:
    """Invalid selections cannot silently choose a different implementation."""
    with pytest.raises(ValueError):
        platform = PricingPlatform(PricingOptions(device=device, gpu_backend=backend))
        platform.price_batch([_problem()])


def test_unavailable_cuda_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing CUDA dependency cannot silently select CPU or Metal pricing."""
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(GPUUnavailable, match="[Cc][Uu][Dd][Aa]"):
        platform = PricingPlatform(PricingOptions(device="gpu", gpu_backend="cuda"))
        platform.price_batch([_problem()])


def test_snapshot_summaries_preserve_rounding_and_interrupted_publication():
    """Shared summaries match ordered FP64 bounds, including signed out-of-domain rows."""
    from freespace_sim.planner.colgen.gpu_prices import summary_for

    cfg = _cfg()
    raw = {RowKey.cell((-999, 99), 0, 1): -1e20,
           RowKey.cell((0, 0), 0, 2): -3.25,
           RowKey.term("terminal", -100): 1e20}
    view = pricing.DualView(raw, cfg)
    calls = 0

    def interrupt():
        """Stop before publication rather than retaining partially accumulated bounds."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise pricing.PricingTimeout()

    with pytest.raises(pricing.PricingTimeout):
        summary_for(view, interrupt)
    assert view._gpu_summary is None
    negative = absolute = 0.0
    for price in raw.values():
        if price < 0:
            negative = math.nextafter(negative + price, -math.inf)
        if price:
            absolute = math.nextafter(absolute + abs(price), math.inf)
    result = summary_for(view)
    assert (result.negative, result.absolute) == (negative, absolute)
    assert summary_for(view) is result
    raw[RowKey.term("terminal", -100)] = -7.0
    assert summary_for(view) is result
    assert summary_for(pricing.DualView(raw, cfg)).negative < result.negative
