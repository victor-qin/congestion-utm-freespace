"""Endpoint masks, original-price range minima and device proposals preserve row unions."""

from dataclasses import replace
from decimal import Decimal, localcontext
import math
from types import SimpleNamespace

import numpy as np
import pytest

from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.gpu_endpoint import EndpointBounds
from freespace_sim.planner.colgen.gpu_runtime import ComputeRuntime
from freespace_sim.planner.colgen.network import RowKey
from freespace_sim.planner.colgen.pricing_platform import PricingOptions
from freespace_sim.planner.colgen.translate import Column
from tests.test_colgen_gpu_bounds import _bounds
from tests.test_colgen_gpu_future import _case
from tests.test_colgen_pricing_platform import _assert_canonical, _oracle, _platform, _problem


def _endpoint_case(monkeypatch: pytest.MonkeyPatch, start: int) -> tuple:
    """Build 70 priced endpoint identities with overlapping roots and visit windows.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): supplies nominal guaranteed endpoint incidence.
    - start (int): absolute clock offset shared by every row and arrival.

    Return
    --------
    - output (tuple): bound inputs, ordered positive rows and independent endpoint membership.
    """
    problem, _, _, _ = _case()
    rows = tuple(RowKey.cell((i, 0), 0, start + i % 4) for i in range(70))
    negative = RowKey.terminal("credit", start)
    prices = {row: 10.0 + (70 - i) * 0.001 for i, row in enumerate(rows)}
    prices[negative] = -100.3
    arrivals = {
        start + offset: frozenset((*rows[:70 - offset], negative)) for offset in range(7)
    }

    def endpoints(graph, cfg, *, origin, step, timing_steps):
        """Only nominal endpoint claims are valid inputs to the geometric cache."""
        assert not origin and timing_steps == 0
        return arrivals[step]

    monkeypatch.setattr(pricing, "_endpoint_claims", endpoints)
    view = pricing.DualView(prices, problem.cfg)
    view._offsets = (-1, 1)
    bounds = SimpleNamespace(
        problem=replace(problem, duals=prices), view=view, air_per_hop=0.3,
        topology=SimpleNamespace(
            max_step=start + 6, cell_q=np.arange(70), cell_r=np.zeros(70),
        ),
        variants=SimpleNamespace(n_variants=2, start_step=np.array([start, start + 2])),
        origin_claims=lambda variant: frozenset((rows[31], rows[63])) if variant == 0 else frozenset(),
    )
    return bounds, rows, arrivals


@pytest.mark.parametrize("start", [-100, 100])
@pytest.mark.parametrize("air", [0.0, 0.3, 2e38, 1e-50])
def test_endpoint_masks_and_all_intervals_cover_exact_original_price_unions(
    monkeypatch: pytest.MonkeyPatch, start: int, air: float,
) -> None:
    """Both words, truncation and minima cover decimal sums over every tiny interval.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): isolates endpoint incidence.
    - start (int): shifts all absolute clocks without changing costs.
    - air (float): includes zero slope and FP32 overflow/underflow while verification stays FP64.

    Return
    --------
    - output (None): validates every Q/H query against original-price Decimal arithmetic.
    """
    bounds, rows, arrivals = _endpoint_case(monkeypatch, start)
    bounds.air_per_hop = air
    endpoint = EndpointBounds(bounds, PricingOptions(), lambda: None)
    assert endpoint.rows == rows[:64]
    assert endpoint.mask((rows[31], rows[63], rows[64])) == (1 << 31, 1 << 31)
    np.testing.assert_array_equal(endpoint.root_masks[0], [1 << 31, 1 << 31])
    for bit in (31, 63):
        indices, masks = endpoint.visit_masks[rows[bit].step]
        words = masks[np.flatnonzero(indices == bit)[0]]
        assert int(words[bit // 32]) & (1 << 31)
    masks = np.array([(0, 0), (1 << 31, 1 << 31), (2**32 - 1, 2**32 - 1), (7, 29)], np.uint32)
    endpoint.consider(masks)
    assert all(tree.dtype == np.float64 for tree in endpoint.entries.values())
    assert any(float(np.float32(w)) != w for w in endpoint.weights)
    gpu = endpoint.gpu_trees()
    original = np.concatenate([tree.ravel() for tree in endpoint.entries.values()])
    assert np.all(gpu.astype(np.float64) <= original)
    with localcontext() as context:
        context.prec = 100
        air = Decimal.from_float(bounds.air_per_hop)
        for low, high in masks:
            key = int(low), int(high)
            paid = {row for bit, row in enumerate(endpoint.rows)
                    if key[bit // 32] & (1 << (bit % 32))}
            for current in range(start, start + 7):
                for first in range(current, start + 7):
                    for last in range(first, start + 7):
                        q, d = endpoint.residual(key, current, first, last)
                        exact = [sum((Decimal.from_float(bounds.view.row_cost(row))
                                      for row in (arrivals[t] & set(endpoint.rows)) - paid), Decimal(0))
                                 for t in range(first, last + 1)]
                        assert Decimal.from_float(q) <= min(exact)
                        assert Decimal.from_float(d) <= min(
                            value + air * (t - current)
                            for t, value in zip(range(first, last + 1), exact, strict=True)
                        )
    assert endpoint.residual((1234, 5678), start, start, start + 6) is None
    assert endpoint.residual((0, 0), start, start + 7, start + 6) is None


@pytest.mark.parametrize("limited", ["bytes", "work", "zero"])
def test_endpoint_cache_budgets_refresh_and_interrupted_builds(
    monkeypatch: pytest.MonkeyPatch, limited: str,
) -> None:
    """Snapshots reuse only geometry; exhausted or interrupted caches remain safe misses.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): isolates endpoint membership.
    - limited (str): byte limit, arithmetic-work limit, or disabled weighted storage.

    Return
    --------
    - output (None): checks refreshed signs, new rows, exact FP64 prices and atomic publication.
    """
    bounds, rows, _ = _endpoint_case(monkeypatch, 0)
    options = PricingOptions(endpoint_cache_bytes=0)
    empty = EndpointBounds(bounds, options, lambda: None)
    byte_cost, work = 4 * empty.leaves * 16 + 32, empty.size * 66 + 2 * empty.leaves
    options = replace(options, endpoint_cache_bytes=(
        0 if limited == "zero" else byte_cost if limited == "bytes" else 1_048_576),
                      endpoint_cache_work=work if limited == "work" else 1_000_000)
    endpoint = EndpointBounds(bounds, options, lambda: None)
    assert len(endpoint.entries) == (0 if limited == "zero" else 1)
    assert endpoint.stats["endpoint_cache_bytes"] <= options.endpoint_cache_bytes
    assert endpoint.stats["endpoint_build_work"] <= options.endpoint_cache_work
    endpoint.consider(np.array([(123, 456)], np.uint32))
    assert endpoint.residual((123, 456), 0, 0, 6) is None
    prices = {row: -price for row, price in bounds.view._duals.items()}
    prices[rows[69]] = 123.123456789
    fresh_bounds = SimpleNamespace(**vars(bounds))
    fresh_bounds.view = pricing.DualView(prices, bounds.problem.cfg)
    fresh_bounds.problem = replace(bounds.problem, duals=prices)
    refreshed = EndpointBounds(fresh_bounds, PricingOptions(), lambda: None)
    assert refreshed.membership is endpoint.membership
    assert rows[31] not in refreshed.bits and rows[63] not in refreshed.bits
    assert refreshed.rows[0] == rows[69]
    assert refreshed.weights[0] == prices[rows[69]]
    np.testing.assert_array_equal(refreshed.root_masks, 0)
    assert not refreshed.entries[(0, 0)].flags.writeable
    assert endpoint.weights[31] == bounds.view.row_cost(rows[31])

    calls = 0

    def expire():
        """Interrupt an unfinished weighted array without making it queryable."""
        nonlocal calls
        calls += 1
        if calls == 3:
            raise pricing.PricingTimeout("test deadline")

    empty.options, empty.check = PricingOptions(), expire
    with pytest.raises(pricing.PricingTimeout):
        empty.consider(np.array([(123, 456)], np.uint32))
    assert not empty.entries and empty.stats["endpoint_cache_bytes"] == 0


@pytest.mark.parametrize("start", [-100, 100, 2**31 - 8])
@pytest.mark.parametrize("chunk_size", [1, 2])
def test_metal_mask_words_history_and_endpoint_only_prune_proposals(
    start: int, chunk_size: int,
) -> None:
    """The device updates high bits/history and proposes deletions unavailable to old bounds.

    Parameters
    ------------
    - start (int): absolute clock offset.
    - chunk_size (int): labels per dispatch, including a split between two flights.

    Return
    --------
    - output (None): checks bit 31/63, disjoint price accounting and invalid-cache fallback.
    """
    _platform("metal")
    runtime = ComputeRuntime("metal")
    labels = np.array([[0, i, 0, i, -1, -1, 2, -(2**31), 0, 0, -1] for i in range(2)], np.int32)
    graph = np.full((2, 14), -1, np.int32)
    graph[:, 6:12] = 0
    graph[0, 0], graph[0, 6], graph[1, 12], graph[1, 13] = 1, 10, 0, 1
    roots = np.array([[i, 0, 1, start + 3, 10, 1] for i in range(2)], np.int32)
    data = np.array([[10, 0, 0, 0, 0, 1, 0, 0, 3, 1, np.inf]] * 2, np.float32)
    visits = np.array([[0, 0, np.inf], [8, 8, np.inf]], np.float32)
    masks = np.array([[0, 0], [1 << 31, 1 << 31]], np.uint32)
    meta = np.array([[start, start + 3, 4, 0, 1], [start, start + 3, 4, 1, 1]], np.int32)
    entries = np.array([[1 << 31, 1 << 31, 0, 1]], np.uint32).view(np.int32)
    trees = np.full(16, 2, np.float32)
    for valid in (True, False):
        entries[0, 3] = int(valid)
        for first in range(0, 2, chunk_size):
            current = labels[first:first + chunk_size]
            count = len(current) * 6
            records, scores = runtime.run("expand", count, (
                np.array([len(current), 11, start, 0, 0], np.int32), current,
                np.array([[7, 3]] * len(current), np.float32), graph, roots, data, visits,
                np.zeros(2, np.int32), np.array([[1, start + 1]], np.int32),
                np.array([3], np.float32), np.empty((count, 11), np.int32),
                np.empty((count, 3), np.float32), masks, meta, entries, trees,
            ), (10, 11), write_only=(10, 11))
            for index in range(0, count, 6):
                record = records[index]
                np.testing.assert_array_equal(record[7:9].view(np.uint32), [1 << 31, 1 << 31])
                np.testing.assert_array_equal(record[9:11], [1, 0])
                assert scores[index, 1] == 8.0  # Origin/visit overlap costs 3 only once.
                stronger = valid and int(record[1]) == 0
                assert bool(record[6] & 4) is stronger
                assert bool(record[6] & 8) is stronger
                assert scores[index, 2] == (0.0 if stronger else 2.0)
    runtime.clear_cache()


@pytest.mark.parametrize("options", [
    {"endpoint_bounds": False}, {"endpoint_max_rows": 1},
    {"endpoint_cache_bytes": 0}, {"endpoint_cache_work": 0},
])
def test_endpoint_omission_or_cache_exhaustion_preserves_exhaustive_optimum(
    monkeypatch: pytest.MonkeyPatch, options: dict,
) -> None:
    """Dropping extra-bound information cannot remove routes or invoke CPU full pricing.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): forbids original full-pricing implementations.
    - options (dict): disables the bound, truncates rows, or exhausts one cache budget.

    Return
    --------
    - output (None): checks the enumerated optimum and a validated returned route on Metal.
    """
    problem = _problem(signed=True)
    optimum = _oracle(problem)

    def forbidden(*args, **kwargs):
        """Prohibit both original constrained CPU pricing implementations."""
        pytest.fail("CPU full pricing fallback")

    for name in ("price_flight", "_best_column", "_best_column_compiled"):
        monkeypatch.setattr(pricing, name, forbidden)
    platform = _platform("metal", chunk_size=1)
    platform.options = replace(platform.options, **options)
    result, = platform.price_batch([problem])
    assert result.certified and result.upper_bound >= optimum - 1e-9
    assert result.reduced_cost == pytest.approx(optimum, abs=1e-8)
    _assert_canonical(problem, result)


@pytest.mark.parametrize("buffer", [0.0, 4.0])
@pytest.mark.parametrize("magnitude", [1.0, 1e8])
def test_arrival_bounds_cover_every_prefix_of_every_canonical_completion(
    buffer: float, magnitude: float,
) -> None:
    """Arrival-only bounds cover exhaustive completions despite signed-price cancellation.

    Parameters
    ------------
    - buffer (float): visit-window width, including legal revisits and endpoint overlap.
    - magnitude (float): scales non-FP32-exact original prices and cancellation.

    Return
    --------
    - output (None): verifies each root/prefix bound against every valid canonical suffix.
    """
    original = _problem(buffer=buffer, signed=True)
    prices = {row: price * magnitude + 0.123456789 for row, price in original.duals.items()}
    problem = replace(original, duals=prices, pi_f=10000.0)
    bounds, variants = _bounds(problem)
    endpoint = EndpointBounds(bounds, PricingOptions(endpoint_cache_work=10_000_000), lambda: None)
    graph, cfg, view = problem.graph, problem.cfg, bounds.view
    negative = 0.0
    for price in view._duals.values():
        if price < 0:
            negative = math.nextafter(negative + price, -math.inf)
    cells = {(int(q), int(r)): i for i, (q, r) in enumerate(
        zip(bounds.topology.cell_q, bounds.topology.cell_r, strict=True))}
    paths = [(graph.origin_cell,)]
    checked = 0
    while paths:
        path = paths.pop()
        hops = len(path) - 1
        if hops and path[-1] == graph.dest_cell:
            for variant in range(variants.n_variants):
                start = int(variants.start_step[variant])
                try:
                    column = pricing.certify_column(Column(
                        graph.request.flight_id, int(variants.departure_step[variant]),
                        0, None, None, path, 0.0,
                    ), graph, cfg, bounds.model)
                except ValueError:
                    continue
                actual = bounds.model.reduced_cost(
                    benefit=problem.params.M, cost=column.delay_s,
                    dual_cost=view.claim_cost(column.claims), pi_f=problem.pi_f,
                )
                union = bounds.origin_claims(variant)
                for hop, cell in enumerate(path):
                    step = start + hop
                    union |= pricing._visit_claims(cell, 0, step, view.offsets)
                    paid = endpoint.mask(union)
                    endpoint.consider(np.array([paid], np.uint32))
                    remaining = int(bounds.topology.rev_remaining[cells[cell]])
                    residual = endpoint.residual(paid, step, step + remaining, min(
                        bounds.topology.max_step, start + bounds.topology.air_hop_limit,
                    ))
                    assert residual is not None
                    prefix = 0.0
                    for row in union:
                        if view.row_cost(row) > 0:
                            prefix = max(0.0, math.nextafter(prefix + view.row_cost(row), -math.inf))
                    upper = bounds.endpoint_upper(
                        variant, hop, remaining, prefix, residual,
                        problem.params.M, problem.pi_f, negative,
                    )
                    assert math.isfinite(upper)
                    assert upper >= actual
                    checked += 1
        if hops < graph.max_air_hops:
            depth = view.offsets[1] - view.offsets[0]
            recent = path[-depth:] if depth else ()
            paths.extend((*path, neighbor) for neighbor in graph.outgoing_neighbors(path[-1])
                         if neighbor not in recent)
    assert checked > 100


@pytest.mark.parametrize("expire", [False, True])
def test_rejected_proposals_and_completions_cannot_certify_unresolved_work(
    monkeypatch: pytest.MonkeyPatch, expire: bool,
) -> None:
    """Untrusted GPU proposals and rejected completions cannot become validated incumbents.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): forces bad proposals and rejects the first completion.
    - expire (bool): interrupts after that rejection instead of allowing a complete retry.

    Return
    --------
    - output (None): certified mode recovers the oracle optimum or reports an explicit deadline.
    """
    problem = _problem(signed=True)
    optimum = _oracle(problem)
    run = ComputeRuntime.run
    canonical = pricing._canonical_candidate
    rejected = 0
    proposed = 0

    def force_proposals(runtime, name, count, arrays, outputs, **kwargs):
        """Every numeric device claim remains subject to original-price CPU verification."""
        nonlocal proposed
        result = run(runtime, name, count, arrays, outputs, **kwargs)
        if name == "expand":
            records, scores = result
            valid = records[:, 6] != 0
            records[valid, 6] |= 4
            records[valid, 7:9] = 0
            scores[valid, 2] = -1e30
            proposed += int(np.count_nonzero(valid))
        return result

    def reject_first(*args, **kwargs):
        """Reject before returning a score, then either retry or terminate cooperatively."""
        nonlocal rejected
        if not rejected:
            rejected += 1
            return None
        if expire:
            raise pricing.PricingTimeout("deadline after rejected completion")
        return canonical(*args, **kwargs)

    monkeypatch.setattr(pricing, "_shortest_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(ComputeRuntime, "run", force_proposals)
    monkeypatch.setattr(pricing, "_canonical_candidate", reject_first)
    platform = _platform("metal", chunk_size=1)
    result, = platform.price_batch([problem])
    assert rejected == 1 and proposed > 0
    if expire:
        assert result.status == "deadline" and not result.certified
        assert result.upper_bound is None
    else:
        assert result.certified and result.reduced_cost == pytest.approx(optimum, abs=1e-8)
        assert result.upper_bound >= optimum - 1e-9
        assert result.stats["cpu_recomputations"] > result.stats["verified_prunes"]
        _assert_canonical(problem, result)
