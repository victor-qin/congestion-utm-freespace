"""Endpoint masks, original-price range minima and device proposals preserve row unions."""

from dataclasses import replace
from decimal import Decimal, localcontext
import math
from types import SimpleNamespace

import numpy as np
import pytest

from freespace_sim.planner.colgen import pricing
from freespace_sim.planner.colgen.gpu_endpoint import EndpointBounds
from freespace_sim.planner.colgen.gpu_future import FutureBounds
from freespace_sim.planner.colgen.gpu_runtime import ComputeRuntime
from freespace_sim.planner.colgen.network import RowKey
from freespace_sim.planner.colgen.pricing_platform import PricingOptions
from freespace_sim.planner.colgen.translate import Column
from tests.test_colgen_gpu_bounds import _bounds
from tests.test_colgen_gpu_future import _ReferenceRuntime, _case
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
    nominal = {
        start + offset: frozenset((*rows[:70 - offset], negative)) - {rows[63]}
        for offset in range(7)
    }
    arrivals = {step: claims | {RowKey.cell((63, 0), 0, step + delta) for delta in (-1, 0, 1)}
                for step, claims in nominal.items()}

    def endpoints(graph, cfg, *, origin, step, timing_steps):
        """Omit a priced final-visit row so only the complete arrival union can include it."""
        assert not origin and timing_steps == 0
        return nominal[step]

    monkeypatch.setattr(pricing, "_endpoint_claims", endpoints)
    from freespace_sim.planner.colgen import gpu_endpoint

    def membership(problem, destinations, first, last, offsets, check):
        """Supply synthetic incidence to the weighted-bound tests via the builder boundary."""
        packed = {}
        for offset, step in enumerate(range(first, last + 1)):
            check()
            for row in gpu_endpoint._arrival_claims(problem, destinations, step, offsets):
                packed.setdefault(row, []).append(offset)
        result = {row: np.asarray(indices, np.intp) for row, indices in packed.items()}
        for indices in result.values():
            indices.flags.writeable = False
        return result

    monkeypatch.setattr(gpu_endpoint, "_build_arrivals", membership)
    view = pricing.DualView(prices, problem.cfg)
    view._offsets = (-1, 1)
    bounds = SimpleNamespace(
        problem=replace(problem, duals=prices), view=view, air_per_hop=0.3,
        topology=SimpleNamespace(
            max_step=start + 6, cell_q=np.arange(70), cell_r=np.zeros(70),
            dest_mask=np.arange(70) == 63,
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
    fresh_bounds.view._offsets = bounds.view.offsets
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


def test_arrival_union_preserves_destination_choices_and_changed_visit_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A final-visit charge is mandatory only for a shared final cell and current window.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): supplies an endpoint set omitting the priced last visit.

    Return
    --------
    - output (None): alternate destinations stay available, caches refresh, and paid rows vanish once.
    """
    bounds, rows, _ = _endpoint_case(monkeypatch, 0)
    previous = None
    for destinations, offsets, required in (
        ((63,), (-1, 1), True), ((63, 64), (-1, 1), False),
        ((63,), (-1, 1), True), ((63,), (0, 0), False),
        ((), (-1, 1), False),
    ):
        bounds.topology.dest_mask = np.isin(np.arange(70), destinations)
        bounds.view._offsets = offsets
        endpoint = EndpointBounds(bounds, PricingOptions(), lambda: None)
        assert endpoint.membership is not previous
        previous = endpoint.membership
        paid = endpoint.mask(row for row in rows if row != rows[63])
        endpoint.consider(np.array([paid], np.uint32))
        q, _ = endpoint.residual(paid, 0, 2, 2)
        expected = bounds.view.row_cost(rows[63]) if required else 0.0
        assert q == pytest.approx(expected, rel=0.0, abs=1e-12)
        assert q <= expected
        all_paid = endpoint.mask(rows)
        endpoint.consider(np.array([all_paid], np.uint32))
        assert endpoint.residual(all_paid, 0, 2, 2)[0] == 0.0


@pytest.mark.parametrize("start", [-100, 100, 2**31 - 8])
@pytest.mark.parametrize("chunk_size", [1, 2])
@pytest.mark.parametrize("integer_bounds", [False, True])
def test_metal_mask_words_history_and_endpoint_only_prune_proposals(
    start: int, chunk_size: int, integer_bounds: bool,
) -> None:
    """The device updates high bits/history and proposes deletions unavailable to old bounds.

    Parameters
    ------------
    - start (int): absolute clock offset.
    - chunk_size (int): labels per dispatch, including a split between two flights.
    - integer_bounds (bool): exercise the checked wrapper with exact row membership.

    Return
    --------
    - output (None): checks bit 31/63, disjoint price accounting and invalid-cache fallback.
    """
    _platform("metal")
    from freespace_sim.planner.colgen.gpu_integer import IntegerInterval, IntegerScale

    runtime = ComputeRuntime("metal", integer_bounds=integer_bounds)
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
    extra = ()
    stride = 11
    if integer_bounds:
        scale = IntegerScale(-40)
        stride += 15
        prefix = scale.encode_array(np.array([[3., 3., 0.]] * 2)).reshape(2, 15).view(np.int32)
        labels = np.concatenate((labels, prefix), axis=1)
        extra = (
            np.array([[-40, 10, 0, 0]] * 2, np.int32),
            scale.encode_array(np.array([[10., 0., 0., 0., 0., 3., 3.]] * 2)),
            scale.encode_array(np.zeros(22)),
            np.array([2, 0, 0, 0, 2, start, start + 1], np.int32),
            scale.encode_array(np.array([3., 5.])), scale.encode_array(trees.astype(np.float64)),
            np.zeros((2, 5), np.uint32), scale.encode_array(np.ones(2)),
        )
    for valid in (True, False):
        entries[0, 3] = int(valid)
        for first in range(0, 2, chunk_size):
            current = labels[first:first + chunk_size]
            count = len(current) * 6
            records, scores = runtime.run("expand", count, (
                np.array([len(current), stride, start, -1, 0, 0], np.int32), current,
                np.array([[7, 3]] * len(current), np.float32), graph, roots, data, visits[:, 2].copy(),
                np.zeros(2, np.int32), np.array([[1, start]], np.int32),
                np.array([3], np.float32), np.empty((count, stride), np.int32),
                np.empty((count, 3), np.float32), np.array([0, 1], np.int32),
                np.array([[0, 0], [0, 1]], np.int32), np.array([start + 1], np.int32),
                np.array([[8, 8]], np.float32), np.array([[1, -np.inf]] * 2, np.float32),
                masks, meta, entries, trees, *extra,
            ), (10, 11), write_only=(10, 11))
            for index in range(0, count, 6):
                record = records[index]
                np.testing.assert_array_equal(record[7:9].view(np.uint32), [1 << 31, 1 << 31])
                np.testing.assert_array_equal(record[9:11], [1, 0])
                assert scores[index, 1] == 8.0  # Origin/visit overlap costs 3 only once.
                stronger = valid and int(record[1]) == 0
                assert bool(record[6] & (64 if integer_bounds else 4)) is stronger
                assert bool(record[6] & 8) is (stronger and not integer_bounds)
                if integer_bounds:
                    positive = IntegerInterval.unpacked(record[-15:-10].view(np.uint32))
                    assert positive.valid and positive.lower * scale.quantum <= 8 <= positive.upper * scale.quantum
                assert scores[index, 2] == (0.0 if stronger else 2.0)
    runtime.clear_cache()


@pytest.mark.parametrize("options", [
    {}, {"gpu_prune_epsilon": 0.001, "gpu_prune_audit": True},
    {"endpoint_bounds": False}, {"endpoint_max_rows": 1},
    {"endpoint_cache_bytes": 0}, {"endpoint_cache_work": 0},
    {"gpu_integer_bounds": True}, {"gpu_integer_bounds": True, "endpoint_cache_bytes": 0},
    {"gpu_integer_bounds": True, "gpu_integer_audit": True},
])
@pytest.mark.parametrize("hop_budget", [0, 64 * 1024 * 1024])
def test_endpoint_omission_or_cache_exhaustion_preserves_exhaustive_optimum(
    monkeypatch: pytest.MonkeyPatch, options: dict, hop_budget: int,
) -> None:
    """Dropping extra-bound information cannot remove routes or invoke CPU full pricing.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): forbids original full-pricing implementations.
    - options (dict): disables the bound, truncates rows, or exhausts one cache budget.
    - hop_budget (int): disables or enables remaining-hop tables in the GPU search.

    Return
    --------
    - output (None): checks the enumerated optimum and a validated returned route on Metal.
    """
    problem = _problem(signed=True)
    optimum = _oracle(problem)
    from freespace_sim.planner.colgen.gpu_pricing import _Batch

    prepare, build = _Batch.prepare, FutureBounds.__init__

    def future(instance, *args, **kwargs):
        """Exercise complete optional tables and missing-table fallback on the same domain."""
        return build(instance, *args, **kwargs, hop_max_bytes=hop_budget)

    def prepare_future(batch):
        """Install the normally delayed bound before a small complete GPU enumeration."""
        prepare(batch)
        batch.prepare_future()

    monkeypatch.setattr(FutureBounds, "__init__", future)
    monkeypatch.setattr(_Batch, "prepare", prepare_future)

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
@pytest.mark.parametrize("magnitude", [0.0, 1.0, 1e8])
def test_arrival_bounds_cover_every_prefix_of_every_canonical_completion(
    buffer: float, magnitude: float,
) -> None:
    """Mandatory arrival unions cover exhaustive completions despite signed-price cancellation.

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
    from freespace_sim.planner.colgen.gpu_integer import IntegerScale
    from freespace_sim.planner.colgen.gpu_integer_bounds import (
        prepare_reference_operands, reference_upper,
    )

    integer_scale = IntegerScale.for_values(
        [problem.params.M, problem.pi_f, *prices.values()],
        max_terms=len(prices) + bounds.topology.air_hop_limit + 32,
    )
    endpoint = EndpointBounds(bounds, PricingOptions(endpoint_cache_work=10_000_000), lambda: None)
    graph, cfg, view = problem.graph, problem.cfg, bounds.view
    future = FutureBounds(
        problem, bounds.topology, variants, bounds, view, _ReferenceRuntime(), None,
    )
    negative = 0.0
    for price in view._duals.values():
        if price < 0:
            negative = math.nextafter(negative + price, -math.inf)
    from freespace_sim.planner.colgen.gpu_approach import ApproachBounds

    approach = ApproachBounds(bounds, lambda: None, max_work=10_000_000)
    root_uppers = [approach.upper(i, problem.params.M, negative, 1e100)
                   for i in range(variants.n_variants)]
    assert all(map(math.isfinite, root_uppers))
    from freespace_sim.planner.colgen.gpu_pricing import _Batch

    screening = SimpleNamespace(
        problems=[problem], models=[bounds.model], benefits=[problem.params.M],
        negative=[negative], best=[(-math.inf, None)], stats={"screened_departures": 0},
        check=lambda: None,
    )
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
                assert root_uppers[variant] >= actual
                screening.best[0] = math.nextafter(actual, -math.inf), None
                assert _Batch.keep_departure(screening, 0, column.departure_step)
                union = bounds.origin_claims(variant)
                visit_union = frozenset()
                for hop, cell in enumerate(path):
                    step = start + hop
                    visit_union |= pricing._visit_claims(cell, 0, step, view.offsets)
                    union |= visit_union
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
                    visit_lower = 0.0
                    for row in visit_union:
                        visit_lower = max(0.0, math.nextafter(
                            visit_lower + max(0.0, view.row_cost(row)), -math.inf,
                        ))
                    suffix = (future.root_lookup(cells[cell], step) if hop == 0 else
                              future.lookup(cells[cell], step, graph.max_air_hops - hop))
                    assert bounds.future_upper(
                        variant, hop, visit_lower, suffix,
                        problem.params.M, problem.pi_f, negative,
                    ) >= actual
                    integer_positive = integer_scale.positive_sum(
                        (max(0.0, view.row_cost(row)) for row in union), extra_rounds=hop,
                    )
                    integer_visits = integer_scale.positive_sum(
                        (max(0.0, view.row_cost(row)) for row in visit_union), extra_rounds=hop,
                    )
                    operands = prepare_reference_operands(
                        integer_scale, bounds, variant, hop, remaining, integer_positive,
                        integer_visits, suffix, residual, problem.params.M, negative,
                    )
                    integer_upper = reference_upper(integer_scale, operands)
                    # Canonical completion coverage is independent of shader/FP64 parity.
                    assert integer_upper.valid
                    assert integer_scale.upper_float(integer_upper) >= actual
                    from freespace_sim.planner.colgen.gpu_pricing import completion_upper_bound

                    reference = min(
                        upper,
                        bounds.future_upper(variant, hop, visit_lower, suffix,
                                            problem.params.M, problem.pi_f, negative),
                        completion_upper_bound(problem.params.M, problem.pi_f,
                                               bounds.cost_lower(variant, hop + remaining),
                                               (prefix,), negative),
                    )
                    assert integer_scale.upper_float(integer_upper) >= reference
                    checked += 1
        if hops < graph.max_air_hops:
            depth = view.offsets[1] - view.offsets[0]
            recent = path[-depth:] if depth else ()
            paths.extend((*path, neighbor) for neighbor in graph.outgoing_neighbors(path[-1])
                         if neighbor not in recent)
    assert checked > 100


def test_approach_budget_cutoff_deadline_and_changed_snapshot_fail_open() -> None:
    """Incomplete alternative scans never expose an optimistic partial root upper.

    Parameters
    ------------
    - none: uses original-price roots and independently changes their prices and restrictions.

    Return
    --------
    - output (None): completed uppers enclose enumeration; exhausted/expired scans cannot prune.
    """
    from freespace_sim.planner.colgen.gpu_approach import ApproachBounds
    from freespace_sim.planner.colgen.gpu_prices import summary_for

    problem = _problem()
    bounds, _ = _bounds(problem)
    helper = ApproachBounds(bounds, lambda: None)
    initial_work = helper.work
    negative = summary_for(bounds.view).negative
    upper = helper.upper(0, problem.params.M, negative, 1e100)
    assert math.isfinite(upper)
    assert max(helper.upper(i, problem.params.M, negative, 1e100)
               for i in range(bounds.variants.n_variants)) >= _oracle(problem)
    assert ApproachBounds(bounds, lambda: None, 0).upper(0, problem.params.M, negative, 1e100) == math.inf
    limited = ApproachBounds(bounds, lambda: None, initial_work + 2)
    assert limited.upper(0, problem.params.M, negative, 1e100) == math.inf
    assert limited.work <= limited.max_work
    assert ApproachBounds(bounds, lambda: None).upper(
        0, problem.params.M, negative, math.nextafter(upper, -math.inf),
    ) == math.inf

    def expire() -> None:
        """Stop after helper construction so a partial root scan cannot be published."""
        raise pricing.PricingTimeout("approach deadline")

    helper.check = expire
    with pytest.raises(pricing.PricingTimeout):
        helper.upper(0, problem.params.M, negative, 1e100)
    prices = {row: -value + 0.125 for row, value in problem.duals.items()}
    prices[RowKey.terminal("unreachable-credit", -7)] = -1e4
    changed = replace(problem, duals=prices, pi_f=0.5)
    fresh, variants = _bounds(changed)
    refreshed = ApproachBounds(fresh, lambda: None)
    new_negative = summary_for(fresh.view).negative
    assert max(refreshed.upper(i, changed.params.M, new_negative, 1e100)
               for i in range(variants.n_variants)) >= _oracle(changed)
    assert refreshed.upper(0, changed.params.M, -math.inf, 1e100) == math.inf


@pytest.mark.parametrize("endpoint_kind", ["one_hop", "origin_terminal", "destination_terminal"])
@pytest.mark.parametrize("magnitude", [1e-200, 1.0, 1e150])
def test_approach_unions_cover_short_routes_and_terminal_lane_choices(
    endpoint_kind: str, magnitude: float,
) -> None:
    """Root overlap, alternate terminal lanes and extreme signed prices enclose canonical routes.

    Parameters
    ------------
    - endpoint_kind (str): one-hop origin/penultimate overlap or alternate terminal lanes.
    - magnitude (float): tiny, ordinary and extreme finite original resource prices.

    Return
    --------
    - output (None): independently scored canonical routes remain under their own root upper.
    """
    from freespace_sim.planner.colgen.gpu_approach import ApproachBounds
    from freespace_sim.planner.colgen.gpu_prices import summary_for
    from freespace_sim.planner.colgen.network import build_flight_graph
    from freespace_sim.planner.colgen.pricing_platform import PricingProblem
    from freespace_sim.types import Terminal
    from tests.test_colgen_solver import _cfg, _params, _request

    cfg = _cfg(max_ground_delay_s=0.0, time_buffer_s=4.0, max_detour_factor=5.0)
    params = _params(max_air_overrun_hops=0)
    request = _request(1, (0, 0), (1, 0) if endpoint_kind == "one_hop" else (8, 2), cfg)
    catalog = ()
    if endpoint_kind != "one_hop":
        terminal = Terminal("test-terminal", 4, radius=120.0)
        origin = endpoint_kind == "origin_terminal"
        request = replace(request, **{"origin_terminal" if origin else "dest_terminal": terminal})
        catalog = ((request.origin if origin else request.dest, terminal),)
    graph = build_flight_graph(request, cfg, catalog, params)
    problem = PricingProblem(graph, cfg, params, {}, 0.0)
    unpriced, variants = _bounds(problem)
    t = unpriced.topology
    cells = tuple(zip(map(int, t.cell_q), map(int, t.cell_r)))
    columns = []
    for variant in range(variants.n_variants):
        for target in np.flatnonzero(t.dest_mask):
            path = pricing._shortest_cell_path(graph, cells[int(variants.cell[variant])], cells[int(target)])
            if path is None or len(path) < 2:
                continue
            lane = next((i for i, lane in enumerate(graph.dest_lanes) if lane.cell == path[-1]), None)
            try:
                column = pricing.certify_column(Column(
                    1, int(variants.departure_step[variant]), 0,
                    None if variants.lane_idx[variant] < 0 else int(variants.lane_idx[variant]),
                    lane, path, 0.0,
                ), graph, cfg, unpriced.model)
            except ValueError:
                continue
            columns.append((variant, column))
    assert columns
    if endpoint_kind != "one_hop":
        assert variants.n_variants > 1 or int(t.dest_mask.sum()) > 1
    all_rows = sorted(frozenset().union(*(column.claims for _, column in columns)))
    prices = {row: magnitude * (3.25 if i % 4 else -1.125) for i, row in enumerate(all_rows)}
    problem = replace(problem, duals=prices)
    bounds, current = _bounds(problem)
    np.testing.assert_array_equal(current.cell, variants.cell)
    helper = ApproachBounds(bounds, lambda: None, max_work=10_000_000)
    negative = summary_for(bounds.view).negative
    uppers = {i: helper.upper(i, params.M, negative, 1e300) for i, _ in columns}
    for variant, column in columns:
        rc = bounds.model.reduced_cost(benefit=params.M, cost=column.delay_s,
                                       dual_cost=bounds.view.claim_cost(column.claims), pi_f=0.0)
        assert math.isfinite(uppers[variant]) and uppers[variant] >= rc


@pytest.mark.parametrize("expire", [False, True])
@pytest.mark.parametrize("resident", [False, True])
def test_rejected_proposals_and_completions_cannot_certify_unresolved_work(
    monkeypatch: pytest.MonkeyPatch, expire: bool, resident: bool,
) -> None:
    """Untrusted GPU proposals and rejected completions cannot become validated incumbents.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): forces bad proposals and rejects the first completion.
    - expire (bool): interrupts after that rejection instead of allowing a complete retry.
    - resident (bool): also force proposals and corrupt paid bits inside the resident GPU pass.

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
    if resident:
        from freespace_sim.planner.colgen import gpu_runtime
        from freespace_sim.planner.colgen.gpu_pricing import _Batch

        source = gpu_runtime._source

        def forced_source(backend: str) -> str:
            """Corrupt device mask proposals; the verifier must rebuild original paid membership."""
            return source(backend).replace(
                "int flags = (finish ? 1 : 0) | (onward ? 2 : 0);",
                "int flags = (finish ? 1 : 0) | (onward ? 2 : 0) | 4;",
            ).replace("upper = fmin(upper, endpoint);", "upper = -1e30f; low = high = 0xffffffffu;")

        monkeypatch.setattr(gpu_runtime, "_source", forced_source)
        monkeypatch.setattr(_Batch, "bootstrap", lambda self: None)
    platform = _platform("metal", chunk_size=1)
    result, = platform.price_batch([problem])
    assert rejected == 1 and (proposed > 0 or result.stats["restored_prunes"] > 0)
    if expire:
        assert result.status == "deadline" and not result.certified
        assert result.upper_bound is None
    else:
        assert result.certified and result.reduced_cost == pytest.approx(optimum, abs=1e-8)
        assert result.upper_bound >= optimum - 1e-9
        assert result.stats["cpu_recomputations"] > result.stats["verified_prunes"]
        _assert_canonical(problem, result)


@pytest.mark.parametrize("cache_bytes", [0, 1_048_576])
def test_preliminary_selection_prices_paid_endpoint_words_and_preserves_diversity(
    monkeypatch: pytest.MonkeyPatch, cache_bytes: int,
) -> None:
    """The bounded preliminary beam favors paid endpoints without starving alternate roots.

    Parameters
    ------------
    - monkeypatch (pytest.MonkeyPatch): supplies independent endpoint membership.
    - cache_bytes (int): exercises available residuals and safe missing-table priority.

    Return
    --------
    - output (None): both paid words affect ranking; distinct root/arrival choices survive.
    """
    from freespace_sim.planner.colgen.gpu_pricing import _Batch

    bounds, rows, _ = _endpoint_case(monkeypatch, 100)
    endpoint = EndpointBounds(bounds, PricingOptions(endpoint_cache_bytes=cache_bytes), lambda: None)
    records = np.zeros((8, 9), np.int32)
    records[:, 0] = np.arange(8)
    records[:, 1] = [0, 0, 0, 0, 0, 0, 1, 0]
    records[4, 7:9] = np.asarray(endpoint.mask((rows[31], rows[63])), np.uint32).view(np.int32)
    graph = np.zeros((8, 14), np.int32)
    graph[:, 12] = [1, 1, 1, 1, 1, 1, 1, 2]
    scores = np.zeros((8, 3), np.float32)
    scores[:, 0] = [10, 9, 8, 7, 6, 5, 4, 3]
    batch = SimpleNamespace(
        root_flights=[0, 0], graph=graph, root_values=np.zeros((2, 11), np.float32),
        problems=[bounds.problem], endpoints=[endpoint], future=[None],
        topologies=[SimpleNamespace(max_step=106, air_hop_limit=6)], check=lambda: None,
    )
    selected = _Batch.select_beam(batch, records, scores, 100, 6)
    assert len(selected) == len(set(selected)) == 6
    assert selected[0] == (4 if cache_bytes else 0)
    assert 6 in selected and 7 in selected


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("dt,start", [(2.0, -100), (0.3, 10), (0.3, 2**31 - 20)])
@pytest.mark.parametrize("destination_count", [0, 1, 2])
def test_compact_arrivals_equal_canonical_unions(
    terminal: bool, dt: float, start: int, destination_count: int,
) -> None:
    """Compact identities preserve all rows, duplicate unions and absolute clock boundaries."""
    from freespace_sim.planner.colgen.gpu_endpoint import _arrival_claims, _build_arrivals
    from freespace_sim.planner.colgen.network import build_flight_graph
    from freespace_sim.planner.colgen.pricing_platform import PricingProblem
    from freespace_sim.types import Terminal
    from tests.test_colgen_solver import _cfg, _params, _request

    cfg = _cfg(dt_s=dt, nominal_speed_mps=120.0 / dt, hover_time_s=0.6, time_buffer_s=0.1)
    params = _params()
    request = _request(1, (0, 0), (2, 0), cfg)
    catalog = ()
    if terminal:
        term = Terminal("arrival", 4, radius=120.0)
        request = replace(request, dest_terminal=term)
        catalog = ((request.dest, term),)
    graph = build_flight_graph(request, cfg, catalog, params)
    problem = PricingProblem(graph, cfg, params, {}, 0.0)
    destinations = ((2, 0), (3, 0))[:destination_count]
    for offsets in ((-2, 1), (0, 0), (2, 4)):
        actual = _build_arrivals(problem, destinations, start, start + 7, offsets, lambda: None)
        expected = {}
        for offset in range(8):
            for row in _arrival_claims(problem, destinations, start + offset, offsets):
                expected.setdefault(row, []).append(offset)
        assert actual.keys() == expected.keys()
        for row, indices in actual.items():
            assert not indices.flags.writeable
            np.testing.assert_array_equal(indices, expected[row])
