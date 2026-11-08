"""Functional worker ownership, domain parity, snapshot, bounds and failure contracts."""

from dataclasses import fields, replace
import math
import socket
import time
import weakref

import numpy as np
import pytest

from freespace_sim.planner.colgen import gpu_prepare_pool as pool_module, pricing
from freespace_sim.planner.colgen.gpu_prepared import PreparationStopped, prepare_flight
from freespace_sim.planner.colgen.gpu_prepare_pool import GPUPreparationError, GPUPreparationPool
from freespace_sim.planner.colgen.network import RowKey, build_flight_graph
from freespace_sim.planner.colgen.pricing_platform import PricingOptions
from tests.test_colgen_pricing_platform import _problem


_REAL_WORKER = pool_module._worker_main


def _forbidden(*args, **kwargs):
    """Fail if preparation tries full CPU pricing or creates any GPU runtime."""
    raise AssertionError("Preparation attempted search/device initialization")


def _guarded_worker(sock: socket.socket, options: PricingOptions) -> None:
    """Enforce process ownership at the actual spawned worker entry point."""
    from freespace_sim.planner.colgen import gpu_runtime
    pricing.price_flight = pricing._best_column = pricing._best_column_compiled = _forbidden
    gpu_runtime.ComputeRuntime = _forbidden
    measure = pool_module._GraphSize.measure
    unretained = []

    def checked_size(self, graph):
        """Check conservative accounting after every price/domain/cache transition."""
        if options.prepare_cache_bytes == 0:
            assert all(reference() is None for reference in unretained)
            unretained.append(weakref.ref(self))
        size = measure(self, graph)
        assert size >= pool_module.retained_bytes(graph)
        return size

    pool_module._GraphSize.measure = checked_size
    _REAL_WORKER(sock, options)


def _stalled_worker(sock: socket.socket, options: PricingOptions) -> None:
    """Simulate a worker that never drains the owner's potentially large request."""
    sock.sendall(pool_module._encode(("ready",), options.prepare_max_bytes))
    time.sleep(30)


def _assert_preparation(actual, expected) -> None:
    """Compare complete domains and original bounds, including zero-priced union membership."""
    assert actual.status == expected.status == "complete"
    assert actual.best[0] == expected.best[0]
    assert actual.negative == expected.negative
    np.testing.assert_array_equal(actual.graph, expected.graph)
    for field in fields(actual.variants):
        a, b = getattr(actual.variants, field.name), getattr(expected.variants, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b
    for root in range(actual.variants.n_variants):
        assert actual.bounds.origin_claims(root) == expected.bounds.origin_claims(root)
        assert actual.bounds.root_positive(root) == pytest.approx(
            expected.bounds.root_positive(root), rel=1e-14, abs=1e-12)
        for hops in (0, actual.topology.air_hop_limit):
            assert actual.bounds.cost_lower(root, hops) == expected.bounds.cost_lower(root, hops)
    assert (actual.endpoint is None) == (expected.endpoint is None)
    if actual.endpoint is not None:
        assert actual.endpoint.rows == expected.endpoint.rows
        np.testing.assert_array_equal(actual.endpoint.root_masks, expected.endpoint.root_masks)
        np.testing.assert_array_equal(actual.endpoint.gpu_trees(), expected.endpoint.gpu_trees())
        assert actual.endpoint.visit_masks.keys() == expected.endpoint.visit_masks.keys()
        for step, (indices, masks) in actual.endpoint.visit_masks.items():
            np.testing.assert_array_equal(indices, expected.endpoint.visit_masks[step][0])
            np.testing.assert_array_equal(masks, expected.endpoint.visit_masks[step][1])


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_persistent_preparation_matches_domains_and_refreshes_snapshots(monkeypatch, workers):
    """Repricing/restrictions/objectives refresh while graph memberships survive on CPU workers."""
    monkeypatch.setattr(pool_module, "_worker_main", _guarded_worker)
    options = PricingOptions(device="gpu", gpu_backend="metal", prepare_workers=workers)
    pool = GPUPreparationPool(options)
    originals = [_problem(signed=True), _problem(buffer=0.0)]
    originals[1] = replace(originals[1], graph=replace(originals[1].graph,
                           request=replace(originals[1].graph.request, flight_id=2)))
    wide = _problem(buffer=128.0)
    wide = replace(wide, duals={RowKey.cell(wide.graph.dest_cell, 0, step): 1.0
                               for step in range(-32, wide.graph.max_step + 33)})
    originals.append(wide)
    try:
        for phase in range(4):
            problems = []
            for p in originals:
                prices = dict(p.duals)
                if phase >= 2:
                    prices = {key: -value for key, value in prices.items()}
                    prices[RowKey.cell(p.graph.dest_cell, 0, p.graph.max_step + 1)] = 7.0
                view = pricing.DualView(prices, p.cfg)
                forbidden = frozenset([next(iter(prices))]) if phase == 2 else frozenset()
                params = replace(p.params, objective="total_delay") if phase == 3 else p.params
                problems.append(replace(p, duals=view, forbidden_rows=forbidden, params=params))
            actual = pool.prepare(problems, time.monotonic() + 45)
            expected = [prepare_flight(p, p.duals, options, None) for p in problems]
            for a, e in zip(actual, expected, strict=True):
                _assert_preparation(a, e)
            if phase == 0:
                endpoint = actual[-1].endpoint
                assert endpoint is not None and len(endpoint.rows) == 64
                assert endpoint.mask((endpoint.rows[31], endpoint.rows[63])) == (1 << 31, 1 << 31)
            with pytest.raises(ValueError, match="snapshot"):
                actual[0].check(problems[0], pricing.DualView({}, problems[0].cfg), options)
            assert all(ch.job is None and not ch.outgoing and not ch.incoming for ch in pool.channels)
        assert pool.stats["graph_sends"] == 3
        assert pool.stats["peak_inflight"] <= workers
        assert pool.stats["peak_jobs"] == 3
        # A changed graph/config gets a new version even with the same flight ID.
        changed = _problem(buffer=12.0)
        changed = replace(changed, duals=pricing.DualView(changed.duals, changed.cfg))
        result, = pool.prepare([changed])
        _assert_preparation(result, prepare_flight(changed, changed.duals, options, None))
        assert pool.stats["graph_sends"] == 4
        shifted = replace(changed, graph=build_flight_graph(
            replace(changed.graph.request,
                    t_departure=changed.graph.request.t_departure + 3 * changed.cfg.dt_s),
            changed.cfg, (), changed.params,
        ))
        result, = pool.prepare([shifted])
        _assert_preparation(result, prepare_flight(shifted, shifted.duals, options, None))
        assert result.topology.base_step != changed.graph.base_step
        assert pool.stats["graph_sends"] == 5
        # One immutable snapshot serves several workers without repeated row serialization.
        shared = pricing.DualView(originals[0].duals, originals[0].cfg)
        group = [replace(originals[0], duals=shared, graph=replace(
            originals[0].graph, request=replace(originals[0].graph.request, flight_id=100 + i)))
                 for i in range(workers + 1)]
        encoded = pool.stats["snapshot_encodes"]
        for _ in range(2):
            actual = pool.prepare(group)
            for result, problem in zip(actual, group, strict=True):
                _assert_preparation(result, prepare_flight(problem, shared, options, None))
        assert pool.stats["snapshot_encodes"] == encoded + 1
        assert pool.stats["peak_snapshot_bytes"] <= options.prepare_max_bytes + 8
    finally:
        pool.close()


@pytest.mark.parametrize("failure", ["crash", "stale", "deadline", "backpressure", "oversized", "cancel"])
def test_failed_preparation_is_bounded_and_reaps_workers(monkeypatch, failure):
    """Crash, stale replies, blocked transport and limits fail explicitly without fallback."""
    options = PricingOptions(device="gpu", gpu_backend="metal", prepare_workers=1,
                             prepare_max_bytes=4096 if failure == "oversized" else 128 * 1024**2)
    pool = GPUPreparationPool(options)
    p = _problem(signed=True)
    prices = p.duals
    if failure == "backpressure":
        monkeypatch.setattr(pool_module, "_worker_main", _stalled_worker)
        prices = {RowKey.cell((i, 0), 0, 0): float(i) for i in range(100_000)}
    p = replace(p, duals=pricing.DualView(prices, p.cfg))
    try:
        pool._start(time.monotonic() + 20)
        processes = [ch.process for ch in pool.channels]
        if failure == "crash":
            processes[0].terminate()
        if failure == "stale":
            accept = pool._accept

            def stale(ch, message, deadline):
                """Substitute a previous invocation's version tag at the transport boundary."""
                if message[0] == "result":
                    message = (message[0], (-1, *message[1][1:]), *message[2:])
                return accept(ch, message, deadline)

            monkeypatch.setattr(pool, "_accept", stale)
        if failure == "cancel":
            def cancel(deadline):
                """Interrupt an active invocation as a caller cancellation would."""
                raise KeyboardInterrupt

            monkeypatch.setattr(pool, "_pump", cancel)
        deadline = time.monotonic() + (0.1 if failure == "backpressure" else 30)
        if failure == "deadline":
            deadline = time.monotonic() - 1
        expected = PreparationStopped if failure in {"deadline", "backpressure"} else GPUPreparationError
        if failure == "cancel":
            expected = KeyboardInterrupt
        started = time.monotonic()
        with pytest.raises(expected):
            pool.prepare([p], deadline)
        assert pool.closed and not pool.channels
        assert all(process._closed for process in processes)
        if failure in {"deadline", "backpressure"}:
            assert time.monotonic() - started < 3
    finally:
        pool.close()


def test_cache_eviction_and_empty_domains_preserve_results(monkeypatch):
    """Zero cache budget safely rebuilds; an empty prepared domain remains a complete domain."""
    monkeypatch.setattr(pool_module, "_worker_main", _guarded_worker)
    options = PricingOptions(device="gpu", gpu_backend="metal", prepare_workers=1,
                             prepare_cache_bytes=0, prepare_batch_size=1)
    pool = GPUPreparationPool(options)
    p = _problem()
    p = replace(p, duals=pricing.DualView(p.duals, p.cfg), forbidden_rows=frozenset(p.duals))
    try:
        for _ in range(2):
            result, = pool.prepare([p])
            assert result.status == "complete" and result.variants.n_variants == 0
            assert result.best[0] == -math.inf
            assert result.stats["worker_cache_bytes"] == pool.stats["owner_cache_bytes"] == 0
        assert pool.stats["graph_sends"] == 2
        with pytest.raises(GPUPreparationError, match="group"):
            pool.prepare([p, p])
    finally:
        pool.close()

    # An LRU can replace all values without changing length. Such growth must be
    # charged before admission rather than hidden by the previous size measurement.
    accounting = pool_module._GraphSize()
    cache = p.graph._search_cache.endpoint_claims
    cache.clear()
    monkeypatch.setattr(pricing, "_MAX_ENDPOINT_CLAIMS", 2)
    for step in range(5):
        pricing._endpoint_claims(p.graph, p.cfg, origin=False, step=step, timing_steps=step)
        assert accounting.measure(p.graph) >= pool_module.retained_bytes(p.graph)
        assert len(cache) <= 2
