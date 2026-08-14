"""The parallel pricing sweep must change work, never answers."""
from __future__ import annotations

import inspect
import multiprocessing as mp
import os
import pickle
import time

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen import pricing_pool
from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import DualView
from freespace_sim.planner.colgen.pricing_pool import (
    PricingPool,
    StalePricingWorker,
    SweepResult,
    _accepted_prefix,
    _lane_assignment,
    _sweep_results,
    price_sweep,
)
from freespace_sim.planner.colgen.solver import ColGenSolver
from freespace_sim.types import FlightRequest, vec


def _cfg(**overrides) -> SimConfig:
    values = {
        "planner": "colgen",
        "flight_levels_m": (100.0,),
        "airspace_ceiling_m": 125.0,
        "region_size_m": (20_000.0, 20_000.0),
        "terminal_airspace_always_active": True,
        "max_ground_delay_s": 48.0,
    }
    values.update(overrides)
    return SimConfig(**values)


def _request(flight_id: int, origin, destination, cfg: SimConfig) -> FlightRequest:
    def point(cell):
        x, y = hg.hex_center(*cell, hg.circumradius(cfg))
        return vec(x, y, cfg.ground_level_m)

    return FlightRequest(flight_id, point(origin), point(destination), 0.0, 0.0)


def _params(**overrides) -> ColGenParams:
    values = {
        "solver": "highs",
        "max_air_overrun_hops": 0,
        "max_iterations": 3,
        "time_limit_s": 120.0,
        "n_heuristic_tries": 16,
        # PINNED to the sequential loop, deliberately not the shipped default of 4.  These
        # are unit tests on five-flight problems: a pool would spawn four processes that
        # each re-import numba and rebuild every graph, which costs seconds per test and
        # buys nothing.  The tests that mean to exercise the pool ask for it by name.  The
        # SHIPPED default is pinned separately, in `test_experiment_run.py`.
        "n_pricing_workers": 0,
    }
    values.update(overrides)
    return ColGenParams(**values)


def _fingerprint(result):
    """Everything a performance knob is forbidden to move."""

    return (
        result.stats["objective"],
        result.stats["selected_flights"],
        result.stats["n_columns"],
        result.stats["iterations"],
        result.stats["termination_reason"],
        tuple(sorted(result.stats["denied_flight_ids"])),
        tuple(
            (
                flight_id,
                column.departure_step,
                repr(column.delay_s),
                tuple(tuple(cell) for cell in column.cell_path),
                tuple(sorted(tuple(row) for row in column.claims)),
            )
            for flight_id, column in sorted(result.columns.items())
        ),
    )


@pytest.mark.parametrize("field", ["n_pricing_workers", "pricing_chunksize"])
def test_pool_knobs_reject_non_integers(field):
    with pytest.raises(TypeError):
        ColGenParams(**{field: 1.5})
    with pytest.raises(TypeError):
        ColGenParams(**{field: True})


def test_pool_knobs_reject_out_of_range_values():
    with pytest.raises(ValueError):
        ColGenParams(n_pricing_workers=-1)
    with pytest.raises(ValueError):
        ColGenParams(pricing_chunksize=0)
    # The ceiling exists because the failure past it is an OOM rather than a message: each
    # worker rebuilds every graph and holds its own label pool.
    with pytest.raises(ValueError):
        ColGenParams(n_pricing_workers=4 * (os.cpu_count() or 1) + 1)


def test_the_pool_width_has_exactly_one_home():
    """It had two, and they could disagree.

    ``solve`` used to take a separate ``ParallelPricingConfig`` whose own ``n_workers``
    default competed with this one, and ``batch`` mapped an explicit 0 to ``None`` -- which
    ``price_sweep`` resolved as *the dataclass default*, not *sequential*.  Raising that
    default would therefore have turned ``--colgen-workers 0`` into a pool.  Assert both
    that the setting lives on the params object and that no second home has come back.
    """

    # 0, i.e. OPT-IN.  It was briefly defaulted to 4 on the strength of `rss_children`
    # reading flat across worker counts -- which it always does, because
    # `getrusage(RUSAGE_CHILDREN).ru_maxrss` is the largest single child and never the
    # sum.  Sampling the process TREE instead put x50 density at 3.9 GB sequential,
    # 12.5 GB at 4 workers and 22.7 GB at 8: linear, and unaffordable by default on a
    # 4 GB/core node.
    assert ColGenParams().n_pricing_workers == 0
    assert ColGenParams(n_pricing_workers=4).n_pricing_workers == 4
    assert "parallel" not in inspect.signature(ColGenSolver.solve).parameters
    assert not hasattr(pricing_pool, "ParallelPricingConfig")


def test_sweep_result_reports_completeness_from_the_timeout_field():
    complete = SweepResult((1, 2), (0.5, 0.25), (None, None), None)
    assert complete.complete
    assert not SweepResult((1,), (0.5,), (None,), 2).complete


def test_parallel_sweep_returns_exactly_the_sequential_answer():
    """The contract, end to end.

    A pool changes only WHEN a subproblem finishes.  Two things would break that and
    neither is visible from a single-arm run: accepting results in completion order feeds
    `master.upper_bound`'s non-associative `sum` a different order, and accepting a flight
    priced after a timeout admits a column the sequential sweep never reached.
    """

    cfg = _cfg()
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
        _request(3, (-4, 4), (4, -4), cfg),
        _request(4, (4, -4), (-4, 4), cfg),
    ]
    sequential = ColGenSolver().solve(requests, cfg, (), _params())
    parallel = ColGenSolver().solve(requests, cfg, (), _params(n_pricing_workers=2))
    assert _fingerprint(parallel) == _fingerprint(sequential)


@pytest.mark.parametrize("n_workers", [0, 2])
def test_an_expired_deadline_yields_an_empty_prefix_in_both_paths(n_workers):
    """The timeout path, which only the PARALLEL arm can get wrong.

    A worker cannot raise across the pool cheaply, so it reports the timeout in its return
    value -- and an ``object()`` sentinel would have been the natural way to write that and
    silently wrong, because pickling preserves value, not identity.  Nothing in the
    sequential path pickles, so only this parametrisation catches it.
    """

    cfg = _cfg()
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]
    params = _params(n_pricing_workers=n_workers)
    catalog = StaticTerminalCatalog((), cfg)
    graphs = {
        request.flight_id: build_flight_graph(request, cfg, catalog, params)
        for request in requests
    }
    order = [1, 2]
    result = price_sweep(
        order,
        requests,
        graphs,
        cfg,
        params,
        catalog,
        {},
        DualView({}, cfg),
        dict.fromkeys(order, 0.0),
        {},
        deadline=time.monotonic() - 1.0,
    )
    assert not result.complete
    assert result.timeout_flight_id == order[0]
    # The sequential loop breaks at the first timeout, so nothing is accepted -- and a
    # worker that finished flight 2 first must not smuggle it in.
    assert result.flight_ids == ()
    assert result.reduced_costs == ()
    assert result.columns == ()


def test_a_timeout_discards_later_results_that_completed():
    """The rule the module exists for, on the sequence neither arm above can produce.

    The deadline is a wall clock, so `price_sweep` is either past it -- every flight times
    out, and the empty prefix above would survive this rule being deleted -- or inside it,
    where nothing times out at all.  What a POOL produces and the sequential loop never
    does is a gap with completed flights on both sides of it, and accepting flight 9 there
    admits a column the sequential sweep would have stopped before reaching.
    """

    accepted = _accepted_prefix(iter([
        (7, True, 0.5, None, 1.0, {"priced": 1}, {}),
        (3, False, 0.0, None, 0.25, {}, {}),
        (9, True, 4.0, None, 8.0, {"priced": 1, "fell_back": 1}, {}),
    ]))

    assert accepted.flight_ids == (7,)
    assert accepted.reduced_costs == (0.5,)
    assert accepted.columns == (None,)
    assert accepted.timeout_flight_id == 3
    assert not accepted.complete
    # Neither the timed-out task's seconds nor the discarded one's may reach the telemetry:
    # the first is not work the sweep kept and the second is work it threw away.
    assert accepted.task_total_s == 1.0
    assert (accepted.kernel_priced, accepted.kernel_fell_back) == (1, 0)


def test_a_complete_sweep_keeps_index_order_and_sums_the_worker_tallies():
    """Index order, not sorted and not completion order -- the reduced costs are summed
    downstream by a plain non-associative `sum`, so the sequence itself is the contract."""

    accepted = _accepted_prefix(iter([
        (7, True, 0.5, None, 1.0, {"priced": 1}, {}),
        (3, True, 0.25, None, 0.5,
         {"priced": 1, "fell_back": 1, "declined_label_pool_exhausted": 1}, {}),
    ]))

    assert accepted.complete
    assert accepted.timeout_flight_id is None
    assert accepted.flight_ids == (7, 3)
    assert accepted.reduced_costs == (0.5, 0.25)
    assert accepted.task_total_s == 1.5
    assert (accepted.kernel_priced, accepted.kernel_fell_back) == (2, 1)
    # `Counter.update` ADDS; a plain `dict.update` here would report only the last task's
    # tally, silently, and every count would read as 1.
    assert accepted.kernel_counters["priced"] == 2
    # The reason rides along with the count, which is the whole point of the dict: a run
    # can say WHICH cause sent a flight back to Python, not merely that one did.
    assert accepted.kernel_counters["declined_label_pool_exhausted"] == 1


def test_a_worker_that_cannot_initialise_reports_from_its_first_task(monkeypatch):
    """An initializer that RAISES hangs the sweep; one that records gives a traceback.

    `mp.Pool` calls the initializer outside the try/except that wraps a task, so a worker
    dying there is replaced by one that dies identically, forever, while `imap` waits on a
    result nobody will produce.  Rebuilding every graph is the worker's largest allocation,
    so `MemoryError` -- the failure the `n_workers` ceiling is about -- lands exactly here.
    """

    def out_of_memory(*_args, **_kwargs):
        raise MemoryError("label pool")

    cfg = _cfg()
    # A fresh dict rather than the module's, so this in-process call cannot leak worker
    # state into a test that spawns real ones.
    monkeypatch.setattr(pricing_pool, "_WORKER", {})
    monkeypatch.setattr(pricing_pool, "build_flight_graph", out_of_memory)

    # The contract is that THIS does not raise.
    pricing_pool._init_worker(
        0, [_request(1, (-4, 0), (4, 0), cfg)], cfg, _params(),
        StaticTerminalCatalog((), cfg),
    )
    with pytest.raises(RuntimeError, match="failed to initialise") as raised:
        pricing_pool._price_one(("epoch", 1), 1)
    # The original cause, carried as text because the exception object itself has to
    # survive being pickled back to the parent.
    assert "MemoryError: label pool" in str(raised.value)
    # The readiness probe has to re-raise it too, or `PricingPool.start()` would block
    # forever against a worker whose replacement dies the same way.
    with pytest.raises(RuntimeError, match="failed to initialise"):
        pricing_pool._worker_identity(None)
    # And a state load must not overwrite the initializer's traceback with a failure
    # caused BY it -- the cause is the useful message.
    pricing_pool._load_sweep_state(("epoch", 1), pickle.dumps({}), {}, {}, None)
    with pytest.raises(RuntimeError, match="failed to initialise"):
        pricing_pool._price_one(("epoch", 1), 1)


def test_explicit_zero_workers_matches_no_config_at_all():
    """`n_workers=0` is the sequential loop, not a one-worker pool."""

    cfg = _cfg()
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
    ]
    default = ColGenSolver().solve(requests, cfg, (), _params())
    pinned = ColGenSolver().solve(requests, cfg, (), _params(n_pricing_workers=0))
    assert _fingerprint(pinned) == _fingerprint(default)


class _LostResult:
    """An ``imap`` iterator whose k-th result never arrives, like a task whose worker died.

    `mp.Pool` does not fail that task -- it starts a replacement process and the lost
    result is simply never produced -- so the real object blocks in `next(timeout)` until
    the timeout and then raises. A fake is the only way to state that contract: killing a
    worker for real is a race, and the hang it produces has no natural end.
    """

    def __init__(self, results, lose_at):
        self._results, self._lose_at, self._i = results, lose_at, 0

    def next(self, timeout=None):
        if self._i == self._lose_at:
            raise mp.TimeoutError("worker died; this result will never arrive")
        if self._i >= len(self._results):
            raise StopIteration
        value = self._results[self._i]
        self._i += 1
        return value


def test_a_lost_result_ends_the_sweep_instead_of_blocking_forever():
    """The hang this guards is silent and unbounded, and the wrong fix is worse than it.

    A truncated prefix that reported itself COMPLETE would let `master.upper_bound` take a
    bound over flights that were never priced -- a wrong answer rather than a slow one. So
    the lost flight has to come back named, with `complete` False.
    """

    order = [7, 3, 9]
    priced = [
        (7, True, 0.5, None, 1.0, {"priced": 1}, {}),
        (3, True, 0.25, None, 0.5, {"priced": 1, "fell_back": 1}, {}),
    ]
    # One lane, so "the k-th result" and "the k-th flight of this lane" coincide and the
    # loss lands on flight 9 exactly as it did before lanes existed.
    accepted = _accepted_prefix(
        _sweep_results(
            [_LostResult([[r] for r in priced], lose_at=2)], order,
            {7: 0, 3: 0, 9: 0}, deadline=0.0,
        )
    )

    assert accepted.flight_ids == (7, 3)
    assert accepted.timeout_flight_id == 9
    assert not accepted.complete
    # The lost task contributes no seconds and no kernel tally: it is not work that landed.
    assert accepted.task_total_s == 1.5
    assert (accepted.kernel_priced, accepted.kernel_fell_back) == (2, 1)


def test_every_result_arriving_in_time_is_still_a_complete_sweep():
    """The guard must not turn an ordinary finished sweep into a phantom timeout."""

    order = [7, 3]
    priced = [
        (7, True, 0.5, None, 1.0, {"priced": 1}, {}),
        (3, True, 0.25, None, 0.5, {"priced": 1, "fell_back": 1}, {}),
    ]
    accepted = _accepted_prefix(
        _sweep_results(
            [_LostResult([[r] for r in priced], lose_at=None)], order,
            {7: 0, 3: 0}, deadline=None,
        )
    )

    assert accepted.complete
    assert accepted.timeout_flight_id is None
    assert accepted.flight_ids == (7, 3)


def test_the_lane_merge_yields_pricing_order_when_lanes_finish_out_of_order():
    """The replacement for the global-FIFO property the shared queue used to give.

    `mp.Pool.imap` guaranteed "the k-th result is the k-th chunk" across ALL workers. With a
    queue per lane that is gone, and the merge has to rebuild index order from per-lane
    streams -- which works because a lane's own results stay FIFO in the order that lane's
    flights appear in `pricing_order`. Lane 1 is fully ready here while lane 0 dribbles, so
    a merge that emitted whatever was available would visibly reorder.
    """

    order = [10, 11, 12, 13]          # lane 0 owns 10 and 12; lane 1 owns 11 and 13
    lane_of = {10: 0, 11: 1, 12: 0, 13: 1}
    lane0 = [[(10, True, 1.0, None, 0.1, {}, {})], [(12, True, 3.0, None, 0.1, {}, {})]]
    lane1 = [[(11, True, 2.0, None, 0.1, {}, {}), (13, True, 4.0, None, 0.1, {}, {})]]

    accepted = _accepted_prefix(
        _sweep_results(
            [_LostResult(lane0, lose_at=None), _LostResult(lane1, lose_at=None)],
            order, lane_of, deadline=None,
        )
    )

    assert accepted.flight_ids == (10, 11, 12, 13)
    # Index order is not cosmetic: `master.upper_bound` sums these with plain `sum`, and
    # float addition is not associative.
    assert accepted.reduced_costs == (1.0, 2.0, 3.0, 4.0)


def test_a_lane_that_returns_the_wrong_flight_is_refused():
    """A merge bug must crash, not transpose two flights' reduced costs.

    Silently swapping them would be a wrong answer with no symptom: both flights were
    priced, the prefix is full length, and `complete` is True.
    """

    order = [10, 11]
    lane_of = {10: 0, 11: 0}
    wrong = [[(11, True, 2.0, None, 0.1, {}, {})], [(10, True, 1.0, None, 0.1, {}, {})]]

    with pytest.raises(RuntimeError, match="where pricing_order expects"):
        _accepted_prefix(
            _sweep_results([_LostResult(wrong, lose_at=None)], order, lane_of, deadline=None)
        )


def test_lane_assignment_is_stable_and_balanced():
    """Fixed for the solve, and even on sparse ids -- which `flight_id % n` is not."""

    sparse = [900, 3, 41, 7, 12, 88, 5]
    lanes = _lane_assignment(sparse, 3)
    assert set(lanes) == set(sparse)
    counts = [sum(1 for v in lanes.values() if v == lane) for lane in range(3)]
    assert max(counts) - min(counts) <= 1, counts
    # Independent of the order the ids arrive in: the map is keyed on the flight, so a
    # re-sorted `pricing_order` cannot move a flight to a different worker.
    assert _lane_assignment(list(reversed(sparse)), 3) == lanes
    # `flight_id % n_lanes` would have put 900, 3, 12 and 88 unevenly; sorted-index does not.
    assert _lane_assignment(range(6), 3) == {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}


def test_both_arms_report_the_same_per_flight_rows():
    """A diagnostic that exists only under a pool cannot be checked against anything.

    These rows are for hunting the straggler, and the straggler only matters under a pool
    -- but that is exactly the arm whose numbers nobody can verify by hand.  Producing the
    same SHAPE sequentially is what makes the pooled table trustworthy: same flights, same
    order, same keys.  The clocks differ (contention is real) so only structure is compared.
    """

    cfg = _cfg()
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
        _request(3, (-4, 4), (4, -4), cfg),
    ]
    catalog = StaticTerminalCatalog((), cfg)
    order = [1, 2, 3]

    def sweep(n_workers):
        params = _params(n_pricing_workers=n_workers)
        graphs = {
            r.flight_id: build_flight_graph(r, cfg, catalog, params) for r in requests
        }
        return price_sweep(
            order, requests, graphs, cfg, params, catalog, {}, DualView({}, cfg),
            dict.fromkeys(order, 0.0), {}, deadline=None,
        )

    seq, par = sweep(0), sweep(2)

    assert [r["flight_id"] for r in seq.flight_records] == order
    assert [r["flight_id"] for r in par.flight_records] == order
    assert all(r["priced"] for r in seq.flight_records + par.flight_records)
    # Same keys, so a field that only the in-process arm can fill would fail here rather
    # than silently reading as absent in every pooled run.
    assert [sorted(r) for r in seq.flight_records] == [sorted(r) for r in par.flight_records]
    # The clock is per-flight and positive in both arms -- the whole point of the rows.
    assert all(r["task_s"] > 0.0 for r in par.flight_records)


def test_a_flight_that_never_reached_the_kernel_reports_no_stale_labels():
    """`clear_search_record` is what stops one flight's 67M labels being filed under another.

    The record is a module global, so without the clear a flight that declines BEFORE the
    compiled search -- no numba, multi-level graph, refused topology -- inherits whatever
    the previous flight left there.  In a straggler hunt that is worse than a missing row:
    it names the wrong flight.
    """

    from freespace_sim.planner.colgen import pricing as pricing_mod

    pricing_mod._LAST_SEARCH.update(flight_id=999, n_labels=67_108_864, attempts=7)
    pricing_mod.clear_search_record()
    assert pricing_mod.last_search_record() == {}

    # `lane`/`pid` describe the sequential arm as written -- one lane, this process. The
    # pool overwrites both once it knows which worker actually ran the flight.
    record = pricing_pool._flight_record(4, 1.5, priced=True)
    assert record == {
        "flight_id": 4, "task_s": 1.5, "priced": True, "lane": 0, "pid": os.getpid(),
    }


@pytest.mark.parametrize("chunksize", [1, 3])
def test_a_real_pool_sweep_survives_every_supported_chunksize(chunksize):
    """The gap that let a P1 ship: the fakes above all expose `.next(timeout)`.

    A real `Pool.imap` only returns an `IMapIterator` -- the one type with that method --
    when its OWN chunksize is 1; above 1 CPython returns
    `(item for chunk in result for item in chunk)`, a plain generator. So the deadline
    guard raised `AttributeError` before yielding anything, and every sweep at the
    documented `--chunksize 8` died. Only a REAL pool has the right return type, which is
    why this test pays for one.
    """

    cfg = _cfg()
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
        _request(3, (-4, 4), (4, -4), cfg),
    ]
    params = _params(n_pricing_workers=2, pricing_chunksize=chunksize)
    catalog = StaticTerminalCatalog((), cfg)
    graphs = {
        r.flight_id: build_flight_graph(r, cfg, catalog, params) for r in requests
    }
    order = [1, 2, 3]
    result = price_sweep(
        order, requests, graphs, cfg, params, catalog, {}, DualView({}, cfg),
        dict.fromkeys(order, 0.0), {}, deadline=None,
    )

    assert result.complete
    assert result.flight_ids == tuple(order)
    assert [r["flight_id"] for r in result.flight_records] == order


def test_chunked_and_unchunked_pools_agree_exactly():
    """`chunksize` is a dispatch knob, so it must not move a single number."""

    cfg = _cfg()
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
        _request(3, (-4, 4), (4, -4), cfg),
        _request(4, (4, -4), (-4, 4), cfg),
    ]
    one = ColGenSolver().solve(
        requests, cfg, (), _params(n_pricing_workers=2, pricing_chunksize=1)
    )
    many = ColGenSolver().solve(
        requests, cfg, (), _params(n_pricing_workers=2, pricing_chunksize=3)
    )
    assert _fingerprint(many) == _fingerprint(one)


# ------------------------------------------------- the pool outlives the sweep (issue #88)


def _worker_fixture(cfg, flight_ids=(1,)):
    """Bring one worker up in-process, exactly as `_init_worker` would in a spawned one."""

    requests = [_request(fid, (-4, 0), (4, 0), cfg) for fid in flight_ids]
    pricing_pool._init_worker(
        0, requests, cfg, _params(), StaticTerminalCatalog((), cfg)
    )
    assert pricing_pool._WORKER.get("init_error") is None
    return requests


def _duals_that_reach_the_compiled_search(request, cfg, params):
    """Duals on rows the seed actually claims, which is what forces the DAG search to run.

    `price_flight` short-circuits when the shortest-path seed pays no row price: with a zero
    dual on every claim, non-negative duals can only make alternatives weakly worse, so it
    returns the seed and the compiled path is never entered (pricing.py, the
    `seed_dual_cost == 0.0 and max_negative_credit == 0.0` branch). A fixture built on empty
    duals therefore rebuilds NOTHING and passes a packing-reuse test vacuously -- which is
    why the test below asserts the first sweep really did build.
    """

    from freespace_sim.planner.colgen import pricing as _pricing

    graph = build_flight_graph(request, cfg, StaticTerminalCatalog((), cfg), params)
    seed = _pricing.seed_column(graph, cfg)
    return {row: 3.0 for row in list(seed.claims)[:4]}


def test_the_packing_is_built_once_across_two_sweeps(monkeypatch):
    """Issue #88, stated as a COUNT: a second sweep must not rebuild the compiled packing.

    This is the assertion the whole change exists for, and it is deliberately a count rather
    than a wall clock -- the packing costs ~184 ms a flight, which is real but far smaller
    than the run-to-run variance on a shared machine, so a timing assertion would be both
    weaker and flakier. `prepare_topology` running exactly once over two sweeps is exact.

    Runs entirely in-process: `_init_worker` and `_load_sweep_state` are the two halves a
    real worker executes, and driving them directly is what lets a unit test see across a
    sweep boundary at all.
    """

    cfg = _cfg()
    params = _params()
    monkeypatch.setattr(pricing_pool, "_WORKER", {})
    requests = _worker_fixture(cfg)
    duals = _duals_that_reach_the_compiled_search(requests[0], cfg, params)

    from freespace_sim.planner.colgen import dp_prepare

    calls = []
    real = dp_prepare.prepare_topology
    monkeypatch.setattr(
        dp_prepare, "prepare_topology",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )

    blob = pickle.dumps(duals)
    per_sweep = []
    for sweep in (1, 2):
        epoch = ("uid", sweep)
        pricing_pool._load_sweep_state(epoch, blob, {1: 0.0}, {}, None)
        assert pricing_pool._WORKER.get("sweep_error") is None
        priced = pricing_pool._price_batch((epoch, [1]))
        assert priced[0][1] is True
        per_sweep.append(len(calls))

    # The fixture has to REACH the compiled path, or "it did not rebuild" is vacuous.
    assert per_sweep[0] == 1, "the first sweep never built a packing; fixture is inert"
    assert per_sweep[1] == 1, f"the packing was rebuilt: {per_sweep[1]} builds over 2 sweeps"
    graph = pricing_pool._WORKER["graphs"][1]
    assert graph._search_cache.prepared is not None


def test_a_worker_refuses_a_task_from_a_sweep_it_does_not_hold(monkeypatch):
    """The stale-dual guard. Without it this is a wrong NUMBER, not an error.

    `mp.Pool` replaces a dead worker by re-running the original initargs, which under a
    solve-scoped pool carry solve constants only -- so the replacement holds no duals at
    all. Pricing against the previous sweep's duals would return a reduced cost that
    `master.upper_bound` accepts as a valid bound.
    """

    cfg = _cfg()
    monkeypatch.setattr(pricing_pool, "_WORKER", {})
    _worker_fixture(cfg)
    pricing_pool._load_sweep_state(("uid", 1), pickle.dumps({}), {1: 0.0}, {}, None)

    with pytest.raises(StalePricingWorker, match="holds sweep state"):
        pricing_pool._price_one(("uid", 2), 1)
    # The message has to name the worker, because the operator's next question is which one
    # died and whether it was the OOM killer.
    try:
        pricing_pool._price_one(("uid", 2), 1)
    except StalePricingWorker as exc:
        assert "lane 0" in str(exc) and str(os.getpid()) in str(exc)


def test_a_respawned_worker_has_no_sweep_state_and_says_so(monkeypatch):
    """A replacement worker runs the initializer and nothing else -- so it must refuse."""

    cfg = _cfg()
    monkeypatch.setattr(pricing_pool, "_WORKER", {})
    _worker_fixture(cfg)
    with pytest.raises(StalePricingWorker, match="holds sweep state None"):
        pricing_pool._price_batch((("uid", 1), [1]))


def test_a_failed_state_load_does_not_leave_the_previous_sweep_readable(monkeypatch):
    """The subtlest bug in the design, and the reason `sweep` is cleared BEFORE the try.

    If building the `DualView` raises -- `MemoryError` is the realistic one, since it is
    dense per resource -- a worker that kept the previous tuple would price iteration k
    against iteration k-1's duals: plausible, wrong, and silent.
    """

    cfg = _cfg()
    monkeypatch.setattr(pricing_pool, "_WORKER", {})
    _worker_fixture(cfg)
    pricing_pool._load_sweep_state(("uid", 1), pickle.dumps({}), {1: 0.0}, {}, None)
    assert pricing_pool._WORKER["sweep"] is not None

    def out_of_memory(*_args, **_kwargs):
        raise MemoryError("dual prefix series")

    monkeypatch.setattr(pricing_pool, "DualView", out_of_memory)
    pricing_pool._load_sweep_state(("uid", 2), pickle.dumps({}), {1: 0.0}, {}, None)

    assert pricing_pool._WORKER["sweep"] is None, "sweep 1's duals survived a failed load"
    with pytest.raises(RuntimeError, match="failed to load sweep state"):
        pricing_pool._price_one(("uid", 2), 1)


def test_a_persistent_pool_reuses_its_workers_and_pins_each_flight():
    """The end-to-end shape of the fix: same processes, same flight->worker, every sweep.

    Driven through `PricingPool` directly rather than through a solve, because how many
    sweeps a solve runs depends on when column generation converges -- a three-flight
    problem finishes in one iteration, and a one-sweep run cannot show reuse ACROSS sweeps
    no matter what it asserts. Worker identity is also exactly what an in-process test
    cannot see, so this pays for real processes.
    """

    cfg = _cfg()
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
        _request(3, (-3, 1), (3, -1), cfg),
    ]
    order = [1, 2, 3]
    flight_duals = dict.fromkeys(order, 0.0)
    with PricingPool(
        requests, cfg, _params(n_pricing_workers=2), StaticTerminalCatalog((), cfg)
    ) as pool:
        sweeps = [pool.run_sweep(order, {}, flight_duals, {}, None) for _ in range(3)]

    seen = [
        {(r["flight_id"], r["lane"], r["pid"]) for r in sweep.flight_records}
        for sweep in sweeps
    ]
    pids = [{pid for _f, _l, pid in rows} for rows in seen]
    assert all(p == pids[0] for p in pids), f"workers were replaced between sweeps: {pids}"
    assert len(pids[0]) == 2
    assert all(rows == seen[0] for rows in seen), "a flight moved between workers"
    assert os.getpid() not in pids[0], "the pool priced in the parent process"
    # Two lanes over three flights, so the split is 2/1 and neither lane is empty.
    assert {lane for _f, lane, _p in seen[0]} == {0, 1}


def test_the_solver_hands_its_sweeps_one_pool():
    """`solve` has to OWN the pool, or every sweep pays the launch ramp again.

    Distinct from the test above: that one proves `PricingPool` reuses workers, this one
    proves the solver actually passes one rather than letting `price_sweep` build a
    throwaway per sweep. `pricing_pool_setup_s` is the observable -- nonzero because a pool
    was started, and far below `pricing_wall_s` because it was started once.
    """

    cfg = _cfg()
    requests = [
        _request(1, (-4, 0), (4, 0), cfg),
        _request(2, (0, -4), (0, 4), cfg),
        _request(3, (-3, 1), (3, -1), cfg),
    ]
    rows = []

    def on_iteration(state):
        rows.extend(state.get("sweep_flight_records") or ())

    result = ColGenSolver().solve(
        requests, cfg, (), _params(n_pricing_workers=2, max_iterations=3),
        on_iteration=on_iteration,
    )
    assert result.stats["pricing_pool_setup_s"] > 0.0
    assert rows and all("lane" in r and "pid" in r for r in rows)
    assert os.getpid() not in {r["pid"] for r in rows}
    # Lane assignment is a pure function of the flight ids, so the solver's pool must have
    # produced the same split this does.
    assert {r["flight_id"]: r["lane"] for r in rows} == _lane_assignment([1, 2, 3], 2)


def test_pool_setup_is_charged_to_the_first_sweep_only():
    """`pool_setup_s` is what makes "the launch ramp stopped recurring" visible in a run."""

    cfg = _cfg()
    requests = [_request(1, (-4, 0), (4, 0), cfg), _request(2, (0, -4), (0, 4), cfg)]
    with PricingPool(requests, cfg, _params(n_pricing_workers=2), StaticTerminalCatalog((), cfg)) as pool:
        first = pool.run_sweep([1, 2], {}, {1: 0.0, 2: 0.0}, {}, None)
        second = pool.run_sweep([1, 2], {}, {1: 0.0, 2: 0.0}, {}, None)
    assert first.pool_setup_s > 0.0
    assert second.pool_setup_s == 0.0
    assert first.complete and second.complete
    assert first.flight_ids == second.flight_ids == (1, 2)


def test_the_pool_refuses_a_second_sweep_after_an_incomplete_one():
    """An incomplete sweep abandons running tasks, so the pool must not be reused.

    Safe to refuse because an incomplete sweep ALWAYS ends the solve (`solver` sets
    `pricing_complete = False` and breaks). A future refactor that changes that gets this
    exception rather than sweep k's leftovers interleaved into sweep k+1.
    """

    cfg = _cfg()
    requests = [_request(1, (-4, 0), (4, 0), cfg), _request(2, (0, -4), (0, 4), cfg)]
    with PricingPool(requests, cfg, _params(n_pricing_workers=2), StaticTerminalCatalog((), cfg)) as pool:
        expired = pool.run_sweep([1, 2], {}, {1: 0.0, 2: 0.0}, {}, time.monotonic() - 1.0)
        assert not expired.complete
        with pytest.raises(RuntimeError, match="no longer usable"):
            pool.run_sweep([1, 2], {}, {1: 0.0, 2: 0.0}, {}, None)


def test_closing_the_pool_twice_is_safe():
    """`close` runs from a `finally` that may fire on a path where `start` never did."""

    cfg = _cfg()
    pool = PricingPool(
        [_request(1, (-4, 0), (4, 0), cfg)], cfg, _params(n_pricing_workers=1),
        StaticTerminalCatalog((), cfg),
    )
    pool.close()          # never started
    pool.close()          # idempotent


def test_a_pool_that_raised_mid_sweep_refuses_another(monkeypatch):
    """An exception leaves the lanes half-consumed, exactly as an early return does.

    The incomplete-sweep path already poisons; this is the same hazard reached the other
    way -- the merge's own consistency check, or a task raising. Without it a caller that
    caught the error and retried would get sweep k's leftovers merged into sweep k+1.
    """

    cfg = _cfg()
    requests = [_request(1, (-4, 0), (4, 0), cfg), _request(2, (0, -4), (0, 4), cfg)]
    with PricingPool(
        requests, cfg, _params(n_pricing_workers=2), StaticTerminalCatalog((), cfg)
    ) as pool:
        pool.start()
        monkeypatch.setattr(
            pricing_pool, "_accepted_prefix",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("merge blew up")),
        )
        with pytest.raises(RuntimeError, match="merge blew up"):
            pool.run_sweep([1, 2], {}, {1: 0.0, 2: 0.0}, {}, None)
        monkeypatch.undo()
        with pytest.raises(RuntimeError, match="no longer usable"):
            pool.run_sweep([1, 2], {}, {1: 0.0, 2: 0.0}, {}, None)


def test_a_pool_that_fails_to_start_does_not_leak_its_workers(monkeypatch):
    """A partial `start()` must leave what it built reachable by `close()`.

    `MemoryError` while spawning the nth worker is the realistic trigger -- it is what the
    `n_pricing_workers` ceiling exists for -- and it is precisely when leaking the other
    n-1 worker processes would hurt most.
    """

    cfg = _cfg()
    requests = [_request(i, (-4, 0), (4, 0), cfg) for i in (1, 2, 3)]
    pool = PricingPool(
        requests, cfg, _params(n_pricing_workers=3), StaticTerminalCatalog((), cfg)
    )
    real_pool = mp.get_context("spawn").Pool
    calls = []

    def explode_on_the_third(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise MemoryError("no room for another worker")
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(
        pricing_pool.mp, "get_context",
        lambda _name: type("Ctx", (), {"Pool": staticmethod(explode_on_the_third)})(),
    )
    with pytest.raises(MemoryError):
        pool.start()
    # `close()` ran on the way out, so nothing is left holding worker processes.
    assert pool._pools is None
    pool.close()
