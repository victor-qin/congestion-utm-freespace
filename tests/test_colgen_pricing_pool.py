"""The parallel pricing sweep must change work, never answers."""
from __future__ import annotations

import inspect
import os
import time

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.colgen import pricing_pool
from freespace_sim.planner.colgen.network import StaticTerminalCatalog, build_flight_graph
from freespace_sim.planner.colgen.params import ColGenParams
from freespace_sim.planner.colgen.pricing import DualView
from freespace_sim.planner.colgen.pricing_pool import (
    SweepResult,
    _accepted_prefix,
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

    assert ColGenParams().n_pricing_workers == 4
    assert ColGenParams(n_pricing_workers=0).n_pricing_workers == 0
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
        (7, True, 0.5, None, 1.0, {"priced": 1}),
        (3, False, 0.0, None, 0.25, {}),
        (9, True, 4.0, None, 8.0, {"priced": 1, "fell_back": 1}),
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
        (7, True, 0.5, None, 1.0, {"priced": 1}),
        (3, True, 0.25, None, 0.5,
         {"priced": 1, "fell_back": 1, "declined_label_pool_exhausted": 1}),
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
        [_request(1, (-4, 0), (4, 0), cfg)], cfg, _params(),
        StaticTerminalCatalog((), cfg), {}, {1: 0.0}, {}, None,
    )
    with pytest.raises(RuntimeError, match="failed to initialise") as raised:
        pricing_pool._price_one(1)
    # The original cause, carried as text because the exception object itself has to
    # survive being pickled back to the parent.
    assert "MemoryError: label pool" in str(raised.value)


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
