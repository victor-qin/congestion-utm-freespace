"""Fan one iteration's pricing sweep across worker processes.

The sweep's subproblems are independent given the iteration's duals -- each one searches a
different flight's DAG and touches nothing shared -- so the only thing parallelism can
change is *when* results arrive. This module exists to make sure that is all it changes.

**Determinism, two rules, both load-bearing.**

1. *Longest completed prefix.* The sequential sweep ``break``s at the first
   :class:`PricingTimeout`, so flights after it are never priced. A pool has no such
   ordering: a worker can finish flight 40 while flight 5 times out. Results are collected
   by ``pricing_order`` index and everything at or past the first timeout is DISCARDED, so
   the accepted set is exactly the prefix the sequential loop would have produced.
2. *Index order, not completion order.* ``master.upper_bound`` sums the reduced costs with
   plain ``sum`` (master.py), not ``math.fsum``, and float addition is not associative --
   so appending them as workers finish perturbs the global bound at ulp level and the
   solve can terminate an iteration earlier or later. They are appended by index.

``n_workers=0`` runs the sequential loop in-process and is byte-identical to no pool at
all; it is the default and the parity baseline.

**Processes, not threads.** Pricing is ~90% Python outside the numba kernel, so threads
would contend on the GIL for the part that is not compiled. The costs processes bring are
measured rather than assumed: a ``FlightGraph`` pickles to 38.6 KB but REBUILDS in 0.11 ms
against 0.40 ms to pickle, so workers are handed the flight *requests* and build their own
graphs. The caches (`_search_cache`) do not survive pickling either way, so nothing is lost
by rebuilding that would have been kept by shipping.

**``mp.Pool``, never ``ProcessPoolExecutor``.** The latter deadlocks on CPython 3.14.2 when
``max_tasks_per_child`` fires -- see ``[[colgen-parallel-pricing-pool]]``. For the same
reason no worker-recycling budget is set here.

**macOS spawn hazard, deliberately accepted.** Under the ``spawn`` start method a caller
script that does work at module level rather than under ``if __name__ == "__main__":`` dies
in ``_check_not_importing_main``. Every shipped entry point already guards its main.

**A worker KILLED mid-task hangs the sweep, and that is accepted too.** ``mp.Pool`` does
not fail a task whose worker died: ``_repopulate_pool_static`` starts a replacement and the
task's result is simply never produced, so ``imap`` waits for it forever. The realistic
trigger is the OOM killer, which is what the ``n_workers`` ceiling below exists to keep out
of reach, and the alternative pool implementation that would detect it is the one that
deadlocks. A worker that fails in its *initializer* is NOT accepted -- that failure is
reachable rather than hypothetical, so :func:`_init_worker` reports it instead of raising.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, replace
from typing import Any

from ...config import SimConfig
from ...types import FlightRequest
from .network import StaticTerminalCatalog, build_flight_graph
from .params import ColGenParams
from .pricing import DualView, PricingTimeout, kernel_stats, price_flight
from .translate import Column

__all__ = [
    "ParallelPricingConfig",
    "SweepResult",
    "price_sweep",
]


@dataclass(frozen=True, slots=True)
class ParallelPricingConfig:
    """How wide to fan the sweep. ``n_workers=0`` is the sequential loop."""

    n_workers: int = 0
    chunksize: int = 1

    def __post_init__(self) -> None:
        for name in ("n_workers", "chunksize"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.n_workers < 0:
            raise ValueError("n_workers must be non-negative")
        if self.chunksize < 1:
            raise ValueError("chunksize must be positive")
        # An upper bound, because the failure past it is not an error message.  Each worker
        # rebuilds every graph and carries its own label pool -- roughly 1.5 GB on density
        # -- so an over-large count from a config file OOMs the host rather than running
        # slowly.  Measured, more lanes stop paying long before here anyway: 8 and 12
        # workers were within noise of each other, because added lanes add memory-system
        # contention as fast as they add throughput.
        ceiling = 4 * (os.cpu_count() or 1)
        if self.n_workers > ceiling:
            raise ValueError(
                f"n_workers={self.n_workers} exceeds {ceiling} (4x this host's "
                f"{os.cpu_count()} cores); each worker holds its own label pool"
            )


@dataclass(frozen=True, slots=True)
class SweepResult:
    """One sweep's accepted prefix, in ``pricing_order`` index order.

    ``reduced_costs`` and ``columns`` are the same length and cover only the accepted
    prefix; ``timeout_flight_id`` names the flight that ended it, or is ``None`` when the
    sweep completed.
    """

    flight_ids: tuple[int, ...]
    reduced_costs: tuple[float, ...]
    columns: tuple[Column | None, ...]
    timeout_flight_id: int | None
    # Seconds of WORK the accepted tasks actually did, against the wall the caller waited.
    # The two separate the only losses a pool can suffer: `wall * n_workers - task_total`
    # is idle worker time, which is scheduling loss (a straggler nobody can fill in behind),
    # while the gap between `task_total` and the sequential sweep's total is dispatch
    # overhead.  Only the second is what `chunksize` can attack, and telling them apart
    # from wall clock alone is impossible.
    task_total_s: float = 0.0
    # Wall the caller waited for the sweep, measured inside `price_sweep` so it covers pool
    # construction and teardown -- the costs a per-sweep pool actually pays.
    wall_s: float = 0.0
    # Exact-pricing calls this sweep, and how many fell back to the Python reference.  A
    # fallback is invisible downstream -- same column, same objective, 3-4.5x the time -- so
    # a nonzero count here is the only signal a production run gets.  Summed ACROSS workers,
    # since each keeps its own per-process tally.
    kernel_priced: int = 0
    kernel_fell_back: int = 0

    @property
    def complete(self) -> bool:
        return self.timeout_flight_id is None

    def efficiency(self, n_workers: int) -> float:
        """Fraction of the pool's worker-seconds that did real work.

        UNDERSTATES after a timeout, and unavoidably so: the sweep stops reading results at
        the first gap, so tasks that completed but land past it are never observed, while
        the wall clock already includes the time spent producing them.  On a sweep that
        completed (``timeout_flight_id is None``) the figure is exact.
        """

        if n_workers <= 0 or self.wall_s <= 0.0:
            return 1.0
        return self.task_total_s / (self.wall_s * n_workers)


# --------------------------------------------------------------------------- worker side

# Populated once per worker per sweep by `_init_worker`. A plain module global because that
# is the only state a spawned worker can carry between tasks without re-pickling it.
_WORKER: dict[str, Any] = {}


def _init_worker(
    requests: list[FlightRequest],
    cfg: SimConfig,
    params: ColGenParams,
    catalog: StaticTerminalCatalog,
    duals: dict,
    flight_duals: dict[int, float],
    known_columns: dict[int, Column],
    deadline: float | None,
) -> None:
    """Rebuild this worker's view of the batch. Runs once per worker per sweep.

    The catalog is shipped rather than rebuilt from raw terminals: every graph in a solve
    must see the identical wall catalogue, and re-deriving it here would be a second source
    of truth for something the parent already snapshotted.  It pickles to 12.2 KB.

    A failure here is RECORDED, never raised, and that is a hang the parent would otherwise
    have no way out of.  ``mp.Pool`` calls the initializer outside the try/except that wraps
    a task, so an exception kills the worker before it reads its first one,
    ``_repopulate_pool_static`` immediately starts a replacement, and the replacement dies
    on the same argument -- forever, at full CPU, while the parent's ``imap`` waits for a
    result no worker will ever produce and nothing is printed.  Rebuilding every graph is
    also the largest allocation a worker makes, so ``MemoryError`` -- the failure the
    ``n_workers`` ceiling is about -- lands exactly here.  :func:`_price_one` re-raises it
    from the first task instead, which reaches the caller as an ordinary traceback.
    """

    _WORKER.clear()
    try:
        _WORKER["graphs"] = {
            request.flight_id: build_flight_graph(request, cfg, catalog, params)
            for request in requests
        }
        _WORKER["cfg"] = cfg
        _WORKER["params"] = params
        # `DualView` is rebuilt here rather than pickled so the worker owns its own caches.
        _WORKER["dual_view"] = DualView(duals, cfg)
        _WORKER["flight_duals"] = flight_duals
        _WORKER["known_columns"] = known_columns
        # `time.monotonic` is a system-wide clock on both Linux and macOS, so a deadline
        # taken in the parent is directly comparable here. It would NOT be across hosts.
        _WORKER["deadline"] = deadline
    except Exception:
        # The traceback as TEXT, because the exception object itself may not survive the
        # trip back: it is re-raised below as a `RuntimeError`, which always pickles.
        _WORKER["init_error"] = traceback.format_exc()


def _price_one(flight_id: int):
    """Price one flight in a worker.

    Returns ``(flight_id, priced, rc, column, task_s, n_priced, n_fell_back)`` -- the shape
    :func:`_accepted_prefix` reduces, and the reason that function takes a sequence of these
    rather than a pool.

    ``priced`` is an explicit BOOLEAN and not an identity sentinel, which is a correctness
    requirement rather than a style choice: a module-level ``object()`` pickles happily and
    arrives in the parent as a DIFFERENT instance, so ``result is _SENTINEL`` is always
    False across a process boundary.  The sequential path would never have caught it --
    nothing is pickled there -- so the bug would surface only on the first production
    timeout under a pool.

    A timeout is reported rather than raised for a related reason: propagating it would
    make `mp.Pool` pickle and re-raise the exception in the parent, and a timeout here is
    an ordinary outcome, not a failure.
    """

    init_error = _WORKER.get("init_error")
    if init_error is not None:
        # This worker never built its batch view.  Raising from here rather than from the
        # initializer is the whole point -- see `_init_worker`.
        raise RuntimeError(f"pricing worker failed to initialise:\n{init_error}")
    # Deltas, not absolutes: the worker's tally is cumulative across every task it has run,
    # so shipping the absolute would double-count on the second task and beyond.
    before = kernel_stats()
    started = time.perf_counter()
    try:
        reduced_cost, column = price_flight(
            _WORKER["graphs"][flight_id],
            _WORKER["dual_view"],
            _WORKER["flight_duals"][flight_id],
            _WORKER["cfg"],
            _WORKER["params"],
            known_column=_WORKER["known_columns"].get(flight_id),
            deadline=_WORKER["deadline"],
        )
    except PricingTimeout:
        return flight_id, False, 0.0, None, time.perf_counter() - started, 0, 0
    after = kernel_stats()
    return (
        flight_id,
        True,
        float(reduced_cost),
        column,
        time.perf_counter() - started,
        after["priced"] - before["priced"],
        after["fell_back"] - before["fell_back"],
    )


# --------------------------------------------------------------------------- parent side


def price_sweep(
    pricing_order: list[int],
    requests: list[FlightRequest],
    graphs: dict,
    cfg: SimConfig,
    params: ColGenParams,
    catalog: StaticTerminalCatalog,
    duals: dict,
    dual_view: DualView,
    flight_duals: dict[int, float],
    known_columns: dict[int, Column],
    *,
    deadline: float | None = None,
    config: ParallelPricingConfig | None = None,
) -> SweepResult:
    """Price every flight in ``pricing_order``, sequentially or across processes.

    ``graphs`` is used only by the sequential path; workers build their own from
    ``requests`` (cheaper than pickling, see the module docstring).
    """

    config = config or ParallelPricingConfig()
    if config.n_workers == 0:
        return _sweep_sequential(
            pricing_order, graphs, cfg, params, dual_view, flight_duals,
            known_columns, deadline,
        )
    return _sweep_parallel(
        pricing_order, requests, cfg, params, catalog, duals, flight_duals,
        known_columns, deadline, config,
    )


def _sweep_sequential(
    pricing_order, graphs, cfg, params, dual_view, flight_duals, known_columns, deadline
) -> SweepResult:
    """The original in-process loop, kept verbatim as the parity baseline."""

    flight_ids: list[int] = []
    reduced_costs: list[float] = []
    columns: list[Column | None] = []
    task_total_s = 0.0
    before = kernel_stats()
    sweep_started = time.perf_counter()
    for flight_id in pricing_order:
        task_started = time.perf_counter()
        try:
            reduced_cost, column = price_flight(
                graphs[flight_id],
                dual_view,
                flight_duals[flight_id],
                cfg,
                params,
                known_column=known_columns.get(flight_id),
                deadline=deadline,
            )
        except PricingTimeout:
            after = kernel_stats()
            return SweepResult(
                tuple(flight_ids), tuple(reduced_costs), tuple(columns), flight_id,
                task_total_s, time.perf_counter() - sweep_started,
                after["priced"] - before["priced"],
                after["fell_back"] - before["fell_back"],
            )
        task_total_s += time.perf_counter() - task_started
        flight_ids.append(flight_id)
        reduced_costs.append(float(reduced_cost))
        columns.append(column)
    after = kernel_stats()
    # One worker's worth of work, by definition -- which is what makes it the denominator
    # the parallel arm is compared against.
    return SweepResult(
        tuple(flight_ids), tuple(reduced_costs), tuple(columns), None,
        task_total_s, time.perf_counter() - sweep_started,
        after["priced"] - before["priced"],
        after["fell_back"] - before["fell_back"],
    )


def _sweep_parallel(
    pricing_order, requests, cfg, params, catalog, duals, flight_duals,
    known_columns, deadline, config
) -> SweepResult:
    """Fan the sweep across a pool built and torn down for this sweep alone.

    The pool is per-sweep because the duals change every iteration and they reach the
    workers through the initializer -- once per worker rather than once per task, which is
    the difference between pickling them ~n_workers times and ~n_flights times.

    The initializer's largest payload is NOT the duals, though: it is `known_columns`, one
    `Column` per flight at ~13.7 KB with 619 claims, so ~13.7 MB per worker at 1,000
    flights and linear in both flights and workers.  Invisible against a sweep of hundreds
    of seconds, and the first term that would bite at several thousand flights -- at which
    point the fix is to send only the columns each worker's own flights need, which
    requires a static rather than dynamic assignment.
    """

    # `spawn` explicitly rather than by platform default: `fork` would inherit the parent's
    # numba runtime and thread state, which is exactly the combination that is unsafe.
    ctx = mp.get_context("spawn")
    init_args = (
        list(requests), cfg, params, catalog, duals, flight_duals,
        dict(known_columns), deadline,
    )
    # Started before the pool exists on purpose: spawning workers, re-importing numba and
    # rebuilding graphs are real costs of a per-sweep pool, and excluding them would report
    # an efficiency the caller never experiences.
    sweep_started = time.perf_counter()
    with ctx.Pool(
        processes=config.n_workers, initializer=_init_worker, initargs=init_args
    ) as pool:
        # `imap` and not `imap_unordered`: results must arrive in `pricing_order` index
        # order so the accepted prefix is the one the sequential loop would have produced,
        # and so the reduced costs reach `master.upper_bound`'s non-associative `sum` in a
        # fixed order. See the module docstring.
        accepted = _accepted_prefix(
            pool.imap(_price_one, pricing_order, config.chunksize)
        )
    return replace(accepted, wall_s=time.perf_counter() - sweep_started)


def _accepted_prefix(results) -> SweepResult:
    """Reduce worker results, IN THE ORDER GIVEN, to the prefix the sequential loop keeps.

    Split out from :func:`_sweep_parallel` because this, and not the pool, is the rule the
    module exists to enforce -- and a live pool cannot demonstrate it.  ``deadline`` is a
    wall clock, so a sweep is either past it (every flight times out and the prefix is
    empty for reasons that would survive the rule being deleted) or comfortably inside it
    (nothing times out at all).  The shape the rule is FOR -- a flight that timed out with
    COMPLETED flights on both sides of it, which is what a pool produces and a sequential
    loop never sees -- has no reproducible wall-clock recipe, but as a sequence of results
    it is three lines of test.

    ``wall_s`` is left at zero: the caller owns the clock, because it has to start before
    the pool exists.
    """

    flight_ids: list[int] = []
    reduced_costs: list[float] = []
    columns: list[Column | None] = []
    task_total_s = 0.0
    kernel_priced = 0
    kernel_fell_back = 0
    for flight_id, priced, reduced_cost, column, task_s, n_priced, n_fell_back in results:
        if not priced:
            # Past the first gap nothing is accepted, so returning here stops consuming --
            # which abandons the outstanding tasks, and leaving the caller's `with` block
            # terminates the pool.  The timed-out task's own seconds are deliberately not
            # added: they are not work the sweep kept.
            return SweepResult(
                tuple(flight_ids), tuple(reduced_costs), tuple(columns), flight_id,
                task_total_s, 0.0, kernel_priced, kernel_fell_back,
            )
        task_total_s += task_s
        kernel_priced += n_priced
        kernel_fell_back += n_fell_back
        flight_ids.append(flight_id)
        reduced_costs.append(float(reduced_cost))
        columns.append(column)
    return SweepResult(
        tuple(flight_ids), tuple(reduced_costs), tuple(columns), None,
        task_total_s, 0.0, kernel_priced, kernel_fell_back,
    )
