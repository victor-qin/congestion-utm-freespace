"""Fan one iteration's pricing sweep across worker processes.

The sweep's subproblems are independent given the iteration's duals -- each one searches a
different flight's DAG and touches nothing shared -- so the only thing parallelism can
change is *when* results arrive. This module exists to make sure that is all it changes.

**Determinism, two rules, both load-bearing.**

1. *Longest completed prefix.* The sequential sweep ``break``s at the first
   :class:`PricingTimeout`, so flights after it are never priced. A pool has no such
   ordering: a worker can finish flight 40 while flight 5 times out. Results are collected
   by ``pricing_order`` index and everything at or past the first timeout is DISCARDED, so
   the accepted set has the same SHAPE the sequential loop would have produced -- a prefix
   of ``pricing_order``, never a set with holes in it.

   Same shape, not the same prefix. ``PricingTimeout`` fires off a wall clock, and
   ``solver`` fixes one absolute ``pricing_deadline`` per iteration, so sequentially flight
   *k* is reached only after the cumulative single-core time of the flights before it while
   a pool runs them concurrently. More of them therefore finish before that same instant,
   and the prefix a pool keeps is generally LONGER. That is strictly more pricing done
   inside the budget rather than a defect, but it is a different column set, so it is only
   correct to call a pool answer-identical on a sweep that never times out. Parity runs
   accordingly pin ``n_workers=0``.
2. *Index order, not completion order.* ``master.upper_bound`` sums the reduced costs with
   plain ``sum`` (master.py), not ``math.fsum``, and float addition is not associative --
   so appending them as workers finish perturbs the global bound at ulp level and the
   solve can terminate an iteration earlier or later. They are appended by index.

``n_workers=0`` runs the sequential loop in-process and is byte-identical to no pool at
all; it is both the default and the parity baseline. The pool is OPT-IN because its memory
is linear in workers -- 3.9 GB in-process against 12.5 GB at 4 workers and 22.7 GB at 8, on
a 50-flight density instance -- and an OOM-killed worker hangs this sweep rather than
failing it (see the deadlock note below). Speed is not the constraint: 3.5x at 4 workers.

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

**A worker KILLED mid-task no longer hangs the sweep.** ``mp.Pool`` does not fail a task
whose worker died: ``_repopulate_pool_static`` starts a replacement and the task's result
is simply never produced, so an unguarded ``imap`` waits for it forever -- and the solver's
own time limit never fires, because the parent is blocked inside the sweep rather than
checking a clock. The realistic trigger is the OOM killer, which is reachable rather than
hypothetical: memory is linear in workers.

:func:`_results_before_deadline` bounds the wait by the caller's own ``pricing_deadline``
and reports a lost result as that flight timing out, so the sweep degrades to a short
prefix instead of blocking. Bounding it by a deadline the caller already owns is what makes
this safe to add without inventing a grace period, and without reaching for the pool
implementation that deadlocks on this interpreter.

A worker that fails in its *initializer* is a different failure with the same symptom, and
is handled separately: ``mp.Pool`` calls the initializer outside the try/except that wraps
a task, so :func:`_init_worker` records the traceback and :func:`_price_one` re-raises it.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from ...config import SimConfig
from ...types import FlightRequest
from .network import StaticTerminalCatalog, build_flight_graph
from .params import ColGenParams
from .pricing import (
    DualView,
    PricingTimeout,
    clear_search_record,
    kernel_stats,
    last_search_record,
    price_flight,
)
from .translate import Column

__all__ = [
    "SweepResult",
    "price_sweep",
]


# HOW WIDE TO FAN THE SWEEP LIVES ON `ColGenParams`, as `n_pricing_workers` (0 is the
# sequential loop) and `pricing_chunksize`.  There used to be a `ParallelPricingConfig`
# dataclass here that `solve` took as a separate `parallel=` keyword, and it was pure
# duplication: `price_sweep` already receives the params object, so the second one carried
# no information the first did not.  It also cost a real bug -- two defaults that could
# disagree, with `batch.py` mapping an explicit 0 to `None` and `price_sweep` resolving
# `None` as "whatever the dataclass defaults to" rather than "sequential", so raising that
# default would have silently turned `--colgen-workers 0` into a pool.  Both the worker
# ceiling and the chunksize validation moved to `ColGenParams.__post_init__` with it.


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
    # `pricing.kernel_stats()` summed ACROSS workers, since each keeps its own per-process
    # tally: exact-pricing calls, how many fell back, the label-pool restarts and declines,
    # and one `declined_<reason>` key per `pricing.Declined` member that fired.  A fallback
    # is invisible downstream -- same column, same objective, 3-4.5x the time -- so these are
    # the only signal a production run gets.
    #
    # ONE DICT rather than a field per counter, deliberately.  The two that used to be
    # fields grew to four and then to a reason histogram whose keys depend on the instance;
    # as positional fields that is a constructor argument per counter, in a dataclass built
    # positionally at four sites and hand-built in the tests.
    kernel_counters: Counter[str] = field(default_factory=Counter)
    #: One record per ACCEPTED flight, in the same index order as `flight_ids`, plus the
    #: timed-out flight when there is one.  Diagnostics only -- nothing in the solve reads
    #: these, and an empty tuple is a valid sweep.
    #:
    #: They exist because `task_total_s` is a SUM, and a sum cannot answer the question a
    #: pool actually poses: a sweep can never finish faster than its slowest single task,
    #: so "which flight is the straggler, and was it one of the flights that improved"
    #: decides whether skip-filtering the cheap flights would buy any wall time at all.
    #: Measured at 500 flights, the largest task is >=26x the mean, so this is the
    #: difference between attacking the binding term and a non-binding one.
    flight_records: tuple[dict[str, Any], ...] = ()

    @property
    def kernel_priced(self) -> int:
        return self.kernel_counters.get("priced", 0)

    @property
    def kernel_fell_back(self) -> int:
        return self.kernel_counters.get("fell_back", 0)

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

    Returns ``(flight_id, priced, rc, column, task_s, counter_deltas)`` -- the shape
    :func:`_accepted_prefix` reduces, and the reason that function takes a sequence of these
    rather than a pool.

    The counters travel as ONE dict rather than a trailing int each, because there are now
    four of them plus a `declined_<reason>` key per cause, and the reachable set of those
    depends on the instance.

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
    # Cleared, not merely read after: a flight that declines BEFORE reaching the kernel
    # leaves the PREVIOUS flight's record in place, and attributing one flight's 67M labels
    # to another is worse than reporting nothing.
    clear_search_record()
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
        # The record still ships: a flight that timed out mid-search is the most
        # interesting row in a straggler hunt, and it is exactly the one the sequential
        # loop never produces a number for.
        return (
            flight_id, False, 0.0, None, time.perf_counter() - started, {},
            last_search_record(),
        )
    after = kernel_stats()
    # Subtraction over the UNION of keys, not over `before`'s: a `declined_<reason>` key
    # appears the first time that cause fires, so it exists in `after` and not in `before`.
    return (
        flight_id,
        True,
        float(reduced_cost),
        column,
        time.perf_counter() - started,
        {key: value - before.get(key, 0) for key, value in after.items()},
        last_search_record(),
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
) -> SweepResult:
    """Price every flight in ``pricing_order``, sequentially or across processes.

    Width comes from ``params.n_pricing_workers``; 0 is the sequential loop and is
    byte-identical to no pool at all.  ``graphs`` is used only by the sequential path;
    workers build their own from ``requests`` (cheaper than pickling, see the module
    docstring).
    """

    if params.n_pricing_workers == 0:
        return _sweep_sequential(
            pricing_order, graphs, cfg, params, dual_view, flight_duals,
            known_columns, deadline,
        )
    return _sweep_parallel(
        pricing_order, requests, cfg, params, catalog, duals, flight_duals,
        known_columns, deadline,
    )


def _sweep_sequential(
    pricing_order, graphs, cfg, params, dual_view, flight_duals, known_columns, deadline
) -> SweepResult:
    """The original in-process loop, kept verbatim as the parity baseline."""

    flight_ids: list[int] = []
    reduced_costs: list[float] = []
    columns: list[Column | None] = []
    task_total_s = 0.0
    before = Counter(kernel_stats())
    # Built here too, and not only in the pool: the two arms must report the same SHAPE or
    # a diagnostic that only exists under `--colgen-workers N` cannot be sanity-checked
    # against the loop it is supposed to reproduce.
    records: list[dict[str, Any]] = []
    sweep_started = time.perf_counter()
    for flight_id in pricing_order:
        task_started = time.perf_counter()
        clear_search_record()
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
            records.append(_flight_record(
                flight_id, time.perf_counter() - task_started, priced=False
            ))
            return SweepResult(
                tuple(flight_ids), tuple(reduced_costs), tuple(columns), flight_id,
                task_total_s, time.perf_counter() - sweep_started,
                Counter(kernel_stats()) - before, tuple(records),
            )
        task_s = time.perf_counter() - task_started
        records.append(_flight_record(flight_id, task_s, priced=True))
        task_total_s += task_s
        flight_ids.append(flight_id)
        reduced_costs.append(float(reduced_cost))
        columns.append(column)
    # One worker's worth of work, by definition -- which is what makes it the denominator
    # the parallel arm is compared against.
    return SweepResult(
        tuple(flight_ids), tuple(reduced_costs), tuple(columns), None,
        task_total_s, time.perf_counter() - sweep_started,
        Counter(kernel_stats()) - before, tuple(records),
    )


def _flight_record(flight_id: int, task_s: float, *, priced: bool, search=None) -> dict:
    """One flight's diagnostic row: its clock, plus whatever the kernel recorded.

    `search` is `pricing.last_search_record()` -- passed in by the pool arm, which read it
    inside the worker, and read here directly by the sequential arm. Empty when the flight
    never reached the compiled search at all, which is itself the answer to "why was this
    one cheap".
    """

    record = {"flight_id": int(flight_id), "task_s": float(task_s), "priced": bool(priced)}
    record.update(last_search_record() if search is None else search)
    # After the update, so a stale `flight_id` in the search record can never rename the
    # flight this row is about.
    record["flight_id"] = int(flight_id)
    return record


def _sweep_parallel(
    pricing_order, requests, cfg, params, catalog, duals, flight_duals,
    known_columns, deadline
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
        processes=params.n_pricing_workers, initializer=_init_worker, initargs=init_args
    ) as pool:
        # `imap` and not `imap_unordered`: results must arrive in `pricing_order` index
        # order so the accepted prefix is the one the sequential loop would have produced,
        # and so the reduced costs reach `master.upper_bound`'s non-associative `sum` in a
        # fixed order. See the module docstring.
        # Chunked HERE rather than by `imap`, and that is a correctness requirement.
        # `Pool.imap` only returns an `IMapIterator` when `chunksize == 1`; above 1 it
        # returns `(item for chunk in result for item in chunk)` -- a plain generator, which
        # has no `.next(timeout)`, so the deadline guard below would raise `AttributeError`
        # before yielding a single result. Batching the work ourselves and asking `imap` for
        # chunks of one keeps the iterator type fixed while giving the identical work
        # distribution `Pool._get_tasks` would have produced.
        chunks = [
            pricing_order[i:i + params.pricing_chunksize]
            for i in range(0, len(pricing_order), params.pricing_chunksize)
        ]
        accepted = _accepted_prefix(
            _results_before_deadline(
                pool.imap(_price_batch, chunks, 1), chunks, deadline
            )
        )
    return replace(accepted, wall_s=time.perf_counter() - sweep_started)


def _price_batch(flight_ids):
    """Price one chunk of flights in a worker, in order.

    Module level because `spawn` pickles by qualified name. Exists so `_sweep_parallel` can
    hand `imap` a chunksize of one whatever `pricing_chunksize` is -- see the note there.
    """

    return [_price_one(flight_id) for flight_id in flight_ids]


def _results_before_deadline(results, chunks, deadline):
    """Yield ``imap`` results, giving up on the one the parent is still waiting for at
    ``deadline``.

    Without this the sweep can wait forever, and the solver's own time limit never gets a
    chance to fire. ``mp.Pool`` does not fail a task whose worker DIED: the sentinel thread
    starts a replacement and the lost task's result is simply never produced, so the
    ordered iterator blocks on a value nobody will send. The realistic trigger is the OOM
    killer, which is reachable rather than hypothetical here -- memory is linear in workers
    (22.7 GB at 8 on density x100) and that is what the ``n_workers`` ceiling exists for.

    The bound is the caller's OWN absolute deadline, not a grace period invented here.
    That distinction is the reason this is safe to add: past ``deadline`` the sweep was
    going to stop regardless, so nothing is abandoned that would otherwise have been kept,
    and no new number needs justifying.

    A give-up is reported as **that flight timing out**, which is deliberate: it re-enters
    the discard rule the sequential loop already defines, so the caller gets a short prefix
    with ``timeout_flight_id`` set and ``complete`` False. Returning a truncated prefix that
    merely *looked* complete would be far worse than the hang -- `master.upper_bound` would
    take a bound over flights that were never priced.

    ``results`` yields CHUNKS, because `imap` only hands back an ``IMapIterator`` -- the one
    type with ``.next(timeout)`` -- when its own chunksize is 1. `_sweep_parallel` therefore
    batches the work itself and asks for chunks of one, so this stays ordered and the k-th
    chunk is ``chunks[k]``, which is what lets a lost result be named at all.

    With ``deadline=None`` this blocks exactly as before, which is the documented behaviour
    for a sweep that was given no time limit -- `solve` always passes one.
    """

    for index in range(len(chunks)):
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            chunk = results.next(remaining)
        except StopIteration:
            return
        except mp.TimeoutError:
            # Exactly the shape `_price_one` returns for a `PricingTimeout`, so the reducer
            # needs no special case: no seconds, no counters, `priced` False -- and an
            # EMPTY search record, because the worker that would have filled it is the one
            # that never reported back.  Named with the chunk's FIRST flight: nothing in it
            # reported, and the accepted prefix ends at the earliest of them either way.
            yield chunks[index][0], False, 0.0, None, 0.0, {}, {}
            return
        yield from chunk


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
    # `Counter.update` ADDS where `dict.update` would REPLACE, which is the whole reason
    # this is a Counter: a plain dict here would silently report only the last task's tally.
    counters: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for flight_id, priced, reduced_cost, column, task_s, deltas, search in results:
        if not priced:
            # Past the first gap nothing is accepted, so returning here stops consuming --
            # which abandons the outstanding tasks, and leaving the caller's `with` block
            # terminates the pool.  The timed-out task's own seconds are deliberately not
            # added: they are not work the sweep kept.
            # The timed-out flight's row IS kept even though its seconds are not: it is
            # the one the sweep stopped for, so a straggler hunt needs it most.
            records.append(_flight_record(flight_id, task_s, priced=False, search=search))
            return SweepResult(
                tuple(flight_ids), tuple(reduced_costs), tuple(columns), flight_id,
                task_total_s, 0.0, counters, tuple(records),
            )
        records.append(_flight_record(flight_id, task_s, priced=True, search=search))
        task_total_s += task_s
        counters.update(deltas)
        flight_ids.append(flight_id)
        reduced_costs.append(float(reduced_cost))
        columns.append(column)
    return SweepResult(
        tuple(flight_ids), tuple(reduced_costs), tuple(columns), None,
        task_total_s, 0.0, counters, tuple(records),
    )
