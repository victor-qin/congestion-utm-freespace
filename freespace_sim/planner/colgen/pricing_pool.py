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

**The pool is SOLVE-scoped, and the worker assignment is why that is worth anything.**
:class:`PricingPool` holds W single-worker ``mp.Pool`` instances and pins each flight to one
of them for the whole solve (:func:`_worker_assignment`), so a worker keeps the state it
derived for its own flights instead of deriving it again every iteration. Per-sweep duals
arrive as a task rather than through the initializer, stamped with an epoch that
:func:`_price_one` refuses to price against if it does not match -- because ``mp.Pool``
replaces a dead worker by re-running the ORIGINAL initargs, and the replacement would
otherwise price with no duals at all and return a plausible wrong number.

**Two costs this removes, and the second is the one that was invisible.**

* A flight's compiled packing (``dp_prepare.prepared_for``) is a pure function of its graph
  and config, built lazily on first pricing touch and kept on ``_search_cache``. A worker
  that dies at end of sweep takes it with it: 989 rebuilds a sweep at 1000 flights, ~184 ms
  each, ~180 s of worker CPU per iteration to reconstruct something already computed.
* Worker launch is parent-SERIAL. ``_repopulate_pool_static`` starts workers in a plain
  loop and ``spawn`` re-pickles the initargs for each, measured at ~4.2-4.8 s per worker,
  so idle worker-seconds grew as ``W(W-1)/2`` and not with W. That is why 8 workers ran at
  79% efficiency where 16 ran at 49%, and why the second half of a 16-core machine bought
  almost nothing. Paid once per solve, it stops being the binding term.

**Processes, not threads.** Pricing is ~90% Python outside the numba kernel, so threads
would contend on the GIL for the part that is not compiled. The costs processes bring are
measured rather than assumed: a ``FlightGraph`` pickles to 38.6 KB but REBUILDS in 0.11 ms
against 0.40 ms to pickle, so workers are handed the flight *requests* and build their own
graphs.

That last argument used to end "the caches (``_search_cache``) do not survive pickling
either way, so nothing is lost by rebuilding that would have been kept by shipping", and
that was the premise this module got wrong. It is true of everything the comparison covered
and false of the expensive thing, because the packing is not on the graph when the graph is
weighed: it is built lazily, on first pricing touch, long after transport. What the graph
costs to rebuild (0.11 ms) and what its cache costs to rebuild (~184 ms) are three orders
of magnitude apart, and only the first was ever measured. Keeping the WORKER is what keeps
the cache; shipping the graph never could.

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

:func:`_sweep_results` bounds the wait by the caller's own ``pricing_deadline`` and reports
a lost result as that flight timing out, so the sweep degrades to a short prefix instead of
blocking. Bounding it by a deadline the caller already owns is what makes this safe to add
without inventing a grace period, and without reaching for the pool implementation that
deadlocks on this interpreter. It also merges the per-worker result
streams back into ``pricing_order`` order, which rule 2 requires and which a queue per
worker no longer gives for free.

A worker that fails in its *initializer* is a different failure with the same symptom, and
is handled separately: ``mp.Pool`` calls the initializer outside the try/except that wraps
a task, so :func:`_init_worker` records the traceback and :func:`_price_one` re-raises it.
"""
from __future__ import annotations

import collections
import itertools
import multiprocessing as mp
import os
import pickle
import time
import traceback
import uuid
import warnings
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
    "PricingPool",
    "StalePricingWorker",
    "SweepResult",
    "price_sweep",
]


class StalePricingWorker(RuntimeError):
    """A worker was handed a task from a sweep whose state it does not hold.

    Carries a single string so it survives the trip back through ``imap`` intact.

    There is exactly one way to reach this, and it is why the class exists: ``mp.Pool``
    silently replaces a worker that DIED, and ``_repopulate_pool_static`` re-runs the
    ORIGINAL ``initargs``.  Under a solve-scoped pool those initargs carry solve constants
    only -- the duals arrive per sweep, as a task -- so the replacement comes back with no
    sweep state at all.  Without this check it would price against whatever it happened to
    hold and return a reduced cost that ``master.upper_bound`` would treat as a valid
    bound: a wrong number, with no exception and no log line.  The OOM killer makes that
    reachable rather than hypothetical (see the memory note in the module docstring).
    """


def _worker_assignment(flight_ids, n_workers: int) -> dict[int, int]:
    """Fix each flight to one worker for the whole solve: ``{flight_id -> worker}``.

    Keyed on the FLIGHT, not on its position in ``pricing_order`` -- that order is re-sorted
    every sweep (``solver`` ranks by the heuristic's delay), so a positional rule would send
    a flight to a different worker each iteration and the retained packing would never be
    hit.  That is not a small effect: on ``mp.Pool``'s single shared queue a flight visits
    ``W(1 - (1 - 1/W)^I)`` distinct workers, which is 5.14 of 6 at 16 workers and 6
    iterations, so a solve-scoped pool WITHOUT this recovers about 14% of the rebuild.

    Round-robin over SORTED ids rather than ``flight_id % n_workers`` so sparse or
    non-contiguous ids still split evenly; the modulus balances only when ids happen to be
    dense.  Cost is not modelled -- see ``PricingPool`` for why that is deliberate, and for
    the counter that will say whether it needs to be.
    """

    return {flight_id: i % n_workers for i, flight_id in enumerate(sorted(flight_ids))}


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
    #: Seconds THIS sweep spent starting worker processes.  Nonzero on the sweep that
    #: brought the pool up and 0.0 on every later one, which is the whole point of a
    #: solve-scoped pool: worker launch is parent-SERIAL (`Pool._repopulate_pool_static`
    #: starts them in a plain loop and `spawn` re-pickles the initargs per worker), so it
    #: cost ~4.2-4.8 s per worker per sweep, measured. Kept as its own field rather than
    #: folded into `wall_s` so a run can still SEE that cost after it stops recurring.
    pool_setup_s: float = 0.0

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

# Populated once per worker per SOLVE by `_init_worker`, and once per sweep by
# `_load_sweep_state`. A plain module global because that is the only state a spawned worker
# can carry between tasks without re-pickling it.
#
# The split is the point.  Everything `_init_worker` stores is fixed for the solve, so a
# worker that outlives the sweep keeps it -- including, transitively, each graph's
# `_search_cache.prepared`, the compiled packing that used to be rebuilt from scratch every
# iteration at ~184 ms a flight.  Everything that moves per iteration lives under "sweep",
# as ONE tuple, so there is no reachable state where new duals sit beside old flight duals.
_WORKER: dict[str, Any] = {}


def _init_worker(
    worker: int,
    worker_requests: list[FlightRequest],
    cfg: SimConfig,
    params: ColGenParams,
    catalog: StaticTerminalCatalog,
) -> None:
    """Build this worker's view of ITS OWN flights. Runs once per worker per solve.

    Takes only the worker's requests, not the batch's: under a fixed worker assignment a worker
    can never be asked for a flight outside its slice, so building the rest was ~n/W useful
    work and the remainder waste -- at 1000 flights and 16 workers, 16x more graphs than any
    worker could use, in every worker.

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
    from the first task instead, which reaches the caller as an ordinary traceback.  That
    contract matters MORE under a solve-scoped pool, not less: the respawn loop it prevents
    would now last the whole solve rather than one sweep.
    """

    _WORKER.clear()
    _WORKER["sweep"] = None
    try:
        _WORKER["worker_index"] = int(worker)
        _WORKER["graphs"] = {
            request.flight_id: build_flight_graph(request, cfg, catalog, params)
            for request in worker_requests
        }
        _WORKER["worker_flights"] = frozenset(_WORKER["graphs"])
        _WORKER["cfg"] = cfg
        _WORKER["params"] = params
    except Exception:
        # The traceback as TEXT, because the exception object itself may not survive the
        # trip back: it is re-raised below as a `RuntimeError`, which always pickles.
        _WORKER["init_error"] = traceback.format_exc()


def _load_sweep_state(
    epoch: tuple,
    duals_blob: bytes,
    flight_duals: dict[int, float],
    known_columns: dict[int, Column],
    deadline: float | None,
) -> None:
    """Install one sweep's duals in this worker. Runs once per worker per sweep.

    ``duals_blob`` is BYTES, already pickled: the mapping is global to the iteration and
    cannot be sliced per worker (``DualView._max_negative_credit`` is an ``fsum`` over EVERY
    row and is consumed as a pricing bound, so a restricted view would move answers), but
    the parent can serialize it ONCE and hand every worker the same buffer.  The outer pickle
    is then a memcpy instead of W traversals of a mapping whose keys are ``RowKey`` -- a
    tuple subclass with a validating ``__new__``, so each key costs its own function call in
    each direction.  ``parallel.py`` ships its committed reservations the same way.

    ``flight_duals`` and ``known_columns`` ARE sliced to the worker; the latter is ~13.7 KB a
    flight, so broadcasting all of them cost ~13.7 MB per worker at 1000 flights for the
    ~1/W of it each could use.

    ``_WORKER["sweep"] = None`` happens BEFORE the try, and that ordering is the whole
    safety argument: if building the ``DualView`` raises -- ``MemoryError`` is the realistic
    one -- the worker must be left holding NOTHING, so the next task raises. Leaving the
    previous sweep's tuple in place would let it price iteration k against iteration k-1's
    duals and return a plausible wrong number.
    """

    if _WORKER.get("init_error") is not None:
        # `_price_one` re-raises the initializer's traceback, which is the more useful
        # error; overwriting it with a failure caused BY it would bury the cause.
        return
    _WORKER["sweep"] = None
    _WORKER.pop("sweep_error", None)
    try:
        # Not untrusted input: this buffer was produced by `PricingPool.run_sweep` in the
        # parent of this very process and handed over `mp.Pool`'s own task queue, which
        # pickles every argument anyway. Pre-pickling changes WHEN the mapping is
        # serialized (once, not once per worker), not whether.
        duals = pickle.loads(duals_blob)
        # `DualView` is rebuilt here rather than pickled so the worker owns its own caches.
        # `time.monotonic` is a system-wide clock on both Linux and macOS, so a deadline
        # taken in the parent is directly comparable here. It would NOT be across hosts.
        _WORKER["sweep"] = (
            epoch, DualView(duals, _WORKER["cfg"]), flight_duals, known_columns, deadline,
        )
    except Exception:
        _WORKER["sweep_error"] = traceback.format_exc()


def _worker_identity(_ignored=None) -> tuple:
    """``(worker, pid)`` -- the only way a caller can prove a worker was REUSED.

    Re-raises an initializer failure for the same reason :func:`_price_one` does: this runs
    as :meth:`PricingPool.start`'s readiness probe, and a probe that ignored `init_error`
    would block forever against a worker whose replacement dies the same way.
    """

    init_error = _WORKER.get("init_error")
    if init_error is not None:
        raise RuntimeError(f"pricing worker failed to initialise:\n{init_error}")
    return (_WORKER.get("worker_index"), os.getpid())


def _price_one(epoch: tuple, flight_id: int):
    """Price one flight in a worker.

    Returns ``(flight_id, priced, rc, column, task_s, counter_deltas, search_record)`` --
    the shape :func:`_accepted_prefix` reduces, and the reason that function takes a
    sequence of these rather than a pool.

    ``epoch`` names the sweep whose duals the caller believes this worker holds, and is
    checked rather than trusted; see :class:`StalePricingWorker` for the failure it is
    there to convert from a wrong number into an exception.

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
    sweep_error = _WORKER.get("sweep_error")
    if sweep_error is not None:
        raise RuntimeError(f"pricing worker failed to load sweep state:\n{sweep_error}")
    state = _WORKER.get("sweep")
    if state is None or state[0] != epoch:
        held = None if state is None else state[0]
        raise StalePricingWorker(
            f"worker {_WORKER.get('worker_index')!r} (pid {os.getpid()}) holds sweep state {held!r} "
            f"but was asked to price flight {flight_id} for sweep {epoch!r}; it was almost "
            f"certainly respawned after dying -- see mp.Pool._repopulate_pool_static"
        )
    if flight_id not in _WORKER["worker_flights"]:
        # A named error rather than a bare KeyError three frames down, and a real
        # possibility: the worker split and the result merge are two expressions of one
        # assignment, and this is the cheap place to catch them disagreeing.
        raise StalePricingWorker(
            f"flight {flight_id} is not in worker {_WORKER.get('worker_index')!r}"
        )
    _, dual_view, flight_duals, known_columns, deadline = state
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
            dual_view,
            flight_duals[flight_id],
            _WORKER["cfg"],
            _WORKER["params"],
            known_column=known_columns.get(flight_id),
            deadline=deadline,
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
    pool: "PricingPool | None" = None,
) -> SweepResult:
    """Price every flight in ``pricing_order``, sequentially or across processes.

    Width comes from ``params.n_pricing_workers``; 0 is the sequential loop and is
    byte-identical to no pool at all.  ``graphs`` is used only by the sequential path;
    workers build their own from ``requests`` (cheaper than pickling, see the module
    docstring).

    ``pool`` is a caller-owned :class:`PricingPool` that outlives the sweep. Passing one is
    what makes each flight's compiled packing survive to the next iteration; omitting it
    keeps the old shape -- a pool built and torn down here -- so a caller that prices a
    single sweep needs to know nothing about lifetimes.
    """

    if params.n_pricing_workers == 0:
        return _sweep_sequential(
            pricing_order, graphs, cfg, params, dual_view, flight_duals,
            known_columns, deadline,
        )
    return _sweep_parallel(
        pricing_order, requests, cfg, params, catalog, duals, flight_duals,
        known_columns, deadline, pool,
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

    # `worker`/`pid` describe the SEQUENTIAL arm as written -- one worker, this process. The
    # pool overwrites both in `PricingPool._annotate`, where the true values are known.
    # Setting them here rather than only there is what keeps the two arms' rows the same
    # SHAPE, which is the property `test_both_arms_report_the_same_per_flight_rows` pins.
    record = {
        "flight_id": int(flight_id), "task_s": float(task_s), "priced": bool(priced),
        "worker": 0, "pid": os.getpid(),
    }
    record.update(last_search_record() if search is None else search)
    # After the update, so a stale `flight_id` in the search record can never rename the
    # flight this row is about.
    record["flight_id"] = int(flight_id)
    return record


class PricingPool:
    """Worker processes that outlive the sweep, one worker each.

    **Why W single-worker pools and not one W-worker pool.** ``mp.Pool`` has a single shared
    task queue, so it can express neither "keep this worker" nor "send this flight to THAT
    worker" -- and the second is what makes the first worth having (see
    :func:`_worker_assignment`). One single-worker pool each gives both for free, and keeps everything
    ``mp.Pool`` was chosen for: ``imap(..., 1)`` still returns the ``IMapIterator`` whose
    ``.next(timeout)`` the deadline guard needs, :func:`_init_worker`'s record-don't-raise
    contract is untouched, and the feeder thread means the parent never blocks pushing a
    multi-megabyte dual payload into a worker that is still returning results -- the
    deadlock class ``parallel.py`` had to invent an idle-only outbox flush to avoid.

    **What this actually buys.** Two things, and the second was the larger one when measured:

    1. Each flight's compiled packing (``dp_prepare.prepared_for``, ~184 ms) is built once
       per SOLVE instead of once per sweep. It lives on the graph's ``_search_cache``, which
       a spawned worker starts cold, so a per-sweep pool threw away every one of them --
       989 rebuilds a sweep at 1000 flights, ~180 s of worker CPU.
    2. Worker launch is parent-SERIAL: ``_repopulate_pool_static`` starts workers in a plain
       loop and ``spawn`` re-pickles the initargs for each one. Measured at ~4.2-4.8 s per
       worker, three ways, so at 16 workers the last one started ~60 s into a ~63 s sweep.
       That is why idle worker-seconds scaled as ``W(W-1)/2`` rather than with W, and why 8
       workers ran at 79% efficiency where 16 ran at 49%. A solve-scoped pool pays it once.

    **Memory.** Retained packings do NOT raise the peak: a worker already held its whole
    sweep's worth at end-of-sweep, so pinning converts "the same peak, discarded and rebuilt"
    into "the same peak, kept". It is *dynamic* dispatch on a persistent pool that would
    regress -- every worker would converge toward holding all n flights' packings and drained
    hop tables -- which is a second reason the worker split is not optional.
    """

    def __init__(self, requests, cfg: SimConfig, params: ColGenParams, catalog) -> None:
        self._requests = list(requests)
        self._cfg = cfg
        self._params = params
        self._catalog = catalog
        self._n_workers = int(params.n_pricing_workers)
        if self._n_workers <= 0:
            raise ValueError("PricingPool needs at least one worker")
        self._worker_of = _worker_assignment(
            [request.flight_id for request in self._requests], self._n_workers
        )
        self._worker_requests: list[list] = [[] for _ in range(self._n_workers)]
        for request in self._requests:
            self._worker_requests[self._worker_of[request.flight_id]].append(request)
        self._pools: list | None = None
        self._worker_pid: dict[int, int] = {}
        # A pool instance's own id, so an epoch from a DIFFERENT pool can never compare
        # equal to one of ours -- belt and braces against a future caller that builds two.
        self._uid = uuid.uuid4().hex
        self._counter = itertools.count(1)
        self._poisoned: str | None = None

    @property
    def worker_of(self) -> dict[int, int]:
        return dict(self._worker_of)

    def start(self) -> float:
        """Bring the workers up. Idempotent; returns seconds spent, 0.0 if already running."""

        if self._pools is not None:
            return 0.0
        started = time.perf_counter()
        # `spawn` explicitly rather than by platform default: `fork` would inherit the
        # parent's numba runtime and thread state, which is exactly the combination that is
        # unsafe.
        ctx = mp.get_context("spawn")
        # Appended one at a time, and `self._pools` bound BEFORE the loop, so a failure
        # partway through leaves the pools that were created reachable by `close()` rather
        # than orphaned with live worker processes.  `MemoryError` here is the realistic
        # case and is exactly when leaking 15 of 16 workers would hurt most.
        self._pools = []
        try:
            for worker in range(self._n_workers):
                self._pools.append(ctx.Pool(
                    processes=1,
                    initializer=_init_worker,
                    initargs=(
                        worker, self._worker_requests[worker], self._cfg, self._params,
                        self._catalog,
                    ),
                ))
            # Block until every worker has finished its initializer. Two reasons:
            # `pool_setup_s` is then honest rather than half-charged to the first sweep's
            # tasks, and an initializer `MemoryError` surfaces HERE, named, instead of on
            # some later flight.
            for worker, pool in enumerate(self._pools):
                _, pid = pool.apply(_worker_identity, (None,))
                self._worker_pid[worker] = pid
        except BaseException:
            self.close()
            raise
        return time.perf_counter() - started

    def run_sweep(
        self, pricing_order, duals, flight_duals, known_columns, deadline
    ) -> SweepResult:
        """Price one sweep across the workers, in ``pricing_order`` order."""

        if self._poisoned is not None:
            raise RuntimeError(f"pricing pool is no longer usable: {self._poisoned}")
        setup_s = self.start()
        epoch = (self._uid, next(self._counter))
        # Pickled ONCE for all workers -- see `_load_sweep_state` for why the mapping cannot
        # be sliced and why one buffer beats W traversals.
        duals_blob = pickle.dumps(duals, protocol=pickle.HIGHEST_PROTOCOL)

        worker_order: list[list[int]] = [[] for _ in range(self._n_workers)]
        for flight_id in pricing_order:
            worker_order[self._worker_of[flight_id]].append(flight_id)

        # Started before any work is dispatched, and after `start()`, so `wall_s` measures
        # the sweep and `pool_setup_s` measures the launch. The predecessor folded the two
        # together because it had no way to separate them.
        sweep_started = time.perf_counter()
        worker_iters = []
        for worker, pool in enumerate(self._pools):
            own = set(worker_order[worker])
            # FIFO on this worker's queue, so it is guaranteed to run before the tasks below
            # it -- which is the entire delivery mechanism for per-sweep duals, and works
            # only because the worker has exactly one worker.
            pool.apply_async(
                _load_sweep_state,
                (
                    epoch,
                    duals_blob,
                    {f: flight_duals[f] for f in own if f in flight_duals},
                    {f: c for f, c in known_columns.items() if f in own},
                    deadline,
                ),
            )
            # Chunked HERE rather than by `imap`, and that is a correctness requirement.
            # `Pool.imap` only returns an `IMapIterator` when `chunksize == 1`; above 1 it
            # returns `(item for chunk in result for item in chunk)` -- a plain generator,
            # which has no `.next(timeout)`, so the deadline guard would raise
            # `AttributeError` before yielding a single result.
            chunks = [
                (epoch, worker_order[worker][i:i + self._params.pricing_chunksize])
                for i in range(0, len(worker_order[worker]), self._params.pricing_chunksize)
            ]
            worker_iters.append(pool.imap(_price_batch, chunks, 1))

        try:
            accepted = _accepted_prefix(
                _sweep_results(worker_iters, pricing_order, self._worker_of, deadline)
            )
        except BaseException as exc:
            # Same reasoning as the incomplete-sweep case below: the per-worker iterators are
            # half consumed and their tasks are still running, so this pool cannot be
            # trusted for another sweep.  Reached when the merge catches the worker split and
            # itself disagreeing, and when a task raises `StalePricingWorker` outside the
            # merge's own handler.
            self._poisoned = f"sweep raised {type(exc).__name__}: {exc}"
            raise
        accepted = replace(
            accepted,
            wall_s=time.perf_counter() - sweep_started,
            pool_setup_s=setup_s,
            flight_records=self._annotate(accepted.flight_records),
        )
        if not accepted.complete:
            # The prefix stopped early, so this worker's `imap` is half-consumed and tasks are
            # still running behind it. Rather than reason about how those interleave with a
            # next sweep, refuse one: an incomplete sweep ALWAYS ends the solve (`solver`
            # sets `pricing_complete = False` and breaks), so there is no reachable caller
            # that wants another. A future refactor that changes that gets this exception
            # instead of a silently interleaved sweep.
            self._poisoned = (
                f"sweep ended early at flight {accepted.timeout_flight_id}; "
                f"outstanding tasks were abandoned"
            )
        return accepted

    def _annotate(self, records):
        """Add ``worker`` and ``pid`` to each diagnostic row.

        Done in the PARENT rather than returned by the worker, which is possible only
        because the assignment is static: the worker follows from the flight id alone, and the
        pid from the worker. That keeps `_price_one`'s result tuple, `_accepted_prefix` and
        `_flight_record` untouched -- the reduction rule this module exists to enforce reads
        as unchanged in the diff, which is worth more than saving a dict copy.

        These two keys are what make worker skew observable in every future run: the risk the
        worker split takes is that `mp.Pool`'s rebalancing is gone, and `max` over `min` of
        per-worker `task_s` is the number that will say whether it needs a cost-aware split.
        """

        annotated = []
        for record in records:
            worker = self._worker_of.get(record["flight_id"])
            annotated.append({
                **record, "worker": worker, "pid": self._worker_pid.get(worker, -1),
            })
        return tuple(annotated)

    def close(self) -> None:
        """Terminate the workers. Idempotent, and safe after any failure."""

        self._poisoned = self._poisoned or "closed"
        for pool in self._pools or ():
            try:
                pool.terminate()
            except Exception:
                pass
            try:
                pool.join()
            except Exception:
                pass
        self._pools = None

    def __enter__(self) -> "PricingPool":
        return self

    def __exit__(self, *_exc) -> bool:
        self.close()
        return False


def _sweep_parallel(
    pricing_order, requests, cfg, params, catalog, duals, flight_duals,
    known_columns, deadline, pool: "PricingPool | None" = None,
) -> SweepResult:
    """Fan the sweep across the workers, on a caller-owned pool when there is one.

    With no pool the sweep builds one for itself and tears it down, which is what
    `price_sweep` does when called outside a solve. That path is now a special case of the
    solve-scoped one rather than a second implementation, so there is exactly one route
    through the workers and the tests that drive `price_sweep` directly still exercise it.
    """

    if pool is not None:
        return pool.run_sweep(pricing_order, duals, flight_duals, known_columns, deadline)
    with PricingPool(requests, cfg, params, catalog) as own:
        return own.run_sweep(pricing_order, duals, flight_duals, known_columns, deadline)


def _price_batch(task):
    """Price one chunk of flights in a worker, in order.

    Module level because `spawn` pickles by qualified name. Exists so the sweep can hand
    `imap` a chunksize of one whatever `pricing_chunksize` is -- see the note in
    :meth:`PricingPool.run_sweep`.

    Takes ONE argument, still, so `imap`'s return type stays `IMapIterator`; the epoch rides
    inside it rather than as a second parameter.
    """

    epoch, flight_ids = task
    return [_price_one(epoch, flight_id) for flight_id in flight_ids]


def _sweep_results(worker_iters, pricing_order, worker_of, deadline):
    """Merge the per-worker result streams into ``pricing_order`` order, bounded by ``deadline``.

    Without the deadline the sweep can wait forever, and the solver's own time limit never
    gets a chance to fire. ``mp.Pool`` does not fail a task whose worker DIED: the sentinel
    thread starts a replacement and the lost task's result is simply never produced, so the
    ordered iterator blocks on a value nobody will send. The realistic trigger is the OOM
    killer, which is reachable rather than hypothetical here -- memory is linear in workers
    (22.7 GB at 8 on density x100) and that is what the ``n_workers`` ceiling exists for.

    The bound is the caller's OWN absolute deadline, not a grace period invented here.
    That distinction is the reason this is safe: past ``deadline`` the sweep was going to
    stop regardless, so nothing is abandoned that would otherwise have been kept.

    A give-up is reported as **that flight timing out**, which is deliberate: it re-enters
    the discard rule the sequential loop already defines, so the caller gets a short prefix
    with ``timeout_flight_id`` set and ``complete`` False. Returning a truncated prefix that
    merely *looked* complete would be far worse than the hang -- `master.upper_bound` would
    take a bound over flights that were never priced.

    **Why there is no reorder buffer.** The predecessor of this function leaned on one
    global FIFO -- "the k-th chunk is ``chunks[k]``" -- and that is gone, because each worker
    is now its own queue. It is not needed: a worker's results are FIFO in the order that
    worker's flights appear in ``pricing_order``, so the j-th result out of worker *w* IS the
    j-th ``pricing_order`` entry assigned to *w*. Walking ``pricing_order`` and blocking on
    the one worker that owns the next index therefore yields exactly index order, with a
    deque per worker holding whatever arrived early.

    Blocking on that single worker costs nothing, and this is the subtle part: index order
    means the next result is REQUIRED before any other can be emitted, so time spent
    waiting for it is not time another worker's result could have used. That is what keeps
    ``master.upper_bound``'s non-associative ``sum`` fed in a fixed order -- rule 2.
    """

    pending = [collections.deque() for _ in worker_iters]
    for flight_id in pricing_order:
        worker = worker_of[flight_id]
        while not pending[worker]:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                pending[worker].extend(worker_iters[worker].next(remaining))
            except StopIteration:
                raise RuntimeError(
                    f"pricing worker {worker} is exhausted but pricing_order still expects "
                    f"flight {flight_id}: the worker split and this merge disagree"
                ) from None
            except mp.TimeoutError:
                # Exactly the shape `_price_one` returns for a `PricingTimeout`, so the
                # reducer needs no special case: no seconds, no counters, `priced` False --
                # and an EMPTY search record, because the worker that would have filled it
                # is the one that never reported back.  Names the flight the parent is
                # actually blocked on, which the chunk-indexed predecessor could only
                # approximate with its chunk's first flight.
                yield flight_id, False, 0.0, None, 0.0, {}, {}
                return
            except StalePricingWorker as exc:
                # A worker died and `mp.Pool` quietly replaced it with one holding no sweep
                # state.  End the sweep the way a lost result does -- the prefix so far is
                # still valid -- but count it, so `solve` can report why rather than
                # blaming the clock.
                warnings.warn(
                    f"pricing worker {worker} lost its worker: {exc}", RuntimeWarning,
                    stacklevel=2,
                )
                yield flight_id, False, 0.0, None, 0.0, {"pool_worker_lost": 1}, {}
                return
        result = pending[worker].popleft()
        if result[0] != flight_id:
            # One comparison per flight, and it turns a merge bug from a WRONG NUMBER --
            # reduced costs silently transposed between two flights -- into a crash.
            raise RuntimeError(
                f"pricing worker {worker} returned flight {result[0]} where pricing_order "
                f"expects {flight_id}"
            )
        yield result


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
