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
:class:`PricingPool` holds W long-lived processes, one duplex ``Pipe`` each, and pins every
flight to one of them for the whole solve (:func:`_worker_assignment`) -- so a worker keeps
the state it derived for its own flights instead of deriving it again every iteration.
Per-sweep duals arrive as a message rather than through an initializer, stamped with an
epoch :func:`_price_one` refuses to price against if it does not match.

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

**Raw processes and pipes, not ``mp.Pool`` and not ``ProcessPoolExecutor``.** The executor
deadlocks on CPython 3.14.2 when ``max_tasks_per_child`` fires. ``mp.Pool`` was used here
first and had a worse problem for a SOLVE-scoped pool: its handler threads sit between the
parent and the worker, and the task handler blocks forever writing to the input queue of a
worker that was SIGKILLed while holding that queue's lock. ``terminate()`` takes no timeout,
so teardown could only be bounded from outside -- leaving the blocked thread behind. A pipe
has no intermediary, so ``close`` is bounded by construction. ``freespace_sim.parallel``
runs the same shape for the A* speculative runner.

**macOS spawn hazard, deliberately accepted.** Under the ``spawn`` start method a caller
script that does work at module level rather than under ``if __name__ == "__main__":`` dies
in ``_check_not_importing_main``. Every shipped entry point already guards its main.

**A worker KILLED mid-sweep is an event, not a timeout.** The parent waits on every
worker's pipe AND its process sentinel, so an OOM kill -- reachable rather than
hypothetical, since memory is linear in workers -- wakes the wait immediately and is
reported as ``pool_worker_lost``. Under ``mp.Pool`` the same failure produced a task whose
result was simply never sent, so it could only be inferred after the entire pricing
deadline had elapsed, and even then only by comparing pids against a silently spawned
replacement.

:func:`_sweep_results` still bounds its wait by the caller's own ``pricing_deadline``, so a
worker that is merely SLOW degrades to a short prefix rather than blocking, and the two
outcomes are reported differently: a death says use fewer workers, an expiry says raise the
budget. It also merges the per-worker result streams back into ``pricing_order`` order,
which rule 2 requires and which a queue per worker does not give for free.

A worker that fails in its *initializer* is a different failure with the same symptom:
:func:`_init_worker` records the traceback rather than raising, :func:`_worker_main` sends
it with the readiness message, and :meth:`PricingPool.start` fails the whole pool with the
cause attached.
"""
from __future__ import annotations

import collections
import itertools
import logging
import multiprocessing as mp
import multiprocessing.connection as mp_connection
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
from ...progress import RollingRate
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

log = logging.getLogger(__name__)


__all__ = [
    "PricingPool",
    "StalePricingWorker",
    "SweepResult",
    "price_sweep",
]


# How long `PricingPool.close` waits on a worker it has already SIGKILLed. Short on
# purpose: the process has been killed, so this is reaping a zombie rather than waiting for
# work, and the only way it expires is a process stuck in uninterruptible I/O -- which
# waiting longer would not fix either.
_CLOSE_JOIN_TIMEOUT_S = 5.0

# How long `close` waits for workers to stop POLITELY before killing them. An idle
# worker exits in milliseconds; anything slower is one that cannot see the message at
# all, so waiting longer only delays a teardown that is already going to be a kill.
_GRACEFUL_STOP_S = 0.5


# How long the result merge blocks before re-checking whether a worker is still alive.
# Not a deadline of its own: the sweep still stops at the caller's, this only decides
# how promptly a death inside it is NOTICED. Short enough that a worker lost early does
# not burn a long budget, long enough that the check is free against real task times
# (a density flight prices in seconds).
_WORKER_POLL_S = 2.0


class _SweepProgress:
    """Log a pricing sweep's advance while it is still running.

    A sweep is the longest single block in a colgen solve -- 205 s for one iteration of the
    full 4,636-flight density scenario -- and until now it printed nothing between "started"
    and "finished". A run that is merely slow was indistinguishable from one that had
    wedged, which matters most exactly when it is worst.

    **Two tracks, because one does not cover both ends of the scale**, the same shape as
    ``sim._MilestoneLog``: every ``every_n`` flights, which stays informative on a long
    sweep, and every 10%, which still says something when the whole batch is smaller than
    ``every_n``. The percentage track is suppressed below ``_MIN_FOR_PERCENT`` flights,
    where ten lines to describe fifty flights is noise rather than progress.

    The ETA rides the ROLLING rate rather than the cumulative one, so it re-forecasts
    through a slowdown instead of trusting an average that lags it -- and the two are
    printed together, because them diverging is the actual signal that the sweep is
    degrading (a straggler, or a worker that has stopped returning).
    """

    _MIN_FOR_PERCENT = 100

    def __init__(self, total: int, n_workers: int, every_n: int = 1000, window: int = 100):
        self.total = int(total)
        self.n_workers = int(n_workers)
        self.every_n = max(1, int(every_n))
        self.rate = RollingRate(min(window, max(1, self.total // 4)))
        self.started = time.monotonic()
        self._prev = self.started
        self._marks = (
            [(pct, max(1, self.total * pct // 100)) for pct in range(10, 100, 10)]
            if self.total >= self._MIN_FOR_PERCENT else []
        )
        self._next_mark = 0

    def begin(self) -> None:
        log.info(
            "pricing sweep: %d flights across %s",
            self.total,
            f"{self.n_workers} workers" if self.n_workers else "1 process (sequential)",
        )

    def advance(self, done: int) -> None:
        now = time.monotonic()
        self.rate.add(now - self._prev)
        self._prev = now
        due = done % self.every_n == 0
        while self._next_mark < len(self._marks) and done >= self._marks[self._next_mark][1]:
            due = True
            self._next_mark += 1
        if due and done < self.total:
            log.info("  priced %s", self._line(done, now))

    def finish(self, done: int, complete: bool) -> None:
        log.info(
            "pricing sweep %s: %s",
            "done" if complete else "STOPPED EARLY",
            self._line(done, time.monotonic()),
        )

    def _line(self, done: int, now: float) -> str:
        elapsed = now - self.started
        pct = 100.0 * done / max(self.total, 1)
        rolling, eta = self.rate.roll_ms(), self.rate.eta_s(done, self.total)
        return (
            f"{done}/{self.total} ({pct:.0f}%)  elapsed={elapsed:.0f}s  "
            f"wall/flight avg={self.rate.avg_ms(done):.0f}ms "
            f"roll[{self.rate.window}]={'n/a' if rolling is None else f'{rolling:.0f}ms'}  "
            f"ETA {'n/a' if eta is None else f'{eta:.0f}s'}"
        )


class _WorkerStartTimeout(RuntimeError):
    """Startup did not finish inside the caller's deadline.

    Internal: :meth:`PricingPool.run_sweep` turns it into an ordinary timed-out sweep,
    because "the clock ran out before any flight could be priced" is what it means and
    the caller already has a shape for that -- an empty accepted prefix. Raising out of
    `run_sweep` instead would make an expired deadline an ERROR on the parallel path and
    a short prefix on the sequential one, for the same input.
    """


class _WorkerChannel:
    """One worker process and the pipe to it."""

    __slots__ = ("index", "proc", "conn", "pid", "dead")

    def __init__(self, index: int, proc, conn):
        self.index = index
        self.proc = proc
        self.conn = conn
        self.pid: int | None = None
        self.dead = False

    def send(self, message) -> None:
        self.conn.send(message)

    def close(self) -> None:
        for shut in (self.conn.close,):
            try:
                shut()
            except Exception:
                pass


class _PipeCollector:
    """Drain whatever the workers have produced, and notice the ones that have died.

    Waits on every live worker's pipe AND its process sentinel in one
    ``multiprocessing.connection.wait``. The sentinel is what makes death immediate: it
    becomes readable the moment the process exits, so an OOM kill is observed as an event
    rather than inferred from a result that never comes.

    Drains EVERY ready pipe, not only the one the merge is currently blocked on. That is
    required, not tidy: a column is ~13.7 KB against a ~64 KB pipe buffer, so a worker whose
    results nobody reads blocks after a handful of them and stops making progress on flights
    the merge will need later.
    """

    def __init__(self, channels):
        self._channels = list(channels)

    def __call__(self, timeout: float | None):
        live = [ch for ch in self._channels if not ch.dead]
        if not live:
            return {}, set()
        waitables: dict[Any, tuple[_WorkerChannel, bool]] = {}
        for ch in live:
            waitables[ch.conn] = (ch, True)
            waitables[ch.proc.sentinel] = (ch, False)
        ready = mp_connection.wait(list(waitables), timeout)

        collected: dict[int, list] = {}
        died: set[int] = set()
        for handle in ready:
            channel, is_pipe = waitables[handle]
            if not is_pipe:
                # The sentinel fired. Its pipe may still hold results the worker sent
                # before exiting, and those are perfectly good -- drain them below rather
                # than discarding work that was already done.
                channel.dead = True
                died.add(channel.index)
                continue
            while True:
                try:
                    if not channel.conn.poll():
                        break
                    message = channel.conn.recv()
                except (EOFError, OSError):
                    channel.dead = True
                    died.add(channel.index)
                    break
                if message[0] == "results":
                    collected.setdefault(channel.index, []).extend(message[1])
                elif message[0] == "error":
                    raise RuntimeError(
                        f"pricing worker {channel.index} raised:\n{message[1]}"
                    )
        # A worker that exited cleanly at the end of its assignment is not a failure, but
        # the merge cannot tell the difference and does not have to: it only consults
        # `died` when it is still WAITING on that worker for a flight.
        return collected, died


class StalePricingWorker(RuntimeError):
    """A worker was handed a task from a sweep whose state it does not hold.

    Carries a single string so it survives the trip back down the pipe intact.

    Kept as defence in depth rather than as the load-bearing guard it once was. Under
    ``mp.Pool`` a dead worker was silently REPLACED, re-running the original initargs --
    which carry solve constants only, since the duals arrive per sweep -- so the
    replacement would price against no duals at all and return a reduced cost
    ``master.upper_bound`` accepts as a valid bound: a wrong number, no exception, no log
    line. Raw processes are not replaced, so that path is gone; what remains is a parent
    bug dispatching against state a worker does not hold, which this still turns into an
    exception rather than a number.
    """


def _worker_assignment(flight_ids, n_workers: int) -> dict[int, int]:
    """Fix each flight to one worker for the whole solve: ``{flight_id -> worker}``.

    Keyed on the FLIGHT, not on its position in ``pricing_order`` -- that order is re-sorted
    every sweep (``solver`` ranks by the heuristic's delay), so a positional rule would send
    a flight to a different worker each iteration and the retained packing would never be
    hit.  That is not a small effect: on a single shared queue a flight visits
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

    A failure here is RECORDED, never raised, and :func:`_worker_main` ships it with the
    readiness message so the parent fails the pool with the cause attached. Rebuilding
    every graph is the largest allocation a worker makes, so ``MemoryError`` -- the failure
    the ``n_workers`` ceiling is about -- lands exactly here, and a worker that merely
    exited would otherwise be indistinguishable from one that was killed.
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
        # parent of this very process and handed down a pipe that pickles every message
        # anyway. Pre-pickling changes WHEN the mapping is serialized (once, not once per
        # worker), not whether.
        duals = pickle.loads(duals_blob)
        # `DualView` is rebuilt here rather than pickled so the worker owns its own caches.
        # `time.monotonic` is a system-wide clock on both Linux and macOS, so a deadline
        # taken in the parent is directly comparable here. It would NOT be across hosts.
        _WORKER["sweep"] = (
            epoch, DualView(duals, _WORKER["cfg"]), flight_duals, known_columns, deadline,
        )
    except Exception:
        _WORKER["sweep_error"] = traceback.format_exc()


def _worker_main(conn, worker_index, requests, cfg, params, catalog) -> None:
    """One pricing worker: build this worker's flights, then serve the parent's pipe.

    Replaces ``mp.Pool``'s task queue, and the reason is teardown. A pool's handler THREADS
    sit between the parent and the worker, and the task handler can block forever writing to
    the input queue of a worker that was SIGKILLed while holding that queue's lock -- so
    ``terminate()`` had no bound, and bounding it from outside could only ever leave the
    blocked thread behind. A pipe has no such intermediary: the parent holds the only other
    end, the process either exits or is killed, and both are observable and bounded.

    Two more things fall out, neither of which ``mp.Pool`` could give:

    * **Death is a sentinel, not a timeout.** The parent waits on this pipe AND on the
      process sentinel, so a worker that dies is known immediately instead of after the
      pricing deadline expires -- which is how a worker lost in the first second of a
      1,200 s budget used to consume all 1,200.
    * **No silent respawn.** ``_repopulate_pool_static`` restarted a dead worker with the
      ORIGINAL initargs, which carry no duals, so the replacement would have priced against
      nothing. Here a dead worker stays dead and is reported.

    Messages, FIFO per pipe (the ordering `_load_sweep_state` relies on -- a sweep's duals
    are always drained before the tasks that read them):

      ("sweep", epoch, duals_blob, flight_duals, known_columns, deadline)  -> install duals
      ("price", epoch, flight_ids, chunksize)  -> stream ("results", [...]) per chunk
      ("stop",)                                -> exit

    Results are streamed per CHUNK rather than per message-batch so the parent can drain
    them as they are produced. That is flow control, not cosmetics: a column is ~13.7 KB and
    a pipe buffer is ~64 KB, so a worker that produced everything before sending would block
    on a full pipe while the parent waited for a different one.
    """

    try:
        _init_worker(worker_index, requests, cfg, params, catalog)
        # The initializer RECORDS rather than raises (see `_init_worker`), so the error --
        # if any -- rides the readiness message. The parent then fails the whole pool with
        # the cause, instead of discovering it on some later flight.
        conn.send(("ready", os.getpid(), _WORKER.get("init_error")))
    except BaseException:  # noqa: BLE001 - the parent must learn about ANY startup failure
        try:
            conn.send(("ready", os.getpid(), traceback.format_exc()))
        except Exception:
            pass
        return

    while True:
        try:
            message = conn.recv()
        except (EOFError, KeyboardInterrupt):
            return
        tag = message[0]
        if tag == "stop":
            return
        if tag == "sweep":
            _, epoch, duals_blob, flight_duals, known_columns, deadline = message
            _load_sweep_state(epoch, duals_blob, flight_duals, known_columns, deadline)
            continue
        if tag == "price":
            _, epoch, flight_ids, chunksize = message
            try:
                for start in range(0, len(flight_ids), chunksize):
                    chunk = flight_ids[start:start + chunksize]
                    conn.send(("results", _price_batch((epoch, chunk))))
                    # NOT polled for `stop` between chunks, which was tried and does not
                    # work: when the parent abandons a sweep it stops draining, the pipe
                    # fills, and the worker blocks inside `send` -- so it never reaches a
                    # poll no matter where one is put. Teardown handles it by killing,
                    # which is why `close` keeps its grace window short.
            except BaseException:  # noqa: BLE001 - reported, never allowed to kill the pipe
                try:
                    conn.send(("error", traceback.format_exc()))
                except Exception:
                    pass
            continue


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
    make the parent re-raise it, and a timeout here is an ordinary outcome, not a
    failure.
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
            f"certainly a dispatch against state it was never sent"
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
    # The sequential arm reports too, and has to: it is the arm a long production run uses
    # by default (`n_pricing_workers` ships at 0), so progress that existed only under a
    # pool would be missing exactly where a sweep takes longest.
    progress = _SweepProgress(len(pricing_order), 0)
    progress.begin()
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
            progress.finish(len(flight_ids), complete=False)
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
        progress.advance(len(flight_ids))
    progress.finish(len(flight_ids), complete=True)
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

    **Why a pipe per worker rather than a shared task queue.** A queue can express neither
    "keep this worker" nor "send this flight to THAT worker", and the second is what makes
    the first worth having (see :func:`_worker_assignment`). A pipe per process gives both,
    and gives two things a queue cannot: the parent can wait on a worker's PROCESS SENTINEL
    beside its results, so death is an event rather than a result that never comes; and
    teardown has no handler thread to get stuck behind, so :meth:`close` is bounded by
    construction rather than by a timeout wrapped around something unbounded.

    Sending never blocks on a busy peer, which is the hazard ``parallel.py`` had to invent
    an idle-only outbox flush to avoid. Both parent sends happen at sweep start, when every
    worker is idle in ``recv``; the only large payload flows the other way, and the parent
    drains every ready pipe on each wait rather than only the one it needs.

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
        self._channels: list | None = None
        self._worker_pid: dict[int, int] = {}
        # A pool instance's own id, so an epoch from a DIFFERENT pool can never compare
        # equal to one of ours -- belt and braces against a future caller that builds two.
        self._uid = uuid.uuid4().hex
        self._counter = itertools.count(1)
        self._poisoned: str | None = None

    @property
    def worker_of(self) -> dict[int, int]:
        return dict(self._worker_of)

    def start(self, deadline: float | None = None) -> float:
        """Bring the workers up. Idempotent; returns seconds spent, 0.0 if already running.

        ``deadline`` bounds the readiness wait, and bounding it matters: a worker killed
        during its initializer produces no exception and no result, so an unbounded wait
        would sit outside every clock the solver owns. Bounded by the caller's OWN deadline,
        for the same reason :func:`_sweep_results` is.
        """

        if self._channels is not None:
            return 0.0
        started = time.perf_counter()
        # `spawn` explicitly rather than by platform default: `fork` would inherit the
        # parent's numba runtime and thread state, which is exactly the combination that is
        # unsafe.
        ctx = mp.get_context("spawn")
        # Bound to the list BEFORE the loop, so a failure partway through leaves the
        # processes that were created reachable by `close()` rather than orphaned.
        self._channels = []
        try:
            for worker in range(self._n_workers):
                # CHECKED BEFORE EACH SPAWN, not only before the readiness wait. Starting a
                # process is synchronous and `spawn` re-pickles the arguments each time, so
                # a deadline consulted only afterwards still pays the whole serial ramp --
                # ~4.5 s per worker, about a minute at 16, for a solve already out of time.
                if deadline is not None and time.monotonic() >= deadline:
                    raise _WorkerStartTimeout(
                        f"the pricing deadline passed while starting worker {worker} of "
                        f"{self._n_workers}"
                    )
                parent_conn, child_conn = ctx.Pipe()
                proc = ctx.Process(
                    target=_worker_main,
                    args=(
                        child_conn, worker, self._worker_requests[worker], self._cfg,
                        self._params, self._catalog,
                    ),
                    daemon=True,
                )
                proc.start()
                # The parent's copy of the CHILD end is closed immediately, and that is what
                # makes EOF meaningful: while any copy stays open in this process, a dead
                # worker's pipe never reports end-of-file and death would be invisible.
                child_conn.close()
                self._channels.append(_WorkerChannel(worker, proc, parent_conn))

            # Wait for every worker to report. Blocking here rather than on the first flight
            # keeps `pool_setup_s` honest and surfaces an initializer failure named, at
            # startup, instead of on some later flight.
            for channel in self._channels:
                remaining = (
                    None if deadline is None else max(0.0, deadline - time.monotonic())
                )
                if not channel.conn.poll(remaining):
                    raise _WorkerStartTimeout(
                        f"pricing worker {channel.index} did not report ready before the "
                        f"pricing deadline"
                    )
                try:
                    tag, pid, init_error = channel.conn.recv()
                except EOFError as exc:
                    raise _WorkerStartTimeout(
                        f"pricing worker {channel.index} died before reporting ready"
                    ) from exc
                if tag != "ready":
                    raise RuntimeError(f"unexpected startup message: {tag!r}")
                if init_error is not None:
                    raise RuntimeError(
                        f"pricing worker {channel.index} failed to "
                        f"initialise:\n{init_error}"
                    )
                channel.pid = pid
                self._worker_pid[channel.index] = pid
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
        # Distinct from the `sweep_started` below, which starts after `start()` so that
        # `wall_s` measures the sweep and `pool_setup_s` measures the launch. This one
        # only ever reports the failed-startup path.
        attempt_started = time.perf_counter()
        try:
            setup_s = self.start(deadline)
        except _WorkerStartTimeout as exc:
            # An empty accepted prefix, which is what "the clock ran out before anything
            # could be priced" already means everywhere else in this module -- and what the
            # sequential arm returns for the same input. Raising instead would make an
            # already-expired deadline an ERROR on the parallel path and a short prefix on
            # the sequential one.
            self._poisoned = f"workers never started: {exc}"
            return SweepResult(
                (), (), (), pricing_order[0] if pricing_order else None,
                0.0, time.perf_counter() - attempt_started, Counter(), (), 0.0,
            )

        epoch = (self._uid, next(self._counter))
        # Pickled ONCE for all workers -- see `_load_sweep_state` for why the mapping cannot
        # be sliced and why one buffer beats W traversals.
        duals_blob = pickle.dumps(duals, protocol=pickle.HIGHEST_PROTOCOL)

        worker_order: list[list[int]] = [[] for _ in range(self._n_workers)]
        for flight_id in pricing_order:
            worker_order[self._worker_of[flight_id]].append(flight_id)

        sweep_started = time.perf_counter()
        for channel in self._channels:
            own = set(worker_order[channel.index])
            # FIFO on this worker's pipe, so the duals are installed before the tasks that
            # read them. That ordering is the whole delivery mechanism, and it works because
            # the worker is idle in `recv` at this moment -- so neither send can block on a
            # busy peer.
            channel.send((
                "sweep",
                epoch,
                duals_blob,
                {f: flight_duals[f] for f in own if f in flight_duals},
                {f: c for f, c in known_columns.items() if f in own},
                deadline,
            ))
            # ONE message carrying every flight this worker owns, rather than one per chunk:
            # it is a list of ints, so it cannot fill the pipe, and the worker streams the
            # RESULTS back in `pricing_chunksize` pieces, which is where flow control is
            # actually needed.
            channel.send((
                "price", epoch, worker_order[channel.index],
                max(1, int(self._params.pricing_chunksize)),
            ))

        progress = _SweepProgress(len(pricing_order), self._n_workers)
        try:
            accepted = _accepted_prefix(
                _sweep_results(
                    _PipeCollector(self._channels), self._n_workers, pricing_order,
                    self._worker_of, deadline, progress,
                )
            )
        except BaseException as exc:
            # The workers are mid-assignment and their pipes hold results nobody will read,
            # so this pool cannot be trusted for another sweep.
            self._poisoned = f"sweep raised {type(exc).__name__}: {exc}"
            raise
        accepted = replace(
            accepted,
            wall_s=time.perf_counter() - sweep_started,
            pool_setup_s=setup_s,
            flight_records=self._annotate(accepted.flight_records),
        )
        if not accepted.complete:
            # The prefix stopped early, so the workers are still working and their pipes
            # still hold results. Rather than reason about how those interleave with a next
            # sweep, refuse one: an incomplete sweep ALWAYS ends the solve (`solver` sets
            # `pricing_complete = False` and breaks), so no reachable caller wants another.
            self._poisoned = (
                f"sweep ended early at flight {accepted.timeout_flight_id}; "
                f"outstanding work was abandoned"
            )
        return accepted

    def _annotate(self, records):
        """Add ``worker`` and ``pid`` to each diagnostic row.

        Done in the PARENT rather than returned by the worker, which is possible only
        because the assignment is static: the worker follows from the flight id alone, and
        the pid from the worker. That keeps `_price_one`'s result tuple, `_accepted_prefix`
        and `_flight_record` untouched -- the reduction rule this module exists to enforce
        reads as unchanged in the diff.

        These two keys are what make worker skew observable -- the risk the fixed split
        takes is that dynamic rebalancing is gone, and `max` over `mean` of per-worker
        `task_s` is the number that says whether it needs to be cost-aware.

        Observable is not the same as recorded: these rows reach a caller only through
        `solve`'s per-iteration callback, are never returned in `stats`, and are skipped
        when a sweep ends early. Anything wanting the skew afterwards has to aggregate them
        as they go, which is what `analysis/run_colgen_timed.py:_worker_skew` does.
        """

        annotated = []
        for record in records:
            worker = self._worker_of.get(record["flight_id"])
            annotated.append({
                **record, "worker": worker, "pid": self._worker_pid.get(worker, -1),
            })
        return tuple(annotated)

    def close(self) -> None:
        """Stop the workers. Idempotent, bounded, and it leaks nothing.

        Three escalating steps, none of which can block indefinitely: ask each worker to
        stop, wait a bounded time for it to exit, then SIGKILL whatever is still alive and
        reap it. There are no handler threads between the parent and the worker, which is
        what makes this statement possible at all -- the previous ``mp.Pool`` transport
        could only ever bound ``terminate()`` from outside and leave its blocked thread
        behind, because that thread was waiting on the input-queue lock of a worker that
        had died holding it.
        """

        self._poisoned = self._poisoned or "closed"
        channels = self._channels or ()
        for channel in channels:
            try:
                channel.send(("stop",))
            except (BrokenPipeError, OSError, ValueError):
                pass
        # ONE short grace window shared by all of them, not `_CLOSE_JOIN_TIMEOUT_S` each.
        # An idle worker sees `stop` and exits in milliseconds, so the polite path costs
        # nothing; a worker mid-assignment is blocked writing into a pipe the parent has
        # stopped draining and will NEVER see `stop`, so waiting on it is pure delay. That
        # is not hypothetical -- per-worker joins made teardown after a lost worker take
        # 15 s for three survivors, on a solve that had already failed.
        graceful_until = time.monotonic() + _GRACEFUL_STOP_S
        for channel in channels:
            try:
                channel.proc.join(max(0.0, graceful_until - time.monotonic()))
            except Exception:
                pass
        for channel in channels:
            try:
                if channel.proc.is_alive():
                    channel.proc.kill()
                    # Reaping a SIGKILLed process, which is prompt; the longer bound is
                    # only here so a process wedged in uninterruptible I/O cannot hang us.
                    channel.proc.join(_CLOSE_JOIN_TIMEOUT_S)
            except Exception:
                pass
            channel.close()
        self._channels = None

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

    Module level because `spawn` pickles by qualified name. Kept as its own function, taking
    one packed argument, because the in-process tests drive it directly to exercise a
    worker's pricing path without spawning anything.
    """

    epoch, flight_ids = task
    return [_price_one(epoch, flight_id) for flight_id in flight_ids]


def _sweep_results(collect, n_workers, pricing_order, worker_of, deadline, progress=None):
    """Merge the per-worker result streams into ``pricing_order`` order, bounded by ``deadline``.

    ``collect(timeout) -> (results_by_worker, died)`` is the only thing this needs from the
    transport, which is what lets it be tested without processes at all.

    **Death is an event now, not an inference.** ``collect`` waits on process sentinels
    alongside the pipes, so a worker that dies is reported in ``died`` immediately. The
    predecessor could only notice by waiting out the entire remaining budget and then asking
    whether the process had changed -- so a worker lost in the first second of a 1,200 s
    solve consumed all 1,200 before anyone said so, and even then it was a race.

    The deadline still bounds the wait, and the bound is the caller's OWN absolute deadline
    rather than a grace period invented here: past it the sweep was going to stop regardless,
    so nothing is abandoned that would otherwise have been kept.

    Both give-ups are reported as **that flight timing out**, which re-enters the discard
    rule the sequential loop already defines: the caller gets a short prefix with
    ``timeout_flight_id`` set and ``complete`` False. A truncated prefix that merely *looked*
    complete would be far worse than a hang -- ``master.upper_bound`` would take a bound over
    flights that were never priced.

    **Why there is no reorder buffer.** Each worker is its own queue, so the global FIFO the
    original leaned on is gone. It is not needed: a worker's results are FIFO in the order
    that worker's flights appear in ``pricing_order``, so the j-th result out of worker *w*
    IS the j-th ``pricing_order`` entry assigned to *w*. Walking ``pricing_order`` and taking
    from the one worker that owns the next index therefore yields exactly index order, which
    rule 2 requires.
    """

    pending = [collections.deque() for _ in range(n_workers)]
    if progress is None:
        progress = _SweepProgress(len(pricing_order), n_workers)
    dead: set[int] = set()
    done = 0
    progress.begin()
    for flight_id in pricing_order:
        worker = worker_of[flight_id]
        while not pending[worker]:
            if worker in dead:
                warnings.warn(
                    f"pricing worker {worker} died mid-sweep; its result for flight "
                    f"{flight_id} will never arrive (OOM killer?)",
                    RuntimeWarning, stacklevel=2,
                )
                progress.finish(done, complete=False)
                yield flight_id, False, 0.0, None, 0.0, {"pool_worker_lost": 1}, {}
                return
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            # COLLECT FIRST, then check the clock. Testing the deadline before draining
            # would discard results that had already arrived merely because the clock
            # passed while they sat in the pipe -- work that was done, paid for, and is
            # part of the prefix the sequential loop would have kept.
            #
            # Sliced rather than waiting the whole remaining budget in one call, so a
            # worker that dies is noticed within a poll interval even if its sentinel
            # somehow does not wake the wait.
            slice_s = (
                _WORKER_POLL_S if remaining is None else min(remaining, _WORKER_POLL_S)
            )
            collected, died = collect(slice_s)
            for index, results in collected.items():
                pending[index].extend(results)
            dead |= died
            if pending[worker] or worker in dead:
                continue
            if remaining is not None and time.monotonic() >= deadline:
                progress.finish(done, complete=False)
                yield flight_id, False, 0.0, None, 0.0, {}, {}
                return
        result = pending[worker].popleft()
        if result[0] != flight_id:
            # One comparison per flight, and it turns a merge bug from a WRONG NUMBER --
            # reduced costs silently transposed between two flights -- into a crash.
            raise RuntimeError(
                f"pricing worker {worker} returned flight {result[0]} where pricing_order "
                f"expects {flight_id}"
            )
        done += 1
        progress.advance(done)
        yield result
    progress.finish(done, complete=True)


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
            # And its COUNTERS are kept, which the seconds argument above does not extend
            # to.  The only counter an unpriced result ever carries is the synthetic
            # `pool_worker_lost` marker `_sweep_results` attaches when it finds the worker
            # gone -- an ordinary `PricingTimeout` returns `{}` -- so dropping them here
            # discarded the one signal that distinguishes "a worker died" from "the clock
            # ran out", and made `solver`'s `pricing_worker_lost` branch unreachable.
            counters.update(deltas)
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
