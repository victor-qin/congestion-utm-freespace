"""Process-parallel pricing sweep for column generation.

Within one CG iteration the pricing subproblems are genuinely independent: ``solve`` builds
``dual_view`` and ``flight_duals`` once before the sweep and never mutates them inside it, each
call touches only its own ``FlightGraph``, and the resulting columns enter the restricted master
afterwards in a sorted order.  So this is a plain fan-out -- no speculation, no read-envelope
validation, nothing like :mod:`freespace_sim.parallel` (which exists because the A* sim has a
shared ledger and commit ordering).

**Processes, not threads.**  ``price_flight`` is still mostly Python around the compiled kernel;
the GIL-free fraction was measured at 19% on a real workload, which caps thread parallelism near
1.2x.  Processes also give the property threads cannot: a worker that exits returns its whole
arena to the OS.  That matters here because the label pool is allocated and freed per flight, and
the churn leaves a large allocator residue -- 2.5 GB survived ``gc.collect()`` with every
``FlightGraph`` unreachable in a 100-flight density measurement.  ``max_tasks_per_child`` turns
that from a leak-shaped curve into a sawtooth bounded by k flights of churn.

**One task per flight, not chunks.**  The per-flight cost is extremely skewed -- 0.09s to 34.4s
in the measured density sweep -- so a static chunking would leave workers idle behind one long
flight.  Dynamic single-flight dispatch is what absorbs that.

**Cost of the fan-out, all measured on density_faa_wing_zipline:** a ``FlightGraph`` pickles to
39 KB (0.3 ms dump, 3.7 ms load) because ``__reduce__`` ships inputs and the geometry rebuilds
lazily; ``DualView`` is 193 KB (5 ms / 33 ms); a cold worker costs 0.36s (0.17s import + 0.19s
``warm_kernel``).  Duals therefore ride in ``initargs`` (once per worker, not once per task) and
graphs ride per task.

**Determinism.**  The sequential sweep ``break``s at the first ``PricingTimeout``, so which
flights got priced is a function of ``pricing_order``.  Dispatching in parallel and keeping
whatever finished would make that wall-clock dependent, and the accepted column set -- hence the
master LP -- would stop being reproducible.  :func:`price_sweep` instead accepts the longest
completed *prefix* of ``pricing_order``, discarding results past the first timeout.  That
reproduces sequential semantics exactly, at the cost of throwing away work when the deadline
binds.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import resource
import sys
import time
from dataclasses import dataclass, field

from .params import ColGenParams

# ru_maxrss is bytes on macOS, kilobytes on Linux.
_MAXRSS_SCALE = 1 if sys.platform == "darwin" else 1024


@dataclass(frozen=True, slots=True)
class ParallelPricingConfig:
    """Knobs for the parallel pricing sweep.  Pure performance -- results are unchanged.

    ``n_workers`` 0 (default) keeps the sequential sweep.  ``max_tasks_per_child`` bounds how
    much allocator residue a worker can accumulate before it is replaced, counted in
    FLIGHTS; ``None`` never recycles.  ``start_method`` defaults to the platform default
    (``spawn`` on macOS, ``fork`` on Linux) -- ``fork`` additionally gives workers the
    parent's graphs copy-on-write, but is unsafe in a process that already has threads
    running.
    """

    n_workers: int = 0
    # In FLIGHTS, which is the unit residue accumulates in -- a label pool is allocated
    # and freed per flight.  `mp.Pool`'s own `maxtasksperchild` counts CHUNKS, so the two
    # knobs silently multiplied before `pool_maxtasksperchild` converted between them:
    # 4 here with chunksize=4 gave a worker a 16-flight life, 4x the documented bound.
    max_tasks_per_child: int | None = 4
    # Flights handed to a worker per dispatch.  1 is maximum load balance and one IPC
    # round-trip per flight; k amortises that over k flights but lets one slow flight
    # hold k-1 others behind it.  1 was right when per-flight cost spanned 0.09-34s.
    # Note k also divides the respawn count, so raising it buys fewer worker restarts
    # (~0.36s of import + warm_kernel each) on top of the dispatch saving.
    chunksize: int = 1
    start_method: str | None = None
    # Called in the PARENT as each task returns, with
    # (done, total, flight_id, wall_s, kernel_stats).  A long sweep is otherwise opaque
    # while it runs: the per-flight record below is only assembled once every task is
    # back, so a 15-minute run shows nothing until it ends.
    on_progress: object = None

    def __post_init__(self) -> None:
        if self.n_workers < 0:
            raise ValueError("n_workers must be non-negative (0 disables parallel pricing)")
        if self.max_tasks_per_child is not None and self.max_tasks_per_child < 1:
            raise ValueError("max_tasks_per_child must be >= 1 or None")
        if self.chunksize < 1:
            raise ValueError("chunksize must be >= 1")

    @property
    def enabled(self) -> bool:
        return self.n_workers > 0

    @property
    def pool_maxtasksperchild(self) -> int | None:
        """``max_tasks_per_child`` expressed in the CHUNKS ``mp.Pool`` actually counts.

        `Pool`'s worker loop increments its completion counter once per item pulled off
        the inqueue, and with ``imap_unordered(chunksize=k)`` one such item is a chunk of
        k flights.  Passing the flight budget through unconverted therefore recycles k
        times too late.  Measured directly (64 items, 2 workers, maxtasksperchild=4):
        chunksize=1 gives 16 distinct pids, chunksize=4 gives 4.

        Rounded UP so a flight budget smaller than one chunk still yields a legal value;
        the effective bound is then one chunk, which is the finest granularity the pool
        can express at that chunksize.
        """

        if self.max_tasks_per_child is None:
            return None
        return max(1, -(-self.max_tasks_per_child // self.chunksize))


# --------------------------------------------------------------------------------- worker side

# Set by _init_worker, read by _price_one.  A module global rather than a closure because the
# pool pickles the callable, and the whole point is to keep the per-sweep constants
# (cfg, params, duals) out of the per-task payload.
_WORKER: dict = {}


def _init_worker(repo_root: str | None, cfg, params: ColGenParams, dual_view) -> None:
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from . import pricing as pricing_mod

    _WORKER["pricing"] = pricing_mod
    _WORKER["cfg"] = cfg
    _WORKER["params"] = params
    _WORKER["duals"] = dual_view
    # Pay the JIT/cache-load here rather than inside the first flight's timing.
    if pricing_mod._dp_kernel is not None:
        pricing_mod._dp_kernel.warm_kernel()


def _price_one(task):
    """Price one flight.

    Returns ``(flight_id, reduced_cost, column, pid, peak_rss_bytes, task_wall_s)``.

    ``PricingTimeout`` is returned as a sentinel rather than raised, so the parent can rebuild
    the sequential prefix semantics instead of losing the whole sweep to one exception.

    ``task_wall_s`` is the useful work in this task.  Summed over a sweep and compared against
    ``n_workers * sweep_wall`` it separates the two ways a fan-out disappoints -- pool overhead
    (sum is much less than the sequential total) versus load imbalance (sum matches, but the
    makespan is pinned by one long task).  Without it, a bad speedup is unattributable.
    """
    started = time.perf_counter()
    flight_id, graph, pi_f, deadline, known_column = task
    pricing_mod = _WORKER["pricing"]
    try:
        reduced_cost, column = pricing_mod.price_flight(
            graph, _WORKER["duals"], pi_f, _WORKER["cfg"], _WORKER["params"],
            known_column=known_column, deadline=deadline,
        )
    except pricing_mod.PricingTimeout:
        reduced_cost, column = None, None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _MAXRSS_SCALE
    # Read the kernel's own account of this flight before anything else overwrites it.
    # This is the only route to straggler attribution: the time is spent in the worker,
    # and a parent-side patch cannot reach it because the pool pickles `_price_one` by
    # reference and the worker resolves the original.
    stats = dict(pricing_mod._LAST_KERNEL_STATS)
    return (
        flight_id, reduced_cost, column, os.getpid(), peak,
        time.perf_counter() - started, stats,
    )


# --------------------------------------------------------------------------------- parent side


@dataclass(slots=True)
class SweepResult:
    """Outcome of one parallel sweep, shaped to drop into the sequential loop's variables."""

    priced: list                      # (flight_id, reduced_cost, column) in pricing_order
    complete: bool
    timeout_flight_id: int | None
    worker_pids: set
    worker_peak_rss_bytes: int        # max over workers, not the simultaneous total
    tasks_discarded: int              # completed past the first timeout, dropped for determinism
    sweep_wall_s: float               # makespan: submit -> last task back
    task_wall_total_s: float          # sum of per-task work; ~= the sequential sweep cost
    task_wall_max_s: float            # the straggler; a hard floor under sweep_wall_s
    # (flight_id, wall_s, worker_peak_rss_bytes, kernel_stats) per task, so the
    # straggler can be named and explained rather than just counted.
    per_flight: list = field(default_factory=list)

    @property
    def efficiency(self) -> float:
        """Useful work divided by worker-seconds bought.  1.0 = perfect packing."""
        denom = self.sweep_wall_s * max(1, len(self.worker_pids))
        return self.task_wall_total_s / denom if denom else 0.0


def price_sweep(
    *,
    graphs,
    pricing_order,
    dual_view,
    flight_duals,
    cfg,
    params: ColGenParams,
    known_columns=None,
    deadline: float | None,
    pool_cfg: ParallelPricingConfig,
) -> SweepResult:
    """Run one pricing sweep across a process pool, preserving sequential semantics."""
    repo_root = str(_repo_root())
    ctx = mp.get_context(pool_cfg.start_method) if pool_cfg.start_method else mp.get_context()

    results: dict = {}
    worker_pids: set = set()
    peak_rss = 0
    task_walls: list[float] = []
    per_flight: list[tuple] = []
    sweep_started = time.perf_counter()
    # multiprocessing.Pool rather than concurrent.futures.ProcessPoolExecutor, for one
    # specific reason: ProcessPoolExecutor's `max_tasks_per_child` DEADLOCKS on CPython
    # 3.14.2 as soon as recycling actually fires.  Reproduced with no initializer and 40
    # trivial `return os.getpid()` tasks over 2 spawn workers -- the main thread parks in
    # as_completed (_base.py:237) and the executor thread in wait_result_broken_or_wakeup,
    # with zero workers alive.  `max_tasks_per_child=None` on the same code is fine, which
    # is why a small smoke test misses it: with fewer tasks than n_workers * k, recycling
    # never triggers.  mp.Pool's `maxtasksperchild` is an independent implementation and
    # recycles correctly (40 tasks / k=4 -> exactly 10 distinct pids).
    #
    # imap_unordered with chunksize=1 (the default) preserves the dynamic
    # one-task-per-flight dispatch that the skewed per-flight cost needs; it also pickles
    # each graph lazily as a worker becomes free rather than all of them up front.
    known = known_columns or {}
    tasks = [
        (flight_id, graphs[flight_id], flight_duals[flight_id], deadline,
         known.get(flight_id))
        for flight_id in pricing_order
    ]
    with ctx.Pool(
        processes=pool_cfg.n_workers,
        initializer=_init_worker,
        initargs=(repo_root, cfg, params, dual_view),
        maxtasksperchild=pool_cfg.pool_maxtasksperchild,
    ) as pool:
        for (
            flight_id, reduced_cost, column, pid, peak, task_wall, kstats
        ) in pool.imap_unordered(_price_one, tasks, chunksize=pool_cfg.chunksize):
            results[flight_id] = (reduced_cost, column)
            worker_pids.add(pid)
            peak_rss = max(peak_rss, peak)
            task_walls.append(task_wall)
            per_flight.append((flight_id, task_wall, peak, kstats))
            if pool_cfg.on_progress is not None:
                pool_cfg.on_progress(
                    len(per_flight), len(tasks), flight_id, task_wall, kstats
                )
    sweep_wall = time.perf_counter() - sweep_started

    # Rebuild the sequential prefix: stop at the first flight that timed out, and drop
    # everything after it even though it completed.  See the module docstring.
    priced: list = []
    complete = True
    timeout_flight_id = None
    for flight_id in pricing_order:
        reduced_cost, column = results[flight_id]
        if reduced_cost is None:
            complete = False
            timeout_flight_id = flight_id
            break
        priced.append((flight_id, reduced_cost, column))

    return SweepResult(
        priced=priced,
        complete=complete,
        timeout_flight_id=timeout_flight_id,
        worker_pids=worker_pids,
        worker_peak_rss_bytes=peak_rss,
        tasks_discarded=len(results) - len(priced) - (0 if complete else 1),
        sweep_wall_s=sweep_wall,
        task_wall_total_s=sum(task_walls),
        task_wall_max_s=max(task_walls, default=0.0),
        per_flight=per_flight,
    )


def _repo_root():
    """The tree this module lives in, so a spawned worker imports the same code.

    ``spawn`` re-executes the interpreter without inheriting ``sys.path`` additions, so a
    worker running under a bare ``python analysis/foo.py`` would otherwise import a
    *different* ``freespace_sim`` (or none).  Passing the root explicitly is the same
    discipline the analysis harnesses assert.
    """
    from pathlib import Path

    return Path(__file__).resolve().parents[3]
