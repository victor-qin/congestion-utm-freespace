"""Persistent process workers for Zeta initial columns and frozen-dual pricing.

Each flight has one owner at a time, so its seed geometry and canonical-column caches survive
successive pricing sweeps. Before each pricing sweep, owners are rebalanced from the previous
measured task times (sticky longest-first packing): a few capped searches otherwise pile onto
one round-robin worker and set the sweep's wall time. Only columns a worker has not yet
received are shipped. Results are restored to flight order before master insertion.
The transport follows the production DP pool's raw spawn/pipe design: no queue-handler
threads, no worker respawn, explicit sweep epochs, and bounded process teardown.
"""

from __future__ import annotations

import gc
import itertools
from collections import OrderedDict
import math
import multiprocessing as mp
import multiprocessing.connection as mp_connection
import os
import pickle
import resource
import sys
import time
import traceback
import uuid
from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from types import TracebackType
from typing import Any

from .network import RowKey
from .objective import cost_model
from .params import ColGenParams
from .pricing_pool import _worker_assignment
from .translate import Column
from .zeta import ZetaGraph, ZetaPricingResult, price_zeta

_GRACEFUL_STOP_S = 0.5
_KILL_JOIN_S = 5.0
# A flight leaves its current owner only when keeping it would exceed the balanced load by
# this factor; moving costs the new owner a cold graph and cache build.
_REBALANCE_SLACK = 1.05
# Measured worker memory (.agent/zeta-runtime-main-20261120/README.md, "Safety cap"), all
# rounded up: peak resident growth of one best-first search per stored label (2.9-3.9 KB,
# linear from 100k to 4M labels), an idle worker (118 MB) and one priced flight graph kept
# between sweeps (38 MB). Re-measure whenever label storage changes (e.g. a numba loop).
SEARCH_BYTES_PER_LABEL = 4096
WORKER_BASE_BYTES = 128 * 2**20
GRAPH_RESIDENT_BYTES = 40 * 2**20
# Fraction of the memory limit the solve may plan to use; the rest absorbs allocator
# fragmentation, the operating system and measurement spread.
_MEMORY_HEADROOM = 0.8


@dataclass(frozen=True, slots=True)
class ZetaSeedResult:
    """Canonical initial columns from one flight, including a valid partial prefix.

    Attributes
    ------------
    - columns (tuple[Column, ...]): supplied, nominal and departure-ladder columns in order.
    - provided_primary (Column | None): first accepted supplied column, if any.
    - complete (bool): all seed candidates were examined before the deadline.
    - reason (str): ``complete`` or ``deadline``; neither certifies pricing.
    """

    columns: tuple[Column, ...]
    provided_primary: Column | None
    complete: bool
    reason: str


def build_seed_columns(
    graph: ZetaGraph, params: ColGenParams, supplied: Sequence[Column] = (),
    forbidden_rows: AbstractSet[RowKey] = frozenset(), deadline: float | None = None,
) -> ZetaSeedResult:
    """Construct and validate the same ordered initial pool in serial or in its owner.

    Parameters
    ------------
    - graph (ZetaGraph): persistent graph whose geometry remains available to pricing.
    - params (ColGenParams): objective and nominal/departure-ladder settings.
    - supplied (Sequence[Column]): optional columns to recanonicalize first. An explicit
      ``seed_validation_radius_m`` admits full-graph seeds at that radius into the fixed
      initial pool; later pricing still searches only the graph's connection domain.
    - forbidden_rows (AbstractSet[RowKey]): reservations unavailable to this solve.
    - deadline (float | None): absolute monotonic cutoff shared with pricing.

    Return
    --------
    - output (ZetaSeedResult): canonical, unique, allowed columns and completion status.
    """
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("deadline must be finite")
    if deadline is not None and time.perf_counter() >= deadline:
        return ZetaSeedResult((), None, False, "deadline")
    candidates = list(supplied)
    supplied_count = len(candidates)
    supplied_graph = graph
    if (supplied_count and graph.options.seed_validation_radius_m is not None
            and (graph.options.connection_mode != "full"
                 or graph.options.seed_validation_radius_m != graph.options.connection_radius_m)):
        supplied_graph = ZetaGraph(
            graph.fg, graph.cfg, graph.row_index,
            replace(graph.options, connection_mode="full", traversal="best_first",
                    connection_radius_m=graph.options.seed_validation_radius_m,
                    seed_validation_radius_m=None),
        )
    nominal = graph.seed_columns(params, deadline=deadline) if params.seed_nominal_routes else ()
    candidates.extend(nominal)
    if nominal:
        primary = nominal[0]
        for shift in range(1, params.seed_ladder_steps + 1):
            departure = primary.departure_step + shift * params.seed_ladder_stride
            candidates.append(replace(primary, departure_step=departure))
    model = cost_model(graph.cfg, params)
    columns: list[Column] = []
    seen: set[Column] = set()
    provided_primary: Column | None = None
    for index, raw in enumerate(candidates):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        authority = supplied_graph if index < supplied_count else graph
        column = authority.canonical_column(raw, model)
        if column is None or column in seen or not column.claims.isdisjoint(forbidden_rows):
            continue
        columns.append(column)
        seen.add(column)
        if index < supplied_count and provided_primary is None:
            provided_primary = column
    complete = deadline is None or time.perf_counter() < deadline
    return ZetaSeedResult(tuple(columns), provided_primary, complete,
                          "complete" if complete else "deadline")


@dataclass(frozen=True, slots=True)
class ZetaSweepResult:
    """Collected task results, distinct from individual search certificates.

    Attributes
    ------------
    - results (tuple[tuple[int, ZetaPricingResult], ...]): received results in flight order.
    - complete (bool): every flight returned a result; capped searches still count.
    - failure (str | None): deadline, worker, or protocol failure; caps are not failures.
    - setup_s (float): startup time paid by this sweep, included in ``wall_s``.
    - wall_s (float): elapsed sweep time, including transport and cleanup on failure.
    - task_s (float): summed received task times, including first-touch graph construction.
    - flight_records (tuple[dict[str, Any], ...]): per-flight time, worker and search evidence.
    - worker_peak_rss_bytes (dict[int, int]): process lifetime high-water RSS by worker.
    - failure_detail (str | None): diagnostic text accompanying a failure.
    """

    results: tuple[tuple[int, ZetaPricingResult], ...]
    complete: bool
    failure: str | None
    setup_s: float
    wall_s: float
    task_s: float
    flight_records: tuple[dict[str, Any], ...] = ()
    worker_peak_rss_bytes: dict[int, int] = field(default_factory=dict)
    failure_detail: str | None = None

    @property
    def termination_reason(self) -> str | None:
        """Expose transport termination separately from individual search cap reasons."""
        return self.failure


@dataclass(frozen=True, slots=True)
class ZetaSeedSweepResult(ZetaSweepResult):
    """Collected initial columns and transport diagnostics, never a pricing certificate.

    Attributes
    ------------
    - results (tuple[tuple[int, ZetaSeedResult], ...]): received seeds in flight order.
    """

    results: tuple[tuple[int, ZetaSeedResult], ...]


@dataclass(slots=True)
class _WorkerChannel:
    """A process and its parent's pipe endpoint.

    Attributes
    ------------
    - index (int): stable worker index.
    - proc (BaseProcess): spawned worker, retained until bounded cleanup.
    - conn (Connection): parent's endpoint; the child endpoint is closed after spawn.
    - pid (int | None): PID received in the startup handshake.
    """

    index: int
    proc: BaseProcess
    conn: Connection
    pid: int | None = None


class _PoolFailure(RuntimeError):
    """An explicit transport failure that cannot certify unreturned flight searches."""

    def __init__(self, code: str, detail: str) -> None:
        """Keep the stable failure code separate from diagnostic text."""
        super().__init__(detail)
        self.code = code


def _peak_rss_bytes() -> int:
    """Normalize Unix process high-water RSS to bytes on macOS and Linux."""
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _current_rss_bytes() -> int:
    """Return resident memory now on Linux; elsewhere the high-water mark, an overestimate."""
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return _peak_rss_bytes()


class _GraphBudget:
    """Release least recently priced flight graphs once their charged memory exceeds a limit.

    Resident-memory growth measured around each task is charged to the flight it priced.
    Freed Python memory is reused rather than returned to the system, so a release lowers the
    charge without waiting for RSS to fall. A released graph is rebuilt cold from its
    definition on next use, exactly as when a flight moves to another worker, so results
    never change; only time is spent rebuilding.

    Attributes
    ------------
    - limit (int | None): charged-byte budget; None never releases, 0 keeps no graph beyond
      the flights still queued in the current sweep.
    """

    def __init__(self, limit: int | None) -> None:
        """Start with no charges; the order is least recently priced first."""
        self.limit = limit
        self._charge: OrderedDict[int, int] = OrderedDict()

    def charge(self, flight_id: int, grown_bytes: int) -> None:
        """Add one task's non-negative memory growth and mark the flight most recent."""
        self._charge[flight_id] = self._charge.get(flight_id, 0) + max(0, int(grown_bytes))
        self._charge.move_to_end(flight_id)

    def forget(self, flight_id: int) -> None:
        """Drop the charge of a graph released for another reason (ownership moved)."""
        self._charge.pop(flight_id, None)

    def victims(self, queued: AbstractSet[int]) -> tuple[int, ...]:
        """Return graphs to release, oldest first, never one still queued in this sweep.

        Parameters
        ------------
        - queued (AbstractSet[int]): flights this sweep has yet to price; their definitions
          were not resent, so their graphs must survive until priced.

        Return
        --------
        - output (tuple[int, ...]): released flight IDs, whose charges are removed.
        """
        if self.limit is None:
            return ()
        total = sum(self._charge.values())
        chosen = []
        for flight_id in list(self._charge):
            if self.limit > 0 and total <= self.limit:
                break
            if flight_id in queued:
                continue
            total -= self._charge.pop(flight_id)
            chosen.append(flight_id)
        return tuple(chosen)


def search_label_cap(memory_bytes: int) -> int:
    """Size the label threshold from measured bytes; best-first may finish one successor batch."""
    return max(1, int(memory_bytes) // SEARCH_BYTES_PER_LABEL)


def search_memory_bytes(
    memory_limit_bytes: int, n_workers: int, n_flights: int, *,
    parent_reserve_bytes: int,
) -> int:
    """Size one pricing search's memory from a worker's share of the machine.

    Every worker may run one search at a time, so the plannable memory (``_MEMORY_HEADROOM``
    of the limit) minus the parent's reserve is split evenly; each share first pays for the
    worker's idle footprint and an equal share of all flight graphs, and the remainder is the
    search's. Ownership can be uneven after rebalancing; this reserves graph memory across
    the whole pool. Cache eviction never increases this budget: RSS-delta charges are a
    retention heuristic, not a bound on live objects or active-search geometry.

    Parameters
    ------------
    - memory_limit_bytes (int): memory this solve may use (job allocation or machine RAM).
    - n_workers (int): pricing worker processes; 0 prices in the parent, as one worker.
    - n_flights (int): flights in the solve, spread evenly over the workers.
    - parent_reserve_bytes (int): the parent's own peak (master LP, graphs, IP solver).

    Return
    --------
    - output (int): positive byte budget for ``ColGenParams.zeta_search_memory_bytes``.
    """
    workers = max(1, int(n_workers))
    owned = math.ceil(max(0, int(n_flights)) / workers)
    graphs = owned * GRAPH_RESIDENT_BYTES
    share = (_MEMORY_HEADROOM * memory_limit_bytes - parent_reserve_bytes) / workers
    search = int(share - WORKER_BASE_BYTES - graphs)
    if search < SEARCH_BYTES_PER_LABEL * 10_000:
        raise ValueError(
            f"a {workers}-worker share of {memory_limit_bytes / 2**30:.1f} GiB leaves "
            f"{max(0, search) / 2**20:.0f} MiB per search; use fewer workers or more memory")
    return search


def _settle_worker_heap(full: bool) -> None:
    """Collect cyclic garbage while the worker is idle, then exempt survivors from later passes.

    Workers never collect automatically: a search allocates millions of short-lived acyclic
    objects, and the passes they trigger rescan every cached graph yet free nothing that
    reference counting does not (~8% of per-expansion time in saved-call profiles). The one
    collection runs after a sweep's last result is sent, before the worker acknowledges
    readiness for another sweep. Freezing keeps the next collection to what the next sweep leaves alive.

    Parameters
    ------------
    - full (bool): also rescan frozen objects, needed after graphs were released or moved:
      a frozen reference cycle is otherwise never reclaimed.

    Return
    --------
    - output (None): the heap is collected and frozen; results never depend on it.
    """
    if full:
        gc.unfreeze()
    gc.collect()
    gc.freeze()


def _limit_worker_threads() -> Any:
    """Limit numerical runtimes inside this worker without changing parent settings."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        from numba import set_num_threads
    except ImportError:
        pass
    else:
        set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return None
    return threadpool_limits(limits=1)


def _worker_main(
    conn: Connection, worker: int, params: ColGenParams,
    native_artifact: tuple[str, str] | None = None,
) -> None:
    """Keep owned graphs alive and stream each flight result to the parent.

    Parameters
    ------------
    - conn (Connection): child endpoint of one duplex pipe.
    - worker (int): stable worker index reported with every message.
    - params (ColGenParams): immutable objective and search controls.
    - native_artifact (tuple | None): parent-prepared compiled loop; workers never build it.

    Return
    --------
    - output (None): exits on stop, pipe closure, or a reported worker exception.
    """
    epoch: tuple[str, int] | None = None
    try:
        # This process owns its collector; `_settle_worker_heap` is the only collection.
        gc.disable()
        # Retain the optional controller for the lifetime of this process.
        _thread_limits = _limit_worker_threads()
        if native_artifact is not None:
            from ..zeta.native import install_compiled_loop
            install_compiled_loop(native_artifact)
        graphs: dict[int, ZetaGraph] = {}
        budget = _GraphBudget(params.zeta_worker_cache_bytes)
        # Pricing columns accumulate here; the parent ships only the unsent suffix.
        known_by_flight: dict[int, list[Column]] = {}
        _settle_worker_heap(full=False)
        conn.send(("ready", worker, os.getpid(), _peak_rss_bytes()))
        previous_epoch: tuple[str, int] | None = None
        while True:
            message = conn.recv()
            if message == ("stop",):
                return
            tag, epoch, duals_blob, tasks, definitions, forbidden, deadline, drops = message
            if tag not in ("sweep", "seed") or any(
                task[0] not in graphs and task[0] not in definitions for task in tasks
            ):
                raise RuntimeError("worker received an invalid sweep assignment")
            if previous_epoch is not None and (
                epoch[0] != previous_epoch[0] or epoch[1] <= previous_epoch[1]
            ):
                raise RuntimeError("worker received a stale sweep epoch")
            previous_epoch = epoch
            full_settle = bool(drops)
            for flight_id in drops:
                graphs.pop(flight_id, None)
                known_by_flight.pop(flight_id, None)
                budget.forget(flight_id)
            duals = pickle.loads(duals_blob) if tag == "sweep" else {}
            for position, (flight_id, flight_dual, columns, extend) in enumerate(tasks):
                measure_cache = budget.limit is not None and budget.limit > 0
                rss_before = _current_rss_bytes() if measure_cache else 0
                started = time.perf_counter()
                if deadline is not None and started >= deadline:
                    conn.send(("deadline", epoch, worker, _peak_rss_bytes()))
                    return
                if flight_id not in graphs:
                    graphs[flight_id] = ZetaGraph(*definitions[flight_id])
                if tag == "seed":
                    result = build_seed_columns(
                        graphs[flight_id], params, columns, forbidden, deadline,
                    )
                else:
                    if extend:
                        known_by_flight.setdefault(flight_id, []).extend(columns)
                    else:
                        known_by_flight[flight_id] = list(columns)
                    result = price_zeta(
                        graphs[flight_id], duals, flight_dual, params,
                        known_columns=known_by_flight[flight_id], forbidden_rows=forbidden,
                        deadline=deadline,
                    )
                task_s = time.perf_counter() - started
                # Known columns stay: the parent's record of what this worker holds is unchanged.
                released = ()
                if budget.limit is not None:
                    budget.charge(flight_id, _current_rss_bytes() - rss_before if measure_cache else 0)
                    released = budget.victims({task[0] for task in tasks[position + 1:]})
                for victim in released:
                    graphs.pop(victim, None)
                full_settle = full_settle or bool(released)
                conn.send(("result", epoch, worker, flight_id, result,
                           task_s, _peak_rss_bytes(), released))
                if result.reason == "deadline":
                    return
            # Cleanup must finish before another potentially large send can reach this pipe.
            # The parent waits for this acknowledgement under the sweep deadline, including
            # workers assigned no flights after rebalancing.
            _settle_worker_heap(full_settle)
            conn.send(("settled", epoch, worker))
    except (EOFError, BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
        return
    except BaseException:
        try:
            conn.send(("error", epoch, worker, traceback.format_exc()))
        except (OSError, ValueError):
            pass
    finally:
        conn.close()


class ZetaPricingPool:
    """Solve-scoped, cost-rebalanced ownership of persistent Zeta pricing processes.

    Attributes
    ------------
    - worker_of (dict[int, int]): a copy of the current flight-to-worker assignment.
    - n_workers (int): requested worker count capped to the number of flights.
    - graphs_released (int): flight graphs workers released under
      ``ColGenParams.zeta_worker_cache_bytes``; each is rebuilt on its next sweep.
    """

    def __init__(self, graphs: Mapping[int, ZetaGraph], params: ColGenParams) -> None:
        """Prepare ownership and cold graph definitions without starting workers.

        Parameters
        ------------
        - graphs (Mapping[int, ZetaGraph]): fixed route domains indexed by flight ID.
        - params (ColGenParams): includes a positive ``n_pricing_workers`` setting.

        Return
        --------
        - output (None): creates a lazy pool; parent graph caches never cross the pipe.
        """
        if params.n_pricing_workers <= 0:
            raise ValueError("ZetaPricingPool needs at least one requested worker")
        if params.pricing_work_stealing:
            raise ValueError("ZetaPricingPool currently requires pricing_work_stealing=False")
        self._flight_ids = tuple(sorted(graphs))
        if any(fid != graphs[fid].fg.request.flight_id for fid in self._flight_ids):
            raise ValueError("graph keys must match their request flight IDs")
        self._definitions = {
            fid: (graph.fg, graph.cfg, graph.row_index, graph.options)
            for fid, graph in graphs.items()
        }
        self._params = params
        self._n_workers = min(params.n_pricing_workers, len(graphs))
        self._worker_of = _worker_assignment(self._flight_ids, self._n_workers)
        self._owned = tuple(
            tuple(fid for fid in self._flight_ids if self._worker_of[fid] == worker)
            for worker in range(self._n_workers)
        )
        self._channels: list[_WorkerChannel] | None = None
        # Per worker: flights whose graph it holds, columns it already holds per flight
        # (by identity, to detect non-append-only callers), and graphs it must release.
        self._loaded: list[set[int]] = [set() for _ in range(self._n_workers)]
        self._shipped: list[dict[int, tuple[Column, ...]]] = [{} for _ in range(self._n_workers)]
        self._drops: list[set[int]] = [set() for _ in range(self._n_workers)]
        # Latest measured task seconds per flight; seed times stand in until a sweep runs.
        self._cost: dict[int, float] = {}
        self._priced = False
        self.graphs_released = 0
        self._rss: dict[int, int] = {}
        self._uid = uuid.uuid4().hex
        self._epochs = itertools.count(1)
        self._poisoned: str | None = None
        self._busy = False
        self._setup_s = 0.0
        self._close_s = 0.0

    @property
    def worker_of(self) -> dict[int, int]:
        """Return a copy so callers cannot mutate worker ownership."""
        return dict(self._worker_of)

    def _rebalance(self) -> None:
        """Repack owners longest-first from measured task times, keeping owners that fit.

        Flights are visited heaviest first. Each stays with its current owner unless that
        would push the owner past the balanced load (the larger of mean load and heaviest
        task) times ``_REBALANCE_SLACK``; otherwise it goes to the least-loaded worker.
        Moved flights are released by their old owner and rebuilt cold by the new one, so
        results are unchanged: only which process prices a flight changes.
        """
        if self._n_workers <= 1 or not self._cost:
            return
        default = math.fsum(self._cost.values()) / len(self._cost)
        cost = {fid: self._cost.get(fid, default) for fid in self._flight_ids}
        target = max(math.fsum(cost.values()) / self._n_workers, max(cost.values()))
        loads = [0.0] * self._n_workers
        assignment: dict[int, int] = {}
        order = sorted(self._flight_ids, key=lambda fid: (-cost[fid], fid))
        for fid in order:
            worker = self._worker_of[fid]
            if loads[worker] + cost[fid] > target * _REBALANCE_SLACK:
                worker = min(range(self._n_workers), key=lambda index: (loads[index], index))
            loads[worker] += cost[fid]
            assignment[fid] = worker
        for fid, worker in assignment.items():
            previous = self._worker_of[fid]
            if worker != previous:
                self._drops[previous].add(fid)
                self._drops[worker].discard(fid)
                self._loaded[previous].discard(fid)
                self._shipped[previous].pop(fid, None)
        self._worker_of = assignment
        # Heaviest first within a worker, so a deadline leaves the cheap tail unpriced.
        self._owned = tuple(
            tuple(fid for fid in order if assignment[fid] == worker)
            for worker in range(self._n_workers)
        )

    def _column_payload(
        self, worker: int, fid: int, columns: Sequence[Column],
    ) -> tuple[tuple[Column, ...], bool]:
        """Return the columns ``worker`` lacks for ``fid`` and whether they extend its list.

        Parameters
        ------------
        - worker (int): worker index that will price ``fid``.
        - fid (int): flight whose known columns are being shipped.
        - columns (Sequence[Column]): the caller's complete current known columns.

        Return
        --------
        - output (tuple[tuple[Column, ...], bool]): an unsent suffix with ``True`` when the
          worker's list is an identity prefix of ``columns``; otherwise all columns and
          ``False`` so the worker replaces its list.
        """
        current = tuple(columns)
        sent = self._shipped[worker].get(fid)
        self._shipped[worker][fid] = current
        if sent is not None and len(sent) <= len(current) and all(
            old is new for old, new in zip(sent, current)
        ):
            return current[len(sent):], True
        return current, False

    @property
    def n_workers(self) -> int:
        """Return the number of useful worker processes for this flight set."""
        return self._n_workers

    @property
    def setup_s(self) -> float:
        """Return aggregate worker startup seconds, retained after close."""
        return self._setup_s

    @property
    def close_s(self) -> float:
        """Return aggregate teardown seconds, retained after close."""
        return self._close_s

    @property
    def worker_peak_rss_bytes(self) -> dict[int, int]:
        """Return observed worker lifetime RSS peaks, retained after close."""
        return dict(self._rss)

    def start(self, deadline: float | None = None) -> float:
        """Spawn workers once, watching process death as well as readiness pipes.

        Parameters
        ------------
        - deadline (float | None): absolute monotonic cutoff shared with pricing.

        Return
        --------
        - output (float): setup seconds, zero if started; failure closes all workers.
        """
        if self._poisoned is not None:
            raise RuntimeError(f"Zeta pricing pool is unavailable: {self._poisoned}")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        if self._channels is not None:
            return 0.0
        started = time.perf_counter()
        self._channels = []
        ctx = mp.get_context("spawn")
        try:
            native_artifact = None
            if any(getattr(definition[3], "compiled_loop", False)
                   for definition in self._definitions.values()):
                from ..zeta.native import prepare_compiled_loop
                try:
                    native_artifact = prepare_compiled_loop(deadline)
                except TimeoutError as error:
                    raise _PoolFailure("deadline", str(error)) from error
            for worker in range(self._n_workers):
                if deadline is not None and time.perf_counter() >= deadline:
                    raise _PoolFailure("deadline", "deadline expired during worker startup")
                parent_conn, child_conn = ctx.Pipe()
                worker_args = (child_conn, worker, self._params)
                if native_artifact is not None:
                    worker_args += (native_artifact,)
                proc = ctx.Process(
                    target=_worker_main, args=worker_args,
                    daemon=True,
                )
                channel = _WorkerChannel(worker, proc, parent_conn)
                self._channels.append(channel)
                try:
                    proc.start()
                finally:
                    child_conn.close()
            pending = set(range(self._n_workers))
            while pending:
                messages, dead = self._collect(deadline, pending)
                for worker, message in messages:
                    if message[0] == "error":
                        raise _PoolFailure("worker_error", str(message[-1]))
                    if len(message) != 4 or message[:2] != ("ready", worker):
                        raise _PoolFailure("protocol_error", "invalid worker readiness message")
                    if worker not in pending:
                        raise _PoolFailure("protocol_error", "duplicate worker readiness message")
                    self._channels[worker].pid = int(message[2])
                    self._rss[worker] = int(message[3])
                    pending.remove(worker)
                if dead:
                    raise _PoolFailure("worker_lost", f"workers died during startup: {sorted(dead)}")
                if pending and deadline is not None and time.perf_counter() >= deadline:
                    raise _PoolFailure("deadline", "deadline expired waiting for worker readiness")
        except BaseException as exc:
            self._setup_s += time.perf_counter() - started
            self._poisoned = str(exc)
            self.close()
            raise
        elapsed = time.perf_counter() - started
        self._setup_s += elapsed
        return elapsed

    def _collect(
        self, deadline: float | None, workers: AbstractSet[int],
    ) -> tuple[list[tuple[int, tuple[Any, ...]]], set[int]]:
        """Drain ready pipes and process sentinels in one bounded wait.

        Parameters
        ------------
        - deadline (float | None): absolute cutoff; expired waits still drain ready data.
        - workers (AbstractSet[int]): worker indices whose messages are expected.

        Return
        --------
        - output (tuple): received worker/message pairs and observed dead worker indices.
        """
        channels = self._channels or []
        waitables: dict[Any, tuple[_WorkerChannel, bool]] = {}
        for worker in workers:
            channel = channels[worker]
            waitables[channel.conn] = (channel, True)
            waitables[channel.proc.sentinel] = (channel, False)
        remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
        ready = mp_connection.wait(list(waitables), remaining)
        messages: list[tuple[int, tuple[Any, ...]]] = []
        dead: set[int] = set()
        # Drain pipes before handling sentinels: a dying worker may have returned a result.
        for worker in sorted(workers):
            channel = channels[worker]
            if channel.conn not in ready and channel.proc.sentinel not in ready:
                continue
            try:
                while channel.conn.poll():
                    messages.append((worker, channel.conn.recv()))
            except (EOFError, OSError):
                dead.add(worker)
            if channel.proc.sentinel in ready:
                dead.add(worker)
        return messages, dead

    def run_sweep(
        self, duals: Mapping[RowKey, float], flight_duals: Mapping[int, float],
        known_columns: Mapping[int, Sequence[Column]],
        forbidden_rows: AbstractSet[RowKey] = frozenset(), deadline: float | None = None,
    ) -> ZetaSweepResult:
        """Price every flight using one immutable dual snapshot per worker.

        Parameters
        ------------
        - duals (Mapping[RowKey, float]): frozen row prices, including negative prices.
        - flight_duals (Mapping[int, float]): the current dual for every flight.
        - known_columns (Mapping[int, Sequence[Column]]): incumbents revalidated each call.
        - forbidden_rows (AbstractSet[RowKey]): all saturated rows, even at zero price.
        - deadline (float | None): absolute monotonic cutoff including startup and transport.

        Return
        --------
        - output (ZetaSweepResult): all received results in flight order. Missing tasks forbid
          a complete sweep certificate. Deadline or worker failure closes the pool; ordinary
          search caps leave it reusable. Invalid caller inputs raise after cleanup.
        """
        return self._run_tasks("sweep", duals, flight_duals, known_columns,
                               forbidden_rows, deadline)

    def run_seed_sweep(
        self, seed_columns: Mapping[int, Sequence[Column]],
        forbidden_rows: AbstractSet[RowKey] = frozenset(), deadline: float | None = None,
    ) -> ZetaSeedSweepResult:
        """Build initial columns in the persistent processes that will later price them.

        Parameters
        ------------
        - seed_columns (Mapping[int, Sequence[Column]]): supplied routes by flight ID.
        - forbidden_rows (AbstractSet[RowKey]): unavailable rows, including unpriced ones.
        - deadline (float | None): absolute cutoff for setup, construction and transport.

        Return
        --------
        - output (ZetaSeedSweepResult): canonical columns in deterministic flight order.
          Failure preserves received partial results but closes this pool.
        """
        return self._run_tasks("seed", {}, dict.fromkeys(self._flight_ids, 0.0),
                               seed_columns, forbidden_rows, deadline)

    def _run_tasks(
        self, operation: str, duals: Mapping[RowKey, float],
        flight_duals: Mapping[int, float], columns: Mapping[int, Sequence[Column]],
        forbidden_rows: AbstractSet[RowKey], deadline: float | None,
    ) -> ZetaSweepResult | ZetaSeedSweepResult:
        """Share bounded transport and ownership checks between seed and pricing work.

        Parameters
        ------------
        - operation (str): ``seed`` or ``sweep`` selects the worker operation.
        - duals (Mapping[RowKey, float]): frozen row prices for a pricing sweep.
        - flight_duals (Mapping[int, float]): one flight dual per dispatched task.
        - columns (Mapping[int, Sequence[Column]]): supplied seeds or known pool columns.
        - forbidden_rows (AbstractSet[RowKey]): unavailable reservations for either operation.
        - deadline (float | None): absolute cutoff including worker startup and transport.

        Return
        --------
        - output (ZetaSweepResult | ZetaSeedSweepResult): received results and failure evidence.
        """
        if self._poisoned is not None:
            raise RuntimeError(f"Zeta pricing pool is unavailable: {self._poisoned}")
        started = time.perf_counter()
        setup_s = 0.0
        results: dict[int, ZetaPricingResult | ZetaSeedResult] = {}
        records: dict[int, dict[str, Any]] = {}
        failure: str | None = None
        detail: str | None = None
        try:
            if deadline is not None and not math.isfinite(deadline):
                raise ValueError("deadline must be finite")
            if deadline is not None and started >= deadline:
                raise _PoolFailure("deadline", "deadline expired before pricing startup")
            setup_before = self._setup_s
            try:
                setup_s = self.start(deadline)
            except BaseException:
                setup_s = self._setup_s - setup_before
                raise
            if operation == "sweep":
                self._rebalance()
            duals_blob = pickle.dumps(dict(duals), protocol=pickle.HIGHEST_PROTOCOL)
            forbidden = frozenset(forbidden_rows)
            epoch = (self._uid, next(self._epochs))
            pending = set(self._flight_ids)
            unsettled = set(range(self._n_workers))
            self._busy = True
            for channel in self._channels or ():
                if deadline is not None and time.perf_counter() >= deadline:
                    raise _PoolFailure("deadline", "deadline expired dispatching pricing tasks")
                worker = channel.index
                owned = self._owned[worker]
                if operation == "sweep":
                    tasks = tuple(
                        (fid, flight_duals[fid],
                         *self._column_payload(worker, fid, columns.get(fid, ())))
                        for fid in owned
                    )
                else:
                    tasks = tuple((fid, flight_duals[fid], tuple(columns.get(fid, ())), False)
                                  for fid in owned)
                definitions = {fid: self._definitions[fid] for fid in owned
                               if fid not in self._loaded[worker]}
                drops = tuple(sorted(self._drops[worker]))
                try:
                    # Every worker is idle in recv at this point; never enqueue to a busy
                    # worker whose result pipe may be full while the parent is sending.
                    channel.conn.send((operation, epoch, duals_blob, tasks, definitions,
                                       forbidden, deadline, drops))
                except (OSError, EOFError) as exc:
                    raise _PoolFailure("worker_lost", f"worker {worker}: {exc}") from exc
                self._loaded[worker].update(owned)
                self._drops[worker].clear()
            while unsettled:
                messages, dead = self._collect(deadline, unsettled)
                for worker, message in messages:
                    if len(message) < 3 or message[1:3] != (epoch, worker):
                        raise _PoolFailure("protocol_error", "stale or misrouted worker message")
                    tag = message[0]
                    if tag == "settled":
                        if (len(message) != 3 or worker not in unsettled
                                or any(self._worker_of[fid] == worker for fid in pending)):
                            raise _PoolFailure("protocol_error", "premature worker settlement")
                        unsettled.remove(worker)
                        continue
                    if tag == "error":
                        failure, detail = "worker_error", str(message[-1])
                        continue
                    if tag == "deadline":
                        self._rss[worker] = max(self._rss.get(worker, 0), int(message[3]))
                        failure, detail = "deadline", "worker reached the pricing deadline"
                        continue
                    if tag != "result" or len(message) != 8:
                        raise _PoolFailure("protocol_error", "invalid pricing result message")
                    _, _, _, fid, result, task_s, rss, released = message
                    if any(self._worker_of.get(victim) != worker for victim in released):
                        raise _PoolFailure("protocol_error", "worker released a graph it does not own")
                    # Released graphs get their definitions resent on their next sweep.
                    self._loaded[worker].difference_update(released)
                    self.graphs_released += len(released)
                    expected_type = ZetaSeedResult if operation == "seed" else ZetaPricingResult
                    if (fid not in pending or self._worker_of[fid] != worker
                            or not isinstance(result, expected_type)):
                        raise _PoolFailure("protocol_error", "duplicate or misrouted flight result")
                    if isinstance(result, ZetaSeedResult) and (
                        any(column.flight_id != fid for column in result.columns)
                        or (result.provided_primary is not None
                            and result.provided_primary not in result.columns)
                    ):
                        raise _PoolFailure("protocol_error", "misrouted seed column")
                    pending.remove(fid)
                    results[fid] = result
                    if operation == "sweep" or not self._priced:
                        self._cost[fid] = float(task_s)
                    records[fid] = {
                        "flight_id": fid, "worker": worker,
                        "pid": self._channels[worker].pid, "task_s": float(task_s),
                        "complete": result.complete,
                        "reason": result.reason,
                        "graphs_released": len(released),
                        "worker_peak_rss_bytes": int(rss),
                    }
                    if isinstance(result, ZetaSeedResult):
                        records[fid]["columns"] = len(result.columns)
                    else:
                        records[fid].update({
                            name: getattr(result, name) for name in (
                                "expanded", "generated", "history_pruned", "history_expired_rows",
                                "history_comparisons", "bound_build_s", "bound_table_states",
                                "bound_lookups", "bound_pruned",
                            )
                        })
                    self._rss[worker] = max(self._rss.get(worker, 0), int(rss))
                    if result.reason == "deadline":
                        failure, detail = "deadline", "pricing search reached its deadline"
                if dead and failure is None:
                    failure, detail = "worker_lost", f"workers exited: {sorted(dead)}"
                if failure is not None:
                    raise _PoolFailure(failure, detail or failure)
                if unsettled and deadline is not None and time.perf_counter() >= deadline:
                    raise _PoolFailure("deadline", "deadline expired waiting for pricing or cleanup")
            self._busy = False
            self._priced = self._priced or operation == "sweep"
        except _PoolFailure as exc:
            failure, detail = exc.code, str(exc)
            self._poisoned = f"{failure}: {detail}"
            self.close()
        except BaseException as exc:
            self._poisoned = f"sweep raised {type(exc).__name__}: {exc}"
            self.close()
            raise
        ordered = tuple(sorted(results))
        result_type = ZetaSeedSweepResult if operation == "seed" else ZetaSweepResult
        return result_type(
            tuple((fid, results[fid]) for fid in ordered),
            len(results) == len(self._flight_ids), failure, setup_s,
            time.perf_counter() - started,
            math.fsum(records[fid]["task_s"] for fid in ordered),
            tuple(records[fid] for fid in ordered), dict(self._rss), detail,
        )

    def close(self) -> None:
        """Stop idle workers, then kill and reap within two shared grace windows."""
        started = time.perf_counter()
        graceful = self._poisoned is None and not self._busy
        self._poisoned = self._poisoned or "closed"
        channels = self._channels or ()
        self._channels = None
        # A failed/interrupted send may have filled a worker's input pipe. Only a fully
        # returned healthy sweep proves that stop can be sent without blocking teardown.
        if graceful:
            for channel in channels:
                try:
                    channel.conn.send(("stop",))
                except (OSError, ValueError):
                    pass
            graceful_until = time.monotonic() + _GRACEFUL_STOP_S
            for channel in channels:
                try:
                    if channel.proc.pid is not None:
                        channel.proc.join(max(0.0, graceful_until - time.monotonic()))
                except Exception:
                    pass
        for channel in channels:
            try:
                if channel.proc.pid is not None and channel.proc.is_alive():
                    channel.proc.kill()
            except Exception:
                pass
        kill_until = time.monotonic() + _KILL_JOIN_S
        for channel in channels:
            try:
                if channel.proc.pid is not None:
                    channel.proc.join(max(0.0, kill_until - time.monotonic()))
            except Exception:
                pass
            finally:
                channel.conn.close()
        self._close_s += time.perf_counter() - started

    def __enter__(self) -> ZetaPricingPool:
        """Return this lazy pool; the first sweep owns startup timing."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """Close every worker while preserving any exception from the caller."""
        self.close()
        return False
