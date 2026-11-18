"""Persistent process workers for Zeta initial columns and frozen-dual pricing.

Each flight has one stable owner, so its seed geometry and canonical-column caches survive
successive pricing sweeps. Results are restored to flight order before master insertion.
The transport follows the production DP pool's raw spawn/pipe design: no queue-handler
threads, no worker respawn, explicit sweep epochs, and bounded process teardown.
"""

from __future__ import annotations

import itertools
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
            replace(graph.options, connection_mode="full", cheap_connection_mode=None,
                    traversal="best_first",
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
    conn: Connection, worker: int, flight_ids: tuple[int, ...], params: ColGenParams,
    native_artifact: tuple[str, str] | None = None,
) -> None:
    """Keep owned graphs alive and stream each flight result to the parent.

    Parameters
    ------------
    - conn (Connection): child endpoint of one duplex pipe.
    - worker (int): stable owner index reported with every message.
    - flight_ids (tuple[int, ...]): only flight IDs this worker may price.
    - params (ColGenParams): immutable objective and search controls.
    - native_artifact (tuple | None): parent-prepared compiled loop; workers never build it.

    Return
    --------
    - output (None): exits on stop, pipe closure, or a reported worker exception.
    """
    epoch: tuple[str, int] | None = None
    try:
        # Retain the optional controller for the lifetime of this process.
        _thread_limits = _limit_worker_threads()
        if native_artifact is not None:
            from ..zeta.native import install_compiled_loop
            install_compiled_loop(native_artifact)
        graphs: dict[int, ZetaGraph] = {}
        conn.send(("ready", worker, os.getpid(), _peak_rss_bytes()))
        previous_epoch: tuple[str, int] | None = None
        while True:
            message = conn.recv()
            if message == ("stop",):
                return
            tag, epoch, duals_blob, tasks, definitions, forbidden, deadline = message
            if tag not in ("sweep", "seed") or tuple(task[0] for task in tasks) != flight_ids:
                raise RuntimeError("worker received an invalid sweep assignment")
            if previous_epoch is not None and (
                epoch[0] != previous_epoch[0] or epoch[1] <= previous_epoch[1]
            ):
                raise RuntimeError("worker received a stale sweep epoch")
            previous_epoch = epoch
            duals = pickle.loads(duals_blob) if tag == "sweep" else {}
            for flight_id, flight_dual, known in tasks:
                started = time.perf_counter()
                if deadline is not None and started >= deadline:
                    conn.send(("deadline", epoch, worker, _peak_rss_bytes()))
                    return
                if flight_id not in graphs:
                    graphs[flight_id] = ZetaGraph(*definitions[flight_id])
                if tag == "seed":
                    result = build_seed_columns(
                        graphs[flight_id], params, known, forbidden, deadline,
                    )
                else:
                    result = price_zeta(
                        graphs[flight_id], duals, flight_dual, params,
                        known_columns=known, forbidden_rows=forbidden, deadline=deadline,
                    )
                task_s = time.perf_counter() - started
                conn.send(("result", epoch, worker, flight_id, result,
                           task_s, _peak_rss_bytes()))
                if result.reason == "deadline":
                    return
            # The final result is the handshake that permits the next sweep's send.
            # No extra completion message may remain unread between sweep epochs.
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
    """Solve-scoped static ownership of persistent Zeta pricing processes.

    Attributes
    ------------
    - worker_of (dict[int, int]): a copy of the stable flight-to-worker assignment.
    - n_workers (int): requested worker count capped to the number of flights.
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
        self._loaded: set[int] = set()
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
            for worker, owned in enumerate(self._owned):
                if deadline is not None and time.perf_counter() >= deadline:
                    raise _PoolFailure("deadline", "deadline expired during worker startup")
                parent_conn, child_conn = ctx.Pipe()
                worker_args = (child_conn, worker, owned, self._params)
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
            duals_blob = pickle.dumps(dict(duals), protocol=pickle.HIGHEST_PROTOCOL)
            forbidden = frozenset(forbidden_rows)
            epoch = (self._uid, next(self._epochs))
            pending = set(self._flight_ids)
            self._busy = True
            for channel in self._channels or ():
                if deadline is not None and time.perf_counter() >= deadline:
                    raise _PoolFailure("deadline", "deadline expired dispatching pricing tasks")
                owned = self._owned[channel.index]
                tasks = tuple((fid, flight_duals[fid], tuple(columns.get(fid, ())))
                              for fid in owned)
                definitions = {fid: self._definitions[fid] for fid in owned
                               if fid not in self._loaded}
                try:
                    # Every worker is idle in recv at this point; never enqueue to a busy
                    # worker whose result pipe may be full while the parent is sending.
                    channel.conn.send((operation, epoch, duals_blob, tasks, definitions,
                                       forbidden, deadline))
                except (OSError, EOFError) as exc:
                    raise _PoolFailure("worker_lost", f"worker {channel.index}: {exc}") from exc
                self._loaded.update(owned)
            while pending:
                active = {self._worker_of[fid] for fid in pending}
                messages, dead = self._collect(deadline, active)
                for worker, message in messages:
                    if len(message) < 3 or message[1:3] != (epoch, worker):
                        raise _PoolFailure("protocol_error", "stale or misrouted worker message")
                    tag = message[0]
                    if tag == "error":
                        failure, detail = "worker_error", str(message[-1])
                        continue
                    if tag == "deadline":
                        self._rss[worker] = max(self._rss.get(worker, 0), int(message[3]))
                        failure, detail = "deadline", "worker reached the pricing deadline"
                        continue
                    if tag != "result" or len(message) != 7:
                        raise _PoolFailure("protocol_error", "invalid pricing result message")
                    _, _, _, fid, result, task_s, rss = message
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
                    records[fid] = {
                        "flight_id": fid, "worker": worker,
                        "pid": self._channels[worker].pid, "task_s": float(task_s),
                        "complete": result.complete,
                        "reason": result.reason,
                    }
                    if isinstance(result, ZetaSeedResult):
                        records[fid]["columns"] = len(result.columns)
                    else:
                        records[fid].update({
                            name: getattr(result, name) for name in (
                                "expanded", "history_pruned", "history_expired_rows",
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
                if pending and deadline is not None and time.perf_counter() >= deadline:
                    raise _PoolFailure("deadline", "deadline expired waiting for pricing results")
            self._busy = False
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
