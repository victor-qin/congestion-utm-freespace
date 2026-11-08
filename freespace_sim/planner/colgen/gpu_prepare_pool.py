"""Bounded persistent CPU workers; every group drains before GPU execution.

Transport is framed and nonblocking in the owner. There is at most one request
or reply per worker, and no background refill while GPU search is running.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
import io
import math
import multiprocessing as mp
import os
import pickle
import resource
import selectors
import socket
import struct
import sys
import time
from typing import Any

import numpy as np

from . import pricing
from .gpu_prepared import (
    PreparedFlight, check_deadline, export_flight, prepare_flight, restore_flight,
)
from .pricing_platform import PricingOptions, PricingProblem
from .network import FlightGraph, _LazyForbiddenHops
from .windows import derive_cell_window


class GPUPreparationError(RuntimeError):
    """Preparation worker/protocol failure; never a request for CPU full pricing."""


class GPUPreparationLimit(GPUPreparationError):
    """A preparation message exceeded its explicit serialized byte budget."""


class GPUPreparationWorkerLost(GPUPreparationError):
    """A process exited or disconnected before its preparation result was received."""


class _LimitedBuffer(io.BytesIO):
    """Prevent pickle from allocating an unbounded serialized payload.

    Attributes
    ------------
    - limit (int): maximum serialized frame body accepted by the buffer.
    """

    def __init__(self, limit: int) -> None:
        """Set the maximum number of bytes accepted by the serializer."""
        super().__init__()
        self.limit = limit

    def write(self, data: bytes) -> int:
        """Reject an oversized write before growing the backing buffer."""
        if self.tell() + memoryview(data).nbytes > self.limit:
            raise GPUPreparationLimit(f"GPU preparation payload exceeds {self.limit} bytes")
        return super().write(data)


def _encode(message: Any, limit: int) -> bytes:
    """Serialize a bounded message with an unsigned length prefix."""
    output = _LimitedBuffer(limit)
    pickle.Pickler(output, protocol=pickle.HIGHEST_PROTOCOL).dump(message)
    data = output.getvalue()
    return struct.pack("!Q", len(data)) + data


def _receive(sock: socket.socket, size: int) -> bytes:
    """Read one complete worker-side frame or fail when the owner closes its socket."""
    data = bytearray()
    while len(data) < size:
        part = sock.recv(min(size - len(data), 1024 * 1024))
        if not part:
            raise EOFError("GPU preparation owner disconnected")
        data.extend(part)
    return bytes(data)


def retained_bytes(value: Any, measured: dict[int, int] | None = None) -> int:
    """Count retained Python containers and array storage once, excluding interpreter state.

    Parameters
    ------------
    - value (Any): owned cache graph; locks and external native objects count as shallow objects.
    - measured (dict[int, int] | None): conservative sizes of unchanged owned components.

    Return
    --------
    - output (int): accounted retained bytes, not a process RSS or transient allocation limit.
    """
    seen: set[int] = set()
    pending = [value]
    total = 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if measured is not None and id(item) in measured:
            total += measured[id(item)]
            continue
        total += sys.getsizeof(item)
        if isinstance(item, np.ndarray):
            if item.base is not None:
                pending.append(item.base)
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset, deque)):
            pending.extend(item)
        elif not isinstance(item, (type, str, bytes, bytearray, memoryview)):
            if hasattr(item, "__dict__"):
                pending.append(vars(item))
            for slot in getattr(type(item), "__slots__", ()):
                if slot not in ("__dict__", "__weakref__") and hasattr(item, slot):
                    pending.append(getattr(item, slot))
    return total


class _GraphSize:
    """Reuse sizes of immutable or append-only graph components, never weighted answers.

    Attributes
    ------------
    - parts (dict): bounded component names mapped to owner, change token and measured bytes.
      Strong identities prevent id reuse; replaced components are released on every measurement.
    """

    def __init__(self) -> None:
        """Start without any retained component references."""
        self.parts: dict[str, tuple[Any, Any, int]] = {}

    def measure(self, graph: FlightGraph) -> int:
        """Recount changed components and the remaining mutable cache storage.

        Parameters
        ------------
        - graph (FlightGraph): private worker graph after synchronous CPU preparation.

        Return
        --------
        - output (int): conservative accounted bytes including this small size registry.
          Components are counted independently, so shared objects can be overcounted.
        """
        cache = graph._search_cache
        parts = {name: (getattr(cache, name), 0)
                 for name in ("prepared", "gpu_geometry", "gpu_endpoints")}
        parts["endpoint_claims"] = cache.endpoint_claims, None
        packing = cache.gpu_packing
        if packing is not None:
            for name in ("cells", "origin_data", "graph"):
                parts["packing_" + name] = getattr(packing, name), 0
            # Membership admission only appends immutable keys/sets. Starts can
            # replace values and are always recounted.
            parts["claims"] = packing.claims, len(packing.claims)
        hops = graph.forbidden_hops
        if type(hops) is _LazyForbiddenHops:
            for name in ("_allowed", "_roles", "_blocked"):
                value = getattr(hops, name)
                parts[name] = value, len(value)
        current = {}
        for name, (value, count) in parts.items():
            if value is None:
                continue
            previous = self.parts.get(name)
            unchanged = previous is not None and previous[0] is value
            if name == "endpoint_claims":
                # A shallow identity snapshot detects LRU replacement even at the
                # same entry count. Its strong references also prevent id reuse.
                unchanged = (unchanged and len(previous[1]) == len(value)
                             and all(previous[1].get(k) is v for k, v in value.items()))
                count = previous[1] if unchanged else dict(value)
            else:
                unchanged = unchanged and previous[1] == count
            size = previous[2] if unchanged else retained_bytes(value)
            current[name] = value, count, size
        self.parts = current
        measured = {id(value): size for value, _, size in current.values()}
        bookkeeping = sys.getsizeof(self) + sys.getsizeof(vars(self)) + sys.getsizeof(current)
        bookkeeping += sum(sys.getsizeof(key) + sys.getsizeof(entry)
                           + sys.getsizeof(entry[1]) + sys.getsizeof(entry[2])
                           for key, entry in current.items())
        return retained_bytes(graph, measured) + bookkeeping


def _worker_main(sock: socket.socket, options: PricingOptions) -> None:
    """Keep bounded graph caches and one price snapshot without importing a GPU runtime.

    Parameters
    ------------
    - sock (socket.socket): private owner connection; one request/reply at a time.
    - options (PricingOptions): preparation, cache and message budgets.

    Return
    --------
    - output (None): exits on disconnect, shutdown, or process cancellation.
    """
    graphs: OrderedDict[int, tuple[Any, int, _GraphSize]] = OrderedDict()
    snapshot, view = None, None
    limit = options.prepare_max_bytes
    try:
        sock.sendall(_encode(("ready",), limit))
        while True:
            size, = struct.unpack("!Q", _receive(sock, 8))
            if size > limit:
                raise GPUPreparationLimit("Oversized GPU preparation request")
            started = time.monotonic()
            message = pickle.loads(_receive(sock, size))
            decode_s = time.monotonic() - started
            if message is None:
                return
            key, version, graph, cfg, params, snap, prices, pi, known, forbidden, improving, \
                deadline, include_topology = message
            try:
                if graph is None:
                    graph, _, accounting = graphs.pop(version)
                else:
                    graphs.pop(version, None)
                    accounting = _GraphSize()
                snapshot_s = 0.0
                if prices is not None:
                    measured = time.monotonic()
                    snapshot, view = snap, pricing.DualView(pickle.loads(prices[8:]), cfg)
                    snapshot_s = time.monotonic() - measured
                if snapshot != snap or view.offsets != derive_cell_window(cfg):
                    raise GPUPreparationError("Stale GPU preparation price snapshot")
                problem = PricingProblem(graph, cfg, params, view, pi, known, forbidden, improving)
                prepared = prepare_flight(problem, view, options, deadline)
                result = export_flight(prepared, include_topology=include_topology)
                measured = time.monotonic()
                size = accounting.measure(graph)
                graphs[version] = graph, size, accounting
                cache_bytes = sum(entry[1] for entry in graphs.values())
                evictions = 0
                while graphs and (cache_bytes > options.prepare_cache_bytes or len(graphs) > 128):
                    # Keep only the byte count: an unpacking temporary would retain
                    # the evicted graph/accounting owner while this worker is idle.
                    removed = graphs.popitem(last=False)[1][1]
                    cache_bytes -= removed
                    evictions += 1
                result.stats.update(
                    cache_account_s=time.monotonic() - measured, worker_receive_decode_s=decode_s,
                    worker_snapshot_s=snapshot_s,
                    worker_pid=os.getpid(),
                    worker_cache_bytes=cache_bytes, worker_cache_evictions=evictions,
                    worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                )
                measured = time.monotonic()
                data = _encode(result, limit)
                serialize_s = time.monotonic() - measured
                payload = _encode(("result", key, data, tuple(graphs), serialize_s), limit)
            except Exception as exc:
                payload = _encode(("error", key, type(exc).__name__, str(exc)[:1000]), limit)
            sock.sendall(payload)
            # Idle workers retain only admitted graph caches and their price snapshot,
            # not the last flight's candidate, complete payload, or decode buffers.
            problem = prepared = result = data = graph = known = prices = forbidden = None
            message = payload = None
            accounting = None
    except (EOFError, OSError):
        pass
    finally:
        sock.close()


@dataclass(slots=True)
class _Job:
    """One version-tagged task, with result storage owned by the synchronous caller.

    Attributes
    ------------
    - key (tuple): invocation, position, graph version and snapshot version.
    - problem (PricingProblem): normalized immutable task inputs.
    - result (PreparedFlight | None): set only after exact tag validation.
    """

    key: tuple[int, int, int, int]
    problem: PricingProblem
    result: PreparedFlight | None = None


@dataclass(slots=True)
class _Channel:
    """One spawned process and bounded transport buffers.

    Attributes
    ------------
    - process, socket: owned process/connection handles.
    - outgoing, incoming: at most one bounded frame each.
    - job (_Job | None): sole in-flight request; None means idle.
    - snapshot, graphs: versions acknowledged by this worker.
    """

    process: Any
    socket: socket.socket
    outgoing: memoryview = field(default_factory=lambda: memoryview(b""))
    incoming: bytearray = field(default_factory=bytearray)
    ready: bool = False
    job: _Job | None = None
    snapshot: int | None = None
    graphs: set[int] = field(default_factory=set)


class GPUPreparationPool:
    """Persistent CPU preparation with a complete barrier before each GPU execution group.

    Attributes
    ------------
    - options (PricingOptions): explicit worker, group, payload and cache bounds.
    - channels (list): at most prepare_workers processes, one job each.
    - stats (dict): cumulative transport/startup/restore time and high-water counts.
    """

    def __init__(self, options: PricingOptions) -> None:
        """Store explicit limits; worker creation is deferred to the first preparation call.

        Parameters
        ------------
        - options (PricingOptions): positive worker count and bounded preparation resources.

        Return
        --------
        - output (None): initializes empty owner state without spawning workers.
        """
        if options.prepare_workers < 1:
            raise ValueError("A preparation pool requires at least one worker")
        self.options = options
        self.channels: list[_Channel] = []
        self._graphs: OrderedDict[int, tuple[Any, int, int]] = OrderedDict()
        self._fixed: OrderedDict[int, tuple[Any, int]] = OrderedDict()
        self._snapshots: OrderedDict[int, tuple[Any, int]] = OrderedDict()
        self._encoded_snapshot: tuple[int, bytes] | None = None
        self._epoch = self._version = self._snapshot = self._assignment = 0
        self._active = self.closed = False
        self._queues = [deque() for _ in range(options.prepare_workers)]
        self.stats = dict(request_bytes=0, reply_bytes=0, graph_sends=0, snapshot_sends=0,
                          startup_s=0.0, serialize_s=0.0, restore_s=0.0, group_wall_s=0.0,
                          peak_jobs=0, peak_inflight=0, owner_cache_bytes=0,
                          snapshot_encodes=0, snapshot_encode_s=0.0, peak_snapshot_bytes=0)

    def _start(self, deadline: float | None) -> None:
        """Spawn workers and wait for readiness under the caller's deadline.

        Parameters
        ------------
        - deadline (float | None): includes startup and readiness transfer.

        Return
        --------
        - output (None): all workers ready, or an explicit failure/timeout.
        """
        if self.channels:
            return
        started = time.monotonic()
        ctx = mp.get_context("spawn")
        for _ in range(self.options.prepare_workers):
            check_deadline(deadline)
            parent, child = socket.socketpair()
            process = ctx.Process(target=_worker_main, args=(child, self.options), daemon=True)
            try:
                process.start()
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            parent.setblocking(False)
            self.channels.append(_Channel(process, parent))
        while not all(ch.ready for ch in self.channels):
            self._pump(deadline)
        self.stats["startup_s"] += time.monotonic() - started

    def _schedule(self, deadline: float | None) -> None:
        """Schedule only idle workers; never queue a second request behind a blocked reply.

        Parameters
        ------------
        - deadline (float | None): checked before/after bounded serialization.

        Return
        --------
        - output (None): at most one frame per worker is pending.
        """
        for index, ch in enumerate(self.channels):
            if ch.job is not None or not self._queues[index]:
                continue
            check_deadline(deadline)
            job = self._queues[index].popleft()
            p, version, snap = job.problem, job.key[2], job.key[3]
            graph = None if version in ch.graphs else p.graph
            started = time.monotonic()
            prices = None
            if ch.snapshot != snap:
                if self._encoded_snapshot is None or self._encoded_snapshot[0] != snap:
                    prices = _encode(p.duals._duals, self.options.prepare_max_bytes)
                    self._encoded_snapshot = snap, prices
                    self.stats["snapshot_encodes"] += 1
                    self.stats["snapshot_encode_s"] += time.monotonic() - started
                    self.stats["peak_snapshot_bytes"] = max(
                        self.stats["peak_snapshot_bytes"], len(prices))
                prices = self._encoded_snapshot[1]
            message = (job.key, version, graph, p.cfg, p.params, snap,
                       prices,
                       p.pi_f, p.known_column, p.forbidden_rows, p.require_improving,
                       deadline, version not in self._fixed)
            ch.outgoing = memoryview(_encode(message, self.options.prepare_max_bytes))
            self.stats["serialize_s"] += time.monotonic() - started
            self.stats["request_bytes"] += len(ch.outgoing)
            self.stats["graph_sends"] += graph is not None
            self.stats["snapshot_sends"] += ch.snapshot != snap
            ch.job, ch.snapshot = job, snap
            check_deadline(deadline)
        self.stats["peak_inflight"] = max(self.stats["peak_inflight"], sum(
            ch.job is not None for ch in self.channels))

    def _accept(self, ch: _Channel, message: Any, deadline: float | None) -> None:
        """Accept only the exact invocation/domain/snapshot tag, then reconstruct owner data.

        Parameters
        ------------
        - ch (_Channel): worker with exactly one outstanding task.
        - message (Any): decoded bounded frame.
        - deadline (float | None): guards reconstruction and future lazy endpoint work.

        Return
        --------
        - output (None): matching result published; stale/error replies raise.
        """
        if message == ("ready",) and not ch.ready:
            ch.ready = True
            return
        job = ch.job
        if job is None or message[1] != job.key:
            raise GPUPreparationError("Discarded stale/mismatched GPU preparation reply")
        if message[0] == "error":
            error = GPUPreparationLimit if message[2] == "GPUPreparationLimit" else GPUPreparationError
            raise error(f"GPU preparation worker {message[2]}: {message[3]}")
        if message[0] != "result":
            raise GPUPreparationError("Unexpected GPU preparation reply")
        started = time.monotonic()
        data, version = pickle.loads(message[2][8:]), job.key[2]
        data.stats["worker_serialize_s"] = message[4]
        fixed = data.fixed
        if fixed is None:
            cached = self._fixed.get(version)
            fixed = None if cached is None else cached[0]
            if cached is not None:
                self._fixed.move_to_end(version)
        else:
            self._fixed[version] = fixed, retained_bytes(fixed)
        job.result = restore_flight(data, job.problem, job.problem.duals, fixed,
                                    self.options, lambda: check_deadline(deadline))
        ch.graphs = set(message[3])
        ch.job = None
        # Keep requested topology available until all current replies are restored.
        self.stats["restore_s"] += time.monotonic() - started

    def _pump(self, deadline: float | None) -> None:
        """Advance interruptible partial I/O and detect worker death.

        Parameters
        ------------
        - deadline (float | None): absolute deadline, checked around each transfer.

        Return
        --------
        - output (None): bounded buffers advance or a failure/timeout raises.
        """
        check_deadline(deadline)
        with selectors.DefaultSelector() as selector:
            for ch in self.channels:
                events = selectors.EVENT_READ | (selectors.EVENT_WRITE if ch.outgoing else 0)
                selector.register(ch.socket, events, (ch, False))
                selector.register(ch.process.sentinel, selectors.EVENT_READ, (ch, True))
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            ready = selector.select(timeout)
        for event, mask in ready:
            ch, dead = event.data
            if dead:
                raise GPUPreparationWorkerLost(f"GPU preparation worker {ch.process.pid} exited")
            try:
                if mask & selectors.EVENT_WRITE:
                    sent = ch.socket.send(ch.outgoing[:1024 * 1024])
                    ch.outgoing = (ch.outgoing[sent:] if sent < len(ch.outgoing)
                                   else memoryview(b""))
                if mask & selectors.EVENT_READ:
                    part = ch.socket.recv(min(1024 * 1024,
                                              self.options.prepare_max_bytes + 8 - len(ch.incoming)))
                    if not part:
                        raise GPUPreparationWorkerLost(f"GPU preparation worker {ch.process.pid} lost")
                    ch.incoming.extend(part)
                    if len(ch.incoming) >= 8:
                        size, = struct.unpack("!Q", ch.incoming[:8])
                        if size > self.options.prepare_max_bytes:
                            raise GPUPreparationLimit("Oversized GPU preparation reply")
                        if len(ch.incoming) >= size + 8:
                            payload = ch.incoming[8:size + 8]
                            del ch.incoming[:size + 8]
                            self.stats["reply_bytes"] += size + 8
                            self._accept(ch, pickle.loads(payload), deadline)
            except BlockingIOError:
                pass
            except OSError as exc:
                raise GPUPreparationWorkerLost("GPU preparation connection failed") from exc
        check_deadline(deadline)

    def prepare(
        self, problems: Sequence[PricingProblem], deadline: float | None = None,
    ) -> tuple[PreparedFlight, ...]:
        """Finish a bounded ordered group before returning; workers are then all idle.

        Parameters
        ------------
        - problems (Sequence[PricingProblem]): immutable inputs with normalized DualViews.
        - deadline (float | None): includes startup, transfer and reconstruction.

        Return
        --------
        - output (tuple[PreparedFlight, ...]): one matching result per input in caller order.
          Timeout/cancellation/failure reaps all workers and never certifies.
        """
        if self.closed or self._active:
            raise GPUPreparationError("GPU preparation pool is closed or already active")
        if deadline is not None and not math.isfinite(deadline):
            raise ValueError("deadline must be finite")
        if len(problems) > self.options.prepare_batch_size:
            raise GPUPreparationLimit("Preparation group exceeds its bounded size")
        if not problems:
            return ()
        self._active = True
        self._epoch += 1
        started = time.monotonic()
        jobs = []
        try:
            for position, p in enumerate(problems):
                check_deadline(deadline)
                if not isinstance(p.duals, pricing.DualView):
                    raise ValueError("Preparation requires immutable DualView snapshots")
                if p.duals.offsets != derive_cell_window(p.cfg):
                    raise ValueError("DualView visit offsets must match the problem configuration")
                saved = self._graphs.get(id(p.graph))
                if saved is None:
                    self._version += 1
                    worker = self._assignment % self.options.prepare_workers
                    self._assignment += 1
                    saved = p.graph, self._version, worker
                    self._graphs[id(p.graph)] = saved
                self._graphs.move_to_end(id(p.graph))
                snap = self._snapshots.get(id(p.duals))
                if snap is None:
                    self._snapshot += 1
                    snap = p.duals, self._snapshot
                    self._snapshots[id(p.duals)] = snap
                self._snapshots.move_to_end(id(p.duals))
                job = _Job((self._epoch, position, saved[1], snap[1]), p)
                self._queues[saved[2]].append(job)
                jobs.append(job)
            self.stats["peak_jobs"] = max(self.stats["peak_jobs"], len(jobs))
            self._start(deadline)
            self._schedule(deadline)
            while any(job.result is None for job in jobs) or any(ch.outgoing for ch in self.channels):
                self._pump(deadline)
                self._schedule(deadline)
            check_deadline(deadline)
            return tuple(job.result for job in jobs)
        except BaseException:
            self.close()
            raise
        finally:
            self._active = False
            # Input owners already retain graphs. These references are count-bounded;
            # new numeric topology storage has a separate accounted byte limit.
            while len(self._graphs) > 128 * self.options.prepare_workers:
                self._graphs.popitem(last=False)
            while len(self._snapshots) > 2:
                self._snapshots.popitem(last=False)
            cache_bytes = sum(item[1] for item in self._fixed.values())
            while self._fixed and cache_bytes > self.options.prepare_cache_bytes:
                _, (_, size) = self._fixed.popitem(last=False)
                cache_bytes -= size
            self.stats["owner_cache_bytes"] = cache_bytes
            self.stats["group_wall_s"] += time.monotonic() - started

    def close(self) -> None:
        """Cancel and reap workers without waiting on their output locks.

        Parameters
        ------------
        - self (GPUPreparationPool): owns all process and transport handles.

        Return
        --------
        - output (None): idempotent shutdown within shared bounded grace periods.
        """
        self.closed = True
        for ch in self.channels:
            ch.socket.close()
            if ch.process.is_alive():
                ch.process.terminate()
        until = time.monotonic() + 0.5
        for ch in self.channels:
            ch.process.join(max(0.0, until - time.monotonic()))
        for ch in self.channels:
            if ch.process.is_alive():
                ch.process.kill()
        until = time.monotonic() + 1.0
        for ch in self.channels:
            ch.process.join(max(0.0, until - time.monotonic()))
            ch.process.close()
        self.channels.clear()
        self._graphs.clear()
        self._fixed.clear()
        self._snapshots.clear()
        self._encoded_snapshot = None
        for queue in self._queues:
            queue.clear()
