"""Bounded device-owned search states with resumable CPU verification checkpoints."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING

import numpy as np

from .gpu_buffers import DeviceBuffer, Dispatch

if TYPE_CHECKING:
    from .gpu_pricing import _Batch

_MAX_BYTES = 128 * 1024 * 1024
_QUEUE_CAPACITY = 65_536
_LAYER_BLOCK = 2
_CHUNK_BLOCK = 8


class ResidentSearch:
    """Own full predecessor records, ready queues and pending numerical decisions on device.

    Only validated incumbents set cutoffs. A queued deletion owns its complete
    record until FP64 verification, and rejected proposals rejoin their original
    absolute time. Ordinary histories are compacted but never merged.
    """

    def __init__(self, batch: _Batch) -> None:
        """Allocate bounded resident storage and borrow immutable batch/snapshot inputs.

        Parameters
        ------------
        - batch (_Batch): prepared certified search, after the independent incumbent search.

        Return
        --------
        - output (None): initialized device state; no root or alternative has been removed.
        """
        self.batch, self.runtime = batch, batch.runtime
        self.capacity, self.stride = batch.search_label_limit, batch.stride
        self.queue_capacity = min(_QUEUE_CAPACITY, max(6, self.capacity))
        self.chunk = min(batch.options.chunk_size, max(1, self.queue_capacity // 6))
        self.bytes = 0
        self.arena = self.empty((self.capacity, self.stride), np.int32)
        self.values = self.empty((self.capacity, 2), np.float32)
        self.labels = self.empty((self.chunk, self.stride), np.int32)
        self.label_values = self.empty((self.chunk, 2), np.float32)
        self.candidates = self.empty((6 * self.chunk, self.stride), np.int32)
        self.scores = self.empty((6 * self.chunk, 3), np.float32)
        self.event_records = self.empty((self.queue_capacity, self.stride), np.int32)
        self.event_values = self.empty((self.queue_capacity, 2), np.float32)
        self.event_meta = self.empty((self.queue_capacity, 5), np.int32)
        self.event_upper = self.empty((self.queue_capacity,), np.float32)
        self.counts = self.empty((7,), np.int32)
        self.cutoffs = self.empty((len(batch.problems), 2), np.float32)
        self.layer_buffers = [(self.empty((len(batch.cells),), np.float32),
                               self.empty((len(batch.cells),), np.int32),
                               self.empty((len(batch.cells), 2), np.uint32)) for _ in range(2)]
        self.known = np.zeros(self.capacity, bool)
        self.ready: dict[int, tuple[DeviceBuffer, int]] = {}
        self.free: list[DeviceBuffer] = []
        self.pending: list[Dispatch] = []
        self.graph = self.runtime.immutable(batch.graph, "graph")
        self.roots = self.runtime.immutable(batch.roots, "roots")
        self.paid_rows = self.runtime.immutable(batch.paid_rows, "paid_rows")
        self.paid_values = self.runtime.immutable(batch.paid_values, "paid_values")
        self.price_map = self.runtime.immutable(batch.price_map, "price_map")
        self.root_values = self.runtime.upload(batch.root_values)
        # Reuse masks discovered by roots and the preliminary pass. Missing masks
        # keep the weaker bound; never read full resident frontiers just to build caches.
        batch.pack_endpoints()
        self.endpoint_meta = self.runtime.immutable(batch.endpoint_meta, "endpoint_meta")
        self.endpoint_entries = self.runtime.immutable(batch.endpoint_entries, "endpoint_entries")
        self.endpoint_trees = self.runtime.immutable(batch.endpoint_trees, "endpoint_trees")
        self.empty_indices = self.runtime.upload(np.empty(0, np.int32))
        for name in ("resident_blocks", "resident_layers", "resident_multi_layer_blocks",
                     "resident_queue_peak", "resident_prefix_reads", "restored_prunes"):
            batch.stats[name] = batch.stats.get(name, 0)

    def stop(self, reason: str) -> None:
        """Report explicit incompletion rather than losing work on capacity or ownership errors."""
        from .gpu_pricing import _Stopped

        raise _Stopped(reason)

    def empty(self, shape: tuple[int, ...], dtype: np.dtype) -> DeviceBuffer:
        """Enforce the resident-storage byte budget before any device allocation."""
        size = max(4, int(np.prod(shape)) * np.dtype(dtype).itemsize)
        if self.bytes + size > _MAX_BYTES:
            self.stop("buffer_limit")
        result = self.runtime.empty(shape, dtype)
        self.bytes += size
        self.batch.stats["resident_bytes"] = max(self.bytes,
                                                 self.batch.stats.get("resident_bytes", 0))
        return result

    def control(self, values: tuple[int, ...]) -> DeviceBuffer:
        """Upload a small immutable command description retained by its dispatch."""
        return self.runtime.upload(np.asarray(values, np.int32))

    def frontier(self, step: int) -> tuple[DeviceBuffer, int]:
        """Take a ready time queue, reusing bounded scratch only after its preceding work ends."""
        found = self.ready.pop(step, None)
        if found is not None:
            return found
        buffer = self.free.pop() if self.free else self.empty((self.capacity,), np.int32)
        return buffer, 0

    def copy(
        self, records: DeviceBuffer, values: DeviceBuffer, indices: DeviceBuffer,
        count: int, begin: int, frontier: DeviceBuffer, offset: int, indexed: bool,
    ) -> Dispatch:
        """Schedule roots or restored proposals into stable IDs and one ready time queue.

        Parameters
        ------------
        - records (DeviceBuffer): complete source records, including opaque extension words.
        - values (DeviceBuffer): aligned FP32 score/positive pairs.
        - indices (DeviceBuffer): selected source indices when indexed is true.
        - count (int): records to retain; capacity is checked before scheduling.
        - begin (int): first reserved stable predecessor ID.
        - frontier (DeviceBuffer): destination ready queue.
        - offset (int): first vacant ready slot.
        - indexed (bool): select sparse source records rather than a contiguous prefix.

        Return
        --------
        - output (Dispatch): ordered copy which must complete before the new states expand.
        """
        if begin + count > self.capacity or offset + count > self.capacity:
            self.stop("label_limit")
        return Dispatch("resident_copy", count, (
            self.control((count, self.stride, begin, offset, int(indexed))), records, values,
            indices, self.arena, self.values, frontier,
        ))

    def ensure_prefix(self, parent: int) -> None:
        """Recover only ancestors needed for current FP64 verification or canonical completion.

        Parameters
        ------------
        - parent (int): stable resident predecessor ID, or -1 for an empty prefix.

        Return
        --------
        - output (None): original-price prefix sums and CPU predecessor mirror are available.
        """
        pending = []
        if parent < -1 or parent >= self.batch.used:
            self.stop("invalid_predecessor")
        while parent >= 0 and not self.known[parent]:
            self.batch.check()
            record = self.runtime.read_rows(self.arena, parent, 1)[0]
            previous = int(record[3])
            if previous >= parent or previous < -1:
                self.stop("invalid_predecessor")
            pending.append((parent, record))
            parent = previous
        for index, record in reversed(pending):
            self.batch.store_prefix(index, record)
            self.known[index] = True
        self.batch.stats["resident_prefix_reads"] += len(pending)

    def checkpoint(self, counts: DeviceBuffer, step: int, output: DeviceBuffer) -> None:
        """Resolve compact decision queues and preserve rejected proposals at their own time.

        Parameters
        ------------
        - counts (DeviceBuffer): completed block counters and explicit overflow flag.
        - step (int): absolute time belonging to the final continuing frontier.
        - output (DeviceBuffer): final frontier, including any earlier queued states at that time.

        Return
        --------
        - output (None): all decisions resolved; restored records are scheduled before reuse.
        """
        b = self.batch
        used, _, continuing, events, overflow, candidates, direct = map(int, self.runtime.read(counts))
        b.used = min(used, self.capacity)
        b.stats["candidates"] += candidates
        b.stats["gpu_direct_prunes"] += direct
        b.stats["gpu_prune_proposals"] += direct
        b.stats["resident_blocks"] += 1
        b.stats["resident_queue_peak"] = max(b.stats["resident_queue_peak"], events)
        if overflow:
            self.stop("queue_limit" if overflow == 2 else "label_limit")
        if continuing:
            self.ready[step] = output, continuing
        else:
            self.free.append(output)
        restored: dict[int, list[int]] = defaultdict(list)
        metadata = self.runtime.read_rows(self.event_meta, 0, events)
        uppers = (self.runtime.read_rows(self.event_upper, 0, events)
                  if b.options.gpu_prune_audit else None)
        for index, (cell, root, hops, parent, flags) in enumerate(metadata):
            b.check()
            root, parent, cell, hops, flags = map(int, (root, parent, cell, hops, flags))
            self.ensure_prefix(parent)
            if flags & 4:
                b.stats["gpu_prune_proposals"] += 1
                b.stats["gpu_endpoint_prune_proposals"] += int(bool(flags & 8))
                if uppers is not None:
                    b.audit_upper = float(uppers[index])
                    b.audit_direct = bool(flags & 32)
                if flags & 128:
                    b.audit_integer_record(self.runtime.read_rows(self.event_records, index, 1)[0])
                if not b.retain(root, parent, cell, hops):
                    continue
            record = np.zeros(self.stride, np.int32)
            record[:4] = cell, root, hops, parent
            record[6] = flags
            if flags & 1:
                b.complete(record, b.path(parent, cell))
            if flags & 4 and flags & 2:
                flight, variant = b.root_flights[root], b.root_variants[root]
                time = int(b.variants[flight].start_step[variant]) + hops
                restored[time].append(index)
        for time, indices in restored.items():
            frontier, offset = self.frontier(time)
            self.pending.append(self.copy(
                self.event_records, self.event_values,
                self.runtime.upload(np.asarray(indices, np.int32)), len(indices), b.used,
                frontier, offset, True,
            ))
            b.used += len(indices)
            b.stats["restored_prunes"] += len(indices)
            self.ready[time] = frontier, offset + len(indices)
        b.stats["max_frontier"] = max(b.stats["max_frontier"],
                                       max((n for _, n in self.ready.values()), default=0))

    def expand(
        self, calls: list[Dispatch], step: int, frontier: DeviceBuffer, output: DeviceBuffer,
        counts: DeviceBuffer, start: int, count: int, cutoffs: DeviceBuffer,
        future: DeviceBuffer, blocked: DeviceBuffer, visit_masks: DeviceBuffer,
    ) -> None:
        """Append dependent load, parallel expansion and device compaction kernels.

        Parameters
        ------------
        - calls (list[Dispatch]): ordered command block under construction.
        - step (int): absolute source time.
        - frontier (DeviceBuffer): ready state IDs, possibly with a device-computed count.
        - output (DeviceBuffer): next-time state IDs.
        - counts (DeviceBuffer): shared block counters, including actual frontier size.
        - start (int): first ready slot in this chunk.
        - count (int): bounded upper limit; padded inactive slots are explicitly invalidated.
        - cutoffs (DeviceBuffer): only canonically validated per-flight pruning cutoffs.
        - future (DeviceBuffer): flight-specific future bounds at the target time.
        - blocked (DeviceBuffer): current flight-specific forbidden-row flags.
        - visit_masks (DeviceBuffer): exact paid-arrival bits for the target time.

        Return
        --------
        - output (None): adds kernels without a host wait or frontier readback.
        """
        prices = self.batch.device_prices
        numeric = (self.batch.numeric.arguments(step + 1)
                   if self.batch.options.gpu_integer_bounds else ())
        calls.extend((
            Dispatch("resident_load", count, (
                self.control((count, self.stride, start)), self.arena, self.values,
                frontier, counts, self.labels, self.label_values,
            )),
            Dispatch("expand", count * 6, (
                self.control((count, self.stride, step, *self.batch.offsets,
                              self.batch.prune_mode)),
                self.labels, self.label_values, self.graph, self.roots, self.root_values,
                future, blocked, self.paid_rows, self.paid_values, self.candidates, self.scores,
                self.price_map, prices.ranges, prices.steps, prices.values, cutoffs,
                visit_masks, self.endpoint_meta, self.endpoint_entries, self.endpoint_trees,
                *numeric,
            )),
            Dispatch("resident_compact", count * 6, (
                self.control((count * 6, self.stride, self.capacity, self.queue_capacity)),
                self.candidates, self.scores, self.arena, self.values, output, counts,
                self.event_records, self.event_values, self.event_meta, self.event_upper,
            )),
        ))

    def layer_inputs(self, step: int, slot: int = 0) -> tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer]:
        """Upload only flight-specific bounds/exclusions; prices stay in the shared store."""
        visits, blocked = self.batch.visits(step + 1)
        future_buffer, blocked_buffer, masks_buffer = self.layer_buffers[slot]
        # The unrestricted future table relaxes every resident parent's hop budget.
        # Exact remaining-hop lookup remains in FP64 verification and the preliminary pass.
        self.runtime.write(future_buffer, np.ascontiguousarray(visits[:, 2]))
        self.runtime.write(blocked_buffer, blocked)
        self.runtime.write(masks_buffer, self.batch.endpoint_visits(step + 1))
        return future_buffer, blocked_buffer, masks_buffer

    def run(self, first_step: int, first_records: np.ndarray, first_values: np.ndarray) -> None:
        """Search all roots and resumable alternatives in bounded ordered layer blocks.

        Parameters
        ------------
        - first_step (int): earliest time with roots surviving FP64 verification.
        - first_records (np.ndarray): these roots, already allocated in the CPU predecessor mirror.
        - first_values (np.ndarray): aligned initial FP32 score/positive pairs.

        Return
        --------
        - output (None): complete exhaustion; limits or unresolved device failures raise.
        """
        b = self.batch
        roots = deque(step for step in sorted(b.start_buckets) if step >= first_step)
        last_step = max(t.max_step for t in b.topologies)
        while roots or self.ready:
            b.check()
            b._visit_positive.clear()
            step = min(roots[0] if roots else last_step + 1, min(self.ready, default=last_step + 1))
            if step > last_step:
                self.stop("unresolved_time")
            frontier, count = self.frontier(step)
            if roots and roots[0] == step:
                roots.popleft()
                if step == first_step:
                    records, values = first_records, first_values
                    begin = b.used - len(records)
                else:
                    begin = b.used
                    records, values = b.roots_at(step)
                self.known[begin:b.used] = True
                if len(records):
                    records[:, 3] = -1
                    self.pending.append(self.copy(
                        self.runtime.upload(records), self.runtime.upload(values), self.empty_indices,
                        len(records), begin, frontier, count, False,
                    ))
                    count += len(records)
            if not count:
                self.free.append(frontier)
                continue
            b.stats["max_frontier"] = max(b.stats["max_frontier"], count)
            next_existing = self.ready.get(step + 1, (None, 0))[1]
            upper_next = 6 * count + next_existing
            two_layers = (_LAYER_BLOCK >= 2 and step < last_step and
                          (not roots or roots[0] > step + 1) and upper_next <= self.chunk and
                          6 * (count + upper_next) <= self.queue_capacity and
                          b.used + 6 * (count + upper_next) <= self.capacity)
            future, blocked, visit_masks = self.layer_inputs(step)
            position = 0
            while position < count:
                b.check()
                calls, self.pending = self.pending, []
                output, existing = self.frontier(step + 1)
                counts, cutoffs = self.counts, self.cutoffs
                self.runtime.write(counts, np.asarray((b.used, count, existing, 0, 0, 0, 0), np.int32))
                self.runtime.write(cutoffs, b.cutoff_values())
                budget = self.queue_capacity
                initial_position = position
                chunks = 0
                while (position < count and chunks < _CHUNK_BLOCK
                       and budget >= 6 * min(self.chunk, count - position)):
                    n = min(self.chunk, count - position)
                    self.expand(calls, step, frontier, output, counts, position, n,
                                cutoffs, future, blocked, visit_masks)
                    position += n
                    budget -= 6 * n
                    chunks += 1
                if position == initial_position:
                    self.stop("queue_limit")
                end_step = step + 1
                consumed = []
                if two_layers:
                    following, existing = self.frontier(step + 2)
                    calls.append(Dispatch("resident_advance", 1, (self.control((existing,)), counts)))
                    following_future, following_blocked, following_masks = self.layer_inputs(step + 1, 1)
                    self.expand(calls, step + 1, output, following, counts, 0, upper_next,
                                cutoffs, following_future, following_blocked, following_masks)
                    consumed.append(output)
                    output = following
                    end_step += 1
                    b.stats["resident_multi_layer_blocks"] += 1
                self.runtime.run_many(calls)
                b.stats["resident_layers"] += 2 if two_layers else 1
                self.free.extend(consumed)
                self.checkpoint(counts, end_step, output)
            self.free.append(frontier)
        if self.pending:
            self.stop("unresolved_decisions")
