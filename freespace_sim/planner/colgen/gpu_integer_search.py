"""Opt-in integer pruning with exact memberships and original canonical completions."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from fractions import Fraction
import math
import time
import weakref

import numpy as np

from .gpu_buffers import DeviceBuffer
from .gpu_integer import IntegerInterval, IntegerScale
from .gpu_integer_bounds import cost_operands, prepare_reference_operands
from .gpu_pricing import _Batch, _BOUND_CLOSURE_TOL, _Stopped

_NUMERIC_BYTES = 64 * 1024 * 1024
_STATE_WORDS = 15


class IntegerData:
    """Own immutable numeric operands for exactly one prepared pricing batch.

    Attributes
    ------------
    - batch (_Batch): immutable invocation inputs and current validated incumbents.
    - scales (list[IntegerScale]): explicit grids for each flight's price/objective snapshot.
    - roots (np.ndarray): original reference constants and both root prefix sums.
    - uppers (np.ndarray): GPU-computed root bound intervals, refreshed after future preparation.
    - layers (OrderedDict): at most two original-FP64 suffix rows retained on the device.
    """

    def __init__(self, batch: _Batch) -> None:
        """Prepare small root data before allocating any continuation-specific tables.

        Parameters
        ------------
        - batch (_Batch): fully prepared original-price graph/root domain.

        Return
        --------
        - output (None): prepares GPU root bounds; large tables remain lazy for closed domains.
        """
        self.batch, self.runtime = weakref.proxy(batch), batch.runtime
        self.scales = []
        for i, bounds in enumerate(batch.bounds):
            operands = (batch.benefits[i], batch.problems[i].pi_f, batch.negative[i],
                        bounds._absolute_prices, bounds._cost_scale,
                        float(batch.variants[i].ground_delay_s.max(initial=0.0)))
            self.scales.append(IntegerScale.for_values(
                (value for value in operands if math.isfinite(value)), max_terms=32,
            ))
        self.roots = np.zeros((len(batch.root_flights), 7, 5), np.uint32)
        self.root_empty = np.zeros(len(self.roots), bool)
        self.uppers = np.zeros((len(self.roots), 5), np.uint32)
        self.layers: OrderedDict[int, DeviceBuffer] = OrderedDict()
        self.buffers: dict[str, DeviceBuffer] = {}
        self.buffer_sizes: dict[str, int] = {}
        self.static_bytes = self.roots.nbytes + self.uppers.nbytes + self.root_empty.nbytes
        self.account()
        self.endpoint_generation = None
        self.cutoff_key = None
        self.live_roots: set[int] = set()
        self.cost_roots: set[int] = set()
        self.cost_arrays: list[np.ndarray] = []
        self.cost_size = 0
        self.built = False
        self.update_roots()

    def account(self, extra: int = 0) -> None:
        """Bound persistent numeric host/device storage separately from resident state arenas."""
        size = self.static_bytes + extra + sum(buffer.nbytes for buffer in self.layers.values())
        if size > _NUMERIC_BYTES:
            raise _Stopped("integer_buffer_limit")
        self.batch.stats["integer_bytes"] = max(size, self.batch.stats.get("integer_bytes", 0))

    def upload(self, name: str, array: np.ndarray) -> DeviceBuffer:
        """Replace a numeric generation only after checking its persistent-storage budget."""
        size = array.nbytes
        old = self.buffer_sizes.get(name, 0)
        self.account(size - old)
        buffer = self.runtime.upload(array)
        self.buffers[name] = buffer
        self.buffer_sizes[name] = size
        self.static_bytes += size - old
        return buffer

    def root_operands(self, root: int) -> np.ndarray:
        """Encode one root from exact original-price unions, geometry and optional bounds.

        Parameters
        ------------
        - root (int): packed departure/lane root in this invocation.

        Return
        --------
        - output (np.ndarray): eleven intervals in BoundOperands order, without FP32 inputs.
        """
        b = self.batch
        i, variant = b.root_flights[root], b.root_variants[root]
        bounds, scale = b.bounds[i], self.scales[i]
        cell = int(b.variants[i].cell[variant]) + b.cell_offsets[i]
        start = int(b.variants[i].start_step[variant])
        remaining = int(b.graph[cell, 12])
        last = min(b.topologies[i].max_step, start + b.topologies[i].air_hop_limit)
        self.root_empty[root] = start + remaining > last
        positive = scale.encode(bounds.root_positive(variant))
        visits = scale.encode(b.visit_lower(root, -1, cell, 0))
        endpoint, future = b.endpoints[i], b.future[i]
        residual = (endpoint.residual(tuple(map(int, endpoint.root_masks[variant])), start,
                                      start + remaining, last) if endpoint is not None else None)
        suffix = future.root_lookup(cell - b.cell_offsets[i], start) if future is not None else math.inf
        operands = prepare_reference_operands(scale, bounds, variant, 0, remaining,
                                              positive, visits, suffix, residual,
                                              b.benefits[i], b.negative[i])
        self.roots[root] = np.stack((operands.benefit.packed(), operands.assignment.packed(),
                                     operands.negative.packed(), operands.allowance.packed(),
                                     scale.encode(bounds.air_per_hop).packed(), positive.packed(),
                                     visits.packed()))
        return operands.packed()

    def update_roots(self) -> None:
        """Rebuild root bounds after a changed future generation without retaining stale layers.

        Parameters
        ------------
        - self (IntegerData): current invocation and its newly available future bounds.

        Return
        --------
        - output (None): replaces checked root uppers and releases obsolete suffix rows.
        """
        b = self.batch
        started = time.monotonic()
        self.layers.clear()
        if self.built:
            # Future preparation restarts the entire root domain with no commands in flight.
            self.live_roots.clear()
            self.cost_roots.clear()
            self.static_bytes -= sum(array.nbytes for array in self.cost_arrays)
            self.cost_arrays.clear()
            self.cost_size = 0
            self.metadata[:, 2] = -1
            for name in ("costs", "meta"):
                self.static_bytes -= self.buffer_sizes.pop(name, 0)
                self.buffers.pop(name, None)
        inputs = []
        for root in range(len(self.roots)):
            b.check()
            inputs.append(self.root_operands(root))
        packed = np.zeros((len(inputs), 12, 5), np.uint32)
        packed[:, :11] = np.asarray(inputs, np.uint32).reshape(-1, 11, 5)
        for root, upper in b.approach_upper.items():
            packed[root, 11] = self.scales[b.root_flights[root]].encode(upper).packed()
        exponents = np.asarray([len(inputs), *(self.scales[i].exponent for i in b.root_flights)], np.int32)
        (self.uppers,) = self.runtime.run("integer_roots", len(inputs),
                                        (exponents, packed, np.empty((len(inputs), 5), np.uint32)), (2,))
        if self.built:
            self.upload("roots", self.roots)
        b.stats["integer_prepare_s"] = b.stats.get("integer_prepare_s", 0.0) + time.monotonic() - started

    def build(self) -> None:
        """Pack exact positive row membership and original-FP64 root/hop cost tables lazily.

        Parameters
        ------------
        - self (IntegerData): one immutable invocation; no table is shared across snapshots.

        Return
        --------
        - output (None): publishes complete checked device tables, or exits incomplete on limits.
        """
        b = self.batch
        started = time.monotonic()
        self.metadata = np.asarray([
            (self.scales[i].exponent, b.topologies[i].air_hop_limit, -1, int(b.options.gpu_integer_audit))
            for i in b.root_flights
        ], np.int32).reshape(-1, 4)
        self.account(self.metadata.nbytes)
        self.static_bytes += self.metadata.nbytes
        self.upload("roots", self.roots)
        rows: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for i, view in enumerate(b.views):
            t, v = b.topologies[i], b.variants[i]
            first = int(v.start_step.min(initial=t.min_step)) + b.offsets[0]
            last = t.max_step + b.offsets[1]
            lookup = {b.cells[b.cell_offsets[i] + cell]: b.cell_offsets[i] + cell
                      for cell in range(t.n_cells)}
            for index, (row, price) in enumerate(view._duals.items()):
                if index % 256 == 0:
                    b.check()
                if (price > 0 and row.kind == "cell" and row.level == 0
                        and first <= row.step <= last and row.cell_coord in lookup):
                    rows[lookup[row.cell_coord]].append((row.step, price))
        ranges, times, prices = [], [], []
        for cell, i in enumerate(b.cell_flights):
            b.check()
            values = sorted(rows.get(cell, ()))
            ranges.extend((len(times), len(times) + len(values)))
            times.extend(step for step, _ in values)
            if values:
                prices.append(self.scales[i].encode_array(np.asarray([price for _, price in values])))
        if len(times) >= 2**27:
            raise _Stopped("integer_buffer_limit")
        self.upload("prices", np.asarray((len(b.cells), *ranges, *times), np.int32))
        self.upload("price_values", np.concatenate(prices) if prices else np.empty((0, 5), np.uint32))
        self.built = True
        b.stats["integer_prepare_s"] = b.stats.get("integer_prepare_s", 0.0) + time.monotonic() - started

    def prepare_costs(self) -> None:
        """Append cost operands only for roots that survived the certified root comparison.

        Parameters
        ------------
        - self (IntegerData): current invocation and exact retained-root identities.

        Return
        --------
        - output (None): publishes immutable cost/offset buffers with stable existing offsets.
          Missing or interrupted data cannot authorize a deletion.
        """
        pending = sorted(self.live_roots - self.cost_roots)
        if not pending:
            return
        b = self.batch
        started = time.monotonic()
        extra = sum(2 * (b.topologies[b.root_flights[root]].air_hop_limit + 1) for root in pending)
        total = self.cost_size + extra
        if total * 5 >= 2**31:
            raise _Stopped("integer_buffer_limit")
        # Include retained host rows and old/new device generations during replacement.
        self.account((extra + total) * 20 + self.metadata.nbytes)
        for root in pending:
            b.check()
            i, variant = b.root_flights[root], b.root_variants[root]
            array = self.scales[i].encode_array(cost_operands(b.bounds[i], variant)).reshape(-1, 5)
            self.metadata[root, 2] = self.cost_size
            self.cost_size += len(array)
            self.static_bytes += array.nbytes
            self.cost_arrays.append(array)
            self.cost_roots.add(root)
        self.upload("costs", np.concatenate(self.cost_arrays))
        self.upload("meta", self.metadata)
        b.stats["integer_cost_roots"] = len(self.cost_roots)
        b.stats["integer_prepare_s"] = b.stats.get("integer_prepare_s", 0.0) + time.monotonic() - started

    def cutoffs(self) -> DeviceBuffer:
        """Use downward cutoff integers and retain a conservative envelope of discarded maxima.

        Parameters
        ------------
        - self (IntegerData): current original-price validated incumbents.

        Return
        --------
        - output (DeviceBuffer): immutable lower cutoffs; the host retains their original FP64
          envelope so every discarded integer upper remains covered after interruption/retry.
        """
        b = self.batch
        values = tuple(best + _BOUND_CLOSURE_TOL for best, _ in b.best)
        if values != self.cutoff_key:
            encoded = np.stack([scale.encode(value).packed()
                                for scale, value in zip(self.scales, values, strict=True)])
            self.upload("cutoffs", encoded)
            self.cutoff_key = values
        # Every device deletion is <= its dispatch cutoff. Keeping that envelope
        # avoids lossy FP32 readback or a nonportable 64-bit atomic maximum.
        for i, value in enumerate(values):
            if math.isfinite(value):
                b.closed_upper[i] = max(b.closed_upper[i], value)
        return self.buffers["cutoffs"]

    def arguments(self, step: int) -> tuple[DeviceBuffer, ...]:
        """Bind one completed numeric generation and exact original-FP64 suffix row.

        Parameters
        ------------
        - step (int): absolute child visit time for this dispatch.

        Return
        --------
        - output (tuple[DeviceBuffer, ...]): immutable arguments retained through the command block.
        """
        b = self.batch
        if not self.built:
            self.build()
        self.prepare_costs()
        started = time.monotonic()
        if self.endpoint_generation != b._endpoint_generation:
            arrays = [self.scales[i].encode_array(tree).reshape(-1, 5)
                      for i, endpoint in enumerate(b.endpoints) if endpoint is not None
                      for tree in endpoint.entries.values()]
            self.upload("endpoints", np.concatenate(arrays) if arrays else np.empty((0, 5), np.uint32))
            self.endpoint_generation = b._endpoint_generation
        if step not in self.layers:
            while len(self.layers) >= 2:
                self.layers.popitem(last=False)
            packed = np.zeros((len(b.cells), 5), np.uint32)
            for i, future in enumerate(b.future):
                if future is not None and future.first_step <= step <= future.last_step:
                    first, count = b.cell_offsets[i], b.topologies[i].n_cells
                    # The unrestricted suffix is weaker than the reference's hop refinement.
                    packed[first:first + count] = self.scales[i].encode_array(future.row(step))
            self.account(packed.nbytes)
            self.layers[step] = self.runtime.upload(packed)
        self.layers.move_to_end(step)
        cutoff = self.cutoffs()
        b.stats["integer_gather_s"] = b.stats.get("integer_gather_s", 0.0) + time.monotonic() - started
        return (*(self.buffers[name] for name in
                  ("meta", "roots", "costs", "prices", "price_values", "endpoints")),
                self.layers[step], cutoff)


class IntegerBatch(_Batch):
    """Search with integer deletion bounds while retaining original canonical completions.

    Attributes
    ------------
    - numeric (IntegerData): snapshot-local checked operands and device prefix representation.
    """

    def prepare(self) -> None:
        """Extend ordinary preparation with checked root bounds and opaque numeric record words.

        Parameters
        ------------
        - self (IntegerBatch): current graph, snapshot, restrictions and canonical seeds.

        Return
        --------
        - output (None): prepares original-price numeric data without allocating resident arenas.
        """
        super().prepare()
        self.stride += _STATE_WORDS
        if (self.options.max_labels * self.stride >= 2**31
                or 6 * min(self.options.chunk_size, self.options.max_labels) * self.stride >= 2**31):
            raise ValueError("Integer GPU label indexing exceeds int32 range")
        self.numeric = IntegerData(self)
        self.runtime.integer_data = self.numeric
        self.stats["integer_audited"] = 0
        self.stats["integer_bounds"] = True
        self.stats["integer_roots"] = len(self.root_flights)

    def evaluate_roots(self) -> None:
        """Keep FP32 root ranking separate from refreshed certified integer root bounds."""
        super().evaluate_roots()
        if hasattr(self, "numeric"):
            self.numeric.update_roots()

    def roots_at(
        self, step: int, allowed_roots: frozenset[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retain every legal root not closed by its checked GPU-computed upper.

        Parameters
        ------------
        - step (int): absolute root time.
        - allowed_roots (frozenset[int] | None): optional preliminary-search-only selection.

        Return
        --------
        - output (tuple[np.ndarray, np.ndarray]): complete records and FP32 ranking values.
        """
        records, values = [], []
        for root in self.start_buckets.get(step, ()):
            self.check()
            if allowed_roots is not None and root not in allowed_roots:
                continue
            if self.numeric.root_empty[root]:
                continue
            i, variant = self.root_flights[root], self.root_variants[root]
            cell = self.cell_offsets[i] + int(self.variants[i].cell[variant])
            scale = self.numeric.scales[i]
            upper = IntegerInterval.unpacked(self.numeric.uppers[root])
            if scale.closes(upper, self.best[i][0] + _BOUND_CLOSURE_TOL):
                if self.options.gpu_integer_audit:
                    self.check_upper(root, -1, cell, 0, upper)
                self.closed_upper[i] = max(self.closed_upper[i], scale.upper_float(upper))
                self.stats["root_prunes"] += 1
                if root in self.approach_upper:
                    self.stats["approach_root_prunes"] = self.stats.get("approach_root_prunes", 0) + 1
                self.stats["gpu_direct_prunes"] += 1
                self.stats["gpu_prune_proposals"] += 1
                continue
            record = np.full(self.stride, -1, np.int32)
            record[:7] = cell, root, 0, -1, -1, -1, 2
            record[7:9] = self.root_endpoint[root, :2]
            record[9] = cell
            record[-15:-5] = self.numeric.roots[root, 5:7].reshape(-1).view(np.int32)
            record[-5:] = self.numeric.uppers[root].view(np.int32)
            records.append(record)
            values.append((self.variants[i].score[variant], self.bounds[i].root_positive(variant)))
        packed = np.asarray(records, np.int32).reshape(-1, self.stride)
        allocated = self.allocate(packed)
        self.numeric.live_roots.update(map(int, packed[:, 1]))
        return allocated, np.asarray(values, np.float32).reshape(-1, 2)

    def approach_open(self, root: int, cutoff: float) -> bool:
        """Avoid optional preparation when the original checked device interval already closes."""
        scale = self.numeric.scales[self.root_flights[root]]
        return not self.numeric.root_empty[root] and not scale.closes(
            IntegerInterval.unpacked(self.numeric.uppers[root]), cutoff,
        )

    def check_upper(self, root: int, parent: int, cell: int, hops: int, upper: IntegerInterval) -> None:
        """Audit a device upper against the original FP64 verifier without trusting device masks.

        Parameters
        ------------
        - root (int): current packed departure/lane root.
        - parent (int): independently recovered predecessor, or -1 for a root.
        - cell (int): packed current cell.
        - hops (int): completed moves from the root.
        - upper (IntegerInterval): actual device-generated upper, before binary64 conversion.

        Return
        --------
        - output (None): raises on any containment failure; a failed audit cannot certify.
        """
        reference = _Batch.bound(self, root, parent, cell, hops)
        scale = self.numeric.scales[self.root_flights[root]]
        if (not upper.valid or math.isnan(reference) or reference == math.inf
                or (reference != -math.inf and upper.upper * scale.quantum < Fraction(reference))):
            raise AssertionError("Integer GPU upper does not contain the original FP64 bound")
        self.stats["integer_audited"] += 1

    def audit_integer_record(self, record: np.ndarray) -> None:
        """Check a queued integer proposal after its original-price predecessor is available."""
        root, parent, cell, hops = map(int, (record[1], record[3], record[0], record[2]))
        upper = IntegerInterval.unpacked(record[-5:].view(np.uint32))
        self.check_upper(root, parent, cell, hops, upper)
        self._integer_audit_pending = (root, parent, cell, hops, upper)

    def retain(self, root: int, parent: int, cell: int, hops: int) -> bool:
        """Resolve only already-audited integer proposals; ordinary FP32 flags cannot delete.

        Parameters
        ------------
        - root (int): packed root of the audited proposal.
        - parent (int): stable predecessor ID.
        - cell (int): packed current cell.
        - hops (int): completed moves.

        Return
        --------
        - output (bool): false only after a matching independent containment audit; otherwise raises.
        """
        pending = getattr(self, "_integer_audit_pending", None)
        if not self.options.gpu_integer_audit or pending is None or pending[:4] != (root, parent, cell, hops):
            raise AssertionError("Unexpected per-prune CPU verification in integer mode")
        i = self.root_flights[root]
        upper = self.numeric.scales[i].upper_float(pending[4])
        self.closed_upper[i] = max(self.closed_upper[i], upper)
        self.stats["verified_prunes"] += 1
        self._integer_audit_pending = None
        return False
