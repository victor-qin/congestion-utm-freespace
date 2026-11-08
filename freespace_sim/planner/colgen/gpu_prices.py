"""Sparse visit prices shared by every GPU batch using one LP price snapshot."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
import math
from typing import TYPE_CHECKING

import numpy as np

from .pricing import DualView

if TYPE_CHECKING:
    from .gpu_runtime import ComputeRuntime


class PriceSummary:
    """Original-price validation and outward sums shared by one immutable snapshot."""

    def __init__(self, view: DualView, check: Callable[[], None]) -> None:
        """Retain the exact iteration order and FP64 rounding of the per-flight bounds.

        Parameters
        ------------
        - view (DualView): fixed normalized prices, including every signed/terminal row.
        - check (Callable[[], None]): cooperative deadline check during construction.

        Return
        --------
        - output (None): downward negative credit and upward absolute-price allowance.
        """
        negative = absolute = 0.0
        for index, price in enumerate(view._duals.values()):
            if index % 256 == 0:
                check()
            if not math.isfinite(price):
                raise ValueError("Normalized dual prices must be finite")
            if price < 0.0:
                negative = math.nextafter(negative + price, -math.inf)
            if price != 0.0:
                absolute = math.nextafter(absolute + abs(price), math.inf)
        check()

        self.negative, self.absolute = negative, absolute


def summary_for(view: DualView, check: Callable[[], None] = lambda: None) -> PriceSummary:
    """Publish a complete summary once per immutable dual snapshot.

    Parameters
    ------------
    - view (DualView): snapshot owning the cache; raw mapping callers must normalize anew.
    - check (Callable[[], None]): optional deadline check; failures leave no partial cache.

    Return
    --------
    - output (PriceSummary): shared numerical summaries with unchanged FP64 semantics.
    """
    if view._gpu_summary is None:
        view._gpu_summary = PriceSummary(view, check)
    return view._gpu_summary


class SparseVisitPrices:
    """Sum each priced resource/visit window once, independently of flight graphs.

    The owning DualView supplies an immutable normalized price snapshot. Keep
    levels and absolute times in the keys so later batches can gather any graph.
    Flight restrictions and destination bounds do not belong in this cache.
    """

    def __init__(self, view: DualView, check: Callable[[], None]) -> None:
        """Build sparse signed and positive sums with the original FP64 arithmetic.

        Parameters
        ------------
        - view (DualView): fixed normalized row prices and inclusive visit offsets.
        - check (Callable[[], None]): raises when the caller's deadline expires.

        Return
        --------
        - output (None): stores sums by (cell, level) and absolute visit time.
        """
        sums: dict[tuple[tuple[int, int], int], dict[int, list[float]]] = {}
        lo, hi = view.offsets
        for index, (row, price) in enumerate(view._duals.items()):
            if index % 256 == 0:
                check()
            if row.kind != "cell" or price == 0.0:
                continue
            windows = sums.setdefault((row.cell_coord, row.level), {})
            for offset, step in enumerate(range(row.step - hi, row.step - lo + 1)):
                if offset and offset % 256 == 0:
                    check()
                windows.setdefault(step, []).append(price)
        self.cells: dict[tuple[tuple[int, int], int], dict[int, tuple[float, float]]] = {}
        for resource, windows in sums.items():
            check()
            values = {}
            for index, (step, prices) in enumerate(windows.items()):
                if index % 256 == 0:
                    check()
                values[step] = math.fsum(prices), math.fsum(max(0.0, p) for p in prices)
            self.cells[resource] = values
        check()


class DeviceVisitPrices:
    """One snapshot's sparse signed/positive visit sums, retained on one device."""

    def __init__(
        self, prices: SparseVisitPrices, runtime: ComputeRuntime, check: Callable[[], None],
    ) -> None:
        """Upload exact CPU sums once, with complete physical resource/time support.

        Parameters
        ------------
        - prices (SparseVisitPrices): original FP64 fsum results for one immutable snapshot.
        - runtime (ComputeRuntime): device owner; tables cannot cross runtimes.
        - check (Callable[[], None]): deadline check during construction and upload.

        Return
        --------
        - output (None): device CSR ranges, sorted absolute times and FP32 price proposals.
        """
        self.slots = {}
        ranges, steps, values = [(0, 0)], [], []
        for resource, windows in prices.cells.items():
            check()
            start = len(steps)
            for step in sorted(windows):
                # Supported GPU graphs use int32 clocks, checked before search. These
                # windows cannot match any such visit; summaries still include all rows.
                if -(2**31) <= step < 2**31:
                    steps.append(step)
                    values.append(windows[step])
            self.slots[resource] = len(ranges)
            ranges.append((start, len(steps)))
        if len(steps) >= 2**30 or len(ranges) >= 2**30:
            raise ValueError("GPU price indexing exceeds int32 range")
        arrays = (np.asarray(ranges, np.int32), np.asarray(steps, np.int32),
                  np.asarray(values, np.float32).reshape(-1, 2))
        uploaded = []
        for array in arrays:
            check()
            uploaded.append(runtime.upload(array))
        check()
        self.ranges, self.steps, self.values = uploaded
        self.nbytes = sum(max(4, array.nbytes) for array in arrays)


class DevicePriceCache:
    """Bounded LRU of snapshot-owned device tables; eviction never changes prices."""

    def __init__(self, max_snapshots: int = 2, max_bytes: int = 128 * 1024 * 1024) -> None:
        """Bound retained tables; a larger live table is returned without caching it."""
        self.max_snapshots, self.max_bytes = max_snapshots, max_bytes
        self.stores: OrderedDict[DualView, DeviceVisitPrices] = OrderedDict()
        self.nbytes = 0

    def clear(self) -> None:
        """Release the cache's ownership; any active batch keeps its own table alive."""
        self.stores.clear()
        self.nbytes = 0

    def get(
        self, view: DualView, runtime: ComputeRuntime, check: Callable[[], None],
    ) -> DeviceVisitPrices:
        """Reuse only the identical immutable snapshot, publishing complete uploads.

        Parameters
        ------------
        - view (DualView): normalized snapshot; raw maps must be copied on each public call.
        - runtime (ComputeRuntime): device owning this cache.
        - check (Callable[[], None]): deadline check for builds and warm hits.

        Return
        --------
        - output (DeviceVisitPrices): live table independent of flight restrictions and bounds.
        """
        check()
        if view in self.stores:
            self.stores.move_to_end(view)
            runtime.counters["price_cache_hits"] += 1
            return self.stores[view]
        if view._gpu_visit_prices is None:
            view._gpu_visit_prices = SparseVisitPrices(view, check)
        store = DeviceVisitPrices(view._gpu_visit_prices, runtime, check)
        runtime.counters["price_uploads"] += 1
        runtime.counters["price_upload_bytes"] += store.nbytes
        if self.max_snapshots > 0 and store.nbytes <= self.max_bytes:
            while self.stores and (len(self.stores) >= self.max_snapshots
                                  or self.nbytes + store.nbytes > self.max_bytes):
                _, old = self.stores.popitem(last=False)
                self.nbytes -= old.nbytes
            self.stores[view] = store
            self.nbytes += store.nbytes
        return store
