"""Sparse visit prices shared by every GPU batch using one LP price snapshot."""

from __future__ import annotations

from collections.abc import Callable
import math

from .pricing import DualView


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
