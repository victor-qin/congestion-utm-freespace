"""Sparse visit prices shared by every GPU batch using one LP price snapshot."""

from __future__ import annotations

from collections.abc import Callable
import math

from .pricing import DualView


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
