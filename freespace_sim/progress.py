"""Rolling-rate arithmetic shared by every long-running stage's progress reporting.

Lives in its own module rather than in :mod:`freespace_sim.sim` because the colgen pricing
sweep needs it too, and ``sim`` imports ``planner`` -- so a planner reaching back into
``sim`` is an import cycle. The alternative was a second copy of the same ETA, which is how
two progress reporters end up disagreeing about one run.
"""
from __future__ import annotations

from collections import deque


class RollingRate:
    """Cumulative + rolling mean of a per-item duration (s), and the ETA it implies.

    Two means rather than one, and that is the point: a whole-run average lags, so a stage
    that slows down mid-run hides inside it. The rolling value pulls ABOVE the cumulative
    average as soon as the slowdown starts, and the two diverging is the signal.

    ``roll_s``/``eta_s`` return ``None`` until ``window`` samples have accrued, so a caller
    never publishes an estimate drawn from a partial sample.
    """

    def __init__(self, window: int = 100):
        """Start an empty tracker whose rolling mean spans the last ``window`` samples."""
        self.window = window
        self._recent: deque[float] = deque(maxlen=window)
        self._sum = 0.0

    def add(self, dt_s: float) -> None:
        """Record one item's duration ``dt_s`` (s) into the cumulative sum and rolling window."""
        self._sum += dt_s
        self._recent.append(dt_s)

    @property
    def total_s(self) -> float:
        """Cumulative total duration (s) recorded so far."""
        return self._sum

    def avg_s(self, done: int) -> float:
        """Mean seconds per item over all ``done`` items (guards ``done`` = 0)."""
        return self._sum / max(done, 1)

    def avg_ms(self, done: int) -> float:
        """Mean milliseconds per item over all ``done`` items."""
        return 1000.0 * self.avg_s(done)

    def roll_s(self) -> float | None:
        """Rolling mean seconds per item, or ``None`` until ``window`` samples have accrued."""
        if len(self._recent) < self.window:
            return None
        return sum(self._recent) / self.window

    def roll_ms(self) -> float | None:
        """Rolling mean milliseconds per item, or ``None`` until the window is full."""
        rolling = self.roll_s()
        return None if rolling is None else 1000.0 * rolling

    def eta_s(self, done: int, total: int) -> float | None:
        """Seconds left for ``total - done`` items, or ``None`` until the window fills."""
        rolling = self.roll_s()
        return None if rolling is None else max(0, total - done) * rolling
