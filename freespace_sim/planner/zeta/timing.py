"""Deterministic bounded-speed timing shared by Zeta search and route reconstruction."""

from __future__ import annotations

import math
from collections.abc import Sequence


def edge_end_times(
    start: float,
    durations: Sequence[float],
    arrival_snap_s: float,
    speed_fraction: float,
) -> tuple[float, ...]:
    """Snap one candidate leg's arrival while retaining its nominal subdivision.

    Parameters
    ------------
    - start (float): absolute departure time from the candidate waypoint.
    - durations (Sequence[float]): positive nominal durations of its corridor subparts.
    - arrival_snap_s (float): absolute arrival-grid period; zero disables snapping.
    - speed_fraction (float): permitted fractional deviation from nominal speed, below one.

    Return
    --------
    - output (tuple[float, ...]): strictly increasing subpart arrival times. The nearest
      feasible grid arrival wins, with exact ties preferring the later arrival. Disabled
      snapping or an unreachable grid retains the builder's sequential nominal additions.
    """
    if not math.isfinite(start):
        raise ValueError("edge start must be finite")
    if not math.isfinite(arrival_snap_s) or arrival_snap_s < 0.0:
        raise ValueError("arrival_snap_s must be finite and nonnegative")
    if not math.isfinite(speed_fraction) or not 0.0 <= speed_fraction < 1.0:
        raise ValueError("speed_fraction must lie in [0, 1)")
    nominal_ends = []
    stamp = start
    for duration in durations:
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("edge durations must be finite and positive")
        next_stamp = stamp + duration
        if not math.isfinite(next_stamp) or next_stamp <= stamp:
            raise ValueError("edge durations must advance the representable clock")
        nominal_ends.append(next_stamp)
        stamp = next_stamp
    if len(durations) == 0 or arrival_snap_s == 0.0:
        return tuple(nominal_ends)

    nominal = math.fsum(durations)
    target = start + nominal
    grid_position = target / arrival_snap_s
    if not math.isfinite(grid_position):
        return tuple(nominal_ends)
    lower = nominal / (1.0 + speed_fraction)
    upper = nominal / (1.0 - speed_fraction)
    before = math.floor(grid_position)
    candidates = (before * arrival_snap_s, (before + 1) * arrival_snap_s)
    for arrival in sorted(candidates, key=lambda value: (abs(value - target), -value)):
        if not math.isfinite(arrival):
            continue
        elapsed = arrival - start
        # Only representation error in absolute timestamps may cross a speed boundary.
        tolerance = math.ulp(start) + math.ulp(arrival) + math.ulp(nominal)
        if elapsed <= 0.0 or not lower - tolerance <= elapsed <= upper + tolerance:
            continue
        ends = tuple(
            start + elapsed * (math.fsum(durations[:index]) / nominal)
            for index in range(1, len(durations))
        ) + (arrival,)
        if all(end > previous for previous, end in zip((start, *ends), ends)):
            return ends
    return tuple(nominal_ends)
