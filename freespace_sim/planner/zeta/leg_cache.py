"""Bounded process-local caches of price-independent Zeta leg data."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import struct
import sys
from threading import RLock
from typing import cast

import numpy as np

from ...config import SimConfig
from ...geometry import BoxSpec, CylinderSpec
from ...volumes import corridor_segment_volume
from .geometry import volume_spatial_cells
from .timing import edge_end_times

Point = tuple[float, float, float]
Space = tuple[int, int, int]
Timing = tuple[np.ndarray, np.ndarray, np.ndarray]

GEOMETRY_CACHE_BYTES = 64 * 1024 * 1024
TIMING_CACHE_BYTES = 32 * 1024 * 1024
_MISSING = object()
_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class LegPart:
    """One immutable horizontal subpart before flight-specific access checks.

    Attributes
    ------------
    - start (Point): exact interpolated start coordinates.
    - end (Point): exact interpolated end coordinates.
    - shape (BoxSpec): simulator-owned spatial box, without a terminal exemption.
    - duration (float): nominal sequential duration increment.
    - length (float): exact length used by the original graph builder.
    """

    start: Point
    end: Point
    shape: BoxSpec
    duration: float
    length: float


class _Store:
    """LRU storage bounded by conservative retained-entry estimates.

    Attributes
    ------------
    - limit (int): maximum estimated entry storage in bytes.
    - size (int): summed estimates for currently retained entries, not process RSS.
    - entries (OrderedDict): keys mapped to immutable values and their estimates.
    """

    def __init__(self, limit: int) -> None:
        """Create an empty store; all access is protected by the module lock."""
        if limit < 0:
            raise ValueError("cache byte limits must be nonnegative")
        self.limit = limit
        self.size = 0
        self.entries: OrderedDict[tuple[object, ...], tuple[object, int]] = OrderedDict()

    def get(self, key: tuple[object, ...]) -> object:
        """Return and promote an entry, or the private missing sentinel."""
        entry = self.entries.get(key)
        if entry is None:
            return _MISSING
        self.entries.move_to_end(key)
        return entry[0]

    def put(self, key: tuple[object, ...], value: object, size: int) -> None:
        """Retain a complete fitting entry, evicting only least-recently-used data."""
        if size > self.limit:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.size -= previous[1]
        while self.size + size > self.limit:
            _, removed = self.entries.popitem(last=False)
            self.size -= removed[1]
        self.entries[key] = value, size
        self.size += size


_GEOMETRY = _Store(GEOMETRY_CACHE_BYTES)
_TIMING = _Store(TIMING_CACHE_BYTES)
_COUNTS = dict.fromkeys(("leg_hits", "leg_misses", "footprint_hits", "footprint_misses",
                       "timing_hits", "timing_misses"), 0)


def get_leg(a: Point, b: Point, cfg: SimConfig) -> tuple[LegPart, ...]:
    """Reuse exact horizontal subparts without caching terminal or obstacle decisions.

    Parameters
    ------------
    - a (Point): absolute source waypoint coordinates at the route's cruise altitude.
    - b (Point): absolute destination waypoint coordinates at that same altitude.
    - cfg (SimConfig): nominal speed, subdivision length and corridor cross-section.

    Return
    --------
    - output (tuple[LegPart, ...]): immutable builder-identical spatial subparts;
      terminal tags and every flight-specific feasibility rule remain the caller's job.
    """
    # Preserve coordinate bits, including signed zero, rather than translating legs.
    key = ("leg", struct.pack("!6d", *a, *b), cfg.corridor_segment_len_m, cfg.nominal_speed_mps,
           cfg.corridor_width_m, cfg.corridor_height_m)
    with _LOCK:
        cached = _GEOMETRY.get(key)
        if cached is not _MISSING:
            _COUNTS["leg_hits"] += 1
            return cast(tuple[LegPart, ...], cached)
        _COUNTS["leg_misses"] += 1
    dx, dy = b[0] - a[0], b[1] - a[1]
    total = math.sqrt(dx * dx + dy * dy)
    nsub = max(1, math.ceil(total / cfg.corridor_segment_len_m))
    parts = []
    for k in range(1, nsub + 1):
        # Preserve graph.parts' interpolation, operation order and duration clamp.
        f0, f1 = (k - 1) / nsub, k / nsub
        sa = a[0] + f0 * dx, a[1] + f0 * dy, a[2]
        sb = a[0] + f1 * dx, a[1] + f1 * dy, a[2]
        hx, hy = sb[0] - sa[0], sb[1] - sa[1]
        length = math.sqrt(hx * hx + hy * hy)
        duration = max(length / cfg.nominal_speed_mps, 1e-3)
        shape = corridor_segment_volume(sa, 0.0, sb, duration, cfg).shape
        parts.append(LegPart(sa, sb, shape, duration, length))
    result = tuple(parts)
    # Includes key tuples, coordinates, frozen boxes and LRU/Python object overhead.
    with _LOCK:
        _GEOMETRY.put(key, result, 1024 + 2048 * len(result))
    return result


def get_footprint(shape: BoxSpec | CylinderSpec, cfg: SimConfig) -> frozenset[Space]:
    """Reuse the full simulator spatial coverage independently of current priced rows.

    Parameters
    ------------
    - shape (BoxSpec | CylinderSpec): exact immutable physical reservation shape.
    - cfg (SimConfig): hex pitch and cruise-level bands used by the simulator converter.

    Return
    --------
    - output (frozenset[Space]): every intersected hex/level cell, including unpriced cells.
      No forbidden-row or terminal-access result is retained.
    """
    key = ("footprint", shape, cfg.corridor_segment_len_m, cfg.corridor_height_m,
           cfg.flight_levels_m)
    with _LOCK:
        cached = _GEOMETRY.get(key)
        if cached is not _MISSING:
            _COUNTS["footprint_hits"] += 1
            return cast(frozenset[Space], cached)
        _COUNTS["footprint_misses"] += 1
    # Avoid also retaining the same shape in the converter's separate global LRU.
    result = volume_spatial_cells.__wrapped__(shape, cfg)
    with _LOCK:
        size = (2048 + sys.getsizeof(result) + 192 * len(result)
                + sys.getsizeof(cfg.flight_levels_m) + 32 * len(cfg.flight_levels_m))
        _GEOMETRY.put(key, result, size)
    return result


def get_timing(
    start: float, durations: tuple[float, ...], snap: float, speed_fraction: float,
    dt: float, buffer: float,
) -> Timing:
    """Reuse exact subpart clocks and buffered periods without current row identifiers.

    Parameters
    ------------
    - start (float): exact absolute departure timestamp; no phase rounding is applied.
    - durations (tuple[float, ...]): nominal subpart increments in their original order.
    - snap (float): arrival-grid period, or zero for sequential nominal timing.
    - speed_fraction (float): permitted speed variation used by the unchanged timing rule.
    - dt (float): positive reservation-period length.
    - buffer (float): nonnegative symmetric reservation time buffer.

    Return
    --------
    - output (Timing): bytes-backed read-only arrays of exact end times, inclusive first
      periods and exclusive stop periods. Empty time windows have equal first/stop values.
      Prices, blocked flags and already-paid history must be evaluated by the caller.
    """
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("reservation period must be finite and positive")
    if not math.isfinite(buffer) or buffer < 0.0:
        raise ValueError("reservation buffer must be finite and nonnegative")
    key = (start, durations, snap, speed_fraction, dt, buffer)
    with _LOCK:
        cached = _TIMING.get(key)
        if cached is not _MISSING:
            _COUNTS["timing_hits"] += 1
            return cast(Timing, cached)
        _COUNTS["timing_misses"] += 1
    ends = edge_end_times(start, durations, snap, speed_fraction)
    firsts, stops = [], []
    stamp = start
    for end in ends:
        t0, t1 = stamp - buffer, end + buffer
        if t0 == t1:
            firsts.append(0)
            stops.append(0)
            stamp = end
            continue
        # Match prepare_timed_edge's multiplied-grid boundary repairs exactly.
        first = math.floor(t0 / dt)
        while (first + 1) * dt <= t0:
            first += 1
        while first * dt > t0:
            first -= 1
        stop = math.ceil(t1 / dt)
        while (stop - 1) * dt >= t1:
            stop -= 1
        while stop * dt < t1:
            stop += 1
        firsts.append(first)
        stops.append(stop)
        stamp = end
    result = (
        np.frombuffer(np.asarray(ends, dtype=np.float64).tobytes(), dtype=np.float64),
        np.frombuffer(np.asarray(firsts, dtype=np.int64).tobytes(), dtype=np.int64),
        np.frombuffer(np.asarray(stops, dtype=np.int64).tobytes(), dtype=np.int64),
    )
    with _LOCK:
        _TIMING.put(key, result, 1536 + 128 * len(durations))
    return result


def stats() -> dict[str, int]:
    """Snapshot cumulative lookup counts and current estimated retained storage.

    Parameters
    ------------
    - None: statistics are local to this worker process.

    Return
    --------
    - output (dict[str, int]): hit/miss counters and bounded geometry/timing byte estimates;
      these estimates do not include graph-owned references or represent process RSS.
    """
    with _LOCK:
        return {**_COUNTS, "geometry_bytes": _GEOMETRY.size, "timing_bytes": _TIMING.size,
                "geometry_entries": len(_GEOMETRY.entries), "timing_entries": len(_TIMING.entries)}


def clear_caches() -> None:
    """Discard both stores and reset counters using the current byte-limit constants.

    Parameters
    ------------
    - None: callers may set the module limits before resetting controlled experiments.

    Return
    --------
    - output (None): new empty stores; existing immutable caller references remain valid.
    """
    global _GEOMETRY, _TIMING
    with _LOCK:
        geometry, timing = _Store(GEOMETRY_CACHE_BYTES), _Store(TIMING_CACHE_BYTES)
        _GEOMETRY, _TIMING = geometry, timing
        for name in _COUNTS:
            _COUNTS[name] = 0
