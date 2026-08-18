"""Geometry-derived space-time row windows for the column-generation planner.

The helpers in this module are the sole source of truth for translating the
ledger's continuous, half-open reservation volumes into ``colgen`` capacity
rows.  In particular, the temporal offsets are measured from volumes built by
:func:`~freespace_sim.volumes.corridor_segment_volume`; callers must not copy a
hard-coded three-period window.
"""

from __future__ import annotations

import math
import operator
import sys
from dataclasses import replace
from functools import lru_cache

import numpy as np

from ...config import SimConfig
from ...conflict import volumes_conflict
from ...types import Vec
from ...volumes import Volume4D, corridor_segment_volume
from ..hexgrid import AXIAL_NEIGHBORS, circumradius, enu_to_axial, hex_center, hex_distance

AxialCell = tuple[int, int]
CellWindow = tuple[int, int]

# ``build_reservation_from_corners`` accumulates segment durations in floating
# point.  Endpoint claims start from the discrete clock, so their rounding pad
# must grow with the number of rebuilt steps rather than assuming every route
# has the short-path drift measured by the parity fixtures.
_MIN_ENDPOINT_TIME_PAD_S = 1e-9


def _endpoint_time_pad_s(t0: float, t1: float, dt: float, timing_steps: int) -> float:
    """Conservatively bound floating accumulation in the corner-rebuild clock."""

    scale = max(abs(t0), abs(t1), dt, 1.0)
    # A nominal hop can split into two subsegments at the pitch-rounding tripwire.
    # The operation factor covers interpolation, norms, division, and clock addition.
    operations = max(1, 2 * timing_steps + 1)
    return max(
        _MIN_ENDPOINT_TIME_PAD_S,
        64.0 * sys.float_info.epsilon * operations * scale,
    )


def _periods_overlapping(t0: float, t1: float, dt: float) -> range:
    """Return grid periods whose half-open intervals overlap ``[t0, t1)``.

    Period ``j`` denotes ``[j * dt, (j + 1) * dt)``.  Consequently the first
    touched period is ``floor(t0 / dt)`` and the exclusive stop is
    ``ceil(t1 / dt)``.  This helper deliberately has no tolerance padding:
    template corridor times are created directly by the ledger builder and the
    expected default offset tuple is exactly ``(-1, 1)``.
    """

    # Division can put an exact constructed boundary on the wrong side of its
    # integer (for example ``(3 * 0.7) / 0.7 < 3``).  Correct the quotient by
    # comparing against the same multiplied grid values that define the row
    # intervals.  Unlike an epsilon snap, this still treats ``nextafter(k*dt,
    # +inf)`` as genuinely inside the following period.
    start = math.floor(t0 / dt)
    while (start + 1) * dt <= t0:
        start += 1
    while start * dt > t0:
        start -= 1

    stop = math.ceil(t1 / dt)
    while (stop - 1) * dt >= t1:
        stop -= 1
    while stop * dt < t1:
        stop += 1
    return range(start, stop)


def visit_rows(v: int, offsets: CellWindow) -> range:
    """Return all cell-capacity row periods claimed by a visit at step ``v``.

    ``offsets`` is the inclusive ``(lo, hi)`` tuple returned by
    :func:`derive_cell_window`.  With the default reservation geometry this is
    ``(-1, 1)``, so a visit at step ``v`` claims ``v-1, v, v+1``.  Returning a
    :class:`range` keeps both pricing and claim construction allocation-light.

    Args:
        v: Integer step at which the trajectory centre reaches the cell.
        offsets: Inclusive temporal offsets relative to ``v``.

    Raises:
        ValueError: If the offset interval is empty or reversed.
    """

    lo, hi = offsets
    if lo > hi:
        raise ValueError(f"cell-window offsets must satisfy lo <= hi, got {offsets!r}")
    return range(v + lo, v + hi + 1)


def _point(cell: AxialCell, z: float, radius: float) -> Vec:
    """Return a 3-D lattice-centre point for a small template construction."""

    xy = hex_center(*cell, radius)
    return np.array((float(xy[0]), float(xy[1]), float(z)), dtype=float)


def _shift_volume(volume: Volume4D, steps: int, dt: float) -> Volume4D:
    """Translate only a template volume's time interval by ``steps`` periods."""

    shift = steps * dt
    return replace(volume, t_start=volume.t_start + shift, t_end=volume.t_end + shift)


def _template_arcs(cfg: SimConfig, radius: float) -> list[tuple[Volume4D, tuple[int, ...]]]:
    """Build representative hop, hover, and rung arcs through cell ``(0, 0)``.

    Each result pairs a ledger volume with the centre-cell visit steps that
    would claim it.  Hop templates have one visit to the common cell.  Hover
    and rung templates remain in that cell for every elapsed step; although
    those arc types are outside colgen v1's column universe, including them is
    a cheap guard that the measured per-visit window still covers the ledger's
    existing geometry classes.
    """

    dt = cfg.dt_s
    z = cfg.cruise_level_m
    centre = _point((0, 0), z, radius)
    templates: list[tuple[Volume4D, tuple[int, ...]]] = []

    # Every directed lateral arc incident to the centre, anchored so its
    # centre-cell visit is step zero.  Pairwise scans cover same-edge, swap,
    # 60/120-degree crossing, and adjacent trailing classes.
    for dq, dr in AXIAL_NEIGHBORS:
        neighbour = _point((dq, dr), z, radius)
        templates.append((corridor_segment_volume(neighbour, -dt, centre, 0.0, cfg), (0,)))
        templates.append((corridor_segment_volume(centre, 0.0, neighbour, dt, cfg), (0,)))

    # A one-step in-cell hover in both temporal orientations around visit 0.
    templates.append((corridor_segment_volume(centre, -dt, centre, 0.0, cfg), (-1, 0)))
    templates.append((corridor_segment_volume(centre, 0.0, centre, dt, cfg), (0, 1)))

    # Existing A* rungs use an integral number of steps.  Marking each elapsed
    # step is the conservative cell-presence representation a future colgen
    # rung would consume.
    xy = hex_center(0, 0, radius)
    for z0, z1 in zip(cfg.flight_levels_m, cfg.flight_levels_m[1:]):
        n_steps = max(1, math.ceil((z1 - z0) / (cfg.climb_rate_mps * dt)))
        duration = n_steps * dt
        low = np.array((float(xy[0]), float(xy[1]), float(z0)), dtype=float)
        high = np.array((float(xy[0]), float(xy[1]), float(z1)), dtype=float)
        for start, end in ((low, high), (high, low)):
            templates.append(
                (
                    corridor_segment_volume(start, -duration, end, 0.0, cfg),
                    tuple(range(-n_steps, 1)),
                )
            )
            templates.append(
                (
                    corridor_segment_volume(start, 0.0, end, duration, cfg),
                    tuple(range(0, n_steps + 1)),
                )
            )

    return templates


def _cross_check_conflicts(cfg: SimConfig, offsets: CellWindow, radius: float) -> None:
    """Assert that template conflicts are covered by a common cell row.

    The offset tuple is measured from volume time footprints, while the
    authoritative ledger predicate also includes exact 3-D geometry.  This
    finite scan ties the two together: for every representative arc class and
    every temporal displacement at which the volumes can meet, a geometric
    conflict must imply an intersecting row claim.
    """

    dt = cfg.dt_s
    templates = _template_arcs(cfg, radius)
    max_extent = max(max(abs(volume.t_start), abs(volume.t_end)) for volume, _visits in templates)
    # Beyond twice the largest anchored extent the two half-open time windows
    # cannot overlap.  Two guard steps make the scan robust at exact boundaries.
    shift_limit = 2 * math.ceil(max_extent / dt) + 2

    claims = [
        frozenset(row for visit in visits for row in visit_rows(visit, offsets))
        for _volume, visits in templates
    ]
    for i, (first, _first_visits) in enumerate(templates):
        for j, (second, second_visits) in enumerate(templates):
            for shift in range(-shift_limit, shift_limit + 1):
                shifted = _shift_volume(second, shift, dt)
                if not volumes_conflict(first, shifted):
                    continue
                shifted_claims = frozenset(
                    row for visit in second_visits for row in visit_rows(visit + shift, offsets)
                )
                if claims[i].isdisjoint(shifted_claims):
                    raise RuntimeError(
                        "derived cell window does not cover a ledger template conflict: "
                        f"offsets={offsets}, templates=({i}, {j}), shift={shift}"
                    )


@lru_cache(maxsize=64)
def validate_edge_locality(cfg: SimConfig) -> None:
    """Require every conflicting pair of lattice hops to share an endpoint cell.

    Visit-only rows can cover a hop conflict only through one of the hop's two
    visited cells.  The shipped 60 m corridor on a 120 m pitch has this
    property, but wider otherwise-valid corridor configurations need not.  A
    finite translation-invariant local sweep fails closed until hex sizing is
    derived from corridor geometry (GitHub issue #72).
    """

    radius = circumradius(cfg)
    pitch = float(cfg.corridor_segment_len_m)
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError(f"corridor pitch must be finite and positive, got {pitch!r}")
    z = cfg.flight_levels_m[0]
    origin = (0, 0)
    origin_point = _point(origin, z, radius)
    first_hops = []
    for direction in AXIAL_NEIGHBORS:
        target = direction
        volume = corridor_segment_volume(
            origin_point, 0.0, _point(target, z, radius), cfg.dt_s, cfg
        )
        first_hops.append((frozenset((origin, target)), volume, volume.aabb()))

    # A hop box reaches at most roughly one pitch plus one corridor width from
    # its source.  Twice that reach, plus two rings of rounding margin, bounds
    # every second source whose box could meet a first hop.
    # At axial distance n, lattice-centre Euclidean distance is at least
    # sqrt(3)/2 * n * pitch.  Convert the spatial reach through that lower
    # bound so very wide experimental corridors cannot escape the sweep.
    min_distance_per_ring = math.sqrt(3.0) * pitch / 2.0
    rings = max(
        3,
        math.ceil(2.0 * (pitch + cfg.corridor_width_m) / min_distance_per_ring) + 2,
    )
    for source in _cells_in_ring(origin, rings):
        source_point = _point(source, z, radius)
        for direction in AXIAL_NEIGHBORS:
            target = source[0] + direction[0], source[1] + direction[1]
            endpoints = frozenset((source, target))
            second = corridor_segment_volume(
                source_point, 0.0, _point(target, z, radius), cfg.dt_s, cfg
            )
            second_lo, second_hi = second.aabb()
            for first_endpoints, first, (first_lo, first_hi) in first_hops:
                if not first_endpoints.isdisjoint(endpoints):
                    continue
                if bool(np.any(first_hi < second_lo) or np.any(second_hi < first_lo)):
                    continue
                if volumes_conflict(first, second):
                    raise NotImplementedError(
                        "colgen visit rows require nonincident lattice edges to be conflict-free; "
                        f"corridor_width_m={cfg.corridor_width_m:g} and pitch={pitch:g} violate "
                        f"that invariant on {tuple(first_endpoints)} vs {tuple(endpoints)}. "
                        "Hex sizing from corridor geometry is tracked in GitHub issue #72."
                    )


def _cells_in_ring(origin: AxialCell, rings: int):
    """Yield the axial disk around ``origin`` in deterministic order."""

    oq, or_ = origin
    for q in range(oq - rings, oq + rings + 1):
        for r in range(or_ - rings, or_ + rings + 1):
            if hex_distance(origin, (q, r)) <= rings:
                yield q, r


@lru_cache(maxsize=64)
def derive_cell_window(cfg: SimConfig) -> CellWindow:
    """Measure inclusive row offsets for one centre crossing from ledger volumes.

    A visit at step zero is incident to an inbound hop ``[-dt, 0]`` and an
    outbound hop ``[0, dt]``.  Both are built with
    :func:`corridor_segment_volume`, so their actual buffered half-open time
    windows—not an assumed width—determine the periods in which the cell's
    reservation is present.  The returned tuple is inclusive.

    The default geometry yields ``(-1, 1)``: the pad is leading-only, so the
    inbound hop spans ``[-dt, buf)`` and the outbound ``[0, dt + buf)``,
    touching three periods.  Setting ``time_buffer_s=0`` yields ``(-1, 0)``,
    which is *not* centred on the visit — that shifted case is why deriving
    only a scalar width and reconstructing offsets later would be incorrect.  A
    representative FCL conflict scan is run once per cached configuration as an
    independent coverage check.

    Args:
        cfg: Simulation geometry and global-clock configuration.

    Returns:
        Inclusive ``(lo_offset, hi_offset)`` row offsets relative to a visit.

    Raises:
        ValueError: If the clock or time-buffer configuration is invalid.
        RuntimeError: If the measured footprint is non-contiguous or fails the
            representative ledger-conflict coverage check.
    """

    dt = float(cfg.dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt_s must be finite and positive, got {cfg.dt_s!r}")
    if not math.isfinite(cfg.time_buffer_s) or cfg.time_buffer_s < 0.0:
        raise ValueError(
            f"time_buffer_s must be finite and non-negative, got {cfg.time_buffer_s!r}"
        )

    radius = circumradius(cfg)
    z = cfg.cruise_level_m
    centre = _point((0, 0), z, radius)
    previous = _point((-1, 0), z, radius)
    following = _point((1, 0), z, radius)
    incident = (
        corridor_segment_volume(previous, -dt, centre, 0.0, cfg),
        corridor_segment_volume(centre, 0.0, following, dt, cfg),
    )
    touched = {
        period
        for volume in incident
        for period in _periods_overlapping(volume.t_start, volume.t_end, dt)
    }
    if not touched:
        raise RuntimeError("template visit produced no occupied cell periods")

    lo, hi = min(touched), max(touched)
    if touched != set(range(lo, hi + 1)):
        raise RuntimeError(f"template visit produced a non-contiguous footprint: {sorted(touched)}")

    offsets = (lo, hi)
    _cross_check_conflicts(cfg, offsets, radius)
    return offsets


def endpoint_claim_cells(point: Vec, radius: float, cfg: SimConfig) -> list[AxialCell]:
    """Return cells conservatively reached by an endpoint cylinder.

    A cylinder centred at ``point`` with radius ``radius`` can conflict with a
    transit box whose centreline reaches beyond its endpoint and laterally by
    the configured corridor width.  Candidate hexes are therefore tested
    against a disk enlarged by the larger of that width and another endpoint's
    hover radius, plus one hex circumradius.  The latter turns the centre test
    into a conservative hex-intersection test.

    Enumeration is bounded to a finite axial ring around ``cell(point)`` and
    the result is sorted lexicographically for deterministic column claims.

    Args:
        point: Endpoint position in local ENU metres; only x/y are used.
        radius: Cylinder radius in metres.
        cfg: Simulation geometry configuration.

    Returns:
        Deterministically ordered axial ``(q, r)`` cells.

    Raises:
        ValueError: If the point or radius is non-finite, or radius is negative.
    """

    px, py = float(point[0]), float(point[1])
    radius = float(radius)
    if not math.isfinite(px) or not math.isfinite(py):
        raise ValueError(f"endpoint coordinates must be finite, got ({px!r}, {py!r})")
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError(
            f"endpoint cylinder radius must be finite and non-negative, got {radius!r}"
        )

    hex_radius = circumradius(cfg)
    pitch = float(cfg.corridor_segment_len_m)
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError(f"corridor pitch must be finite and positive, got {pitch!r}")

    # At defaults both terms are 60 m, reproducing the plan's r_cyl + 60 + R
    # bound.  Taking the maximum keeps cylinder/cylinder coverage valid when a
    # caller increases the independent hover-radius knob.
    footprint_reach = max(float(cfg.corridor_width_m), float(cfg.effective_hover_radius_m))
    threshold = radius + footprint_reach + hex_radius
    ring_radius = math.ceil(threshold / pitch) + 1
    origin = enu_to_axial(px, py, hex_radius)

    cells: list[AxialCell] = []
    oq, or_ = origin
    for q in range(oq - ring_radius, oq + ring_radius + 1):
        for r in range(or_ - ring_radius, or_ + ring_radius + 1):
            cell = (q, r)
            if hex_distance(origin, cell) > ring_radius:
                continue
            cx, cy = hex_center(q, r, hex_radius)
            if math.hypot(float(cx) - px, float(cy) - py) < threshold:
                cells.append(cell)
    return cells


def endpoint_claim_steps(
    t0: float,
    t1: float,
    cfg: SimConfig,
    *,
    timing_steps: int = 0,
) -> range:
    """Return row periods conservatively touched by endpoint window ``[t0,t1)``.

    The row index denotes the physical period ``[j*dt, (j+1)*dt)``.  Rounding
    the endpoint interval outward makes it intersect the derived
    :func:`visit_rows` of every transit box whose ledger time window can overlap
    the endpoint.  A floating-point error bound based on the number of rebuilt
    lateral steps covers clock drift; direct callers retain a minimum 1 ns pad.

    Exact grid-boundary endpoints intentionally claim one adjacent period on
    each side because the rebuilt timestamp may drift either way.  This is a
    small conservative expansion, never an under-approximation.

    Args:
        t0: Inclusive endpoint-cylinder start time in seconds.
        t1: Exclusive endpoint-cylinder end time in seconds.
        cfg: Simulation clock configuration.
        timing_steps: Lateral steps accumulated by the corner-rebuild clock.

    Returns:
        A range of integer capacity-row periods.  A zero-duration interval
        returns an empty range.

    Raises:
        ValueError: If times are non-finite, reversed, or ``dt_s`` is invalid.
    """

    t0, t1 = float(t0), float(t1)
    dt = float(cfg.dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt_s must be finite and positive, got {cfg.dt_s!r}")
    if not math.isfinite(t0) or not math.isfinite(t1):
        raise ValueError(f"endpoint times must be finite, got ({t0!r}, {t1!r})")
    if t1 < t0:
        raise ValueError(f"endpoint interval must satisfy t0 <= t1, got ({t0!r}, {t1!r})")
    try:
        timing_steps = operator.index(timing_steps)
    except TypeError as exc:
        raise TypeError("timing_steps must be an integer") from exc
    if timing_steps < 0:
        raise ValueError("timing_steps must be non-negative")
    if t1 == t0:
        period = math.floor(t0 / dt)
        return range(period, period)

    pad = _endpoint_time_pad_s(t0, t1, dt, timing_steps)
    start = math.floor((t0 - pad) / dt)
    stop = math.ceil((t1 + pad) / dt)
    return range(start, stop)


def terminal_claim_steps(t0: float, t1: float, cfg: SimConfig) -> range:
    """Return the exact clock periods overlapped by terminal dwell ``[t0, t1)``.

    Terminal rows model pad occupancy rather than a conservative geometric
    envelope.  They must therefore use the same half-open interval semantics as
    :class:`~freespace_sim.planner.terminal_capacity.TerminalCapacity`: period
    ``j`` is claimed exactly when ``[j*dt, (j+1)*dt)`` overlaps the dwell.  In
    particular, a dwell ending on a grid boundary does not claim the following
    period, and one starting there does not claim the preceding period.

    Unlike :func:`endpoint_claim_steps`, this helper deliberately applies no
    floating-point padding.  Customer endpoint cylinders need that outward
    geometric cover; terminal capacity is an exact scheduling resource.
    """

    t0, t1 = float(t0), float(t1)
    dt = float(cfg.dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt_s must be finite and positive, got {cfg.dt_s!r}")
    if not math.isfinite(t0) or not math.isfinite(t1):
        raise ValueError(f"terminal dwell times must be finite, got ({t0!r}, {t1!r})")
    if t1 < t0:
        raise ValueError(f"terminal dwell interval must satisfy t0 <= t1, got ({t0!r}, {t1!r})")
    if t1 == t0:
        period = math.floor(t0 / dt)
        return range(period, period)
    return _periods_overlapping(t0, t1, dt)


__all__ = [
    "AxialCell",
    "CellWindow",
    "derive_cell_window",
    "endpoint_claim_cells",
    "endpoint_claim_steps",
    "terminal_claim_steps",
    "validate_edge_locality",
    "visit_rows",
]
