"""Simulator-owned reservation geometry for the finite-waypoint any-angle prototype.

Spatial rows cover the actual filed boxes and customer cylinders. Terminal cylinders
instead use pad rows only when the caller enforces their matching permanent walls;
tagged corridor boxes still reserve ordinary cell rows.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache
from typing import AbstractSet, Hashable

from shapely.geometry import MultiPoint, Point, Polygon

from ...config import SimConfig
from ...geometry import BoxSpec, CylinderSpec
from ...types import FlightRequest, OperationalIntent
from ...volumes import Volume4D, column_dwell_s
from ..route_domain import RowIndex, RowKey
from .model import Route
from ..route_geometry import route_to_corners, route_to_intent
from ..reservation_windows import terminal_claim_steps
from .timing import edge_end_times


@lru_cache(maxsize=32768)
def _hex_polygon(q: int, r: int, pitch: float) -> Polygon:
    """Construct the closed hex tile used by the capacity rows."""
    radius = pitch / math.sqrt(3.0)
    x = pitch * (q + r / 2.0)
    y = pitch * math.sqrt(3.0) * r / 2.0
    return Polygon([
        (x + radius * math.cos(math.radians(30 + 60 * index)),
         y + radius * math.sin(math.radians(30 + 60 * index)))
        for index in range(6)
    ])


@lru_cache(maxsize=32768)
def volume_spatial_cells(
    shape: BoxSpec | CylinderSpec,
    cfg: SimConfig,
) -> frozenset[tuple[int, int, int]]:
    """Cover a static volume footprint with all intersected hex/level cells.

    Parameters
    ------------
    - shape (BoxSpec | CylinderSpec): filed spatial shape, independent of its time window.
    - cfg (SimConfig): hex pitch and existing cruise-level bands.

    Return
    --------
    - output (frozenset[tuple[int, int, int]]): closed intersecting cells as (q, r, level).
      Boundary contact is included with a small conservative numerical guard.
    """
    pitch = float(cfg.corridor_segment_len_m)
    if not math.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("hex pitch must be finite and positive")
    xmin, ymin, zmin, xmax, ymax, zmax = shape.flat_aabb()
    if not all(math.isfinite(value) for value in (xmin, ymin, zmin, xmax, ymax, zmax)):
        raise ValueError("reservation shape bounds must be finite")
    levels = tuple(
        level for level, height in enumerate(cfg.flight_levels_m)
        if height + cfg.corridor_height_m / 2.0 >= zmin - 1e-8
        and height - cfg.corridor_height_m / 2.0 <= zmax + 1e-8
    )
    if isinstance(shape, CylinderSpec):
        footprint = Point(shape.cx, shape.cy)
        reach = shape.radius
    elif isinstance(shape, BoxSpec):
        vertices = []
        for signs in itertools.product((-1, 1), repeat=3):
            offsets = tuple(signs[i] * shape.extents[i] / 2.0 for i in range(3))
            vertices.append(tuple(
                shape.center[i] + sum(shape.rot[3 * i + j] * offsets[j] for j in range(3))
                for i in range(2)
            ))
        footprint = MultiPoint(vertices).convex_hull
        reach = 0.0
    else:
        raise TypeError(f"unsupported reservation shape {type(shape).__name__}")

    radius = pitch / math.sqrt(3.0)
    row_height = pitch * math.sqrt(3.0) / 2.0
    cells = set()
    for r in range(math.floor((ymin - radius) / row_height) - 1,
                   math.ceil((ymax + radius) / row_height) + 2):
        for q in range(math.floor((xmin - radius) / pitch - r / 2.0) - 1,
                       math.ceil((xmax + radius) / pitch - r / 2.0) + 2):
            if _hex_polygon(q, r, pitch).distance(footprint) <= reach + 1e-8:
                cells.update((q, r, level) for level in levels)
    return frozenset(cells)


@lru_cache(maxsize=32768)
def _cell_claims(
    shape: BoxSpec | CylinderSpec, cfg: SimConfig, first: int, stop: int,
) -> frozenset[RowKey]:
    """Reuse immutable claims when exact geometry and overlapping period indices agree."""
    return frozenset(
        RowKey.cell(q, r, level, step)
        for q, r, level in volume_spatial_cells(shape, cfg)
        for step in range(first, stop)
    )


def volume_claims(
    volume: Volume4D,
    cfg: SimConfig,
    *,
    row_index: RowIndex,
    protected_terminal_ids: AbstractSet[Hashable] = frozenset(),
) -> frozenset[RowKey]:
    """Reserve a filed volume's complete footprint and exact half-open time periods.

    Parameters
    ------------
    - volume (Volume4D): one actual simulator reservation volume.
    - cfg (SimConfig): spatial and temporal capacity discretization.
    - row_index (RowIndex): terminal capacity authority; tagged cylinders must be registered.
    - protected_terminal_ids (AbstractSet[Hashable]): terminals whose permanent walls are
      enforced against every route and fixed commitment by the caller. Only those cylinders
      may omit cell claims, preserving the simulator's same-terminal exemption.

    Return
    --------
    - output (frozenset[RowKey]): unique capacity coefficients. Raises ValueError for an
      unprotected terminal cylinder and KeyError for an unregistered terminal capacity.
    """
    if isinstance(volume.shape, CylinderSpec) and volume.terminal_id is not None:
        if volume.terminal_id not in protected_terminal_ids:
            raise ValueError("terminal cylinder claims require matching permanent-wall protection")
        rows = frozenset(
            RowKey.term(volume.terminal_id, step)
            for step in terminal_claim_steps(volume.t_start, volume.t_end, cfg)
        )
        # Validate registration even for an empty interval.
        row_index.cap(RowKey.term(volume.terminal_id, 0))
        return rows
    periods = terminal_claim_steps(volume.t_start, volume.t_end, cfg)
    return _cell_claims(volume.shape, cfg, periods.start, periods.stop)


def reservation_claims(
    volumes: Sequence[Volume4D],
    cfg: SimConfig,
    *,
    row_index: RowIndex,
    protected_terminal_ids: AbstractSet[Hashable] = frozenset(),
) -> frozenset[RowKey]:
    """Union complete filed-volume claims so each route pays for a row only once.

    Parameters
    ------------
    - volumes (Sequence[Volume4D]): exact simulator reservation to offer to the master.
    - cfg (SimConfig): spatial and temporal capacity discretization.
    - row_index (RowIndex): registered terminal-capacity authority.
    - protected_terminal_ids (AbstractSet[Hashable]): permanent-wall protection guaranteed
      by the caller, with the same precondition as volume_claims.

    Return
    --------
    - output (frozenset[RowKey]): every covered cell/time or terminal/time row once.
    """
    return frozenset(
        row for volume in volumes
        for row in volume_claims(
            volume, cfg, row_index=row_index, protected_terminal_ids=protected_terminal_ids,
        )
    )


def anyangle_to_intent(
    column: Route,
    request: FlightRequest,
    cfg: SimConfig,
    *,
    arrival_snap_s: float = 0.0,
    speed_fraction: float = 0.05,
) -> OperationalIntent:
    """File finite-waypoint segments with optional bounded-speed arrival snapping.

    Parameters
    ------------
    - column (Route): route whose integer cells identify candidate bend locations; adjacent
      cells are not required. Departure step, lane choices and level retain their old meaning.
    - request (FlightRequest): exact endpoints and terminal metadata to attach to the route.
    - cfg (SimConfig): unchanged departure, terminal, buffer, detour and cost rules.
    - arrival_snap_s (float): candidate-leg arrival-grid period; zero retains nominal timing.
    - speed_fraction (float): fractional horizontal-speed allowance used by arrival snapping.

    Return
    --------
    - output (OperationalIntent): translated accepted or budget-rejected intent. The caller
      must still check its volumes against permanent walls and fixed commitments.
    """
    intent = route_to_intent(column, request, cfg, _any_angle=True)
    if not intent.accepted or arrival_snap_s == 0.0:
        return intent
    corners, _, _, stamp, _, destination_terminal = route_to_corners(
        column, request, cfg, _any_angle=True,
    )
    volumes, centerline = intent.volumes, intent.centerline
    assert volumes is not None and centerline is not None
    retimed_volumes = [volumes[0]]
    retimed_centerline = [centerline[0]]
    part_index = 1
    for a, b in zip(corners, corners[1:]):
        ax, ay = float(a[0]), float(a[1])
        dx, dy = float(b[0]) - ax, float(b[1]) - ay
        total = math.sqrt(dx * dx + dy * dy)
        nsub = max(1, math.ceil(total / cfg.corridor_segment_len_m))
        durations = []
        for k in range(1, nsub + 1):
            f0, f1 = (k - 1) / nsub, k / nsub
            hx = (ax + f1 * dx) - (ax + f0 * dx)
            hy = (ay + f1 * dy) - (ay + f0 * dy)
            durations.append(max(math.sqrt(hx * hx + hy * hy) / cfg.nominal_speed_mps, 1e-3))
        for end in edge_end_times(stamp, durations, arrival_snap_s, speed_fraction):
            retimed_volumes.append(replace(
                volumes[part_index],
                t_start=stamp - cfg.time_buffer_s,
                t_end=end + cfg.time_buffer_s,
            ))
            retimed_centerline.append((centerline[part_index][0], end))
            part_index += 1
            stamp = end
    assert part_index == len(centerline) == len(volumes) - 1
    destination = volumes[-1]
    retimed_volumes.append(replace(
        destination,
        t_start=stamp,
        t_end=stamp + cfg.hover_time_s + column_dwell_s(
            request.dest, destination_terminal, cfg, cfg.flight_levels_m[column.level],
        ),
    ))
    return replace(intent, volumes=retimed_volumes, centerline=retimed_centerline)
