"""4D volumes and the two ASTM operational-intent builders.

A `Volume4D` (ASTM §3.2.2) is a 3D shape + a time window. A **corridor** (trajectory-based intent,
§4.3.5) is a chain of oriented boxes — one per timestep — that overlap in space and time. A
**hover reservation** (area-based intent, §4.3.5) is a single vertical cylinder covering the
takeoff/landing climb/descent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np

from .config import SimConfig
from .geometry import BoxSpec, CylinderSpec, box_from_segment
from .types import TimedPoint, Vec, as_terminal

ShapeSpec = BoxSpec | CylinderSpec


@dataclass(frozen=True)
class Volume4D:
    """A 3D shape (Box or Cylinder) active over the half-open time window [t_start, t_end).

    ``terminal_id`` optionally marks this volume as part of a *shared vertiport terminal* — the
    airspace column over a multi-pad hub. Two volumes with the same non-None ``terminal_id`` do NOT
    conflict (they're the same vertiport's flights sharing its terminal; pad capacity is enforced
    separately). ``None`` (the default) is ordinary airspace and deconflicts normally — so existing
    flights are unaffected. See :func:`conflict.volumes_conflict`.
    """

    shape: ShapeSpec
    t_start: float
    t_end: float
    terminal_id: Hashable = None

    def to_fcl(self):
        return self.shape.to_fcl()

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.shape.aabb()

    def time_overlaps(self, other: "Volume4D") -> bool:
        return self.t_start < other.t_end and other.t_start < self.t_end

    @property
    def z_range(self) -> tuple[float, float]:
        lo, hi = self.shape.aabb()
        return float(lo[2]), float(hi[2])


def corridor_segment_volume(
    p0: Vec, t0: float, p1: Vec, t1: float, cfg: SimConfig, *, terminal_id: Hashable = None
) -> Volume4D:
    """Build the single corridor box for one segment (p0,t0)→(p1,t1).

    **This is the contract between the planners and the ledger.** A planner (RRT* per edge, or the
    straight-line planner via :func:`build_corridor`) checks *this exact* box against the ledger and
    commits *this exact* box — there is no separate post-hoc inflation that could reintroduce a
    conflict. The box is purely segment-local (depends only on its own endpoints + cfg), which is
    what makes per-edge checking equivalent to whole-corridor checking.

    Geometry: configured width/height, extended longitudinally by half the corridor width at each
    end so consecutive boxes overlap (ASTM §4.3.5 contiguity); time window buffered by
    ``time_buffer_s`` on both sides so neighbours overlap in time too.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    u = d / length if length > 1e-9 else np.array([1.0, 0.0, 0.0])
    ext = cfg.corridor_width_m / 2.0
    a = p0 - u * ext        # extend behind the start
    b = p1 + u * ext        # and beyond the end → overlap with neighbours
    spec = box_from_segment(a, b, cfg.corridor_width_m, cfg.corridor_height_m)
    return Volume4D(spec, t0 - cfg.time_buffer_s, t1 + cfg.time_buffer_s, terminal_id=terminal_id)


def build_corridor(centerline: list[TimedPoint], cfg: SimConfig) -> list[Volume4D]:
    """Chop a timed 3D polyline into one oriented-box Volume4D per segment (ASTM §4.3.5).

    A thin loop over :func:`corridor_segment_volume` — so the whole-path corridor is exactly the
    concatenation of the per-edge boxes a planner checks during search.
    """
    return [
        corridor_segment_volume(p0, t0, p1, t1, cfg)
        for (p0, t0), (p1, t1) in zip(centerline, centerline[1:])
    ]


def terminal_radius(term, cfg: SimConfig) -> float:
    """A terminal's column radius — its own ``radius`` if set, else the hover footprint."""
    return term.radius if term.radius is not None else cfg.effective_hover_radius_m


def build_reservation_from_corners(
    corners: list[Vec], origin: Vec, dest: Vec, t_depart: float, g_delay: float, cfg: SimConfig,
    *, origin_term=None, dest_term=None,
) -> tuple[list[Volume4D], list[TimedPoint], float, float]:
    """Resample a corner polyline to ≤segment-length boxes, time at nominal speed, assemble.

    Shared by the RRT* smoother, the NLP/MILP planners, and the shortcut refiner so they all emit the
    *same* contract-preserving boxes (checked == committed). When ``origin_term``/``dest_term`` are
    given, only the hub **hover column** is tagged shared (and sized to the terminal's radius) — every
    corridor box stays strict, so it deconflicts against everything, same-hub flights included, even the
    bit that dips into the terminal by ``corridor_overlap``. Returns (volumes, centerline, horiz, dz).
    """
    origin_term, dest_term = as_terminal(origin_term), as_terminal(dest_term)
    t = t_depart + g_delay + cfg.climb_time_s
    centerline: list[TimedPoint] = [(np.asarray(corners[0], float).copy(), t)]
    edges: list[Volume4D] = []
    cum_horiz = cum_dz = 0.0
    seg = cfg.corridor_segment_len_m
    for a, b in zip(corners, corners[1:]):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        d = b - a
        total = float(np.linalg.norm(d))
        nsub = max(1, int(np.ceil(total / seg)))
        for k in range(1, nsub + 1):
            sa = a + (k - 1) / nsub * d
            sb = a + k / nsub * d
            horiz = float(np.linalg.norm((sb - sa)[:2]))
            dz = abs(float(sb[2] - sa[2]))
            t_next = t + max(horiz / cfg.nominal_speed_mps, dz / cfg.climb_rate_mps, 1e-3)
            edges.append(corridor_segment_volume(sa, t, sb, t_next, cfg))   # corridor boxes stay strict
            centerline.append((sb.copy(), t_next))
            t = t_next
            cum_horiz += horiz
            cum_dz += dz
    volumes = [
        hover_reservation(origin, t_depart + g_delay, cfg,
                          terminal_id=origin_term.id if origin_term else None,
                          radius=origin_term.radius if origin_term else None),
        *edges,
        hover_reservation(dest, t, cfg,
                          terminal_id=dest_term.id if dest_term else None,
                          radius=dest_term.radius if dest_term else None),
    ]
    return volumes, centerline, cum_horiz, cum_dz


def hover_reservation(center: Vec, t0: float, cfg: SimConfig, *, terminal_id: Hashable = None,
                      radius: float | None = None) -> Volume4D:
    """A vertical hover cylinder at ``center`` (ASTM area-based intent, §4.3.5).

    ``radius`` (default ``effective_hover_radius_m``) lets a multi-pad vertiport size its shared column
    bigger than a single pad. Altitude band [ground, cruise] so it covers the climb or descent; active
    for ``hover_time_s + climb_time_s`` from ``t0``. When ``terminal_id`` is set this cylinder is a
    shared terminal column — transparent to its own hub's flights, opaque to everyone else (see
    :func:`conflict.volumes_conflict`).
    """
    center = np.asarray(center, float)
    spec = CylinderSpec(
        cx=float(center[0]),
        cy=float(center[1]),
        radius=cfg.effective_hover_radius_m if radius is None else float(radius),
        z_lo=cfg.ground_level_m,
        z_hi=cfg.cruise_level_m,
    )
    return Volume4D(spec, t0, t0 + cfg.hover_time_s + cfg.climb_time_s, terminal_id=terminal_id)
