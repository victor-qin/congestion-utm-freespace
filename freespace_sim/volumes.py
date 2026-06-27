"""4D volumes and the two ASTM operational-intent builders.

A `Volume4D` (ASTM §3.2.2) is a 3D shape + a time window. A **corridor** (trajectory-based intent,
§4.3.5) is a chain of oriented boxes — one per timestep — that overlap in space and time. A
**hover reservation** (area-based intent, §4.3.5) is a single vertical cylinder covering the
takeoff/landing climb/descent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    p0: Vec, t0: float, p1: Vec, t1: float, cfg: SimConfig, *, terminal_id: Hashable = None,
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
    """A terminal's column radius — its own ``radius`` if set, else ``cfg.terminal_radius_m`` (90 m)."""
    return term.radius if term.radius is not None else cfg.terminal_radius_m


def exit_radius(term, cfg: SimConfig) -> float:
    """A hub's exit-lane inner edge — flush with the column edge by default (``corridor_overlap = 0``).

    Inner edge = R − overlap, so the reserved lane/fold starts FLUSH with the column edge; the exit-lane box
    is tagged with the hub and the column-involved exemption (:func:`conflict.volumes_conflict`) makes it
    transparent to same-hub COLUMNS, while two same-hub *corridor* boxes still contend (box↔box stays
    strict), so divergent lanes need the column wide enough not to crowd (``cfg.terminal_radius_m`` 90 m
    default). ``overlap > 0`` penetrates the column; ``< 0`` leaves a clearance gap. (Issue #10.)

    The single source of truth for the fold/lane radius — used by the A* head/tail fold
    (:func:`planner.astar._fold_path`, which drives both the commit and the landing gate) and
    :meth:`planner.terminal_capacity.TerminalCapacity.exit_clear` — so the gate, the commit, and the
    exit-lane check all root the lane at the same edge and cannot drift."""
    ov = term.corridor_overlap if term.corridor_overlap is not None else 0.0
    return terminal_radius(term, cfg) + cfg.corridor_width_m / 2.0 - ov


def segment_overlaps_column(a, b, center, radius: float, cfg: SimConfig) -> bool:
    """Does the corridor box for segment ``a→b`` reach into the disk of ``radius`` at ``center`` (xy)?

    Accounts for the box geometry corridor_segment_volume builds: the centerline is extended by half
    the corridor width at each end, and the box has a half-width of ``corridor_width/2``. So the box
    overlaps the column iff the distance from ``center`` to the *extended* centerline is below
    ``radius + corridor_width/2``.

    Used to tag EVERY near-hub box that reaches into a flight's own column — not just box[0]/box[-1].
    The count of such boxes is geometry-dependent (radius × exit angle), so a fixed "tag the first N"
    rule is unsound (e.g. a 500 m column can need boxes [1] and [2] tagged); this geometric test scales.
    Far cruise boxes stay untagged, so foreign/same-hub overflight still deconflicts."""
    a2 = np.asarray(a, float)[:2]
    b2 = np.asarray(b, float)[:2]
    c2 = np.asarray(center, float)[:2]
    seg = b2 - a2
    L = float(np.linalg.norm(seg))
    u = seg / L if L > 1e-9 else np.array([1.0, 0.0])
    ext = cfg.corridor_width_m / 2.0
    p0, p1 = a2 - u * ext, b2 + u * ext              # box centerline incl. longitudinal extension
    ab = p1 - p0
    t = float(np.clip((c2 - p0).dot(ab) / max(ab.dot(ab), 1e-12), 0.0, 1.0))
    d = float(np.linalg.norm(c2 - (p0 + t * ab)))    # distance center → extended centerline
    return d < radius + cfg.corridor_width_m / 2.0   # + box half-width


def build_reservation_from_corners(
    corners: list[Vec], origin: Vec, dest: Vec, t_depart: float, g_delay: float, cfg: SimConfig,
    *, origin_term=None, dest_term=None,
) -> tuple[list[Volume4D], list[TimedPoint], float, float]:
    """Resample a corner polyline to ≤segment-length boxes, time at nominal speed, assemble.

    Shared by the RRT* smoother, the NLP/MILP planners, and the shortcut refiner so they all emit the
    *same* contract-preserving boxes (checked == committed). When ``origin_term``/``dest_term`` are
    given, the hub **hover column** is tagged shared (sized to the terminal's radius) AND **every corridor
    box that reaches into that column** (``segment_overlaps_column`` — not just the first/last) is tagged
    with the hub, so the column-involved exemption lets the near-hub corridor pass through the shared
    column; every box clear of the column stays strict (untagged). Returns (volumes, centerline, horiz, dz).
    """
    origin_term, dest_term = as_terminal(origin_term), as_terminal(dest_term)
    t = t_depart + g_delay + cfg.climb_time_s
    centerline: list[TimedPoint] = [(np.asarray(corners[0], float).copy(), t)]
    edges: list[Volume4D] = []
    cum_horiz = cum_dz = 0.0
    seg = cfg.corridor_segment_len_m
    o_xy = np.asarray(origin, float)[:2] if origin_term is not None else None
    d_xy = np.asarray(dest, float)[:2] if dest_term is not None else None
    o_r = terminal_radius(origin_term, cfg) if origin_term is not None else 0.0
    d_r = terminal_radius(dest_term, cfg) if dest_term is not None else 0.0
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
            # Tag EVERY box reaching into its hub's own column (not just first/last), so a near-hub
            # cruise box grazing the shared column is column-exempt rather than a CONFLICT_FILED. See
            # segment_overlaps_column; mirrors astar._build's per-box tagging.
            tid = (origin_term.id if o_xy is not None and segment_overlaps_column(sa, sb, o_xy, o_r, cfg)
                   else dest_term.id if d_xy is not None and segment_overlaps_column(sa, sb, d_xy, d_r, cfg)
                   else None)
            edges.append(corridor_segment_volume(sa, t, sb, t_next, cfg, terminal_id=tid))
            centerline.append((sb.copy(), t_next))
            t = t_next
            cum_horiz += horiz
            cum_dz += dz
    if cfg.fixed_exit_lanes and edges and (origin_term is not None or dest_term is not None):
        # Fixed exit lanes: force the hub tag on the first/last (boundary-cell) box. It leaves from /
        # arrives at the column edge and can graze the shared column; an untagged box grazing it would
        # conflict at commit (different tid) — the cruise-box-clip. ``segment_overlaps_column`` tags
        # interior boxes; this guarantees the boundary box too (mirrors ``astar._build``).
        if origin_term is not None:
            edges[0] = replace(edges[0], terminal_id=origin_term.id)
        if dest_term is not None:
            edges[-1] = replace(edges[-1], terminal_id=dest_term.id)
    volumes = [
        hover_reservation(origin, t_depart + g_delay, cfg,
                          terminal_id=origin_term.id if origin_term else None,
                          radius=terminal_radius(origin_term, cfg) if origin_term else None),
        *edges,
        hover_reservation(dest, t, cfg,
                          terminal_id=dest_term.id if dest_term else None,
                          radius=terminal_radius(dest_term, cfg) if dest_term else None),
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
