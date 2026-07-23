"""4D volumes and the two ASTM operational-intent builders.

A `Volume4D` (ASTM §3.2.2) is a 3D shape + a time window. A **corridor** (trajectory-based intent,
§4.3.5) is a chain of oriented boxes — one per timestep — that overlap in space and time. A
**hover reservation** (area-based intent, §4.3.5) is a single vertical cylinder covering the
takeoff/landing climb/descent.
"""

from __future__ import annotations

import math
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

    **This is the contract between the planners and the ledger.** A planner (A* per edge, or the
    straight-line planner via :func:`build_corridor`) checks *this exact* box against the ledger and
    commits *this exact* box — there is no separate post-hoc inflation that could reintroduce a
    conflict. The box is purely segment-local (depends only on its own endpoints + cfg), which is
    what makes per-edge checking equivalent to whole-corridor checking.

    Geometry: configured width/height, extended longitudinally at each end so consecutive boxes overlap
    (ASTM §4.3.5 contiguity); time window buffered by ``time_buffer_s`` on both sides so neighbours
    overlap in time too. The extension is **anisotropic** — half the box's footprint *in the travel
    direction*: ``corridor_width/2`` in the horizontal plane, ``corridor_height/2`` in the vertical. This
    matters for a mid-route layer change (a fixed-xy segment moving in z): a flat ``corridor_width/2``
    would balloon the box in z past the levels it traverses (and above the ceiling); the vertical term
    keeps its z-extent at ``[z0, z1] ± corridor_height/2`` — the drone's real vertical footprint.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    u = d / length if length > 1e-9 else np.array([1.0, 0.0, 0.0])
    # half the cross-section along travel: width when horizontal, height when vertical (== width/2 for a
    # level cruise/exit box → those are unchanged; == height/2 for a pure climb → no z overshoot).
    ext = 0.5 * math.hypot(cfg.corridor_width_m * math.hypot(u[0], u[1]), cfg.corridor_height_m * u[2])
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
    Far cruise boxes stay untagged, so foreign/same-hub overflight still deconflicts.

    The xy point-to-segment distance is computed with scalars (norm via ``math.sqrt``, dot as a scalar sum)
    — bit-for-bit identical to the numpy form but without its per-call ufunc dispatch, since this runs once
    per corridor sub-box during every rebuild (issue #30 lever #8; same idiom as ``geometry.segment_frame``
    and ``astar.h_air``). See ``tests/test_volumes.py`` for the frozen-numpy byte-identity oracle."""
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    cx, cy = float(center[0]), float(center[1])
    segx, segy = bx - ax, by - ay
    length = math.sqrt(segx * segx + segy * segy)         # == np.linalg.norm(seg) on the xy pair
    ux, uy = (segx / length, segy / length) if length > 1e-9 else (1.0, 0.0)
    ext = cfg.corridor_width_m / 2.0
    p0x, p0y = ax - ux * ext, ay - uy * ext               # box centerline incl. longitudinal extension
    p1x, p1y = bx + ux * ext, by + uy * ext
    abx, aby = p1x - p0x, p1y - p0y
    t = ((cx - p0x) * abx + (cy - p0y) * aby) / max(abx * abx + aby * aby, 1e-12)
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t         # == np.clip(t, 0.0, 1.0)
    dx, dy = cx - (p0x + t * abx), cy - (p0y + t * aby)
    d = math.sqrt(dx * dx + dy * dy)                      # distance center → extended centerline
    return d < radius + cfg.corridor_width_m / 2.0        # + box half-width


def build_reservation_from_corners(
    corners: list[Vec], origin: Vec, dest: Vec, t_depart: float, g_delay: float, cfg: SimConfig,
    *, origin_term=None, dest_term=None,
) -> tuple[list[Volume4D], list[TimedPoint], float, float]:
    """Resample a corner polyline to ≤segment-length boxes, time at nominal speed, assemble.

    Shared by the MILP planner and the shortcut refiner so they all emit the
    *same* contract-preserving boxes (checked == committed). When ``origin_term``/``dest_term`` are
    given, the hub **hover column** is tagged shared (sized to the terminal's radius) AND **every corridor
    box that reaches into that column** (``segment_overlaps_column`` — not just the first/last) is tagged
    with the hub, so the column-involved exemption lets the near-hub corridor pass through the shared
    column; every box clear of the column stays strict (untagged). Returns (volumes, centerline, horiz, dz).
    """
    origin_term, dest_term = as_terminal(origin_term), as_terminal(dest_term)
    # the corner z is the source of truth for climb timing: cruise starts after the climb to the
    # FIRST corner's altitude (its flight level), not a fixed preferred-level climb.
    z_takeoff = float(np.asarray(corners[0], float)[2])
    z_land = float(np.asarray(corners[-1], float)[2])
    t = t_depart + g_delay + cfg.climb_time_to(z_takeoff)
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
        # Single-box hub→hub corridor: edges[-1] IS edges[0]; tag dest only when distinct so it can't
        # clobber the origin tag above (mirrors astar._build).
        if dest_term is not None and not (origin_term is not None and len(edges) == 1):
            edges[-1] = replace(edges[-1], terminal_id=dest_term.id)
    volumes = [
        hover_reservation(origin, t_depart + g_delay, cfg,
                          terminal_id=origin_term.id if origin_term else None,
                          radius=terminal_radius(origin_term, cfg) if origin_term else None,
                          climb_time_s=cfg.climb_time_to(z_takeoff)),
        *edges,
        hover_reservation(dest, t, cfg,
                          terminal_id=dest_term.id if dest_term else None,
                          radius=terminal_radius(dest_term, cfg) if dest_term else None,
                          climb_time_s=cfg.climb_time_to(z_land)),
    ]
    return volumes, centerline, cum_horiz, cum_dz


def hover_reservation(center: Vec, t0: float, cfg: SimConfig, *, terminal_id: Hashable = None,
                      radius: float | None = None, z_hi: float | None = None,
                      climb_time_s: float | None = None) -> Volume4D:
    """A vertical hover cylinder at ``center`` (ASTM area-based intent, §4.3.5).

    ``radius`` (default ``effective_hover_radius_m``) lets a multi-pad vertiport size its shared column
    bigger than a single pad. Altitude band [ground, ``z_hi``] — ``z_hi`` defaults to
    ``airspace_ceiling_m`` so the column spans the full regulated tube (a vertiport owns its vertical
    column of regulated airspace). Active for ``hover_time_s + climb_time_s`` from ``t0``; pass
    ``climb_time_s`` (e.g. :meth:`SimConfig.climb_time_to` of the flight's cruise level) to size the
    window to the actual climb instead of the preferred-level default. When ``terminal_id`` is set this
    cylinder is a shared terminal column — transparent to its own hub's flights, opaque to everyone else
    (see :func:`conflict.volumes_conflict`).
    """
    center = np.asarray(center, float)
    z_hi = cfg.airspace_ceiling_m if z_hi is None else float(z_hi)
    ct = cfg.climb_time_s if climb_time_s is None else float(climb_time_s)
    spec = CylinderSpec(
        cx=float(center[0]),
        cy=float(center[1]),
        radius=cfg.effective_hover_radius_m if radius is None else float(radius),
        z_lo=cfg.ground_level_m,
        z_hi=z_hi,
    )
    return Volume4D(spec, t0, t0 + cfg.hover_time_s + ct, terminal_id=terminal_id)


# Effectively-unbounded but FINITE t_end for the time-invariant terminal wall (see
# permanent_terminal_reservation). Huge (~31000 yr) so no committed corridor — incl. a late-departing return
# — can outlast the wall. Finite (not inf) belt-and-suspenders; the real safety is that a static wall is never
# committed, so it never reaches step-range code (where a huge t_end would HANG the range(), not just overflow).
_WALL_T_END_S = 1e12


def permanent_terminal_reservation(center: Vec, term, cfg: SimConfig) -> Volume4D:
    """A hub's whole-horizon terminal-airspace reservation — the ledger volume that makes an
    ``cfg.terminal_airspace_always_active`` wall a first-class part of the committed airspace (visible to
    ``ledger.any_conflict`` / ``verify`` / the ledger-only refiners) instead of an off-ledger occupancy
    side-structure.

    Spans the full ``[ground, ceiling]`` tube for the whole horizon and is tagged with ``terminal_id`` so
    the column-involved exemption in :func:`conflict.volumes_conflict` keeps it transparent to its own
    hub's flights while walling foreign cruise.

    **Radius = ``terminal_radius`` — the reserved column, exactly what the per-flight dwell column reserves
    (:func:`hover_reservation` in ``build_reservation_from_corners`` / ``AStarPlanner._build``).** The ledger
    records only the *safety-critical reserved volume* (the hover column where drones actually are); it does
    NOT include the ``+corridor_width/2`` of ``exit_radius`` (that is exit-LANE geometry — where lanes start
    flush with the column edge — a routing/lane concern, not a reservation) nor the wider ``terminal_cells``
    flood-fill (A*'s discrete keep-out, for search margin). So the permanent wall is byte-identical to the
    transient dwell column, just permanent — the "active ⟺ on the ledger" model applied to the *same* volume
    (built by reusing :func:`hover_reservation`, so the two cannot drift). Because ``terminal_radius ⊂
    terminal_cells``, any corridor A* routes around ``terminal_cells`` also clears this column with margin (no
    spurious commit-time denials).

    **Time-invariant — active for ALL time, mirroring the occupancy routing wall.** The A* occupancy
    ``static_col`` blocks these cells at EVERY queried step (it has no time dimension), so the ledger wall
    must too. Any finite, ``cfg``-derived ``t_end`` has a hole: a committed corridor can land after it — most
    sharply a return flight departing at ``t_request + est_trip + turnaround_s > horizon_s`` (``turnaround_s``
    is a demand-model field, invisible here) — and then a foreign crossing in that window would escape
    ``any_conflict`` / ``verify``. So ``t_end`` is a large sentinel (:data:`_WALL_T_END_S`): effectively
    unbounded, but FINITE (not ``inf``) as belt-and-suspenders. It is safe because a static wall is never
    committed, so it never reaches the step-range/bucketing arithmetic (``ledger._steps`` /
    ``hexgrid.rasterize_volume``); it surfaces only via ``ledger.conflicts`` (the ``-1`` sentinel), where the
    sole arithmetic reader — ``straight``'s jump-to-gap ``min(cv.t_end)`` — is refused under always-active."""
    term = as_terminal(term)
    return replace(
        hover_reservation(center, 0.0, cfg, terminal_id=term.id, radius=terminal_radius(term, cfg)),
        t_end=_WALL_T_END_S)
