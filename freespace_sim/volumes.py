"""4D volumes and the two ASTM operational-intent builders.

A ``Volume4D`` (ASTM §3.2.2) is a 3D shape + a time window. A corridor (trajectory-based intent,
§4.3.5) is a chain of oriented boxes — one per timestep — that overlap in space and time. A hover
reservation (area-based intent, §4.3.5) is a single vertical cylinder covering the takeoff/landing
climb/descent.
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
        """Delegate to the underlying shape's ``to_fcl`` serialization."""
        return self.shape.to_fcl()

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """The underlying shape's axis-aligned bounding box (delegates to ``shape.aabb``)."""
        return self.shape.aabb()

    def flat_aabb(self) -> tuple[float, float, float, float, float, float]:
        """World AABB as six plain floats ``(xmin, ymin, zmin, xmax, ymax, zmax)`` — allocation-free twin
        of :meth:`aabb` for the scalar broadphase (``ledger._flat_aabb`` / ``terminal_capacity``)."""
        return self.shape.flat_aabb()

    def time_overlaps(self, other: "Volume4D") -> bool:
        """True if this volume's half-open time window overlaps ``other``'s."""
        return self.t_start < other.t_end and other.t_start < self.t_end

    @property
    def z_range(self) -> tuple[float, float]:
        """The volume's vertical extent ``(z_min, z_max)`` in metres."""
        lo, hi = self.shape.aabb()
        return float(lo[2]), float(hi[2])


def corridor_segment_volume(
    p0: Vec, t0: float, p1: Vec, t1: float, cfg: SimConfig, *, terminal_id: Hashable = None,
) -> Volume4D:
    """Build the single corridor box for one segment ``(p0, t0)`` → ``(p1, t1)``.

    This is the contract between the planners and the ledger: a planner (A* per edge, or the
    straight-line planner via :func:`build_corridor`) checks this exact box against the ledger and
    commits this exact box — there is no separate post-hoc inflation that could reintroduce a
    conflict. The box is purely segment-local (depends only on its own endpoints and cfg), which is
    what makes per-edge checking equivalent to whole-corridor checking.

    Geometry (see context/figures/corridor_box_extension.png): longitudinally extended so
    consecutive boxes overlap (ASTM §4.3.5 contiguity) and time-buffered by ``time_buffer_s`` so
    neighbours overlap in time. The extension is anisotropic (``corridor_width/2`` horizontal,
    ``corridor_height/2`` vertical) so a climb does not overshoot the ceiling.

    Parameters
    ------------
    - p0 (Vec): segment start position (metres, ENU).
    - t0 (float): time (s) the flight is at ``p0``.
    - p1 (Vec): segment end position (metres, ENU).
    - t1 (float): time (s) the flight is at ``p1``.
    - cfg (SimConfig): supplies corridor width/height and ``time_buffer_s``.
    - terminal_id (Hashable): tags the box as part of a shared terminal column, or ``None``.

    Return
    --------
    - output (Volume4D): the buffered, longitudinally extended corridor box for this segment.
    """
    # Scalar hot path (hundreds of thousands of sub-boxes per refined plan): plain floats, no
    # per-box np.asarray / norm / vector alloc. Bit-identical to the numpy form, pinned by
    # segment_frame's oracle.
    p0x, p0y, p0z = float(p0[0]), float(p0[1]), float(p0[2])
    p1x, p1y, p1z = float(p1[0]), float(p1[1]), float(p1[2])
    dx, dy, dz = p1x - p0x, p1y - p0y, p1z - p0z
    length = math.sqrt(dx * dx + dy * dy + dz * dz)                 # == float(np.linalg.norm(p1 - p0))
    ux, uy, uz = (dx / length, dy / length, dz / length) if length > 1e-9 else (1.0, 0.0, 0.0)
    # half the cross-section: width/2 when level, height/2 for a climb — no z overshoot.
    ext = 0.5 * math.hypot(cfg.corridor_width_m * math.hypot(ux, uy), cfg.corridor_height_m * uz)
    a = (p0x - ux * ext, p0y - uy * ext, p0z - uz * ext)           # extend behind the start
    b = (p1x + ux * ext, p1y + uy * ext, p1z + uz * ext)           # and beyond the end → overlap neighbours
    spec = box_from_segment(a, b, cfg.corridor_width_m, cfg.corridor_height_m)
    return Volume4D(spec, t0 - cfg.time_buffer_s, t1 + cfg.time_buffer_s, terminal_id=terminal_id)


def build_corridor(centerline: list[TimedPoint], cfg: SimConfig) -> list[Volume4D]:
    """Chop a timed 3D polyline into one oriented-box :class:`Volume4D` per segment (ASTM §4.3.5).

    A thin loop over :func:`corridor_segment_volume`, so the whole-path corridor is exactly the
    concatenation of the per-edge boxes a planner checks during search.

    Parameters
    ------------
    - centerline (list[TimedPoint]): timed waypoints; each consecutive pair becomes one box.
    - cfg (SimConfig): corridor geometry, forwarded to :func:`corridor_segment_volume`.

    Return
    --------
    - output (list[Volume4D]): one corridor box per centreline segment (empty if < 2 points).
    """
    return [
        corridor_segment_volume(p0, t0, p1, t1, cfg)
        for (p0, t0), (p1, t1) in zip(centerline, centerline[1:])
    ]


def terminal_radius(term, cfg: SimConfig) -> float:
    """A terminal's column radius — its own ``radius`` if set, else ``cfg.terminal_radius_m`` (90 m)."""
    return term.radius if term.radius is not None else cfg.terminal_radius_m


def exit_radius(term, cfg: SimConfig) -> float:
    """A hub's exit-lane inner edge — flush with the column edge when ``corridor_overlap`` is 0
    (see context/figures/exit_radius.png).

    The exit-lane box is tagged with the hub and the column-involved exemption
    (:func:`conflict.volumes_conflict`) makes it transparent to same-hub COLUMNS, while two same-hub
    corridor boxes still contend (box↔box stays strict), so divergent lanes need the column wide
    enough not to crowd (``cfg.terminal_radius_m`` 90 m default).

    The single source of truth for the fold/lane radius — used by the A* head/tail fold
    (:func:`planner.astar.planner._fold_path`, driving both the commit and the landing gate) and
    :meth:`planner.terminal_capacity.TerminalCapacity.exit_clear` — so the gate, the commit, and the
    exit-lane check all root the lane at the same edge and cannot drift.

    Parameters
    ------------
    - term (Terminal): the hub, read for ``radius`` and ``corridor_overlap``.
    - cfg (SimConfig): supplies the default column radius and ``corridor_width_m``.

    Return
    --------
    - output (float): the lane inner-edge radius, ``terminal_radius + corridor_width/2 − overlap``.
    """
    ov = term.corridor_overlap if term.corridor_overlap is not None else 0.0
    return terminal_radius(term, cfg) + cfg.corridor_width_m / 2.0 - ov


def segment_overlaps_column(a, b, center, radius: float, cfg: SimConfig) -> bool:
    """Does segment ``a→b``'s corridor box reach into the disk of ``radius`` at ``center`` (xy)?

    The box reaches the column iff the distance from ``center`` to its extended centreline is below
    ``radius + corridor_width/2`` (see context/figures/segment_overlaps_column.png).

    Used to tag EVERY near-hub box reaching into a flight's own column — not just box[0]/box[-1].
    The box count is geometry-dependent (radius × exit angle), so a fixed "tag the first N" rule is
    unsound (e.g. a 500 m column can need boxes [1] and [2] tagged); this geometric test scales.
    Far cruise boxes stay untagged, so foreign/same-hub overflight still deconflicts.

    The xy point-to-segment distance is computed with scalars (norm via ``math.sqrt``, dot as a
    scalar sum) — bit-for-bit identical to the numpy form but without its per-call ufunc dispatch,
    since this runs once per corridor sub-box during every rebuild. See ``tests/test_volumes.py``
    for the frozen-numpy byte-identity oracle.

    Parameters
    ------------
    - a (Vec): segment start (only xy is used).
    - b (Vec): segment end (only xy is used).
    - center (Vec): column centre (only xy is used).
    - radius (float): column radius to test against.
    - cfg (SimConfig): supplies ``corridor_width_m`` for the extension and half-width.

    Return
    --------
    - output (bool): True if the extended corridor box overlaps the column disk.
    """
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


def column_dwell_s(center, term, cfg: SimConfig, z: float) -> float:
    """How long a flight occupies its terminal column above the pad: climb, then egress traverse.

    THE SINGLE SOURCE OF TRUTH for the column window, in the style of :func:`exit_radius`. Every
    site that books, gates, or clocks a terminal column must call this — ``astar._build``,
    :func:`build_reservation_from_corners`, and every ``TerminalCapacity`` window — so the gate, the
    commit, and the corridor start cannot drift apart (they did when callers assembled the window
    themselves: a booked column shorter than the gated one, a corridor starting before egress flew).

    Returns the portion AFTER the pad hover — i.e. exactly what :func:`hover_reservation` takes as
    ``climb_time_s``, and exactly how long after takeoff the corridor may begin.

    Parameters
    ------------
    - center (Vec): the hub centre (pad location).
    - term (Terminal): the terminal, for its exit-lane traverse geometry.
    - cfg (SimConfig): supplies the climb rate and lane traverse timing.
    - z (float): the flight's cruise altitude, setting the climb time.

    Return
    --------
    - output (float): seconds after takeoff hover before the corridor may begin (climb + egress).
    """
    from .planner.hexgrid import max_lane_traverse_s   # local: hexgrid imports this module

    return cfg.climb_time_to(z) + max_lane_traverse_s(center, term, cfg)


# --- The three en-route rulers ----------------------------------------------------------------
#
# Spatial layout and the reference / flown / detour definitions: see
# context/figures/enroute_rulers.png (generated by context/figures/make_figures.py).
#
# Both rulers start and end on the SAME circles (exit_radius), so their difference is pure en-route
# detour. Nothing inside a hub column counts — terminal flying consumes that hub's CAPACITY (tagged
# column + pad gate), not en-route distance or delay — and there is no phantom shortcut from
# mismatched endpoints, which would let a refined flight read stretch < 1 ("shorter than a
# straight line"). Every planner gate and metrics._flown/_straight go through these three
# functions, so the gate enforces exactly the ratio stretch later reports and planner-vs-metrics
# drift is structurally impossible. The reference is deliberately lane-AGNOSTIC — the minimum over
# every lane choice, which is what makes stretch >= 1 a triangle-inequality theorem — while the
# flown side reflects the lane actually taken, so a bad or traffic-forced lane choice reads as
# detour (in lattice_overhead_m) instead of inflating the baseline.


def enroute_reference_m(origin, dest, origin_term, dest_term, cfg: SimConfig) -> float:
    """IDEAL ruler (see context/figures/enroute_rulers.png): centre distance minus both
    :func:`exit_radius`, floored at 0.

    LATENT EDGE — the 0-clamp (columns covering the whole trip) is NOT benign: every caller's
    ``straight > _EPS`` guard then skips the ``max_detour_factor`` gate and :func:`enroute_detour_m`
    books 0, so the flight escapes the only length term in ``trajectory_cost`` and ``stretch`` goes
    NaN. Unreached in shipped scenarios (``HubRadiusDemand``'s ``min_r``), but ``min_r`` does not
    scale with ``corridor_overlap``, so ``--corridor-overlap <= -60`` reaches it. Short flights are
    geometry-dominated either way — read ``stretch``/``delay_pct`` with care there.

    Parameters
    ------------
    - origin (Vec): origin position (xy used).
    - dest (Vec): destination position (xy used).
    - origin_term (Terminal | tuple | None): origin terminal, normalized via :func:`as_terminal`.
    - dest_term (Terminal | tuple | None): destination terminal, normalized via :func:`as_terminal`.
    - cfg (SimConfig): supplies the exit-lane radii.

    Return
    --------
    - output (float): edge-to-edge straight-line reference distance (m), floored at 0.
    """
    o = np.asarray(origin, float)[:2]
    d = np.asarray(dest, float)[:2]
    o_term, d_term = as_terminal(origin_term), as_terminal(dest_term)
    gap = float(np.linalg.norm(d - o))
    gap -= exit_radius(o_term, cfg) if o_term is not None else 0.0
    gap -= exit_radius(d_term, cfg) if d_term is not None else 0.0
    return max(0.0, gap)


def enroute_detour_m(flown_m: float, reference_m: float) -> float:
    """The verdict (see context/figures/enroute_rulers.png): flown minus reference, floored at 0;
    0 when the reference is 0.

    The 0-reference guard is the point: without it ``max(0.0, flown - 0.0)`` books the ENTIRE path
    of a flight that never left terminal airspace as detour. No en-route segment means no en-route
    detour. Callers still guard ``reference > _EPS`` for ``stretch``, which stays NaN — undefined is
    the honest answer.

    Parameters
    ------------
    - flown_m (float): actual en-route distance flown (m), from :func:`enroute_flown_m`.
    - reference_m (float): ideal reference distance (m), from :func:`enroute_reference_m`.

    Return
    --------
    - output (float): detour ``max(0, flown − reference)`` m, or 0 when the reference is 0.
    """
    return 0.0 if reference_m <= 1e-9 else max(0.0, flown_m - reference_m)


def enroute_flown_m(points, origin, dest, origin_term, dest_term, cfg: SimConfig) -> float:
    """ACTUAL ruler (see context/figures/enroute_rulers.png): the path folded to both column edges,
    then summed in the plane.

    Folds through the SAME :func:`fold_corners_to_columns` every reservation uses, then sums the
    segment lengths in the horizontal plane. Single owner: every planner's ``air_detour_m``/gate and
    ``metrics._flown_horizontal_m`` call this, so the two layers cannot drift.

    Three contracts:

    - An endpoint with NO terminal extends to the true ``origin``/``dest``, so A*'s endpoint snap
      onto a hex centre stays on A*'s bill (in ``lattice_overhead_m``) instead of reading as a
      phantom shortcut (``stretch`` < 1) — see ``metrics.flight_row``.
    - Re-folding an already-folded path is NEARLY idempotent, not exactly (the edge re-roots toward
      a different first waypoint, a few metres): conservative — a second fold is not free.
    - Fold bail-outs pass through unfolded. Contained: every bail reachable under
      ``fixed_exit_lanes`` with shipped demand has ``enroute_reference_m == 0`` (detour 0,
      ``stretch`` NaN). If a guard is added, fall back to the reference length (stretch -> 1,
      detour -> 0), NOT NaN — NaN would flow into ``air_detour_m`` -> cost -> ``total_delay_s``.

    Parameters
    ------------
    - points (Sequence[Vec]): centreline waypoints to measure (consumed via ``list``).
    - origin (Vec): origin position, appended when there is no origin terminal.
    - dest (Vec): destination position, appended when there is no dest terminal.
    - origin_term (Terminal | tuple | None): origin terminal, normalized via :func:`as_terminal`.
    - dest_term (Terminal | tuple | None): destination terminal, normalized via :func:`as_terminal`.
    - cfg (SimConfig): supplies the exit-lane radii for the fold.

    Return
    --------
    - output (float): horizontal en-route distance (m) of the folded path (0 if < 2 points).
    """
    pts = list(fold_corners_to_columns(list(points), origin, dest, origin_term, dest_term, cfg))
    if as_terminal(origin_term) is None:
        pts.insert(0, np.asarray(origin, float))
    if as_terminal(dest_term) is None:
        pts.append(np.asarray(dest, float))
    xy = np.asarray([np.asarray(p, float)[:2] for p in pts], float)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) >= 2 else 0.0


def fold_corners_to_columns(corners, origin, dest, origin_term, dest_term, cfg: SimConfig):
    """Drop in-column head/tail corners and re-root polyline at the column edge (``exit_radius``)
    (see context/figures/fold_corners.png).

    The centre→edge leg is flown but UNRESERVED — the tagged hover column covers it. This is the
    continuous planners' analogue of A*'s boundary-lane rooting: a strict corridor rooted at the hub
    CENTRE would contend box↔box with every same-hub sibling's corridor (box↔box pairs are never
    exempt), serializing divergent launches that the shared column is meant to admit concurrently.

    Degenerate flights bail out UNFOLDED (returned as given): the whole path inside a column, a
    perimeter direction that is undefined (waypoint at the hub centre), or fewer than two surviving
    points. Bailing is safe — tags still apply, only same-hub concurrency degrades for that flight.

    Parameters
    ------------
    - corners (list[Vec]): the polyline to fold (head/tail corners may sit inside a column).
    - origin (Vec): origin hub centre.
    - dest (Vec): destination hub centre.
    - origin_term (Terminal | tuple | None): origin terminal, normalized via :func:`as_terminal`.
    - dest_term (Terminal | tuple | None): destination terminal, normalized via :func:`as_terminal`.
    - cfg (SimConfig): supplies the exit-lane radii.

    Return
    --------
    - output (list[Vec]): the polyline re-rooted at the column edge(s), or ``corners`` unchanged on
      a degenerate bail.
    """
    pts = [np.asarray(p, float) for p in corners]
    o_term, d_term = as_terminal(origin_term), as_terminal(dest_term)

    def _outside(p, center, r):
        """True if ``p`` lies on or outside the radius-``r`` circle at ``center`` (xy).

        Parameters
        ------------
        - p (np.ndarray): the point to test (only xy is read).
        - center (np.ndarray): the circle centre (only xy is read).
        - r (float): the circle radius in metres.

        Return
        --------
        - output (bool): True iff the xy distance from ``center`` to ``p`` is at least ``r``.
        """
        dx, dy = float(p[0]) - float(center[0]), float(p[1]) - float(center[1])
        return math.sqrt(dx * dx + dy * dy) >= r

    def _edge_point(center, toward, r):
        """The radius-``r`` point on ``center`` toward ``toward`` (``None`` if degenerate).

        Parameters
        ------------
        - center (np.ndarray): the circle centre (xy used; z ignored).
        - toward (np.ndarray): point that sets the direction; its z is copied to the output.
        - r (float): the circle radius in metres.

        Return
        --------
        - output (np.ndarray | None): the xy point at radius ``r`` from ``center`` toward
          ``toward`` with ``toward``'s z, or ``None`` when the two coincide in xy (degenerate).
        """
        dx, dy = float(toward[0]) - float(center[0]), float(toward[1]) - float(center[1])
        n = math.sqrt(dx * dx + dy * dy)
        if n < 1e-9:
            return None
        return np.array([float(center[0]) + r * dx / n, float(center[1]) + r * dy / n,
                         float(toward[2])])

    if o_term is not None:
        r_o = exit_radius(o_term, cfg)
        i = next((k for k, p in enumerate(pts) if _outside(p, origin, r_o)), None)
        if i is None:
            return corners                       # whole path inside the origin column → bail
        ep = _edge_point(origin, pts[i], r_o)
        if ep is None:
            return corners
        pts = [ep] + pts[i:]
    if d_term is not None:
        r_d = exit_radius(d_term, cfg)
        j = next((k for k in range(len(pts) - 1, -1, -1) if _outside(pts[k], dest, r_d)), None)
        if j is None:
            return corners                       # whole (head-folded) path inside the dest column → bail
        ep = _edge_point(dest, pts[j], r_d)
        if ep is None:
            return corners
        pts = pts[: j + 1] + [ep]
    if len(pts) < 2:
        return corners
    return pts


def build_reservation_from_corners(
    corners: list[Vec], origin: Vec, dest: Vec, t_depart: float, g_delay: float, cfg: SimConfig,
    *, origin_term=None, dest_term=None, corridor_t0: float | None = None,
) -> tuple[list[Volume4D], list[TimedPoint], float, float]:
    """Resample a corner polyline to ≤segment-length boxes, time at nominal speed, assemble.

    Shared by the MILP planner and the shortcut refiner so they all emit the same
    contract-preserving boxes (checked == committed). When ``origin_term``/``dest_term`` are given,
    the hub hover column is tagged shared (sized to the terminal's radius) AND every corridor box
    that reaches into that column (:func:`segment_overlaps_column` — not just the first/last) is
    tagged with the hub, so the column-involved exemption lets the near-hub corridor pass through
    the shared column; every box clear of the column stays strict (untagged).

    Parameters
    ------------
    - corners (list[Vec]): the corner polyline to resample.
    - origin (Vec): origin hub centre.
    - dest (Vec): destination hub centre.
    - t_depart (float): filed departure time (s).
    - g_delay (float): ground delay held on the pad before departure (s).
    - cfg (SimConfig): supplies geometry, speeds, and timing.
    - origin_term: origin terminal (``Terminal`` or ``(id, capacity, ...)`` tuple), or ``None``.
    - dest_term: destination terminal, or ``None``.
    - corridor_t0 (float | None): verified corridor start stamp; overrides the derived start.

    Return
    --------
    - output (tuple[list[Volume4D], list[TimedPoint], float, float]): the reservation volumes
      (origin column, corridor boxes, dest column), the timed centreline, cumulative horizontal
      distance (m), and cumulative altitude change (m).
    """
    origin_term, dest_term = as_terminal(origin_term), as_terminal(dest_term)
    # the corner z is the source of truth for climb timing: cruise starts after the climb to the
    # FIRST corner's altitude (its flight level), not a fixed preferred-level climb.
    z_takeoff = float(np.asarray(corners[0], float)[2])
    z_land = float(np.asarray(corners[-1], float)[2])
    # The corridor may not start until the climb AND egress traverse are flown. ``corridor_t0``
    # overrides that derivation with a VERIFIED stamp: a refiner must anchor the corridor exactly
    # where the inner planner's ledger-checked centerline starts. Re-deriving here would mix this
    # CONTINUOUS clock (climb_time_to + WORST lane) with A*'s QUANTISED stamp and drift each box.
    t = corridor_t0 if corridor_t0 is not None else (
        t_depart + g_delay + column_dwell_s(origin, origin_term, cfg, z_takeoff))
    centerline: list[TimedPoint] = [(np.asarray(corners[0], float).copy(), t)]
    edges: list[Volume4D] = []
    cum_horiz = cum_dz = 0.0
    seg = cfg.corridor_segment_len_m
    o_xy = np.asarray(origin, float)[:2] if origin_term is not None else None
    d_xy = np.asarray(dest, float)[:2] if dest_term is not None else None
    o_r = terminal_radius(origin_term, cfg) if origin_term is not None else 0.0
    d_r = terminal_radius(dest_term, cfg) if dest_term is not None else 0.0
    # Scalar hot path (hundreds of thousands of sub-boxes per refined plan): sa/sb are plain-float
    # tuples, so segment_overlaps_column and corridor_segment_volume allocate no per-sub-box arrays.
    # Bit-identical to the numpy form (the frozen oracles and the scenario A/B SHA256 pin it);
    # centerline keeps its np.ndarray points.
    for a, b in zip(corners, corners[1:]):
        ax, ay, az = float(a[0]), float(a[1]), float(a[2])
        bx, by, bz = float(b[0]), float(b[1]), float(b[2])
        dx, dy, dz_seg = bx - ax, by - ay, bz - az
        total = math.sqrt(dx * dx + dy * dy + dz_seg * dz_seg)   # == float(np.linalg.norm(b - a))
        nsub = max(1, math.ceil(total / seg))                    # == max(1, int(np.ceil(total / seg)))
        for k in range(1, nsub + 1):
            f0, f1 = (k - 1) / nsub, k / nsub
            sa = (ax + f0 * dx, ay + f0 * dy, az + f0 * dz_seg)
            sb = (ax + f1 * dx, ay + f1 * dy, az + f1 * dz_seg)
            hx, hy = sb[0] - sa[0], sb[1] - sa[1]
            horiz = math.sqrt(hx * hx + hy * hy)                 # == float(np.linalg.norm((sb - sa)[:2]))
            dz = abs(sb[2] - sa[2])
            t_next = t + max(horiz / cfg.nominal_speed_mps, dz / cfg.climb_rate_mps, 1e-3)
            # Tag EVERY box reaching into its hub's own column (not just first/last), so a near-hub
            # cruise box grazing the shared column is column-exempt rather than a CONFLICT_FILED. See
            # segment_overlaps_column; mirrors astar._build's per-box tagging.
            tid = (origin_term.id if o_xy is not None and segment_overlaps_column(sa, sb, o_xy, o_r, cfg)
                   else dest_term.id if d_xy is not None and segment_overlaps_column(sa, sb, d_xy, d_r, cfg)
                   else None)
            edges.append(corridor_segment_volume(sa, t, sb, t_next, cfg, terminal_id=tid))
            centerline.append((np.array([sb[0], sb[1], sb[2]]), t_next))
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
                          climb_time_s=column_dwell_s(origin, origin_term, cfg, z_takeoff)),
        *edges,
        hover_reservation(dest, t, cfg,
                          terminal_id=dest_term.id if dest_term else None,
                          radius=terminal_radius(dest_term, cfg) if dest_term else None,
                          climb_time_s=column_dwell_s(dest, dest_term, cfg, z_land)),
    ]
    return volumes, centerline, cum_horiz, cum_dz


def ground_dwell_reservation(center: Vec, t0: float, duration_s: float, cfg: SimConfig, *,
                             radius: float | None = None) -> Volume4D:
    """
    The pad an aircraft occupies while SITTING on it, between the legs of a round trip.

    A LOW box, not the full column :func:`hover_reservation` books — the descent that put the aircraft
    there and the climb that takes it away sweep the column and are reserved separately. Same
    footprint radius as the hover column, so another flight may overfly at altitude but none may land
    here while this one is parked. See context/figures/itinerary_reservation.png.

    Parameters
    ------------
    - center (Vec): pad centre
    - t0 (float): when the aircraft is down, i.e. the arrival column's ``t_end``
    - duration_s (float): how long it stays parked
    - cfg (SimConfig): supplies ``ground_box_height_m`` and the default footprint radius
    - radius (float): footprint radius; None uses ``effective_hover_radius_m``

    Return
    --------
    - volume (Volume4D): the parked-aircraft reservation
    """
    center = np.asarray(center, float)
    spec = CylinderSpec(
        cx=float(center[0]),
        cy=float(center[1]),
        radius=cfg.effective_hover_radius_m if radius is None else float(radius),
        z_lo=cfg.ground_level_m,
        z_hi=cfg.ground_level_m + cfg.ground_box_height_m,
    )
    # Deliberately UNTAGGED, and no knob to tag it: `conflict.volumes_conflict` makes two same-hub
    # volumes transparent whenever either is a cylinder, so a tagged box would let another flight of
    # that hub land its column on top of the parked aircraft.
    return Volume4D(spec, t0, t0 + float(duration_s))


def hover_reservation(center: Vec, t0: float, cfg: SimConfig, *, terminal_id: Hashable = None,
                      radius: float | None = None, z_hi: float | None = None,
                      climb_time_s: float | None = None) -> Volume4D:
    """A vertical hover cylinder at ``center`` (ASTM area-based intent, §4.3.5).

    Altitude band [ground, ``z_hi``]; ``z_hi`` defaults to ``airspace_ceiling_m`` so the column
    spans the full regulated tube (a vertiport owns its vertical column of regulated airspace).
    Active for ``hover_time_s + climb_time_s`` from ``t0``. When ``terminal_id`` is set this
    cylinder is a shared terminal column — transparent to its own hub's flights, opaque to everyone
    else (see :func:`conflict.volumes_conflict`).

    Parameters
    ------------
    - center (Vec): cylinder centre (xy); z spans [ground, ``z_hi``].
    - t0 (float): time (s) the reservation becomes active.
    - cfg (SimConfig): supplies default radius, ground level, ceiling, and hover/climb times.
    - terminal_id (Hashable): shared-terminal tag, or ``None`` for ordinary airspace.
    - radius (float | None): column radius; ``None`` ⇒ ``effective_hover_radius_m`` (lets a
      multi-pad vertiport size its shared column bigger than a single pad).
    - z_hi (float | None): band top; ``None`` ⇒ ``airspace_ceiling_m``.
    - climb_time_s (float | None): climb time added to the window; ``None`` ⇒ the preferred-level
      default (pass e.g. :meth:`SimConfig.climb_time_to` of the cruise level to match the real
      climb).

    Return
    --------
    - output (Volume4D): the hover-cylinder reservation, active over
      ``[t0, t0 + hover_time_s + climb_time_s)``.
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


# Effectively-unbounded but FINITE t_end (~31000 yr) for the time-invariant terminal wall: no
# committed corridor — even a late-departing return — can outlast it. Finite (not inf) is
# belt-and-suspenders (a huge t_end would HANG a range()); the real safety is that a static wall
# is never committed. See permanent_terminal_reservation.
_WALL_T_END_S = 1e12


def permanent_terminal_reservation(center: Vec, term, cfg: SimConfig) -> Volume4D:
    """A hub's whole-horizon terminal-airspace reservation — the ledger volume that makes a
    ``cfg.terminal_airspace_always_active`` wall a first-class part of the committed airspace
    (visible to ``ledger.any_conflict`` / ``verify`` / the ledger-only refiners) instead of an
    off-ledger occupancy side-structure.

    Spans the full ``[ground, ceiling]`` tube for the whole horizon and is tagged with
    ``terminal_id`` so the column-involved exemption in :func:`conflict.volumes_conflict` keeps it
    transparent to its own hub's flights while walling foreign cruise.

    Radius = :func:`terminal_radius` — byte-identical to the per-flight dwell column (built by
    reusing :func:`hover_reservation`, so the two cannot drift). It deliberately excludes the
    ``+corridor_width/2`` of :func:`exit_radius` (exit-LANE routing geometry, not a reservation) and
    the wider ``terminal_cells`` flood-fill (A*'s discrete keep-out); since ``terminal_radius ⊂
    terminal_cells``, any corridor routed around those cells clears this column with margin (no
    spurious commit-time denials).

    Time-invariant — active for ALL time, mirroring the A* occupancy ``static_col`` (which blocks
    these cells at EVERY step). Any finite ``cfg``-derived ``t_end`` has a hole: a return departing
    at ``t_request + est_trip + turnaround_s > horizon_s`` (``turnaround_s`` invisible here) can
    land past it and let a foreign crossing escape ``any_conflict`` / ``verify``. So ``t_end`` is a
    finite sentinel (:data:`_WALL_T_END_S`). Safe because a static wall is never committed, so it
    never reaches step-range arithmetic (``ledger._steps`` / ``hexgrid.rasterize_volume``); it
    surfaces only via ``ledger.conflicts``, where the sole arithmetic reader (``straight``'s
    ``min(cv.t_end)``) is refused under always-active.

    Parameters
    ------------
    - center (Vec): the hub centre (column location).
    - term (Terminal | tuple): the terminal, normalized via :func:`as_terminal`; sets the tag and
      radius.
    - cfg (SimConfig): supplies the column radius, ground level, and ceiling.

    Return
    --------
    - output (Volume4D): the tagged hover column with ``t_end`` set to :data:`_WALL_T_END_S`.
    """
    term = as_terminal(term)
    return replace(
        hover_reservation(center, 0.0, cfg, terminal_id=term.id, radius=terminal_radius(term, cfg)),
        t_end=_WALL_T_END_S)
