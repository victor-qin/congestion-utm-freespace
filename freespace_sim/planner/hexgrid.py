"""Axial hex lattice in the local ENU plane + rasterization of committed volumes.

The grid is a fixed global pointy-top hex tiling anchored at ENU (0,0) and shared by every
flight, so the blocked-set built from committed volumes is global and incremental (the continuous
analogue of the sibling project's occupancy ledger). The pitch (centre-to-centre) is tied to
``nominal_speed · dt`` so one hex move is exactly one timestep at nominal speed — which keeps the
A* time axis clean and makes the MILP's "slow-down-for-free" / "hop-a-thin-wall" exploits
structurally impossible.

Rasterization is deliberately conservative (see context/figures/rasterisation_coverage.png): a
cell is blocked if a committed volume, inflated by the new corridor's half-width PLUS one hex
circumradius, reaches its centre. Over-blocking by up to a hex is safe — A* avoids a hair more
than necessary, the NLP recovers the slack by smoothing into the true continuous gap, and FCL
verify is the backstop.
"""

from __future__ import annotations

import math
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ..config import SimConfig
from ..geometry import BoxSpec, CylinderSpec
from ..types import as_terminal
from ..volumes import Volume4D, exit_radius

SQRT3 = math.sqrt(3.0)
# The six pointy-top axial neighbour directions around a centre hex, in ring order (see
# context/figures/hex_layout.png).
AXIAL_NEIGHBORS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

# ---- compiled footprint sweep (see hexgrid_kernel) ------------------------------------------
# The reference sweep below (`_candidate_slack` + a mask) stays the oracle and the fallback. The
# kernel evaluates the same candidates in compiled code and emits only the ones that pass — removing
# the ~84% of candidates the host would otherwise materialise just to discard.
try:
    from .hexgrid_kernel import sweep_box as _sweep_box, sweep_cyl as _sweep_cyl
    _COMPILED = True
    _compiled_error = None
except Exception as exc:                             # optional acceleration must fail open
    _COMPILED = False
    _compiled_error = exc
#: Runtime switch. False selects the numpy oracle; True selects the compiled sweep when available.
#: Box cells within a conservative floating-point envelope of either threshold are re-evaluated by
#: the oracle, so this switch cannot change raster decisions.
USE_COMPILED = True
_compiled_warned = False


def _warn_no_kernel(error: Exception | None = None) -> None:
    """One stderr line, once per process, when optional compiled acceleration cannot run."""
    global _compiled_warned
    if _compiled_warned:
        return
    _compiled_warned = True
    detail = f" ({type(error).__name__}: {error})" if error is not None else ""
    print(f"WARNING: compiled hex rasteriser unavailable{detail} — using the numpy reference "
          "sweep. Fix: run via plain `uv run` (numba is in tool.uv default-groups) or `uv sync`.",
          file=sys.stderr)


def _disable_compiled(error: Exception) -> None:
    """Permanently fall back after a lazy numba/cache/LLVM failure in this process."""
    global _COMPILED, _compiled_error
    _COMPILED = False
    _compiled_error = error
    cache = globals().get("_RANGE_CACHE")
    if cache is not None:
        cache.clear()
    _warn_no_kernel(error)


def set_compiled_rasterizer(enabled: bool) -> None:
    """Select the compiled or reference sweep, invalidating rows the old backend produced.

    Parameters
    ------------
    - enabled (bool): True selects the compiled sweep (when available), False the numpy oracle.

    Return
    --------
    - output (None): sets the ``USE_COMPILED`` global and clears ``_RANGE_CACHE`` on a change,
      so no cached row outlives the backend that produced it.
    """
    global USE_COMPILED
    enabled = bool(enabled)
    if USE_COMPILED != enabled:
        USE_COMPILED = enabled
        cache = globals().get("_RANGE_CACHE")
        if cache is not None:
            cache.clear()


@contextmanager
def rasterizer_backend(compiled: bool):
    """Temporarily select a rasterizer backend, restoring the prior selection on every exit.

    Parameters
    ------------
    - compiled (bool): backend for the ``with`` body (see :func:`set_compiled_rasterizer`).

    Return
    --------
    - output (Generator): a context manager yielding nothing; the prior ``USE_COMPILED`` is restored
      in a ``finally`` even if the body raises.
    """
    previous = USE_COMPILED
    set_compiled_rasterizer(compiled)
    try:
        yield
    finally:
        set_compiled_rasterizer(previous)


def hex_neighbors(q: int, r: int) -> list[tuple[int, int]]:
    """The 6 axial neighbours of hex (q, r)."""
    return [(q + dq, r + dr) for dq, dr in AXIAL_NEIGHBORS]


def hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Axial lattice distance in STEPS: the fewest 6-neighbour moves from ``a`` to ``b``.

    Every lateral A* edge is exactly one pitch, so this times the pitch is the shortest path the
    lattice permits between two cells — the obstacle-free reference a detour is measured against.

    Parameters
    ------------
    - a (tuple[int, int]): first axial cell ``(q, r)``.
    - b (tuple[int, int]): second axial cell ``(q, r)``.

    Return
    --------
    - output (int): the axial (hex) distance in whole 6-neighbour steps.
    """
    dq, dr = b[0] - a[0], b[1] - a[1]
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2



def lattice_overhead_m(cells, pitch, air_detour_m):
    """The share of ``air_detour_m`` that is hex geometry rather than traffic, in metres.

    The traffic share is derived EXACTLY and subtracted: every lateral edge is one pitch, so ``moves
    actually flown − the lattice geodesic between the first and last cell`` is the berth traffic
    forced, in whole hex steps. That residual is exactly 0 for an unimpeded flight at ANY bearing —
    the invariant that makes the congestion reading trustworthy.

    Everything else in ``air_detour_m`` is geometry and lands here: the staircase a 6-direction
    lattice imposes on an off-axis bearing (0 on-axis, peaking at 2/√3 − 1 ≈ 15.5% at 30° off — see
    context/figures/hex_lattice_overhead.png), plus the endpoint snap of origin/dest onto cell
    centres. The in-column fold does NOT land here: ``air_detour_m`` is measured exit lane →
    exit lane on both sides, so hub centre → column edge is outside the measurement (that hub's
    capacity). The column edge → lane-cell hop DOES: A*'s path starts on a boundary hex, not the
    reference circle, so folding EXTENDS it — lane snap, still geometry not traffic.

    Parameters
    ------------
    - cells (Sequence[tuple[int, int]]): the flown axial path, one ``(q, r)`` per waypoint.
    - pitch (float): centre-to-centre hex spacing (m); one lateral move is one pitch.
    - air_detour_m (float): total measured en-route air detour (m) to attribute.

    Return
    --------
    - output (float): the geometry-only share of ``air_detour_m`` (m), floored at 0 (the exact
      traffic-forced residual removed).
    """
    moves = sum(1 for a, b in zip(cells, cells[1:]) if a != b)
    forced = max(0, moves - hex_distance(cells[0], cells[-1])) * pitch
    return max(0.0, air_detour_m - forced)

def circumradius(cfg: SimConfig) -> float:
    """Hex circumradius R, from pitch = nominal_speed·dt and pitch = √3·R."""
    return cfg.nominal_speed_mps * cfg.dt_s / SQRT3


def hex_center(q: int, r: int, R: float) -> np.ndarray:
    """ENU (x, y) of the centre of axial hex (q, r) — pointy-top axial layout
    (see context/figures/hex_layout.png)."""
    return np.array([R * SQRT3 * (q + r / 2.0), R * 1.5 * r])


def enu_to_axial(x: float, y: float, R: float) -> tuple[int, int]:
    """Nearest axial hex to ENU ``(x, y)`` — the inverse of :func:`hex_center`.

    Parameters
    ------------
    - x (float): ENU easting (m).
    - y (float): ENU northing (m).
    - R (float): hex circumradius (m), from :func:`circumradius`.

    Return
    --------
    - output (tuple[int, int]): the axial ``(q, r)`` whose centre is nearest ``(x, y)``.
    """
    qf = (SQRT3 / 3.0 * x - 1.0 / 3.0 * y) / R
    rf = (2.0 / 3.0 * y) / R
    return _axial_round(qf, rf)


def _axial_round(qf: float, rf: float) -> tuple[int, int]:
    """Round fractional axial ``(qf, rf)`` to the nearest hex centre via cube-coordinate rounding:
    round all three coords, then recompute whichever drifted most so ``x + y + z == 0`` holds."""
    xf, zf = qf, rf
    yf = -xf - zf
    rx, ry, rz = round(xf), round(yf), round(zf)
    dx, dy, dz = abs(rx - xf), abs(ry - yf), abs(rz - zf)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


@dataclass(frozen=True)
class Lane:
    """A hub's canonical exit/approach lane: a boundary hex just outside the column.

    Same-hub launches are deconflicted by exact cell occupancy at plan time (``HexOccupancyService``),
    not by any per-lane graze graph — two flights whose corridors share a footprint cell serialise,
    divergent ones stay concurrent. So a lane is just its ``cell`` plus descriptors: ``bearing`` (the
    stable sort order of the ring) and ``dist`` (used by the A* heuristic and the takeoff edge cost)."""

    cell: tuple[int, int]
    bearing: float            # degrees, from the hub centre (the lane ring's stable sort key)
    dist: float               # metres, hub centre → cell centre
    steps: int = 0            # dt steps to translate ``dist`` at cruise speed (egress traverse)


_LANE_CACHE: dict = {}
_COVERED_CACHE: dict = {}


def _bearing_deg(cell, cx: float, cy: float, R: float) -> float:
    """Bearing in degrees from hub ``(cx, cy)`` to ``cell``'s ENU centre (the lane sort key).

    Parameters
    ------------
    - cell (tuple[int, int]): the axial ``(q, r)`` cell whose centre the bearing points at.
    - cx (float): hub centre easting (ENU x, m).
    - cy (float): hub centre northing (ENU y, m).
    - R (float): hex circumradius (m).

    Return
    --------
    - output (float): bearing from the hub to the cell centre, in degrees.
    """
    bx, by = hex_center(*cell, R)
    return math.degrees(math.atan2(by - cy, bx - cx))


def _covered_boundary(center, term, cfg: SimConfig) -> tuple[set, set]:
    """Flood-fill a hub's column into ``covered`` hexes (centre within ``exit_radius``) and
    the ``boundary`` ring just outside them — neighbours of a covered cell that aren't covered (see
    context/figures/exit_radius.png). Shared by :func:`terminal_lanes` (boundary = exit lanes) and
    :func:`terminal_cells` (covered ∪ boundary = the full reserved terminal airspace).

    Memoised per ``(center, term, cfg)`` — hubs are fixed, and ``terminal_cells`` sits on
    the compiled A* hot path (the own-hub overlay, once per flight), so without the cache every
    same-hub plan re-ran this flood-fill. Callers treat the returned sets as read-only
    (``terminal_cells`` unions into a fresh set; ``terminal_lanes`` sorts/iterates), so sharing the
    cached instances is safe. Mirrors ``_LANE_CACHE``.

    Parameters
    ------------
    - center (Vec): the hub centre (column location).
    - term (Terminal | tuple): the terminal, normalized via :func:`as_terminal`.
    - cfg (SimConfig): supplies the circumradius and exit radius.

    Return
    --------
    - output (tuple[set, set]): ``(covered, boundary)`` — covered column hexes and their outside
      neighbour ring; both may be shared cached instances (treat as read-only).
    """
    term = as_terminal(term)
    cx, cy = float(center[0]), float(center[1])
    key = (round(cx, 3), round(cy, 3), term, cfg)
    cached = _COVERED_CACHE.get(key)
    if cached is not None:
        return cached
    R = circumradius(cfg)
    er = exit_radius(term, cfg)
    covered: set = set()
    stack = [enu_to_axial(cx, cy, R)]                    # the home hex is always covered (er > R)
    while stack:
        h = stack.pop()
        if h in covered:
            continue
        hx, hy = hex_center(*h, R)
        if math.hypot(hx - cx, hy - cy) < er - 1e-9:
            covered.add(h)
            stack.extend(hex_neighbors(*h))
    boundary = {n for h in covered for n in hex_neighbors(*h) if n not in covered}
    result = (covered, boundary)
    _COVERED_CACHE[key] = result
    return result


def terminal_cells(center, term, cfg: SimConfig) -> set:
    """Every hex of a hub's reserved terminal airspace — the column (covered) plus its exit lanes
    (boundary); see context/figures/exit_radius.png.

    Walls the terminal off from FOREIGN cruise traffic when ``cfg.terminal_airspace_always_active``
    (foreign flights route around instead of crossing): the occupancy derives its routing wall from
    this set via :meth:`~freespace_sim.ledger.ReservationLedger.register_static_terminal` and
    :meth:`~freespace_sim.planner.astar.occupancy.HexOccupancyService._on_static`.

    Parameters
    ------------
    - center (Vec): the hub centre (column location).
    - term (Terminal | tuple): the terminal, normalized via :func:`as_terminal`.
    - cfg (SimConfig): supplies the circumradius and exit radius.

    Return
    --------
    - output (set): the axial ``(q, r)`` cells of the column plus its boundary ring.
    """
    covered, boundary = _covered_boundary(center, term, cfg)
    return covered | boundary


def terminal_lanes(center, term, cfg: SimConfig) -> list[Lane]:
    """A hub's fixed exit-lane set: the boundary hexes of its column, sorted by bearing.

    Classify hexes by centre distance to the hub — covered if within ``exit_radius`` (flood-filled
    out from the home hex), boundary if not covered but hex-adjacent to a covered cell (see
    context/figures/exit_radius.png). The boundary ring is the canonical exit-lane set, fully
    determined by the hub position and the fixed grid (no snapping). Deterministic in
    ``(center, term, cfg)`` and memoised — hubs don't move during a run.

    Parameters
    ------------
    - center (Vec): the hub centre (column location).
    - term (Terminal | tuple): the terminal, normalized via :func:`as_terminal`.
    - cfg (SimConfig): supplies the circumradius, exit radius, and egress-traverse timing.

    Return
    --------
    - output (list[Lane]): one :class:`Lane` per boundary hex, ordered by bearing from the hub.
    """
    term = as_terminal(term)
    cx, cy = float(center[0]), float(center[1])
    key = (round(cx, 3), round(cy, 3), term, cfg)
    cached = _LANE_CACHE.get(key)
    if cached is not None:
        return cached
    R = circumradius(cfg)
    _covered, boundary = _covered_boundary(center, term, cfg)
    cells = sorted(boundary, key=lambda c: _bearing_deg(c, cx, cy, R))
    hub = np.array([cx, cy])
    lanes = [
        Lane(cell=c, bearing=_bearing_deg(c, cx, cy, R),
             dist=(d := float(np.linalg.norm(hex_center(*c, R) - hub))),
             steps=int(math.ceil(d / (cfg.nominal_speed_mps * cfg.dt_s))))
        for c in cells
    ]
    _LANE_CACHE[key] = lanes
    return lanes


def _levels_overlapped(vol: Volume4D, cfg: SimConfig) -> list[int]:
    """Indices of the flight levels whose corridor band ``[z_L ± corridor_height/2]`` overlaps the
    volume's z-AABB. A level corridor box touches exactly one; a slanted/climb box touches ≥2; a
    ``[ground, ceiling]`` hover/terminal column touches every in-band level (the regulated tube)."""
    _x0, _y0, zlo, _x1, _y1, zhi = vol.flat_aabb()   # scalars, no throwaway arrays; flat_aabb is
    #                                                   pinned bit-for-bit against aabb()
    half = cfg.corridor_height_m / 2.0
    return [L for L, z in enumerate(cfg.flight_levels_m) if zlo <= z + half and z - half <= zhi]


def _hexes_in_box(amin, amax, R):
    """Yield all axial hexes whose centres could lie in the xy AABB [amin, amax] — a superset
    of the kept footprint (see context/figures/rasterisation_coverage.png).

    Parameters
    ------------
    - amin (Sequence[float]): lower ``(x, y)`` corner of the AABB (m).
    - amax (Sequence[float]): upper ``(x, y)`` corner of the AABB (m).
    - R (float): hex circumradius (m).

    Return
    --------
    - output (Iterator[tuple[int, int]]): axial ``(q, r)`` cells covering the box (a superset).
    """
    qs, rs = [], []
    for x in (amin[0], amax[0]):
        for y in (amin[1], amax[1]):
            q, r = enu_to_axial(x, y, R)
            qs.append(q)
            rs.append(r)
    for q in range(min(qs) - 1, max(qs) + 2):
        for r in range(min(rs) - 1, max(rs) + 2):
            yield q, r


def _footprint_contains(shape, c: np.ndarray, infl: float, cfg: SimConfig,
                        z: float | None = None) -> bool:
    """True if point ``c`` (at altitude ``z``, default cruise) lies inside ``shape`` inflated by
    ``infl`` metres. The scalar per-hex membership test; :func:`_footprint_slack` is its vectorized
    twin, and ``_hexes_in_box`` + this is the test suite's independent oracle for the sweeps.

    Parameters
    ------------
    - shape (BoxSpec | CylinderSpec): the volume footprint to test membership against.
    - c (np.ndarray): the xy point ``(x, y)`` whose membership is tested.
    - infl (float): footprint inflation (m).
    - cfg (SimConfig): supplies the default cruise level.
    - z (float | None): altitude probe (m); None ⇒ ``cfg.cruise_level_m``.

    Return
    --------
    - output (bool): True iff ``c`` at altitude ``z`` lies inside ``shape`` inflated by ``infl``.
    """
    z = cfg.cruise_level_m if z is None else z
    p = np.array([c[0], c[1], z])
    if isinstance(shape, BoxSpec):
        local = shape.rotation().T @ (p - np.array(shape.center, float))
        half = np.array(shape.extents, float) / 2.0 + infl
        return bool(np.all(np.abs(local) <= half))
    # cylinder
    d = float(np.hypot(p[0] - shape.cx, p[1] - shape.cy))
    return d <= shape.radius + infl and (shape.z_lo - infl <= z <= shape.z_hi + infl)


def _footprint_slack(shape, cx: np.ndarray, cy: np.ndarray, cfg: SimConfig,
                     z: float | None = None) -> np.ndarray:
    """Vectorized inflation margin for hex centres (cx, cy arrays) at altitude probe ``z`` (defaults to
    the preferred cruise level): a centre is inside the footprint at inflation ``x`` iff ``slack <= x``.
    Computes the shape geometry ONCE for all candidate hexes, so a single matmul/reduce replaces the
    millions of per-hex :func:`_footprint_contains` calls. Callers probe once per flight level ``z_L``.

    Equivalence to the scalar test: for a box, ``all(|local_d| <= half_d + x)`` ⟺
    ``max_d(|local_d| - half_d) <= x``; for a cylinder the radial (``d - radius``) and altitude-band
    margins both reduce to ``margin <= x``. ``rotᵀ·v`` (column) equals ``v·rot`` (row), batched.

    Parameters
    ------------
    - shape (BoxSpec | CylinderSpec): the volume footprint to measure margins against.
    - cx (np.ndarray): x-coordinates of the candidate hex centres (m).
    - cy (np.ndarray): y-coordinates of the candidate hex centres (m).
    - cfg (SimConfig): supplies the default cruise level.
    - z (float | None): altitude probe (m); None ⇒ ``cfg.cruise_level_m``.

    Return
    --------
    - output (np.ndarray): per-centre slack; inside at inflation ``x`` iff ``slack <= x``.
    """
    z = cfg.cruise_level_m if z is None else z
    if isinstance(shape, BoxSpec):
        center = np.array(shape.center, float)
        p = np.column_stack([cx, cy, np.full(cx.shape, z)])
        local = np.abs((p - center) @ shape.rotation())
        half = np.array(shape.extents, float) / 2.0
        return np.max(local - half, axis=1)
    radial = np.hypot(cx - shape.cx, cy - shape.cy) - shape.radius
    z_slack = max(shape.z_lo - z, z - shape.z_hi)
    return np.maximum(radial, z_slack)


def _axial_rect(xmin: float, ymin: float, xmax: float, ymax: float, R: float):
    """Inclusive axial rectangle ``(q0, q1, r0, r1)`` covering every hex whose centre could lie in the
    xy box — the same superset :func:`_hexes_in_box` walks, as bounds instead of a generator.

    Both sweeps derive their candidates from HERE rather than each rolling their own, because a
    compiled sweep that enumerated a differently-sized rectangle would silently keep a different
    cell set at the margin. Rounding stays in Python on purpose: :func:`_axial_round` uses banker's
    ``round``, whose numba semantics differ, so the kernel is handed bounds and never rounds.

    Parameters
    ------------
    - xmin (float): lower x bound of the xy box (m).
    - ymin (float): lower y bound of the xy box (m).
    - xmax (float): upper x bound of the xy box (m).
    - ymax (float): upper y bound of the xy box (m).
    - R (float): hex circumradius (m).

    Return
    --------
    - output (tuple[int, int, int, int]): inclusive axial bounds ``(q0, q1, r0, r1)``.
    """
    qs, rs = [], []
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            q, r = enu_to_axial(x, y, R)
            qs.append(q)
            rs.append(r)
    return min(qs) - 1, max(qs) + 1, min(rs) - 1, max(rs) + 1


def _candidate_slack(vol: Volume4D, cfg: SimConfig, R: float, infl: float, z: float | None = None):
    """Candidate axial hexes (centres within the volume AABB inflated by ``infl``) and each one's
    :func:`_footprint_slack` at altitude probe ``z``. The (q, r) enumeration reproduces
    :func:`_hexes_in_box` as arrays.

    The numpy REFERENCE sweep: kept as the oracle the compiled kernel is pinned against, and as the
    fallback when numba is absent or ``USE_COMPILED`` is off.

    Parameters
    ------------
    - vol (Volume4D): the committed volume whose footprint is swept.
    - cfg (SimConfig): supplies the default cruise level (via :func:`_footprint_slack`).
    - R (float): hex circumradius (m).
    - infl (float): footprint inflation (m) sizing the candidate rectangle.
    - z (float | None): altitude probe (m); None ⇒ ``cfg.cruise_level_m``.

    Return
    --------
    - output (tuple[np.ndarray, np.ndarray, np.ndarray]): ``(q_grid, r_grid, slack)`` — the axial
      ``q`` and ``r`` of each candidate cell and its :func:`_footprint_slack`.
    """
    lo, hi = vol.aabb()
    q0, q1, r0, r1 = _axial_rect(lo[0] - infl, lo[1] - infl, hi[0] + infl, hi[1] + infl, R)
    q_grid, r_grid = np.meshgrid(
        np.arange(q0, q1 + 1), np.arange(r0, r1 + 1), indexing="ij"
    )
    q_grid = q_grid.ravel()
    r_grid = r_grid.ravel()
    cx = R * SQRT3 * (q_grid + r_grid / 2.0)
    cy = R * 1.5 * r_grid
    return q_grid, r_grid, _footprint_slack(vol.shape, cx, cy, cfg, z=z)


def _sweep_kept(vol: Volume4D, cfg: SimConfig, R: float, infl_blocked: float, infl_pad: float,
                z: float | None = None):
    """``(qs, rs, in_blocked)`` for the cells of ``vol``'s footprint at altitude probe ``z`` — the
    single place the compiled/reference choice is made.

    All three public rasterisers funnel through here so they cannot drift apart and so one switch
    (``USE_COMPILED``) controls the A/B and the rollback for every one of them. ``infl_pad`` sizes
    the candidate rectangle (it is the wider inflation, so the blocked cells are a subset);
    ``in_blocked`` flags membership in the narrower corridor footprint.

    Parameters
    ------------
    - vol (Volume4D): the committed volume whose footprint is swept.
    - cfg (SimConfig): supplies the default cruise level and flight levels.
    - R (float): hex circumradius (m).
    - infl_blocked (float): inflation (m) for the narrower corridor footprint (sets ``in_blocked``).
    - infl_pad (float): wider inflation (m) that sizes the candidate rectangle.
    - z (float | None): altitude probe (m); None ⇒ ``cfg.cruise_level_m``.

    Return
    --------
    - output (tuple[list[int], list[int], list[bool]]): ``(qs, rs, in_blocked)`` per kept cell —
      axial ``q``/``r`` and whether the cell also lies in the narrower corridor footprint.
    """
    z = cfg.cruise_level_m if z is None else z
    if _COMPILED and USE_COMPILED:
        x0, y0, _z0, x1, y1, _z1 = vol.flat_aabb()   # scalars; pinned bit-for-bit against aabb()
        q0, q1, r0, r1 = _axial_rect(x0 - infl_pad, y0 - infl_pad,
                                     x1 + infl_pad, y1 + infl_pad, R)
        n_cand = (q1 - q0 + 1) * (r1 - r0 + 1)       # exact upper bound: overflow is impossible
        oq = np.empty(n_cand, np.int64)
        orr = np.empty(n_cand, np.int64)
        ob = np.empty(n_cand, np.bool_)
        s = vol.shape
        if isinstance(s, BoxSpec):
            oa = np.empty(n_cand, np.bool_)
            m, e = s.rot, s.extents
            try:
                n = _sweep_box(q0, q1, r0, r1, R, s.center[0], s.center[1], s.center[2],
                               m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
                               e[0] / 2.0, e[1] / 2.0, e[2] / 2.0,
                               z, infl_pad, infl_blocked, oq, orr, ob, oa)
            except Exception as exc:
                _disable_compiled(exc)
            else:
                ambiguous = np.flatnonzero(oa[:n])
                if ambiguous.size:
                    aq, ar = oq[ambiguous], orr[ambiguous]
                    cx = R * SQRT3 * (aq + ar / 2.0)
                    cy = R * 1.5 * ar
                    slack = _footprint_slack(s, cx, cy, cfg, z=z)
                    keep = np.ones(n, dtype=np.bool_)
                    keep[ambiguous] = slack <= infl_pad
                    ob[ambiguous] = slack <= infl_blocked
                    return (oq[:n][keep].tolist(), orr[:n][keep].tolist(),
                            ob[:n][keep].tolist())
                return oq[:n].tolist(), orr[:n].tolist(), ob[:n].tolist()
        else:
            try:
                n = _sweep_cyl(q0, q1, r0, r1, R, s.cx, s.cy, s.radius, s.z_lo, s.z_hi,
                               z, infl_pad, infl_blocked, oq, orr, ob)
            except Exception as exc:
                _disable_compiled(exc)
            else:
                return oq[:n].tolist(), orr[:n].tolist(), ob[:n].tolist()
    if not _COMPILED and USE_COMPILED:
        _warn_no_kernel(_compiled_error)
    q_grid, r_grid, slack = _candidate_slack(vol, cfg, R, infl_pad, z=z)
    in_pad = slack <= infl_pad
    return (q_grid[in_pad].tolist(), r_grid[in_pad].tolist(),
            (slack[in_pad] <= infl_blocked).tolist())


def _step_range(vol: Volume4D, cfg: SimConfig) -> range:
    """The inclusive step span a committed volume blocks, widened by the corridor box's temporal
    extent: a move ARRIVING at step s commits a box spanning ``[(s−1)·dt − buffer, s·dt + buffer]``,
    so s is blocked whenever that box could overlap the obstacle window. Without the widening A*
    enters a just-cleared cell and the rebuilt box clips it.
    """
    dt = cfg.dt_s
    s0 = int(math.floor((vol.t_start - cfg.time_buffer_s) / dt))
    s1 = int(math.floor((vol.t_end + dt + cfg.time_buffer_s) / dt))
    return range(s0, s1 + 1)


def _cylinder_z_independent(vol: Volume4D, cfg: SimConfig, levels: list[int]) -> bool:
    """True when ``vol`` is a vertical cylinder whose z-band contains every overlapped level. Its (q,r)
    footprint is then z-INDEPENDENT — the radial slack doesn't depend on z and the altitude-band slack is
    ≤ 0 in-band, so the mask reduces to the radial term — hence the masked cell set is identical at each
    level and the per-level loop can compute it ONCE. (All committed hover/terminal columns span the whole
    [ground, ceiling] tube, so this is their common case.)"""
    return (isinstance(vol.shape, CylinderSpec)
            and vol.shape.z_lo <= cfg.flight_levels_m[levels[0]]
            and cfg.flight_levels_m[levels[-1]] <= vol.shape.z_hi)


def rasterize_volume(vol: Volume4D, cfg: SimConfig, R: float, infl: float | None = None):
    """Yield ``(q, r, L, s)`` cells a committed volume blocks (conservatively inflated), for every
    overlapped flight level ``L`` and every blocked step ``s``.

    ``infl`` overrides the footprint inflation (metres). It defaults to the corridor half-width plus
    one hex — correct for the swept corridor. Callers checking pad occupancy (the takeoff/landing
    hover cylinder) pass ``effective_hover_radius_m + R`` instead, so the blocked footprint matches
    the wider cylinder rather than the corridor.

    Parameters
    ------------
    - vol (Volume4D): the committed volume to rasterize.
    - cfg (SimConfig): supplies the flight levels, corridor width, and step timing.
    - R (float): hex circumradius (m).
    - infl (float | None): footprint inflation (m); ``None`` ⇒ ``corridor_width/2 + R``.

    Return
    --------
    - output (Iterator): yields ``(q, r, L, s)`` per blocked (cell, level, step); empty when the
      volume overlaps no flight level. Vectorized — see :func:`_footprint_slack`.
    """
    levels = _levels_overlapped(vol, cfg)
    if not levels:
        return
    if infl is None:
        infl = cfg.corridor_width_m / 2.0 + R      # corridor half-width + one hex (conservative)
    steps = _step_range(vol, cfg)
    if _cylinder_z_independent(vol, cfg, levels):          # z-independent footprint → compute once
        qs, rs, _b = _sweep_kept(vol, cfg, R, infl, infl, z=cfg.flight_levels_m[levels[0]])
        cells = list(zip(qs, rs))
        for L in levels:
            for q, r in cells:
                for s in steps:
                    yield q, r, L, s
        return
    for L in levels:
        qs, rs, _b = _sweep_kept(vol, cfg, R, infl, infl, z=cfg.flight_levels_m[L])
        for q, r in zip(qs, rs):
            for s in steps:
                yield q, r, L, s


def rasterize_volume_dual(
    vol: Volume4D, cfg: SimConfig, R: float, infl_blocked: float, infl_pad: float
):
    """One vectorized sweep yielding ``(q, r, L, s, in_blocked)`` over the pad footprint, where
    ``in_blocked`` flags membership in the (smaller) corridor footprint.

    Requires ``infl_pad >= infl_blocked`` so pad cells are a superset of blocked cells. Replaces two
    :func:`rasterize_volume` passes with a single geometry computation per volume (the A* hot path).

    Parameters
    ------------
    - vol (Volume4D): the committed volume to rasterize.
    - cfg (SimConfig): supplies the flight levels and step timing.
    - R (float): hex circumradius (m).
    - infl_blocked (float): inflation (m) for the narrower corridor (``in_blocked``) footprint.
    - infl_pad (float): inflation (m) for the wider pad footprint that sizes the candidate set.

    Return
    --------
    - output (Iterator): yields ``(q, r, L, s, in_blocked)`` per pad (cell, level, step); empty when
      the volume overlaps no flight level.
    """
    levels = _levels_overlapped(vol, cfg)
    if not levels:
        return
    steps = _step_range(vol, cfg)
    if _cylinder_z_independent(vol, cfg, levels):          # z-independent footprint → compute once
        rows = list(zip(*_sweep_kept(vol, cfg, R, infl_blocked, infl_pad,
                                     z=cfg.flight_levels_m[levels[0]])))
        for L in levels:
            for q, r, b in rows:
                for s in steps:
                    yield q, r, L, s, b
        return
    for L in levels:
        qp, rp, in_blk = _sweep_kept(vol, cfg, R, infl_blocked, infl_pad, z=cfg.flight_levels_m[L])
        for q, r, b in zip(qp, rp, in_blk):
            for s in steps:
                yield q, r, L, s, b


def rasterize_volume_ranges(
    vol: Volume4D, cfg: SimConfig, R: float, infl_blocked: float, infl_pad: float
):
    """Like :func:`rasterize_volume_dual`, but yields one ``(q, r, L, s_lo, s_hi, in_blocked)`` per
    cell with the step axis collapsed to its inclusive ``[s_lo, s_hi]`` range instead of one row per
    (cell, step).

    A committed volume occupies each cell over a contiguous step span (``_step_range`` is a plain
    ``range``), so the per-step form yields ``S`` rows the consumers then process one at a time.
    Handing the range straight through lets a consumer block the span in ONE operation (e.g. the
    compiled interval pool's single split). Byte-identical coverage: expanding every range back
    over ``s_lo..s_hi`` reproduces the dual sweep exactly.

    Parameters
    ------------
    - vol (Volume4D): the committed volume to rasterize.
    - cfg (SimConfig): supplies the flight levels and step timing.
    - R (float): hex circumradius (m).
    - infl_blocked (float): inflation (m) for the narrower corridor footprint.
    - infl_pad (float): inflation (m) for the wider pad footprint.

    Return
    --------
    - output (Iterator): yields ``(q, r, L, s_lo, s_hi, in_blocked)`` per cell; empty if the volume
      overlaps no flight level or spans no step.
    """
    levels = _levels_overlapped(vol, cfg)
    if not levels:
        return
    steps = _step_range(vol, cfg)
    s_lo, s_hi = steps.start, steps.stop - 1               # range(s0, s1+1) → inclusive [s0, s1]
    if s_hi < s_lo:
        return
    if _cylinder_z_independent(vol, cfg, levels):
        rows = list(zip(*_sweep_kept(vol, cfg, R, infl_blocked, infl_pad,
                                     z=cfg.flight_levels_m[levels[0]])))
        for L in levels:
            for q, r, b in rows:
                yield q, r, L, s_lo, s_hi, b
        return
    for L in levels:
        qp, rp, in_blk = _sweep_kept(vol, cfg, R, infl_blocked, infl_pad, z=cfg.flight_levels_m[L])
        for q, r, b in zip(qp, rp, in_blk):
            yield q, r, L, s_lo, s_hi, b


_RANGE_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
# Both occupancy images consume the SAME volume with the SAME params on each commit (hex then
# compiled), so the geometry sweep (`_sweep_kept`) is memoized once and reused by the second
# consumer. The cap must exceed the reuse WINDOW for that sharing to fire. For the FCFS sim that
# window is one flight's volumes; under LNS it is one NEIGHBORHOOD's, because the claim index
# (`lns.state._index_add`, a third consumer at the same inflations) rasterizes the victims only
# after all of them have been committed — a cap below the neighborhood size loses most of its rows
# before reaching it. An entry is ~10 rows (a corridor box covers ~10 hexes), so 1024 costs ~1 MB.
# Overshooting only wastes a miss's recompute, never correctness; undershooting silently loses the
# sharing, so this errs high.
#
# 1024 is a FLOOR: `prepare_range_cache_for_commit` raises the cap when a single commit carries
# more volumes than that, so a long trajectory stays resident through the fan-out. It never
# lowers the cap below this floor, which is what keeps the LNS neighborhood window (spanning SEVERAL
# commits) intact.
_RANGE_CACHE_MIN_CAP = 1024
_RANGE_CACHE_CAP = 1024

_COL_HEX_CACHE: "OrderedDict[tuple, frozenset]" = OrderedDict()
# Sized like `_RANGE_CACHE` and for the same reason. The key is `(cols, R)` and `cols` comes from a
# flight's own terminal CYLINDERS, so with hub-radius demand it is really per-HUB — hundreds of
# distinct hubs; a smaller cap thrashes. An entry is ~50 hexes. Undershooting silently loses
# the sharing, so this errs high.
_COL_HEX_CACHE_CAP = 1024
# Unsynchronized, exactly like `_RANGE_CACHE`: the parallel LNS path SPAWNS processes, so each has
# its own copy and there is no cross-thread reader to race with.


def column_hexes(cols: tuple, R: float) -> frozenset:
    """Axial hexes whose CENTRE lies inside any ``(cx, cy, radius)`` disc in ``cols``.

    Four consumers ask "is this cell inside the committing flight's own terminal column?" — the two
    occupancy images, ``HexOccupancyService.enable_blocked``'s re-derive, and the LNS claim index —
    and the answer depends only on ``(cols, R)``, both constant for a flight, so it is resolved once
    per flight and shared instead of recomputed per rasterized cell.

    Exact by construction, not by approximation: the set is built with the SAME ``hex_center`` and the
    SAME ``<=`` comparison the per-cell test uses, and ``_hexes_in_box`` yields a SUPERSET
    of the hexes whose centres could lie in a box — so every centre within ``rad`` of a disc centre,
    which necessarily lies in that disc's AABB, is enumerated.

    Parameters
    ------------
    - cols (tuple): the flight's terminal discs as ``(cx, cy, radius)`` tuples.
    - R (float): hex circumradius (m).

    Return
    --------
    - output (frozenset): the axial ``(q, r)`` cells whose centre lies inside any disc (empty when
      ``cols`` is empty).
    """
    if not cols:
        return frozenset()
    key = (cols, R)
    hit = _COL_HEX_CACHE.get(key)
    if hit is not None:
        _COL_HEX_CACHE.move_to_end(key)
        return hit
    out = set()
    for cx, cy, rad in cols:
        for q, r in _hexes_in_box((cx - rad, cy - rad), (cx + rad, cy + rad), R):
            c = hex_center(q, r, R)
            if (c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= rad * rad:
                out.add((q, r))
    res = frozenset(out)
    _COL_HEX_CACHE[key] = res
    _COL_HEX_CACHE.move_to_end(key)
    if len(_COL_HEX_CACHE) > _COL_HEX_CACHE_CAP:
        _COL_HEX_CACHE.popitem(last=False)
    return res


def prepare_range_cache_for_commit(volumes) -> None:
    """Size the shared raster cache for one synchronous ledger commit.

    Every occupancy observer calls this before walking ``volumes``. The first call raises the active
    LRU to the commit size when that exceeds ``_RANGE_CACHE_MIN_CAP``; later observers repeat the same
    cheap assignment and therefore see every row the first produced. ``ReservationLedger.commit``
    invokes observers synchronously, so no unrelated rasterization interleaves with that window. It
    never drops below the floor — LNS's claim index reads victims rasterized across SEVERAL earlier
    commits, and sizing down to the latest commit would evict exactly the rows that window keeps.

    Parameters
    ------------
    - volumes (Sequence): the volumes about to be walked in this commit; only its length is read.

    Return
    --------
    - output (None): raises the global ``_RANGE_CACHE_CAP`` (never below the floor) and evicts
      any overflow from ``_RANGE_CACHE``.
    """
    global _RANGE_CACHE_CAP
    _RANGE_CACHE_CAP = max(_RANGE_CACHE_MIN_CAP, len(volumes))
    while len(_RANGE_CACHE) > _RANGE_CACHE_CAP:
        _RANGE_CACHE.popitem(last=False)


def rasterize_ranges(vol: Volume4D, cfg: SimConfig, R: float, infl_blocked: float, infl_pad: float):
    """Materialized :func:`rasterize_volume_ranges`, shared between a commit's occupancy consumers.

    These per-commit rasterizations dominate the serial commit floor and the geometry is
    identical between consumers, so it is computed once here and reused. Consumers are A*'s
    ``HexOccupancyService`` + ``CompiledHexOccupancy``, LNS's claim index, and on a SIPP plan
    additionally ``SafeIntervalIndex`` — all derive ``R``/``infl_blocked``/``infl_pad`` identically
    from ``cfg``, and within a commit they share the ``cfg`` object and backend flag too, which is
    what lets them share a key.

    The reuse window is ONE commit, and eviction is not the hazard. ``ledger.commit`` fans out
    synchronously (``for cb in self._observers``) with no yield point: the first consumer inserts a
    flight's rows, the rest read them immediately, and :func:`prepare_range_cache_for_commit` has
    already sized the cache for the whole flight. Once that fan-out returns the entries are dead —
    the next commit carries different volume objects and would miss them anyway — so evicting them
    costs nothing. The exception is LNS, whose claim index reads victims rasterized across SEVERAL
    earlier commits; that longer window is what the 1024 floor exists for.

    Per-plan callers therefore stay on the generator, NOT to protect the commit rows (routing the
    own-column overlay through here was measured answer-neutral) but because it would be pure loss:
    such a caller builds a FRESH ``hover_reservation`` per call, so its lookups all miss and each
    insert would pin a dead volume and its rows alive for the life of its cache slot.

    ``hit[0] is vol`` / ``hit[1] is cfg`` are defensive only: the entry holds a STRONG reference to
    both, so a cached object cannot be freed and its ``id`` recycled. Keep the guard — it
    is the invariant that would catch a future weak-ref rewrite.

    Parameters
    ------------
    - vol (Volume4D): the committed volume to rasterize.
    - cfg (SimConfig): supplies flight levels, step timing, and inflations; its identity is part of
      the cache key.
    - R (float): hex circumradius (m).
    - infl_blocked (float): inflation (m) for the narrower corridor footprint.
    - infl_pad (float): inflation (m) for the wider pad footprint.

    Return
    --------
    - output (list): the materialized ``(q, r, L, s_lo, s_hi, in_blocked)`` rows for ``vol``, cached
      under ``(id(vol), id(cfg), R, infl_blocked, infl_pad, backend)``.
    """
    # Backend and config identity are output inputs too. Keeping the identities in the key and the
    # objects in the value makes id reuse harmless while avoiding a hash of the full frozen config.
    backend = bool(_COMPILED and USE_COMPILED)
    key = (id(vol), id(cfg), R, infl_blocked, infl_pad, backend)
    hit = _RANGE_CACHE.get(key)
    if hit is not None and hit[0] is vol and hit[1] is cfg:
        _RANGE_CACHE.move_to_end(key)
        return hit[2]
    rows = list(rasterize_volume_ranges(vol, cfg, R, infl_blocked, infl_pad))
    _RANGE_CACHE[key] = (vol, cfg, rows)
    _RANGE_CACHE.move_to_end(key)
    while len(_RANGE_CACHE) > _RANGE_CACHE_CAP:
        _RANGE_CACHE.popitem(last=False)
    return rows



def max_lane_traverse_s(center, term, cfg: SimConfig) -> float:
    """Worst-case seconds to translate from the hub CENTRE out to one of its exit-lane cells.

    Egress is modelled sequentially — climb inside the column, then translate out — so the drone is
    still inside its own column this long after topping out, and the column must stay reserved for
    it. Taking the WORST lane keeps the window per-level, not per-(lane, level), which matters
    because ``dwell_ok_levels`` sits in the A* ground-state hot path.

    Measured off ``Lane.steps * dt``, NOT ``dist / speed``: the A* clock advances in whole ``dt``
    steps, so a continuous ``dist / speed`` window would be SHORTER than the imposed delay
    and would release the column while the drone is still inside it. The exact traverse depends on
    where the hub sits relative to the hex lattice; quantising here keeps the window and the clock
    identical by construction, which is what makes the guarantee position-independent.

    Parameters
    ------------
    - center (Vec): the hub centre (column location).
    - term (Terminal | tuple | None): the terminal, normalized via :func:`as_terminal`.
    - cfg (SimConfig): supplies the exit-lane geometry and ``dt``.

    Return
    --------
    - output (float): worst-case egress-traverse seconds; 0 when there are no lanes to traverse (no
      terminal, or ``fixed_exit_lanes`` off — the legacy path folds instead).
    """
    term = as_terminal(term)
    if term is None or not cfg.fixed_exit_lanes:
        return 0.0
    lanes = terminal_lanes(center, term, cfg)
    return max((ln.steps for ln in lanes), default=0) * cfg.dt_s
