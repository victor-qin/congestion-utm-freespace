"""Axial hex lattice in the local ENU plane + rasterization of committed volumes.

The grid is a *fixed global* pointy-top hex tiling anchored at ENU (0,0) and shared by every
flight, so the blocked-set built from committed volumes is global and incremental (the continuous
analogue of the sibling project's occupancy ledger). The pitch (centre-to-centre) is tied to
``nominal_speed · dt`` so one hex move is exactly one timestep at nominal speed — which keeps the
A* time axis clean and makes the MILP's "slow-down-for-free" / "hop-a-thin-wall" exploits
structurally impossible.

Rasterization is deliberately *conservative*: a cell is blocked if a committed volume, inflated by
the new corridor's half-width PLUS one hex circumradius, reaches its centre. Over-blocking by up to
a hex is safe — A* avoids a hair more than necessary, the NLP recovers the slack by smoothing into
the true continuous gap, and FCL verify is the backstop.
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
AXIAL_NEIGHBORS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

# ---- compiled footprint sweep (see hexgrid_kernel) ------------------------------------------
# The reference sweep below (`_candidate_slack` + a mask) stays the oracle and the fallback. The
# kernel evaluates the same candidates in compiled code and emits only the ones that pass, which is
# what removes the ~84% of candidates the host currently materialises just to discard.
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
    """Select the compiled or reference sweep and invalidate rows produced by the old backend."""
    global USE_COMPILED
    enabled = bool(enabled)
    if USE_COMPILED != enabled:
        USE_COMPILED = enabled
        cache = globals().get("_RANGE_CACHE")
        if cache is not None:
            cache.clear()


@contextmanager
def rasterizer_backend(compiled: bool):
    """Temporarily select a rasterizer backend, restoring the prior selection on every exit."""
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
    """Axial lattice distance in STEPS: the fewest 6-neighbour moves from `a` to `b`.

    Every lateral A* edge is exactly one pitch, so this times the pitch is the shortest path the
    lattice permits between two cells — the obstacle-free reference a detour is measured against.
    """
    dq, dr = b[0] - a[0], b[1] - a[1]
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2



def lattice_overhead_m(cells, pitch, air_detour_m):
    """The share of ``air_detour_m`` that is hex geometry rather than traffic, in metres.

    The traffic share is derived EXACTLY and subtracted: every lateral edge is exactly one pitch, so
    ``moves actually flown − the lattice geodesic between the first and last cell`` is the berth
    traffic forced, in whole hex steps. That residual is exactly 0 for an unimpeded flight at ANY
    bearing — which is the invariant that makes the congestion reading trustworthy.

    Everything else in ``air_detour_m`` is geometry and lands here: the staircase a 6-direction
    lattice imposes on an off-axis bearing (0 on-axis, peaking at 2/√3 − 1 ≈ 15.5% at 30° off), plus
    the endpoint snap of origin/dest onto cell centres.

    The IN-COLUMN part of the terminal fold does not land here: ``air_detour_m`` is measured exit
    lane → exit lane on both sides (issue #50), so hub centre → column edge is outside the measurement
    entirely — terminal operations, accounted as that hub's capacity.

    What DOES land here is the column edge → lane-cell hop, because A*'s path starts on a boundary
    hex rather than on the reference circle, so folding EXTENDS it (measured +169.58 m of a 724.41 m
    band on an unimpeded hub flight). That is lane snap — the same quantization as (2), just at a
    terminal — so the bucket is still "geometry, not traffic".
    """
    moves = sum(1 for a, b in zip(cells, cells[1:]) if a != b)
    forced = max(0, moves - hex_distance(cells[0], cells[-1])) * pitch
    return max(0.0, air_detour_m - forced)

def circumradius(cfg: SimConfig) -> float:
    """Hex circumradius R, from pitch = nominal_speed·dt and pitch = √3·R."""
    return cfg.nominal_speed_mps * cfg.dt_s / SQRT3


def hex_center(q: int, r: int, R: float) -> np.ndarray:
    """ENU (x, y) of the centre of axial hex (q, r) — pointy-top."""
    return np.array([R * SQRT3 * (q + r / 2.0), R * 1.5 * r])


def enu_to_axial(x: float, y: float, R: float) -> tuple[int, int]:
    """Nearest axial hex to ENU (x, y)."""
    qf = (SQRT3 / 3.0 * x - 1.0 / 3.0 * y) / R
    rf = (2.0 / 3.0 * y) / R
    return _axial_round(qf, rf)


def _axial_round(qf: float, rf: float) -> tuple[int, int]:
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
    """A hub's canonical exit/approach lane: a boundary hex just outside the column (issue #18).

    Same-hub launches are deconflicted by exact cell occupancy at plan time (``HexOccupancyService``),
    not by any per-lane graze graph — two flights whose corridors share a footprint cell serialise,
    divergent ones stay concurrent. So a lane is just its ``cell`` plus descriptors: ``bearing`` (the
    stable sort order of the ring) and ``dist`` (used by the A* heuristic and the takeoff edge cost)."""

    cell: tuple[int, int]
    bearing: float            # degrees, from the hub centre (the lane ring's stable sort key)
    dist: float               # metres, hub centre → cell centre
    steps: int = 0            # dt steps to translate ``dist`` at cruise speed (issue #52 egress)


_LANE_CACHE: dict = {}
_COVERED_CACHE: dict = {}


def _bearing_deg(cell, cx: float, cy: float, R: float) -> float:
    bx, by = hex_center(*cell, R)
    return math.degrees(math.atan2(by - cy, bx - cx))


def _covered_boundary(center, term, cfg: SimConfig) -> tuple[set, set]:
    """Flood-fill a hub's column: ``covered`` hexes (centre within ``exit_radius`` of the hub) and the
    ``boundary`` ring just outside them (neighbours of a covered cell that aren't themselves covered).
    Shared by :func:`terminal_lanes` (boundary = the exit lanes) and :func:`terminal_cells`
    (covered ∪ boundary = the full reserved terminal airspace).

    Memoised per ``(center, term, cfg)`` — hubs don't move during a run, and ``terminal_cells`` is now on
    the compiled A* hot path (the own-hub overlay, once per terminal flight); without the cache each of
    the thousands of same-hub plans re-ran this flood-fill. The returned sets are treated as read-only by
    callers (``terminal_cells`` unions into a fresh set; ``terminal_lanes`` sorts/iterates), so sharing the
    cached instances is safe. Mirrors ``_LANE_CACHE``."""
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
    (boundary). Used to wall the terminal off from FOREIGN cruise traffic when
    ``cfg.terminal_airspace_always_active`` (foreign flights route around instead of crossing); see
    :meth:`freespace_sim.ledger.ReservationLedger.register_static_terminal` (which the occupancy derives its
    routing wall from via :meth:`~freespace_sim.planner.astar.occupancy.HexOccupancyService._on_static`)."""
    covered, boundary = _covered_boundary(center, term, cfg)
    return covered | boundary


def terminal_lanes(center, term, cfg: SimConfig) -> list[Lane]:
    """A hub's fixed exit-lane set: the **boundary hexes** of its column, sorted by bearing.

    Classify hexes by centre distance to the hub: *covered* if within ``exit_radius`` (flood-filled out
    from the home hex), *boundary* if not covered but hex-adjacent to a covered cell. The boundary ring
    is the canonical exit-lane set — fully determined by the hub position and the fixed grid (no
    snapping). Deterministic in ``(center, term, cfg)`` and memoised — hubs don't move during a run.
    See issue #18."""
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
    """Yield all axial hexes whose centres could lie in the xy AABB [amin, amax] (a superset)."""
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
    fallback when numba is absent or ``USE_COMPILED`` is off."""
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
    # Expand the blocked step range by the corridor box's temporal extent: a move ARRIVING at step s
    # commits a box spanning [(s−1)·dt − buffer, s·dt + buffer], so block s if that box could overlap
    # the obstacle window — otherwise A* enters a just-cleared cell and the rebuilt box clips it.
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
    """Yield (q, r, step) cruise-level cells a committed volume blocks (conservatively inflated).

    ``infl`` overrides the footprint inflation (metres). It defaults to the corridor half-width plus
    one hex — correct for the swept corridor. Callers checking *pad* occupancy (the takeoff/landing
    hover cylinder) pass ``effective_hover_radius_m + R`` instead, so the blocked footprint matches
    the wider cylinder rather than the corridor. Vectorized — see :func:`_footprint_slack`.
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
    """One vectorized sweep yielding ``(q, r, step, in_blocked)`` over the *pad* footprint, where
    ``in_blocked`` flags membership in the (smaller) corridor footprint. Requires
    ``infl_pad >= infl_blocked`` so pad cells are a superset of blocked cells. Replaces two
    :func:`rasterize_volume` passes with a single geometry computation per volume (the A* hot path).
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
    *cell* with the step axis collapsed to its inclusive ``[s_lo, s_hi]`` range instead of one row
    per (cell, step).

    A committed volume occupies each cell over a *contiguous* step span (``_step_range`` is a plain
    ``range``), so the per-step form yields ``S`` rows the consumers then process one at a time —
    ``S`` interval-splits into the compiled pool, ``S`` dict inserts into the hex service. Handing the
    range straight through lets the compiled pool block the whole span in ONE split (issue #8 Phase E:
    the pool-splice half of the commit floor drops by the ~10-30 steps a volume spans). Byte-identical
    coverage: expanding every yielded range back over ``s_lo..s_hi`` reproduces the dual sweep exactly."""
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
# after all of them have been committed — at 128 a default 8-flight neighborhood evicted most of
# its own rows before reaching it. An entry is ~10 rows (a corridor box covers ~10 hexes), so 1024
# costs ~1 MB. Overshooting only wastes a miss's recompute, never correctness; undershooting
# silently loses the sharing, so this errs high.
#
# 1024 is a FLOOR, not the whole story: `prepare_range_cache_for_commit` raises the active cap when a
# single commit carries more volumes than that, so an unusually long trajectory still stays resident
# through the fan-out. It never lowers the cap below this floor, which is what keeps the LNS
# neighborhood window (spanning SEVERAL commits) intact.
_RANGE_CACHE_MIN_CAP = 1024
_RANGE_CACHE_CAP = 1024

_COL_HEX_CACHE: "OrderedDict[tuple, frozenset]" = OrderedDict()
# Sized like `_RANGE_CACHE` and for the same reason. The key is `(cols, R)` and `cols` comes from a
# flight's own terminal CYLINDERS, so with hub-radius demand it is really per-HUB: 476 distinct hubs
# on `density_future_wing_zipline`, which a smaller cap would thrash. An entry is ~50 hexes.
# Undershooting silently loses the sharing, so this errs high.
_COL_HEX_CACHE_CAP = 1024
# Unsynchronized, exactly like `_RANGE_CACHE`: the parallel LNS path SPAWNS processes, so each has
# its own copy and there is no cross-thread reader to race with.


def column_hexes(cols: tuple, R: float) -> frozenset:
    """Axial hexes whose CENTRE lies inside any ``(cx, cy, radius)`` disc in ``cols``.

    Four consumers ask "is this cell inside the committing flight's own terminal column?" — the two
    occupancy images, ``HexOccupancyService.enable_blocked``'s re-derive, and the LNS claim index —
    and every one of them used to ask per RASTERIZED CELL: 4.2 M calls each at density_faa, each
    allocating a numpy array via `hex_center`. The answer depends only on ``(cols, R)``, both constant
    for a flight, so it is resolved once per flight and shared.

    Exact by construction, not by approximation: the set is built with the SAME ``hex_center`` and the
    SAME ``<=`` comparison the per-cell test used, and ``_hexes_in_box`` yields a documented SUPERSET
    of the hexes whose centres could lie in a box — so every centre within ``rad`` of a disc centre,
    which necessarily lies in that disc's AABB, is enumerated."""
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
    cheap assignment and therefore see every row produced by the first. ``ReservationLedger.commit``
    invokes observers synchronously, so no unrelated rasterization can interleave with that window.

    It never drops below the floor. That matters for LNS, whose claim index reads victim rows
    rasterized across SEVERAL earlier commits — sizing down to the latest commit would evict exactly
    the rows that window exists to keep.
    """
    global _RANGE_CACHE_CAP
    _RANGE_CACHE_CAP = max(_RANGE_CACHE_MIN_CAP, len(volumes))
    while len(_RANGE_CACHE) > _RANGE_CACHE_CAP:
        _RANGE_CACHE.popitem(last=False)


def rasterize_ranges(vol: Volume4D, cfg: SimConfig, R: float, infl_blocked: float, infl_pad: float):
    """Materialized :func:`rasterize_volume_ranges`, shared between a commit's occupancy consumers.

    ~98% of the coordinator's serial commit floor is these per-commit rasterizations (issue #8 Phase
    E), and the geometry is identical between them, so it is computed once here and reused. Consumers
    are A*'s ``HexOccupancyService`` + ``CompiledHexOccupancy``, LNS's claim index, and on a SIPP plan
    additionally ``CompiledOccupancy`` + ``SafeIntervalIndex`` (issue #114) — all derive ``R``/
    ``infl_blocked``/``infl_pad`` identically from ``cfg``, and within a commit they share the ``cfg``
    object and the backend flag too, which is what makes them share a key.

    WHY THE WINDOW IS ONE COMMIT, AND WHY EVICTION IS NOT THE HAZARD. ``ledger.commit`` fans out
    synchronously (``for cb in self._observers``), so the consumers run back-to-back with no yield
    point: the first inserts a flight's rows, the rest read them immediately, and
    :func:`prepare_range_cache_for_commit` has already sized the cache to hold the whole flight. Once
    that fan-out returns the entries are dead — the next commit carries different volume objects and
    would miss on them anyway. So evicting them costs nothing, and eviction is already the steady
    state (measured on `dallas_hub_2uss_large`: 44,469 evictions, hit rate pinned at its ceiling).
    The exception is LNS, where the claim index reads victims rasterized across SEVERAL earlier
    commits — that longer window is what the 1024 floor exists for.

    Per-plan callers therefore stay on the generator NOT to protect the commit rows — routing the
    own-column overlay through here was measured to change the commit hit rate by exactly zero
    (133,791/178,388 either way) — but because it would be pure loss: that caller builds a FRESH
    ``hover_reservation`` per call, so its lookups are a measured 0/1,286 hits, all overhead, and each
    insert would pin a dead volume and its rows alive for the life of its cache slot.

    ``hit[0] is vol`` is defensive only: the entry holds a STRONG reference to ``vol``, so a cached
    volume cannot be freed and its ``id`` cannot be recycled while cached (measured: 0 guard failures
    in 179,674 lookups). Keep it — it is the invariant that would catch a future weak-ref rewrite."""
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

    The egress is modelled sequentially — climb inside the column, then translate out — so the drone
    is still inside its own column for this long after topping out, and the column must stay reserved
    for it. Taking the WORST lane keeps the window per-level rather than per-(lane, level), which
    matters because ``dwell_ok_levels`` sits in the A* ground-state hot path.

    Measured off ``Lane.steps * dt``, NOT ``dist / speed``. The A* clock advances in whole ``dt``
    steps, so a continuous ``dist / speed`` window is SHORTER than the delay the planner actually
    imposes and the column is released while the drone is still inside it — 10.583 s of window against
    a 12.000 s clock for a 180 m hub AT THE GRID ORIGIN, i.e. 1.417 s of unreserved occupancy that two
    same-hub flights could overlap in. (The exact traverse depends on where the hub sits relative to
    the hex lattice — 10.799 s at (500,500), 10.695 s at (10000,10000) — so treat the figure as
    illustrative; the quantisation below is what makes the guarantee position-independent.) Quantising here keeps the window and the clock identical by
    construction. 0 when there are no lanes to traverse (no terminal, or ``fixed_exit_lanes`` off —
    the legacy path folds instead).
    """
    term = as_terminal(term)
    if term is None or not cfg.fixed_exit_lanes:
        return 0.0
    lanes = terminal_lanes(center, term, cfg)
    return max((ln.steps for ln in lanes), default=0) * cfg.dt_s
