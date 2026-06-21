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

import numpy as np

from ..config import SimConfig
from ..geometry import BoxSpec
from ..volumes import Volume4D

SQRT3 = math.sqrt(3.0)
AXIAL_NEIGHBORS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


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


def _cruise_overlap(vol: Volume4D, cfg: SimConfig) -> bool:
    """Does the volume's altitude span overlap the cruise band the new corridor occupies?"""
    lo, hi = vol.aabb()
    band_lo = cfg.cruise_level_m - cfg.corridor_height_m / 2.0
    band_hi = cfg.cruise_level_m + cfg.corridor_height_m / 2.0
    return lo[2] <= band_hi and band_lo <= hi[2]


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


def _footprint_contains(shape, c: np.ndarray, infl: float, cfg: SimConfig) -> bool:
    p = np.array([c[0], c[1], cfg.cruise_level_m])
    if isinstance(shape, BoxSpec):
        local = shape.rotation().T @ (p - np.array(shape.center, float))
        half = np.array(shape.extents, float) / 2.0 + infl
        return bool(np.all(np.abs(local) <= half))
    # cylinder
    d = float(np.hypot(p[0] - shape.cx, p[1] - shape.cy))
    return d <= shape.radius + infl and (shape.z_lo - infl <= cfg.cruise_level_m <= shape.z_hi + infl)


def _footprint_slack(shape, cx: np.ndarray, cy: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Vectorized inflation margin for hex centres (cx, cy arrays): a centre is inside the footprint
    at inflation ``x`` iff ``slack <= x``. Computes the shape geometry ONCE for all candidate hexes,
    so a single matmul/reduce replaces the millions of per-hex :func:`_footprint_contains` calls.

    Equivalence to the scalar test: for a box, ``all(|local_d| <= half_d + x)`` ⟺
    ``max_d(|local_d| - half_d) <= x``; for a cylinder the radial (``d - radius``) and altitude-band
    margins both reduce to ``margin <= x``. ``rotᵀ·v`` (column) equals ``v·rot`` (row), batched.
    """
    if isinstance(shape, BoxSpec):
        center = np.array(shape.center, float)
        p = np.column_stack([cx, cy, np.full(cx.shape, cfg.cruise_level_m)])
        local = np.abs((p - center) @ shape.rotation())
        half = np.array(shape.extents, float) / 2.0
        return np.max(local - half, axis=1)
    radial = np.hypot(cx - shape.cx, cy - shape.cy) - shape.radius
    z_slack = max(shape.z_lo - cfg.cruise_level_m, cfg.cruise_level_m - shape.z_hi)
    return np.maximum(radial, z_slack)


def _candidate_slack(vol: Volume4D, cfg: SimConfig, R: float, infl: float):
    """Candidate axial hexes (centres within the volume AABB inflated by ``infl``) and each one's
    :func:`_footprint_slack`. The (q, r) enumeration reproduces :func:`_hexes_in_box` as arrays."""
    lo, hi = vol.aabb()
    amin = lo[:2] - infl
    amax = hi[:2] + infl
    qs, rs = [], []
    for x in (amin[0], amax[0]):
        for y in (amin[1], amax[1]):
            q, r = enu_to_axial(x, y, R)
            qs.append(q)
            rs.append(r)
    q_grid, r_grid = np.meshgrid(
        np.arange(min(qs) - 1, max(qs) + 2), np.arange(min(rs) - 1, max(rs) + 2), indexing="ij"
    )
    q_grid = q_grid.ravel()
    r_grid = r_grid.ravel()
    cx = R * SQRT3 * (q_grid + r_grid / 2.0)
    cy = R * 1.5 * r_grid
    return q_grid, r_grid, _footprint_slack(vol.shape, cx, cy, cfg)


def _step_range(vol: Volume4D, cfg: SimConfig) -> range:
    # Expand the blocked step range by the corridor box's temporal extent: a move ARRIVING at step s
    # commits a box spanning [(s−1)·dt − buffer, s·dt + buffer], so block s if that box could overlap
    # the obstacle window — otherwise A* enters a just-cleared cell and the rebuilt box clips it.
    dt = cfg.dt_s
    s0 = int(math.floor((vol.t_start - cfg.time_buffer_s) / dt))
    s1 = int(math.floor((vol.t_end + dt + cfg.time_buffer_s) / dt))
    return range(s0, s1 + 1)


def rasterize_volume(vol: Volume4D, cfg: SimConfig, R: float, infl: float | None = None):
    """Yield (q, r, step) cruise-level cells a committed volume blocks (conservatively inflated).

    ``infl`` overrides the footprint inflation (metres). It defaults to the corridor half-width plus
    one hex — correct for the swept corridor. Callers checking *pad* occupancy (the takeoff/landing
    hover cylinder) pass ``effective_hover_radius_m + R`` instead, so the blocked footprint matches
    the wider cylinder rather than the corridor. Vectorized — see :func:`_footprint_slack`.
    """
    if not _cruise_overlap(vol, cfg):
        return
    if infl is None:
        infl = cfg.corridor_width_m / 2.0 + R      # corridor half-width + one hex (conservative)
    q_grid, r_grid, slack = _candidate_slack(vol, cfg, R, infl)
    mask = slack <= infl
    steps = _step_range(vol, cfg)
    for q, r in zip(q_grid[mask].tolist(), r_grid[mask].tolist()):
        for s in steps:
            yield q, r, s


def rasterize_volume_dual(
    vol: Volume4D, cfg: SimConfig, R: float, infl_blocked: float, infl_pad: float
):
    """One vectorized sweep yielding ``(q, r, step, in_blocked)`` over the *pad* footprint, where
    ``in_blocked`` flags membership in the (smaller) corridor footprint. Requires
    ``infl_pad >= infl_blocked`` so pad cells are a superset of blocked cells. Replaces two
    :func:`rasterize_volume` passes with a single geometry computation per volume (the A* hot path).
    """
    if not _cruise_overlap(vol, cfg):
        return
    q_grid, r_grid, slack = _candidate_slack(vol, cfg, R, infl_pad)
    in_pad = slack <= infl_pad
    in_blk = (slack[in_pad] <= infl_blocked).tolist()
    qp = q_grid[in_pad].tolist()
    rp = r_grid[in_pad].tolist()
    steps = _step_range(vol, cfg)
    for q, r, b in zip(qp, rp, in_blk):
        for s in steps:
            yield q, r, s, b
