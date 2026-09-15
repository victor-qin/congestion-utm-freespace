"""Compiled (numba) footprint sweeps for :mod:`freespace_sim.planner.hexgrid`.

Rasterisation turns a committed ``Volume4D`` into the hex cells it blocks. It is not on the path
into the ledger — ``commit()`` stores the continuous volume verbatim — it runs in the ledger's
subscribers (``HexOccupancyService`` / ``CompiledHexOccupancy``), rebuilding the discrete obstacle
map the next A* search reads. Under LNS that map is rebuilt constantly: every iteration re-commits a
neighbourhood, and a rejected iteration commits twice (the repair, then the restored incumbent).

Why this is compiled. The reference sweep (``hexgrid._candidate_slack`` + a mask) enumerates the
axial bounding rectangle of the inflated AABB and evaluates every candidate with numpy — dozens of
tiny-array candidates per volume, only a handful kept, so the ufunc/BLAS call overhead dwarfs the
arithmetic. Tightening the enumeration is a dead end: the waste is the inflated AABB against the
oriented box, not the axial rectangle against the AABB, so a tighter rectangle removes little.
The fix is instead to make each candidate cheap AND to emit only the cells that pass, so the host
never materialises the ~84% that don't.

Relationship to the reference. ``_candidate_slack`` / ``_footprint_slack`` stay in ``hexgrid``
unchanged and remain the oracle. The scalar box arithmetic is NOT bit-identical to numpy's matrix
multiply, so this kernel marks cells close enough to either inflation threshold for the host to
re-evaluate with that oracle; it also emits near-pad cells just outside the scalar threshold, since
rounding may put the numpy result just inside. Ordinary cells stay entirely in compiled code; the
boundary repair makes the public result decision-identical for every finite input rather than
depending on a measured scenario margin. The cylinder path is already bit-identical (numba's
``np.hypot`` is numpy's).

Conventions follow ``astar_kernel``: ``@njit(cache=True, nogil=True)``, flat scalars in,
caller-allocated output arrays, no Python objects and no rounding (``hexgrid._axial_round`` uses
Python's banker's ``round``, whose numba semantics differ, so the candidate rectangle is computed by
the host and passed in).
"""

from __future__ import annotations

import numpy as np
from numba import njit

_SQRT3 = 1.7320508075688772
# Conservative roundoff envelope for the difference between two length-three dot-product
# evaluations (numba scalars vs numpy matmul), including abs/subtract/max and threshold comparison.
# The scale is the sum of the absolute products, so cancellation cannot make the guard too small.
_ROUND_GUARD = 64.0 * np.finfo(np.float64).eps


@njit(cache=True, nogil=True)
def sweep_box(q0, q1, r0, r1, R,
              ox, oy, oz, m0, m1, m2, m3, m4, m5, m6, m7, m8, h0, h1, h2,
              z, infl_pad, infl_blk, out_q, out_r, out_b, out_ambiguous):
    """Kept cells of an oriented box's footprint at altitude probe ``z``.

    Mirrors ``_footprint_slack``: for a box, ``all(|local_d| <= half_d + x)`` iff
    ``max_d(|local_d| - half_d) <= x``, and ``rot^T . v`` (column) equals ``v . rot`` (row), so
    column ``j`` of ``(p - centre) @ rot`` is ``dx*m[j] + dy*m[3+j] + dz*m[6+j]``.

    The loop nest is q outer, r inner because that is the order
    ``meshgrid(..., indexing="ij").ravel()`` produces in the reference. Transposing it would still
    emit every correct cell, but in a different SEQUENCE — which silently reorders
    ``HexOccupancyService._rows``, ``CompiledHexOccupancy._claims`` and the interval pool's
    ``block_range`` applications. Nothing would raise, so the parity test compares ORDERED rows.

    Parameters
    ------------
    - q0, q1, r0, r1 (int): inclusive axial candidate rectangle (computed host-side; the module
      docstring explains why rounding stays in Python).
    - R (float): hex circumradius (m).
    - ox, oy, oz (float): box centre in ENU (m).
    - m0..m8 (float): ``BoxSpec.rot`` row-major.
    - h0, h1, h2 (float): box half-extents (m).
    - z (float): altitude probe (m).
    - infl_pad, infl_blk (float): pad and blocked inflation (m), ``infl_pad >= infl_blk``.
    - out_q, out_r (int[]): caller-allocated output arrays receiving the kept cells' axial coords.
    - out_b (bool[]): output flag per kept cell — inside the blocked (narrower) footprint.
    - out_ambiguous (bool[]): output flag per kept cell — within the roundoff guard of a threshold,
      so the host must re-check it against the numpy oracle.

    Return
    --------
    - output (int): the number of kept cells written to the head of each output array.
    """
    n = 0
    dz = z - oz
    for q in range(q0, q1 + 1):
        for r in range(r0, r1 + 1):
            cx = R * _SQRT3 * (q + r / 2.0)
            cy = R * 1.5 * r
            dx = cx - ox
            dy = cy - oy
            x0, y0, z0 = dx * m0, dy * m3, dz * m6
            x1, y1, z1 = dx * m1, dy * m4, dz * m7
            x2, y2, z2 = dx * m2, dy * m5, dz * m8
            s = abs(x0 + y0 + z0) - h0
            l1 = abs(x1 + y1 + z1) - h1
            if l1 > s:
                s = l1
            l2 = abs(x2 + y2 + z2) - h2
            if l2 > s:
                s = l2
            scale = abs(x0) + abs(y0) + abs(z0) + abs(h0)
            scale1 = abs(x1) + abs(y1) + abs(z1) + abs(h1)
            if scale1 > scale:
                scale = scale1
            scale2 = abs(x2) + abs(y2) + abs(z2) + abs(h2)
            if scale2 > scale:
                scale = scale2
            if abs(infl_pad) > scale:
                scale = abs(infl_pad)
            if abs(infl_blk) > scale:
                scale = abs(infl_blk)
            if scale < 1.0:
                scale = 1.0
            guard = _ROUND_GUARD * scale
            near_pad = abs(s - infl_pad) <= guard
            near_blk = abs(s - infl_blk) <= guard
            # Include near-pad scalar misses: numpy may round to the other side of the threshold.
            if s <= infl_pad + guard:
                out_q[n] = q
                out_r[n] = r
                out_b[n] = s <= infl_blk
                out_ambiguous[n] = near_pad or near_blk
                n += 1
    return n


@njit(cache=True, nogil=True)
def sweep_cyl(q0, q1, r0, r1, R, ccx, ccy, rad, z_lo, z_hi,
              z, infl_pad, infl_blk, out_q, out_r, out_b):
    """Kept cells of a vertical cylinder's footprint at altitude probe ``z``.

    Mirrors ``_footprint_slack``'s cylinder branch: the radial margin ``hypot(dx, dy) - radius`` and
    the altitude-band margin ``max(z_lo - z, z - z_hi)``, combined with ``maximum``. ``np.hypot`` is
    deliberate and not interchangeable with ``sqrt(dx*dx + dy*dy)`` — hypot is correctly rounded and
    overflow-safe, and the naive form differs in the last ULP; numba's ``np.hypot`` measures
    bit-identical to numpy's over every committed cylinder in a density cut.

    Parameters
    ------------
    - q0, q1, r0, r1 (int): inclusive axial candidate rectangle (computed host-side).
    - R (float): hex circumradius (m).
    - ccx, ccy (float): cylinder centre in ENU (m).
    - rad (float): cylinder radius (m).
    - z_lo, z_hi (float): cylinder altitude band (m).
    - z (float): altitude probe (m).
    - infl_pad, infl_blk (float): pad and blocked inflation (m), ``infl_pad >= infl_blk``.
    - out_q, out_r (int[]): caller-allocated output arrays receiving the kept cells' axial coords.
    - out_b (bool[]): output flag per kept cell — inside the blocked (narrower) footprint.

    Return
    --------
    - output (int): the number of kept cells written to the head of each output array.
    """
    n = 0
    zs = z_lo - z
    if z - z_hi > zs:
        zs = z - z_hi
    for q in range(q0, q1 + 1):                     # q-major, r-minor — see `sweep_box`
        for r in range(r0, r1 + 1):
            cx = R * _SQRT3 * (q + r / 2.0)
            cy = R * 1.5 * r
            s = np.hypot(cx - ccx, cy - ccy) - rad
            if zs > s:
                s = zs
            if s <= infl_pad:
                out_q[n] = q
                out_r[n] = r
                out_b[n] = s <= infl_blk
                n += 1
    return n
