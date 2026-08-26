"""Compiled (numba) footprint sweeps for :mod:`freespace_sim.planner.hexgrid`.

Rasterisation turns a committed ``Volume4D`` into the hex cells it blocks. It is not on the path
*into* the ledger — ``commit()`` stores the continuous volume verbatim — it runs in the ledger's
subscribers (``HexOccupancyService`` / ``CompiledHexOccupancy``), rebuilding the discrete obstacle
map the next A* search reads. Under LNS that map is rebuilt constantly: every iteration re-commits
a neighbourhood, and a *rejected* iteration commits twice (the repair, then the restored incumbent).

**Why this is compiled.** The reference sweep (``hexgrid._candidate_slack`` + a mask) enumerates the
axial bounding rectangle of the inflated AABB and evaluates every candidate with numpy — measured
at **62 candidates per volume of which 10 are kept**, for ~10 ufunc/BLAS calls on 62-element arrays.
At that size the arithmetic is free and the call overhead is everything (~3.4 us per numpy entry,
33.9 us per volume total). Tightening the enumeration instead is measured dead: the exact per-row
``q`` interval only removes 11% of the candidates, because the waste is the inflated AABB against
the oriented box, not the axial rectangle against the AABB. So the fix is to make each candidate
cheap rather than to have fewer of them — and to **emit only the cells that pass**, so the host
never materialises the ~84% that don't.

**Relationship to the reference.** ``_candidate_slack`` / ``_footprint_slack`` stay in ``hexgrid``
unchanged and remain the oracle: ``hexgrid.USE_COMPILED = False`` restores them bit-for-bit, and the
scalar-geometry oracle in ``tests/test_hexgrid.py`` gates both paths. These kernels are **not**
bit-identical to numpy on the box path — numpy's ``(N,3) @ (3,3)`` does not sum in the order a
register-scalar expression does. Measured over 359,512 real cell evaluations: 84.8% bit-identical,
**max |delta| 1.1e-13 m, zero kept-set flips**, against a boundary margin where **0 of 370,077
cells** came within 1e-9 m of an inflation threshold. The cylinder path measures **100% identical**
(numba's ``np.hypot`` is numpy's). The contract is therefore *decision*-identical, verified per
volume on whole scenarios — see ``.context/perf/verify_rasteriser.py``.

Conventions follow ``astar_kernel``: ``@njit(cache=True, nogil=True)``, flat scalars in,
caller-allocated output arrays, no Python objects and no rounding (``hexgrid._axial_round`` uses
Python's banker's ``round``, whose numba semantics differ, so the candidate rectangle is computed by
the host and passed in).
"""

from __future__ import annotations

import numpy as np
from numba import njit

_SQRT3 = 1.7320508075688772


@njit(cache=True, nogil=True)
def sweep_box(q0, q1, r0, r1, R,
              ox, oy, oz, m0, m1, m2, m3, m4, m5, m6, m7, m8, h0, h1, h2,
              z, infl_pad, infl_blk, out_q, out_r, out_b):
    """Kept cells of an oriented box's footprint at altitude probe ``z``. Returns the count.

    ``m0..m8`` is ``BoxSpec.rot`` row-major, ``h*`` the half-extents. Mirrors ``_footprint_slack``:
    for a box, ``all(|local_d| <= half_d + x)`` iff ``max_d(|local_d| - half_d) <= x``, and
    ``rot^T . v`` (column) equals ``v . rot`` (row), so column ``j`` of ``(p - centre) @ rot`` is
    ``dx*m[j] + dy*m[3+j] + dz*m[6+j]``.

    The loop nest is **q outer, r inner** because that is the order
    ``meshgrid(..., indexing="ij").ravel()`` produces in the reference. Transposing it would still
    emit every correct cell, but in a different sequence — which silently reorders
    ``HexOccupancyService._rows``, ``CompiledHexOccupancy._claims`` and the interval pool's
    ``block_range`` applications. Nothing would raise, so the parity test compares ORDERED rows.
    """
    n = 0
    dz = z - oz                                     # loop-invariant, but the three PRODUCTS below
    #                                                 are deliberately NOT hoisted: lifting `dz*m6`
    #                                                 out changes how many multiplies the adds can
    #                                                 contract into FMAs, which changes rounding.
    #                                                 This expression is the one parity was measured
    #                                                 on (84.8% bit-identical, max delta 1.1e-13 m).
    for q in range(q0, q1 + 1):
        for r in range(r0, r1 + 1):
            cx = R * _SQRT3 * (q + r / 2.0)
            cy = R * 1.5 * r
            dx = cx - ox
            dy = cy - oy
            s = abs(dx * m0 + dy * m3 + dz * m6) - h0
            l1 = abs(dx * m1 + dy * m4 + dz * m7) - h1
            if l1 > s:
                s = l1
            l2 = abs(dx * m2 + dy * m5 + dz * m8) - h2
            if l2 > s:
                s = l2
            if s <= infl_pad:
                out_q[n] = q
                out_r[n] = r
                out_b[n] = s <= infl_blk
                n += 1
    return n


@njit(cache=True, nogil=True)
def sweep_cyl(q0, q1, r0, r1, R, ccx, ccy, rad, z_lo, z_hi,
              z, infl_pad, infl_blk, out_q, out_r, out_b):
    """Kept cells of a vertical cylinder's footprint at altitude probe ``z``. Returns the count.

    Mirrors ``_footprint_slack``'s cylinder branch: the radial margin ``hypot(dx, dy) - radius`` and
    the altitude-band margin ``max(z_lo - z, z - z_hi)``, combined with ``maximum``. ``np.hypot`` is
    deliberate and not interchangeable with ``sqrt(dx*dx + dy*dy)`` — hypot is correctly rounded and
    overflow-safe, and the naive form differs in the last ULP. numba's ``np.hypot`` measures
    bit-identical to numpy's over every committed cylinder in a density cut.
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
