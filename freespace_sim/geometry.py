"""3D geometry primitives backed by python-fcl.

Two shapes carry every reservation:
- an **oriented Box** for a corridor segment (ASTM trajectory-based volume), and
- a vertical **Cylinder** for a hover reservation (ASTM area-based volume).

Each shape is stored as a small, immutable *spec* of plain floats (so `Volume4D` stays
hashable/serialisable) and builds an `fcl.CollisionObject` on demand. Specs also expose a
world-frame axis-aligned bounding box (AABB) for the ledger's cheap broadphase prune.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import fcl
import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])


def _cross3(a, b):
    """3-vector cross product as plain scalars — bit-for-bit identical to ``np.cross`` for length-3
    inputs, but without numpy's per-call ufunc dispatch (``moveaxis`` / ``normalize_axis_tuple``), which
    dominates the cost on length-3 arrays. Same scalar-hot-path idiom as the ledger's ``_aabb_miss`` and
    A*'s ``h_air``."""
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def segment_frame(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float]:
    """Orthonormal rotation whose local x-axis runs p0→p1 (length returned separately).

    Columns are the local axes expressed in world coordinates (local→world), which is exactly
    what ``fcl.Transform`` wants. The lateral (y) axis is chosen perpendicular to both the segment
    and world-up so a level corridor is "flat"; for a (near-)vertical segment we fall back to
    world-x as the reference to avoid a degenerate cross product.

    Axes are computed with scalars (cross via :func:`_cross3`, norm via ``math.sqrt``) — bit-for-bit
    identical to the numpy form (see ``tests/test_geometry.py`` for the frozen-numpy byte-identity oracle)
    but without the per-call ufunc dispatch, since ``segment_frame`` runs once per corridor sub-box
    (hundreds of thousands of times per refined plan).
    """
    d = np.asarray(p1, float) - np.asarray(p0, float)
    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)        # == float(np.linalg.norm(d))
    if length < 1e-9:
        return np.eye(3), 0.0
    x = (dx / length, dy / length, dz / length)
    # ref = WORLD_UP unless near-vertical; |dot(x, WORLD_UP)| == |x[2]| since WORLD_UP = (0, 0, 1)
    ref = (0.0, 0.0, 1.0) if abs(x[2]) < 0.99 else (1.0, 0.0, 0.0)
    yx, yy, yz = _cross3(ref, x)
    yn = math.sqrt(yx * yx + yy * yy + yz * yz)            # == np.linalg.norm(y)
    y = (yx / yn, yy / yn, yz / yn)
    z = _cross3(x, y)
    R = np.array([[x[0], y[0], z[0]],                      # columns x, y, z == np.column_stack([x, y, z])
                  [x[1], y[1], z[1]],
                  [x[2], y[2], z[2]]])
    return R, length


@dataclass(frozen=True)
class BoxSpec:
    """Oriented 3D box: full extents (L, W, H) in a local frame, posed at ``center``."""

    center: tuple[float, float, float]
    rot: tuple[float, ...]                  # 9 values, row-major 3x3 (local→world)
    extents: tuple[float, float, float]     # full lengths L, W, H

    def rotation(self) -> np.ndarray:
        return np.array(self.rot, float).reshape(3, 3)

    def to_fcl(self) -> fcl.CollisionObject:
        L, W, H = self.extents
        tf = fcl.Transform(self.rotation(), np.array(self.center, float))
        return fcl.CollisionObject(fcl.Box(L, W, H), tf)

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        # world half-extent |R| @ half, from the flat rot tuple with scalars — bit-for-bit identical to the
        # numpy matmul (verified) but without rebuilding a 3x3 array + ufunc dispatch on every call (aabb
        # runs >1e6 times per refined plan via the ledger broadphase). rotation() is left intact for its
        # matrix consumers (hexgrid / opt / milp / viz).
        r = self.rot
        h0, h1, h2 = self.extents[0] / 2.0, self.extents[1] / 2.0, self.extents[2] / 2.0   # == extents / 2
        ext = np.array([abs(r[0]) * h0 + abs(r[1]) * h1 + abs(r[2]) * h2,
                        abs(r[3]) * h0 + abs(r[4]) * h1 + abs(r[5]) * h2,
                        abs(r[6]) * h0 + abs(r[7]) * h1 + abs(r[8]) * h2])
        c = np.array(self.center, float)
        return c - ext, c + ext


@dataclass(frozen=True)
class CylinderSpec:
    """Vertical cylinder (axis along world-z): radius and altitude band [z_lo, z_hi]."""

    cx: float
    cy: float
    radius: float
    z_lo: float
    z_hi: float

    def to_fcl(self) -> fcl.CollisionObject:
        height = self.z_hi - self.z_lo
        cz = (self.z_lo + self.z_hi) / 2.0
        tf = fcl.Transform(np.eye(3), np.array([self.cx, self.cy, cz], float))
        return fcl.CollisionObject(fcl.Cylinder(self.radius, height), tf)

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array([self.cx - self.radius, self.cy - self.radius, self.z_lo], float),
            np.array([self.cx + self.radius, self.cy + self.radius, self.z_hi], float),
        )


def box_from_segment(p0: np.ndarray, p1: np.ndarray, width: float, height: float) -> BoxSpec:
    """Build an oriented box bounding the segment p0→p1 with the given lateral width and height."""
    R, length = segment_frame(p0, p1)
    center = (np.asarray(p0, float) + np.asarray(p1, float)) / 2.0
    return BoxSpec(
        center=tuple(center.tolist()),
        rot=tuple(R.flatten().tolist()),
        extents=(max(length, 1e-6), float(width), float(height)),
    )
