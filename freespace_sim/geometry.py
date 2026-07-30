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


def _segment_frame_scalars(p0, p1) -> tuple[tuple[float, ...], float]:
    """Scalar core of :func:`segment_frame`: the 9 row-major rotation floats (local→world, columns x/y/z)
    plus the segment length, computed with plain scalars — no ``np.array`` build.

    :func:`segment_frame` wraps these into the 3x3 ``np.ndarray`` its matrix consumers + the frozen
    byte-identity oracle expect; :func:`box_from_segment` (which stores ``rot`` FLAT anyway) consumes the
    tuple directly, skipping a per-sub-box array build + ``flatten().tolist()`` round-trip. Bit-for-bit
    identical to the numpy form — the same scalar idiom already pinned for the axes here and for
    ``aabb`` / ``segment_overlaps_column`` (issue #30). The degenerate (near-zero length) case returns the
    identity frame flattened + length ``0.0``, exactly as the numpy original did (``np.eye(3), 0.0``)."""
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    dz = float(p1[2]) - float(p0[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)        # == float(np.linalg.norm(p1 - p0)) on a 3-vector
    if length < 1e-9:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), 0.0   # np.eye(3) flattened, length 0
    x = (dx / length, dy / length, dz / length)
    # ref = WORLD_UP unless near-vertical; |dot(x, WORLD_UP)| == |x[2]| since WORLD_UP = (0, 0, 1)
    ref = (0.0, 0.0, 1.0) if abs(x[2]) < 0.99 else (1.0, 0.0, 0.0)
    yx, yy, yz = _cross3(ref, x)
    yn = math.sqrt(yx * yx + yy * yy + yz * yz)            # == np.linalg.norm(y)
    y = (yx / yn, yy / yn, yz / yn)
    z = _cross3(x, y)
    # row-major flatten of the 3x3 whose COLUMNS are x, y, z == np.column_stack([x, y, z]).flatten()
    return (x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]), length


def segment_frame(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float]:
    """Orthonormal rotation whose local x-axis runs p0→p1 (length returned separately).

    Columns are the local axes expressed in world coordinates (local→world), which is exactly
    what ``fcl.Transform`` wants. The lateral (y) axis is chosen perpendicular to both the segment
    and world-up so a level corridor is "flat"; for a (near-)vertical segment we fall back to
    world-x as the reference to avoid a degenerate cross product.

    Thin ``np.ndarray`` wrapper over :func:`_segment_frame_scalars` (the scalar core), kept for its matrix
    consumers + the frozen-numpy byte-identity oracle in ``tests/test_geometry.py``. The scalar axes shed
    numpy's per-call ufunc dispatch; ``box_from_segment`` bypasses this wrapper entirely on the hot path.
    """
    rot, length = _segment_frame_scalars(p0, p1)
    if length == 0.0:                                      # degenerate (near-zero length) → identity frame
        return np.eye(3), 0.0
    R = np.array([[rot[0], rot[1], rot[2]],                # columns x, y, z == np.column_stack([x, y, z])
                  [rot[3], rot[4], rot[5]],
                  [rot[6], rot[7], rot[8]]])
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

    def flat_aabb(self) -> tuple[float, float, float, float, float, float]:
        """World AABB as six plain floats ``(xmin, ymin, zmin, xmax, ymax, zmax)`` — the allocation-free
        twin of :meth:`aabb` for the scalar broadphase hot path (``ledger._flat_aabb`` /
        ``terminal_capacity.column_clear``), which only ever want floats yet paid for two throwaway
        ``np.array``s per call (~1e6/refined plan; the profile's #1 self-time line). Mirrors :meth:`aabb`'s
        exact expressions and left-to-right summation order, and is pinned bit-for-bit against it in
        ``tests/test_geometry.py`` (which in turn pins ``aabb`` to the frozen numpy oracle)."""
        r = self.rot
        h0, h1, h2 = self.extents[0] / 2.0, self.extents[1] / 2.0, self.extents[2] / 2.0
        ex = abs(r[0]) * h0 + abs(r[1]) * h1 + abs(r[2]) * h2
        ey = abs(r[3]) * h0 + abs(r[4]) * h1 + abs(r[5]) * h2
        ez = abs(r[6]) * h0 + abs(r[7]) * h1 + abs(r[8]) * h2
        cx, cy, cz = self.center
        return (cx - ex, cy - ey, cz - ez, cx + ex, cy + ey, cz + ez)


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

    def flat_aabb(self) -> tuple[float, float, float, float, float, float]:
        """World AABB as six plain floats — the allocation-free twin of :meth:`aabb` (see
        ``BoxSpec.flat_aabb``); pinned bit-for-bit against :meth:`aabb` in ``tests/test_geometry.py``."""
        return (self.cx - self.radius, self.cy - self.radius, self.z_lo,
                self.cx + self.radius, self.cy + self.radius, self.z_hi)


def box_from_segment(p0: np.ndarray, p1: np.ndarray, width: float, height: float) -> BoxSpec:
    """Build an oriented box bounding the segment p0→p1 with the given lateral width and height.

    Consumes :func:`_segment_frame_scalars` (flat rot floats) and builds the center with scalars, so the
    hot per-sub-box path (hundreds of thousands of BoxSpecs per refined plan) allocates no intermediate
    ``np.ndarray``. The ``rot`` / ``center`` tuples are byte-identical to the prior
    ``tuple(R.flatten().tolist())`` / ``tuple(((p0 + p1) / 2).tolist())`` (pinned in ``tests/test_geometry.py``).
    """
    rot, length = _segment_frame_scalars(p0, p1)
    cx = (float(p0[0]) + float(p1[0])) / 2.0
    cy = (float(p0[1]) + float(p1[1])) / 2.0
    cz = (float(p0[2]) + float(p1[2])) / 2.0
    return BoxSpec(
        center=(cx, cy, cz),
        rot=rot,
        extents=(max(length, 1e-6), float(width), float(height)),
    )
