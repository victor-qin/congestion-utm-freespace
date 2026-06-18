"""3D geometry primitives backed by python-fcl.

Two shapes carry every reservation:
- an **oriented Box** for a corridor segment (ASTM trajectory-based volume), and
- a vertical **Cylinder** for a hover reservation (ASTM area-based volume).

Each shape is stored as a small, immutable *spec* of plain floats (so `Volume4D` stays
hashable/serialisable) and builds an `fcl.CollisionObject` on demand. Specs also expose a
world-frame axis-aligned bounding box (AABB) for the ledger's cheap broadphase prune.
"""

from __future__ import annotations

from dataclasses import dataclass

import fcl
import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])


def segment_frame(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, float]:
    """Orthonormal rotation whose local x-axis runs p0→p1 (length returned separately).

    Columns are the local axes expressed in world coordinates (local→world), which is exactly
    what ``fcl.Transform`` wants. The lateral (y) axis is chosen perpendicular to both the segment
    and world-up so a level corridor is "flat"; for a (near-)vertical segment we fall back to
    world-x as the reference to avoid a degenerate cross product.
    """
    d = np.asarray(p1, float) - np.asarray(p0, float)
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return np.eye(3), 0.0
    x = d / length
    ref = WORLD_UP if abs(float(np.dot(x, WORLD_UP))) < 0.99 else np.array([1.0, 0.0, 0.0])
    y = np.cross(ref, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack([x, y, z]), length


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
        c = np.array(self.center, float)
        half = np.array(self.extents, float) / 2.0
        ext = np.abs(self.rotation()) @ half     # world half-extent of an oriented box
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
