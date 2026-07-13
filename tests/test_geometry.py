import fcl
import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.geometry import (
    WORLD_UP,
    BoxSpec,
    CylinderSpec,
    box_from_segment,
    segment_frame,
)
from freespace_sim.types import vec
from freespace_sim.volumes import build_reservation_from_corners


def _collide(a, b) -> bool:
    return fcl.collide(a.to_fcl(), b.to_fcl(), fcl.CollisionRequest(), fcl.CollisionResult()) > 0


def test_box_along_x_extent_and_aabb():
    spec = box_from_segment(np.array([0, 0, 0.0]), np.array([100, 0, 0.0]), width=60, height=30)
    L, W, H = spec.extents
    assert np.isclose(L, 100)
    assert (W, H) == (60, 30)
    lo, hi = spec.aabb()
    assert np.allclose(lo, [0, -30, -15])
    assert np.allclose(hi, [100, 30, 15])


def test_box_along_y_rotates_axes():
    spec = box_from_segment(np.array([0, 0, 0.0]), np.array([0, 100, 0.0]), width=60, height=30)
    lo, hi = spec.aabb()
    # length now runs along y, width along x
    assert np.allclose(lo, [-30, 0, -15])
    assert np.allclose(hi, [30, 100, 15])


def test_segment_frame_is_orthonormal_even_when_vertical():
    for p1 in ([100, 0, 0.0], [0, 0, 100.0], [50, 50, 50.0]):
        R, length = segment_frame(np.zeros(3), np.array(p1, float))
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
        assert np.isclose(length, np.linalg.norm(p1))


def test_box_fcl_collision_overlap_and_separation():
    a = box_from_segment(np.array([0, 0, 0.0]), np.array([100, 0, 0.0]), 60, 30)
    near = box_from_segment(np.array([50, 0, 0.0]), np.array([150, 0, 0.0]), 60, 30)
    far = box_from_segment(np.array([500, 0, 0.0]), np.array([600, 0, 0.0]), 60, 30)
    assert _collide(a, near)
    assert not _collide(a, far)


def test_cylinder_aabb_and_collision():
    cyl = CylinderSpec(cx=0, cy=0, radius=25, z_lo=0, z_hi=150)
    lo, hi = cyl.aabb()
    assert np.allclose(lo, [-25, -25, 0])
    assert np.allclose(hi, [25, 25, 150])
    box_at_cruise = box_from_segment(np.array([0, 0, 150.0]), np.array([40, 0, 150.0]), 10, 30)
    box_above = box_from_segment(np.array([0, 0, 300.0]), np.array([40, 0, 300.0]), 10, 30)
    assert _collide(cyl, box_at_cruise)       # cylinder spans z∈[0,150], box at 150 → touch/overlap
    assert not _collide(cyl, box_above)       # box at z=300 is above the cylinder


# ---------------- byte-identity of the scalarized geometry hot path (issue #30 lever #4) ----------------
# segment_frame and BoxSpec.aabb were rewritten from numpy (np.cross / |R|@half) to plain scalars, to shed
# numpy's per-call ufunc dispatch overhead. The scalar output must be BIT-FOR-BIT identical to the original
# numpy code. The two oracles below are frozen VERBATIM copies of the pre-change bodies — DO NOT EDIT them;
# they are the reference the live scalar code is pinned against (exact ==, not allclose).


def _segment_frame_numpy_original(p0, p1):
    """Frozen original (pre-scalarization) segment_frame. Do not edit — byte-identity reference only."""
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


def _boxspec_aabb_numpy_original(spec):
    """Frozen original (pre-scalarization) BoxSpec.aabb. Do not edit — byte-identity reference only."""
    c = np.array(spec.center, float)
    half = np.array(spec.extents, float) / 2.0
    ext = np.abs(np.array(spec.rot, float).reshape(3, 3)) @ half
    return c - ext, c + ext


def _real_corridor_boxes():
    """Every BoxSpec build_reservation_from_corners emits for an actual multi-corner climb/cruise/descent —
    the exact geometry the shortcut refiner rebuilds, so byte-identity is proven on real inputs, not only
    synthetic ones."""
    cfg = SimConfig()
    corners = [vec(0, 0, 30), vec(1200, 300, 70), vec(2400, -200, 110), vec(3600, 100, 30)]
    vols, *_ = build_reservation_from_corners(corners, vec(0, 0, 30), vec(3600, 100, 30), 0.0, 0.0, cfg)
    return [v.shape for v in vols if isinstance(v.shape, BoxSpec)]


def test_segment_frame_byte_identical_to_original_numpy():
    """The scalarized segment_frame is bit-for-bit identical to the frozen numpy original — random,
    degenerate/boundary, and real corridor geometry. Exact array equality + exact length equality."""
    import random
    rng = random.Random(0)

    def check(p0, p1):
        r_new, len_new = segment_frame(np.asarray(p0, float), np.asarray(p1, float))
        r_ref, len_ref = _segment_frame_numpy_original(p0, p1)
        assert np.array_equal(r_new, r_ref), f"R differs for {p0}->{p1}:\n{r_new}\n!=\n{r_ref}"
        assert len_new == len_ref, f"length {len_new!r} != {len_ref!r} for {p0}->{p1}"

    for _ in range(5000):                                  # random, varied magnitude + sign (incl. negative)
        scale = 10 ** rng.uniform(-2, 4)
        p0 = [rng.uniform(-1, 1) * scale for _ in range(3)]
        p1 = [rng.uniform(-1, 1) * scale for _ in range(3)]
        check(p0, p1)

    origin = [0.0, 0.0, 0.0]
    for p1 in ([0.0, 0.0, 0.0],                            # zero-length → length<1e-9 branch
               [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],         # ±x
               [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],         # ±y
               [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],         # exactly vertical → ref switches to world-x
               [0.99, 0.0, 0.99], [0.01, 0.0, 0.99],      # straddle the |x[2]|==0.99 reference switch
               [1.0, 1.0, 1.0], [1e-8, 0.0, 0.0], [1e6, -2e6, 3e6]):
        check(origin, p1)

    boxes = _real_corridor_boxes()                        # real geometry: reconstruct each box's segment
    assert len(boxes) > 5, "real corridor must contribute several boxes"
    for spec in boxes:
        r = np.array(spec.rot, float).reshape(3, 3)
        c = np.array(spec.center, float)
        half_len = spec.extents[0] / 2.0
        check(c - r[:, 0] * half_len, c + r[:, 0] * half_len)


def test_boxspec_aabb_byte_identical_to_original_numpy():
    """The scalarized BoxSpec.aabb is bit-for-bit identical to the frozen numpy original — random specs
    (incl. NON-orthonormal rot, to stress the |R|@half summation order) and every real corridor box."""
    import random
    rng = random.Random(1)

    def check(spec):
        lo_new, hi_new = spec.aabb()
        lo_ref, hi_ref = _boxspec_aabb_numpy_original(spec)
        assert np.array_equal(lo_new, lo_ref) and np.array_equal(hi_new, hi_ref), f"aabb differs for {spec}"

    for _ in range(5000):
        spec = BoxSpec(
            center=tuple(rng.uniform(-1e5, 1e5) for _ in range(3)),
            rot=tuple(rng.uniform(-2, 2) for _ in range(9)),        # arbitrary, non-orthonormal
            extents=tuple(rng.uniform(1e-3, 1e4) for _ in range(3)),
        )
        check(spec)

    boxes = _real_corridor_boxes()
    assert len(boxes) > 5
    for spec in boxes:
        check(spec)
