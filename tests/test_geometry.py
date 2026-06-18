import fcl
import numpy as np

from freespace_sim.geometry import CylinderSpec, box_from_segment, segment_frame


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
