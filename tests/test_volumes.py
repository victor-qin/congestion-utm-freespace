import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.conflict import volumes_conflict
from freespace_sim.types import vec
from freespace_sim.volumes import Volume4D, build_corridor, hover_reservation

CFG = SimConfig()


def _straight_centerline():
    # cruise leg along +x at the cruise level, one waypoint per timestep
    seg = CFG.corridor_segment_len_m  # 120 m
    return [
        (vec(0, 0, CFG.cruise_level_m), 0.0),
        (vec(seg, 0, CFG.cruise_level_m), CFG.dt_s),
        (vec(2 * seg, 0, CFG.cruise_level_m), 2 * CFG.dt_s),
    ]


def test_corridor_volume_count_and_time_monotone():
    vols = build_corridor(_straight_centerline(), CFG)
    assert len(vols) == 2
    assert all(isinstance(v, Volume4D) for v in vols)
    assert vols[0].t_start < vols[1].t_start
    assert vols[0].t_end < vols[1].t_end


def test_corridor_consecutive_boxes_overlap_contiguity():
    # ASTM §4.3.5: consecutive trajectory volumes overlap in space AND time → conflict-positive.
    vols = build_corridor(_straight_centerline(), CFG)
    assert volumes_conflict(vols[0], vols[1])


def test_corridor_time_windows_buffered():
    vols = build_corridor(_straight_centerline(), CFG)
    # first segment spans [0, dt] before buffering
    assert np.isclose(vols[0].t_start, -CFG.time_buffer_s)
    assert np.isclose(vols[0].t_end, CFG.dt_s + CFG.time_buffer_s)


def test_hover_reservation_geometry_and_window():
    h = hover_reservation(vec(100, 200, 0), t0=10.0, cfg=CFG)
    lo, hi = h.aabb()
    assert np.allclose(lo, [100 - CFG.effective_hover_radius_m, 200 - CFG.effective_hover_radius_m, 0])
    assert np.allclose(hi, [100 + CFG.effective_hover_radius_m, 200 + CFG.effective_hover_radius_m, CFG.cruise_level_m])
    assert np.isclose(h.t_start, 10.0)
    assert np.isclose(h.t_end, 10.0 + CFG.hover_time_s + CFG.climb_time_s)
