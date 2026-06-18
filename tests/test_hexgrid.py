import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.types import vec
from freespace_sim.volumes import corridor_segment_volume

CFG = SimConfig()
R = hg.circumradius(CFG)


def test_pitch_is_speed_times_dt():
    pitch = hg.SQRT3 * R
    assert abs(pitch - CFG.nominal_speed_mps * CFG.dt_s) < 1e-6   # one hex move == one timestep


def test_axial_enu_round_trip():
    for q, r in [(0, 0), (3, -1), (-2, 5), (17, 0)]:
        c = hg.hex_center(q, r, R)
        assert hg.enu_to_axial(c[0], c[1], R) == (q, r)


def test_neighbors_are_one_pitch_away():
    c0 = hg.hex_center(0, 0, R)
    pitch = hg.SQRT3 * R
    for dq, dr in hg.AXIAL_NEIGHBORS:
        assert abs(float(np.linalg.norm(hg.hex_center(dq, dr, R) - c0)) - pitch) < 1e-6


def test_rasterize_blocks_cells_near_a_corridor_and_not_far():
    box = corridor_segment_volume(
        vec(1000, 0, CFG.cruise_level_m), 0.0, vec(1120, 0, CFG.cruise_level_m), CFG.dt_s, CFG
    )
    cells = set(hg.rasterize_volume(box, CFG, R))
    assert cells                                            # blocks something
    near_q, near_r = hg.enu_to_axial(1050, 0, R)
    assert (near_q, near_r, 0) in cells                     # a cell on the corridor is blocked
    far_q, far_r = hg.enu_to_axial(5000, 3000, R)
    assert (far_q, far_r, 0) not in cells                   # a distant cell is not
