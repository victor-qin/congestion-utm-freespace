import math

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.types import vec
from freespace_sim.volumes import corridor_segment_volume, hover_reservation

CFG = SimConfig()
R = hg.circumradius(CFG)


def _scalar_rasterize(vol, cfg, r_circ, infl):
    """The pre-vectorization scalar algorithm, kept as an independent reference oracle — now probed
    once per flight level it reaches, yielding ``(q, r, L, s)``."""
    levels = hg._levels_overlapped(vol, cfg)
    if not levels:
        return set()
    s0 = int(math.floor((vol.t_start - cfg.time_buffer_s) / cfg.dt_s))
    s1 = int(math.floor((vol.t_end + cfg.dt_s + cfg.time_buffer_s) / cfg.dt_s))
    lo, hi = vol.aabb()
    amin = lo[:2] - infl
    amax = hi[:2] + infl
    out = set()
    for L in levels:
        z = cfg.flight_levels_m[L]
        for q, r in hg._hexes_in_box(amin, amax, r_circ):
            if hg._footprint_contains(vol.shape, hg.hex_center(q, r, r_circ), infl, cfg, z=z):
                out.update((q, r, L, s) for s in range(s0, s1 + 1))
    return out


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
    assert any((near_q, near_r, L, 0) in cells for L in range(CFG.n_levels))   # on the corridor
    far_q, far_r = hg.enu_to_axial(5000, 3000, R)
    assert not any((far_q, far_r, L, 0) in cells for L in range(CFG.n_levels))  # distant: clear


def test_vectorized_rasterize_matches_scalar_reference():
    """The vectorized rasterizer (single + dual) must be byte-identical to the scalar oracle, across
    several box orientations and a hover cylinder."""
    z = CFG.cruise_level_m
    vols = [
        corridor_segment_volume(vec(800, 200, z), 40.0, vec(920, 260, z), 44.0, CFG),   # diagonal
        corridor_segment_volume(vec(0, 0, z), 0.0, vec(120, 0, z), 4.0, CFG),           # axis-aligned
        corridor_segment_volume(vec(-500, 300, z), 100.0, vec(-560, 420, z), 104.0, CFG),
        hover_reservation(vec(1500, -700, 0.0), 60.0, CFG),                             # cylinder
    ]
    infl_b = CFG.corridor_width_m / 2.0 + R
    infl_p = CFG.effective_hover_radius_m + R
    assert infl_p >= infl_b
    for v in vols:
        # single-inflation path == scalar oracle (default corridor inflation)
        assert set(hg.rasterize_volume(v, CFG, R)) == _scalar_rasterize(v, CFG, R, infl_b)
        # dual sweep reconstructs BOTH inflation sets exactly
        blk, pad = set(), set()
        for q, r, L, s, in_blocked in hg.rasterize_volume_dual(v, CFG, R, infl_b, infl_p):
            pad.add((q, r, L, s))
            if in_blocked:
                blk.add((q, r, L, s))
        assert blk == _scalar_rasterize(v, CFG, R, infl_b)
        assert pad == _scalar_rasterize(v, CFG, R, infl_p)


def test_rasterize_box_lands_on_its_level_only():
    """A level corridor box marks cells at exactly its own flight level."""
    z = CFG.level_z(1)                                      # 70 m
    box = corridor_segment_volume(vec(0, 0, z), 0.0, vec(120, 0, z), CFG.dt_s, CFG)
    cells = set(hg.rasterize_volume(box, CFG, R))
    assert cells
    assert {L for (_, _, L, _) in cells} == {1}


def test_climb_box_spans_two_levels():
    """A slanted climb box from level 0 to level 1 marks cells at both levels."""
    box = corridor_segment_volume(
        vec(0, 0, CFG.level_z(0)), 0.0, vec(120, 0, CFG.level_z(1)), CFG.dt_s, CFG
    )
    levels = {L for (_, _, L, _) in hg.rasterize_volume(box, CFG, R)}
    assert levels == {0, 1}


def test_terminal_column_spans_all_inband_levels():
    """A [ground, ceiling] hover/terminal column registers at every in-band flight level."""
    from freespace_sim.geometry import CylinderSpec
    from freespace_sim.volumes import Volume4D

    col = Volume4D(CylinderSpec(0.0, 0.0, 60.0, CFG.ground_level_m, CFG.airspace_ceiling_m), 0.0, 60.0)
    levels = {L for (_, _, L, _) in hg.rasterize_volume(col, CFG, R)}
    assert levels == {0, 1, 2}


def test_single_level_rasterize_tags_zero():
    """With one flight level the (q,r,s) projection matches a single-plane raster, all at L==0."""
    cfg1 = SimConfig(flight_levels_m=(75.0,))               # one level, ceiling stays 125
    box = corridor_segment_volume(vec(0, 0, 75.0), 0.0, vec(120, 0, 75.0), cfg1.dt_s, cfg1)
    cells = set(hg.rasterize_volume(box, cfg1, R))
    assert cells
    assert {L for (_, _, L, _) in cells} == {0}
    assert {(q, r, s) for (q, r, L, s) in cells} == {
        (q, r, s) for (q, r, L, s) in _scalar_rasterize(box, cfg1, R, cfg1.corridor_width_m / 2.0 + R)
    }


def test_vertical_climb_box_overlaps_only_its_traversed_levels():
    # A 30→70 climb box must map to levels {0,1} only — never level 2. (The ±corridor_width/2 z-inflation
    # reached z=100 and _levels_overlapped wrongly returned [0, 1, 2].)
    box = corridor_segment_volume(vec(500, 0, CFG.level_z(0)), 0.0,
                                  vec(500, 0, CFG.level_z(1)), 2 * CFG.dt_s, CFG)
    assert hg._levels_overlapped(box, CFG) == [0, 1]
