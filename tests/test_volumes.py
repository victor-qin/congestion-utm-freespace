import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.conflict import volumes_conflict
from freespace_sim.types import vec
from freespace_sim.volumes import (
    Volume4D, build_corridor, build_reservation_from_corners, corridor_segment_volume, hover_reservation,
    segment_overlaps_column,
)

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
    assert np.allclose(hi, [100 + CFG.effective_hover_radius_m, 200 + CFG.effective_hover_radius_m,
                            CFG.airspace_ceiling_m])   # column now spans the regulated tube [0, ceiling]
    assert np.isclose(h.t_start, 10.0)
    assert np.isclose(h.t_end, 10.0 + CFG.hover_time_s + CFG.climb_time_s)


def test_hover_duration_uses_per_level_climb_time():
    # the dwell window covers hover + the ACTUAL climb to the given level, not the preferred plane
    ct = CFG.climb_time_to(CFG.level_z(0))                       # 30 m / 6 = 5 s
    h = hover_reservation(vec(0, 0, 0), t0=0.0, cfg=CFG, climb_time_s=ct)
    assert np.isclose(h.t_end, CFG.hover_time_s + ct)


def test_build_from_corners_times_cruise_start_per_level():
    # cruise starts after the climb to the FIRST corner's altitude (its flight level), not the preferred
    z = CFG.level_z(1)                                          # 70 m
    corners = [vec(0, 0, z), vec(120, 0, z), vec(240, 0, z)]
    _, centerline, _, _ = build_reservation_from_corners(
        corners, vec(0, 0, 0), vec(240, 0, 0), 0.0, 0.0, CFG
    )
    assert np.isclose(centerline[0][1], CFG.climb_time_to(z))   # 70/6, not the preferred 75/6


# --- vertical layer-change (climb) box geometry -----------------------------------------------------
# A mid-route climb segment is fixed in (x, y) and moves in z. `corridor_segment_volume` extends every
# segment by half the corridor WIDTH at each end for horizontal contiguity; for a vertical segment that
# extension must NOT land in z (it would balloon the box past the levels the drone actually traverses).

def _climb_box(z_from, z_to, x=500.0):
    """A pure-vertical layer-change box: fixed (x, 0), z climbs z_from → z_to over two timesteps."""
    return corridor_segment_volume(vec(x, 0, z_from), 0.0, vec(x, 0, z_to), 2 * CFG.dt_s, CFG)


def test_vertical_climb_box_spans_only_the_drone_vertical_footprint():
    # A 30→70 climb reserves the climb range plus the drone's ±corridor_height/2 footprint → z-AABB
    # [15, 85]. NOT ±corridor_width/2 (which would give [0, 100]).
    lo, hi = _climb_box(CFG.level_z(0), CFG.level_z(1)).shape.aabb()
    half = CFG.corridor_height_m / 2.0
    assert np.isclose(lo[2], CFG.level_z(0) - half)                 # 15, not 0
    assert np.isclose(hi[2], CFG.level_z(1) + half)                 # 85, not 100


def test_vertical_climb_box_stays_within_the_regulated_ceiling():
    # The top rung (70→110) must not poke above airspace_ceiling_m; the ±width/2 bug reached z = 140.
    _, hi = _climb_box(CFG.level_z(1), CFG.level_z(2)).shape.aabb()
    assert hi[2] <= CFG.airspace_ceiling_m + 1e-9                   # 125, not 140


def test_horizontal_corridor_box_geometry_is_unchanged():
    # Anisotropic extension must leave HORIZONTAL boxes byte-identical: ext stays corridor_width/2 in x,
    # z is the corridor_height slab centred on the level.
    z, seg = CFG.level_z(0), CFG.corridor_segment_len_m
    lo, hi = corridor_segment_volume(vec(0, 0, z), 0.0, vec(seg, 0, z), CFG.dt_s, CFG).shape.aabb()
    half_h, ext = CFG.corridor_height_m / 2.0, CFG.corridor_width_m / 2.0
    assert np.isclose(lo[2], z - half_h) and np.isclose(hi[2], z + half_h)   # z slab unchanged
    assert np.isclose(lo[0], -ext) and np.isclose(hi[0], seg + ext)          # x extension unchanged


def test_climb_box_stays_contiguous_with_the_cruise_before_it():
    # The smaller vertical extension must not open a contiguity gap: a level-0 cruise box arriving at
    # (500,0,30) and the climb box leaving it (30→70) must still overlap in space+time (ASTM §4.3.5).
    seg = CFG.corridor_segment_len_m
    cruise = corridor_segment_volume(vec(500 - seg, 0, CFG.level_z(0)), 0.0,
                                     vec(500, 0, CFG.level_z(0)), CFG.dt_s, CFG)
    climb = corridor_segment_volume(vec(500, 0, CFG.level_z(0)), CFG.dt_s,
                                    vec(500, 0, CFG.level_z(1)), 3 * CFG.dt_s, CFG)
    assert volumes_conflict(cruise, climb)


def test_climb_box_does_not_conflict_with_the_level_above():
    # THE headline correctness case: a 30→70 climb box must not FCL-touch a level-2 (110 m) cruise box
    # over the SAME column and time — the climber tops out at 70 m, 40 m below level 2. (The z-inflation
    # made the climb box span z 0..100, so it spuriously conflicted with level-2 traffic.)
    climb = _climb_box(CFG.level_z(0), CFG.level_z(1))                        # (500,0), 30→70
    cruise = corridor_segment_volume(vec(440, 0, CFG.level_z(2)), 0.0,
                                     vec(560, 0, CFG.level_z(2)), 2 * CFG.dt_s, CFG)  # level 2 over (500,0)
    assert not volumes_conflict(climb, cruise)


# ---------------- byte-identity of the scalarized hub-tagging test (issue #30 lever #8) ----------------
# segment_overlaps_column was rewritten from numpy (np.linalg.norm / .dot / np.clip) to plain scalars to
# shed numpy's per-call ufunc dispatch. Its boolean output must be identical to the original numpy code; the
# oracle below is a VERBATIM frozen copy of the pre-change body — DO NOT EDIT (byte-identity reference only).


def _segment_overlaps_column_numpy_original(a, b, center, radius, cfg):
    a2 = np.asarray(a, float)[:2]
    b2 = np.asarray(b, float)[:2]
    c2 = np.asarray(center, float)[:2]
    seg = b2 - a2
    L = float(np.linalg.norm(seg))
    u = seg / L if L > 1e-9 else np.array([1.0, 0.0])
    ext = cfg.corridor_width_m / 2.0
    p0, p1 = a2 - u * ext, b2 + u * ext
    ab = p1 - p0
    t = float(np.clip((c2 - p0).dot(ab) / max(ab.dot(ab), 1e-12), 0.0, 1.0))
    d = float(np.linalg.norm(c2 - (p0 + t * ab)))
    return d < radius + cfg.corridor_width_m / 2.0


def test_segment_overlaps_column_byte_identical_to_original_numpy():
    """The scalarized segment_overlaps_column returns the SAME boolean as the frozen numpy original — over
    random, degenerate (zero-length), near-threshold, and real corridor geometry. The underlying distance is
    bit-exact (verified separately), so the `<` decision cannot flip."""
    import random
    rng = random.Random(0)

    def check(a, b, center, radius):
        got = segment_overlaps_column(a, b, center, radius, CFG)
        exp = _segment_overlaps_column_numpy_original(a, b, center, radius, CFG)
        assert got == exp, f"overlap {got} != {exp} for a={a} b={b} c={center} r={radius}"

    for _ in range(5000):                                  # random, varied magnitude (incl. negative coords)
        scale = 10 ** rng.uniform(-1, 4)
        a = [rng.uniform(-1, 1) * scale for _ in range(3)]
        b = [rng.uniform(-1, 1) * scale for _ in range(3)]
        center = [rng.uniform(-1, 1) * scale for _ in range(2)]
        check(a, b, center, rng.uniform(0, 200))

    check([5, 5, 0], [5, 5, 0], [5, 10, 0], 3.0)           # zero-length segment → the length<1e-9 branch
    thresh = 90.0 + CFG.corridor_width_m / 2.0              # exact overlap boundary for a level segment
    for off in (0.0, -1e-6, 1e-6, -5.0, 5.0):              # straddle the `d < radius + width/2` threshold
        check([0, 0, 0], [100, 0, 0], [50, thresh + off, 0], 90.0)

    # real geometry: the hub-tagging calls build_reservation_from_corners actually makes near a hub column
    corners = [vec(0, 0, 30), vec(1200, 300, 70), vec(2400, -200, 110)]
    for a, b in zip(corners, corners[1:]):
        for r in (90.0, 180.0):
            check(a, b, np.asarray(corners[0], float)[:2], r)
            check(a, b, np.asarray(corners[-1], float)[:2], r)
