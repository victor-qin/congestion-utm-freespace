import math

import numpy as np
import pytest

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


@pytest.fixture
def sweep_mode(request, monkeypatch):
    """Run a test body once per rasteriser backend. ``True`` is a no-op when numba is absent, so a
    numba-less environment still exercises the reference rather than silently skipping."""
    monkeypatch.setattr(hg, "USE_COMPILED", request.param and hg._COMPILED)
    hg._RANGE_CACHE.clear()                       # the memo is keyed on the volume, not the backend
    yield
    hg._RANGE_CACHE.clear()


@pytest.mark.parametrize("sweep_mode", [False, True], indirect=True)
def test_vectorized_rasterize_matches_scalar_reference(sweep_mode):
    """The vectorized rasterizer (single + dual) must be byte-identical to the scalar oracle, across
    several box orientations and a hover cylinder — for BOTH backends. The oracle
    (``_hexes_in_box`` + ``_footprint_contains``) is independent of either sweep, so it is what
    stops the compiled path and the numpy path from being wrong together."""
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


def test_rasterize_ranges_expand_to_dual_and_reuse(monkeypatch):
    """``rasterize_ranges`` (issue #8 Phase E) collapses the step axis to a contiguous span per cell.
    Expanding every range back over its steps must reproduce :func:`rasterize_volume_dual` EXACTLY
    (byte-coverage: the compiled pool blocks the whole span in one split, and this is why that stays
    byte-identical), and the memo must reuse the geometry (the point: one sweep for both images)."""
    z = CFG.cruise_level_m
    vols = [
        corridor_segment_volume(vec(800, 200, z), 40.0, vec(920, 260, z), 44.0, CFG),   # multi-step box
        hover_reservation(vec(1500, -700, 0.0), 60.0, CFG),                             # cylinder
    ]
    infl_b = CFG.corridor_width_m / 2.0 + R
    infl_p = CFG.effective_hover_radius_m + R

    calls = {"n": 0}
    real = hg.rasterize_volume_ranges
    def _counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(hg, "rasterize_volume_ranges", _counting)
    hg._RANGE_CACHE.clear()

    for v in vols:
        want = list(hg.rasterize_volume_dual(v, CFG, R, infl_b, infl_p))
        before = calls["n"]
        ranges = hg.rasterize_ranges(v, CFG, R, infl_b, infl_p)   # cold: computes
        again = hg.rasterize_ranges(v, CFG, R, infl_b, infl_p)    # warm: reused, no recompute
        assert again is ranges                                    # the SECOND consumer reuses it
        assert calls["n"] == before + 1                           # exactly one underlying sweep
        expanded = [(q, r, L, s, b) for q, r, L, s_lo, s_hi, b in ranges
                    for s in range(s_lo, s_hi + 1)]
        assert expanded == want                                   # ranges ⇒ dual sweep, byte-for-byte
        assert len(ranges) < len(want)                            # the collapse actually happened
    hg._RANGE_CACHE.clear()


def test_block_range_matches_free_set_oracle():
    """``_Pool.block_range`` (issue #8 Phase E) vs an INDEPENDENT oracle — a plain per-cell set of
    still-free steps. (``block`` now delegates to ``block_range``, so comparing the two would be
    circular; the oracle validates the interval surgery from first principles instead.) Random spans,
    some straddling holes earlier spans punched (the multi-interval case) and some running off both
    ends, must leave ``blocked_at`` agreeing with the oracle at every step of every cell."""
    from freespace_sim.planner.astar.compiled_hex_occupancy import _Pool

    NC, MAXS = 4, 40
    rng = np.random.default_rng(0)
    pool = _Pool(NC, MAXS)
    free = [set(range(MAXS + 1)) for _ in range(NC)]      # oracle: the still-free steps per cell
    for _ in range(80):
        c = int(rng.integers(0, NC))
        lo = int(rng.integers(-3, MAXS + 2))              # spans may run off either end
        hi = lo + int(rng.integers(0, 14))
        pool.block_range(c, lo, hi)
        free[c] -= set(range(max(0, lo), min(MAXS, hi) + 1))   # clamp mirrors block_range
        for cc in range(NC):
            for s in range(0, MAXS + 1):
                assert pool.blocked_at(cc, s) == (s not in free[cc]), (cc, s, lo, hi)


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


# ------------------------------------------------------ compiled sweep (hexgrid_kernel)
@pytest.mark.slow
def test_compiled_rasteriser_rows_match_reference_on_a_real_cut():
    """Every committed volume of a real cut, compiled vs reference, compared as ORDERED rows.

    Ordering is the point. The compiled sweep is not bit-identical to numpy on the box path (numpy's
    ``(N,3) @ (3,3)`` does not sum in the order a register-scalar expression does; measured max
    delta 1.1e-13 m), so what is being asserted is that no cell's ``slack <= infl`` DECISION moved.
    And a transposed loop nest would keep every cell while changing the sequence, which silently
    reorders ``HexOccupancyService._rows``, ``CompiledHexOccupancy._claims`` and the interval pool's
    ``block_range`` applications — a set comparison would pass that.

    Synthetic volumes cannot stand in for this: only a real schedule produces corridor boxes at
    every bearing, terminal columns, and climb boxes spanning several levels.
    """
    if not hg._COMPILED:
        pytest.skip("numba unavailable — nothing to compare against")
    from freespace_sim import sim
    from freespace_sim.scenarios import get_scenario
    from freespace_sim.scenarios.spec import with_overrides

    spec = with_overrides(get_scenario("density_faa_wing_zipline"),
                          demand_duration_s=30.0, horizon_s=900.0)
    cfg = spec.config()
    res = sim.run(cfg, demand=spec.demand_model(), planner_name="astar", progress=False)
    r_circ = hg.circumradius(cfg)
    infl_b = cfg.corridor_width_m / 2.0 + r_circ
    infl_p = cfg.effective_hover_radius_m + r_circ
    vols = [v for _fid, v in res.ledger.iter_committed()]
    assert vols, "the cut committed nothing — the comparison would be vacuous"

    def sweep(vol, compiled):
        hg.USE_COMPILED = compiled
        hg._RANGE_CACHE.clear()
        return (list(hg.rasterize_volume_ranges(vol, cfg, r_circ, infl_b, infl_p)),
                list(hg.rasterize_volume_dual(vol, cfg, r_circ, infl_b, infl_p)),
                list(hg.rasterize_volume(vol, cfg, r_circ)))

    try:
        for vol in vols:
            assert sweep(vol, True) == sweep(vol, False)
    finally:
        hg.USE_COMPILED = False
        hg._RANGE_CACHE.clear()


def test_rasteriser_falls_back_when_kernel_is_absent(monkeypatch):
    """With ``USE_COMPILED`` on but no kernel, the reference must still run (and say so once), so a
    numba-less environment keeps working instead of raising a NameError deep in a commit hook."""
    monkeypatch.setattr(hg, "_COMPILED", False)
    monkeypatch.setattr(hg, "USE_COMPILED", True)
    monkeypatch.setattr(hg, "_compiled_warned", False)
    hg._RANGE_CACHE.clear()
    z = CFG.cruise_level_m
    vol = corridor_segment_volume(vec(800, 200, z), 40.0, vec(920, 260, z), 44.0, CFG)
    infl_b = CFG.corridor_width_m / 2.0 + R
    infl_p = CFG.effective_hover_radius_m + R

    got = list(hg.rasterize_volume_ranges(vol, CFG, R, infl_b, infl_p))
    monkeypatch.setattr(hg, "USE_COMPILED", False)
    hg._RANGE_CACHE.clear()
    assert got == list(hg.rasterize_volume_ranges(vol, CFG, R, infl_b, infl_p))
    assert hg._compiled_warned                       # warned, once
