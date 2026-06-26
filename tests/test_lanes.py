"""Fixed terminal exit lanes — the lane-creation + conflict-graph schedule foundation (issue #18).

These pin the geometry that the A* wiring (``fixed_exit_lanes``) is built on: how many boundary-hex
lanes a column of a given radius produces, that each is a valid just-outside-the-column cell, the graze
structure, and the per-cell ``cell_admits`` reservation. No A* / full sim here — the foundation is
proven correct before any planner reads it.
"""

import math

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.terminal_capacity import TerminalCapacity
from freespace_sim.types import Terminal
from freespace_sim.volumes import exit_radius, graze_angle

CFG = SimConfig()
R = hg.circumradius(CFG)


def _term(radius):
    return Terminal("H", 1, radius)


def _covered(cell, hub, term):
    cx, cy = hub
    hx, hy = hg.hex_center(*cell, R)
    return math.hypot(hx - cx, hy - cy) < exit_radius(term, CFG) - 1e-9


def _ang_gap(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _min_bearing_gap(lanes):
    bs = sorted(L.bearing % 360.0 for L in lanes)
    n = len(bs)
    return min((bs[(i + 1) % n] - bs[i]) % 360.0 for i in range(n))


# ----- lane counts & validity -----

@pytest.mark.parametrize("radius,n", [(90, 6), (120, 12), (200, 18), (300, 24)])
def test_lane_count_centered(radius, n):
    """A hub on a hex centre yields a fixed number of boundary-hex lanes per radius."""
    lanes = hg.terminal_lanes(tuple(hg.hex_center(0, 0, R)), _term(radius), CFG)
    assert len(lanes) == n


@pytest.mark.parametrize("radius", [90, 120, 200, 300])
@pytest.mark.parametrize("hub", [(0.0, 0.0), (48.0, 16.0), (30.0, -55.0)])
def test_lanes_are_valid_boundary_cells(radius, hub):
    """Every lane cell is outside the column, not covered, and hex-adjacent to a covered cell;
    cells and bearings are distinct (smooth, correct creation)."""
    term = _term(radius)
    # snap the requested hub into a real hex so (0,0) etc. are exact centres for the offset==0 case
    lanes = hg.terminal_lanes(hub, term, CFG)
    er = exit_radius(term, CFG)
    cells = [L.cell for L in lanes]
    assert len(cells) == len(set(cells))                      # no duplicate cells
    assert len({round(L.bearing, 6) for L in lanes}) == len(lanes)   # distinct bearings
    for L in lanes:
        assert L.dist >= er - 1e-6                            # outside the column
        assert not _covered(L.cell, hub, term)               # boundary, not covered
        assert any(_covered(n, hub, term) for n in hg.hex_neighbors(*L.cell))  # touches covered


@pytest.mark.parametrize("radius", [90, 120, 200, 300])
def test_centered_hub_lanes_never_graze(radius):
    """On a hex centre the boundary ring is uniform, so no two lanes graze (graze sets empty)."""
    lanes = hg.terminal_lanes(tuple(hg.hex_center(0, 0, R)), _term(radius), CFG)
    assert all(L.graze == () for L in lanes)
    assert _min_bearing_gap(lanes) >= graze_angle(_term(radius), CFG)


# ----- graze structure -----

def test_graze_angle_matches_formula_and_shrinks():
    """θ_min = 2·asin(corridor_width / 2r), decreasing in r (bigger columns graze less)."""
    vals = []
    for radius in (90, 120, 200, 300):
        expected = 2.0 * math.degrees(math.asin(CFG.corridor_width_m / (2.0 * radius)))
        got = graze_angle(_term(radius), CFG)
        assert got == pytest.approx(expected, abs=1e-9)
        vals.append(got)
    assert vals == sorted(vals, reverse=True)                 # strictly shrinking


@pytest.mark.parametrize("radius", [90, 200])
def test_no_graze_free_radius_for_arbitrary_hub(radius):
    """No finite radius removes worst-case grazing: sweeping the hub across a hex always finds an
    offset whose closest two lanes are nearer than θ_min (issue #18 analysis, locked as a test)."""
    term = _term(radius)
    th = graze_angle(term, CFG)
    worst = 999.0
    any_graze = False
    for i in range(9):
        for j in range(9):
            hub = (i / 8 * R * math.sqrt(3) * 0.5, (j / 8 - 0.5) * R)
            if math.hypot(*hub) > R:
                continue
            lanes = hg.terminal_lanes(hub, term, CFG)
            worst = min(worst, _min_bearing_gap(lanes))
            any_graze = any_graze or any(L.graze for L in lanes)
    assert worst < th                                         # a grazing offset exists
    assert any_graze                                          # ... and it shows up as a graze set


# ----- smoothness of creation -----

def test_offset_sweep_count_is_smooth():
    """Sliding a hub across one hex keeps the lane count bounded and well-formed (no blow-up / empties /
    cell that is both covered and boundary)."""
    term = _term(200)
    counts = set()
    for i in range(13):
        hub = (i / 12 * R * math.sqrt(3), 0.0)
        lanes = hg.terminal_lanes(hub, term, CFG)
        assert 12 <= len(lanes) <= 22                         # bounded, never empty
        assert all(not _covered(L.cell, hub, term) for L in lanes)
        counts.add(len(lanes))
    assert max(counts) - min(counts) <= 3                     # smooth, no degenerate jumps


# ----- determinism / memoisation -----

def test_deterministic_and_memoised():
    hub, term = (48.0, 16.0), _term(90)
    a = hg.terminal_lanes(hub, term, CFG)
    b = hg.terminal_lanes(hub, term, CFG)
    assert a is b                                             # same object from the cache
    assert hg.terminal_lanes((300.0, 200.0), term, CFG) is not a   # distinct hub → distinct set


def test_two_nearby_hubs_get_distinct_lanes():
    """Guards against a memo-key collision: two hubs a few metres apart must not alias lane sets."""
    a = hg.terminal_lanes((0.0, 0.0), _term(90), CFG)
    b = hg.terminal_lanes((40.0, 0.0), _term(90), CFG)
    assert [L.cell for L in a] != [L.cell for L in b]


# ----- conflict-graph schedule (cell_admits) -----

def test_cell_admits_capacity_one_and_neighbour_lock():
    tc = TerminalCapacity(CFG, ReservationLedger(CFG))
    tc.cell_dwells[("H", (1, 0))] = [(0.0, 50.0)]            # lane (1,0) busy on [0,50)
    assert not tc.cell_admits("H", (1, 0), (), 0.0, 50.0)    # same cell, overlap → locked
    assert tc.cell_admits("H", (1, 0), (), 50.0, 100.0)      # disjoint window → free
    assert not tc.cell_admits("H", (0, 1), ((1, 0),), 0.0, 50.0)  # graze-neighbour of busy cell → locked
    assert tc.cell_admits("H", (0, 1), (), 0.0, 50.0)        # no graze list → independent
    assert tc.cell_admits("H2", (1, 0), (), 0.0, 50.0)       # different hub → independent


def test_on_commit_records_lane_cell_and_evict_clears():
    from freespace_sim.geometry import BoxSpec
    from freespace_sim.volumes import Volume4D
    tc = TerminalCapacity(CFG, ReservationLedger(CFG))
    box = Volume4D(BoxSpec((0, 0, 150), tuple(np.eye(3).flatten()), (60, 60, 30)),
                   t_start=10.0, t_end=40.0, terminal_id="H", lane_cell=(1, 0))
    tc.on_commit(7, [box])
    assert tc.cell_dwells[("H", (1, 0))] == [(10.0, 40.0)]
    tc.evict_before(40.0)                                     # interval ends at 40 → dropped
    assert ("H", (1, 0)) not in tc.cell_dwells
