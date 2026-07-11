"""Fixed terminal exit lanes — the lane-creation foundation (issue #18).

These pin the geometry the A* wiring (``fixed_exit_lanes``) is built on: how many boundary-hex lanes a
column of a given radius produces, that each is a valid just-outside-the-column cell, and that creation
is smooth, deterministic, and memoised. Same-hub deconfliction itself is exact CELL occupancy at plan
time (``HexOccupancyService.is_blocked``, exercised in ``test_terminal``/``test_hub_conflict_filed``),
not a per-lane graze graph — so a lane is just its cell + bearing/dist descriptors. No A* here: the
geometry is proven correct before any planner reads it.
"""

import math

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.planner import hexgrid as hg
from freespace_sim.types import Terminal
from freespace_sim.volumes import exit_radius

CFG = SimConfig()
R = hg.circumradius(CFG)


def _term(radius):
    return Terminal("H", 1, radius)


def _covered(cell, hub, term):
    cx, cy = hub
    hx, hy = hg.hex_center(*cell, R)
    return math.hypot(hx - cx, hy - cy) < exit_radius(term, CFG) - 1e-9


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
    lanes = hg.terminal_lanes(hub, term, CFG)
    er = exit_radius(term, CFG)
    cells = [L.cell for L in lanes]
    assert len(cells) == len(set(cells))                      # no duplicate cells
    assert len({round(L.bearing, 6) for L in lanes}) == len(lanes)   # distinct bearings
    for L in lanes:
        assert L.dist >= er - 1e-6                            # outside the column
        assert not _covered(L.cell, hub, term)               # boundary, not covered
        assert any(_covered(n, hub, term) for n in hg.hex_neighbors(*L.cell))  # touches covered


def test_lanes_sorted_by_bearing():
    """The ring is returned in a stable bearing order (the lane list's deterministic shape)."""
    lanes = hg.terminal_lanes((48.0, 16.0), _term(120), CFG)
    assert [L.bearing for L in lanes] == sorted(L.bearing for L in lanes)


# ----- always-active terminal airspace (terminal_airspace_always_active) -----

def test_terminal_cells_is_column_plus_lanes():
    """terminal_cells = covered column ∪ boundary lanes — a strict superset of the lane ring."""
    center = tuple(hg.hex_center(0, 0, R))
    lanes = {L.cell for L in hg.terminal_lanes(center, _term(120), CFG)}
    cells = hg.terminal_cells(center, _term(120), CFG)
    assert lanes <= cells and len(cells) > len(lanes)   # lanes plus the column interior


def test_static_terminal_walls_foreign_keeps_own_passable():
    """The occupancy's ledger subscribe_static hook (``_on_static``) makes a hub's cells a permanent FOREIGN
    wall (any step), while the hub's own flights pass through (transparent, absent a committed sibling
    corridor). (``_on_static`` is the body the ``ReservationLedger.subscribe_static`` replay drives.)"""
    from freespace_sim.planner.occupancy import HexOccupancyService

    svc = HexOccupancyService(CFG)
    center, term = tuple(hg.hex_center(0, 0, R)), _term(120)
    svc._on_static(center, term)
    q, r = next(iter(hg.terminal_cells(center, term, CFG)))
    top = CFG.n_levels - 1                                  # the always-active column is the [ground,
    #                                                        ceiling] tube ⇒ it walls EVERY flight level
    assert svc.is_blocked(q, r, 0, 5, own=())              # foreign flight → wall (level 0)
    assert svc.is_blocked(q, r, top, 9999, own=("OTHER",)) # foreign at any step AND any level
    assert not svc.is_blocked(q, r, 0, 5, own=(term.id,))  # own hub → transparent (no committed corridor)


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
