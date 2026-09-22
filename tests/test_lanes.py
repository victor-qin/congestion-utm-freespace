"""Fixed terminal exit lanes — the lane-creation foundation (issue #18).

These pin the geometry the A* wiring (``fixed_exit_lanes``) is built on: how many boundary-hex lanes a
column of a given radius produces, that each is a valid just-outside-the-column cell, and that creation
is smooth, deterministic, and memoised. Same-hub deconfliction itself is exact CELL occupancy at plan
time (``HexOccupancyService.is_blocked``, exercised in ``test_terminal``/``test_hub_conflict_filed``),
not a per-lane graze graph — so a lane is just its cell + bearing/dist descriptors. No A* here: the
geometry is proven correct before any planner reads it.
"""

import math
from dataclasses import replace
from itertools import combinations

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.conflict import volumes_conflict
from freespace_sim.planner import hexgrid as hg
from freespace_sim.types import Terminal
from freespace_sim.volumes import exit_radius, lane_link_volume, terminal_radius

CFG = SimConfig()
R = hg.circumradius(CFG)


def _term(radius):
    return Terminal("H", 1, radius)


def _covered(cell, hub, term):
    cx, cy = hub
    hx, hy = hg.hex_center(*cell, R)
    return math.hypot(hx - cx, hy - cy) < exit_radius(term, CFG) - 1e-9


def _link(hub, term, cell):
    """The outbound link box of one ring cell, on a common [0, 1) window so only geometry decides."""
    box = lane_link_volume(hub, terminal_radius(term, CFG), (*hg.hex_center(*cell, R), CFG.level_z(0)), 1.0,
                           CFG, "H", outbound=True)
    return replace(box, t_start=0.0, t_end=1.0)


# ----- lane counts & validity -----

@pytest.mark.parametrize("radius,n", [(120, 6), (200, 12), (300, 18)])
def test_lane_count_centered(radius, n):
    """A hub on a hex centre yields a fixed number of boundary-hex lanes per radius."""
    lanes = hg.terminal_lanes(tuple(hg.hex_center(0, 0, R)), _term(radius), CFG)
    assert len(lanes) == n


@pytest.mark.parametrize("radius", [120, 200, 300])
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


@pytest.mark.parametrize("radius,hub,dropped", [
    (90, (48.0, 16.0), 2),        # a chain of 4 conflicting lanes (5-6-7-8) keeps 2 of them
    (90, (35.0, 20.0), 2),        # chain of 5 (3..7) keeps 3
    (90, (60.0, -20.0), 1),       # one conflicting pair
    (75, (-57.0, 32.9), 5),       # every adjacent pair conflicts: a 9-cycle keeps 4
])
def test_narrow_column_keeps_the_largest_disjoint_lane_subset(radius, hub, dropped):
    """w = 60 m needs r >= (sqrt3/2)·120 m = 103.9 m (SimConfig.max_corridor_width_m) for every ring cell
    to be a lane. A narrower column is not refused: the ring is pruned, once and permanently, to a
    maximum subset whose link boxes are pairwise disjoint (exact FCL) — no larger disjoint subset exists
    (brute-force oracle) — while the wall still covers the whole ring."""
    term = _term(radius)
    covered, ring = hg._covered_boundary(hub, term, CFG)
    lanes = hg.terminal_lanes(hub, term, CFG)
    kept = [L.cell for L in lanes]
    assert set(kept) <= ring and len(kept) == len(ring) - dropped
    link = {c: _link(hub, term, c) for c in ring}
    clash = {frozenset(p) for p in combinations(ring, 2) if volumes_conflict(link[p[0]], link[p[1]])}
    assert all(frozenset(p) not in clash for p in combinations(kept, 2))            # kept links disjoint
    assert not any(all(frozenset(p) not in clash for p in combinations(S, 2))       # and no larger set is
               for S in combinations(sorted(ring), len(kept) + 1))
    assert [L.bearing for L in lanes] == sorted(L.bearing for L in lanes)          # still in bearing order
    assert hg.terminal_cells(hub, term, CFG) == covered | ring                      # wall untouched


@pytest.mark.parametrize("radius,hub", [(90, (48.0, 16.0)), (75, (-57.0, 32.9))])
def test_pruning_prefers_the_widest_spread(radius, hub):
    """Among equally large disjoint subsets the kept lanes are spread as evenly as the ring allows:
    the smallest bearing gap between consecutive kept lanes is the widest any maximum subset achieves
    (the deterministic tie-break, so a hub's lane set is the same in every run)."""
    term = _term(radius)
    ring = hg._covered_boundary(hub, term, CFG)[1]
    link = {c: _link(hub, term, c) for c in ring}
    lanes = hg.terminal_lanes(hub, term, CFG)

    def smallest_gap(cells):
        b = sorted(hg._bearing_deg(c, *hub, R) for c in cells)
        return min([b[i + 1] - b[i] for i in range(len(b) - 1)] + [b[0] + 360.0 - b[-1]])

    rivals = [S for S in combinations(sorted(ring), len(lanes))
              if all(not volumes_conflict(link[a], link[b]) for a, b in combinations(S, 2))]
    assert len(rivals) > 1                                                          # a real tie
    assert smallest_gap([L.cell for L in lanes]) == pytest.approx(max(smallest_gap(S) for S in rivals))


def test_lane_steps_do_not_round_up_at_exact_pitch_multiples():
    """A hub on a hex centre puts its corner lanes at exactly two pitches; float noise in the hypot
    (240.0000000000008 m) must not charge them a third step."""
    lanes = hg.terminal_lanes(tuple(hg.hex_center(0, 0, R)), _term(200), CFG)
    assert {L.steps for L in lanes} == {2}                    # 208 m and 240 m lanes: 2 steps each


@pytest.mark.parametrize("radius", [104, 120, 180])
@pytest.mark.parametrize("hub", [(0.0, 0.0), (48.0, 16.0), (30.0, -55.0), (-57.0, 32.9)])
def test_link_boxes_of_one_hub_never_conflict(radius, hub):
    """Simultaneous link boxes on every lane of a hub are pairwise disjoint (exact FCL) and, at or above
    the (sqrt3/2)·p = 103.9 m the rule needs for w = 60 m, no ring cell is dropped. They are strict
    boxes, so an overlap would serialise two lanes. (-57, 32.9) is the tightest 180 m placement:
    adjacent links pass 1.7 m apart."""
    term = _term(radius)
    assert CFG.corridor_width_m <= CFG.max_corridor_width_m(radius)
    lanes = hg.terminal_lanes(hub, term, CFG)
    assert len(lanes) == len(hg._covered_boundary(hub, term, CFG)[1])
    links = [_link(hub, term, L.cell) for L in lanes]
    for i, a in enumerate(links):
        for b in links[i + 1:]:
            assert not volumes_conflict(a, b)
    # the bound peaks at one pitch and tends to p/2 = 60 m from above, so w = 60 stays admissible
    assert CFG.max_corridor_width_m(120.0) > CFG.max_corridor_width_m(180.0) > CFG.corridor_width_m


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
    from freespace_sim.planner.astar.occupancy import HexOccupancyService

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
    hub, term = (48.0, 16.0), _term(120)
    a = hg.terminal_lanes(hub, term, CFG)
    b = hg.terminal_lanes(hub, term, CFG)
    assert a is b                                             # same object from the cache
    assert hg.terminal_lanes((300.0, 200.0), term, CFG) is not a   # distinct hub → distinct set


def test_two_nearby_hubs_get_distinct_lanes():
    """Guards against a memo-key collision: two hubs a few metres apart must not alias lane sets."""
    a = hg.terminal_lanes((0.0, 0.0), _term(120), CFG)
    b = hg.terminal_lanes((40.0, 0.0), _term(120), CFG)
    assert [L.cell for L in a] != [L.cell for L in b]
