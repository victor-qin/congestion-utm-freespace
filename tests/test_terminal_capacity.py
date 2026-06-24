"""TerminalCapacity — the temporal pad-capacity + column-activation authority.

Migrates the *intent* of the term_cells unit tests (test_occupancy.py Phase B) onto the new
hex-free interval authority: capacity by interval-overlap count, column activation by lazy
union-coverage + ledger fallback, lockstep eviction.
"""

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner.terminal_capacity import TerminalCapacity
from freespace_sim.types import Terminal, vec
from freespace_sim.volumes import Volume4D, hover_reservation

CFG = SimConfig()
DWELL = CFG.hover_time_s + CFG.climb_time_s   # a column cylinder's committed lifetime (55 s default)


def _col(center_xy=(1000.0, 1000.0), t0=0.0, tid="H", radius=90.0):
    """A committed terminal-column cylinder at ``center_xy`` opening at ``t0`` → window [t0, t0+DWELL)."""
    return hover_reservation((center_xy[0], center_xy[1], 0.0), t0, CFG, terminal_id=tid, radius=radius)


def _foreign_through_hub():
    """A foreign (untagged) cruise corridor at y=1000 from x=500→1500 — transits the hub at (1000,1000)."""
    return Volume4D(box_from_segment(vec(500, 1000, 150), vec(1500, 1000, 150), 40, 400), 0.0, 1e6)


# --- capacity (step 2) ------------------------------------------------------------------------

def test_admits_counts_overlapping_same_hub_dwells():
    tcap = TerminalCapacity(CFG, ReservationLedger(CFG))
    tcap.on_commit(1, [_col(t0=0.0)])
    tcap.on_commit(2, [_col(t0=0.0)])
    assert not tcap.admits("H", 0.0, DWELL, capacity=2)   # 2 dwells overlap, no room
    assert tcap.admits("H", 0.0, DWELL, capacity=3)        # room for a third
    assert tcap.admits("H", 1000.0, 1000.0 + DWELL, 1)     # disjoint window → 0 overlap


def test_capacity_one_is_exclusive():
    tcap = TerminalCapacity(CFG, ReservationLedger(CFG))
    tcap.on_commit(1, [_col(t0=0.0)])
    assert not tcap.admits("H", 0.0, DWELL, capacity=1)    # capacity 1 ⟺ the old single pad


def test_on_commit_records_both_cylinders_of_a_roundtrip():
    # a flight tagging BOTH origin and dest at hub H contributes two dwells (mirror add_volume)
    tcap = TerminalCapacity(CFG, ReservationLedger(CFG))
    tcap.on_commit(1, [_col(t0=0.0), _col(t0=200.0)])
    assert len(tcap.dwells["H"]) == 2


def test_radius_must_be_constant_per_hub():
    tcap = TerminalCapacity(CFG, ReservationLedger(CFG))
    tcap.on_commit(1, [_col(radius=90.0)])
    with pytest.raises(ValueError, match="radius must be constant"):
        tcap.on_commit(2, [_col(radius=150.0)])


# --- column activation (step 1): always query the ledger (no unsound skip) --------------------

def test_column_clear_detects_foreign_transit():
    led = ReservationLedger(CFG)
    led.commit(99, [_foreign_through_hub()])
    tcap = TerminalCapacity(CFG, led)
    term, center = Terminal("H", 4, radius=90.0), vec(1000, 1000, 0)
    assert not tcap.column_clear(term, center, 0.0)        # the foreign corridor intrudes → not clear


def test_column_clear_always_queries_even_when_siblings_cover():
    # NO 'already-deployed → skip the ledger' shortcut: it is unsound — a sibling's own near-hub cruise
    # corridor can intrude in a window its column 'covers', so column_clear always consults the ledger.
    led = ReservationLedger(CFG)
    led.commit(99, [_foreign_through_hub()])
    tcap = TerminalCapacity(CFG, led)
    term, center = Terminal("H", 4, radius=90.0), vec(1000, 1000, 0)
    tcap.dwells["H"] = [(0.0, DWELL)]                       # a sibling 'covers' the window...
    assert not tcap.column_clear(term, center, 0.0)         # ...but the ledger still gates the foreign


def test_column_clear_is_clear_in_empty_airspace():
    tcap = TerminalCapacity(CFG, ReservationLedger(CFG))
    term, center = Terminal("H", 4, radius=90.0), vec(1000, 1000, 0)
    assert tcap.column_clear(term, center, 0.0)


# --- the takeoff/landing edge predicate -------------------------------------------------------

def test_dwell_ok_requires_capacity_and_clear():
    tcap = TerminalCapacity(CFG, ReservationLedger(CFG))
    term, center = Terminal("H", 2, radius=90.0), vec(1000, 1000, 0)
    assert tcap.dwell_ok(term, center, 0.0, capacity=2)    # empty: capacity + clear
    tcap.dwells["H"] = [(0.0, DWELL), (0.0, DWELL)]         # two overlapping dwells fill capacity 2
    assert not tcap.dwell_ok(term, center, 0.0, capacity=2)


# --- eviction (lockstep) ----------------------------------------------------------------------

def test_evict_drops_past_dwells_and_is_monotonic():
    tcap = TerminalCapacity(CFG, ReservationLedger(CFG))
    tcap.on_commit(1, [_col(t0=0.0)])                      # [0, DWELL)
    tcap.on_commit(2, [_col(t0=200.0)])                    # [200, 200+DWELL)
    tcap.evict_before(DWELL + 1.0)                         # drops the first (ended), keeps the second
    assert tcap.dwells["H"] == [(200.0, 200.0 + DWELL)]
    tcap.evict_before(10.0)                                # earlier watermark → no-op (monotonic)
    assert tcap.dwells["H"] == [(200.0, 200.0 + DWELL)]
    tcap.evict_before(10_000.0)                            # everything past → hub dropped entirely
    assert "H" not in tcap.dwells


def test_reset_clears_everything():
    tcap = TerminalCapacity(CFG, ReservationLedger(CFG))
    tcap.on_commit(1, [_col(t0=0.0)])
    tcap.reset()
    assert tcap.dwells == {} and tcap.radius == {} and tcap.evicted_before is None


# --- the ledger publish hook (push) -----------------------------------------------------------

def test_subscribe_feeds_on_commit():
    led = ReservationLedger(CFG)
    tcap = TerminalCapacity(CFG, led)
    led.subscribe(tcap.on_commit)
    led.commit(1, [_col(t0=0.0)])                          # fires the publish hook → tcap.on_commit
    assert tcap.dwells.get("H") == [(0.0, DWELL)]
