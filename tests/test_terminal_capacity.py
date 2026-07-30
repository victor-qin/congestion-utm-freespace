"""TerminalCapacity — the temporal pad-capacity + column-activation authority.

Migrates the *intent* of the term_cells unit tests (test_occupancy.py Phase B) onto the new
hex-free interval authority: capacity by interval-overlap count, column activation by lazy
union-coverage + ledger fallback, lockstep eviction.
"""

from dataclasses import replace

import pytest

import freespace_sim.planner.terminal_capacity as tc
from analysis.ab_column_clear import _committed_digest, _intent_digest, _static_digest
from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.terminal_capacity import TerminalCapacity
from freespace_sim.sim import run
from freespace_sim.types import DenialReason, FlightRequest, Terminal, vec
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


def test_always_active_shortcut_requires_this_hubs_registered_wall():
    """The config flag states that walls are intended; only the ledger registration proves this hub has one.

    Lower-level planner callers can set always-active without going through sim.run's wall installation.
    They must retain the real foreign-transit scan instead of turning a ground delay into a commit rejection.
    Registering an unrelated hub is insufficient.
    """
    cfg = replace(CFG, terminal_airspace_always_active=True)
    led = ReservationLedger(cfg)
    led.commit(99, [_foreign_through_hub()])
    tcap = TerminalCapacity(cfg, led)
    term, center = Terminal("H", 4, radius=90.0), vec(1000, 1000, 0)

    assert not led.has_static_terminal(term.id)
    assert not tcap.column_clear(term, center, 0.0, z=70.0)

    other = Terminal("OTHER", 4, radius=90.0)
    led.register_static_terminal(vec(3000, 3000, 0), other)
    assert led.has_static_terminal(other.id)
    assert not led.has_static_terminal(term.id)
    assert not tcap.column_clear(term, center, 0.0, z=70.0)


def test_direct_planner_without_registered_wall_preserves_legacy_gate(monkeypatch):
    """Tripwire for the reproduced config-only regression in lower-level planner callers.

    With no registered wall, the real gate proves the origin column is unavailable and exhausts the
    search budget. The old shortcut changed that false gate to true, built a conflicting corridor, and
    changed the deterministic denial from ``budget_exceeded`` to ``conflict_filed``.
    """
    cfg = SimConfig(
        terminal_airspace_always_active=True,
        max_ground_delay_s=0.0,
        flight_levels_m=(100.0,),
        airspace_ceiling_m=125.0,
        region_size_m=(2500.0, 2500.0),
    )
    foreign = Volume4D(
        box_from_segment(vec(800, 1000, 100), vec(1200, 1000, 100), 40, 200),
        0.0,
        1e6,
    )
    request = FlightRequest(
        1,
        vec(1000, 1000, 0),
        vec(1400, 1000, 0),
        0.0,
        origin_terminal=Terminal("H", 4, radius=90.0),
    )

    def plan(skip):
        monkeypatch.setattr(tc, "SKIP_FOREIGN_WHEN_WALLED", skip)
        ledger = ReservationLedger(cfg)
        ledger.commit(99, [foreign])
        return AStarPlanner().plan(request, ledger, cfg)

    baseline = plan(False)
    patched = plan(True)

    assert baseline.denial_reason is DenialReason.BUDGET_EXCEEDED
    assert _intent_digest(patched) == _intent_digest(baseline)


def test_registered_wall_shortcut_covers_every_flight_level(monkeypatch):
    cfg = replace(CFG, terminal_airspace_always_active=True)
    led = ReservationLedger(cfg)
    term, center = Terminal("H", 4, radius=90.0), vec(1000, 1000, 0)
    led.register_static_terminal(center, term)
    tcap = TerminalCapacity(cfg, led)

    assert led.has_static_terminal(term.id)
    assert led.static_volumes()[0].z_range == (cfg.ground_level_m, cfg.airspace_ceiling_m)

    def _scan_started(*_args, **_kwargs):
        raise AssertionError("registered always-active column should return before the dynamic scan")

    monkeypatch.setattr(tc, "terminal_radius", _scan_started)
    for z in cfg.flight_levels_m:
        assert tcap.column_clear(term, center, 0.0, z=z)


def test_always_active_shortcut_is_exact_with_multiple_flight_levels(monkeypatch):
    """Compact integration guard for the same three-level path used by the 3,000-flight acceptance run."""
    cfg = SimConfig(
        terminal_airspace_always_active=True,
        flight_levels_m=(30.0, 70.0, 110.0),
        airspace_ceiling_m=125.0,
        region_size_m=(6000.0, 6000.0),
        horizon_s=240.0,
        lam_per_hour=300.0,
        seed=2,
    )
    demand = HubRadiusDemand(
        n_hubs_per_uss={"wing_uss": 3},
        radius_m=2000.0,
        pads_per_hub=4,
        terminal_radius_m=120.0,
        return_flights=True,
    )

    monkeypatch.setattr(tc, "SKIP_FOREIGN_WHEN_WALLED", False)
    baseline = run(cfg, demand=demand)
    monkeypatch.setattr(tc, "SKIP_FOREIGN_WHEN_WALLED", True)
    patched = run(cfg, demand=demand)

    assert len(baseline.intents) > 10
    assert baseline.verified and patched.verified
    assert [_intent_digest(intent) for intent in baseline.intents] == [
        _intent_digest(intent) for intent in patched.intents
    ]
    assert _committed_digest(baseline.ledger) == _committed_digest(patched.ledger)
    assert _static_digest(baseline.ledger) == _static_digest(patched.ledger)


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
