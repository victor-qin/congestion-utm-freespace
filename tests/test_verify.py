from freespace_sim.config import SimConfig
from freespace_sim.planner.straight import build_reservation
from freespace_sim.types import FlightRequest, IntentStatus, OperationalIntent, Terminal, vec
from freespace_sim.verify import find_interflight_conflict

CFG = SimConfig()


def _intent(fid, t_dep):
    req = FlightRequest(fid, vec(0, 0, 0), vec(2400, 0, 0), 0.0)
    vols, cl = build_reservation(req.origin, req.dest, t_dep, CFG)
    return OperationalIntent(req, IntentStatus.ACCEPTED, volumes=vols, centerline=cl)


def test_clean_when_separated_in_time():
    a = _intent(0, 0.0)
    b = _intent(1, 1000.0)
    assert find_interflight_conflict([a, b], CFG) is None


def test_detects_injected_inter_flight_overlap():
    a = _intent(0, 0.0)
    b = _intent(1, 0.0)   # identical reservation as a different flight → must be caught
    assert find_interflight_conflict([a, b], CFG) == (1, 0)


def test_verify_flags_static_wall_crossing():
    """Correctness gain: verify now catches a committed corridor that crosses a walled (foreign) static
    terminal — a property it was structurally blind to when the walls lived off-ledger. The partner id is
    the -1 sentinel (the wall has no owning flight). Its path (0,0)->(2400,0) crosses a hub at (1200,0)."""
    a = _intent(7, 0.0)
    hub = Terminal("h#0", 8, 180.0)
    bad = find_interflight_conflict([a], CFG, static_terminals=[((1200.0, 0.0), hub)])
    assert bad == (7, -1), "verify must flag the wall crossing with the -1 static sentinel"


def test_verify_no_false_positive_when_wall_off_path():
    """No false positive from the new static check: the same intent with the hub OFF its path verifies
    clean (guards against the AABB prune or the sentinel path over-triggering)."""
    a = _intent(7, 0.0)
    hub = Terminal("h#0", 8, 180.0)
    assert find_interflight_conflict([a], CFG, static_terminals=[((1200.0, 3000.0), hub)]) is None


def test_verify_flags_wall_crossing_late_in_schedule():
    """Regression for the finite-window bug: the permanent wall must cover the WHOLE schedulable horizon
    (latest departure + ground-delay budget + travel), not just horizon_s. A corridor crossing a hub at a
    time PAST horizon_s must still be caught — otherwise a flight could ground-delay past the wall and fly
    through it undetected. Departs at horizon_s+max_ground_delay+100 (well past the old horizon_s+buffer)."""
    a = _intent(9, CFG.horizon_s + CFG.max_ground_delay_s + 100.0)   # corridor time is far past horizon_s
    hub = Terminal("h#0", 8, 180.0)
    bad = find_interflight_conflict([a], CFG, static_terminals=[((1200.0, 0.0), hub)])
    assert bad == (9, -1), "verify must catch a wall crossing scheduled past horizon_s (whole-window wall)"
