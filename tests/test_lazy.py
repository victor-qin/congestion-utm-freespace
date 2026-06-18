from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.types import FlightRequest, IntentStatus, vec
from freespace_sim.volumes import Volume4D


def _req(fid=1, dest_x=2400.0):
    return FlightRequest(fid, vec(0, 0, 0), vec(dest_x, 0, 0), 0.0)


def _wall():
    spec = box_from_segment(vec(1200, -200, 150), vec(1200, 200, 150), width=40, height=400)
    return Volume4D(spec, 0.0, 1e6)


def _hover_barrier(x, ys, t_start, t_end, cfg):
    """A picket line of hover reservations (drones holding position) across the path at x."""
    return [Volume4D(CylinderSpec(x, y, cfg.effective_hover_radius_m, 0, 150), t_start, t_end)
            for y in ys]


def test_lazy_uses_cheap_straight_tier_when_possible():
    cfg = SimConfig()
    intent = get_planner("lazy").plan(_req(), ReservationLedger(cfg), cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.planner == "lazy"
    assert intent.air_detour_m == 0.0          # straight tier handled it, no spatial search


def test_lazy_escalates_to_rrt_when_straight_denies():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    led.commit(99, [_wall()])
    intent = get_planner("lazy").plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.planner == "lazy"
    assert intent.air_detour_m > 0.0           # escalated to RRT*, went around the wall
    assert not led.any_conflict(intent.volumes)


def test_lazy_escalates_to_rrt_under_persistent_crowding():
    # A wall of long-dwell hovering drones (y ∈ [-300, 300]) blocks the straight path and never
    # clears, so time-shift can't help → lazy must escalate to RRT* and detour around the crowd.
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    barrier = _hover_barrier(2000, (-240, -120, 0, 120, 240), 0.0, 1e6, cfg)
    led.commit(900, barrier)
    req = _req(1, dest_x=4000.0)
    # straight alone is helpless against a persistent barrier
    assert get_planner("straight").plan(req, led, cfg).status is IntentStatus.REJECTED
    intent = get_planner("lazy").plan(req, led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.planner == "lazy"
    assert intent.air_detour_m > 0.0           # RRT* activated and routed around the crowd
    assert intent.ground_delay_s == 0.0        # no waiting needed — it's a spatial problem
    assert not led.any_conflict(intent.volumes)


def test_lazy_uses_cheap_ground_delay_for_a_temporary_block():
    # Same geometry but the crowd disperses at t=200 s (< max_ground_delay). Now the cheapest lever
    # is simply to wait, so lazy stays on the straight tier — no spatial detour, just a delay.
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    barrier = _hover_barrier(2000, (-60, 0, 60), 0.0, 200.0, cfg)
    led.commit(900, barrier)
    intent = get_planner("lazy").plan(_req(1, dest_x=4000.0), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.air_detour_m == 0.0          # straight tier waited it out — no detour
    assert intent.ground_delay_s > 0.0         # it delayed takeoff until the path cleared
    assert not led.any_conflict(intent.volumes)
