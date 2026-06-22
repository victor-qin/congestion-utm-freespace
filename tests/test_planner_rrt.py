from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner.rrt import SpaceTimeRRTStar
from freespace_sim.planner.straight import StraightLineTimeShift
from freespace_sim.types import DenialReason, FlightRequest, IntentStatus, vec
from freespace_sim.volumes import Volume4D


def _req(fid=1):
    return FlightRequest(fid, vec(0, 0, 0), vec(2400, 0, 0), 0.0)


def _wall():
    # a persistent box across the straight path (x≈1200), spanning all altitudes and all time,
    # finite in y so a lateral detour exists but waiting never helps
    spec = box_from_segment(vec(1200, -200, 150), vec(1200, 200, 150), width=40, height=400)
    return Volume4D(spec, 0.0, 1e6)


def test_empty_airspace_path_is_conflict_free():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    intent = SpaceTimeRRTStar().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(intent.volumes)
    assert intent.centerline


def test_reroutes_around_wall_that_straight_cannot_pass():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    led.commit(99, [_wall()])
    # straight-line cannot pass a persistent wall by waiting → it denies
    assert StraightLineTimeShift().plan(_req(), led, cfg).status is IntentStatus.REJECTED
    intent = SpaceTimeRRTStar().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(intent.volumes)
    assert intent.air_detour_m > 0.0           # it went around


def test_deterministic_under_seed():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    led.commit(99, [_wall()])
    a = SpaceTimeRRTStar().plan(_req(), led, cfg)
    b = SpaceTimeRRTStar().plan(_req(), led, cfg)
    assert a.status is IntentStatus.ACCEPTED and b.status is IntentStatus.ACCEPTED
    assert a.cost == b.cost
    assert len(a.centerline) == len(b.centerline)


def test_sample_cap_denies_with_search_exhausted():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    led.commit(99, [_wall()])
    intent = SpaceTimeRRTStar(max_samples=3).plan(_req(), led, cfg)
    assert intent.status is IntentStatus.REJECTED
    assert intent.denial_reason is DenialReason.SEARCH_EXHAUSTED   # compute artifact, not physics


def test_uses_ground_delay_to_clear_a_busy_takeoff_pad():
    # The TAKEOFF pad is occupied early. You cannot detour or air-hold your way out of a blocked
    # origin — the only escape is to wait on the ground — so this isolates the ground-delay lever.
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    blocker = Volume4D(CylinderSpec(0, 0, cfg.effective_hover_radius_m, 0, 150), 0.0, 200.0)
    led.commit(99, [blocker])
    intent = SpaceTimeRRTStar().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.ground_delay_s > 0.0        # ← the ground-delay lever, deterministically forced
    assert not led.any_conflict(intent.volumes)


def test_clears_a_busy_destination_pad_by_deferring_arrival():
    # The landing pad is occupied until t=200. RRT must defer arrival past that window — by ground
    # delay, an air hold, or a long-enough timing detour (any is valid; the planner returns the first
    # feasible connection, not a specific lever). The invariant is: accepted, conflict-free, and the
    # landing dwell starts only after the pad clears.
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    blocker = Volume4D(CylinderSpec(2400, 0, cfg.effective_hover_radius_m, 0, 150), 0.0, 200.0)
    led.commit(99, [blocker])
    intent = SpaceTimeRRTStar().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(intent.volumes)
    assert intent.centerline[-1][1] >= 200.0   # arrival deferred until the pad is free


def test_cruise_stays_on_the_single_airspace_altitude_level():
    # The airspace band is currently collapsed to one flight level (cfg.z_min_m == z_max_m), so
    # altitude is not a deconfliction lever yet: every RRT* cruise waypoint sits on that level and
    # cum_dz is zero. (Sampling across a band would make the top-down replay show vertically
    # separated flights as phantom overlaps.) Widening the band re-enables multi-altitude routing.
    cfg = SimConfig()
    assert cfg.z_min_m == cfg.z_max_m == cfg.cruise_level_m   # single-level airspace (for now)
    led = ReservationLedger(cfg)
    led.commit(99, [_wall()])                       # force a real lateral detour, exercising sampling
    intent = SpaceTimeRRTStar().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert all(abs(float(p[2]) - cfg.cruise_level_m) < 1e-6 for p, _ in intent.centerline)
    # altitude_change is exactly the mandatory climb + descent (no in-cruise vertical travel)
    assert intent.altitude_change_m == 2.0 * (cfg.cruise_level_m - cfg.ground_level_m)


def test_detour_budget_denies_with_budget_exceeded():
    cfg = SimConfig(max_detour_factor=1.01)     # any detour around the wall exceeds 1%
    led = ReservationLedger(cfg)
    led.commit(99, [_wall()])
    intent = SpaceTimeRRTStar().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.REJECTED
    assert intent.denial_reason is DenialReason.BUDGET_EXCEEDED    # real congestion denial
