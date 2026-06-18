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


def test_uses_ground_delay_to_clear_a_busy_destination_pad():
    # The landing pad is occupied early; no spatial detour can avoid the destination itself, so the
    # only fix is to take off later (ground delay) and arrive after the pad clears.
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    blocker = Volume4D(CylinderSpec(2400, 0, cfg.effective_hover_radius_m, 0, 150), 0.0, 200.0)
    led.commit(99, [blocker])
    intent = SpaceTimeRRTStar().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.ground_delay_s > 0.0        # ← the lever we just added
    assert not led.any_conflict(intent.volumes)


def test_detour_budget_denies_with_budget_exceeded():
    cfg = SimConfig(max_detour_factor=1.01)     # any detour around the wall exceeds 1%
    led = ReservationLedger(cfg)
    led.commit(99, [_wall()])
    intent = SpaceTimeRRTStar().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.REJECTED
    assert intent.denial_reason is DenialReason.BUDGET_EXCEEDED    # real congestion denial
