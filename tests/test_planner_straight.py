from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner.straight import StraightLineTimeShift
from freespace_sim.types import FlightRequest, IntentStatus, vec


def _req(fid, t=0.0):
    return FlightRequest(fid, vec(0, 0, 0), vec(2400, 0, 0), t)


def test_empty_airspace_accepts_with_no_delay():
    cfg = SimConfig()
    intent = StraightLineTimeShift().plan(_req(0), ReservationLedger(cfg), cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.ground_delay_s == 0.0
    assert intent.volumes and intent.centerline


def test_blocked_flight_time_shifts_then_accepts():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    planner = StraightLineTimeShift()
    first = planner.plan(_req(0), led, cfg)
    led.commit(0, first.volumes)
    second = planner.plan(_req(1), led, cfg)   # identical path/time → must yield
    assert second.status is IntentStatus.ACCEPTED
    assert second.ground_delay_s > 0.0


def test_denies_when_delay_budget_too_small():
    cfg = SimConfig(max_ground_delay_s=10.0)
    led = ReservationLedger(cfg)
    planner = StraightLineTimeShift()
    led.commit(0, planner.plan(_req(0), led, cfg).volumes)
    second = planner.plan(_req(1), led, cfg)   # needs ~160 s, budget is 10 s
    assert second.status is IntentStatus.REJECTED


def test_cost_reflects_ground_delay():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    planner = StraightLineTimeShift()
    led.commit(0, planner.plan(_req(0), led, cfg).volumes)
    second = planner.plan(_req(1), led, cfg)
    assert second.cost >= cfg.cost_ground_delay_per_s * second.ground_delay_s
