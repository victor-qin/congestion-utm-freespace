from freespace_sim.config import SimConfig
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner.decoupled import DecoupledPlanner
from freespace_sim.planner.straight import build_reservation
from freespace_sim.types import FlightRequest, IntentStatus, vec
from freespace_sim.volumes import Volume4D


def _req(fid=1):
    return FlightRequest(fid, vec(0, 0, 0), vec(2400, 0, 0), 0.0)


def test_empty_airspace_accepts():
    cfg = SimConfig()
    intent = DecoupledPlanner().plan(_req(), ReservationLedger(cfg), cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.planner == "decoupled"


def test_time_clearable_block_is_scheduled_around():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    vols, _ = build_reservation(vec(0, 0, 0), vec(2400, 0, 0), 0.0, cfg)
    led.commit(0, vols)                          # an earlier flight on the same path
    intent = DecoupledPlanner().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED   # finds a free schedule slot


def test_persistent_wall_denies_no_spatial_lever():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    spec = box_from_segment(vec(1200, -200, 150), vec(1200, 200, 150), width=40, height=400)
    led.commit(99, [Volume4D(spec, 0.0, 1e6)])
    intent = DecoupledPlanner().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.REJECTED   # can't bend in space → no schedule helps
