from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.opt import NLPOptPlanner
from freespace_sim.planner.rrt import SpaceTimeRRTStar
from freespace_sim.types import FlightRequest, IntentStatus, vec
from freespace_sim.volumes import Volume4D


def _req(fid=1):
    return FlightRequest(fid, vec(0, 0, 0), vec(2400, 0, 0), 0.0)


def _wall():
    return Volume4D(box_from_segment(vec(1200, -200, 150), vec(1200, 200, 150), 40, 400), 0.0, 1e6)


def test_get_planner_opt():
    assert isinstance(get_planner("opt"), NLPOptPlanner)


def test_opt_empty_airspace_is_optimal_and_conflict_free():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    intent = NLPOptPlanner().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.air_detour_m < 1.0          # straight line is optimal here
    assert not led.any_conflict(intent.volumes)


def test_opt_improves_on_rrt_around_a_wall():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    led.commit(99, [_wall()])
    rrt_intent = SpaceTimeRRTStar().plan(_req(), led, cfg)
    opt_intent = NLPOptPlanner().plan(_req(), led, cfg)
    assert opt_intent.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(opt_intent.volumes)        # rebuilt + re-checked boxes
    assert opt_intent.cost < rrt_intent.cost               # NLP polish beats first-feasible RRT*


def test_opt_never_worse_than_rrt():
    # when RRT* is already optimal (empty airspace) the fallback must prevent any regression
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    rrt_intent = SpaceTimeRRTStar().plan(_req(), led, cfg)
    opt_intent = NLPOptPlanner().plan(_req(), led, cfg)
    assert opt_intent.cost <= rrt_intent.cost + 1e-6


def test_opt_uses_takeoff_delay_for_busy_destination_pad():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    led.commit(99, [Volume4D(CylinderSpec(2400, 0, cfg.effective_hover_radius_m, 0, 150), 0.0, 200.0)])
    opt_intent = NLPOptPlanner().plan(_req(), led, cfg)
    assert opt_intent.status is IntentStatus.ACCEPTED
    assert opt_intent.ground_delay_s > 0.0     # takeoff time is a decision variable in the NLP
    assert not led.any_conflict(opt_intent.volumes)
