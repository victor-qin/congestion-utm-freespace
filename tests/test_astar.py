import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.opt import NLPOptPlanner
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, IntentStatus, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()


def _req(fid=1):
    return FlightRequest(fid, vec(0, 0, 0), vec(2000, 0, 0), 0.0)


def _wall():
    return Volume4D(box_from_segment(vec(1000, -200, 150), vec(1000, 200, 150), 40, 400), 0.0, 1e6)


def test_get_planner_astar_and_opt_astar():
    assert isinstance(get_planner("astar"), AStarPlanner)
    assert isinstance(get_planner("opt_astar"), NLPOptPlanner)


def test_astar_empty_airspace_accepted_and_conflict_free():
    led = ReservationLedger(CFG)
    intent = AStarPlanner().plan(_req(), led, CFG)
    assert intent.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(intent.volumes)
    assert intent.air_detour_m < 0.5 * 2000          # only the hex staircase, not a real detour


def test_astar_reroutes_around_a_wall_that_straight_cannot_pass():
    led = ReservationLedger(CFG)
    led.commit(99, [_wall()])
    assert get_planner("straight").plan(_req(), led, CFG).status is IntentStatus.REJECTED
    intent = AStarPlanner().plan(_req(), led, CFG)
    assert intent.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(intent.volumes)
    assert intent.air_detour_m > 0.0                 # deterministically routed around


def test_astar_uses_ground_delay_for_a_busy_destination_pad():
    led = ReservationLedger(CFG)
    led.commit(99, [Volume4D(CylinderSpec(2000, 0, 60, 0, 150), 0.0, 200.0)])
    intent = AStarPlanner().plan(_req(), led, CFG)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.ground_delay_s > 0.0               # cheapest lever: wait on the pad
    assert not led.any_conflict(intent.volumes)


def test_astar_is_deterministic():
    led = ReservationLedger(CFG)
    led.commit(99, [_wall()])
    a = AStarPlanner().plan(_req(), led, CFG)
    b = AStarPlanner().plan(_req(), led, CFG)
    assert a.cost == b.cost
    assert len(a.centerline) == len(b.centerline)


def test_opt_astar_smooths_the_staircase_and_never_worsens():
    led = ReservationLedger(CFG)
    astar = get_planner("astar").plan(_req(), led, CFG)
    opt = get_planner("opt_astar").plan(_req(), led, CFG)
    assert opt.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(opt.volumes)
    assert opt.air_detour_m <= astar.air_detour_m + 1e-6   # NLP polish never worsens
    assert opt.air_detour_m < astar.air_detour_m           # and in open space it removes the staircase


def test_astar_milp_refiner_keeps_astars_delay_and_smooths():
    # delay-dominated case: A* picks the 120 s wait, the fixed-delay MILP refines the geometry fast
    led = ReservationLedger(CFG)
    led.commit(99, [Volume4D(CylinderSpec(2000, 0, 60, 0, 150), 0.0, 200.0)])
    astar = get_planner("astar").plan(_req(), led, CFG)
    refined = get_planner("astar_milp").plan(_req(), led, CFG)
    assert refined.status is IntentStatus.ACCEPTED
    assert refined.ground_delay_s > 0.0                   # kept A*'s ground-delay choice
    assert refined.air_detour_m <= astar.air_detour_m + 1e-6
    assert not led.any_conflict(refined.volumes)


@pytest.mark.slow
def test_astar_milp_refiner_restructures_the_wide_berth():
    # the MILP refiner cuts A*'s conservative 400 m berth to the global optimum — which the NLP
    # (opt_astar) cannot, because it can't change the number of segments or the homotopy.
    led = ReservationLedger(CFG)
    led.commit(99, [_wall()])
    astar = get_planner("astar").plan(_req(), led, CFG)
    refined = get_planner("astar_milp").plan(_req(), led, CFG)
    assert refined.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(refined.volumes)
    assert refined.cost < astar.cost
    assert refined.air_detour_m < astar.air_detour_m - 100.0   # genuinely restructured, not nudged


def test_astar_demand_run_is_verified():
    cfg = SimConfig(
        planner="astar", lam_per_hour=40.0, horizon_s=900.0, seed=4, region_size_m=(4000.0, 4000.0)
    )
    res = run(cfg)
    assert res.verified
