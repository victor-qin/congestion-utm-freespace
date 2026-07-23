from freespace_sim.config import SimConfig
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.milp import MILPOptPlanner
from freespace_sim.planner.straight import StraightLineTimeShift
from freespace_sim.types import FlightRequest, IntentStatus, vec
from freespace_sim.volumes import Volume4D


def _req(fid=1, dx=2000.0):
    return FlightRequest(fid, vec(0, 0, 0), vec(dx, 0, 0), 0.0)


def _wall(x=1000.0):
    return Volume4D(box_from_segment(vec(x, -200, 150), vec(x, 200, 150), 40, 400), 0.0, 1e6)


def test_get_planner_milp():
    assert isinstance(get_planner("milp"), MILPOptPlanner)


def test_milp_empty_airspace_optimal_and_conflict_free():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    intent = MILPOptPlanner().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.air_detour_m < 5.0
    assert not led.any_conflict(intent.volumes)


def test_milp_global_optimum_around_wall():
    # the straight warm start DENIES here (a permanent wall admits no time-shift), so the accepted
    # intent is the MILP solver's own — the near-optimal berth proves the global solve, not a fallback
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    led.commit(99, [_wall()])                       # planners don't mutate the ledger → share it
    assert StraightLineTimeShift().plan(_req(), led, cfg).status is IntentStatus.REJECTED
    milp = MILPOptPlanner().plan(_req(), led, cfg)
    assert milp.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(milp.volumes)       # rebuilt + re-checked boxes
    assert milp.air_detour_m < 80.0                 # near the ~47 m geometric minimum


def test_milp_never_worse_than_warm():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    warm = StraightLineTimeShift().plan(_req(), led, cfg)
    milp = MILPOptPlanner().plan(_req(), led, cfg)
    assert milp.cost <= warm.cost + 1e-6            # plan() returns min(warm, solver) by cost


def _wide_wall(clear_t):
    # a wall wide enough that going around it costs far more than a short wait
    return Volume4D(box_from_segment(vec(1000, -800, 150), vec(1000, 800, 150), 40, 400), 0.0, clear_t)


def test_milp_waits_when_a_temporary_block_makes_detour_expensive():
    # delay is a decision variable now: a wide wall that CLEARS soon → the global solve WAITS
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    led.commit(99, [_wide_wall(70.0)])
    intent = MILPOptPlanner().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.ground_delay_s > 0.0 and intent.air_detour_m < 10.0   # chose to wait, not detour
    assert not led.any_conflict(intent.volumes)


def test_milp_detours_when_the_same_wide_wall_is_permanent():
    # identical geometry but it never clears → waiting can't help, so the global solve DETOURS
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    led.commit(99, [_wide_wall(1e6)])
    intent = MILPOptPlanner().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert intent.air_detour_m > 10.0
    assert not led.any_conflict(intent.volumes)
