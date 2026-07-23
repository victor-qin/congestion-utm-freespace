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


# --- multi-altitude: the continuous cruise band [z_min_m, z_max_m] ---------------------------------

def _z_profile(intent):
    """Rounded cruise altitudes along the centerline."""
    return [round(float(p[2]), 1) for p, _ in intent.centerline]


def _low_wall(x=1000.0, half_y=800.0):
    """A wide permanent wall spanning z 15..70: blocks the band floor (30) AND the straight warm
    planner's 75 m plane, but leaves the upper band clear — the only cheap lever is to climb."""
    return Volume4D(box_from_segment(vec(x, -half_y, 42.5), vec(x, half_y, 42.5), 40, 55.0), 0.0, 1e6)


def test_milp_cruises_at_band_floor_in_empty_airspace():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    intent = MILPOptPlanner().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert max(_z_profile(intent)) <= cfg.z_min_m + 1e-6      # cheapest descent ⇒ the band floor
    assert abs(intent.altitude_change_m - 2.0 * (cfg.z_min_m - cfg.ground_level_m)) < 1e-6


def test_milp_climbs_over_a_wall_blocking_the_floor_and_the_warm_plane():
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    led.commit(99, [_low_wall()])
    # the straight warm start is blocked too (its 75 m corridor overlaps the wall) → the accepted
    # intent is the MILP solver's own, and climbing beats the ~1.6 km lateral berth
    assert StraightLineTimeShift().plan(_req(), led, cfg).status is IntentStatus.REJECTED
    intent = MILPOptPlanner().plan(_req(), led, cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(intent.volumes)
    assert max(_z_profile(intent)) > cfg.z_min_m + 10.0       # used the vertical lever
    assert intent.air_detour_m < 200.0                        # not a big lateral berth


def test_milp_single_plane_band_recovers_legacy():
    cfg = SimConfig(z_min_m=75.0, z_max_m=75.0)
    intent = MILPOptPlanner().plan(_req(), ReservationLedger(cfg), cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert set(_z_profile(intent)) == {75.0}
    assert abs(intent.altitude_change_m - 2.0 * (75.0 - cfg.ground_level_m)) < 1e-6


def test_milp_long_flight_past_max_steps_capacity_still_solves():
    # regression (#36 denial root-cause): max_steps=60 caps path capacity at 59·v_step = 7,080 m,
    # which made every longer flight's model kinematically INFEASIBLE before obstacles entered
    # (CBC proved it instantly; the flight was denied whenever no warm start rescued it). The N
    # feasibility floor must keep such flights solvable. _solve is called directly so the warm
    # planner cannot mask a vacuous model.
    cfg = SimConfig()
    led = ReservationLedger(cfg)
    cap_m = (MILPOptPlanner().max_steps - 1) * cfg.nominal_speed_mps * cfg.dt_s
    req = FlightRequest(1, vec(0, 0, 0), vec(cap_m + 1000.0, 0, 0), 0.0)
    intent = MILPOptPlanner()._solve(req, led, cfg)
    assert intent is not None and intent.status is IntentStatus.ACCEPTED
    assert intent.air_detour_m < 5.0                          # straight through empty airspace


def test_milp_altitude_change_books_endpoint_formula():
    from freespace_sim.cost import endpoint_altitude_change_m

    cfg = SimConfig()
    intent = MILPOptPlanner().plan(_req(), ReservationLedger(cfg), cfg)
    zs = [float(p[2]) for p, _ in intent.centerline]
    interior = sum(abs(b - a) for a, b in zip(zs, zs[1:]))
    expect = endpoint_altitude_change_m(zs[0], zs[-1], interior, cfg)
    assert abs(intent.altitude_change_m - expect) < 1e-6
