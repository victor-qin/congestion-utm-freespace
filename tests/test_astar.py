import dataclasses as dc
import math

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner, hexgrid as hg
from freespace_sim.planner.astar import AStarPlanner, _committed_arrival
from freespace_sim.planner.occupancy import HexOccupancyService
from freespace_sim.sim import run
from freespace_sim.types import DenialReason, FlightRequest, IntentStatus, Terminal, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()


def _req(fid=1):
    return FlightRequest(fid, vec(0, 0, 0), vec(2000, 0, 0), 0.0)


def _wall():
    return Volume4D(box_from_segment(vec(1000, -200, 150), vec(1000, 200, 150), 40, 400), 0.0, 1e6)


def test_get_planner_astar():
    assert isinstance(get_planner("astar"), AStarPlanner)


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


def test_astar_compute_cap_truncation_is_search_exhausted():
    # Stopping at the expansion cap is a COMPUTE artifact -> SEARCH_EXHAUSTED (a higher cap might have
    # found a path). max_expansions=0 truncates the search on its first expansion.
    led = ReservationLedger(CFG)
    intent = AStarPlanner(max_expansions=0).plan(_req(), led, CFG)
    assert intent.status is IntentStatus.REJECTED
    assert intent.denial_reason is DenialReason.SEARCH_EXHAUSTED


def test_astar_no_feasible_plan_is_budget_exceeded():
    # Exhausting the bounded search with NO feasible plan (here: dest pad blocked past the horizon) is
    # real congestion -> BUDGET_EXCEEDED, not the compute-artifact SEARCH_EXHAUSTED. A* is complete
    # within the horizon, so an emptied queue PROVES infeasibility — distinct from giving up on compute.
    cfg = dc.replace(CFG, max_ground_delay_s=20.0)
    led = ReservationLedger(cfg)
    led.commit(99, [Volume4D(CylinderSpec(400, 0, 60, 0, 150), 0.0, 1e5)])   # dest pad blocked ~forever
    intent = AStarPlanner().plan(FlightRequest(1, vec(0, 0, 0), vec(400, 0, 0), 0.0), led, cfg)
    assert intent.status is IntentStatus.REJECTED
    assert intent.denial_reason is DenialReason.BUDGET_EXCEEDED


def test_astar_is_deterministic():
    led = ReservationLedger(CFG)
    led.commit(99, [_wall()])
    a = AStarPlanner().plan(_req(), led, CFG)
    b = AStarPlanner().plan(_req(), led, CFG)
    assert a.cost == b.cost
    assert len(a.centerline) == len(b.centerline)


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
    # the MILP refiner cuts A*'s conservative 400 m berth to the global optimum — restructuring the
    # segment count within the homotopy, which a pure smoothing polish cannot.
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


def test_committed_arrival_gates_at_the_folded_dest_column_time_not_the_goal_step():
    # Issue #15 tripwire: the landing gate must count capacity at the time _build COMMITS the dest column
    # — the tail-folded column-edge arrival — not the goal-hex step time st[3]*dt. _committed_arrival
    # rebuilds the candidate path and folds it through the SAME _fold_path _build uses, so the gate time
    # and the committed dest-column t_start agree bit-for-bit; and that time is strictly earlier than the
    # goal-hex step (proving we no longer gate at st[3]*dt, which over-subscribed pads on 7/8 dallas seeds).
    # Legacy path only: fixed exit lanes root the corridor at the boundary cell (no tail fold), so
    # _committed_arrival / _fold_path are the fixed_exit_lanes=False landing gate.
    cfg = SimConfig(fixed_exit_lanes=False)
    dt, R = cfg.dt_s, hg.circumradius(cfg)
    dest = vec(0, 0, 0)
    dest_term = Terminal("H", 2, radius=300.0)            # wide column → the straight-in tail clearly folds
    lvl = 0                                                # a single cruise level for the straight-in path
    # a straight-in air path along the q-axis toward the dest hub, one hex per step (q=5→0, steps 10→15)
    air = [("a", q, 0, lvl, 10 + (5 - q)) for q in (5, 4, 3, 2, 1, 0)]
    goal = air[-1]
    came = {air[i]: air[i - 1] for i in range(1, len(air))}
    came[air[0]] = ("g", 5, 0, 9)                          # the takeoff ground state ends the air walk
    origin = vec(*hg.hex_center(5, 0, R), 0.0)

    arr = _committed_arrival(goal, came, R, dt, cfg, origin, dest, None, dest_term)

    # gate == commit: equals the dest-column t_start _build stamps (both fold via _fold_path)
    cruise_wps = [(np.array([*hg.hex_center(q, r, R), cfg.level_z(L)]), s * dt) for (_, q, r, L, s) in air]
    volumes, *_ = AStarPlanner()._build(cruise_wps, origin, dest, 0, 0, cfg, dest_term=dest_term)
    assert arr == volumes[-1].t_start
    # and strictly earlier than the goal-hex step time — the fold moved it (not gating at st[4]*dt)
    assert arr < goal[4] * dt


# --- multi-altitude: discrete flight levels --------------------------------------------------------

def _cruise_levels(intent):
    """Distinct cruise altitudes present in an intent's centerline, rounded to the metre."""
    return sorted({round(float(p[2]), 1) for p, _ in intent.centerline})


def _level_wall(z, x=1000.0, half_y=400.0):
    """A wide, all-time wall centred at altitude ``z`` (height = corridor_height ⇒ blocks ONE level)."""
    return Volume4D(
        box_from_segment(vec(x, -half_y, z), vec(x, half_y, z), 40, CFG.corridor_height_m), 0.0, 1e6
    )


def test_astar_prefers_lowest_level_in_empty_airspace():
    intent = AStarPlanner().plan(_req(), ReservationLedger(CFG), CFG)
    assert intent.status is IntentStatus.ACCEPTED
    assert _cruise_levels(intent) == [CFG.level_z(0)]                 # cheapest descent ⇒ lowest level
    assert intent.altitude_change_m == 2.0 * (CFG.level_z(0) - CFG.ground_level_m)   # 2·30 = 60


def test_astar_climbs_over_a_blocked_low_level_without_lateral_detour():
    led = ReservationLedger(CFG)
    led.commit(99, [_level_wall(CFG.level_z(0))])                     # level 0 walled across the route
    intent = AStarPlanner().plan(_req(), led, CFG)
    assert intent.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(intent.volumes)
    assert CFG.level_z(1) in _cruise_levels(intent)                  # used level 1 (70 m) to get over
    assert intent.air_detour_m < 200.0                               # vertical, not a big lateral berth
    assert intent.altitude_change_m == 2.0 * (CFG.level_z(1) - CFG.ground_level_m)   # 2·70 = 140


def test_two_flights_share_a_corridor_deconflict_by_altitude():
    """Opposite-direction flights on a long shared corridor: the second climbs rather than wait/detour."""
    led = ReservationLedger(CFG)
    a = FlightRequest(1, vec(0, 0, 0), vec(6000, 0, 0), 0.0)
    b = FlightRequest(2, vec(6000, 0, 0), vec(0, 0, 0), 0.0)         # reverse, same departure
    i1 = AStarPlanner().plan(a, led, CFG)
    assert i1.status is IntentStatus.ACCEPTED
    led.commit(1, i1.volumes)
    i2 = AStarPlanner().plan(b, led, CFG)
    assert i2.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(i2.volumes)
    assert max(_cruise_levels(i2)) >= CFG.level_z(1)                 # crossed at a higher level
    assert i2.air_detour_m < 600.0                                   # not a big lateral detour
    assert i2.ground_delay_s < 100.0                                 # not a long ground wait


def test_vertical_edge_step_count_matches_climb_kinematics():
    """Force a mid-route layer change (level 1 walled early, level 0 walled late) and check it spans
    ceil(Δz / (climb_rate·dt)) steps — 40 m / 24 m ⇒ 2 steps. Both walls are mid-route, never over a
    pad (the takeoff/landing tube reserves [ground, ceiling] at the endpoints)."""
    led = ReservationLedger(CFG)
    led.commit(98, [_level_wall(CFG.level_z(1), x=900.0)])           # level 1 blocked early → fly low
    led.commit(97, [_level_wall(CFG.level_z(0), x=1500.0)])          # level 0 blocked late → must climb
    intent = AStarPlanner().plan(_req(), led, CFG)
    assert intent.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(intent.volumes)
    cl = intent.centerline
    climbs = [(cl[i][1], cl[i + 1][1]) for i in range(len(cl) - 1)
              if round(float(cl[i][0][2])) == CFG.level_z(0)
              and round(float(cl[i + 1][0][2])) == CFG.level_z(1)]
    assert climbs, "expected a level-0 → level-1 climb mid-route"
    t_a, t_b = climbs[0]
    assert abs((t_b - t_a) - 2 * CFG.dt_s) < 1e-6                    # 2 timesteps for the 40 m rung


def test_astar_multilevel_is_deterministic():
    i1 = AStarPlanner().plan(_req(), ReservationLedger(CFG), CFG)
    i2 = AStarPlanner().plan(_req(), ReservationLedger(CFG), CFG)
    assert i1.cost == i2.cost
    assert len(i1.centerline) == len(i2.centerline)
    assert [round(float(p[2]), 3) for p, _ in i1.centerline] == \
           [round(float(p[2]), 3) for p, _ in i2.centerline]


def test_single_level_config_recovers_legacy_behavior():
    """One flight level at the old cruise plane ⇒ the legacy single-plane A* (no vertical lever)."""
    cfg = SimConfig(cruise_level_m=150.0, flight_levels_m=(150.0,), airspace_ceiling_m=165.0,
                    z_min_m=150.0, z_max_m=150.0)
    intent = AStarPlanner().plan(_req(), ReservationLedger(cfg), cfg)
    assert intent.status is IntentStatus.ACCEPTED
    assert _cruise_levels(intent) == [150.0]
    assert intent.altitude_change_m == 2.0 * (150.0 - cfg.ground_level_m)


def _air_edges(planner, cfg, svc, st, max_step=999):
    """Expand an AIR state (reroute/hover/vertical-edge only). The ground-branch params (takeoff_steps,
    tcap, …) aren't consulted for an ``("a", …)`` state, so dummies are fine."""
    n, lv = cfg.n_levels, cfg.flight_levels_m
    rung_steps = tuple(max(1, math.ceil((lv[L + 1] - lv[L]) / (cfg.climb_rate_mps * cfg.dt_s)))
                       for L in range(n - 1))
    rung_cost = tuple(cfg.cost_altitude_change_per_m * (lv[L + 1] - lv[L]) for L in range(n - 1))
    return planner._edges(st, cfg, cfg.corridor_segment_len_m, lv, (0,) * n, (0.0,) * n,
                          rung_steps, rung_cost, (1,) * n, cfg.cost_altitude_change_per_m, svc, max_step)


def test_vertical_edge_checks_only_traversed_levels_not_all():
    """A 0→1 layer-change edge must require clearance only on the levels it traverses ({0, 1}): an
    obstacle on the UNtraversed level 2 over the same column must NOT block it, while one on the
    destination level 1 must. (Before the fix the edge required ALL levels clear.)"""
    planner = AStarPlanner()
    q, r, s = 0, 0, 5
    vsteps = max(1, math.ceil((CFG.level_z(1) - CFG.level_z(0)) / (CFG.climb_rate_mps * CFG.dt_s)))
    climb_edge = ("a", q, r, 1, s + vsteps)                          # the 0→1 rung successor

    blocked_above = HexOccupancyService(CFG)
    for sk in range(s + 1, s + vsteps + 1):
        blocked_above.blocked.setdefault(sk, set()).add((q, r, 2))   # obstacle two levels up
    got = {e[0] for e in _air_edges(planner, CFG, blocked_above, ("a", q, r, 0, s))}
    assert climb_edge in got, "an obstacle on untraversed level 2 wrongly blocked a 0→1 climb"

    blocked_dest = HexOccupancyService(CFG)
    for sk in range(s + 1, s + vsteps + 1):
        blocked_dest.blocked.setdefault(sk, set()).add((q, r, 1))    # obstacle on the destination level
    got2 = {e[0] for e in _air_edges(planner, CFG, blocked_dest, ("a", q, r, 0, s))}
    assert climb_edge not in got2, "an obstacle on the destination level must block the climb"
