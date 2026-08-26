import dataclasses as dc
import math

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner, hexgrid as hg
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.astar.occupancy import HexOccupancyService
from freespace_sim.planner.astar.planner import _committed_arrival
from freespace_sim.planner.milp import MILPOptPlanner
from freespace_sim.planner.shortcut import ShortcutRefiner
from freespace_sim.sim import run
from freespace_sim.types import (
    DenialReason,
    FlightRequest,
    IntentStatus,
    OperationalIntent,
    Terminal,
    vec,
)
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


def test_two_flights_share_a_corridor_deconflict_by_the_cheapest_lever():
    """Opposite-direction flights on a long shared corridor: the second must give way, and it picks the
    lever the cost weights actually make cheapest — a short lateral sidestep.

    With the weights normalized to one per-second currency (ground 1x, lateral 3x, hover 3x, climb 4x
    PER SECOND) stepping one hex aside costs 3 s-equivalents, while a rung to the next flight level is
    charged the climb TIME at 4x — so a two-hex berth undercuts both a climb and any comparable hold.
    Asserts that ORDERING rather than magic thresholds, so the test keeps its meaning if the weights are
    retuned; only a change to the ranking itself should force a rewrite.
    """
    led = ReservationLedger(CFG)
    a = FlightRequest(1, vec(0, 0, 0), vec(6000, 0, 0), 0.0)
    b = FlightRequest(2, vec(6000, 0, 0), vec(0, 0, 0), 0.0)         # reverse, same departure
    i1 = AStarPlanner().plan(a, led, CFG)
    assert i1.status is IntentStatus.ACCEPTED
    led.commit(1, i1.volumes)
    i2 = AStarPlanner().plan(b, led, CFG)
    assert i2.status is IntentStatus.ACCEPTED
    assert not led.any_conflict(i2.volumes)

    berth = CFG.cost_air_lateral_per_m * i2.air_detour_m             # what it paid to step aside
    one_rung = CFG.cost_altitude_change_per_m * 2.0 * (CFG.level_z(1) - CFG.level_z(0))
    assert i2.air_detour_m > 0.0                                     # gave way laterally ...
    assert berth < one_rung                                          # ... because that undercuts a climb
    assert _cruise_levels(i2) == [CFG.level_z(0)]                    # so it never left the floor level
    assert CFG.cost_ground_delay_per_s * i2.ground_delay_s < one_rung  # nor did it out-wait a climb
    # The berth is traffic-forced, not hex staircase: this corridor is axis-aligned, so an unimpeded run
    # quantizes to zero overhead and every metre of the detour is attributable to flight 1. It is also a
    # whole number of hex steps, because that is the only way A* can move sideways.
    assert i2.lattice_overhead_m == 0.0
    pitch = CFG.nominal_speed_mps * CFG.dt_s
    assert (i2.air_detour_m - i2.lattice_overhead_m) % pitch == 0.0


@pytest.mark.parametrize("deg, on_axis", [(0.0, True), (60.0, True), (10.0, False), (30.0, False)])
def test_lattice_overhead_absorbs_quantization_leaving_no_phantom_deconfliction(deg, on_axis):
    """An unimpeded flight has no traffic to avoid, so every metre of ``air_detour_m`` must land in
    ``lattice_overhead_m`` and NONE in the traffic residual — at ANY bearing.

    ``air_detour_m`` is measured against the Euclidean straight line, which a 6-direction lattice
    simply cannot fly: off a hex axis the shortest lattice path is up to 2/√3 − 1 ≈ 15.5% longer no
    matter how empty the sky is. Without this split an entirely congestion-free run books that pure
    geometry as congestion, and at low load it dominates the reported detour outright.
    """
    cfg = SimConfig(flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
                    region_size_m=(20_000.0, 20_000.0))
    th, d = math.radians(deg), 6000.0
    req = FlightRequest(1, vec(0, 0, 0), vec(d * math.cos(th), d * math.sin(th), 0), 0.0)
    intent = AStarPlanner().plan(req, ReservationLedger(cfg), cfg)
    assert intent.accepted
    assert intent.air_detour_m - intent.lattice_overhead_m == 0.0     # nothing blamed on traffic
    if on_axis:
        assert intent.lattice_overhead_m == 0.0                      # an axis needs no staircase
    else:
        assert intent.lattice_overhead_m > 0.05 * d                  # ... off-axis it is substantial
        # and still under the lattice ceiling, plus the snap of each endpoint onto a cell centre
        assert intent.air_detour_m <= (2 / math.sqrt(3) - 1) * d + 2 * hg.circumradius(cfg)


def test_continuous_planners_report_no_lattice_overhead():
    """``lattice_overhead_m`` is an A*-family diagnostic: milp/straight plan on continuous geometry,
    so their ``air_detour_m`` carries no quantization and must not be discounted by this split."""
    cfg = SimConfig()
    for name in ("milp", "straight"):
        intent = get_planner(name).plan(_req(), ReservationLedger(cfg), cfg)
        assert intent.accepted and intent.lattice_overhead_m == 0.0


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
    cfg = SimConfig(flight_levels_m=(150.0,), airspace_ceiling_m=165.0)   # cruise/z derive to 150
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
                          rung_steps, rung_cost, (1,) * n, cfg.cost_altitude_change_per_m,
                          cfg.cost_air_lateral_per_m, svc, max_step)


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


class _DenyAll:
    """A warm planner that always denies — see :func:`_folded_planner`."""

    def plan(self, req, ledger, cfg):
        return OperationalIntent(request=req, status=IntentStatus.REJECTED,
                                 denial_reason=DenialReason.BUDGET_EXCEEDED, planner="deny")


def _folded_planner(name):
    """Resolve ``name`` to a planner that actually returns a TERMINAL-FOLDED path.

    ``get_planner("milp")`` is a trap here: ``MILPOptPlanner.plan`` returns the CHEAPER of its warm
    ``StraightLineTimeShift`` candidate and its own solve, and in empty airspace the warm one wins.
    That candidate is never folded to the terminal columns — its centerline starts exactly at
    ``req.origin`` — so the milp.py detour site never executes and every assertion below would pass
    vacuously (reverting the milp.py hunk left this test green). Denying the warm start forces the
    MILP's own folded path to come back.

    ``intent.planner == "milp"`` would NOT be a usable guard: ``MILPOptPlanner.plan`` relabels
    whichever candidate wins, including the warm one.
    """
    if name == "milp":
        return MILPOptPlanner(warm_planner=_DenyAll())
    return get_planner(name)


def _terminal_case(**cfg_kw):
    cfg = SimConfig(flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
                    region_size_m=(20_000.0, 20_000.0), terminal_radius_m=180.0, **cfg_kw)
    hub = Terminal("hub#0", 8, 180.0)
    return cfg, FlightRequest(1, vec(0, 0, 0), vec(5000, 2000, 0), 0.0, origin_terminal=hub)


@pytest.mark.parametrize("planner", ["astar", "astar_shortcut", "milp"])
def test_stretch_never_below_one_leaving_a_terminal(planner):
    """Regression for issue #50: a flight cannot fly SHORTER than the straight line.

    One end only: ``_terminal_case`` sets ``origin_terminal`` and leaves ``dest_terminal`` None, so a
    single column is folded — the asymmetric case, which is the harder one for ``stretch >= 1``.

    Under ``fixed_exit_lanes`` the air path starts on a hub boundary lane cell, so the centerline
    spans lane→lane while ``straight_line_m`` spans hub-centre→hub-centre. Comparing them directly
    books a phantom shortcut (mean 210.0 m/flight on density_test, 172.9 on
    dallas_hub_2uss_large) and drives ``stretch`` below 1.
    Bare A*'s hex staircase used to mask it; ``astar_shortcut`` removes the staircase and exposes it
    on ~71% of flights, so both are checked here — as is the continuous MILP.

    Which arms actually carry the regression: reverting ``_flown_horizontal_m`` fails ONLY
    ``astar_shortcut`` (measured stretch 0.9739). ``astar`` passes at 1.1130 because its staircase
    still covers the fold — the very masking described above — and ``milp`` passes at 1.0007 because
    it folds to a continuous column edge and barely leaves the ideal line. Both are kept as guards
    against future drift, not as proof; do not read three green arms as three independent checks.
    """
    from freespace_sim import metrics
    from freespace_sim.volumes import enroute_reference_m

    cfg, req = _terminal_case()
    intent = _folded_planner(planner).plan(req, ReservationLedger(cfg), cfg)
    assert intent.accepted
    # Guard the guard: the hub must actually shorten the baseline, else there is no bug to catch.
    centre = float(np.linalg.norm(np.asarray(req.dest, float)[:2] - np.asarray(req.origin, float)[:2]))
    assert enroute_reference_m(req.origin, req.dest, req.origin_terminal,
                               req.dest_terminal, cfg) < centre - 100.0, \
        f"{planner}: baseline is not lane->lane — this arm proves nothing"
    row = metrics.flight_row(intent, cfg)
    assert row["stretch"] >= 1.0 - 1e-9, f"{planner}: flew shorter than the straight line"
    # ... and the flown length must actually reach both endpoints, not stop at the lane cell
    assert row["flown_m"] >= row["straight_line_m"] - 1e-9


def test_accepted_stretch_respects_the_detour_budget():
    """The ``max_detour_factor`` gate and the reported ``stretch`` must measure the SAME ratio.

    Issue #50's first cut corrected only the readout, leaving every gate comparing the lane->lane
    ``cum_horiz`` against the centre->centre straight line. A terminal flight could then pass a
    ``max_detour_factor`` gate and report a stretch above it — measured 1.1400 against a 1.07 budget.
    Invisible at the default factor of 100.0, so pin it at a value the fold can actually breach.

    NOT parametrized, deliberately. Once the gate is correct the two refiners simply DENY at this
    budget (no path can shrink the unreserved fold), so as separate params they would be silent
    no-ops that look like passing coverage. Looping here lets the test assert that at least one
    planner actually reached the accept path — otherwise the whole check is vacuous.
    """
    from freespace_sim import metrics

    cfg, req = _terminal_case(max_detour_factor=1.07)
    accepted = 0
    for planner in ("astar", "astar_shortcut", "milp"):
        intent = _folded_planner(planner).plan(req, ReservationLedger(cfg), cfg)
        if not intent.accepted:
            continue                      # denying is a legitimate outcome; over-reporting is not
        accepted += 1
        stretch = metrics.flight_row(intent, cfg)["stretch"]
        assert stretch <= cfg.max_detour_factor + 1e-9, (
            f"{planner}: admitted at the gate but reports stretch {stretch:.4f} "
            f"> max_detour_factor {cfg.max_detour_factor}")
    assert accepted, "every planner denied — the budget check never exercised the accept path"


@pytest.mark.parametrize("planner", ["astar", "astar_shortcut", "milp", "astar_milp"])
def test_takeoff_clock_includes_the_egress_traverse(planner):
    """Issue #52: the corridor starts after climb AND the traverse out to the lane cell.

    Parametrized over EVERY planner, not just astar. The first version tested astar alone, and the
    refiners kept the bug for two more review rounds: astar read a feasible 13.60 m/s while
    astar_shortcut implied 54.42 and the MILP family 42.00, because they build through
    ``volumes.build_reservation_from_corners`` rather than ``astar._build`` and that path started the
    corridor before the egress was flown.

    A* used to advance the clock by the climb alone, so the drone teleported sideways out of its own
    column — 272 m in 8 s on a 180 m hub at the 30 m ladder floor, i.e. 34.0 m/s against a 30 m/s
    limit (40.1 m/s is the worst bearing on the same hub; this flight's is 34.0). Nothing in the suite caught the whole change being reverted, so pin the physics directly:
    the implied ground speed of the egress must not exceed nominal_speed_mps.
    """
    cfg = SimConfig(flight_levels_m=(30.0, 70.0, 110.0), airspace_ceiling_m=135.0,
                    region_size_m=(20_000.0, 20_000.0), terminal_radius_m=180.0)
    hub = Terminal("hub#0", 8, 180.0)
    req = FlightRequest(1, vec(10_000, 10_000, 0), vec(15_000, 12_000, 0), 0.0, origin_terminal=hub)
    intent = get_planner(planner).plan(req, ReservationLedger(cfg), cfg)
    assert intent.accepted

    p0 = np.asarray(intent.centerline[0][0], float)
    lead_m = float(np.linalg.norm(p0[:2] - np.asarray(req.origin, float)[:2]))
    t0 = intent.centerline[0][1]                       # when the reserved corridor begins
    # guard the guard: a hub wide enough that the lead does NOT fit inside the floor's climb, else
    # the climb alone would already cover it and this proves nothing
    assert lead_m / cfg.nominal_speed_mps > cfg.climb_time_to(30.0), "lead fits in the climb"
    assert t0 > 0.0
    assert lead_m / t0 <= cfg.nominal_speed_mps + 1e-9, (
        f"{planner}: egress implies {lead_m / t0:.1f} m/s against a "
        f"{cfg.nominal_speed_mps:.0f} m/s limit")
    # ... and the column must still be reserved for the whole of it
    assert intent.volumes[0].t_end >= t0 - 1e-9, "corridor starts after the column is released"


def test_column_window_covers_the_actual_traverse():
    """The reserved column must outlast the egress the drone physically flies (issue #52).

    Asserting only ``window == max(steps)*dt`` would be a TAUTOLOGY — that is what the implementation
    says. It has to be pinned against the PHYSICAL traverse instead: with a tautological assertion,
    changing ``math.ceil`` to ``int`` in ``Lane.steps`` yields an 8.000 s window against a 10.583 s
    traverse (2.583 s of unreserved occupancy, worse than the 1.417 s defect this was written for)
    and still passes.
    """
    cfg = SimConfig(terminal_radius_m=180.0)
    hub = Terminal("hub#0", 8, 180.0)
    centre = vec(0, 0, 0)
    lanes = hg.terminal_lanes(centre, hub, cfg)
    assert lanes, "no lanes — the check would be vacuous"
    window = hg.max_lane_traverse_s(centre, hub, cfg)
    worst = max(ln.dist for ln in lanes)
    assert worst / cfg.nominal_speed_mps > 0.0, "no traverse — vacuous"
    # the independent claim: the window covers every lane's real flight time, at cruise speed
    for ln in lanes:
        assert window >= ln.dist / cfg.nominal_speed_mps - 1e-9, f"lane {ln.cell} outruns the window"
    # ... and matches the clock A* actually imposes, so gate and commit cannot drift
    assert window == max(ln.steps for ln in lanes) * cfg.dt_s


@pytest.mark.parametrize("strategy", ["single_knot", "single_knot_heading", "batched_turns"])
def test_refiner_commits_the_same_terminal_column_as_the_planner_it_refines(strategy):
    """``astar_shortcut`` must book the identical origin/dest column as bare ``astar`` (issue #52).

    Two independent regressions hid here, both invisible to every other test:
      * ``ShortcutRefiner`` recovered ``t_depart`` by subtracting only the climb from centerline[0],
        but #52 put ``Lane.steps*dt`` in there too — so the whole rebuilt reservation started 15 s
        LATE, leaving the origin column unreserved while the drone was still inside it;
      * ``build_reservation_from_corners`` sized both columns ``hover + climb`` with no egress, so the
        rebuilt column was 12 s SHORTER than the one the flight had been gated against.
    Together the refined flight's column was [15, 50] where the planner's was [0, 47].
    """
    cfg = SimConfig(terminal_radius_m=180.0)
    hub = Terminal("hub#0", 8, 180.0)
    req = FlightRequest(1, vec(500, 500, 0), vec(4300, 3100, 0), 0.0,
                        origin_terminal=hub, dest_terminal=hub)
    bare = AStarPlanner().plan(req, ReservationLedger(cfg), cfg)
    refined = ShortcutRefiner(AStarPlanner(), strategy=strategy).plan(req, ReservationLedger(cfg), cfg)
    assert bare.accepted and refined.accepted
    assert refined.planner != bare.planner, "refiner returned the inner intent — vacuous"

    egress = hg.max_lane_traverse_s(req.origin, hub, cfg)
    assert egress > 0.0, "no egress traverse — the check would be vacuous"

    # Departure must be identical: the refiner re-times the same takeoff, it does not move it.
    assert refined.volumes[0].t_start == bare.volumes[0].t_start, (
        f"origin column starts {refined.volumes[0].t_start - bare.volumes[0].t_start:+.1f}s off — "
        "t_depart recovery lost the egress traverse")
    # ... and the CORRIDOR too: the refiner re-times splices, never the takeoff. The rebuild's own
    # clock is continuous (climb_time_to + WORST lane) while A* stamps quantised (climb_steps*dt +
    # CHOSEN lane's steps), so re-deriving the start shifted every rebuilt volume by -3..+1 s,
    # lane-dependent; corridor_t0 anchors the rebuild at the stamp the inner planner verified.
    assert refined.centerline[0][1] == bare.centerline[0][1], (
        f"refined corridor starts {refined.centerline[0][1] - bare.centerline[0][1]:+.1f}s off the "
        "stamp the inner planner verified against the ledger")
    # Both column DURATIONS must match. Arrival legitimately moves earlier (a shorter path lands
    # sooner), so only the window length is comparable at the destination.
    for name, idx in (("origin", 0), ("dest", -1)):
        b, r = bare.volumes[idx], refined.volumes[idx]
        assert math.isclose(r.t_end - r.t_start, b.t_end - b.t_start, abs_tol=1e-9), (
            f"{name} column is {(r.t_end - r.t_start) - (b.t_end - b.t_start):+.1f}s shorter — "
            "the rebuild dropped the egress tail")
    assert refined.volumes[-1].t_start <= bare.volumes[-1].t_start + 1e-9, "refined path arrives later?"


def test_capacity_gate_probes_the_full_column_window_not_the_climb():
    """The pad-capacity gate must probe hover + climb + egress traverse — the window the commit books.

    The binding case is a prober sitting just BEFORE an already-committed dwell: FCFS-ordered probes
    never expose a short gate window, because the RECORDED dwell interval carries the commit-side
    tail regardless. Here A is committed with a delayed takeoff and B probes from t=0 underneath it:
    with the gate reverted to a climb-only window, B is admitted at t=0 and its committed column
    overlaps A's by 7.00 s at pad capacity 1 (measured) — the oversubscription the gate exists to
    prevent.
    """
    cfg = SimConfig(flight_levels_m=(30.0, 70.0, 110.0), airspace_ceiling_m=135.0,
                    region_size_m=(20_000.0, 20_000.0), terminal_radius_m=180.0)
    hub = Terminal("hub#0", 1, 180.0)                  # ONE pad: dwells must never overlap
    led = ReservationLedger(cfg)
    a = AStarPlanner().plan(FlightRequest(1, vec(500, 500, 0), vec(4300, 3100, 0), 0.0,
                                          t_departure=40.0, origin_terminal=hub), led, cfg)
    assert a.accepted
    led.commit(1, a.volumes)
    b = AStarPlanner().plan(FlightRequest(2, vec(500, 500, 0), vec(500, 4500, 0), 0.0,
                                          origin_terminal=hub), led, cfg)
    assert b.accepted
    ca, cb = a.volumes[0], b.volumes[0]
    # Guard the guard: A's dwell must start INSIDE B's climb→climb+traverse reach from t=0, i.e. a
    # climb-only probe window would clear it while the full window conflicts — else nothing binds
    # and the assertion below passes for free.
    short_w = cfg.hover_time_s + cfg.climb_time_to(cfg.flight_levels_m[0])
    full_w = short_w + hg.max_lane_traverse_s(np.asarray(a.request.origin, float), hub, cfg)
    assert short_w < ca.t_start < full_w, "A's dwell no longer sits in the gate-sensitive band"
    overlap = min(ca.t_end, cb.t_end) - max(ca.t_start, cb.t_start)
    assert overlap <= 1e-9, (
        f"same-hub columns overlap {overlap:.2f}s at pad capacity 1 — the capacity gate probed a "
        "shorter window than the commit books")


def test_read_envelope_covers_the_landing_dwell_traverse():
    """Track A: a commit inside the LAST traverse seconds of a landing-dwell read must read as DIRTY.

    A dwell/capacity probe at the plan's final step reads ``hover + climb + lane traverse`` past it,
    but ``hover_tail_steps`` covers hover + max climb + buffer only — at 350 m radius the traverse
    (20 s) outruns the buffer by 4.33 s, and pre-fix a commit in that sliver reported
    ``envelope_intersects == False``: exact-mode revalidation would silently miss a real conflict.

    Load-bearing anchor: for a fixed request the radius enters ``t_hi`` through exactly TWO terms —
    ``search_horizon``'s takeoff term now carries the worst origin-lane steps (one traverse), and
    ``_mk_envelope`` adds the worst-end traverse again for the dwell read past the last step — so
    ``t_hi(350) - t_hi(90) == 2 * (traverse(350) - traverse(90))``. Reverting EITHER widening drops
    the difference to one traverse (or zero for both) and fails this. Anchoring a sliver on
    ``env.t_hi`` alone slides WITH the fix and passes with it reverted — measured, the first
    version of this test did exactly that.
    """
    from freespace_sim.parallel import envelope_intersects
    from freespace_sim.planner.astar.compiled_hex_occupancy import hover_tail_steps

    def env_at(radius):
        cfg = SimConfig(flight_levels_m=(30.0, 70.0, 110.0), airspace_ceiling_m=135.0,
                        region_size_m=(20_000.0, 20_000.0), terminal_radius_m=radius)
        hub = Terminal("hub#0", 8, radius)
        req = FlightRequest(1, vec(500, 500, 0), vec(4300, 3100, 0), 0.0,
                            origin_terminal=hub, dest_terminal=hub)
        p = AStarPlanner()
        p.record_envelope = True
        intent = p.plan(req, ReservationLedger(cfg), cfg)
        assert intent.accepted and p.last_envelope is not None
        return p.last_envelope, hg.max_lane_traverse_s(np.asarray(req.dest, float), hub, cfg), cfg, req

    env90, trav90, _, _ = env_at(90.0)
    env350, trav350, cfg, req = env_at(350.0)
    assert trav350 > trav90, "traverse no longer grows with radius — the anchor premise is gone"
    # Guard the guard: the hover tail alone must NOT cover a last-step dwell read at 350 m, or the
    # widening is unnecessary at this config and the test proves nothing.
    max_climb = max(cfg.climb_time_to(z) for z in cfg.flight_levels_m)
    assert hover_tail_steps(cfg) * cfg.dt_s < cfg.hover_time_s + max_climb + trav350, \
        "hover tail covers the traverse at this config — the test lost its bite"
    # The widening itself, pinned against the fix-independent r=90 run: one traverse from the
    # search-horizon lane term + one from the envelope's dwell-read term.
    assert math.isclose(env350.t_hi - env90.t_hi, 2.0 * (trav350 - trav90), abs_tol=1e-9), (
        f"t_hi grew by {env350.t_hi - env90.t_hi:.1f}s, expected {2.0 * (trav350 - trav90):.1f}s — "
        "either the horizon lane term or the envelope traverse term is missing")
    # ... and the semantics: a commit 1 s past the unwidened bound, at the dest hub, reads DIRTY.
    d = np.asarray(req.dest, float)
    aabb = (d[0] - 1.0, d[1] - 1.0, 0.0, d[0] + 1.0, d[1] + 1.0, 200.0)
    sliver = (aabb, env350.t_hi - trav350 + 1.0, env350.t_hi - trav350 + 5.0)
    assert envelope_intersects(env350, [sliver]), (
        "a commit inside the landing dwell's traverse tail is invisible to the read envelope")
