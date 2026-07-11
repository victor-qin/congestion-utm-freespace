import pytest

from freespace_sim.config import SimConfig
from freespace_sim.geometry import box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner.shortcut import ShortcutRefiner, shortcut_corners
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec
from freespace_sim.volumes import Volume4D, build_reservation_from_corners

CFG = SimConfig()


def _req():
    return FlightRequest(1, vec(0, 0, 0), vec(2400, 0, 0), 0.0)


def _wall_led():
    led = ReservationLedger(CFG)
    led.commit(99, [Volume4D(box_from_segment(vec(1200, -250, 150), vec(1200, 250, 150), 40, 400),
                             0.0, 1e6)])
    return led


def test_get_planner_registers_shortcut_variants():
    assert isinstance(get_planner("astar_shortcut"), ShortcutRefiner)
    assert isinstance(get_planner("astar_milp_shortcut"), ShortcutRefiner)


def test_shortcut_empty_airspace_leaves_straight_path_alone():
    a = AStarPlanner().plan(_req(), ReservationLedger(CFG), CFG)
    s = get_planner("astar_shortcut").plan(_req(), ReservationLedger(CFG), CFG)
    assert s.accepted
    assert abs(s.air_detour_m - a.air_detour_m) < 1.0   # already straight; nothing to remove


def test_shortcut_tightens_astar_berth_and_stays_conflict_free():
    a = AStarPlanner().plan(_req(), _wall_led(), CFG)
    led = _wall_led()
    s = get_planner("astar_shortcut").plan(_req(), led, CFG)
    assert s.accepted
    assert not led.any_conflict(s.volumes)            # build-then-check contract holds
    assert s.air_detour_m < a.air_detour_m - 50.0     # genuinely tighter, not a nudge
    assert s.cost <= a.cost + 1e-6                     # a post-pass never worsens


def test_shortcut_corners_collapses_a_zigzag_in_open_space():
    led = ReservationLedger(CFG)
    z = CFG.cruise_level_m
    corners = [vec(0, 0, z), vec(400, 120, z), vec(800, 0, z), vec(1200, 120, z), vec(1600, 0, z)]
    out = shortcut_corners(corners, vec(0, 0, 0), vec(1600, 0, 0), 0.0, 0.0, CFG, led)
    assert len(out) < len(corners)                    # redundant knots removed
    vols, _, _, _ = build_reservation_from_corners(out, vec(0, 0, 0), vec(1600, 0, 0), 0.0, 0.0, CFG)
    assert not led.any_conflict(vols)                 # rebuilt path is conflict-free


def test_shortcut_demand_run_is_verified():
    cfg = SimConfig(planner="astar_shortcut", lam_per_hour=40.0, horizon_s=900.0, seed=4,
                    region_size_m=(4000.0, 4000.0))
    assert run(cfg).verified


def test_astar_shortcut_runs_under_always_active():
    """The payoff + the lifted ban: with the terminal walls now PERMANENT LEDGER VOLUMES, the shortcut
    refiner's ``any_conflict`` recheck respects them, so ``sim.run`` no longer raises for an A*-wrapping
    planner under ``terminal_airspace_always_active`` (it used to be a hard ``ValueError``) — and the result
    stays verified: the refiner does not straighten a corridor through a walled terminal column."""
    from freespace_sim.scenarios import get_scenario, with_overrides
    spec = with_overrides(get_scenario("dallas_hub_2uss_large"), horizon_s=8.0)
    cfg = spec.config()
    assert cfg.terminal_airspace_always_active
    r = run(cfg, demand=spec.demand_model(), planner_name="astar_shortcut")   # must NOT raise
    assert r.verified, "shortcut refiner must respect the ledger walls (verified conflict-free)"


@pytest.mark.slow
def test_milp_shortcut_never_worsens_the_milp_solution():
    base = get_planner("astar_milp").plan(_req(), _wall_led(), CFG)
    led = _wall_led()
    sc = get_planner("astar_milp_shortcut").plan(_req(), led, CFG)
    assert sc.accepted
    assert not led.any_conflict(sc.volumes)
    assert sc.cost <= base.cost + 1e-6                 # post-MILP shortcut is monotone


# --- multi-altitude: the refiner polishes A*'s multi-level output -----------------------------------

def _climb_walls_led():
    """A ledger forcing a mid-route climb: level 1 walled early, level 0 walled late (both mid-route)."""
    led = ReservationLedger(CFG)
    led.commit(98, [Volume4D(box_from_segment(vec(900, -400, CFG.level_z(1)),
                                              vec(900, 400, CFG.level_z(1)), 40, CFG.corridor_height_m),
                             0.0, 1e6)])
    led.commit(97, [Volume4D(box_from_segment(vec(1500, -400, CFG.level_z(0)),
                                              vec(1500, 400, CFG.level_z(0)), 40, CFG.corridor_height_m),
                             0.0, 1e6)])
    return led


def test_astar_shortcut_preserves_multilevel_climb():
    led = _climb_walls_led()
    s = get_planner("astar_shortcut").plan(_req(), led, CFG)
    assert s.accepted
    assert not led.any_conflict(s.volumes)                              # build-then-check holds
    levels = sorted({round(float(p[2]), 1) for p, _ in s.centerline})
    assert CFG.level_z(0) in levels and CFG.level_z(1) in levels        # the climb knot survived


def test_astar_shortcut_slants_the_climb_staircase():
    a = AStarPlanner().plan(_req(), _climb_walls_led(), CFG)
    led = _climb_walls_led()
    s = get_planner("astar_shortcut").plan(_req(), led, CFG)
    assert s.accepted and not led.any_conflict(s.volumes)
    assert s.cost <= a.cost + 1e-6                                      # a post-pass never worsens
    cl = s.centerline
    slanted = any(abs(float(cl[i + 1][0][0]) - float(cl[i][0][0])) > 1.0
                  and abs(float(cl[i + 1][0][2]) - float(cl[i][0][2])) > 1.0
                  for i in range(len(cl) - 1))
    assert slanted   # A*'s orthogonal cruise→climb→cruise staircase fused into a DIAGONAL climb


def _diagonal_segments(centerline):
    """Count centerline segments that move BOTH horizontally and vertically — a slanted diagonal climb
    box (vs A*'s orthogonal pure-vertical rung / pure-horizontal cruise)."""
    return sum((abs(float(b[0] - a[0])) > 1.0 or abs(float(b[1] - a[1])) > 1.0)
               and abs(float(b[2] - a[2])) > 1.0
               for (a, _), (b, _) in zip(centerline, centerline[1:]))


def test_astar_shortcut_dense_multilevel_run_stays_verified():
    # The refiner's diagonal climb-boxes must stay FCL-conflict-free UNDER LOAD — against other traffic at
    # other levels, not just the static walls the single-flight tests use. Dense crossing traffic with
    # capped ground delay makes altitude the deconfliction lever, forcing many climbs the shortcut fuses
    # into diagonals. (test_shortcut_demand_run_is_verified @ λ=40 is far too sparse to force any climb;
    # multilevel_e2e uses plain astar — so nothing else covers this path under load.)
    cfg = SimConfig(planner="astar_shortcut", lam_per_hour=3000.0, horizon_s=300.0,
                    region_size_m=(1200.0, 1200.0), seed=1, max_ground_delay_s=60.0)
    res = run(cfg)
    assert res.verified                                    # FCL replay: no phantom cross-level collision
    # not vacuous: the run actually produced climbs the refiner slanted into diagonal segments
    floor = 2.0 * cfg.flight_levels_m[0]
    assert sum(i.altitude_change_m > floor + 1.0 for i in res.accepted) >= 5    # many climbed the ladder
    assert sum(_diagonal_segments(i.centerline) for i in res.accepted) >= 5     # ... and were slanted


def test_astar_shortcut_diagonal_climb_deconflicts_from_committed_traffic():
    # Two opposite-direction flights on a shared corridor: the first is committed, the second must climb
    # OVER it — and the refiner slants that climb into a DIAGONAL box which must stay clear of the first's
    # committed corridor (build-then-check + FCL). The diagonal climb-box vs real traffic, deterministic.
    led = ReservationLedger(CFG)
    i1 = get_planner("astar_shortcut").plan(FlightRequest(1, vec(0, 0, 0), vec(6000, 0, 0), 0.0), led, CFG)
    assert i1.accepted
    led.commit(1, i1.volumes)
    i2 = get_planner("astar_shortcut").plan(FlightRequest(2, vec(6000, 0, 0), vec(0, 0, 0), 0.0), led, CFG)
    assert i2.accepted
    assert not led.any_conflict(i2.volumes)                # the (slanted) climb clears the committed flight
    assert max(round(float(p[2]), 1) for p, _ in i2.centerline) >= CFG.level_z(1)   # it climbed a level
    assert _diagonal_segments(i2.centerline) >= 1          # the climb was slanted, not a pure rung
