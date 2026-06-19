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


@pytest.mark.slow
def test_milp_shortcut_never_worsens_the_milp_solution():
    base = get_planner("astar_milp").plan(_req(), _wall_led(), CFG)
    led = _wall_led()
    sc = get_planner("astar_milp_shortcut").plan(_req(), led, CFG)
    assert sc.accepted
    assert not led.any_conflict(sc.volumes)
    assert sc.cost <= base.cost + 1e-6                 # post-MILP shortcut is monotone
