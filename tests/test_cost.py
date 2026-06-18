from freespace_sim.config import SimConfig
from freespace_sim.cost import trajectory_cost
from freespace_sim.types import FlightRequest, IntentStatus, OperationalIntent, vec

CFG = SimConfig()


def _intent(**kw):
    req = FlightRequest(0, vec(0, 0, 0), vec(1, 0, 0), 0.0)
    return OperationalIntent(req, IntentStatus.ACCEPTED, **kw)


def test_zero_when_no_levers_used():
    assert trajectory_cost(_intent(), CFG) == 0.0


def test_each_lever_contributes_its_weight():
    assert trajectory_cost(_intent(ground_delay_s=10), CFG) == CFG.cost_ground_delay_per_s * 10
    assert trajectory_cost(_intent(air_hold_s=10), CFG) == CFG.cost_air_hold_per_s * 10
    assert trajectory_cost(_intent(air_detour_m=10), CFG) == CFG.cost_air_lateral_per_m * 10
    assert trajectory_cost(_intent(altitude_change_m=10), CFG) == CFG.cost_altitude_change_per_m * 10


def test_air_hold_weighted_above_ground_delay():
    # loitering in the air should cost more than waiting on the pad (battery)
    assert CFG.cost_air_hold_per_s > CFG.cost_ground_delay_per_s
