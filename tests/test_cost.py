import dataclasses

import pytest

from freespace_sim.config import SimConfig
from freespace_sim.cost import endpoint_altitude_change_m, trajectory_cost
from freespace_sim.types import FlightRequest, IntentStatus, OperationalIntent, vec

CFG = SimConfig()


def test_endpoint_altitude_change_m_books_both_endpoints_and_interior():
    # climb ground→z0 + interior climb/descent dz + descent z1→ground — the shared altitude booking
    assert endpoint_altitude_change_m(30.0, 70.0, 10.0, CFG) == (30 - 0) + (70 - 0) + 10   # 110
    assert endpoint_altitude_change_m(75.0, 75.0, 0.0, CFG) == 2.0 * 75.0                  # single plane


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


def _per_step_costs(cfg):
    """What ONE timestep of each lever actually costs the search — the only comparable basis, since
    every A* edge advances the clock by a whole number of dt steps."""
    return {
        "ground": cfg.cost_ground_delay_per_s * cfg.dt_s,
        # one lateral hex == one timestep at cruise: the lattice pitch IS nominal_speed * dt
        "lateral": cfg.cost_air_lateral_per_m * cfg.nominal_speed_mps * cfg.dt_s,
        "hover": cfg.cost_air_hold_per_s * cfg.dt_s,
        # one climb step covers climb_rate * dt metres of altitude
        "climb": cfg.cost_altitude_change_per_m * cfg.climb_rate_mps * cfg.dt_s,
    }


def test_levers_are_priced_1_3_3_4_per_step():
    """The cost model's contract: per timestep, waiting on the pad is 1x, flying and hovering are 3x,
    and climbing is 4x. Before the weights were normalized to a per-second currency, lateral and
    altitude were stored PER METRE and so got silently multiplied by pitch (120 m) and climb_rate*dt
    (24 m) while ground/hover were multiplied by dt (4 s) — making the real ratios 1:90:3:24. One hex
    of detour cost as much as 360 s of ground delay, so no detour or climb was ever rational."""
    c = _per_step_costs(CFG)
    assert (c["lateral"], c["hover"], c["climb"]) == pytest.approx(
        (3 * c["ground"], 3 * c["ground"], 4 * c["ground"]))


@pytest.mark.parametrize("kw", [
    {"dt_s": 1.0}, {"dt_s": 7.0},
    {"nominal_speed_mps": 12.0}, {"nominal_speed_mps": 45.0},
    {"climb_rate_mps": 2.5}, {"climb_rate_mps": 9.0},
    {"dt_s": 6.0, "nominal_speed_mps": 12.0, "climb_rate_mps": 9.0},
])
def test_lever_ratios_survive_retiming_and_respeeding(kw):
    """The ratio must be a property of the CONFIG, not a coincidence of three unrelated numbers.
    Deriving the per-metre weights from per-second ones is what buys this: change the timestep, the
    cruise speed or the climb rate and 1:3:3:4 still holds. Pinning per-metre values instead would
    re-introduce the original bug the moment any of these three moved."""
    c = _per_step_costs(SimConfig(**kw))
    assert (c["lateral"] / c["ground"], c["hover"] / c["ground"],
            c["climb"] / c["ground"]) == pytest.approx((3.0, 3.0, 4.0))


def test_per_metre_weights_are_derived_not_stored():
    """They must not be constructor args — a stored per-metre weight is exactly the thing that let
    the currencies drift apart, and asdict() must not round-trip one back into a config."""
    assert {f.name for f in dataclasses.fields(SimConfig)}.isdisjoint(
        {"cost_air_lateral_per_m", "cost_altitude_change_per_m"})
    with pytest.raises(TypeError):
        SimConfig(cost_air_lateral_per_m=3.0)
    cfg = SimConfig()
    assert cfg.cost_air_lateral_per_m == cfg.cost_air_lateral_per_s / cfg.nominal_speed_mps
    assert cfg.cost_altitude_change_per_m == cfg.cost_altitude_change_per_s / cfg.climb_rate_mps
