"""SimConfig flight-level ladder: derivation helpers and __post_init__ validation."""

import pytest

from freespace_sim.config import SimConfig


def test_default_config_is_multilevel():
    c = SimConfig()
    assert c.flight_levels_m == (30.0, 70.0, 110.0)
    assert c.n_levels == 3
    assert c.airspace_ceiling_m == 125.0
    # cruise + band DERIVE from the ladder: cruise = middle level; MILP band = floor→top
    assert c.cruise_level_m == 70.0
    assert c.z_min_m == 30.0 and c.z_max_m == 110.0


def test_equidistant_levels_builds_ladder():
    assert SimConfig.equidistant_levels(30.0, 110.0, 3) == (30.0, 70.0, 110.0)


def test_equidistant_levels_n1_is_single():
    assert SimConfig.equidistant_levels(30.0, 110.0, 1) == (30.0,)


def test_level_z_and_nearest_level():
    c = SimConfig()
    assert c.level_z(1) == 70.0
    assert c.nearest_level(72.0) == 1
    assert c.nearest_level(10.0) == 0
    assert c.nearest_level(200.0) == 2


def test_climb_time_to_and_steps():
    c = SimConfig()                               # climb_rate 6 m/s, dt 4 s
    assert c.climb_time_to(70.0) == 70.0 / 6.0
    assert c.climb_steps_to(70.0) == 3            # ceil((70/6)/4) = ceil(2.92)
    assert c.climb_steps_to(30.0) == 2            # ceil((30/6)/4) = ceil(1.25)


def test_validation_rejects_unsorted_levels():
    with pytest.raises(ValueError):
        SimConfig(flight_levels_m=(110.0, 70.0, 30.0))


def test_validation_rejects_levels_too_close():
    with pytest.raises(ValueError, match="corridor_height"):
        SimConfig(flight_levels_m=(25.0, 55.0))                        # gap 30 == corridor_height


def test_validation_rejects_top_above_ceiling():
    with pytest.raises(ValueError):
        SimConfig(flight_levels_m=(30.0, 70.0, 130.0))                 # 130 + 15 > 125


def test_validation_rejects_lowest_below_ground():
    with pytest.raises(ValueError):
        SimConfig(flight_levels_m=(10.0, 70.0, 110.0))                 # 10 - 15 < 0


def test_cruise_and_band_derive_from_ladder():
    c = SimConfig(flight_levels_m=(30.0, 70.0, 110.0))   # derived, not settable
    assert c.cruise_level_m == 70.0                      # cruise = middle level (straight/decoupled)
    assert c.z_min_m == 30.0 and c.z_max_m == 110.0      # MILP band = ladder floor→top


def test_altitudes_derive_from_single_level_ladder():
    c = SimConfig(flight_levels_m=(100.0,))          # a scenario pinned to one plane at 100 m
    assert c.cruise_level_m == c.z_min_m == c.z_max_m == 100.0   # single level ⇒ cruise + band collapse


def test_single_level_config_supported():
    c = SimConfig(flight_levels_m=(75.0,))                             # ceiling stays 125
    assert c.n_levels == 1
    assert c.flight_levels_m == (75.0,)
    assert c.cruise_level_m == c.z_min_m == c.z_max_m == 75.0          # all derive onto the lone level


def test_demand_duration_defaults_to_horizon():
    c = SimConfig(horizon_s=900.0)
    assert c.demand_duration_s is None
    assert c.effective_demand_duration_s == 900.0


def test_demand_duration_must_be_positive():
    with pytest.raises(ValueError, match="demand_duration_s must be positive"):
        SimConfig(horizon_s=900.0, demand_duration_s=0.0)


def test_demand_duration_cannot_exceed_horizon():
    with pytest.raises(ValueError, match="exceeds horizon_s"):
        SimConfig(horizon_s=900.0, demand_duration_s=901.0)
