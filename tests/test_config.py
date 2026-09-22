"""SimConfig flight-level ladder: derivation helpers and __post_init__ validation."""

import pytest

from freespace_sim.config import SimConfig


_D = SimConfig()                    # the shipped defaults; every derived expectation below reads them
_H = _D.corridor_height_m


def test_default_config_is_multilevel():
    # The ONE literal pin of the shipped altitude defaults: changing them is a research decision, so
    # it should fail exactly here. Every other ladder test derives from whatever ladder is configured.
    assert _D.flight_levels_m == (70.0, 85.0, 100.0, 115.0)
    assert _D.n_levels > 1
    assert _D.corridor_height_m == 10.0
    assert _D.airspace_ceiling_m == 120.0


def test_equidistant_levels_builds_ladder():
    assert SimConfig.equidistant_levels(30.0, 110.0, 3) == (30.0, 70.0, 110.0)


def test_equidistant_levels_n1_is_single():
    assert SimConfig.equidistant_levels(30.0, 110.0, 1) == (30.0,)


def test_level_z_and_nearest_level():
    for i, z in enumerate(_D.flight_levels_m):
        assert _D.level_z(i) == z
        assert _D.nearest_level(z + 1.0) == i                                # inside the level's half-gap
    assert _D.nearest_level(_D.ground_level_m) == 0                          # below the ladder → floor
    assert _D.nearest_level(_D.airspace_ceiling_m + 100.0) == _D.n_levels - 1   # above it → top


def test_climb_time_to_and_steps():
    c = SimConfig()                               # climb_rate 6 m/s, dt 4 s
    assert c.climb_time_to(70.0) == 70.0 / 6.0
    assert c.climb_steps_to(70.0) == 3            # ceil((70/6)/4) = ceil(2.92)
    assert c.climb_steps_to(30.0) == 2            # ceil((30/6)/4) = ceil(1.25)


@pytest.mark.parametrize(
    ("bad_ladder", "expected_match"),
    [
        pytest.param(tuple(reversed(_D.flight_levels_m)), None, id="unsorted"),
        # adjacent boxes must not even touch: a gap EQUAL to the box height is rejected
        pytest.param((_D.flight_levels_m[0], _D.flight_levels_m[0] + _H), "corridor_height", id="too_close"),
        pytest.param((_D.airspace_ceiling_m - _H / 2.0 + 1.0,), None, id="above_ceiling"),   # top box pokes out
        pytest.param((_D.ground_level_m + _H / 2.0 - 1.0,), None, id="below_ground"),       # bottom box dips
    ],
)
def test_validation_rejects_bad_ladder(bad_ladder, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        SimConfig(flight_levels_m=bad_ladder)


@pytest.mark.parametrize(
    ("kwargs", "expected_match"),
    [
        # A negative dwell makes `ItineraryPlanner` ask leg 2 to depart BEFORE leg 1 lands, which is
        # the one thing the itinerary model exists to make inexpressible.
        pytest.param({"turnaround_s": -1.0}, "turnaround_s", id="negative_turnaround"),
        pytest.param({"ground_box_height_m": 0.0}, "ground_box_height_m", id="flat_ground_box"),
        # A box taller than the headroom under the lowest level's band (levels[0] - h/2 - ground)
        # would rasterize into the lattice and wall off the airspace over its own pad.
        pytest.param({"ground_box_height_m": _D.flight_levels_m[0] - _H / 2.0 - _D.ground_level_m + 1.0},
                     "ground_box_height_m", id="box_reaches_lattice"),
    ],
)
def test_validation_rejects_bad_round_trip_geometry(kwargs, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        SimConfig(**kwargs)


@pytest.mark.parametrize(
    "ladder",
    [_D.flight_levels_m, (30.0, 70.0, 110.0), (100.0,), (75.0,)],
    ids=["default", "three_level", "single_100m", "single_75m"],
)
def test_cruise_and_band_derive_from_ladder(ladder):
    # derived, never stored: cruise = middle level (straight/decoupled), MILP band = ladder floor→top;
    # a single-level ladder collapses all three onto the lone plane (the ceiling is untouched)
    c = SimConfig(flight_levels_m=ladder)
    assert c.flight_levels_m == ladder and c.n_levels == len(ladder)
    assert c.cruise_level_m == ladder[len(ladder) // 2]
    assert (c.z_min_m, c.z_max_m) == (ladder[0], ladder[-1])
    assert c.airspace_ceiling_m == _D.airspace_ceiling_m


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
