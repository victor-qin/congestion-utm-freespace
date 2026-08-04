"""ScenarioSpec — the world recipe that the execute step turns into a run."""

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand, HubVoronoiDemand, UniformPoissonDemand
from freespace_sim.geo import project_lonlat_to_enu
from freespace_sim.scenarios import (
    SCENARIOS,
    DemandSpec,
    ScenarioSpec,
    get_scenario,
    with_overrides,
)
from freespace_sim.scenarios.demand_dfw import (
    DEFAULT_FIXED_TYPES,
    DEFAULT_HUB_CATEGORIES,
    DfwGeoDemand,
)
from freespace_sim.scenarios.density import AMAZON_USS, WING_ZIPLINE_USS
from freespace_sim.scenarios.dfw import DFW_FRAME, DFW_REGION_CENTER_LATLON, DFW_REGION_M


def test_get_scenario_resolves_and_rejects():
    assert get_scenario("metro_2uss") is SCENARIOS["metro_2uss"]
    with pytest.raises(ValueError):
        get_scenario("no_such_scenario")


def test_config_applies_overrides_over_defaults():
    spec = ScenarioSpec("x", region_m=(5000.0, 4000.0), horizon_s=900.0, lam_per_hour=240.0,
                        demand_duration_s=600.0, seed=3, planner="astar_shortcut")
    cfg = spec.config()
    assert cfg.region_size_m == (5000.0, 4000.0)
    assert cfg.horizon_s == 900.0 and cfg.lam_per_hour == 240.0 and cfg.seed == 3
    assert cfg.demand_duration_s == 600.0
    assert cfg.planner == "astar_shortcut"


def test_planner_none_keeps_simconfig_default():
    assert ScenarioSpec("x").config().planner == SimConfig().planner


def test_demand_uniform_single_uss_is_none():
    # bare uniform demand → None → simulator uses its built-in single-"default"-USS demand
    assert DemandSpec().build() is None
    assert ScenarioSpec("x").demand_model() is None


def test_demand_uniform_multi_uss():
    d = DemandSpec(pattern="uniform", uss=("a", "b")).build()
    assert isinstance(d, UniformPoissonDemand)
    assert d.uss_ids == ("a", "b")


def test_demand_hub_builds_hubvoronoi_with_counts():
    d = DemandSpec(pattern="hub", uss=("walmart_uss", "stripmall_uss"), hubs=(4, 12)).build()
    assert isinstance(d, HubVoronoiDemand)
    assert d.n_hubs_per_uss == {"walmart_uss": 4, "stripmall_uss": 12}


def test_demand_hub_defaults_when_counts_omitted():
    d = DemandSpec(pattern="hub").build()
    assert isinstance(d, HubVoronoiDemand)
    assert d.n_hubs_per_uss == {"walmart_uss": 6, "stripmall_uss": 20}


def test_demand_hub_mismatched_counts_raises():
    with pytest.raises(ValueError):
        DemandSpec(pattern="hub", uss=("a", "b", "c"), hubs=(1, 2)).build()


def test_unknown_pattern_raises():
    with pytest.raises(ValueError):
        DemandSpec(pattern="poisson_clustered").build()


def test_with_overrides_replaces_top_and_demand_fields():
    base = SCENARIOS["dallas_hub_2uss"]
    spec = with_overrides(base, lam_per_hour=1200.0, seed=2, demand_overrides={"hubs": (3, 9)})
    assert spec.lam_per_hour == 1200.0 and spec.seed == 2
    assert spec.demand.hubs == (3, 9)
    assert base.lam_per_hour == 600.0 and base.demand.hubs == (6, 20)   # original untouched (frozen)


def test_every_registry_scenario_builds_valid_world():
    for name, spec in SCENARIOS.items():
        cfg = spec.config()
        assert cfg.region_size_m[0] > 0 and cfg.horizon_s > 0
        dm = spec.demand_model()
        # builds without error and is either the default (None) or a real model with a generate()
        assert dm is None or hasattr(dm, "generate")


def test_registry_scenarios_generate_requests_with_expected_uss():
    spec = SCENARIOS["dallas_hub_2uss"]
    reqs = spec.demand_model().generate(spec.config(), np.random.default_rng(0))
    assert {r.uss_id for r in reqs} == {"walmart_uss", "stripmall_uss"}


def test_demand_hub_radius_builds_with_params():
    d = DemandSpec(pattern="hub_radius", uss=("a", "b"), hubs=(3, 7),
                   radius_m=2500.0, pads_per_hub=4, return_flights=False, turnaround_s=90.0).build()
    assert isinstance(d, HubRadiusDemand)
    assert d.n_hubs_per_uss == {"a": 3, "b": 7}
    assert d.radius_m == 2500.0 and d.pads_per_hub == 4
    assert d.return_flights is False and d.turnaround_s == 90.0


def test_demand_hub_radius_plumbs_lam_per_uss_and_departure_offset():
    # The per-USS and clock knobs must survive DemandSpec.build() onto HubRadiusDemand.
    d = DemandSpec(pattern="hub_radius", uss=("a", "b"), hubs=(3, 5),
                   lam_per_uss={"a": 1000.0, "b": 250.0},
                   departure_offset_s={"a": (450.0, 60.0)},
                   timing_mode="departure", paired_return_request=True).build()
    assert isinstance(d, HubRadiusDemand)
    assert d.lam_per_uss == {"a": 1000.0, "b": 250.0}
    assert d.departure_offset_s == {"a": (450.0, 60.0)}
    assert d.timing_mode == "departure"
    assert d.paired_return_request is True


def test_dallas_large_scenario_uses_radius_pads_returns():
    spec = SCENARIOS["dallas_hub_2uss_large"]
    assert spec.region_m == (10000.0, 10000.0)
    d = spec.demand_model()
    assert isinstance(d, HubRadiusDemand)
    assert d.n_hubs_per_uss == {"walmart_uss": 6, "stripmall_uss": 20}
    assert d.pads_per_hub == {"walmart_uss": 40, "stripmall_uss": 16} and d.return_flights is True
    # generates a two-USS round-trip demand (run at a small λ/horizon so the test stays fast)
    small = with_overrides(spec, lam_per_hour=120.0, horizon_s=300.0)
    reqs = small.demand_model().generate(small.config(), np.random.default_rng(0))
    assert {r.uss_id for r in reqs} == {"walmart_uss", "stripmall_uss"}


@pytest.mark.parametrize(
    ("name", "hubs", "rates"),
    [
        (
            "density_faa_wing_zipline",
            {WING_ZIPLINE_USS: 182},
            {WING_ZIPLINE_USS: 4853.94},
        ),
        (
            "density_future_wing_zipline",
            {WING_ZIPLINE_USS: 476},
            {WING_ZIPLINE_USS: 27322.4},
        ),
        (
            "density_faa_wing_zipline_amazon",
            {WING_ZIPLINE_USS: 182, AMAZON_USS: 7},
            {WING_ZIPLINE_USS: 4853.94, AMAZON_USS: 466.69},
        ),
        (
            "density_future_wing_zipline_amazon",
            {WING_ZIPLINE_USS: 476, AMAZON_USS: 14},
            {WING_ZIPLINE_USS: 27322.4, AMAZON_USS: 2198.0},
        ),
    ],
)
def test_density_scenario_matrix(name, hubs, rates):
    spec = SCENARIOS[name]
    cfg = spec.config()
    demand = spec.demand_model()

    assert cfg.region_size_m == (60_000.0, 30_000.0)
    assert cfg.horizon_s == 7200.0
    assert cfg.demand_duration_s == 1800.0
    assert cfg.flight_levels_m == (100.0,)
    assert cfg.fixed_exit_lanes is True
    assert cfg.terminal_airspace_always_active is True
    assert cfg.lam_per_hour == round(sum(rates.values()), 2)

    assert isinstance(demand, HubRadiusDemand)
    assert demand.n_hubs_per_uss == hubs
    assert demand.lam_per_uss == rates
    assert demand.pads_per_hub == dict.fromkeys(hubs, 40)
    assert demand.terminal_radius_m == dict.fromkeys(hubs, 180.0)
    assert demand.radius_m[WING_ZIPLINE_USS] == 16_000.0
    if AMAZON_USS in hubs:
        assert demand.radius_m[AMAZON_USS] == 12_000.0
    assert demand.departure_offset_s[WING_ZIPLINE_USS] == (480.0, 90.0)
    if AMAZON_USS in hubs:
        assert demand.departure_offset_s[AMAZON_USS] == (1800.0, 300.0)
    assert demand.return_flights is True
    assert demand.turnaround_s == 0.0
    assert demand.timing_mode == "departure"
    assert demand.paired_return_request is True


@pytest.mark.parametrize(
    ("stacked_name", "twin_name"),
    [
        ("density_faa_wing_zipline_3lvl", "density_faa_wing_zipline"),
        ("density_future_wing_zipline_3lvl", "density_future_wing_zipline"),
        ("density_faa_wing_zipline_amazon_3lvl", "density_faa_wing_zipline_amazon"),
        ("density_future_wing_zipline_amazon_3lvl", "density_future_wing_zipline_amazon"),
    ],
)
def test_density_stacked_scenarios_are_three_levels(stacked_name, twin_name):
    # config() constructing at all proves the (80, 95, 110) ladder clears SimConfig.__post_init__ — the
    # 15 m gaps only validate because the scenario ships a 14 m corridor box (a 30 m box would raise).
    cfg = SCENARIOS[stacked_name].config()
    assert cfg.flight_levels_m == (80.0, 95.0, 110.0)
    assert cfg.corridor_height_m == 14.0
    assert cfg.n_levels == 3
    assert (cfg.z_min_m, cfg.cruise_level_m, cfg.z_max_m) == (80.0, 95.0, 110.0)

    # the stacked variant changes ONLY altitude — its demand world is identical to the single-level twin,
    # which itself is untouched (still the 100 m plane on SimConfig's default 30 m box).
    stacked, twin = SCENARIOS[stacked_name], SCENARIOS[twin_name]
    assert stacked.lam_per_hour == twin.lam_per_hour
    assert stacked.demand == twin.demand
    twin_cfg = twin.config()
    assert twin_cfg.flight_levels_m == (100.0,)
    assert twin_cfg.corridor_height_m == 30.0


def test_density_mixed_scenarios_use_two_distinct_uss():
    for name in (
        "density_faa_wing_zipline_amazon",
        "density_future_wing_zipline_amazon",
    ):
        assert SCENARIOS[name].demand.uss == (WING_ZIPLINE_USS, AMAZON_USS)


def test_density_single_scenarios_use_only_wing_zipline_uss():
    for name in ("density_faa_wing_zipline", "density_future_wing_zipline"):
        assert SCENARIOS[name].demand.uss == (WING_ZIPLINE_USS,)


@pytest.mark.parametrize(
    ("single_name", "mixed_name"),
    [
        ("density_faa_wing_zipline", "density_faa_wing_zipline_amazon"),
        ("density_future_wing_zipline", "density_future_wing_zipline_amazon"),
    ],
)
def test_density_wing_hub_layout_is_preserved_when_amazon_is_added(single_name, mixed_name):
    single = SCENARIOS[single_name]
    mixed = SCENARIOS[mixed_name]
    single_hubs = single.demand_model().place_hubs(
        single.config(), np.random.default_rng(single.demand_model().hub_seed)
    )
    mixed_model = mixed.demand_model()
    mixed_hubs = mixed_model.place_hubs(
        mixed.config(), np.random.default_rng(mixed_model.hub_seed)
    )
    assert np.array_equal(single_hubs[WING_ZIPLINE_USS], mixed_hubs[WING_ZIPLINE_USS])


def test_density_future_mixed_490_hub_layout_is_feasible():
    spec = SCENARIOS["density_future_wing_zipline_amazon"]
    model = spec.demand_model()
    hubs = model.place_hubs(spec.config(), np.random.default_rng(model.hub_seed))
    assert sum(len(points) for points in hubs.values()) == 490


def test_density_scenarios_have_descriptions_and_remove_density_test():
    density_names = {name for name in SCENARIOS if name.startswith("density_")}
    assert density_names == {
        "density_faa_wing_zipline",
        "density_future_wing_zipline",
        "density_faa_wing_zipline_amazon",
        "density_future_wing_zipline_amazon",
        "density_faa_wing_zipline_3lvl",
        "density_future_wing_zipline_3lvl",
        "density_faa_wing_zipline_amazon_3lvl",
        "density_future_wing_zipline_amazon_3lvl",
    }
    assert all(SCENARIOS[name].description for name in density_names)
    assert "density_test" not in SCENARIOS


# --- DFW real-geography twins: same numbers as density_*, only the spatial generation differs ---

_DFW_TWINS = [
    ("dfw_faa_wing_zipline", "density_faa_wing_zipline"),
    ("dfw_future_wing_zipline", "density_future_wing_zipline"),
    ("dfw_faa_wing_zipline_amazon", "density_faa_wing_zipline_amazon"),
    ("dfw_future_wing_zipline_amazon", "density_future_wing_zipline_amazon"),
    ("dfw_faa_wing_zipline_3lvl", "density_faa_wing_zipline_3lvl"),
    ("dfw_future_wing_zipline_3lvl", "density_future_wing_zipline_3lvl"),
    ("dfw_faa_wing_zipline_amazon_3lvl", "density_faa_wing_zipline_amazon_3lvl"),
    ("dfw_future_wing_zipline_amazon_3lvl", "density_future_wing_zipline_amazon_3lvl"),
]


def test_dfw_registry_set_and_descriptions():
    assert {n for n in SCENARIOS if n.startswith("dfw_")} == {dfw for dfw, _ in _DFW_TWINS}
    assert all(SCENARIOS[dfw].description for dfw, _ in _DFW_TWINS)


@pytest.mark.parametrize(("dfw_name", "den_name"), _DFW_TWINS)
def test_dfw_twin_reuses_all_density_numbers(dfw_name, den_name):
    """The whole point of the family: a dfw_* world keeps every DEMAND number of its density_* parent
    (hub counts, hub sizes, per-USS demand, timing, levels) and changes only how hubs and destinations
    are placed (pattern hub_radius → dfw_geo) — plus the region, which is WIDENED to the full metroplex
    frame so the real metro-wide geography fits. Guards the "reuse density numbers" contract."""
    dfw_spec, den_spec = SCENARIOS[dfw_name], SCENARIOS[den_name]
    for field in ("horizon_s", "demand_duration_s", "lam_per_hour", "flight_levels_m",
                  "corridor_height_m", "fixed_exit_lanes", "terminal_airspace_always_active"):
        assert getattr(dfw_spec, field) == getattr(den_spec, field), field
    # region is intentionally the full ~192×147 km frame, NOT the density 60×30 km window
    assert dfw_spec.region_m == DFW_REGION_M and dfw_spec.region_m != den_spec.region_m
    assert dfw_spec.region_center_latlon == DFW_REGION_CENTER_LATLON
    a, b = dfw_spec.demand, den_spec.demand
    for field in ("uss", "hubs", "radius_m", "pads_per_hub", "terminal_radius_m", "corridor_overlap_m",
                  "return_flights", "turnaround_s", "uss_share", "lam_per_uss", "departure_offset_s",
                  "timing_mode", "paired_return_request", "min_hub_gap_m"):
        assert getattr(a, field) == getattr(b, field), field
    # ONLY the spatial generation differs
    assert b.pattern == "hub_radius" and a.pattern == "dfw_geo"
    assert isinstance(dfw_spec.demand_model(), DfwGeoDemand)
    assert a.sampled_hub_uss == (WING_ZIPLINE_USS,)
    assert a.fixed_hub_uss == ((AMAZON_USS,) if AMAZON_USS in a.uss else ())


def test_dfw_frame_corners_map_onto_the_region_box():
    """The region box IS the lon/lat frame: geo.project_lonlat_to_enu must land the frame's corners
    exactly on [0, w] x [0, h]. If the frame and the box size ever drift apart, every hub and tract
    silently shifts (or gets clipped) with no other symptom. Also pins region_m to plain floats —
    numpy scalars leak into scenario_spec.json and make json.dumps raise."""
    minlon, maxlon, minlat, maxlat = DFW_FRAME
    w, h = DFW_REGION_M
    lat0, lon0 = DFW_REGION_CENTER_LATLON
    corners = project_lonlat_to_enu(
        np.array([minlon, maxlon]), np.array([minlat, maxlat]), lat0, lon0, w, h)
    assert np.allclose(corners, [[0.0, 0.0], [w, h]], atol=1e-6)
    assert all(type(v) is float for v in DFW_REGION_M)


def test_dfw_spec_pins_its_own_hub_siting_rules():
    """An archived scenario_spec.json must fully describe the world it ran. Leaving these () and
    letting DemandSpec.build() fill them from DfwGeoDemand's defaults means editing
    DEFAULT_HUB_CATEGORIES silently replays every archived dfw_* run against a different hub pool —
    exactly the silent reinterpretation the schema_version guard exists to prevent."""
    payload = SCENARIOS["dfw_future_wing_zipline_amazon"].to_json_dict()["demand"]
    assert tuple(payload["hub_categories"]) == DEFAULT_HUB_CATEGORIES
    assert tuple(payload["fixed_hub_types"]) == DEFAULT_FIXED_TYPES


def test_dfw_spec_json_round_trips_under_schema_v2():
    # the new dfw_geo tuple fields must survive to_json_dict/from_json_dict (they default () elsewhere,
    # so a missed coercion would break EVERY scenario's round-trip — this pins it for a geo spec).
    spec = SCENARIOS["dfw_future_wing_zipline_amazon"]
    assert ScenarioSpec.from_json_dict(spec.to_json_dict()) == spec
