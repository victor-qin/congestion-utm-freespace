"""ScenarioSpec — the world recipe that the execute step turns into a run."""

import numpy as np
import pytest

from freespace_sim.config import SimConfig
from freespace_sim.demand import HubRadiusDemand, HubVoronoiDemand, UniformPoissonDemand
from freespace_sim.scenarios import (
    SCENARIOS,
    DemandSpec,
    ScenarioSpec,
    get_scenario,
    with_overrides,
)
from freespace_sim.scenarios.density import (
    AMAZON_USS,
    LEAD_ARM_CLOCK_OFFSET_S,
    LEAD_ARM_OPERATORS,
    LEAD_ARMS,
    WING_ZIPLINE_LEAD_S,
    WING_ZIPLINE_USS,
)


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


_MIXED_WORLDS = ["density_faa_wing_zipline_amazon", "density_future_wing_zipline_amazon"]


def _world(reqs):
    """The identity every arm of a world must agree on — flights and DESIRED departures."""
    return {r.flight_id: (r.uss_id, tuple(r.origin), tuple(r.dest), r.t_departure) for r in reqs}


def _filings(reqs, uss):
    return {r.flight_id: r.t_request for r in reqs if r.uss_id == uss}


def _generate(name):
    spec = SCENARIOS[name]
    cfg = spec.config()
    return spec.demand_model().generate(cfg, np.random.default_rng(cfg.seed))


@pytest.mark.parametrize("base", _MIXED_WORLDS)
@pytest.mark.parametrize("token", sorted(LEAD_ARM_OPERATORS))
@pytest.mark.parametrize("arm", sorted(LEAD_ARMS))
def test_lead_arm_changes_one_operators_lead_and_pins_the_clock(base, token, arm):
    """An arm is its base world with two edits: ONE operator's lead, and a pinned request clock.

    Parametrized over both operators — the arm machinery is operator-agnostic, so Wing/Zipline gets the
    same treatment Amazon does and the sweep can be run from either side.
    """
    _, varied_uss = LEAD_ARM_OPERATORS[token]
    held_uss = AMAZON_USS if varied_uss == WING_ZIPLINE_USS else WING_ZIPLINE_USS
    spec, base_spec = SCENARIOS[f"{base}_{token}lead{arm}"], SCENARIOS[base]
    demand, base_demand = spec.demand_model(), base_spec.demand_model()

    assert demand.departure_offset_s[varied_uss] == LEAD_ARMS[arm]
    # the OTHER operator holds still — that is what makes the arm a contrast rather than a translation
    assert demand.departure_offset_s[held_uss] == base_demand.departure_offset_s[held_uss]
    assert demand.request_clock_offset_s == LEAD_ARM_CLOCK_OFFSET_S
    assert base_demand.request_clock_offset_s is None   # base worlds keep the floating preroll
    assert demand.timing_mode == "departure"            # the pinned clock only applies here

    # everything else — geometry, capacity, rates, horizon — is the base world untouched
    assert demand.n_hubs_per_uss == base_demand.n_hubs_per_uss
    assert demand.lam_per_uss == base_demand.lam_per_uss
    assert demand.radius_m == base_demand.radius_m
    assert demand.pads_per_hub == base_demand.pads_per_hub
    assert demand.terminal_radius_m == base_demand.terminal_radius_m
    assert demand.paired_return_request == base_demand.paired_return_request
    assert (spec.region_m, spec.horizon_s, spec.demand_duration_s, spec.lam_per_hour) == (
        base_spec.region_m, base_spec.horizon_s, base_spec.demand_duration_s, base_spec.lam_per_hour)
    assert spec.flight_levels_m == base_spec.flight_levels_m


def test_lead_ladder_spans_both_operators_defaults():
    # The ladder is operator-agnostic precisely because its ends ARE the two defaults: 8 minutes is
    # Wing/Zipline's own lead, 30 minutes is Amazon's. So "08m" reads as "file like Wing/Zipline" and
    # "30m" as "file like Amazon" whichever operator an arm varies.
    assert LEAD_ARMS["08m"] == WING_ZIPLINE_LEAD_S
    assert LEAD_ARMS["15m"] == (900.0, 150.0)
    assert LEAD_ARMS["30m"] == (1800.0, 300.0)


@pytest.mark.parametrize("base", _MIXED_WORLDS)
def test_az_and_wz_sweeps_share_the_status_quo_pivot(base):
    """``azlead30m`` and ``wzlead08m`` are both operators at their defaults — the SAME recipe under two
    names, and the point both sweeps rotate about. Running both would just pay twice for one world."""
    assert SCENARIOS[f"{base}_azlead30m"].demand == SCENARIOS[f"{base}_wzlead08m"].demand
    # ...and it is the status quo: identical leads to the un-pinned base world
    assert (SCENARIOS[f"{base}_azlead30m"].demand.departure_offset_s
            == SCENARIOS[base].demand.departure_offset_s)


def test_lead_arms_share_one_world_and_move_only_the_varied_operator():
    """THE load-bearing invariant: within a seed, arms differ ONLY in one operator's FCFS position.

    Every arm of a world — across BOTH operator sweeps — must yield the identical flight set: same ids,
    operators, endpoints, and *desired departures*. That is what lets delays be differenced
    flight-by-flight rather than only in aggregate. Two things could silently break it and void every
    comparison built on it: a change to how ``rng.normal`` consumes entropy (which would desync the
    varied operator's stream from arm to arm), and dropping the pinned clock offset (whose
    data-dependent twin translates the whole world per arm). Only the FAA world is generated here — the
    far-future one is ~27k flights and shares the same code path.
    """
    base = "density_faa_wing_zipline_amazon"
    generated = {
        f"{token}lead{arm}": _generate(f"{base}_{token}lead{arm}")
        for token in LEAD_ARM_OPERATORS
        for arm in LEAD_ARMS
    }

    reference = _world(generated["azlead30m"])
    assert reference
    for name, reqs in generated.items():
        assert _world(reqs) == reference, f"{name} perturbed the flight set or desired departures"

    for token, (_, varied_uss) in LEAD_ARM_OPERATORS.items():
        held_uss = AMAZON_USS if varied_uss == WING_ZIPLINE_USS else WING_ZIPLINE_USS
        arms = [generated[f"{token}lead{arm}"] for arm in LEAD_ARMS]

        held = [_filings(reqs, held_uss) for reqs in arms]
        assert all(h == held[0] for h in held), f"{token} sweep moved {held_uss}'s filings"

        # the varied operator's filings DO move, by the difference in lead — that shift IS the treatment
        mean_filing = [float(np.mean(list(_filings(reqs, varied_uss).values()))) for reqs in arms]
        assert mean_filing[0] > mean_filing[1] > mean_filing[2]
        assert mean_filing[0] - mean_filing[2] == pytest.approx(
            LEAD_ARMS["30m"][0] - LEAD_ARMS["08m"][0], abs=60.0)


@pytest.mark.parametrize("base", _MIXED_WORLDS)
@pytest.mark.parametrize("arm", ["azlead30m", "wzlead30m"])
def test_lead_arm_departures_stay_inside_the_horizon(base, arm):
    """The pinned preroll must not push flights past ``horizon_s``, where the compiled A* box guard
    dispatches to the ~5-7x slower reference. Checks the real margin rather than trusting the sizing
    arithmetic in density.py. Both 30-minute arms are exercised because the preroll is a max-order
    statistic over the varied operator's lead draws — ``wzlead30m`` is the binding case, drawing that
    maximum from Wing/Zipline's far larger flight count.
    """
    spec = SCENARIOS[f"{base}_{arm}"]
    cfg = spec.config()
    reqs = spec.demand_model().generate(cfg, np.random.default_rng(cfg.seed))
    assert min(r.t_request for r in reqs) > 0.0        # offset genuinely exceeded the realized preroll
    assert max(r.t_departure for r in reqs) < cfg.horizon_s


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
    } | {
        f"{base}_{token}lead{arm}"
        for base in ("density_faa_wing_zipline_amazon", "density_future_wing_zipline_amazon")
        for token in LEAD_ARM_OPERATORS
        for arm in LEAD_ARMS
    }
    # The arm half of that set is DERIVED from the same constants that generate the scenarios, so it
    # would happily absorb a fourth lead rung or a third operator. Pin the shape explicitly too, or the
    # registry is only checked against itself.
    assert (len(LEAD_ARMS), len(LEAD_ARM_OPERATORS)) == (3, 2)
    assert len(density_names) == 8 + 12
    assert all(SCENARIOS[name].description for name in density_names)
    assert "density_test" not in SCENARIOS
