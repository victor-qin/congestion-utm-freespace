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


def test_get_scenario_resolves_and_rejects():
    assert get_scenario("metro_2uss") is SCENARIOS["metro_2uss"]
    with pytest.raises(ValueError):
        get_scenario("no_such_scenario")


def test_config_applies_overrides_over_defaults():
    spec = ScenarioSpec("x", region_m=(5000.0, 4000.0), horizon_s=900.0, lam_per_hour=240.0,
                        seed=3, planner="astar_shortcut")
    cfg = spec.config()
    assert cfg.region_size_m == (5000.0, 4000.0)
    assert cfg.horizon_s == 900.0 and cfg.lam_per_hour == 240.0 and cfg.seed == 3
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
