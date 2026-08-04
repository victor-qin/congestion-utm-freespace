"""DfwGeoDemand — real-geography hub placement + census-density destinations.

Placement/generation tests need the baked artifacts (freespace_sim/data/dfw/); they skip cleanly if
prep hasn't run. The scenario-number contract (dfw_* twins == density_* parents) lives in
test_scenarios.py and needs no artifacts.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest

from freespace_sim.scenarios import get_scenario, with_overrides

_ARTIFACTS = Path(__file__).resolve().parents[1] / "freespace_sim" / "data" / "dfw"
pytestmark = pytest.mark.skipif(
    not (_ARTIFACTS / "retail_pois.csv").exists(),
    reason="DFW geo artifacts not baked — run analysis/prep_dfw.py (needs the `geo` extra)",
)


def _placed(name):
    spec = get_scenario(name)
    cfg = spec.config()
    dm = spec.demand_model()
    hubs = dm.place_hubs(cfg, np.random.default_rng(dm.hub_seed))
    return spec, cfg, dm, hubs


def test_hubs_respect_separation_and_stay_in_region():
    _, cfg, dm, hubs = _placed("dfw_faa_wing_zipline_amazon")
    w, h = cfg.region_size_m
    flat = [(hubs[u][i], dm._sep_radius(u, cfg)) for u in hubs for i in range(len(hubs[u]))]
    for (c1, r1), (c2, r2) in itertools.combinations(flat, 2):        # non-overlap across operators
        assert float(np.linalg.norm(c1 - c2)) >= r1 + r2 + dm.min_hub_gap_m - 1e-6
    for uid, pts in hubs.items():                                    # every hub inside [0,w]x[0,h]
        assert ((pts[:, 0] >= 0) & (pts[:, 0] <= w) & (pts[:, 1] >= 0) & (pts[:, 1] <= h)).all(), uid


def test_placement_is_deterministic_in_hub_seed():
    spec = get_scenario("dfw_faa_wing_zipline")
    dm = spec.demand_model()
    a = dm.place_hubs(spec.config(), np.random.default_rng(dm.hub_seed))
    b = dm.place_hubs(spec.config(), np.random.default_rng(dm.hub_seed))
    assert set(a) == set(b) and all(np.allclose(a[u], b[u]) for u in a)


def test_wing_zipline_hub_count_matches_density_number():
    # the abundant retail pool supplies the full density hub count for both scales
    _, _, _, faa = _placed("dfw_faa_wing_zipline")
    _, _, _, fut = _placed("dfw_future_wing_zipline")
    assert faa["wing_zipline_uss"].shape[0] == 182
    assert fut["wing_zipline_uss"].shape[0] == 476


def test_amazon_hubs_are_real_in_region_facilities():
    # the full-metroplex frame contains all 14 real last-mile facilities, so the density Amazon count
    # is met exactly (top-7 by tract density for the FAA scenario).
    _, _, _, hubs = _placed("dfw_faa_wing_zipline_amazon")
    assert hubs["amazon_uss"].shape[0] == 7


def test_generate_produces_both_uss_and_destinations_within_radius():
    spec = get_scenario("dfw_faa_wing_zipline_amazon")
    small = with_overrides(spec, horizon_s=600.0, demand_duration_s=120.0)
    cfg, dm = small.config(), small.demand_model()
    reqs = dm.generate(cfg, np.random.default_rng(0))
    assert {r.uss_id for r in reqs} == {"wing_zipline_uss", "amazon_uss"}
    for r in reqs:
        if r.origin_terminal is not None:                            # outbound: origin = hub
            radius = dm._radius_for(r.uss_id)
            assert np.linalg.norm(np.asarray(r.dest[:2]) - np.asarray(r.origin[:2])) <= radius + 1.0


def test_missing_fixed_type_raises_clearly():
    spec = get_scenario("dfw_faa_wing_zipline_amazon")
    cfg, dm = spec.config(), spec.demand_model()
    dm.fixed_hub_types = ("Nonexistent",)                            # no facility of this type survives
    with pytest.raises(ValueError, match="fixed hubs"):
        dm.place_hubs(cfg, np.random.default_rng(dm.hub_seed))


def test_retail_pool_exhaustion_raises_clearly():
    spec = get_scenario("dfw_faa_wing_zipline")
    cfg, dm = spec.config(), spec.demand_model()
    dm.n_hubs_per_uss = {"wing_zipline_uss": 10_000_000}             # more than the pool can seat
    with pytest.raises(ValueError, match="pool exhausted"):
        dm.place_hubs(cfg, np.random.default_rng(dm.hub_seed))
