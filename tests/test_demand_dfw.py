"""DfwGeoDemand — real-geography hub placement + census-density destinations.

Placement/generation tests need the baked artifacts (freespace_sim/data/dfw/); they skip cleanly if
prep hasn't run. The scenario-number contract (dfw_* twins == density_* parents) lives in
test_scenarios.py and needs no artifacts.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest

from freespace_sim.geo import DfwGeo
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


def _two_tract_geo(pop_a: float = 1000.0, pop_b: float = 1000.0):
    """A synthetic :class:`DfwGeo` with two equal-population tracts of very different bbox FILL: a
    solid square (fill 1.00) and an L (area 1.75/4.00 = 0.44). Both sit wholly inside the hub's
    service disk, so a population-proportional sampler must split draws ~50/50 between them."""
    square = np.array([[3000., 4500.], [4000., 4500.], [4000., 5500.], [3000., 5500.]])
    ell = np.array([[6000., 4000.], [8000., 4000.], [8000., 4500.],
                    [6500., 4500.], [6500., 6000.], [6000., 6000.]])
    empty = np.empty((0, 2), float)
    return DfwGeo(
        pois_xy=empty, pois_cat=np.array([]), pois_w=np.array([]),
        amazon_xy=empty, amazon_type=np.array([]), amazon_w=np.array([]),
        tract_pop=np.array([pop_a, pop_b]),
        tract_bbox=np.array([[3000., 4500., 4000., 5500.], [6000., 4000., 8000., 6000.]]),
        tract_rings=[[square], [ell]],
    )


def test_tract_draw_follows_population_not_bbox_fill():
    """REGRESSION: redrawing the TRACT on an in-tract rejection (rather than only the point) weights
    every tract by its bbox fill ratio — 0.18-0.99 across the real DFW tracts, uncorrelated with
    population. Here that skew would hand the solid square 1.00/(1.00+0.44) = 70 % of two
    equal-population tracts; a correct sampler gives 50 %."""
    spec = get_scenario("dfw_faa_wing_zipline")
    cfg, dm = spec.config(), spec.demand_model()
    dm._geo = lambda _cfg: _two_tract_geo()                          # no artifacts needed
    hub, radius, rng = np.array([5000.0, 5000.0]), 4000.0, np.random.default_rng(11)
    pts = np.array([dm._draw_customer(hub, radius, 0.0, 10_000.0, 10_000.0, cfg, rng)
                    for _ in range(3000)])
    in_square = float((pts[:, 0] < 5000.0).mean())
    assert abs(in_square - 0.5) < 0.05, in_square
    assert (np.linalg.norm(pts - hub, axis=1) <= radius).all()       # never leaves the service disk
