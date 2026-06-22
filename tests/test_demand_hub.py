"""HubVoronoiDemand — geographically-anchored demand (Poisson in time, Voronoi origins).

The load-bearing correctness checks are exact, non-flaky invariants: an origin is *exactly* the
nearest hub of its USS to the customer, and the hub layout is fixed by ``hub_seed`` independent of the
demand seed. Distribution-shape checks (lengths, shares, counts) use fixed seeds + loose tolerances.
"""

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.demand import (
    HubRadiusDemand,
    HubVoronoiDemand,
    UniformPoissonDemand,
    _pad_offsets,
    nearest_hub,
)
from freespace_sim.sim import run


def _len(r):
    return float(np.linalg.norm(np.array(r.dest)[:2] - np.array(r.origin)[:2]))


def _xy(p):
    return np.array(p)[:2]


def test_nearest_hub_picks_min_distance():
    hubs = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    assert np.allclose(nearest_hub(np.array([90.0, 5.0]), hubs), [100.0, 0.0])
    assert np.allclose(nearest_hub(np.array([2.0, 3.0]), hubs), [0.0, 0.0])


def test_hubs_stable_across_demand_seeds():
    m = HubVoronoiDemand()
    cfg = SimConfig(region_size_m=(5000.0, 5000.0))
    h1 = m.place_hubs(cfg, np.random.default_rng(m.hub_seed))
    h2 = m.place_hubs(cfg, np.random.default_rng(m.hub_seed))
    assert set(h1) == set(h2)
    for uid in h1:
        assert np.array_equal(h1[uid], h2[uid])   # infrastructure is fixed, not demand-seed dependent


def test_origin_is_voronoi_nearest_hub():
    m = HubVoronoiDemand()
    cfg = SimConfig(region_size_m=(5000.0, 5000.0), lam_per_hour=300.0, horizon_s=3600.0)
    hubs = m.place_hubs(cfg, np.random.default_rng(m.hub_seed))
    for r in m.generate(cfg, np.random.default_rng(7)):
        customer = np.array(r.dest)[:2]       # delivery: dest is the customer
        origin = np.array(r.origin)[:2]
        uss_hubs = hubs[r.uss_id]
        assert np.allclose(origin, nearest_hub(customer, uss_hubs))
        dmin = float(np.linalg.norm(uss_hubs - customer, axis=1).min())
        assert np.isclose(float(np.linalg.norm(origin - customer)), dmin)   # none closer


def test_two_uss_distinct_density_distinct_lengths():
    m = HubVoronoiDemand(n_hubs_per_uss={"walmart_uss": 4, "stripmall_uss": 25})
    cfg = SimConfig(region_size_m=(5000.0, 5000.0), lam_per_hour=600.0, horizon_s=3600.0)
    reqs = m.generate(cfg, np.random.default_rng(3))
    wl = [_len(r) for r in reqs if r.uss_id == "walmart_uss"]
    sl = [_len(r) for r in reqs if r.uss_id == "stripmall_uss"]
    assert np.mean(wl) > np.mean(sl)   # fewer hubs ⇒ bigger cells ⇒ longer flights


def test_flight_lengths_short_vs_uniform():
    cfg = SimConfig(region_size_m=(10000.0, 10000.0), lam_per_hour=600.0, horizon_s=3600.0)
    hub = HubVoronoiDemand().generate(cfg, np.random.default_rng(0))
    uni = UniformPoissonDemand().generate(cfg, np.random.default_rng(0))
    assert np.mean([_len(r) for r in hub]) < np.mean([_len(r) for r in uni])


def test_arrivals_poisson_and_in_horizon():
    cfg = SimConfig(lam_per_hour=600.0, horizon_s=3600.0)
    reqs = HubVoronoiDemand().generate(cfg, np.random.default_rng(5))
    assert 450 < len(reqs) < 770                          # ~Poisson(600), generous band
    ts = [r.t_request for r in reqs]
    assert all(0.0 <= t <= cfg.horizon_s for t in ts)
    assert ts == sorted(ts)                               # FCFS-ordered output


def test_min_od_separation_respected():
    m = HubVoronoiDemand(min_od_separation_m=200.0, n_hubs_per_uss={"a": 3, "b": 3})
    cfg = SimConfig(region_size_m=(5000.0, 5000.0), lam_per_hour=300.0, horizon_s=3600.0)
    for r in m.generate(cfg, np.random.default_rng(2)):
        assert _len(r) >= 200.0 - 1e-6


def test_uss_share_split():
    m = HubVoronoiDemand(
        n_hubs_per_uss={"walmart_uss": 6, "stripmall_uss": 20},
        uss_share={"walmart_uss": 0.8, "stripmall_uss": 0.2},
    )
    cfg = SimConfig(lam_per_hour=2000.0, horizon_s=3600.0)
    reqs = m.generate(cfg, np.random.default_rng(4))
    frac_w = np.mean([r.uss_id == "walmart_uss" for r in reqs])
    assert 0.70 < frac_w < 0.90                           # ≈0.8 by LLN


def test_pickup_direction_swaps_o_d():
    m = HubVoronoiDemand(direction="pickup")
    cfg = SimConfig(region_size_m=(5000.0, 5000.0), lam_per_hour=120.0, horizon_s=3600.0)
    hubs = m.place_hubs(cfg, np.random.default_rng(m.hub_seed))
    for r in m.generate(cfg, np.random.default_rng(1)):
        customer = np.array(r.origin)[:2]                 # pickup: origin is the customer
        dest = np.array(r.dest)[:2]
        assert np.allclose(dest, nearest_hub(customer, hubs[r.uss_id]))   # dest is the hub


def test_single_hub_uss_degenerate():
    m = HubVoronoiDemand(n_hubs_per_uss={"solo": 1})
    cfg = SimConfig(region_size_m=(4000.0, 4000.0), lam_per_hour=200.0, horizon_s=3600.0)
    reqs = m.generate(cfg, np.random.default_rng(0))
    origins = {tuple(np.round(np.array(r.origin)[:2], 3)) for r in reqs}
    assert len(origins) == 1                              # all flights launch from the one hub


def test_hub_demand_run_verified_astar():
    cfg = SimConfig(planner="astar", region_size_m=(4000.0, 4000.0),
                    lam_per_hour=120.0, horizon_s=900.0, seed=1)
    res = run(cfg, demand=HubVoronoiDemand(n_hubs_per_uss={"walmart_uss": 4, "stripmall_uss": 10}))
    assert res.verified
    s = res.summary()
    assert s["n_accepted"] + s["n_denied"] == s["n_requests"]


# --- HubRadiusDemand: multi-pad hubs + radius service areas + return flights -------------------

def _radius_cfg():
    return SimConfig(region_size_m=(20000.0, 20000.0), lam_per_hour=600.0, horizon_s=3600.0)


def test_pad_offsets_spaced_and_centered():
    offs = _pad_offsets(4, 150.0)
    assert offs.shape == (4, 2)
    assert np.allclose(offs.mean(axis=0), 0.0)                    # centered on the hub
    d = np.linalg.norm(offs[:, None, :] - offs[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= 150.0 - 1e-6                                # adjacent pads a full spacing apart
    assert _pad_offsets(1, 150.0).shape == (1, 2) and np.allclose(_pad_offsets(1, 150.0), 0.0)


def test_pads_per_hub_makes_spaced_independent_pads():
    cfg = _radius_cfg()
    m = HubRadiusDemand(n_hubs_per_uss={"a": 3}, pads_per_hub=4)
    pads = m.place_pads(cfg, np.random.default_rng(m.hub_seed))["a"]
    assert pads.shape == (3, 4, 2)
    spacing = 2.5 * cfg.effective_hover_radius_m
    for hub_pads in pads:                                         # within each hub, pads ≥ a spacing apart
        d = np.linalg.norm(hub_pads[:, None] - hub_pads[None, :], axis=2)
        np.fill_diagonal(d, np.inf)
        assert d.min() >= spacing - 1e-6


def test_pads_per_hub_multiplies_distinct_launch_points():
    cfg = _radius_cfg()
    base = HubRadiusDemand(n_hubs_per_uss={"a": 5}, pads_per_hub=1, return_flights=False)
    multi = HubRadiusDemand(n_hubs_per_uss={"a": 5}, pads_per_hub=4, return_flights=False)
    o1 = {tuple(np.round(_xy(r.origin), 3)) for r in base.generate(cfg, np.random.default_rng(0))}
    o4 = {tuple(np.round(_xy(r.origin), 3)) for r in multi.generate(cfg, np.random.default_rng(0))}
    assert len(o1) <= 5                                           # ≤ one launch point per hub
    assert len(o4) > len(o1)                                      # pads add parallel launch points


def test_customer_within_radius_of_a_hub():
    cfg = _radius_cfg()
    m = HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 4, "stripmall_uss": 8},
                        radius_m=2500.0, return_flights=False)
    centers = {uid: pads.mean(axis=1) for uid, pads in
               m.place_pads(cfg, np.random.default_rng(m.hub_seed)).items()}
    for r in m.generate(cfg, np.random.default_rng(7)):           # deliveries: dest is the customer
        c = _xy(r.dest)
        dmin = np.linalg.norm(centers[r.uss_id] - c, axis=1).min()
        assert dmin <= 2500.0 + 1e-6                              # customer in some hub's disk


def test_return_flights_double_count_and_roundtrip():
    cfg = _radius_cfg()
    deliv = HubRadiusDemand(n_hubs_per_uss={"a": 4}, pads_per_hub=2, return_flights=False)
    both = HubRadiusDemand(n_hubs_per_uss={"a": 4}, pads_per_hub=2, return_flights=True)
    nd = len(deliv.generate(cfg, np.random.default_rng(0)))
    rs = both.generate(cfg, np.random.default_rng(0))
    assert len(rs) == 2 * nd                                      # one return per delivery
    # the round trip exists: for every (origin→dest) there is a matching (dest→origin)
    legs = {(tuple(np.round(_xy(r.origin), 2)), tuple(np.round(_xy(r.dest), 2))) for r in rs}
    assert all((d, o) in legs for (o, d) in legs)                 # every leg has its reverse


def test_radius_demand_run_verified_astar():
    cfg = SimConfig(planner="astar", region_size_m=(8000.0, 8000.0),
                    lam_per_hour=120.0, horizon_s=900.0, seed=1)
    res = run(cfg, demand=HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 4, "stripmall_uss": 10},
                                          radius_m=2500.0, pads_per_hub=2, return_flights=True))
    assert res.verified
    s = res.summary()
    assert s["n_accepted"] + s["n_denied"] == s["n_requests"]
