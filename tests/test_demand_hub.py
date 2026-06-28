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


# --- HubRadiusDemand: single-point hubs + terminals + radius areas + returns ------------------

def _radius_cfg():
    return SimConfig(region_size_m=(20000.0, 20000.0), lam_per_hour=600.0, horizon_s=3600.0)


def test_hubs_are_single_points():
    cfg = _radius_cfg()
    hubs = HubRadiusDemand(n_hubs_per_uss={"a": 3, "b": 5}).place_hubs(
        cfg, np.random.default_rng(0))
    assert hubs["a"].shape == (3, 2) and hubs["b"].shape == (5, 2)


def test_delivery_sets_origin_terminal_with_capacity():
    cfg = _radius_cfg()
    m = HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 4}, pads_per_hub=3, return_flights=False)
    reqs = m.generate(cfg, np.random.default_rng(0))
    assert reqs
    for r in reqs:                                   # delivery: origin is a hub terminal of capacity 3
        assert r.origin_terminal is not None and r.dest_terminal is None
        assert r.origin_terminal.capacity == 3 and str(r.origin_terminal.id).startswith("walmart_uss#")


def test_pads_per_hub_is_capacity_not_geometry():
    # pads_per_hub changes the capacity tag, NOT the hub locations (single points, stable across N)
    cfg = _radius_cfg()
    a = HubRadiusDemand(n_hubs_per_uss={"a": 5}, pads_per_hub=1, return_flights=False)
    b = HubRadiusDemand(n_hubs_per_uss={"a": 5}, pads_per_hub=8, return_flights=False)
    ha = a.place_hubs(cfg, np.random.default_rng(a.hub_seed))["a"]
    hb = b.place_hubs(cfg, np.random.default_rng(b.hub_seed))["a"]
    assert np.array_equal(ha, hb)                                 # same infrastructure
    assert {r.origin_terminal[1] for r in b.generate(cfg, np.random.default_rng(0))} == {8}


def test_customer_within_per_uss_radius():
    cfg = _radius_cfg()
    m = HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 4, "stripmall_uss": 8},
                        radius_m={"walmart_uss": 6000.0, "stripmall_uss": 2000.0},
                        return_flights=False)
    hubs = m.place_hubs(cfg, np.random.default_rng(m.hub_seed))
    for r in m.generate(cfg, np.random.default_rng(7)):           # delivery: dest is the customer
        cust = _xy(r.dest)
        radius = 6000.0 if r.uss_id == "walmart_uss" else 2000.0
        dmin = np.linalg.norm(hubs[r.uss_id] - cust, axis=1).min()
        assert dmin <= radius + 1e-6


def test_return_flights_roundtrip_and_terminals():
    cfg = _radius_cfg()
    nd = len(HubRadiusDemand(n_hubs_per_uss={"a": 4}, return_flights=False).generate(
        cfg, np.random.default_rng(0)))
    # clip off so every delivery keeps its return — this test is about the round-trip pairing
    rs = HubRadiusDemand(n_hubs_per_uss={"a": 4}, return_flights=True,
                         clip_returns_to_horizon=False).generate(cfg, np.random.default_rng(0))
    assert len(rs) == 2 * nd                                      # one return per delivery
    deliveries = [r for r in rs if r.origin_terminal is not None]
    returns = [r for r in rs if r.dest_terminal is not None]
    assert len(deliveries) == len(returns) == nd
    # a return lands at a hub that some delivery launched from (same hub_id)
    deliv_hubs = {r.origin_terminal[0] for r in deliveries}
    assert all(r.dest_terminal[0] in deliv_hubs for r in returns)
    # round trip: every (origin→dest) leg has its reverse among the flights
    legs = {(tuple(np.round(_xy(r.origin), 2)), tuple(np.round(_xy(r.dest), 2))) for r in rs}
    assert all((d, o) in legs for (o, d) in legs)


def test_terminal_airspace_filter_drops_foreign_column_customers_keeps_subset():
    """terminal_airspace_always_active: customers inside a FOREIGN hub's column are dropped (spurious),
    and the RNG/fid stream is preserved so the kept flights are a clean subset of the unfiltered run."""
    import dataclasses as dc

    cfg = _radius_cfg()
    cfg_on = dc.replace(cfg, terminal_airspace_always_active=True)
    kw = dict(n_hubs_per_uss={"a": 6}, radius_m=6000.0, terminal_radius_m=1500.0)  # big columns ⇒ drops
    off = HubRadiusDemand(**kw).generate(cfg, np.random.default_rng(0))
    on = HubRadiusDemand(**kw).generate(cfg_on, np.random.default_rng(0))
    assert len(on) < len(off)                                          # spurious customers dropped
    assert {r.flight_id for r in on} <= {r.flight_id for r in off}     # clean subset (rng/fid preserved)


def test_clip_returns_to_horizon_drops_only_post_horizon_returns():
    """clip=True ends demand at the horizon: returns landing past H are dropped, deliveries untouched."""
    cfg = _radius_cfg()
    kw = dict(n_hubs_per_uss={"a": 4}, return_flights=True)
    tail = HubRadiusDemand(**kw, clip_returns_to_horizon=False).generate(cfg, np.random.default_rng(0))
    clip = HubRadiusDemand(**kw, clip_returns_to_horizon=True).generate(cfg, np.random.default_rng(0))

    # every kept return lands strictly before the horizon
    assert all(r.t_request < cfg.horizon_s for r in clip if r.dest_terminal is not None)
    # at least one return was actually past the horizon (else the test proves nothing)
    assert any(r.t_request >= cfg.horizon_s for r in tail if r.dest_terminal is not None)
    # deliveries are an identically-labelled, identical-geometry subset (fid still advances on a drop)
    dt = {r.flight_id: r for r in tail if r.origin_terminal is not None}
    dc = {r.flight_id: r for r in clip if r.origin_terminal is not None}
    assert set(dc) == set(dt)
    assert all(dc[i].t_request == dt[i].t_request and np.allclose(dc[i].dest, dt[i].dest) for i in dc)
    # the only difference is the dropped post-horizon returns
    assert len(clip) < len(tail)


def test_radius_demand_run_verified_astar():
    cfg = SimConfig(planner="astar", region_size_m=(8000.0, 8000.0),
                    lam_per_hour=120.0, horizon_s=900.0, seed=1)
    res = run(cfg, demand=HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 4, "stripmall_uss": 10},
                                          radius_m=2500.0, pads_per_hub=2, return_flights=True))
    assert res.verified
    s = res.summary()
    assert s["n_accepted"] + s["n_denied"] == s["n_requests"]


def test_more_pads_per_hub_cut_ground_delay():
    # the Phase B payoff end-to-end: on a hub-funnelled scenario (few hubs, returns), giving each hub
    # more pads slashes pad-contention ground delay — same demand, only pads_per_hub changes.
    cfg = SimConfig(planner="astar", region_size_m=(8000.0, 8000.0),
                    lam_per_hour=600.0, horizon_s=300.0, seed=1)

    def mean_delay(pads):
        dem = HubRadiusDemand(n_hubs_per_uss={"walmart_uss": 2, "stripmall_uss": 3}, radius_m=2500.0,
                              pads_per_hub=pads, terminal_radius_m=150.0, return_flights=True)
        res = run(cfg, demand=dem)
        assert res.verified
        return float(np.mean([a.ground_delay_s for a in res.accepted]))

    assert mean_delay(4) < 0.5 * mean_delay(1)   # 1→4 pads cuts mean delay by far more than half
