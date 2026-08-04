"""HubVoronoiDemand — geographically-anchored demand (Poisson in time, Voronoi origins).

The load-bearing correctness checks are exact, non-flaky invariants: an origin is *exactly* the
nearest hub of its USS to the customer, and the hub layout is fixed by ``hub_seed`` independent of the
demand seed. Distribution-shape checks (lengths, shares, counts) use fixed seeds + loose tolerances.
"""

import numpy as np
import pytest

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
    rs = HubRadiusDemand(n_hubs_per_uss={"a": 4}, return_flights=True).generate(
        cfg, np.random.default_rng(0))
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


def test_terminal_airspace_filter_drops_foreign_column_customers():
    """terminal_airspace_always_active: a delivery whose customer hex sits in a FOREIGN hub's walled
    terminal_cells is unreachable and dropped; every KEPT delivery is clear of foreign walls. (Asserts
    the real reachability invariant directly — comparing to a taa-off run is meaningless because turning
    the flag on widens the reject radius, relocating every hub.)"""
    import dataclasses as dc

    from freespace_sim.planner.hexgrid import circumradius, enu_to_axial, terminal_cells
    from freespace_sim.types import Terminal

    cfg = dc.replace(_radius_cfg(), terminal_airspace_always_active=True)
    dm = HubRadiusDemand(n_hubs_per_uss={"a": 6}, radius_m=6000.0, terminal_radius_m=1500.0)  # big cols ⇒ drops
    R = circumradius(cfg)
    hubs = dm.place_hubs(cfg, np.random.default_rng(dm.hub_seed))
    foreign: dict = {}                                        # cell -> {walling terminal ids}
    for uid, pts in hubs.items():
        for hj in range(pts.shape[0]):
            term = Terminal(f"{uid}#{hj}", dm._pads_for(uid), dm._terminal_radius_for(uid), dm.corridor_overlap_m)
            for c in terminal_cells(pts[hj], term, cfg):
                foreign.setdefault(c, set()).add(term.id)
    reqs = dm.generate(cfg, np.random.default_rng(0))
    # every KEPT delivery's customer hex is clear of every FOREIGN terminal's walls (own hub exempt) —
    # exactly the reachability invariant the filter enforces
    for r in reqs:
        if r.origin_terminal is not None:                    # a delivery: dest is the customer
            walls = foreign.get(enu_to_axial(r.dest[0], r.dest[1], R))
            assert not (walls and any(t != r.origin_terminal.id for t in walls)), "kept customer in a foreign wall"
    # and the filter actually dropped some flights: fid advances on a drop, so kept fids have gaps
    assert reqs and len(reqs) < max(r.flight_id for r in reqs) + 1


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


# --- HubRadiusDemand: per-USS Poisson rates + Gaussian departure offsets (issue: pure density tests) ---

def test_lam_per_uss_counts_scale_and_ignore_global_lambda():
    # per-USS Poisson streams: counts track lam_per_uss, NOT cfg.lam_per_hour / uss_share.
    cfg = SimConfig(region_size_m=(20000.0, 20000.0), lam_per_hour=99999.0, horizon_s=3600.0)
    assert cfg.effective_demand_duration_s == cfg.horizon_s       # legacy default remains the horizon
    m = HubRadiusDemand(n_hubs_per_uss={"a": 3, "b": 4},
                        lam_per_uss={"a": 1000.0, "b": 250.0},
                        uss_share={"a": 1.0, "b": 9.0},          # a 1:9 split IF the global path were used
                        return_flights=False)
    reqs = m.generate(cfg, np.random.default_rng(0))
    na = sum(r.uss_id == "a" for r in reqs)
    nb = sum(r.uss_id == "b" for r in reqs)
    assert 850 <= na <= 1150 and 175 <= nb <= 325             # ≈ lam·horizon/3600 = 1000 / 250 (Poisson)
    assert 3.0 <= na / nb <= 5.0                              # ≈ 4:1 from lam, NOT 1:9 from share/global


def test_lam_per_uss_absent_uss_gets_zero_demand():
    # a USS present in n_hubs_per_uss but OMITTED from lam_per_uss draws zero flights (an
    # infrastructure-only hub) — the intentional counterpart to the __post_init__ unknown-key guard.
    cfg = _radius_cfg()
    m = HubRadiusDemand(n_hubs_per_uss={"a": 3, "b": 3}, lam_per_uss={"a": 500.0}, return_flights=False)
    reqs = m.generate(cfg, np.random.default_rng(0))
    assert reqs and all(r.uss_id == "a" for r in reqs)        # "b" omitted ⇒ no flights


def test_departure_offset_unset_departs_on_filing():
    # legacy path: with no departure_offset_s every leg departs when filed (t_departure == t_request).
    cfg = _radius_cfg()
    m = HubRadiusDemand(n_hubs_per_uss={"a": 4}, lam_per_uss={"a": 800.0})
    reqs = m.generate(cfg, np.random.default_rng(0))
    assert reqs and all(r.t_departure == r.t_request for r in reqs)


def test_legacy_request_mode_departure_offset_applies_to_both_legs_with_distribution():
    # Legacy, non-paired request-first mode gives BOTH delivery and return independent ~N(mean, std) leads.
    cfg = SimConfig(region_size_m=(20000.0, 20000.0), horizon_s=3600.0)
    m = HubRadiusDemand(n_hubs_per_uss={"a": 4}, lam_per_uss={"a": 2000.0},
                        departure_offset_s={"a": (450.0, 60.0)}, return_flights=True)
    reqs = m.generate(cfg, np.random.default_rng(1))
    deliveries = [r for r in reqs if r.origin_terminal is not None]
    returns = [r for r in reqs if r.dest_terminal is not None]
    assert deliveries and returns
    for legs in (deliveries, returns):                        # each leg carries its own drawn lead
        offs = np.array([r.t_departure - r.t_request for r in legs])
        assert abs(offs.mean() - 450.0) < 25.0 and abs(offs.std() - 60.0) < 20.0


def test_departure_offset_clamped_nonnegative():
    # the max(0, N(mean,std)) floor keeps FlightRequest's t_departure >= t_request even when the Gaussian
    # would go negative (mean 0, wide std ⇒ ~half the draws clamp) — no ValueError, and the clamp fires.
    cfg = SimConfig(region_size_m=(20000.0, 20000.0), horizon_s=3600.0)
    m = HubRadiusDemand(n_hubs_per_uss={"a": 4}, lam_per_uss={"a": 1500.0},
                        departure_offset_s={"a": (0.0, 1000.0)}, return_flights=False)
    reqs = m.generate(cfg, np.random.default_rng(2))
    offs = [r.t_departure - r.t_request for r in reqs]
    assert reqs and all(o >= 0.0 for o in offs)               # invariant t_departure >= t_request holds
    assert any(o == 0.0 for o in offs)                        # the clamp actually fired (draws went negative)


def test_departure_offset_absent_uss_departs_on_filing():
    # a USS not named in departure_offset_s draws NO offset (and no rng) ⇒ departs on filing.
    cfg = _radius_cfg()
    m = HubRadiusDemand(n_hubs_per_uss={"a": 3, "b": 3}, lam_per_uss={"a": 500.0, "b": 500.0},
                        departure_offset_s={"a": (300.0, 10.0)}, return_flights=False)
    reqs = m.generate(cfg, np.random.default_rng(0))
    a_off = [r.t_departure - r.t_request for r in reqs if r.uss_id == "a"]
    b_off = [r.t_departure - r.t_request for r in reqs if r.uss_id == "b"]
    assert a_off and all(o > 0.0 for o in a_off)              # "a" leads its departures (~300 s)
    assert b_off and all(o == 0.0 for o in b_off)             # "b" (absent) departs on filing


def test_unknown_uss_in_lam_or_offset_raises():
    with pytest.raises(ValueError, match="lam_per_uss"):
        HubRadiusDemand(n_hubs_per_uss={"a": 3}, lam_per_uss={"b": 100.0})
    with pytest.raises(ValueError, match="departure_offset_s"):
        HubRadiusDemand(n_hubs_per_uss={"a": 3}, departure_offset_s={"typo": (10.0, 1.0)})


# --- HubRadiusDemand: departure-first demand windows + strategically paired returns ------------

def test_departure_mode_uses_demand_duration_not_sim_horizon():
    cfg = SimConfig(
        region_size_m=(20000.0, 20000.0),
        horizon_s=3600.0,
        demand_duration_s=600.0,
    )
    m = HubRadiusDemand(
        n_hubs_per_uss={"a": 3},
        lam_per_uss={"a": 600.0},
        return_flights=False,
        timing_mode="departure",
    )
    reqs = m.generate(cfg, np.random.default_rng(0))
    departures = np.array([r.t_departure for r in reqs])
    assert 65 <= len(reqs) <= 135                         # Poisson(600 × 600/3600) ≈ 100, not 600
    assert np.ptp(departures) <= cfg.demand_duration_s    # one 10-minute departure window


def test_departure_mode_aligns_uss_departure_windows():
    cfg = SimConfig(
        region_size_m=(20000.0, 20000.0),
        horizon_s=3600.0,
        demand_duration_s=600.0,
    )
    m = HubRadiusDemand(
        n_hubs_per_uss={"a": 3, "b": 3},
        lam_per_uss={"a": 1200.0, "b": 1200.0},
        departure_offset_s={"a": (120.0, 20.0), "b": (1800.0, 300.0)},
        return_flights=False,
        timing_mode="departure",
    )
    reqs = m.generate(cfg, np.random.default_rng(1))
    dep_a = np.array([r.t_departure for r in reqs if r.uss_id == "a"])
    dep_b = np.array([r.t_departure for r in reqs if r.uss_id == "b"])
    assert len(dep_a) > 100 and len(dep_b) > 100
    assert abs(dep_a.min() - dep_b.min()) < 60.0
    assert abs(dep_a.max() - dep_b.max()) < 60.0
    assert max(dep_a.max(), dep_b.max()) - min(dep_a.min(), dep_b.min()) <= 600.0


def test_departure_mode_dynamic_preroll_keeps_filings_nonnegative():
    cfg = SimConfig(
        region_size_m=(20000.0, 20000.0),
        horizon_s=3600.0,
        demand_duration_s=600.0,
    )
    m = HubRadiusDemand(
        n_hubs_per_uss={"a": 3},
        lam_per_uss={"a": 600.0},
        departure_offset_s={"a": (1800.0, 300.0)},
        return_flights=False,
        timing_mode="departure",
    )
    reqs = m.generate(cfg, np.random.default_rng(2))
    assert reqs
    assert min(r.t_request for r in reqs) == pytest.approx(0.0)
    assert all(r.t_request >= 0.0 for r in reqs)
    assert all(r.t_departure >= r.t_request for r in reqs)


def _preroll_model(**kw):
    """A departure-mode model whose realized preroll is a 1800 s-mean lead (so it is comfortably > 0)."""
    return HubRadiusDemand(
        n_hubs_per_uss={"a": 3},
        lam_per_uss={"a": 600.0},
        departure_offset_s={"a": (1800.0, 300.0)},
        return_flights=False,
        timing_mode="departure",
        **kw,
    )


_PREROLL_CFG = SimConfig(
    region_size_m=(20000.0, 20000.0), horizon_s=7200.0, demand_duration_s=600.0)


def test_request_clock_offset_pins_the_shift():
    # The legacy shift is data-dependent (earliest filing lands exactly at 0); a fixed offset shifts by
    # that constant instead, so the earliest filing sits at offset - realized_preroll > 0.
    floating = _preroll_model().generate(_PREROLL_CFG, np.random.default_rng(2))
    pinned = _preroll_model(request_clock_offset_s=3600.0).generate(
        _PREROLL_CFG, np.random.default_rng(2))

    assert min(r.t_request for r in floating) == pytest.approx(0.0)
    realized_preroll = 3600.0 - min(r.t_request for r in pinned)
    assert 0.0 < realized_preroll < 3600.0
    assert all(r.t_request >= 0.0 for r in pinned)
    assert all(r.t_departure >= r.t_request for r in pinned)
    # same world, rigidly translated: every flight moves by the SAME constant, none is reshuffled
    pinned_dep = {r.flight_id: r.t_departure for r in pinned}
    assert pinned_dep.keys() == {r.flight_id for r in floating}
    offsets = {round(pinned_dep[r.flight_id] - r.t_departure, 6) for r in floating}
    assert offsets == {round(3600.0 - realized_preroll, 6)}


def test_request_clock_offset_makes_departures_lead_invariant():
    """The property the scheduling-lead arms rest on: pin the clock and two runs differing ONLY in the
    lead share every flight and every desired departure — solely the filing times (FCFS order) move."""
    def gen(lead):
        m = HubRadiusDemand(
            n_hubs_per_uss={"a": 3},
            lam_per_uss={"a": 600.0},
            departure_offset_s={"a": lead},
            return_flights=False,
            timing_mode="departure",
            request_clock_offset_s=3600.0,
        )
        return m.generate(_PREROLL_CFG, np.random.default_rng(2))

    def world(reqs):
        return {r.flight_id: (tuple(r.origin), tuple(r.dest), r.t_departure) for r in reqs}

    short, long_ = gen((480.0, 90.0)), gen((1800.0, 300.0))
    assert world(short) == world(long_)
    short_lead = np.mean([r.t_departure - r.t_request for r in short])
    long_lead = np.mean([r.t_departure - r.t_request for r in long_])
    assert long_lead - short_lead == pytest.approx(1320.0, abs=60.0)


def test_request_clock_offset_too_small_raises():
    # Clipping to keep filings nonnegative would silently break the pinned-clock guarantee (and a
    # negative t_request breaks the planner's monotonic-t_request eviction), so this is a hard error.
    m = _preroll_model(request_clock_offset_s=10.0)
    with pytest.raises(ValueError, match="realized preroll"):
        m.generate(_PREROLL_CFG, np.random.default_rng(2))


def test_request_clock_offset_rejected_outside_departure_mode():
    with pytest.raises(ValueError, match="timing_mode='departure'"):
        HubRadiusDemand(n_hubs_per_uss={"a": 3}, timing_mode="request",
                        request_clock_offset_s=3600.0)


def test_request_clock_offset_rejects_negative():
    with pytest.raises(ValueError, match="must be >= 0"):
        HubRadiusDemand(n_hubs_per_uss={"a": 3}, timing_mode="departure",
                        request_clock_offset_s=-1.0)


def test_departure_mode_preserves_gaussian_outbound_leads():
    cfg = SimConfig(
        region_size_m=(20000.0, 20000.0),
        horizon_s=3600.0,
        demand_duration_s=600.0,
    )
    m = HubRadiusDemand(
        n_hubs_per_uss={"a": 4},
        lam_per_uss={"a": 3600.0},
        departure_offset_s={"a": (480.0, 90.0)},
        return_flights=False,
        timing_mode="departure",
    )
    reqs = m.generate(cfg, np.random.default_rng(3))
    leads = np.array([r.t_departure - r.t_request for r in reqs])
    assert abs(leads.mean() - 480.0) < 15.0
    assert abs(leads.std() - 90.0) < 15.0


def test_paired_return_shares_filing_time_and_follows_nominal_arrival():
    cfg = SimConfig(
        region_size_m=(20000.0, 20000.0),
        horizon_s=3600.0,
        demand_duration_s=600.0,
    )
    m = HubRadiusDemand(
        n_hubs_per_uss={"a": 3},
        lam_per_uss={"a": 600.0},
        departure_offset_s={"a": (480.0, 90.0)},
        return_flights=True,
        turnaround_s=45.0,
        timing_mode="departure",
        paired_return_request=True,
    )
    reqs = m.generate(cfg, np.random.default_rng(4))
    by_id = {r.flight_id: r for r in reqs}
    outbounds = [r for r in reqs if r.origin_terminal is not None]
    assert outbounds
    for outbound in outbounds:
        returned = by_id[outbound.flight_id + 1]
        assert returned.dest_terminal == outbound.origin_terminal
        assert returned.t_request == outbound.t_request
        expected = m._est_trip_s(outbound.origin, outbound.dest, cfg) + m.turnaround_s
        assert returned.t_departure - outbound.t_departure == pytest.approx(expected)


def test_departure_stream_is_stable_when_second_uss_is_added():
    cfg = SimConfig(
        region_size_m=(20000.0, 20000.0),
        horizon_s=3600.0,
        demand_duration_s=600.0,
    )
    common = {
        "radius_m": {"wing": 3000.0},
        "pads_per_hub": {"wing": 4},
        "terminal_radius_m": {"wing": 180.0},
        "lam_per_uss": {"wing": 600.0},
        "departure_offset_s": {"wing": (480.0, 90.0)},
        "return_flights": False,
        "timing_mode": "departure",
    }
    single = HubRadiusDemand(n_hubs_per_uss={"wing": 3}, **common)
    mixed = HubRadiusDemand(
        n_hubs_per_uss={"wing": 3, "amazon": 2},
        radius_m={"wing": 3000.0, "amazon": 2500.0},
        pads_per_hub={"wing": 4, "amazon": 4},
        terminal_radius_m={"wing": 180.0, "amazon": 180.0},
        lam_per_uss={"wing": 600.0, "amazon": 500.0},
        departure_offset_s={"wing": (480.0, 90.0), "amazon": (1800.0, 300.0)},
        return_flights=False,
        timing_mode="departure",
    )
    a = sorted(single.generate(cfg, np.random.default_rng(5)), key=lambda r: r.flight_id)
    b = sorted(
        (r for r in mixed.generate(cfg, np.random.default_rng(5)) if r.uss_id == "wing"),
        key=lambda r: r.flight_id,
    )
    assert len(a) == len(b)
    assert all(np.array_equal(x.origin, y.origin) for x, y in zip(a, b))
    assert all(np.array_equal(x.dest, y.dest) for x, y in zip(a, b))
    assert np.allclose(
        [x.t_departure - x.t_request for x in a],
        [y.t_departure - y.t_request for y in b],
    )
    translations = np.array([y.t_departure - x.t_departure for x, y in zip(a, b)])
    assert np.ptp(translations) < 1e-9                   # only the shared pre-roll origin may change


def test_invalid_timing_mode_raises():
    with pytest.raises(ValueError, match="timing_mode"):
        HubRadiusDemand(timing_mode="filing-ish")


# --- round-trip returns anchored to the REALIZED outbound arrival (two-pass) ----------------------

def _roundtrip_world(**kw):
    """A small congested round-trip world in the density scenarios' own timing mode: both legs filed
    together, the return's departure anchored to the outbound's NOMINAL arrival."""
    cfg = SimConfig(region_size_m=(12000.0, 12000.0), horizon_s=3600.0, demand_duration_s=300.0,
                    planner="astar_shortcut")
    kw = {"timing_mode": "departure", "paired_return_request": True,
          "departure_offset_s": {"a": (480.0, 90.0)}, **kw}
    model = HubRadiusDemand(
        n_hubs_per_uss={"a": 4}, lam_per_uss={"a": 900.0}, radius_m=2500.0,
        pads_per_hub=2, terminal_radius_m=120.0, return_flights=True, **kw)
    return cfg, model


def test_return_leg_names_its_outbound():
    # The link must be explicit: the old flight_id + 1 parity convention is invisible to any consumer
    # and silently wrong for a model that emits legs in another order.
    cfg, model = _roundtrip_world()
    reqs = model.generate(cfg, np.random.default_rng(0))
    by_id = {r.flight_id: r for r in reqs}
    returns = [r for r in reqs if r.paired_outbound_id is not None]
    assert returns
    assert all(r.dest_terminal is not None for r in returns)      # a return lands at its hub
    for r in returns:
        out = by_id[r.paired_outbound_id]
        assert out.origin_terminal is not None                    # ...and its outbound left from one
        assert np.allclose(out.origin, r.dest)                    # same hub, reversed
        assert np.allclose(out.dest, r.origin)
    assert all(r.paired_outbound_id is None for r in reqs if r.origin_terminal is not None)




def test_realized_anchor_departs_on_the_arrival_that_actually_happened():
    """The property the coupling exists for: every return departs after its aircraft is down."""
    from freespace_sim.sim import realized_arrival_s

    cfg, model = _roundtrip_world()

    def slips(res):
        by = {i.request.flight_id: i for i in res.intents}
        out = []
        for i in res.intents:
            o = by.get(i.request.paired_outbound_id) if i.request.paired_outbound_id else None
            if o is None or not (i.accepted and o.accepted):
                continue
            out.append(realized_arrival_s(o) - i.request.t_departure)   # >0 ⇒ departs before landing
        return np.array(out)

    nominal = run(cfg, demand=model, return_anchor="nominal")
    realized = run(cfg, demand=model, return_anchor="realized")
    s_nom, s_real = slips(nominal), slips(realized)

    assert len(s_nom) and len(s_nom) == len(s_real)
    assert (s_nom > 0).mean() > 0.5                 # the artifact is severe under the nominal anchor
    # ...and the coupling removes it OUTRIGHT — this is exact, not a fixed-point approximation
    assert (s_real > 0).sum() == 0
    # each return leaves exactly one pad dwell after its own outbound touched down
    assert s_real.max() == pytest.approx(-cfg.hover_time_s, abs=cfg.dt_s)
    assert len(realized.intents) == len(nominal.intents)   # same flight set; only departures moved


def test_realized_anchor_keeps_filing_times_and_flight_set_intact():
    # FCFS order (and the monotonic-t_request occupancy eviction that rides on it) must not move.
    cfg, model = _roundtrip_world()
    nominal = run(cfg, demand=model, return_anchor="nominal")
    realized = run(cfg, demand=model, return_anchor="realized")
    key = lambda res: [(i.request.flight_id, i.request.t_request) for i in res.intents]  # noqa: E731
    assert key(nominal) == key(realized)
    # outbound legs are upstream of the coupling, so their departures are untouched
    assert ([i.request.t_departure for i in nominal.intents if i.request.paired_outbound_id is None]
            == [i.request.t_departure for i in realized.intents if i.request.paired_outbound_id is None])


def test_realized_anchor_keeps_nominal_when_the_outbound_is_denied():
    # Dropping or denying the return instead would make the flight SET depend on congestion, which
    # breaks any paired comparison across runs (e.g. the scheduling-lead arms).
    from freespace_sim.sim import realized_arrival_s
    from freespace_sim.types import DenialReason, IntentStatus, OperationalIntent

    cfg, model = _roundtrip_world()
    reqs = model.generate(cfg, np.random.default_rng(0))
    outbound = next(r for r in reqs if r.paired_outbound_id is None)

    # a denied outbound yields no arrival, so the loop leaves its return on the nominal anchor
    denied = OperationalIntent(request=outbound, status=IntentStatus.REJECTED,
                               denial_reason=DenialReason.BUDGET_EXCEEDED)
    assert realized_arrival_s(denied) is None

    # end-to-end the flight set is identical either way — congestion never removes a leg
    realized = run(cfg, demand=model, return_anchor="realized")
    assert len(realized.intents) == len(reqs)
    assert any(r.paired_outbound_id is not None for r in reqs)   # the fixture really pairs legs
    assert all(i.request.t_departure >= i.request.t_request for i in realized.intents)


def test_realized_anchor_rejects_parallel_and_unknown_values():
    from freespace_sim.parallel import ParallelConfig

    cfg, model = _roundtrip_world()
    with pytest.raises(ValueError, match="sequential loop"):
        run(cfg, demand=model, return_anchor="realized", parallel=ParallelConfig(n_workers=2))
    with pytest.raises(ValueError, match="unknown return_anchor"):
        run(cfg, demand=model, return_anchor="whenever")


def test_nominal_anchor_is_byte_identical_to_no_flag():
    cfg, model = _roundtrip_world()
    a = run(cfg, demand=model)
    b = run(cfg, demand=model, return_anchor="nominal")
    assert ([(i.request.flight_id, i.request.t_departure, i.ground_delay_s) for i in a.intents]
            == [(i.request.flight_id, i.request.t_departure, i.ground_delay_s) for i in b.intents])
