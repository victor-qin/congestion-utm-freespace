import math

import numpy as np

from freespace_sim import metrics
from freespace_sim.config import SimConfig
from freespace_sim.geometry import BoxSpec, CylinderSpec, box_from_segment
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()


def test_shape_volume_box_and_cylinder():
    box = box_from_segment(vec(0, 0, 0), vec(10, 0, 0), 4.0, 2.0)  # L≈10, W=4, H=2
    assert math.isclose(metrics.shape_volume_m3(box), 10 * 4 * 2, rel_tol=1e-6)
    cyl = CylinderSpec(0, 0, 3.0, 0.0, 5.0)  # r=3, h=5
    assert math.isclose(metrics.shape_volume_m3(cyl), math.pi * 9 * 5, rel_tol=1e-6)


def test_reserved_volume_seconds_clamps_open_window():
    # an open-ended reservation (t_end ~ 1e6) must be clamped to the horizon, not run away
    box = BoxSpec(center=(0, 0, 0), rot=tuple(np.eye(3).flatten()), extents=(1.0, 1.0, 1.0))
    vol = Volume4D(box, 0.0, 1e6)
    got = metrics.reserved_volume_seconds([vol], horizon_s=100.0)
    assert math.isclose(got, 1.0 * 100.0, rel_tol=1e-9)   # 1 m³ × 100 s, not 1e6 s


def test_flight_frame_one_row_per_intent():
    res = run(SimConfig(planner="straight"),
              requests=[FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0)])
    df = metrics.flight_frame(res)
    assert len(df) == len(res.intents) == 1
    assert df.iloc[0]["accepted"]
    assert math.isclose(df.iloc[0]["stretch"], 1.0, abs_tol=0.05)   # free straight flight


def test_aggregate_arithmetic_and_bounds():
    res = run(SimConfig(planner="straight", lam_per_hour=60.0, horizon_s=1800.0, seed=2))
    agg = metrics.aggregate(res)
    assert agg["n_accepted"] + agg["n_denied"] == agg["n_requests"]
    assert 0.0 <= agg["denial_rate"] <= 1.0
    assert agg["congestion_denial_rate"] <= agg["denial_rate"]   # budget ⊆ all denials
    assert 0.0 <= agg["airspace_utilization"] <= 1.0
    assert agg["throughput_per_h"] <= agg["offered_load_per_h"] + 1e-9


def test_total_delay_decomposes_into_levers():
    # total_delay_s = ground_hold + air_loiter + detour-time; excludes mandatory climb
    res = run(SimConfig(planner="straight"),
              requests=[FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0),
                        FlightRequest(1, vec(0, 0, 0), vec(2400, 0, 0), 0.0)])
    df = metrics.flight_frame(res)
    for _, r in df[df["accepted"]].iterrows():
        expected = r["ground_delay_s"] + r["air_hold_s"] + r["air_detour_m"] / CFG.nominal_speed_mps
        assert math.isclose(r["total_delay_s"], expected, rel_tol=1e-9)
    # the FCFS straight case is pure ground-hold: total delay equals the hold, no detour component
    held = df.loc[df["ground_delay_s"] > 0].iloc[0]
    assert math.isclose(held["total_delay_s"], held["ground_delay_s"], rel_tol=1e-9)


def test_delay_pct_is_bounded_and_consistent():
    res = run(SimConfig(planner="straight", lam_per_hour=120.0, horizon_s=1200.0, seed=2))
    df = metrics.flight_frame(res)
    acc = df[df["accepted"]]
    assert "delay_pct" in df.columns
    assert (acc["delay_pct"] >= 0).all() and (acc["delay_pct"] < 100).all()   # fraction of trip time
    # a zero-delay flight is 0%, and pct moves with total_delay (monotone within a run)
    assert acc.loc[acc["total_delay_s"] == 0.0, "delay_pct"].fillna(0).eq(0.0).all()
    # explicit check against the definition for one flight
    r = acc.iloc[acc["total_delay_s"].argmax()]
    nominal = metrics.nominal_flight_time_s(r["straight_line_m"], res.config)
    assert math.isclose(r["delay_pct"], 100 * r["total_delay_s"] / (nominal + r["total_delay_s"]),
                        rel_tol=1e-9)


def test_delay_sources_sum_to_total_delay():
    # the breakdown is exact: ground_delay + air_hold + detour_time == total_delay (no residual)
    res = run(SimConfig(planner="straight", lam_per_hour=120.0, horizon_s=1200.0, seed=2))
    acc = metrics.flight_frame(res).query("accepted")
    recombined = acc["ground_delay_s"] + acc["air_hold_s"] + acc["detour_time_s"]
    assert ((recombined - acc["total_delay_s"]).abs() < 1e-9).all()


def test_total_delay_is_nan_for_denied():
    # a fully-blocked straight flight is denied → no arrival → NaN delay (not a misleading 0)
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner import get_planner

    led = ReservationLedger(CFG)
    led.commit(99, [Volume4D(box_from_segment(vec(1000, -300, 150), vec(1000, 300, 150), 40, 400),
                             0.0, 1e6)])
    denied = get_planner("straight").plan(FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0), led, CFG)
    assert not denied.accepted
    assert math.isnan(metrics.total_delay_s(denied, CFG))


def test_solve_time_is_recorded_per_flight_and_aggregated():
    res = run(SimConfig(planner="straight", lam_per_hour=60.0, horizon_s=1200.0, seed=2))
    df = metrics.flight_frame(res)
    assert "solve_time_s" in df.columns
    assert (df["solve_time_s"] >= 0).all() and df["solve_time_s"].sum() > 0  # real wall time
    agg = metrics.aggregate(res)
    for k in ("mean_solve_time_s", "p95_solve_time_s", "max_solve_time_s", "total_solve_time_s"):
        assert agg[k] >= 0.0
    assert agg["max_solve_time_s"] >= agg["mean_solve_time_s"]
    # aggregate covers ALL flights (denials too), so total ≈ sum of the per-flight column
    assert math.isclose(agg["total_solve_time_s"], float(df["solve_time_s"].sum()), rel_tol=1e-9)


def test_congestion_rises_with_demand():
    # the headline claim: denser demand reserves more airspace and waits longer
    lo = metrics.aggregate(run(SimConfig(planner="straight", lam_per_hour=40.0,
                                         horizon_s=1800.0, seed=1)))
    hi = metrics.aggregate(run(SimConfig(planner="straight", lam_per_hour=200.0,
                                         horizon_s=1800.0, seed=1)))
    assert hi["reserved_vol_m3_s"] > lo["reserved_vol_m3_s"]
    assert hi["mean_ground_delay_s"] >= lo["mean_ground_delay_s"]


# --- per-USS slicing -------------------------------------------------------------------------

from freespace_sim.demand import UniformPoissonDemand   # noqa: E402


def _two_uss_run():
    cfg = SimConfig(planner="straight", lam_per_hour=120.0, horizon_s=1800.0, seed=4,
                    region_size_m=(5000.0, 5000.0))
    return run(cfg, demand=UniformPoissonDemand(uss_ids=("walmart", "stripmall")))


def test_per_uss_frame_one_row_per_uss():
    res = _two_uss_run()
    pu = metrics.per_uss_frame(res)
    assert set(pu["uss_id"]) == {"walmart", "stripmall"}
    assert len(pu) == 2


def test_per_uss_counts_sum_to_overall():
    res = _two_uss_run()
    pu = metrics.per_uss_frame(res)
    agg = metrics.aggregate(res)
    assert int(pu["n_requests"].sum()) == agg["n_requests"]
    assert int(pu["n_accepted"].sum()) == agg["n_accepted"]
    assert int(pu["n_denied"].sum()) == agg["n_denied"]
    assert math.isclose(float(pu["share_of_accepted"].sum()), 1.0, rel_tol=1e-9)


def test_per_uss_reserved_volume_sums_to_overall():
    res = _two_uss_run()
    pu = metrics.per_uss_frame(res)
    agg = metrics.aggregate(res)
    assert math.isclose(float(pu["reserved_vol_m3_s"].sum()), agg["reserved_vol_m3_s"], rel_tol=1e-9)
    # each operator's utilization is its share of the whole sky → they sum to the overall
    assert math.isclose(float(pu["airspace_utilization"].sum()), agg["airspace_utilization"], rel_tol=1e-9)


def test_aggregate_reports_n_uss_and_spreads():
    res = _two_uss_run()
    agg = metrics.aggregate(res)
    assert agg["n_uss"] == 2
    assert agg["denial_rate_spread"] >= 0.0
    assert agg["mean_delay_spread"] >= 0.0


def test_cross_uss_spread_zero_for_single_uss():
    agg = metrics.aggregate(run(SimConfig(planner="straight", lam_per_hour=60.0, horizon_s=1200.0, seed=1)))
    assert agg["n_uss"] == 1
    assert agg["denial_rate_spread"] == 0.0
    assert agg["mean_delay_spread"] == 0.0
