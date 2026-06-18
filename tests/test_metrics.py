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
