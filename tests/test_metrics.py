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
    # an open-ended reservation (t_end ~ 1e6) must be clamped to the window, not run away
    box = BoxSpec(center=(0, 0, 0), rot=tuple(np.eye(3).flatten()), extents=(1.0, 1.0, 1.0))
    vol = Volume4D(box, 0.0, 1e6)
    got = metrics.reserved_volume_seconds([vol], 0.0, 100.0)
    assert math.isclose(got, 1.0 * 100.0, rel_tol=1e-9)   # 1 m³ × 100 s, not 1e6 s
    # a sub-window clamps on both ends: the reservation's [20, 50) slice → 30 s
    assert math.isclose(metrics.reserved_volume_seconds([vol], 20.0, 50.0), 30.0, rel_tol=1e-9)


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
    # total_delay_s = ground_hold + air_loiter + detour-time + traffic-forced climb; excludes the
    # mandatory climb. (Single-plane 'straight' has no altitude lever ⇒ that 4th term is 0.)
    res = run(SimConfig(planner="straight"),
              requests=[FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0),
                        FlightRequest(1, vec(0, 0, 0), vec(2400, 0, 0), 0.0)])
    df = metrics.flight_frame(res)
    for _, r in df[df["accepted"]].iterrows():
        expected = (r["ground_delay_s"] + r["air_hold_s"] + r["air_detour_m"] / CFG.nominal_speed_mps
                    + r["altitude_delay_phys_s"])
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
    # the breakdown is exact: ground_delay + air_hold + detour_time + altitude_delay_phys == total_delay
    res = run(SimConfig(planner="straight", lam_per_hour=120.0, horizon_s=1200.0, seed=2))
    acc = metrics.flight_frame(res).query("accepted")
    recombined = (acc["ground_delay_s"] + acc["air_hold_s"] + acc["detour_time_s"]
                  + acc["altitude_delay_phys_s"])
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


# --- cost ⇄ time dual decomposition (the vertical-flight / cost-transparency metrics) ----------

from freespace_sim.cost import trajectory_cost                       # noqa: E402
from freespace_sim.types import IntentStatus, OperationalIntent      # noqa: E402


def _accepted(**kw):
    """Synthesize an accepted intent with `cost` set exactly as a planner would (trajectory_cost). The
    excess-altitude baseline keys on cfg.planner (the run), not intent.planner, so a test passes the cfg
    whose planner it means — CFG (astar → ladder floor) or SimConfig(planner='straight') (cruise)."""
    intent = OperationalIntent(FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0),
                               IntentStatus.ACCEPTED, **kw)
    intent.cost = trajectory_cost(intent, CFG)
    return intent


def test_cost_breakdown_reconciles_to_intent_cost():
    # the four COST levers sum to exactly intent.cost (== trajectory_cost) for every accepted flight
    res = run(SimConfig(planner="straight", lam_per_hour=120.0, horizon_s=1200.0, seed=2))
    acc = metrics.flight_frame(res).query("accepted")
    recombined = (acc["ground_delay_cost"] + acc["air_hold_cost"]
                  + acc["air_detour_cost"] + acc["altitude_cost"])
    assert ((recombined - acc["cost"]).abs() < 1e-6).all()


def test_cost_and_time_diverge_by_their_weights():
    # the headline of "exposing costs AND their real-time": a lever's COST and its TIME differ by exactly
    # its weight — that gap is why we record both currencies.
    intent = _accepted(air_hold_s=10.0, air_detour_m=300.0)
    cb, db = metrics.cost_breakdown(intent, CFG), metrics.delay_breakdown_s(intent, CFG)
    # air hold: costs 3×/s but is one real second per second
    assert math.isclose(cb["air_hold_cost"], CFG.cost_air_hold_per_s * db["air_hold_s"])
    assert math.isclose(db["air_hold_s"], 10.0)
    # lateral: a detour metre costs c_lat but is 1/speed real seconds
    assert math.isclose(cb["air_detour_cost"], CFG.cost_air_lateral_per_m * 300.0)
    assert math.isclose(db["detour_time_s"], 300.0 / CFG.nominal_speed_mps)


def test_altitude_recorded_as_cost_and_both_time_readings():
    # a flight pushed up to level 1 (z=70): altitude_change 140, floor 60 ⇒ 80 m of congestion climb
    intent = _accepted(altitude_change_m=2.0 * 70.0)
    cb, db = metrics.cost_breakdown(intent, CFG), metrics.delay_breakdown_s(intent, CFG)
    assert math.isclose(cb["altitude_cost"], CFG.cost_altitude_change_per_m * 140.0)        # FULL climb
    assert math.isclose(db["excess_altitude_m"], 80.0)                                       # above floor
    # A (physical) and B (cost-equivalent) — BOTH recorded, and genuinely different (12× at defaults)
    assert math.isclose(db["altitude_delay_phys_s"], 80.0 / CFG.climb_rate_mps)              # ≈13.3 s
    assert math.isclose(db["altitude_delay_costeq_s"],
                        80.0 * CFG.cost_altitude_change_per_m / CFG.cost_ground_delay_per_s)  # 160 s
    assert db["altitude_delay_costeq_s"] > db["altitude_delay_phys_s"]


def test_excess_altitude_is_zero_at_the_floor():
    # multi-level A* in empty airspace cruises at the lowest level ⇒ no congestion climb, no alt delay
    res = run(SimConfig(planner="astar"),
              requests=[FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0)])
    r = metrics.flight_frame(res).iloc[0]
    assert r["accepted"]
    assert math.isclose(r["altitude_change_m"], metrics.nominal_altitude_change_m(res.config))
    assert math.isclose(r["excess_altitude_m"], 0.0, abs_tol=1e-9)
    assert math.isclose(r["altitude_delay_phys_s"], 0.0, abs_tol=1e-9)
    assert math.isclose(r["altitude_delay_costeq_s"], 0.0, abs_tol=1e-9)


def test_congestion_cost_is_cost_minus_the_mandatory_climb():
    # congestion_cost omits the mandatory floor climb every flight pays (which `cost` carries in full),
    # so it == cost − c_alt·nominal_altitude_change. Detour-robust: holds whatever the hex path does.
    res = run(SimConfig(planner="astar"),
              requests=[FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0)])
    r = metrics.flight_frame(res).iloc[0]
    baseline = res.config.cost_altitude_change_per_m * metrics.nominal_altitude_change_m(res.config)
    assert math.isclose(r["congestion_cost"], r["cost"] - baseline, abs_tol=1e-6)
    assert r["cost"] > 0.0      # still paid the mandatory baseline-altitude cost


def test_breakdowns_are_nan_for_denied():
    denied = OperationalIntent(FlightRequest(1, vec(0, 0, 0), vec(2400, 0, 0), 0.0),
                               IntentStatus.REJECTED)
    assert not denied.accepted
    assert all(math.isnan(v) for v in metrics.cost_breakdown(denied, CFG).values())
    assert all(math.isnan(v) for v in metrics.delay_breakdown_s(denied, CFG).values())


def test_aggregate_exposes_cost_and_altitude_rollups():
    res = run(SimConfig(planner="straight", lam_per_hour=120.0, horizon_s=1200.0, seed=2))
    agg = metrics.aggregate(res)
    for k in ("mean_ground_delay_cost", "mean_air_hold_cost", "mean_air_detour_cost",
              "mean_altitude_cost", "mean_congestion_cost", "mean_excess_altitude_m",
              "p95_excess_altitude_m", "mean_altitude_delay_phys_s", "mean_altitude_delay_costeq_s"):
        assert k in agg and agg[k] >= 0.0


def test_single_plane_planner_reads_no_altitude_congestion():
    # Baseline fix: keyed on cfg.planner, a single-plane run (cruise_level_m, no altitude lever) reads
    # ZERO excess altitude — no spurious congestion floor in empty airspace.
    sp = SimConfig(planner="straight")
    intent = _accepted(altitude_change_m=2.0 * sp.cruise_level_m)       # 2·75 = 150 (cruise round trip)
    db = metrics.delay_breakdown_s(intent, sp)
    assert db["excess_altitude_m"] == 0.0
    assert db["altitude_delay_phys_s"] == 0.0
    assert metrics.total_delay_s(intent, sp) == 0.0                     # no ground/hold/detour either


def test_total_delay_counts_a_traffic_forced_climb_like_congestion_cost():
    # Twin fix: on an A* run, a flight pushed up one level purely by traffic (no ground hold / detour) is
    # NOT zero-congestion — total_delay_s counts the climb TIME; congestion_cost counts the same COST.
    intent = _accepted(altitude_change_m=2.0 * CFG.level_z(1))          # CFG is astar → cruised at level 1
    excess = 2.0 * CFG.level_z(1) - metrics.nominal_altitude_change_m(CFG)   # 140 − 60 = 80 m
    td = metrics.total_delay_s(intent, CFG)
    assert td > 0.0 and math.isclose(td, excess / CFG.climb_rate_mps)   # the vertical lever, in time
    assert math.isclose(metrics.flight_row(intent, CFG)["congestion_cost"],
                        excess * CFG.cost_altitude_change_per_m)         # ... and in cost


# --- steady-state measurement window (issue #25) -----------------------------------------------

from freespace_sim.sim import SimResult   # noqa: E402


def _accepted_over(fid, t0, t1):
    """A minimal accepted intent airborne over [t0, t1] via a 2-point centerline (no volumes → the
    density helpers fall back to the centerline span)."""
    cl = [(vec(0, 0, 75), float(t0)), (vec(100, 0, 75), float(t1))]
    return OperationalIntent(FlightRequest(fid, vec(0, 0, 0), vec(100, 0, 0), 0.0),
                             IntentStatus.ACCEPTED, centerline=cl)


def _synthetic_result(intents, horizon_s=1000.0):
    return SimResult(config=SimConfig(planner="straight", horizon_s=horizon_s),
                     intents=intents, ledger=None, verified=True)


def test_widest_hot_run_picks_widest_earliest():
    hot = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)   # runs [1,2], [4,7], [9,9]
    assert metrics._widest_hot_run(hot) == (4, 7)
    assert metrics._widest_hot_run(np.zeros(5, dtype=bool)) is None
    assert metrics._widest_hot_run(np.array([1, 1, 0, 1, 1], dtype=bool)) == (0, 1)  # earliest on ties


def test_density_timeseries_counts_concurrency():
    intents = [_accepted_over(0, 0, 100), _accepted_over(1, 50, 150), _accepted_over(2, 60, 70)]
    t, d = metrics.density_timeseries(_synthetic_result(intents), dt=1.0)
    assert d.max() == 3.0        # all three overlap around t ∈ [60, 70)
    assert d[10] == 1.0          # at t=10 only flight 0 is airborne
    # nothing flew → safe, non-empty degenerate arrays
    _, d0 = metrics.density_timeseries(_synthetic_result([]), dt=1.0)
    assert d0.max() == 0.0


def test_steady_state_window_recovers_trapezoid_plateau():
    # flight k airborne over [4k, 4k+80] → concurrency trapezoid; the full-overlap plateau is [80, 196]
    step, dur, n = 4.0, 80.0, 50
    res = _synthetic_result([_accepted_over(k, k * step, k * step + dur) for k in range(n)])
    # frac=1 on the raw density → exactly the flat top (smooth_s=0 keeps the trapezoid crisp)
    lo, hi = metrics.steady_state_window(res, frac=1.0, dt=step, smooth_s=0.0)
    assert lo > 0.0 and hi < (n - 1) * step + dur                  # both ramp tails trimmed
    assert lo <= dur + step and hi >= (n - 1) * step - step        # recovers the plateau within dt


def test_steady_state_window_falls_back_when_degenerate():
    # no accepted flights → the whole horizon (steady == whole-run)
    assert metrics.steady_state_window(_synthetic_result([], 1000.0)) == (0.0, 1000.0)
    # accepted but with no geometry (no volumes/centerline) → empty density → whole horizon
    ghost = OperationalIntent(FlightRequest(0, vec(0, 0, 0), vec(1, 0, 0), 0.0), IntentStatus.ACCEPTED)
    assert metrics.steady_state_window(_synthetic_result([ghost], 1000.0)) == (0.0, 1000.0)


def test_density_grid_bounded_against_open_ended_volume():
    # a hand-built accepted intent carrying an open-ended (t_end ~ 1e6) volume must not blow up the grid
    from freespace_sim.geometry import box_from_segment
    box = box_from_segment(vec(0, 0, 75), vec(60, 0, 75), 60, 30)
    ghost = OperationalIntent(FlightRequest(0, vec(0, 0, 0), vec(60, 0, 0), 0.0), IntentStatus.ACCEPTED,
                              volumes=[Volume4D(box, 0.0, 1e6)])
    res = _synthetic_result([ghost], 1000.0)
    t, _ = metrics.density_timeseries(res, dt=4.0)
    assert t[-1] <= 4.0 * res.config.horizon_s + 4.0   # capped at ~4×horizon, not 1e6


def test_windowed_aggregate_uses_window_duration_and_records_provenance():
    res = run(SimConfig(planner="straight", lam_per_hour=200.0, horizon_s=1800.0, seed=3))
    full = metrics.aggregate(res)
    lo, hi = win = metrics.steady_state_window(res)
    w = metrics.aggregate(res, window=win)
    # provenance on the windowed view, absent on the full one
    assert (w["window_lo"], w["window_hi"]) == (lo, hi)
    assert "window_lo" not in full and "window_hi" not in full
    # rate denominators divide by the WINDOW duration, not the horizon
    dur_h = (hi - lo) / 3600.0
    assert math.isclose(w["offered_load_per_h"], w["n_requests"] / dur_h, rel_tol=1e-9)
    assert math.isclose(w["throughput_per_h"], w["n_accepted"] / dur_h, rel_tol=1e-9)
    # airspace_utilization stays a valid fraction under the window
    assert 0.0 <= w["airspace_utilization"] <= 1.0


def test_window_drops_post_horizon_filed_flights():
    # a return flight filed AFTER the horizon (t_request > H) is windowed out by filing-time membership —
    # the principled replacement for the removed clip_returns_to_horizon demand hack
    H = 600.0
    reqs = [FlightRequest(0, vec(0, 0, 0), vec(1500, 0, 0), 10.0),
            FlightRequest(1, vec(0, 0, 0), vec(1500, 0, 0), H + 300.0)]   # filed past the horizon
    res = run(SimConfig(planner="straight", horizon_s=H, region_size_m=(2000.0, 2000.0)), requests=reqs)
    assert len(metrics.flight_frame(res)) == 2                       # whole run keeps both
    win = metrics.flight_frame(res, window=(0.0, H))
    assert set(win["flight_id"]) == {0}                             # windowed set drops the late one


def test_aggregate_with_steady_reports_both_views():
    res = run(SimConfig(planner="straight", lam_per_hour=150.0, horizon_s=1800.0, seed=1))
    agg = metrics.aggregate_with_steady(res)
    assert "steady_state" in agg
    st = agg["steady_state"]
    assert "window_lo" in st and "window_hi" in st
    # run-identity keys live only at the top level, not duplicated in the block
    for k in ("lam_per_hour", "seed", "planner", "n_uss", "verified"):
        assert k in agg and k not in st
    # the top-level (whole-run) numbers are unchanged vs a plain aggregate
    full = metrics.aggregate(res)
    assert agg["mean_total_delay_s"] == full["mean_total_delay_s"]
    assert agg["n_requests"] == full["n_requests"]
