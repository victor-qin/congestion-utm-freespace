import math

import numpy as np
import pytest

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


def test_bare_flight_row_does_not_clamp_at_the_horizon():
    """``flight_row``'s ``window=None`` must mean UNCLAMPED, matching what ``flight_frame`` measures.

    It used to mean ``[0, horizon_s]``. Since ``flight_frame`` switched to ``simulation_window``,
    that default made ``flight_row`` the last surface still truncating the post-horizon return tail —
    two public entry points in this module reporting different ``reserved_vol_m3_s`` for one flight.
    """
    cfg = SimConfig(horizon_s=1800.0, planner="straight")
    # a return leg reserving [1700, 2300): 100 s inside the horizon, 500 s past it
    box = box_from_segment(vec(0, 0, 100), vec(100, 0, 100), 40, 120)
    vol = Volume4D(box, 1700.0, 2300.0)
    unclamped = metrics.reserved_volume_seconds([vol], -math.inf, math.inf)
    at_horizon = metrics.reserved_volume_seconds([vol], 0.0, cfg.horizon_s)
    assert unclamped == pytest.approx(6.0 * at_horizon)   # 600 s of reservation vs the 100 s pre-horizon

    res = run(cfg, requests=[FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0)])
    intent = res.intents[0]
    bare = metrics.flight_row(intent, cfg)["reserved_vol_m3_s"]
    framed = metrics.flight_frame(res).iloc[0]["reserved_vol_m3_s"]
    assert bare == pytest.approx(framed)


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


def _congested_multilevel():
    """A dense A* run on the 3-level ladder where BOTH the climb and lattice levers actually fire.

    The straight/single-plane run used above can't exercise either — it has no lattice and no ladder
    to climb — which is how the missing-climb-band bug survived. Asserted non-vacuous below.
    """
    cfg = SimConfig(planner="astar", lam_per_hour=900.0, horizon_s=600.0,
                    region_size_m=(2500.0, 2500.0), seed=3)
    acc = metrics.flight_frame(run(cfg)).query("accepted")
    assert (acc["excess_altitude_m"] > 0).any(), "no traffic-forced climb — climb-band check is vacuous"
    assert (acc["lattice_overhead_m"] > 0).any(), "no hex staircase — lattice-band check is vacuous"
    return acc


def test_delay_sources_sum_to_total_delay_with_climb_and_lattice():
    # Same exactness claim, but on a run where the climb lever is NON-ZERO: dropping the altitude term
    # (as the chart did) leaves a shortfall that grows with load, and no single-plane run can catch it.
    acc = _congested_multilevel()
    recombined = (acc["ground_delay_s"] + acc["air_hold_s"] + acc["detour_time_s"]
                  + acc["altitude_delay_phys_s"])
    assert ((recombined - acc["total_delay_s"]).abs() < 1e-9).all()


def test_detour_time_splits_exactly_into_lattice_and_traffic():
    # the split is a partition of detour_time_s, not an independent estimate — it must re-sum exactly,
    # and neither half may go negative (the lattice share is clamped to what air_detour_m can carry).
    acc = _congested_multilevel()
    assert (((acc["detour_lattice_s"] + acc["detour_traffic_s"]) - acc["detour_time_s"]).abs()
            < 1e-9).all()
    assert (acc["detour_lattice_s"] >= 0).all() and (acc["detour_traffic_s"] >= 0).all()


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


@pytest.mark.parametrize("planner", ("sipp", "sipp_ref", "sipp_shortcut"))
def test_sipp_family_uses_lattice_floor_for_altitude_metrics(planner):
    """Every SIPP registry variant shares A*'s flight-level ladder and therefore its 30 m baseline."""
    cfg = SimConfig(planner=planner)
    floor = cfg.flight_levels_m[0]
    assert metrics.nominal_altitude_change_m(cfg) == 2.0 * floor
    assert metrics.nominal_flight_time_s(2400.0, cfg) == pytest.approx(
        2400.0 / cfg.nominal_speed_mps + 2.0 * cfg.climb_time_to(floor)
    )

    intent = _accepted(altitude_change_m=2.0 * cfg.level_z(1), planner="sipp")
    row = metrics.flight_row(intent, cfg)
    excess = 2.0 * (cfg.level_z(1) - floor)
    assert row["excess_altitude_m"] == pytest.approx(excess)
    assert row["total_delay_s"] == pytest.approx(excess / cfg.climb_rate_mps)
    assert row["congestion_cost"] == pytest.approx(excess * cfg.cost_altitude_change_per_m)


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
    intent = _accepted(altitude_change_m=2.0 * sp.cruise_level_m)       # 2·70 = 140 (cruise round trip)
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
    # no accepted flights → there is no realized flight timeline
    assert metrics.steady_state_window(_synthetic_result([], 1000.0)) == (0.0, 0.0)
    # accepted but with no geometry (no volumes/centerline) → no realized flight timeline
    ghost = OperationalIntent(FlightRequest(0, vec(0, 0, 0), vec(1, 0, 0), 0.0), IntentStatus.ACCEPTED)
    assert metrics.steady_state_window(_synthetic_result([ghost], 1000.0)) == (0.0, 0.0)


def test_all_denied_run_reports_its_denial_rate_in_the_steady_block():
    """A zero-width window must not publish a block of structural zeros.

    Nothing accepted ⇒ simulation_window is (0, 0) ⇒ a windowed cohort is empty ⇒ denial_rate is
    0/max(1,0) == 0.0, i.e. a saturated run archives byte-identically to a flawless one. The steady
    twin degenerates to the whole-run view instead, which is what the deleted fallbacks guaranteed.
    """
    denied = [OperationalIntent(FlightRequest(i, vec(0, 0, 0), vec(1, 0, 0), float(i)),
                                IntentStatus.REJECTED) for i in range(5)]
    agg = metrics.aggregate_with_steady(_synthetic_result(denied, 600.0))
    assert agg["denial_rate"] == 1.0
    steady = agg["steady_state"]
    assert steady["n_requests"] == 5
    assert steady["denial_rate"] == 1.0          # NOT 0.0
    assert (steady["window_lo"], steady["window_hi"]) == (0.0, 0.0)   # provenance still recorded


def test_steady_block_falls_back_when_a_positive_window_selects_no_takeoffs():
    """A NON-degenerate steady window can still select an empty cohort — guard on the cohort, not width.

    All 30 flights take off in [0, 29] but stay airborne to 2000, so the concurrency plateau (and thus
    steady_state_window) centers hundreds of seconds later, around (808, 1208), where no flight's
    occupancy clock falls. Guarding only zero-width windows (win[1] <= win[0]) would measure over this
    positive-but-empty window and republish structural zeros; the cohort must fall back to the whole
    run. This is reachable on any run whose trips outlast the takeoff burst, not just synthetic ones.
    """
    flights = [_accepted_over(k, float(k), 2000.0) for k in range(30)]
    res = _synthetic_result(flights, 3000.0)

    lo, hi = metrics.steady_state_window(res)
    assert hi > lo, "fixture no longer yields a positive window — the check would be vacuous"
    assert len(metrics.flight_frame(res, window=(lo, hi))) == 0, "fixture no longer empties the cohort"

    steady = metrics.aggregate_with_steady(res)["steady_state"]
    assert steady["n_requests"] == len(flights)              # fell back to the whole-run cohort, not 0
    assert steady["n_accepted"] == len(flights)              # ...with its real (non-zero) numbers
    assert (steady["window_lo"], steady["window_hi"]) == (lo, hi)   # window provenance preserved


def test_window_membership_uses_the_occupancy_clock_not_the_filing_clock():
    """Cohort selection and the window itself must be on one clock.

    ``steady_state_window`` is derived from airborne density; filtering by ``t_request`` against it is
    only correct when flights depart on filing. Under a scheduling lead the two clocks can be fully
    disjoint (measured: t_request [0, 2823] vs t_occupancy [2892, 3780] on a density run), so the
    filing-time cohort was empty and every steady metric published zeros.
    """
    lead = 500.0        # every flight files at t=0 but takes off ~lead later
    flights = [_accepted_over(k, lead + 4.0 * k, lead + 4.0 * k + 80.0) for k in range(30)]
    for f in flights:
        object.__setattr__(f.request, "t_request", 0.0)
    res = _synthetic_result(flights, 2000.0)

    df_all = metrics.flight_frame(res)
    assert "t_occupancy" in df_all.columns, "flights.parquet must carry the clock readouts filter on"
    assert df_all["t_occupancy"].min() >= lead      # airborne clock, not the all-zero filing clock

    lo, hi = metrics.steady_state_window(res)
    assert hi > lo
    sub = metrics.flight_frame(res, window=(lo, hi))
    assert len(sub) > 0, "airborne-derived window selected nobody — clocks are crossed again"
    assert len(sub) < len(df_all)                  # and it is a real subset, not everything


def _open_ended_ghost(t_end: float) -> OperationalIntent:
    """One accepted intent holding a reservation to ``t_end`` — forces the adaptive grid to coarsen."""
    from freespace_sim.geometry import box_from_segment
    box = box_from_segment(vec(0, 0, 75), vec(60, 0, 75), 60, 30)
    return OperationalIntent(FlightRequest(0, vec(0, 0, 0), vec(60, 0, 0), 0.0), IntentStatus.ACCEPTED,
                             volumes=[Volume4D(box, 0.0, t_end)])


def test_density_grid_spans_open_ended_volume_without_unbounded_allocation():
    # A pathological hand-built open-ended fixture is fully represented by coarsening, not truncating:
    # density runs land return traffic AFTER the horizon, so the grid must not clip at 4x horizon.
    # t_end must exceed 250_000 x dt to actually EXERCISE coarsening — at 1e6/dt=4 the grid step is
    # max(4.0, 4.0), i.e. the cap is never reached and the coarsening branch is a no-op.
    res = _synthetic_result([_open_ended_ghost(1e7)], 1000.0)
    t, _ = metrics.density_timeseries(res, dt=4.0)
    assert t[-1] >= 1e7                                 # complete time range, no horizon cutoff
    assert len(t) <= 250_002                            # adaptive grid remains bounded in memory
    assert float(t[1] - t[0]) == pytest.approx(40.0)    # coarsened 10x from dt=4.0 (1e7 / 250_000)


def test_steady_state_smoothing_kernel_is_measured_against_the_coarsened_grid():
    """``smooth_s`` is seconds, so the kernel width must divide by the grid step actually returned.

    ``density_timeseries`` coarsens to ``max(dt, span/250k)``. Dividing ``smooth_s`` by the *requested*
    ``dt`` instead makes the kernel ``grid_dt/dt`` times too wide — 10x on this fixture — which smears
    the plateau across the ramps and mislocates the window.
    """
    step, dur, n = 4.0, 80.0, 50          # plateau of full overlap is [80, 196]
    flights = [_accepted_over(k, k * step, k * step + dur) for k in range(n)]
    res = _synthetic_result([*flights, _open_ended_ghost(1e7)], 1000.0)
    lo, hi = metrics.steady_state_window(res, frac=0.9, dt=step, smooth_s=dur)
    # True full-overlap plateau is [80, 196]; grid_dt is 40 s, so allow one bin of slack each side.
    # Dividing by dt instead of grid_dt yields [0, 440] — the window runs off the front of the ramp.
    assert (lo, hi) == pytest.approx((80.0, 196.0), abs=40.0), f"plateau mislocated: got [{lo}, {hi}]"


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


class _DenyAll:
    """A warm planner that always denies — see :func:`_hub_flight`."""

    def plan(self, req, ledger, cfg):
        from freespace_sim.types import DenialReason, IntentStatus, OperationalIntent

        return OperationalIntent(request=req, status=IntentStatus.REJECTED,
                                 denial_reason=DenialReason.BUDGET_EXCEEDED, planner="deny")


def _hub_flight(planner, radius=180.0, levels=(100.0,)):
    """An unimpeded flight between two hubs of ``radius`` on the given flight-level ladder.

    ``get_planner("milp")`` returns MILPOptPlanner, whose ``plan`` yields the CHEAPER of its warm
    StraightLineTimeShift candidate and its own solve — and in empty airspace the warm one wins. That
    made the ``milp`` arm a byte-identical duplicate of the ``straight`` arm (same 46-point unfolded
    centerline, same 0.0 detour) which never entered milp.py at all. Deny the warm start so the MILP's
    own folded path comes back. Same trap, same fix as ``test_astar._folded_planner``.
    """
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner import get_planner
    from freespace_sim.planner.milp import MILPOptPlanner
    from freespace_sim.types import Terminal

    cfg = SimConfig(flight_levels_m=levels, airspace_ceiling_m=max(levels) + 25.0,
                    region_size_m=(20_000.0, 20_000.0), terminal_radius_m=radius)
    hub = Terminal("hub#0", 8, radius)
    req = FlightRequest(1, vec(0, 0, 0), vec(5000, 2000, 0), 0.0,
                        origin_terminal=hub, dest_terminal=hub)
    resolved = MILPOptPlanner(warm_planner=_DenyAll()) if planner == "milp" else get_planner(planner)
    intent = resolved.plan(req, ReservationLedger(cfg), cfg)
    assert intent.accepted
    return cfg, req, intent


def test_enroute_metrics_exclude_terminal_airspace():
    """Both sides of ``stretch`` span exit lane -> exit lane, never hub centre -> hub centre.

    Flying inside a terminal column is terminal operations: it consumes that hub's capacity and is
    neither en-route distance nor delay. Measuring against the centres books the two unreserved column
    legs as avoidable detour, which drove ``stretch`` BELOW 1 once a refiner removed the hex staircase
    that had been masking it (issue #50).
    """
    from freespace_sim.volumes import enroute_reference_m

    cfg, req, intent = _hub_flight("astar")
    row = metrics.flight_row(intent, cfg)
    centre_to_centre = float(np.linalg.norm(
        np.asarray(req.dest, float)[:2] - np.asarray(req.origin, float)[:2]))
    ref = enroute_reference_m(req.origin, req.dest, req.origin_terminal, req.dest_terminal, cfg)
    # guard the guard: two 180 m hubs must actually shorten the baseline, else this proves nothing
    assert ref < centre_to_centre - 100.0, "baseline is not lane->lane"
    assert math.isclose(row["straight_line_m"], ref, rel_tol=1e-9)
    assert row["stretch"] >= 1.0 - 1e-9
    # Pin the FLOWN side too — asserting only the reference let a full revert of enroute_flown_m pass,
    # because the A* staircase is larger than the column legs so stretch stayed above 1 either way.
    # Use `straight`, whose centerline runs hub centre -> hub centre unfolded: the excluded length is
    # then EXACTLY the two column legs, 2 x exit_radius. (For A* the fold instead EXTENDS the path out
    # to the column edge, since its centerline already starts on a boundary lane cell — so the sign of
    # flown - raw is planner-dependent and only this case pins the exclusion exactly.)
    from freespace_sim.volumes import exit_radius

    cfg_s, req_s, intent_s = _hub_flight("straight")
    raw = float(np.linalg.norm(
        np.diff(np.asarray([np.asarray(p, float)[:2] for p, _ in intent_s.centerline]), axis=0),
        axis=1).sum())
    legs = 2.0 * exit_radius(req_s.origin_terminal, cfg_s)
    assert math.isclose(metrics.flight_row(intent_s, cfg_s)["flown_m"], raw - legs, abs_tol=1e-6), \
        "flown length does not exclude exactly the two terminal column legs"


@pytest.mark.parametrize("planner", ["astar", "astar_shortcut", "milp", "straight"])
def test_planner_detour_and_metrics_stretch_cannot_drift(planner):
    """``air_detour_m`` (planner) and ``flown_m - straight_line_m`` (metrics) must agree EXACTLY.

    They are computed in different layers off different inputs, so they drifted before: the planners
    measured the raw reserved centerline while metrics measured it rooted at the column edges. Both now
    go through ``volumes.enroute_flown_m``. A continuous planner flies the ideal line and must read
    exactly 0 detour / 1.0 stretch — if it does not, the two rulers have diverged again.
    """
    cfg, _req, intent = _hub_flight(planner)
    row = metrics.flight_row(intent, cfg)
    assert math.isclose(intent.air_detour_m, row["flown_m"] - row["straight_line_m"], abs_tol=1e-6)
    assert row["stretch"] >= 1.0 - 1e-9
    if planner == "straight":
        assert math.isclose(row["stretch"], 1.0, abs_tol=1e-9), "straight flies the ideal line exactly"
    if planner == "milp":
        # a REAL milp intent (denied warm start) discretizes into knots, so it lands just off the
        # ideal line rather than exactly on it — but it must not carry the column legs
        assert row["stretch"] < 1.01, f"milp stretch {row['stretch']:.4f} — column legs leaking in?"


@pytest.mark.parametrize("planner", ["astar", "milp", "straight"])
def test_overlapping_terminal_columns_book_no_enroute_detour(planner):
    """A flight with no en-route segment must not have its whole path booked as detour.

    ``enroute_reference_m`` clamps to 0 when two column radii cover the whole trip, and
    ``max(0.0, flown - 0.0)`` then charged the ENTIRE flown path as ``air_detour_m`` — real seconds
    and real cost — while the ``max_detour_factor`` gate was skipped by its own ``> _EPS`` guard.
    Reachable through a shipped flag: ``--corridor-overlap -100`` (``exit_radius`` documents negative
    overlap as "leaves a clearance gap"), which pushes exit_radius to 310 m.
    """
    from freespace_sim.ledger import ReservationLedger
    from freespace_sim.planner import get_planner
    from freespace_sim.types import Terminal
    from freespace_sim.volumes import enroute_reference_m

    cfg = SimConfig(flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
                    region_size_m=(20_000.0, 20_000.0), terminal_radius_m=180.0,
                    max_detour_factor=1.2)
    hub = Terminal("hub#0", 8, 180.0, -100.0)          # exit_radius 310 m > the 250 m trip
    req = FlightRequest(1, vec(5000, 5000, 0), vec(5000, 5250, 0), 0.0, origin_terminal=hub)
    # guard the guard: this configuration must actually reach the clamp
    assert enroute_reference_m(req.origin, req.dest, hub, None, cfg) == 0.0, "clamp not reached"

    intent = get_planner(planner).plan(req, ReservationLedger(cfg), cfg)
    if not intent.accepted:
        return                                          # denying is fine; over-charging is not
    assert intent.air_detour_m == 0.0, (
        f"{planner}: booked {intent.air_detour_m:.1f} m of detour on a flight with no en-route "
        "segment — the whole path was charged")
    assert math.isnan(metrics.flight_row(intent, cfg)["stretch"]), "stretch must be undefined here"
