from freespace_sim.config import SimConfig
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec


def _two_identical():
    return [
        FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0),
        FlightRequest(1, vec(0, 0, 0), vec(2400, 0, 0), 0.0),
    ]


def test_single_flight_accepted_no_delay():
    cfg = SimConfig(planner="straight")
    res = run(cfg, requests=[FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0)])
    assert res.verified
    assert len(res.accepted) == 1
    assert res.accepted[0].ground_delay_s == 0.0


def test_two_crossing_flights_second_yields_and_invariant_holds():
    cfg = SimConfig(planner="straight")
    res = run(cfg, requests=_two_identical())
    assert res.verified                      # core ASTM invariant: no inter-flight conflict
    assert len(res.accepted) == 2
    delays = sorted(i.ground_delay_s for i in res.accepted)
    assert delays[0] == 0.0 and delays[1] > 0.0   # FCFS: first holds airspace, second waits


def test_end_to_end_demand_run_is_consistent():
    # generated Poisson demand should always produce a verified (conflict-free) commit set
    cfg = SimConfig(planner="straight", lam_per_hour=50.0, horizon_s=3600.0, seed=1)
    res = run(cfg)
    assert res.verified
    s = res.summary()
    assert s["n_accepted"] + s["n_denied"] == s["n_requests"]


def test_summary_reports_no_denials_for_two_flights():
    cfg = SimConfig(planner="straight")
    res = run(cfg, requests=_two_identical())
    s = res.summary()
    assert s["n_accepted"] == 2
    assert s["denial_rate"] == 0.0


def test_progress_callback_fires_once_per_flight():
    cfg = SimConfig(planner="straight", lam_per_hour=80.0, horizon_s=1200.0, seed=1)
    calls = []
    run(cfg, progress=lambda done, total, intent: calls.append((done, total)))
    assert calls, "callback never fired"
    n = calls[-1][1]
    assert len(calls) == n                       # exactly one call per flight
    assert [c[0] for c in calls] == list(range(1, n + 1))   # monotone 1..n, total stable
    assert all(c[1] == n for c in calls)


def test_progress_none_is_silent_and_unchanged(capsys):
    cfg = SimConfig(planner="straight")
    run(cfg, requests=[FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0)])  # default progress=None
    assert capsys.readouterr().err == ""         # nothing printed when off


def test_lazy_planner_demand_run_is_verified():
    # the default-style escalation planner on generated demand must stay conflict-free
    cfg = SimConfig(
        planner="lazy", lam_per_hour=80.0, horizon_s=1800.0, seed=3, region_size_m=(5000.0, 5000.0)
    )
    res = run(cfg)
    assert res.verified
    s = res.summary()
    assert s["n_accepted"] + s["n_denied"] == s["n_requests"]


def test_planner_name_override_is_reflected_in_stored_config():
    # run(planner_name=...) overrides the planner used to plan; the stored config must reflect the planner
    # that ACTUALLY flew, so downstream metrics (which key the altitude baseline on cfg.planner) and the
    # reported planner label describe the real planner, not the original cfg.planner.
    from freespace_sim import metrics
    res = run(SimConfig(planner="straight"), planner_name="astar",
              requests=[FlightRequest(0, vec(0, 0, 0), vec(2000, 0, 0), 0.0)])
    assert res.config.planner == "astar"                 # not the original "straight"
    assert res.accepted[0].planner == "astar"            # A* actually planned it
    # the altitude baseline now keys on the real planner (ladder floor), not the "straight" cruise plane
    assert metrics.nominal_altitude_change_m(res.config) == 2.0 * res.config.flight_levels_m[0]
    assert metrics.aggregate(res)["planner"] == "astar"


def test_no_planner_name_override_preserves_config_identity():
    # the common path (no override) must not perturb the stored config — same object, no needless replace()
    cfg = SimConfig(planner="astar")
    res = run(cfg, requests=[FlightRequest(0, vec(0, 0, 0), vec(2000, 0, 0), 0.0)])
    assert res.config is cfg
