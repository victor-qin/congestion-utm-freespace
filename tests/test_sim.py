import logging

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


def test_milestone_recordings_at_horizon_marks(caplog):
    # 5%-of-horizon "recordings": carried by the FIRST flight filing at-or-after each mark; a sparse
    # stretch makes one flight carry every mark it jumped over (one line per mark). horizon=100 →
    # marks at 5,10,…,100. Flights at t=0 (before any mark), t=7 (carries @5%), t=52 (carries
    # @10%…@50%, nine marks). Marks @55%+ never fire — no flight files after them.
    caplog.set_level(logging.INFO, logger="freespace_sim.sim")
    cfg = SimConfig(planner="straight", horizon_s=100.0)
    run(cfg, requests=[
        FlightRequest(0, vec(0, 0, 0), vec(2400, 0, 0), 0.0),
        FlightRequest(1, vec(0, 200, 0), vec(2400, 200, 0), 7.0),
        FlightRequest(2, vec(0, 400, 0), vec(2400, 400, 0), 52.0),
    ])
    recs = [m for m in caplog.messages if m.startswith("recording @")]
    assert len(recs) == 10
    assert recs[0].startswith("recording @5% horizon (mark 5s): flight=1 sim_t=7.0s")
    assert all("flight=2" in m for m in recs[1:])
    assert recs[-1].startswith("recording @50% horizon (mark 50s): flight=2 sim_t=52.0s")
    assert not [m for m in caplog.messages if m.startswith("planned ")]   # 3 flights ≪ every_n=1000


def test_milestone_every_n_planned_flights(caplog):
    # the flight-count cadence, exercised directly with every_n=2 (1000 needs a huge run): lines at
    # done=2 and done=4 carrying the triggering flight + running acc/den; huge horizon → no recordings.
    from types import SimpleNamespace

    from freespace_sim.sim import _MilestoneLog
    from freespace_sim.types import IntentStatus

    caplog.set_level(logging.INFO, logger="freespace_sim.sim")
    acc = SimpleNamespace(accepted=True, status=IntentStatus.ACCEPTED)
    den = SimpleNamespace(accepted=False, status=IntentStatus.REJECTED)
    ml = _MilestoneLog(total=5, horizon_s=1e12, every_n=2)
    for done, outcome in enumerate([acc, den, acc, acc, den], 1):
        ml(done, FlightRequest(done - 1, vec(0, 0, 0), vec(100, 0, 0), float(done)), outcome)
    lines = [m for m in caplog.messages if m.startswith("planned ")]
    assert len(lines) == 2
    assert lines[0].startswith("planned 2/5: flight=1 sim_t=2.0s") and "acc=1 den=1" in lines[0]
    assert lines[1].startswith("planned 4/5: flight=3 sim_t=4.0s") and "acc=3 den=1" in lines[1]
    assert not [m for m in caplog.messages if m.startswith("recording @")]


def test_milestone_mark_hit_exactly_is_carried_by_that_flight(caplog):
    # a flight filing EXACTLY on a horizon mark carries that mark. Guards the mark arithmetic:
    # horizon*0.05*k overshoots the true fraction for many (horizon, k) pairs (1.0*0.05*3 =
    # 0.15000000000000002), which silently deferred the mark past an exactly-on-time flight;
    # marks are now horizon*k/n_marks, exact for round times. h=1 → flight at t=0.15 crosses
    # @5%, @10% and — exactly — @15%.
    from types import SimpleNamespace

    from freespace_sim.sim import _MilestoneLog
    from freespace_sim.types import IntentStatus

    caplog.set_level(logging.INFO, logger="freespace_sim.sim")
    ml = _MilestoneLog(total=1, horizon_s=1.0)
    ml(1, FlightRequest(0, vec(0, 0, 0), vec(100, 0, 0), 0.15),
       SimpleNamespace(accepted=True, status=IntentStatus.ACCEPTED))
    recs = [m for m in caplog.messages if m.startswith("recording @")]
    assert [m.split(":")[0] for m in recs] == [
        "recording @5% horizon (mark 0s)", "recording @10% horizon (mark 0s)",
        "recording @15% horizon (mark 0s)"]                    # @15% NOT deferred past t=0.15
    assert all("flight=0" in m for m in recs)


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
