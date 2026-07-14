"""Telemetry — observer-only capture of the non-recoverable congestion streams (run instrumentation).

Guards the load-bearing properties: telemetry-off is byte-identical (observer-only); a built-then-denied
corridor's filed volumes are captured on the DEFAULT compiled A* path; culprit classification
(static_wall / sibling / foreign); dwell occupancy recovered from reservations; and save_run/load_run
round-trip the parquets + terminal membership.
"""

from freespace_sim import runs
from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.sim import SimResult, run
from freespace_sim.telemetry import TelemetryCollector, conflict_frame, filed_volume_frame, terminal_frame
from freespace_sim.types import (
    FlightRequest,
    IntentStatus,
    OperationalIntent,
    Terminal,
    as_terminal,
    vec,
)
from freespace_sim.volumes import Volume4D


# --- unit: the collector hook + rollup classification (no sim) ------------------------------------

def test_on_deny_captures_filed_volumes_and_culprits():
    c = TelemetryCollector()
    box = Volume4D(box_from_segment(vec(0, 0, 75), vec(60, 0, 75), 60, 30), 0.0, 10.0)
    cyl = Volume4D(CylinderSpec(100, 0, 90, 0, 125), 0.0, 50.0, terminal_id="hubA")
    # a conflict_filed: the rejected corridor (2 boxes) + two culprits (a real flight + the static wall)
    c.on_deny(7, "conflict_filed", [box, box], [(3, cyl), (-1, cyl)])
    assert len(c.filed_volumes) == 2 and {r["reason"] for r in c.filed_volumes} == {"conflict_filed"}
    assert {e["culprit_fid"] for e in c.conflict_events} == {3, -1}
    # a budget_exceeded (detour): filed volumes, NO culprits
    c.on_deny(8, "budget_exceeded", [box], None)
    assert any(r["flight_id"] == 8 for r in c.filed_volumes)
    assert all(e["flight_id"] != 8 for e in c.conflict_events)


def _rejected(flight_id, hub_id):
    req = FlightRequest(flight_id, vec(0, 0, 0), vec(1, 0, 0), 0.0,
                        origin_terminal=Terminal(hub_id, 2))
    return OperationalIntent(req, IntentStatus.REJECTED)


def test_conflict_frame_classifies_static_wall_sibling_foreign():
    c = TelemetryCollector()
    c.conflict_events = [
        {"flight_id": 1, "culprit_fid": -1, "culprit_tid": "hubA", "shape": "CylinderSpec", "t_start": 0, "t_end": 1},
        {"flight_id": 1, "culprit_fid": 9, "culprit_tid": "hubA", "shape": "BoxSpec", "t_start": 0, "t_end": 1},
        {"flight_id": 1, "culprit_fid": 9, "culprit_tid": "hubZ", "shape": "BoxSpec", "t_start": 0, "t_end": 1},
    ]
    res = SimResult(config=SimConfig(planner="astar"), intents=[_rejected(1, "hubA")],
                    ledger=None, verified=True, telemetry=c)
    assert list(conflict_frame(res)["culprit_kind"]) == ["static_wall", "sibling", "foreign"]


# --- integration: real A* (compiled default) ------------------------------------------------------

def test_filed_volumes_captured_on_detour_denial_compiled_path():
    # max_detour_factor < 1 forces EVERY path to trip the detour check AFTER _build → a built-then-denied
    # corridor. Runs the DEFAULT compiled kernel (guards the plan-critic CRIT: capture must fire there).
    cfg = SimConfig(planner="astar", horizon_s=300.0, region_size_m=(3000.0, 3000.0), max_detour_factor=0.5)
    reqs = [FlightRequest(0, vec(500, 500, 0), vec(2500, 500, 0), 0.0)]
    res = run(cfg, requests=reqs, telemetry=True)
    assert not res.accepted and res.intents[0].denial_reason.value == "budget_exceeded"
    ff = filed_volume_frame(res)
    assert len(ff) > 0 and set(ff["reason"]) == {"budget_exceeded"}
    assert len(res.telemetry.conflict_events) == 0   # a detour denial has no blocker


def test_telemetry_off_is_byte_identical():
    cfg = SimConfig(planner="astar", horizon_s=300.0, region_size_m=(3000.0, 3000.0), max_detour_factor=0.5)
    reqs = [FlightRequest(0, vec(500, 500, 0), vec(2500, 500, 0), 0.0)]
    on, off = run(cfg, requests=reqs, telemetry=True), run(cfg, requests=reqs, telemetry=False)
    assert off.telemetry is None
    assert [i.status for i in on.intents] == [i.status for i in off.intents]
    assert [i.denial_reason for i in on.intents] == [i.denial_reason for i in off.intents]
    assert [i.cost for i in on.intents] == [i.cost for i in off.intents]


def _hub_run():
    hub = Terminal("hubA", 2)
    reqs = [FlightRequest(0, vec(1000, 1000, 0), vec(2800, 1000, 0), 0.0, origin_terminal=hub),
            FlightRequest(1, vec(1000, 1000, 0), vec(2800, 2800, 0), 3.0, origin_terminal=hub)]
    cfg = SimConfig(planner="astar", horizon_s=600.0, region_size_m=(4000.0, 4000.0))
    return run(cfg, requests=reqs, telemetry=True)


def test_terminal_snapshot_and_peak_occupancy_from_reservations():
    res = _hub_run()
    assert "hubA" in {str(k) for k in res.telemetry.terminals}
    tf = terminal_frame(res)
    row = tf[tf["tid"] == "hubA"].iloc[0]
    assert row["pads"] == 2 and row["n_departures"] >= 1
    # peak_pad_occupancy is a sweep-line over the persisted terminal cylinders — no live dwell hook
    assert row["peak_pad_occupancy"] >= 1


def test_save_run_persists_telemetry_and_roundtrips_terminals(tmp_path):
    res = _hub_run()
    folder = runs.save_run(res, root=tmp_path, write_replay=False)
    for name in ("terminal_telemetry", "conflict_events", "filed_volumes", "ledger_end"):
        assert (folder / f"{name}.parquet").exists()
    # load_run round-trips terminal (hub) membership onto the rebuilt request
    loaded = runs.load_run(folder)
    r0 = next(i for i in loaded.intents if i.request.flight_id == 0).request
    assert as_terminal(r0.origin_terminal).id == "hubA"
    # a non-telemetry run writes NONE of the extra parquets (separate root: same config+second else collides)
    res_off = run(res.config, requests=[i.request for i in res.intents], telemetry=False)
    folder_off = runs.save_run(res_off, root=tmp_path / "off", write_replay=False)
    assert not (folder_off / "filed_volumes.parquet").exists()
    # the cross-run index carries has_telemetry
    idx = runs.load_index(tmp_path)
    assert "has_telemetry" in idx.columns and bool(idx["has_telemetry"].any())
