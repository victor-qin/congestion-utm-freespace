import json
import math

import pandas as pd

from freespace_sim import metrics, runs
from freespace_sim.config import SimConfig
from freespace_sim.geometry import BoxSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec
from freespace_sim.volumes import Volume4D, box_from_segment


def _small(demand_duration_s=None):
    # one accepted + one forced to yield, so the capture exercises delay + a denied-eligible path
    reqs = [
        FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0),
        FlightRequest(2, vec(0, 0, 0), vec(2000, 0, 0), 0.0),
        FlightRequest(3, vec(1000, -800, 0), vec(1000, 1200, 0), 0.0),
    ]
    return run(SimConfig(planner="straight", horizon_s=600.0,
                         demand_duration_s=demand_duration_s,
                         region_size_m=(2200.0, 2200.0)),
               requests=reqs)


def test_save_run_writes_full_bundle(tmp_path):
    scenario_spec = {
        "name": "tiny",
        "description": "Small resolved unit-test scenario.",
        "horizon_s": 600.0,
        "demand_duration_s": 300.0,
    }
    folder = runs.save_run(_small(), root=tmp_path, label="t", experiment="unit",
                           experiment_args={"k": 1}, scenario_spec=scenario_spec, wall_seconds=0.5)
    for name in ("config.json", "scenario_spec.json", "env.json", "git.json",
                 "experiment.json", "summary.json",
                 "scenario.parquet", "trajectories.parquet", "reservations.parquet",
                 "flights.parquet", "replay.html"):
        assert (folder / name).stat().st_size > 0, name
    meta = json.loads((folder / "experiment.json").read_text())
    assert meta["experiment"] == "unit" and meta["args"] == {"k": 1}
    assert meta["scenario_description"] == scenario_spec["description"]
    assert meta["simulation_duration_s"] > 0.0


def test_save_run_persists_resolved_scenario_spec(tmp_path):
    payload = {
        "name": "resolved",
        "description": "The exact post-override recipe.",
        "lam_per_hour": 321.0,
        "demand": {"lam_per_uss": {"a": 123.0, "b": 198.0}},
    }
    folder = runs.save_run(
        _small(),
        root=tmp_path,
        label="spec",
        scenario="resolved",
        scenario_spec=payload,
        write_replay=False,
    )
    assert json.loads((folder / "scenario_spec.json").read_text()) == payload
    experiment = json.loads((folder / "experiment.json").read_text())
    assert experiment["scenario_description"] == payload["description"]


def test_scenario_frame_includes_every_request():
    res = _small()
    sdf = runs.scenario_frame(res)
    assert len(sdf) == len(res.intents)               # denied flights captured too, not just flown
    assert set(sdf["flight_id"]) == {1, 2, 3}


def test_reservation_frame_rebuilds_exact_volumes():
    res = _small()
    rdf = runs.reservation_frame(res)
    assert set(rdf["kind"]) <= {"box", "cyl"}
    # a rebuilt box/cyl reproduces the same FCL-relevant geometry it was serialized from
    box_row = rdf[rdf["kind"] == "box"].iloc[0]
    v = runs._volume_from_row(box_row)
    assert isinstance(v.shape, BoxSpec)
    assert math.isclose(v.t_start, box_row["t_start"]) and v.shape.center[0] == box_row["cx"]


def test_load_run_roundtrip_is_faithful(tmp_path):
    res = _small(demand_duration_s=300.0)
    folder = runs.save_run(res, root=tmp_path, label="rt")
    loaded = runs.load_run(folder)
    a0, a1 = metrics.aggregate(res), metrics.aggregate(loaded)
    assert a1["n_accepted"] == a0["n_accepted"] and a1["n_denied"] == a0["n_denied"]
    # geometry rebuilt exactly → identical reserved volume-seconds
    assert math.isclose(a1["reserved_vol_m3_s"], a0["reserved_vol_m3_s"], rel_tol=1e-9)
    # centerlines restored → stretch matches
    assert math.isclose(a1["mean_stretch"], a0["mean_stretch"], rel_tol=1e-9)
    assert loaded.config.demand_duration_s == 300.0


def test_load_run_tolerates_dropped_legacy_altitude_keys(tmp_path):
    # runs archived before cruise_level_m/z_min_m/z_max_m became derived @properties carry those keys in
    # config.json; load_run must whitelist-filter them (not TypeError) and re-derive off flight_levels_m.
    import json
    folder = runs.save_run(_small(), root=tmp_path, label="legacy")
    cfg_path = folder / "config.json"
    payload = json.loads(cfg_path.read_text())
    payload.update(cruise_level_m=75.0, z_min_m=42.0, z_max_m=99.0)   # legacy stored fields, now removed
    payload["flight_levels_m"] = list(payload["flight_levels_m"])     # JSON round-trips the tuple as a list
    cfg_path.write_text(json.dumps(payload))
    cfg = runs.load_run(folder).config                               # must NOT raise TypeError
    assert cfg.z_min_m == cfg.flight_levels_m[0]                      # re-derived off the ladder; 42 dropped
    assert cfg.z_max_m == cfg.flight_levels_m[-1]                     # ... 99 dropped too


def test_load_run_back_converts_legacy_per_metre_cost_weights(tmp_path):
    """Runs archived before the per-second normalization stored cost_air_lateral_per_m /
    cost_altitude_change_per_m, which are derived @properties now. Whitelist-filtering alone would
    DROP them and silently replay the run under today's defaults — a different cost model than it was
    planned with (0.1 vs 3.0 per lateral metre, a 30x change). They must be back-converted instead."""
    folder = runs.save_run(_small(), root=tmp_path, label="legacy_costs")
    cfg_path = folder / "config.json"
    payload = json.loads(cfg_path.read_text())
    for k in ("cost_air_lateral_per_s", "cost_altitude_change_per_s"):
        payload.pop(k, None)                                          # the old schema had neither
    payload.update(cost_air_lateral_per_m=3.0, cost_altitude_change_per_m=4.0)
    payload["flight_levels_m"] = list(payload["flight_levels_m"])
    payload["region_size_m"] = list(payload["region_size_m"])
    cfg_path.write_text(json.dumps(payload))

    cfg = runs.load_run(folder).config
    assert cfg.cost_air_lateral_per_m == 3.0                          # the archived weight, preserved
    assert cfg.cost_altitude_change_per_m == 4.0
    # ... which means the per-second knobs were reconstructed, not defaulted
    assert cfg.cost_air_lateral_per_s == 3.0 * cfg.nominal_speed_mps
    assert cfg.cost_altitude_change_per_s == 4.0 * cfg.climb_rate_mps


def test_load_run_replay_payload_matches(tmp_path):
    from freespace_sim import viz_html

    res = _small()
    folder = runs.save_run(res, root=tmp_path, label="rt")
    loaded = runs.load_run(folder)
    p0, p1 = viz_html._payload(res), viz_html._payload(loaded)
    # the replay encodes quantised ints, so a save/load round-trip must reproduce the scene EXACTLY —
    # a stronger claim than the old count-only check, and it also pins the box-rebuild decision
    assert p0 == p1


def test_index_parquet_appends(tmp_path):
    runs.save_run(_small(), root=tmp_path, label="a")
    runs.save_run(_small(), root=tmp_path, label="b")
    idx = pd.read_parquet(tmp_path / "index.parquet")
    assert len(idx) == 2
    assert {"path", "planner", "verified", "mean_total_delay_s"} <= set(idx.columns)


def test_index_records_scenario_tag_demand_for_cross_run_filter(tmp_path):
    runs.save_run(_small(), root=tmp_path, label="sweepX", scenario="metro_2uss", demand="uniform")
    runs.save_run(_small(), root=tmp_path, label="sweepX", scenario="metro_2uss", demand="uniform")
    runs.save_run(_small(), root=tmp_path, label="other", scenario="metro_uniform", demand="uniform")
    idx = runs.load_index(tmp_path)
    assert {
        "scenario",
        "tag",
        "demand",
        "n_uss",
        "horizon_s",
        "simulation_start_s",
        "simulation_end_s",
        "simulation_duration_s",
        "region_w",
        "region_h",
    } <= set(idx.columns)
    # the tag is the join key a cross-run readout filters on
    assert len(idx[idx["tag"] == "sweepX"]) == 2
    assert set(idx[idx["tag"] == "sweepX"]["scenario"]) == {"metro_2uss"}


def test_index_records_description_and_demand_duration(tmp_path):
    payload = {"name": "tiny", "description": "Density metadata test."}
    runs.save_run(
        _small(demand_duration_s=300.0),
        root=tmp_path,
        label="density",
        scenario="tiny",
        scenario_spec=payload,
        write_replay=False,
    )
    row = runs.load_index(tmp_path).iloc[0]
    assert row["scenario_description"] == payload["description"]
    assert row["demand_duration_s"] == 300.0
    assert row["simulation_duration_s"] > 0.0


def test_load_index_empty_when_missing(tmp_path):
    assert runs.load_index(tmp_path).empty


def test_denied_flight_captured_without_volumes(tmp_path):
    # a fully-walled straight flight is denied; the capture must record it (no volumes) and reload it
    led = ReservationLedger(SimConfig())
    led.commit(99, [Volume4D(box_from_segment(vec(1000, -300, 150), vec(1000, 300, 150), 40, 400),
                             0.0, 1e6)])
    denied = get_planner("straight").plan(
        FlightRequest(7, vec(0, 0, 0), vec(2000, 0, 0), 0.0), led, SimConfig())
    assert not denied.accepted
    # hand-assemble a result with the denied intent and round-trip it
    from freespace_sim.sim import SimResult

    res = SimResult(config=SimConfig(region_size_m=(2200.0, 2200.0)),
                    intents=[denied], ledger=led, verified=True)
    folder = runs.save_run(res, root=tmp_path, label="den")
    loaded = runs.load_run(folder)
    assert len(loaded.denied) == 1 and loaded.denied[0].volumes is None
