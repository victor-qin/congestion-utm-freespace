import math

import pandas as pd
import pytest

from freespace_sim import metrics, runs
from freespace_sim.config import SimConfig
from freespace_sim.geometry import BoxSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec
from freespace_sim.volumes import Volume4D, box_from_segment


def _small():
    # one accepted + one forced to yield, so the capture exercises delay + a denied-eligible path
    reqs = [
        FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0),
        FlightRequest(2, vec(0, 0, 0), vec(2000, 0, 0), 0.0),
        FlightRequest(3, vec(1000, -800, 0), vec(1000, 1200, 0), 0.0),
    ]
    return run(SimConfig(planner="straight", horizon_s=600.0, region_size_m=(2200.0, 2200.0)),
               requests=reqs)


def test_save_run_writes_full_bundle(tmp_path):
    folder = runs.save_run(_small(), root=tmp_path, label="t", experiment="unit",
                           experiment_args={"k": 1}, wall_seconds=0.5)
    for name in ("config.json", "env.json", "git.json", "experiment.json", "summary.json",
                 "scenario.parquet", "trajectories.parquet", "reservations.parquet",
                 "flights.parquet", "replay.html"):
        assert (folder / name).stat().st_size > 0, name
    import json
    meta = json.loads((folder / "experiment.json").read_text())
    assert meta["experiment"] == "unit" and meta["args"] == {"k": 1}


def test_scenario_spec_round_trips_through_the_run_folder(tmp_path):
    """A run folder must be able to rebuild the recipe it was launched from.

    ``dataclasses.asdict`` + JSON does not survive the trip: every tuple returns as a list and the
    nested ``demand`` returns a plain dict, so ``ScenarioSpec(**json.load(...)).demand_model()``
    raised ``AttributeError: 'dict' object has no attribute 'build'``. The file existed but nothing
    could read it, while the README advertised the folder as self-contained.
    """
    import json

    from freespace_sim.scenarios import SCENARIOS
    from freespace_sim.scenarios.spec import ScenarioSpec

    spec = SCENARIOS["density_faa_wing_zipline_amazon"]        # tuples + a nested per-USS pair dict
    folder = runs.save_run(_small(), root=tmp_path, label="t", experiment="unit",
                           scenario_spec=spec.to_json_dict(), wall_seconds=0.5)

    back = runs.load_scenario_spec(folder)
    assert back == spec
    assert isinstance(back.region_m, tuple) and isinstance(back.flight_levels_m, tuple)
    assert isinstance(back.demand.uss, tuple) and isinstance(back.demand.hubs, tuple)
    assert all(isinstance(v, tuple) for v in back.demand.departure_offset_s.values())
    back.demand_model()          # used to raise AttributeError
    back.config()

    # every registry scenario must survive the trip, not just the richest one — a scenario-specific
    # tuple field the reconstructor forgets would come back a list and fail equality here.
    for name, s in SCENARIOS.items():
        assert ScenarioSpec.from_json_dict(json.loads(json.dumps(s.to_json_dict()))) == s, name

    # a folder without the file is a normal outcome, not an error
    bare = runs.save_run(_small(), root=tmp_path, label="bare", experiment="unit", wall_seconds=0.5)
    assert runs.load_scenario_spec(bare) is None

    # a newer (or non-numeric) schema must refuse rather than silently reinterpret an unknown layout
    future = json.loads(json.dumps(spec.to_json_dict()))
    future["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        ScenarioSpec.from_json_dict(future)


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
    res = _small()
    folder = runs.save_run(res, root=tmp_path, label="rt")
    loaded = runs.load_run(folder)
    a0, a1 = metrics.aggregate(res), metrics.aggregate(loaded)
    assert a1["n_accepted"] == a0["n_accepted"] and a1["n_denied"] == a0["n_denied"]
    # geometry rebuilt exactly → identical reserved volume-seconds
    assert math.isclose(a1["reserved_vol_m3_s"], a0["reserved_vol_m3_s"], rel_tol=1e-9)
    # centerlines restored → stretch matches
    assert math.isclose(a1["mean_stretch"], a0["mean_stretch"], rel_tol=1e-9)


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
    import json

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
    assert len(p0["flights"]) == len(p1["flights"])
    for a, b in zip(p0["flights"], p1["flights"]):
        assert len(a["boxes"]) == len(b["boxes"]) and len(a["cyls"]) == len(b["cyls"])


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
    assert {"scenario", "tag", "demand", "n_uss", "horizon_s", "region_w", "region_h"} <= set(idx.columns)
    # the tag is the join key a cross-run readout filters on
    assert len(idx[idx["tag"] == "sweepX"]) == 2
    assert set(idx[idx["tag"] == "sweepX"]["scenario"]) == {"metro_2uss"}


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
