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
