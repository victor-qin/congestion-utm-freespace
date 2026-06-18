import json

import numpy as np

from freespace_sim import runs, viz, viz_html
from freespace_sim.config import SimConfig
from freespace_sim.geometry import box_from_segment
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec


def _small_run():
    reqs = [
        FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0),
        FlightRequest(2, vec(1000, -800, 0), vec(1000, 1200, 0), 0.0),
    ]
    return run(SimConfig(planner="straight", horizon_s=600.0, region_size_m=(2200.0, 2200.0)),
               requests=reqs)


def test_flight_color_is_deterministic_and_distinct():
    assert viz.flight_color(1) == viz.flight_color(1)
    assert viz.flight_color(1) != viz.flight_color(2)


def test_box_footprint_is_four_xy_corners_of_right_size():
    spec = box_from_segment(vec(0, 0, 150), vec(120, 0, 150), 60.0, 30.0)
    fp = viz.box_footprint(spec)
    assert fp.shape == (4, 2)
    # axis-aligned east-west box: spans ~120+60 in x (length+overlap-free here) and 60 in y
    assert np.isclose(fp[:, 1].max() - fp[:, 1].min(), 60.0, atol=1e-6)


def test_snapshot_and_heatmap_write_files(tmp_path):
    res = _small_run()
    viz.snapshot(res, t=40.0, out=tmp_path / "snap.png")
    viz.congestion_heatmap(res, out=tmp_path / "heat.png")
    assert (tmp_path / "snap.png").stat().st_size > 0
    assert (tmp_path / "heat.png").stat().st_size > 0


def test_scene_3d_has_geometry():
    res = _small_run()
    scene = viz.scene_3d(res)
    assert len(scene.geometry) > 0


def test_viz_html_is_selfcontained_and_parses(tmp_path):
    res = _small_run()
    out = viz_html.write_html(res, tmp_path / "replay.html")
    html = open(out).read()
    assert "{horizon}" not in html and "{data}" not in html   # all tokens substituted
    payload = viz_html._payload(res)
    assert payload["flights"] and all("path" in f for f in payload["flights"])
    # each flight carries its straight origin→dest endpoints (the dashed reference line)
    assert all(len(f["o"]) == 2 and len(f["d"]) == 2 for f in payload["flights"])
    # the embedded DATA blob must be valid JSON
    blob = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert json.loads(blob)["horizon"] == res.config.horizon_s


def test_delay_histogram_drops_nan_and_writes(tmp_path):
    viz.delay_histogram([10.0, 20.0, float("nan"), 30.0], out=tmp_path / "h.png")
    assert (tmp_path / "h.png").stat().st_size > 0


def test_delay_histograms_by_lambda(tmp_path):
    import pandas as pd

    from freespace_sim import metrics

    frames = []
    for lam in (40.0, 120.0):
        res = run(SimConfig(planner="straight", lam_per_hour=lam, horizon_s=1200.0, seed=1))
        f = metrics.flight_frame(res)
        f["lam_per_hour"] = lam
        frames.append(f)
    viz.delay_histograms_by_lambda(pd.concat(frames, ignore_index=True), out=tmp_path / "byL.png")
    assert (tmp_path / "byL.png").stat().st_size > 0


def test_save_run_roundtrips_parquet(tmp_path):
    import pandas as pd

    res = _small_run()
    folder = runs.save_run(res, root=tmp_path, label="t")
    for name in ("config.json", "env.json", "summary.json", "flights.parquet"):
        assert (folder / name).stat().st_size > 0
    df = pd.read_parquet(folder / "flights.parquet")
    assert len(df) == len(res.intents)


def test_save_sweep_roundtrips(tmp_path):
    import pandas as pd

    from freespace_sim import metrics

    rows = [metrics.aggregate(_small_run())]
    folder = runs.save_sweep(rows, root=tmp_path, label="s")
    df = pd.read_parquet(folder / "sweep.parquet")
    assert len(df) == 1 and "denial_rate" in df.columns
