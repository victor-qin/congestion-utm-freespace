import json

import numpy as np
import pytest

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


def _two_uss_run():
    reqs = [
        FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0, uss_id="walmart"),
        FlightRequest(2, vec(1000, -800, 0), vec(1000, 1200, 0), 0.0, uss_id="stripmall"),
    ]
    return run(SimConfig(planner="straight", horizon_s=600.0, region_size_m=(2200.0, 2200.0)),
               requests=reqs)


def test_flight_color_is_deterministic_and_distinct():
    assert viz.flight_color(1) == viz.flight_color(1)
    assert viz.flight_color(1) != viz.flight_color(2)


def test_uss_hues_distinct_and_deterministic():
    h1 = viz.uss_hues(["walmart", "stripmall"])
    h2 = viz.uss_hues(["stripmall", "walmart"])      # order-independent (sorted internally)
    assert h1 == h2
    assert h1["walmart"] != h1["stripmall"]


def test_flight_color_by_uss_groups_by_hue():
    import colorsys
    hues = viz.uss_hues(["walmart", "stripmall"])
    # same USS, different flight → same hue (only sat/value jitter)
    a = colorsys.rgb_to_hsv(*viz.flight_color_by_uss("walmart", 1, hues))
    b = colorsys.rgb_to_hsv(*viz.flight_color_by_uss("walmart", 7, hues))
    assert abs(a[0] - b[0]) < 1e-9
    # different USS → different hue family
    c = colorsys.rgb_to_hsv(*viz.flight_color_by_uss("stripmall", 1, hues))
    assert abs(a[0] - c[0]) > 1e-9


def test_snapshot_uss_filter_writes_file(tmp_path):
    res = _two_uss_run()
    viz.snapshot(res, t=20.0, out=tmp_path / "walmart.png", uss="walmart")
    assert (tmp_path / "walmart.png").stat().st_size > 0


def test_payload_carries_uss_and_colors():
    res = _two_uss_run()
    payload = viz_html._payload(res)
    assert all("uss" in f for f in payload["flights"])
    assert set(payload["uss_colors"]) == {"walmart", "stripmall"}
    assert all(c.startswith("#") and len(c) == 7 for c in payload["uss_colors"].values())


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


def test_replay_clips_to_horizon_by_default():
    from freespace_sim.geometry import box_from_segment
    from freespace_sim.volumes import Volume4D

    res = _small_run()   # horizon 600 s; every flight clears well before it
    # append a volume that extends past the horizon to one accepted flight (a stand-in return tail)
    acc = res.accepted[0]
    tail = Volume4D(box_from_segment(vec(0, 0, 75), vec(60, 0, 75), 60, 30), 700.0, 900.0)
    acc.volumes = (acc.volumes or []) + [tail]
    # default clips the replay clock at the horizon, excluding the post-H tail
    assert viz_html._payload(res, clip_to_horizon=True)["horizon"] == res.config.horizon_s
    # opting out extends the clock to the last volume (return-flight demo)
    assert viz_html._payload(res, clip_to_horizon=False)["horizon"] >= 900.0


def test_delay_histogram_overlay_writes_two_series(tmp_path):
    # the steady-state overlay draws both distributions on shared bins (issue #25)
    viz.delay_histogram([10.0, 20.0, 30.0, 40.0], overlay=[25.0, 30.0, 35.0],
                        out=tmp_path / "ov.png")
    assert (tmp_path / "ov.png").stat().st_size > 0


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


def test_delay_sources_bands_reconcile_to_total_delay(tmp_path):
    """The stacked chart must account for ALL of ``total_delay_s`` — driven off ``_DELAY_SOURCES``
    itself, so editing the chart's band list can never silently reintroduce a shortfall.

    Run on the 3-level ladder under load: the climb band is non-zero here, which is exactly the case
    the old 3-band stack under-reported (and which no single-plane run can detect).
    """
    from freespace_sim import metrics

    cfg = SimConfig(planner="astar", lam_per_hour=900.0, horizon_s=600.0,
                    region_size_m=(2500.0, 2500.0), seed=3)
    acc = metrics.flight_frame(run(cfg)).query("accepted")
    assert (acc["excess_altitude_m"] > 0).any(), "no traffic-forced climb — the check would be vacuous"
    assert (acc["lattice_overhead_m"] > 0).any(), "no hex staircase — the check would be vacuous"

    stacked = sum(acc[key] for key, _, _, _ in viz._DELAY_SOURCES)
    assert ((stacked - acc["total_delay_s"]).abs() < 1e-9).all()

    out = tmp_path / "delay_sources.png"
    viz.delay_sources(acc, out=out, by=None)
    assert out.exists()


def test_delay_sources_tolerates_legacy_frame_without_detour_split(tmp_path):
    """A pre-#49 flights.parquet has detour_time_s but not the traffic/lattice split; a run older than
    the climb band additionally lacks altitude_delay_phys_s. delay_sources must reconstruct the detour
    split (whole detour → traffic) and zero-fill the legacy-only climb band rather than KeyError, and render."""
    from freespace_sim import metrics
    cfg = SimConfig(planner="astar", lam_per_hour=900.0, horizon_s=600.0,
                    region_size_m=(2500.0, 2500.0), seed=3)
    acc = metrics.flight_frame(run(cfg)).query("accepted")
    # shape of an old parquet: no detour split, and (older still) no climb band
    legacy = acc.drop(columns=["detour_traffic_s", "detour_lattice_s", "altitude_delay_phys_s"])
    out = tmp_path / "delay_sources.png"
    viz.delay_sources(legacy, out=out, by=None)                           # must not raise
    assert out.exists()


def test_delay_sources_raises_on_missing_core_band(tmp_path):
    """The legacy back-stop stays NARROW: every planner emits all five bands (0 where a lever doesn't
    apply), so a current-schema frame missing a core lever (ground_delay_s) is a real upstream defect,
    not a schema gap. delay_sources must still surface it as a KeyError rather than silently plot a 0
    band under a wrong-but-reconciling total — a guard against re-broadening the fill to _DELAY_SOURCES."""
    from freespace_sim import metrics
    cfg = SimConfig(planner="astar", lam_per_hour=900.0, horizon_s=600.0,
                    region_size_m=(2500.0, 2500.0), seed=3)
    acc = metrics.flight_frame(run(cfg)).query("accepted")
    broken = acc.drop(columns=["ground_delay_s"])          # a core band vanishing ⇒ upstream bug, not legacy
    with pytest.raises(KeyError):
        viz.delay_sources(broken, out=tmp_path / "x.png", by=None)
