import base64
import dataclasses
import gzip
import json
import math

import numpy as np

from freespace_sim import metrics, runs, viz, viz_html, volumes
from freespace_sim.config import SimConfig
from freespace_sim.geometry import BoxSpec, box_from_segment
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec


def _embedded(html: str) -> dict:
    """Inflate the replay's base64+gzip scene exactly as the browser does."""
    b64 = html.split('const B64 = "', 1)[1].split('"', 1)[0]
    return json.loads(gzip.decompress(base64.b64decode(b64)))


def _undelta(stream: list[int]) -> list[int]:
    """Mirror of the replay's JS ``undelta``."""
    out, v = [], 0
    for d in stream:
        v += d
        out.append(v)
    return out


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
    # each flight stores an INDEX into payload["usses"], not the operator string (repeated 26k times)
    assert all(payload["usses"][f["u"]] in {"walmart", "stripmall"} for f in payload["flights"])
    assert set(payload["uss_colors"]) == {"walmart", "stripmall"} == set(payload["usses"])
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


def _js_seg_poly(p0, p1, cfg):
    """Python mirror of the replay's JS ``segPoly`` — one corridor box footprint rebuilt from two
    adjacent path points, with NO access to the stored volume.

    Deliberately transcribed here rather than shared with ``viz_html``: the point of the test below is
    that this formula reproduces :func:`volumes.corridor_segment_volume`, so it has to be an independent
    copy of the JS. Change one, change the other.
    """
    dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-9:
        length, (ux, uy, uz) = 0.0, (1.0, 0.0, 0.0)
    else:
        ux, uy, uz = dx / length, dy / length, dz / length
    half_len = length / 2 + 0.5 * math.hypot(cfg.corridor_width_m * math.hypot(ux, uy),
                                             cfg.corridor_height_m * uz)
    cx, cy = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    if abs(uz) < 0.99:                                     # lateral axis = unit(world_up x travel)
        n = math.hypot(ux, uy) or 1.0
        vx, vy = -uy / n, ux / n
    else:                                                  # near-vertical falls back to world-x
        n = math.hypot(uy, uz) or 1.0
        vx, vy = 0.0, -uz / n
    half_w = cfg.corridor_width_m / 2
    return [(cx + a * half_len * ux + b * half_w * vx, cy + a * half_len * uy + b * half_w * vy)
            for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]


def test_replay_rebuilds_corridor_box_footprints_exactly():
    """The replay ships no corridor polygons — it regenerates them in JS from the path (72% of the
    payload). This pins that formula to the real builder across every segment orientation, including
    the two branches the density scenario never exercises (near-vertical and zero-length)."""
    cfg = SimConfig()
    segments = [
        ((0, 0, 100), (120, 0, 100)),                      # level, due east
        ((0, 0, 100), (60, 103.9, 100)),                   # level, 60° bearing
        ((500, 500, 100), (380, 500, 100)),                # level, due west (negative direction)
        ((0, 0, 30), (120, 0, 70)),                        # climbing
        ((0, 0, 110), (84.9, 84.9, 30)),                   # descending, diagonal
        ((0, 0, 30), (2, 0, 110)),                         # near-vertical → world-x fallback branch
        ((0, 0, 30), (0, 0, 110)),                         # pure vertical rung
        ((0, 0, 100), (0, 0, 100)),                        # degenerate: hover in place
    ]
    for p0, p1 in segments:
        want = viz.box_footprint(
            volumes.corridor_segment_volume(np.array(p0, float), 0.0, np.array(p1, float), 4.0, cfg).shape)
        got = _js_seg_poly(p0, p1, cfg)
        assert np.allclose(np.array(got), want, atol=1e-9), f"segment {p0}→{p1}"


def test_payload_omits_rebuildable_boxes_and_round_trips_the_path():
    """Boxes are dropped when they are exactly the swept centerline, and the quantised delta streams
    decode back to the flown path within the decimetre quantum."""
    res = _small_run()
    payload = viz_html._payload(res)
    assert payload["explicit_box_flights"] == 0
    assert all("b" not in f for f in payload["flights"])   # no polygons stored at all
    tol = 0.5 / payload["q"]                               # half a quantum
    for intent, f in zip(res.accepted, payload["flights"]):
        xs, ys, zs = (_undelta(f[k]) for k in "xyz")
        ts = _undelta(f["t"])
        assert len(xs) == len(intent.centerline)
        for (p, t), qx, qy, qz, qt in zip(intent.centerline, xs, ys, zs, ts):
            assert abs(qx / payload["q"] - p[0]) <= tol and abs(qy / payload["q"] - p[1]) <= tol
            assert abs(qz / payload["q"] - p[2]) <= tol
            assert abs(qt / payload["qt"] + payload["simulation_start_s"] - t) <= 0.5 / payload["qt"]


def test_payload_keeps_explicit_boxes_when_not_rebuildable():
    """A planner that reserves something other than the swept centerline must still replay correctly —
    the rebuild is *verified* per flight, not assumed, and a mismatch falls back to real polygons."""
    res = _small_run()
    acc = res.accepted[0]
    vols = list(acc.volumes)
    i = next(k for k, v in enumerate(vols) if isinstance(v.shape, BoxSpec))
    moved = dataclasses.replace(vols[i].shape,
                                center=(vols[i].shape.center[0] + 500.0, *vols[i].shape.center[1:]))
    vols[i] = dataclasses.replace(vols[i], shape=moved)
    acc.volumes = vols

    payload = viz_html._payload(res)
    assert payload["explicit_box_flights"] == 1
    doctored = payload["flights"][0]
    assert len(doctored["b"]) == len(vols) - len([v for v in vols if not isinstance(v.shape, BoxSpec)])
    assert all(len(box) == 11 for box in doctored["b"])    # 8 xy coords + z + t0 + t1
    assert all("b" not in f for f in payload["flights"][1:])


def test_viz_html_is_selfcontained_and_parses(tmp_path):
    res = _small_run()
    out = viz_html.write_html(res, tmp_path / "replay.html")
    html = open(out).read()
    assert "<script>" in html and "src=" not in html    # standalone: no external assets to fetch
    payload = viz_html._payload(res)
    assert payload["flights"] and all(f["x"] and f["t"] for f in payload["flights"])
    # each flight carries its straight origin→dest endpoints (the dashed reference line)
    assert all(len(f["o"]) == 2 and len(f["d"]) == 2 for f in payload["flights"])
    # the embedded scene must inflate to exactly the payload the browser will draw
    assert _embedded(html) == json.loads(json.dumps(payload))
    # the slider bounds are substituted as plain numbers, outside the compressed blob
    assert f'min="{payload["simulation_start_s"]}"' in html


def test_write_html_is_byte_reproducible(tmp_path):
    """gzip stamps the wall clock into its header unless told not to; two dumps of the same run must
    match so archived replays don't churn."""
    res = _small_run()
    a = open(viz_html.write_html(res, tmp_path / "a.html"), "rb").read()
    b = open(viz_html.write_html(res, tmp_path / "b.html"), "rb").read()
    assert a == b


def test_embedded_scene_is_compressed_not_inlined(tmp_path):
    """Size regression guard. Asserts on the embedded blob rather than the file, so the fixed ~16 KB
    HTML shell can't mask a regression on the small runs the suite can afford to build."""
    res = _small_run()
    html = open(viz_html.write_html(res, tmp_path / "replay.html")).read()
    blob = html.split('const B64 = "', 1)[1].split('"', 1)[0]
    raw = json.dumps(viz_html._payload(res), separators=(",", ":"))
    assert len(blob) < len(raw)                          # base64(gzip(scene)) still beats the raw dump
    assert raw not in html                               # the scene is compressed, not inlined
    assert all(c.isalnum() or c in "+/=" for c in blob)  # ...and what IS embedded is pure base64


def test_replay_spans_first_activity_through_final_landing():
    from freespace_sim.geometry import box_from_segment
    from freespace_sim.volumes import Volume4D

    res = _small_run()   # horizon 600 s; every flight clears well before it
    # append a volume that extends past the horizon to one accepted flight (a stand-in return tail)
    acc = res.accepted[0]
    tail = Volume4D(box_from_segment(vec(0, 0, 75), vec(60, 0, 75), 60, 30), 700.0, 900.0)
    acc.volumes = (acc.volumes or []) + [tail]
    payload = viz_html._payload(res)
    expected_start, expected_end = metrics.simulation_window(res)
    assert payload["simulation_start_s"] == expected_start
    assert payload["simulation_end_s"] == expected_end == 900.0
    assert payload["simulation_end_s"] > res.config.horizon_s


def test_replay_starts_at_first_flight_activity_not_zero():
    res = run(
        SimConfig(planner="straight", horizon_s=600.0, region_size_m=(2200.0, 2200.0)),
        requests=[FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 123.0)],
    )
    payload = viz_html._payload(res)
    assert payload["simulation_start_s"] == 123.0
    assert payload["simulation_end_s"] > payload["simulation_start_s"]


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
    """Pre-#49 flights.parquet has detour_time_s but not the traffic/lattice split. delay_sources
    must reconstruct the bands (whole detour → traffic) rather than KeyError, and still render."""
    from freespace_sim import metrics
    cfg = SimConfig(planner="astar", lam_per_hour=900.0, horizon_s=600.0,
                    region_size_m=(2500.0, 2500.0), seed=3)
    acc = metrics.flight_frame(run(cfg)).query("accepted")
    legacy = acc.drop(columns=["detour_traffic_s", "detour_lattice_s"])   # shape of an old parquet
    out = tmp_path / "delay_sources.png"
    viz.delay_sources(legacy, out=out, by=None)                           # must not raise
    assert out.exists()
