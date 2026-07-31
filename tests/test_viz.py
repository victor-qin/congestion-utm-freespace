import base64
import dataclasses
import gzip
import json
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from freespace_sim import metrics, runs, viz, viz_html, volumes
from freespace_sim.config import SimConfig
from freespace_sim.geometry import BoxSpec, box_from_segment
from freespace_sim.sim import run
from freespace_sim.types import FlightRequest, vec


def _blob(html: str) -> str:
    """The embedded base64 scene."""
    return html.split('const B64 = "', 1)[1].split('"', 1)[0]


def _embedded(html: str) -> dict:
    """Inflate the replay's base64+gzip scene exactly as the browser does."""
    return json.loads(gzip.decompress(base64.b64decode(_blob(html))))


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


def test_congestion_heatmap_measures_over_simulation_window_not_horizon_clamp():
    """congestion_heatmap must bin volume-seconds over simulation_window, not clamp at cfg.horizon_s.

    The old clamp dropped every reservation-second past horizon_s — exactly the post-horizon return
    tail the density scenarios exist to produce. This drives the ACTUAL function (via the grid it
    plots onto the returned axis) rather than re-deriving the sum inline, so reverting viz.py to the
    clamp makes it fail: (a) the plotted total no longer reconciles with metrics' reserved_vol_m3_s,
    and (b) it collapses to the horizon-clamped total instead of exceeding it.
    """
    import math

    from freespace_sim import metrics

    res = run(SimConfig(planner="straight", lam_per_hour=200.0, horizon_s=600.0, seed=3))
    t_lo, t_hi = metrics.simulation_window(res)
    assert t_hi > res.config.horizon_s, "fixture has no post-horizon tail — the check is vacuous"

    ax = viz.congestion_heatmap(res)                       # the real function, not a re-implementation
    plotted_total = float(ax.images[0].get_array().sum())  # the grid it actually binned

    # (a) the figure reconciles with the metric measured over the same (simulation) window
    assert math.isclose(plotted_total, metrics.aggregate(res)["reserved_vol_m3_s"], rel_tol=1e-9)

    # (b) and it is strictly MORE than the old horizon-clamped total — the dropped return tail. A
    # revert to `min(v.t_end, horizon_s) - max(v.t_start, 0)` makes plotted_total == clamped, so this
    # fails. (No reservation starts before t_lo, so the clamp's lower bound of 0 loses nothing there.)
    clamped = sum(
        metrics.shape_volume_m3(v.shape)
        * max(0.0, min(v.t_end, res.config.horizon_s) - max(v.t_start, 0.0))
        for i in res.accepted for v in (i.volumes or []))
    assert plotted_total > clamped * (1.0 + 1e-9)


def test_scene_3d_has_geometry():
    res = _small_run()
    scene = viz.scene_3d(res)
    assert len(scene.geometry) > 0


def test_viz_html_is_selfcontained_and_parses(tmp_path):
    res = _small_run()
    out = viz_html.write_html(res, tmp_path / "replay.html")
    html = open(out).read()
    assert "<script>" in html and "<img" not in html    # standalone: no external assets to fetch
    payload = viz_html._payload(res)
    assert payload["flights"] and all(f["x"] and f["t"] for f in payload["flights"])
    # each flight carries its straight origin→dest endpoints (the dashed reference line)
    assert all(len(f["o"]) == 2 and len(f["d"]) == 2 for f in payload["flights"])
    # the embedded scene must inflate to exactly the payload the browser will draw
    assert _embedded(html) == json.loads(json.dumps(payload))
    # the slider bounds are substituted as plain numbers, outside the compressed blob
    assert f'min="{payload["simulation_start_s"]}"' in html
    assert f'max="{payload["simulation_end_s"]}"' in html


def test_replay_spans_the_realized_run_not_the_horizon():
    """The clock is the realized operation, so it EXTENDS past an early horizon and TRIMS a late one.

    `horizon_s` is a planner envelope, not a schedule: on the density scenarios flights are filed from
    t=0 but the first departs ~768 s in, and the last lands ~3300 s before the envelope closes — 57% of
    the old slider was empty sky at one end or the other."""
    from freespace_sim.geometry import box_from_segment
    from freespace_sim.volumes import Volume4D

    res = _small_run()   # horizon 600 s; every flight clears well before it
    # (a) TRIM: nothing is airborne near the horizon, so the clock must stop at the last landing
    payload = viz_html._payload(res)
    lo, hi = metrics.simulation_window(res)
    assert (payload["simulation_start_s"], payload["simulation_end_s"]) == (lo, hi)
    assert hi < res.config.horizon_s, "fixture clears well before the horizon — the trim check is real"

    # (b) EXTEND: a post-horizon return tail must still be reachable on the slider
    acc = res.accepted[0]
    acc.volumes = (acc.volumes or []) + [
        Volume4D(box_from_segment(vec(0, 0, 75), vec(60, 0, 75), 60, 30), 700.0, 900.0)]
    assert viz_html._payload(res)["simulation_end_s"] == 900.0


def test_replay_clock_starts_at_first_departure_not_at_filing():
    """A flight filed at t=0 but departing later must not leave the replay opening on an empty sky."""
    res = run(SimConfig(planner="straight", horizon_s=900.0, region_size_m=(2200.0, 2200.0)),
              requests=[FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0, t_departure=300.0)])
    payload = viz_html._payload(res)
    assert payload["simulation_start_s"] >= 290.0    # the takeoff column opens just before the roll
    assert payload["simulation_duration_s"] > 0


_SEG_CASES = [
    ((0, 0, 100), (120, 0, 100)),                      # level, due east
    ((0, 0, 100), (60, 103.9, 100)),                   # level, 60° bearing
    ((500, 500, 100), (380, 500, 100)),                # level, due west (negative direction)
    ((0, 0, 30), (120, 0, 70)),                        # climbing
    ((0, 0, 110), (84.9, 84.9, 30)),                   # descending, diagonal
    ((0, 0, 30), (2, 0, 110)),                         # near-vertical → world-x fallback branch
    ((0, 0, 30), (0, 0, 110)),                         # pure vertical rung
    ((0, 0, 100), (0, 0, 100)),                        # degenerate: hover in place
]


def _builder_footprints(cfg):
    """What the ledger actually reserved, for each case in `_SEG_CASES`."""
    return [viz.box_footprint(volumes.corridor_segment_volume(
        np.array(p0, float), 0.0, np.array(p1, float), 4.0, cfg).shape) for p0, p1 in _SEG_CASES]


def test_shipped_segpoly_js_reproduces_the_corridor_builder():
    """Runs the REAL shipped `segPoly` source in node against `volumes.corridor_segment_volume`.

    This is the load-bearing pin for the whole size lever: the replay stores no corridor polygons, so
    if the JS drifts from the builder every archived replay silently draws wrong geometry — and
    `_rebuildable` cannot catch it, because it compares Python against Python. A Python transcription
    of the JS would not catch it either; only executing `viz_html._SEG_POLY_JS` itself does.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; cannot execute the shipped JS")
    cfg = SimConfig()
    harness = f"""
const IQ = 1, CW = {cfg.corridor_width_m}, CH = {cfg.corridor_height_m};
{viz_html._SEG_POLY_JS}
const cases = {json.dumps(_SEG_CASES)};
const out = [];
for (const [p0, p1] of cases) {{
  const f = {{x: [p0[0], p1[0]], y: [p0[1], p1[1]], z: [p0[2], p1[2]]}};
  const poly = new Float64Array(8);
  const z = segPoly(f, 0, poly);
  out.push({{poly: Array.from(poly), z}});
}}
console.log(JSON.stringify(out));
"""
    proc = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    got = json.loads(proc.stdout)

    for (p0, p1), rec, want in zip(_SEG_CASES, got, _builder_footprints(cfg)):
        assert np.allclose(np.array(rec["poly"]).reshape(4, 2), want, atol=1e-9), f"segment {p0}→{p1}"
        # the returned altitude drives the flight-level dash styling, so pin it too
        vol = volumes.corridor_segment_volume(np.array(p0, float), 0.0, np.array(p1, float), 4.0, cfg)
        assert abs(rec["z"] - vol.shape.center[2]) < 1e-9, f"box centre altitude for {p0}→{p1}"


def test_shipped_view_transform_zooms_about_a_fixed_point():
    """Runs the shipped zoom/pan maths in node: the world point under the cursor must not move.

    An anchor that drifts is the classic zoom bug — the map slides away as you scroll — and it is
    invisible to any Python-side test, since the transform only exists in the embedded JS.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available; cannot execute the shipped JS")
    html = viz_html._HTML
    # clampView / setZoom / resetView — the whole view transform, and nothing below it, since redraw
    # and wireView touch the DOM
    body = html[html.index("function clampView()"):html.index("function redraw()")]
    harness = """
const PAD = 20, W = 60000, H = 60000;
const cv = {width: 760, height: 760, style: {}};
const S = (cv.width - 2*PAD)/Math.max(W, H);
let Z = 1, VS = S, VX = 0, VY = 0;
const ZMIN = 1, ZMAX = 64;
const wx = px => (px - PAD)/VS + VX, wy = py => (cv.height - PAD - py)/VS + VY;
function redraw(){}
""" + body.replace("{{", "{").replace("}}", "}") + """
const out = [];
for (const [z, ax, ay] of [[8,300,400],[3.7,120,700],[64,700,60],[2,380,380]]) {
  Z = 1; VS = S; VX = 0; VY = 0;                       // fit, then zoom about (ax, ay)
  const before = [wx(ax), wy(ay)];
  setZoom(z, ax, ay);
  const after = [wx(ax), wy(ay)];
  out.push({z: Z, drift: [Math.abs(after[0]-before[0]), Math.abs(after[1]-before[1])],
            spanM: (cv.width - 2*PAD)/VS, VX, VY});
}
Z = 1; VS = S; VX = -1e9; VY = 1e9; clampView();       // the region cannot be panned off-screen
out.push({clampedAtFit: [VX, VY]});
console.log(JSON.stringify(out));
"""
    proc = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    got = json.loads(proc.stdout)

    for rec in got[:-1]:
        # the anchored point holds still to floating-point noise, at every zoom level
        assert max(rec["drift"]) < 1e-6, f"zoom to {rec['z']}x drifted by {rec['drift']} m"
        # and the visible span is exactly the region divided by the zoom
        assert abs(rec["spanM"] - 60000 / rec["z"]) < 1e-6
        assert rec["VX"] >= 0 and rec["VY"] >= 0
    assert got[-1]["clampedAtFit"] == [0, 0]   # at fit there is nowhere to pan to


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
            assert abs(qt / payload["qt"] - t) <= 0.5 / payload["qt"]


def test_sub_quantum_segment_forces_the_explicit_fallback():
    """A segment shorter than the position quantum must NOT be rebuilt in the browser.

    The browser only ever sees the decimetre-quantised path, so a centimetre-scale segment collapses to
    zero length there and JS draws an x-aligned box instead of the real corridor — measured at up to
    32 m off on a real `straight` run, whose final partial timestep leaves exactly such a remainder.
    Verifying against the full-precision centerline passed all of those silently; verifying against the
    quantised path is what makes them fall back.
    """
    cfg = SimConfig(planner="straight", horizon_s=600.0, region_size_m=(2200.0, 2200.0))
    res = run(cfg, requests=[FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0)])
    acc = res.accepted[0]
    cl = list(acc.centerline)
    # splice in a 3 cm near-vertical stub, the shape that quantisation destroys
    (p, t), (_, t_next) = cl[0], cl[1]
    stub_end = vec(float(p[0]), float(p[1]), float(p[2]) + 0.03)
    acc.centerline = [cl[0], (stub_end, t + (t_next - t) / 2), *cl[1:]]
    acc.volumes = volumes.build_corridor(acc.centerline, cfg) + [
        v for v in acc.volumes if not isinstance(v.shape, BoxSpec)]

    with pytest.warns(RuntimeWarning, match="could not have their corridor boxes rebuilt"):
        payload = viz_html._payload(res), viz_html.write_html(res, tempfile.mkstemp(suffix=".html")[1])
    assert payload[0]["explicit_box_flights"] == 1
    assert "b" in payload[0]["flights"][0]      # its polygons ship verbatim rather than being mis-drawn


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
    assert len(doctored["b"]) == len([v for v in vols if isinstance(v.shape, BoxSpec)])
    assert all(len(box) == 11 for box in doctored["b"])    # 8 xy coords + z + t0 + t1
    assert all("b" not in f for f in payload["flights"][1:])


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
    blob = _blob(html)
    raw = viz_html._scene_json(viz_html._payload(res))   # the exact bytes write_html compresses
    assert len(blob) < len(raw)                          # base64(gzip(scene)) still beats the raw dump
    assert raw not in html                               # the scene is compressed, not inlined
    assert all(c.isalnum() or c in "+/=" for c in blob)  # ...and what IS embedded is pure base64


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
