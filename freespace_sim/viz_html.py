"""Self-contained HTML replay — scrub time and watch FCFS deconfliction happen.

Emits a single standalone ``.html`` file (no server, no external assets): the reservation geometry
is serialised to JSON and drawn as projected polygons on a ``<canvas>``, with a play/pause/step/slider
transport. This is the free-space analog of the sibling project's ``viz_html.py`` — corridors and
hover cylinders appear and vanish with their ASTM time windows while drones glide along centerlines.

**The scene is stored compressed** (gzip → base64 → one JS string, inflated in the browser via
``DecompressionStream``), because a dense run writes hundreds of thousands of corridor boxes and the
naive dump reaches 78 MB at 4.7k flights / ~400 MB at 26k. Three encodings stack to ~145x smaller:

1. **Corridor boxes are not stored at all** — they are *rebuilt* in JS. Every box is
   ``volumes.corridor_segment_volume`` of one path segment, so its 4 footprint corners and both time
   bounds are a pure function of two adjacent path points plus ``corridor_width_m`` /
   ``corridor_height_m`` / ``time_buffer_s``. That alone is 72% of the naive payload. :func:`_rebuildable`
   *verifies* the identity per flight rather than assuming it, so a planner that reserves something
   other than the swept centerline keeps explicit polygons and still replays correctly.
2. **Coordinates are quantised integers**, delta-encoded per stream (see :data:`_Q` / :data:`_QT`).
   Path points advance by one lattice pitch per timestep, so the deltas collapse to a short run of
   repeated integers — which is what makes step 3 pay off.
3. **gzip**, which turns those runs into almost nothing.

Only the per-flight streams are quantised; every other number in the payload is plain metres/seconds.
"""

from __future__ import annotations

import base64
import gzip
import json
import warnings

import numpy as np

from . import metrics, volumes
from .geometry import BoxSpec, CylinderSpec
from .sim import SimResult
from .types import as_terminal
from .viz import box_footprint, flight_color_by_uss, result_uss_hues, uss_swatch_hex

_Q = 10        # position quantum: decimetres. One canvas pixel is ~79 m on a 60 km region, so this is
               # ~800x finer than anything that can be seen — and it is finite, unlike a float64 repr.
_QT = 1000     # time quantum: milliseconds. Deliberately NOT dt-steps: astar_shortcut splices refined
               # segments that need not land on the dt grid, and ms costs nothing after delta+gzip.
               # Both quanta must keep the streams inside the browser's Int32Array: at these values a
               # position overflows only past a 214,000 km region and a time past a 24-day run.
# A box counts as rebuildable only if the browser's reconstruction lands this close to what was actually
# reserved. It is deliberately NOT 1e-6: the browser rebuilds from the QUANTISED path, so a faithful
# reconstruction still differs by up to ~a quantum (more on short segments, where the same endpoint
# rounding rotates the direction further). 0.5 m is ~0.006 canvas pixels at 60 km — invisible — while
# still catching the failure that matters: a segment short enough that quantisation collapses it to zero
# length, or flips segment_frame's near-vertical branch, diverges by metres to tens of metres.
_REBUILD_TOL = 0.5           # metres
_REBUILD_TOL_T = 1.0 / _QT   # seconds — one time quantum, the most dequantisation can shift a window
# Warn once this share of flights ships explicit polygons. A few always will — the straight planner's
# final partial timestep leaves a centimetre-scale remainder segment that quantisation cannot preserve
# (~1.5% of flights, no measurable size cost). The threshold is there to catch the *systemic* failure:
# a planner or an archived corridor formula that makes the whole run non-rebuildable.
_FALLBACK_WARN_FRAC = 0.10


def _static_walls(result: SimResult):
    """The run's always-active terminal walls (permanent no-fly columns): from the live ledger
    (``SimResult.ledger._static_vols``) or a loaded run's ``static_walls`` (populated by ``load_run`` from
    ``ledger_end.parquet``). Empty otherwise. These are NOT in any accepted intent's volumes, so the replay
    must render them separately — otherwise a denial's blocker is invisible (see the telemetry design §10)."""
    led = getattr(result, "ledger", None)
    if led is not None and getattr(led, "_static_vols", None):
        return list(led._static_vols)
    return list(getattr(result, "static_walls", None) or [])


def _delta(vals: list[int]) -> list[int]:
    """First value, then successive differences — the form gzip compresses best.

    Written as a zero-seeded running difference so it is visibly the exact inverse of the JS
    ``undelta``, which also starts from 0; the first element is not a special case in either."""
    out, prev = [], 0
    for v in vals:
        out.append(v - prev)
        prev = v
    return out


def _scene_json(payload: dict) -> str:
    """The exact bytes that get gzipped into the page — shared with the tests so a size or
    round-trip assertion cannot drift from what ``write_html`` actually compresses."""
    return json.dumps(payload, separators=(",", ":"))


def _footprint_xy(spec: BoxSpec) -> tuple[float, ...]:
    """The box's 4 ground-plane corners as 8 plain floats — the same quantity :func:`viz.box_footprint`
    returns, but in scalars, since this runs once per box per flight on the write path.

    ``rot`` is row-major with columns as the local axes, so local-x in world is ``(rot[0], rot[3])`` and
    local-y is ``(rot[1], rot[4])``; a corner is ``centre ± (L/2)·x ± (W/2)·y``."""
    cx, cy = spec.center[0], spec.center[1]
    lx, wy = spec.extents[0] / 2.0, spec.extents[1] / 2.0
    ax, ay = lx * spec.rot[0], lx * spec.rot[3]
    bx, by = wy * spec.rot[1], wy * spec.rot[4]
    return (cx + ax + bx, cy + ay + by, cx + ax - bx, cy + ay - by,
            cx - ax - bx, cy - ay - by, cx - ax + bx, cy - ay + by)


def _rebuildable(intent, cfg, quantised_centerline) -> bool:
    """True when the browser's reconstruction of this flight's corridor boxes matches what was reserved.

    When it does, the payload can drop the boxes entirely and let JS rebuild them — the single biggest
    lever on file size (see the module docstring).

    Two details make this an honest check rather than a tautology. First, we compare against the real
    :func:`volumes.build_corridor` instead of reimplementing its geometry, so the check cannot drift from
    the thing it is checking (the JS copy of the formula is pinned separately, by executing it in
    ``test_viz``). Second — and this is the subtle one — we build from the **quantised** path, because
    that is what the browser actually has. Verifying against the full-precision centerline would pass a
    segment that quantisation collapses to zero length or pushes across ``segment_frame``'s near-vertical
    branch, and the replay would then draw a corridor tens of metres from the one that was reserved.
    """
    got = [v for v in (intent.volumes or []) if isinstance(v.shape, BoxSpec)]
    want = volumes.build_corridor(quantised_centerline, cfg)
    if len(got) != len(want):
        return False
    return all(
        abs(a.t_start - b.t_start) <= _REBUILD_TOL_T and abs(a.t_end - b.t_end) <= _REBUILD_TOL_T
        and abs(a.shape.center[2] - b.shape.center[2]) <= _REBUILD_TOL      # level dash styling
        and all(abs(p - q) <= _REBUILD_TOL for p, q in zip(_footprint_xy(a.shape), _footprint_xy(b.shape)))
        for a, b in zip(got, want))


def _payload(result: SimResult) -> dict:
    """Flatten the accepted intents into a compact, JSON-serialisable scene description.

    The clock spans the **realized operation** — :func:`metrics.simulation_window`, i.e. the first
    airspace reservation through the last one to clear — not ``[0, cfg.horizon_s]``. ``horizon_s`` is a
    planner *envelope*, not a schedule: flights are filed from t=0 but depart on a lead of hundreds of
    seconds, and they all land long before the envelope closes. Anchoring on it left the density replays
    with 768 s of nothing at the head and 3329 s at the tail — 57% of the slider scrubbing through an
    empty sky. This is a display choice only; the measurement window is a separate concern that
    :func:`metrics.steady_state_window` owns (issue #25)."""
    cfg = result.config
    hues = result_uss_hues(result)
    usses = sorted(hues)
    uss_ix = {u: i for i, u in enumerate(usses)}
    play_start, play_end = metrics.simulation_window(result)

    def q(v) -> int:                       # metres → quantised int
        return round(float(v) * _Q)

    def qt(t) -> int:                      # absolute seconds → quantised int, relative to play_start
        return round((float(t) - play_start) * _QT)

    flights, n_explicit = [], 0
    for intent in result.accepted:
        req = intent.request
        r, g, b = flight_color_by_uss(req.uss_id, req.flight_id, hues)
        cl = list(intent.centerline or [])
        vols = intent.volumes or []
        # Quantise ONCE, then both encode from it and verify against it — so what we check is exactly
        # what the browser will reconstruct, not the full-precision path it never sees.
        qx = [q(p[0]) for p, _ in cl]
        qy = [q(p[1]) for p, _ in cl]
        qz = [q(p[2]) for p, _ in cl]
        qtime = [qt(t) for _, t in cl]
        rec = {
            "i": req.flight_id,
            "u": uss_ix[req.uss_id],
            "k": f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}",
            "x": _delta(qx),
            "y": _delta(qy),
            "z": _delta(qz),                             # altitude → flight-level styling
            "t": _delta(qtime),
            "o": [q(req.origin[0]), q(req.origin[1])],
            "d": [q(req.dest[0]), q(req.dest[1])],
            "c": [[q(v.shape.cx), q(v.shape.cy), q(v.shape.radius), qt(v.t_start), qt(v.t_end)]
                  for v in vols if isinstance(v.shape, CylinderSpec)],
        }
        dequantised = [(np.array([x / _Q, y / _Q, z / _Q]), t / _QT + play_start)
                       for x, y, z, t in zip(qx, qy, qz, qtime)]
        if not _rebuildable(intent, cfg, dequantised):   # explicit polygons for just this flight
            n_explicit += 1
            rec["b"] = [[*(q(c) for corner in box_footprint(v.shape) for c in corner),
                         q(v.shape.center[2]), qt(v.t_start), qt(v.t_end)]
                        for v in vols if isinstance(v.shape, BoxSpec)]
        flights.append(rec)

    # The hex lattice only exists if an A*-based planner ran (astar / astar_milp / astar_shortcut). When
    # it did, expose its circumradius so the replay can overlay the exact grid A* searched on.
    hex_available = "astar" in cfg.planner
    from .planner.hexgrid import circumradius
    # always-active terminal WALLS (permanent no-fly columns) live in the ledger, NOT in any accepted
    # intent's volumes — render them as permanent overlays so a denial's blocker is visible. Each also
    # carries its EXIT-LANE radius: where the hub's reserved lanes actually begin. The annulus between
    # the two is flown but deliberately unreserved (the vertiport deconflicts inside it tactically), so
    # without the ring the replay shows corridors starting in mid-air a corridor-width off the column.
    terms = {}
    for intent in result.accepted:
        for t in (intent.request.origin_terminal, intent.request.dest_terminal):
            if (t := as_terminal(t)) is not None:
                terms[str(t.id)] = t
    walls = []
    for v in _static_walls(result):
        if not isinstance(v.shape, CylinderSpec):
            continue
        tid = None if v.terminal_id is None else str(v.terminal_id)
        term = terms.get(tid)
        walls.append({"cx": float(v.shape.cx), "cy": float(v.shape.cy), "r": float(v.shape.radius),
                      "tid": tid,
                      # volumes.exit_radius is the single source of truth for the lane edge — the same
                      # radius the A* fold, the commit, and TerminalCapacity.exit_clear all root on.
                      "er": volumes.exit_radius(term, cfg) if term is not None else None})
    return {
        "v": 2,                              # payload schema version
        "simulation_start_s": play_start,
        "simulation_end_s": play_end,
        "simulation_duration_s": play_end - play_start,
        "horizon": play_end,                 # compatibility alias for older payload consumers
        "dt": cfg.dt_s,
        "region": list(cfg.region_size_m),
        "q": _Q, "qt": _QT,                  # quantisation of the per-flight int streams
        "corridor_w": cfg.corridor_width_m,  # the three scalars that let JS rebuild every corridor box
        "corridor_h": cfg.corridor_height_m,
        "t_buffer": cfg.time_buffer_s,
        "flights": flights,
        "explicit_box_flights": n_explicit,  # flights whose boxes could NOT be rebuilt (0 for every
                                             # centerline-swept planner; non-zero is worth investigating)
        "walls": walls,                      # permanent terminal no-fly columns (always-active)
        "usses": usses,                      # index order for each flight's "u"
        "uss_colors": {uid: uss_swatch_hex(uid, hues) for uid in usses},   # legend / per-USS slice
        "hex_available": hex_available,
        "hex_R": circumradius(cfg) if hex_available else 0.0,
        "planner": cfg.planner,
        "flight_levels": [round(float(z), 1) for z in cfg.flight_levels_m],
    }


_SEG_POLY_JS = """
// Rebuild segment i's corridor box exactly as volumes.corridor_segment_volume does: extend the segment
// by half its own cross-section ALONG travel (anisotropic — corridor width in the horizontal plane,
// corridor height in the vertical, so a pure climb doesn't balloon in z), then take the footprint of the
// oriented box. Lateral axis = unit(world_up x travel), falling back to world-x when near-vertical —
// mirrors geometry.segment_frame.
// sqrt(a*a+b*b) rather than Math.hypot: V8's hypot does overflow-safe scaling we cannot need at these
// magnitudes, and it costs 4.5x here — this runs for every box of every active flight, every frame.
const QUAD = [[1,1],[1,-1],[-1,-1],[-1,1]];               // local corners, same order as viz.box_footprint
function segPoly(f, i, out){
  const x0=f.x[i]*IQ, y0=f.y[i]*IQ, z0=f.z[i]*IQ;
  const x1=f.x[i+1]*IQ, y1=f.y[i+1]*IQ, z1=f.z[i+1]*IQ;
  const dx=x1-x0, dy=y1-y0, dz=z1-z0;
  let L=Math.sqrt(dx*dx+dy*dy+dz*dz), ux=1, uy=0, uz=0;
  if(L<1e-9) L=0; else { ux=dx/L; uy=dy/L; uz=dz/L; }
  const h=Math.sqrt(ux*ux+uy*uy), ea=CW*h, eb=CH*uz;
  const hl = L/2 + 0.5*Math.sqrt(ea*ea+eb*eb);
  const cx=(x0+x1)/2, cy=(y0+y1)/2;
  let vx, vy, n;
  if(Math.abs(uz)<0.99){ n=h||1; vx=-uy/n; vy=ux/n; }
  else { n=Math.sqrt(uy*uy+uz*uz)||1; vx=0; vy=-uz/n; }
  for(let k=0;k<4;k++){ const a=QUAD[k][0]*hl, b=QUAD[k][1]*CW/2;
    out[2*k]=cx+a*ux+b*vx; out[2*k+1]=cy+a*uy+b*vy; }
  return (z0+z1)/2;                                       // box centre altitude → level dash styling
}
"""
"""The corridor-box rebuild, kept OUT of ``_HTML`` so a test can execute this exact source in node and
pin it against :func:`volumes.corridor_segment_volume`. Inlining it would make the shipped formula
untestable — the drawn geometry could then silently drift from what was actually reserved, and because
:func:`_rebuildable` compares Python against Python, nothing would notice. Depends on the globals
``IQ`` / ``CW`` / ``CH``, which ``boot()`` sets from the payload."""


_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>FCFS replay</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:0;background:#0e1116;color:#d7dde3}}
 #wrap{{display:flex;flex-direction:column;align-items:center;gap:8px;padding:12px}}
 canvas{{background:#161b22;border:1px solid #30363d;border-radius:6px}}
 #bar{{display:flex;align-items:center;gap:8px;width:760px;flex-wrap:wrap}}
 #bar input[type=range]{{flex:1;min-width:240px}}
 button,select{{background:#21262d;color:#d7dde3;border:1px solid #30363d;border-radius:5px;
        padding:5px 10px;cursor:pointer}}
 #t{{font-variant-numeric:tabular-nums;min-width:120px}}
 label.tog{{display:flex;align-items:center;gap:5px;font-size:13px;color:#8b949e;cursor:pointer}}
 h3{{margin:6px 0 0}} small{{color:#8b949e}} #err{{color:#f87171;max-width:760px}}
</style></head><body><div id="wrap">
 <h3>FCFS strategic deconfliction — free-space replay</h3>
 <small>corridors = trajectory intents · circles = hover reservations · dots = drones · dashed = straight origin→dest<br>
 amber disc = permanent terminal column (no-fly) · fine ring outside it = where its reserved exit lanes begin<br>
 scroll to zoom · drag to pan · double-click to zoom in · <kbd>0</kbd> to fit</small>
 <canvas id="c" width="760" height="760"></canvas>
 <div id="bar"><button id="play">▶ play</button>
  <button id="back" title="step back one timestep">⏮</button>
  <button id="fwd" title="step forward one timestep">⏭</button>
  <input id="slider" type="range" min="{start}" max="{end}" value="{start}" step="1">
  <span id="t">decompressing…</span>
  <label class="tog" for="speed">speed
   <select id="speed" title="playback speed">
    <option value="0.25">0.25&times;</option>
    <option value="0.5">0.5&times;</option>
    <option value="1" selected>1&times;</option>
    <option value="2">2&times;</option>
    <option value="4">4&times;</option>
    <option value="8">8&times;</option>
   </select></label>
  <label class="tog" id="hexWrap"><input type="checkbox" id="hexToggle"> hex grid (A*)<span id="hexHint"></span></label>
  <span class="tog" title="scroll to zoom at the cursor · drag to pan · double-click to zoom in">
   <button id="zoomOut" title="zoom out (-)">&minus;</button>
   <span id="zoomLbl" style="min-width:34px;text-align:center">1.0&times;</span>
   <button id="zoomIn" title="zoom in (+)">+</button>
   <button id="zoomFit" title="fit the whole region (0)">fit</button></span>
  <span id="legend" style="display:flex;gap:10px;flex-wrap:wrap"></span>
 </div>
 <div id="err"></div>
</div><script>
// The scene is gzipped JSON in base64. See freespace_sim/viz_html.py for the encoding; the two halves
// have to agree on the quantisation and on the corridor-box formula rebuilt in segPoly() below.
const B64 = "{b64}";
const START = {start}, END_ABS = {end};   // absolute seconds: the realized run bounds
const LEVEL_DASH = [[], [6,4], [2,4], [1,5], [8,3,2,3]];  // solid / dash / dot / fine / dash-dot — per level
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const slider = document.getElementById('slider');
const hidden = new Set();                                 // USS indices toggled off via the legend
let DATA=null, LEVELS=[], END=END_ABS, DURATION=0, IQ=1, IQT=1;             // IQ/IQT: quantised unit → metres / seconds
let W=0, H=0, S=1, CW=0, CH=0, TBQ=0;                     // TBQ: the ASTM time buffer, in quantised units
let Z=1, VS=1, VX=0, VY=0;                                // zoom, effective px/m (S*Z), view origin (world m)
const ZMIN=1, ZMAX=64;
let buckets=[], BW=1;                                     // time index: buckets[k] = flights active then
const PAD = 20;
const sx = x => PAD + (x - VX)*VS, sy = y => cv.height - PAD - (y - VY)*VS;   // flip y: north is up
const wx = px => (px - PAD)/VS + VX, wy = py => (cv.height - PAD - py)/VS + VY;   // screen px → world m

// ---------------------------------------------------------------- load
async function inflate(b64){{
  if(typeof DecompressionStream !== 'function')
    throw new Error('this browser has no DecompressionStream (needs Chrome 80+ / Safari 16.4+ / Firefox 113+)');
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) bytes[i] = bin.charCodeAt(i);   // charCodeAt loop beats Uint8Array.from here
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new TextDecoder().decode(await new Response(stream).arrayBuffer());
}}
function undelta(a){{                                     // delta stream → absolute values, unboxed
  const out = new Int32Array(a.length); let v = 0;
  for(let i=0;i<a.length;i++){{ v += a[i]; out[i] = v; }}
  return out;
}}
async function boot(){{
  DATA = JSON.parse(await inflate(B64));
  IQ = 1/DATA.q; IQT = 1/DATA.qt;
  LEVELS = DATA.flight_levels || [];
  END = DATA.simulation_end_s; DURATION = Math.max(0, DATA.simulation_duration_s);
  [W, H] = DATA.region; S = (cv.width - 2*PAD)/Math.max(W, H); VS = S; Z = 1; VX = VY = 0;
  CW = DATA.corridor_w; CH = DATA.corridor_h; TBQ = DATA.t_buffer*DATA.qt;
  for(const f of DATA.flights){{ f.x=undelta(f.x); f.y=undelta(f.y); f.z=undelta(f.z); f.t=undelta(f.t); }}
  buildIndex();
  if(!DATA.hex_available) document.getElementById('hexWrap').style.display='none';
  buildLegend(); buildLevelLegend(); wireTransport(); wireView();
  draw(START);
}}
boot().catch(e => {{ document.getElementById('t').textContent = 'failed to load';
  document.getElementById('err').textContent = 'Replay could not be decompressed: ' + e.message; }});

// ---------------------------------------------------------------- geometry
{seg_poly}
function levelOf(z){{ let bi=0, bd=1e9;
  for(let i=0;i<LEVELS.length;i++){{ const d=Math.abs(LEVELS[i]-z); if(d<bd){{ bd=d; bi=i; }} }}
  return bi % LEVEL_DASH.length; }}
// first index with a[i] > v — the active segments of a flight are a contiguous range, because both
// bounds are monotone in i.
function upperBound(a, v){{ let lo=0, hi=a.length;
  while(lo<hi){{ const m=(lo+hi)>>1; if(a[m]<=v) lo=m+1; else hi=m; }} return lo; }}
function posAt(f, tq){{                                   // interpolated [x, y, z] in metres, or null
  const T=f.t, n=T.length;
  if(!n || tq<T[0] || tq>T[n-1]) return null;
  if(n===1) return [f.x[0]*IQ, f.y[0]*IQ, f.z[0]*IQ];
  const i = Math.min(n-2, Math.max(0, upperBound(T, tq)-1));
  const span = T[i+1]-T[i], u = span<=0 ? 0 : (tq-T[i])/span;
  return [(f.x[i]+(f.x[i+1]-f.x[i])*u)*IQ, (f.y[i]+(f.y[i+1]-f.y[i])*u)*IQ,
          (f.z[i]+(f.z[i+1]-f.z[i])*u)*IQ];
}}

// ---------------------------------------------------------------- time index
// Without this a frame would test every box in the run (>400k at 4.7k flights, millions at 26k) just to
// find the few hundred on screen. Bucketing by activity window makes a frame cost what it draws.
function buildIndex(){{
  BW = Math.max(DATA.dt, DURATION/1024 || DATA.dt);
  const n = Math.max(1, Math.ceil(DURATION/BW) + 1);
  buckets = Array.from({{length: n}}, () => []);
  DATA.flights.forEach((f, fi) => {{
    // Widen over EVERY drawable: a flight with no path still has hover cylinders (and, if its boxes
    // weren't rebuildable, explicit polygons) to render — keying off the path alone would hide it.
    let lo = Infinity, hi = -Infinity;
    if(f.t.length){{ lo = f.t[0]*IQT - DATA.t_buffer; hi = f.t[f.t.length-1]*IQT + DATA.t_buffer; }}
    for(const c of f.c){{ lo = Math.min(lo, c[3]*IQT); hi = Math.max(hi, c[4]*IQT); }}
    for(const b of (f.b || [])){{ lo = Math.min(lo, b[9]*IQT); hi = Math.max(hi, b[10]*IQT); }}
    if(lo > hi) return;                                    // nothing this flight can ever draw
    const a = Math.max(0, Math.floor(lo/BW)), z = Math.min(n-1, Math.floor(hi/BW));
    for(let k=a;k<=z;k++) buckets[k].push(fi);             // pushed in flight order → stable z-ordering
  }});
}}

// ---------------------------------------------------------------- draw
function drawHexGrid(){{
  const R = DATA.hex_R; if(!R) return;
  const SQRT3 = Math.sqrt(3);
  // A hex is 2R across. Below ~6 px you cannot see it IS a hexagon, and drawing it anyway costs ~292k
  // strokes on a 60 km region (280 ms/frame, measured) — so say "zoom in" rather than stutter to draw
  // something nobody can read. Above the threshold, cull to the viewport: cost then scales with what
  // is actually on screen, which is what makes the overlay usable at all.
  if(2*R*VS < 6){{ hexHint(true); return; }}
  hexHint(false);
  ctx.strokeStyle = '#39414f'; ctx.lineWidth = 0.4;          // faint lattice beneath the corridors
  const x0 = Math.max(0, wx(PAD)), x1 = Math.min(W, wx(cv.width - PAD));
  const y0 = Math.max(0, wy(cv.height - PAD)), y1 = Math.min(H, wy(PAD));
  const rMax = Math.ceil(y1/(1.5*R)) + 1;
  for(let r=Math.floor(y0/(1.5*R)) - 1; r<=rMax; r++){{
    const qLo = Math.floor(x0/(SQRT3*R) - r/2) - 1, qHi = Math.ceil(x1/(SQRT3*R) - r/2) + 1;
    for(let q=qLo; q<=qHi; q++){{
      const cx = R*SQRT3*(q + r/2), cy = R*1.5*r;
      ctx.beginPath();
      for(let i=0;i<6;i++){{ const a = Math.PI/180*(60*i - 30);   // pointy-top vertices
        const x = cx + R*Math.cos(a), y = cy + R*Math.sin(a);
        i ? ctx.lineTo(sx(x),sy(y)) : ctx.moveTo(sx(x),sy(y)); }}
      ctx.closePath(); ctx.stroke();
    }}
  }}
}}
let hexHinted = null;
function hexHint(on){{                                     // tell the user WHY the lattice is absent
  if(on === hexHinted) return;
  hexHinted = on;
  document.getElementById('hexHint').textContent = on ? ' — zoom in' : '';
}}
function strokePoly(p, color, z){{
  ctx.beginPath(); ctx.moveTo(sx(p[0]),sy(p[1]));
  for(let k=1;k<4;k++) ctx.lineTo(sx(p[2*k]),sy(p[2*k+1]));
  ctx.closePath(); ctx.fillStyle=color+'55'; ctx.fill();
  ctx.save(); ctx.setLineDash(LEVEL_DASH[levelOf(z)]); ctx.strokeStyle=color; ctx.stroke(); ctx.restore();
}}
const POLY = new Float64Array(8);                            // scratch, reused every segment
function draw(t){{
  if(!DATA) return;
  ctx.clearRect(0,0,cv.width,cv.height);
  // set lineWidth explicitly: canvas state survives across frames, so the region border used to render
  // at whatever width the PREVIOUS frame's last flight happened to leave behind.
  ctx.strokeStyle='#30363d'; ctx.lineWidth=0.6; ctx.strokeRect(sx(0),sy(H),W*VS,H*VS);
  if(document.getElementById('hexToggle').checked) drawHexGrid();
  for(const wl of (DATA.walls||[])){{                     // permanent terminal walls (always-active no-fly)
    ctx.beginPath(); ctx.arc(sx(wl.cx),sy(wl.cy),wl.r*VS,0,2*Math.PI);
    ctx.fillStyle='#f59e0b1f'; ctx.fill();
    ctx.save(); ctx.setLineDash([4,4]); ctx.strokeStyle='#b45309aa'; ctx.lineWidth=1; ctx.stroke(); ctx.restore();
    // The exit-lane ring: where this hub's RESERVED lanes begin. Drawn only once it is more than a
    // pixel outside the column — at fit the gap is corridor_width/2 (0.4 px on a 60 km region), so
    // below that it is 490 arcs a frame drawing nothing you can see.
    if(wl.er && (wl.er - wl.r)*VS >= 1.5){{
      ctx.save(); ctx.beginPath(); ctx.arc(sx(wl.cx),sy(wl.cy),wl.er*VS,0,2*Math.PI);
      ctx.setLineDash([2,3]); ctx.strokeStyle='#fbbf2466'; ctx.lineWidth=1; ctx.stroke(); ctx.restore();
    }}
  }}
  const tq = (t - START)*DATA.qt, lo = tq - TBQ, hi = tq + TBQ;
  const bk = buckets[Math.max(0, Math.min(buckets.length-1, Math.floor((t - START)/BW)))] || [];
  let nActive = 0;
  for(const fi of bk){{
    const fl = DATA.flights[fi];
    if(hidden.has(fl.u)) continue;                       // per-USS slice (legend toggles)
    let on = false;
    ctx.lineWidth = 0.6;
    if(fl.b){{                                           // explicit polygons (planner we can't rebuild)
      for(const bx of fl.b){{ if(bx[9]>tq || tq>=bx[10]) continue; on=true;
        for(let k=0;k<8;k++) POLY[k]=bx[k]*IQ;
        strokePoly(POLY, fl.k, bx[8]*IQ); }}
    }} else {{      // segment i spans [t[i]-buf, t[i+1]+buf), so it is live when t[i] <= hi && t[i+1] > lo.
                   // Both bounds are upperBound: the window is half-open, and on a dt-aligned clock the
                   // trailing edge lands exactly on t[i+1] == lo every frame.
      const a = Math.max(0, upperBound(fl.t, lo) - 1);
      const z = Math.min(fl.t.length - 2, upperBound(fl.t, hi) - 1);
      for(let i=a;i<=z;i++){{ on=true; strokePoly(POLY, fl.k, segPoly(fl, i, POLY)); }}
    }}
    for(const cy of fl.c){{ if(cy[3]>tq || tq>=cy[4]) continue; on=true;
      ctx.beginPath(); ctx.arc(sx(cy[0]*IQ),sy(cy[1]*IQ),cy[2]*IQ*VS,0,2*Math.PI);
      ctx.fillStyle=fl.k+'33'; ctx.fill(); ctx.strokeStyle=fl.k; ctx.stroke(); }}
    const p = posAt(fl, tq);
    if(p){{ on=true; ctx.beginPath(); ctx.arc(sx(p[0]),sy(p[1]),4,0,2*Math.PI);
      ctx.fillStyle=fl.k; ctx.fill(); ctx.strokeStyle='#000'; ctx.lineWidth=0.5; ctx.stroke();
      if(LEVELS.length>1){{ ctx.fillStyle='#aeb6c2'; ctx.font='8px monospace';   // altitude readout
        ctx.fillText(Math.round(p[2])+'m', sx(p[0])+6, sy(p[1])-4); }} }}
    if(on){{                                          // dashed straight origin→dest for active flights
      ctx.save(); ctx.setLineDash([6,5]); ctx.lineWidth=1; ctx.strokeStyle=fl.k+'aa';
      ctx.beginPath(); ctx.moveTo(sx(fl.o[0]*IQ),sy(fl.o[1]*IQ));
      ctx.lineTo(sx(fl.d[0]*IQ),sy(fl.d[1]*IQ)); ctx.stroke(); ctx.restore();
      nActive++;
    }}
  }}
  document.getElementById('t').textContent = 't = '+Math.round(t)+' s  ('+nActive+' active)';
}}

// ---------------------------------------------------------------- zoom / pan
// The view is a world-space window: VX/VY is its bottom-left corner in metres, VS its px-per-metre.
// Sizes that should stay legible (drone dots, line widths) are already in px and deliberately do NOT
// scale; sizes that are real geometry (hover radii, the region border) go through VS.
function clampView(){{                                    // keep the region from sliding off-screen
  VX = Math.min(Math.max(0, W - (cv.width - 2*PAD)/VS), Math.max(0, VX));
  VY = Math.min(Math.max(0, H - (cv.height - 2*PAD)/VS), Math.max(0, VY));
}}
function setZoom(z, ax, ay){{                             // zoom about the canvas point (ax, ay)
  const nz = Math.max(ZMIN, Math.min(ZMAX, z));
  if(nz === Z) return;
  const bx = wx(ax), by = wy(ay);                         // world point to hold still
  Z = nz; VS = S*Z;
  VX = bx - (ax - PAD)/VS;
  VY = by - (cv.height - PAD - ay)/VS;
  clampView(); redraw();
}}
function resetView(){{ Z = 1; VS = S; VX = VY = 0; redraw(); }}
function redraw(){{
  document.getElementById('zoomLbl').textContent = Z.toFixed(Z < 10 ? 1 : 0) + '\u00d7';
  draw(+slider.value);
}}
function wireView(){{
  cv.style.cursor = 'grab';
  cv.addEventListener('wheel', e => {{                     // wheel / trackpad pinch → zoom at cursor
    e.preventDefault();
    setZoom(Z * Math.pow(1.0015, -e.deltaY), e.offsetX, e.offsetY);
  }}, {{passive: false}});
  let dragging = false, lx = 0, ly = 0;
  cv.addEventListener('mousedown', e => {{ dragging = true; lx = e.offsetX; ly = e.offsetY;
    cv.style.cursor = 'grabbing'; }});
  window.addEventListener('mouseup', () => {{ dragging = false; cv.style.cursor = 'grab'; }});
  cv.addEventListener('mousemove', e => {{
    if(!dragging) return;
    VX -= (e.offsetX - lx)/VS; VY += (e.offsetY - ly)/VS;   // sy flips y, so dragging down raises VY
    lx = e.offsetX; ly = e.offsetY; clampView(); redraw();
  }});
  cv.addEventListener('dblclick', e => setZoom(Z*2, e.offsetX, e.offsetY));
  const mid = () => [cv.width/2, cv.height/2];
  document.getElementById('zoomIn').onclick  = () => setZoom(Z*1.6, ...mid());
  document.getElementById('zoomOut').onclick = () => setZoom(Z/1.6, ...mid());
  document.getElementById('zoomFit').onclick = resetView;
  document.addEventListener('keydown', e => {{             // + / - / 0, ignoring the transport keys
    if(e.key === '+' || e.key === '=') setZoom(Z*1.6, ...mid());
    else if(e.key === '-' || e.key === '_') setZoom(Z/1.6, ...mid());
    else if(e.key === '0') resetView();
  }});
}}

// ---------------------------------------------------------------- transport
let playing=false, speed=1, clock=START;
function step(d){{ playing=false; document.getElementById('play').textContent='▶ play';
  clock = Math.max(START, Math.min(END, +slider.value + d)); slider.value=clock; draw(clock); }}
// the play position is a FLOAT clock, not the slider value — the range input snaps to step=1, so it
// cannot hold sub-unit advances and speeds < 1 would round away. At 1x the realized run plays in
// ~10s (60fps · duration/600 per frame); speed scales that per-frame step (0.25x ⇒ 40s, 8x ⇒ ~1.25s).
function tick(){{ if(!playing || DURATION<=0) return; clock += speed*DURATION/600;
  if(clock>=END){{ clock=END; slider.value=clock; draw(clock); playing=false;
    document.getElementById('play').textContent='▶ play'; return; }}
  slider.value=clock; draw(clock); requestAnimationFrame(tick); }}
function wireTransport(){{
  slider.oninput=()=>{{ clock=+slider.value; draw(clock); }};   // scrubbing re-seats the play clock
  document.getElementById('back').onclick=()=>step(-DATA.dt);   // one timestep back
  document.getElementById('fwd').onclick=()=>step(+DATA.dt);    // one timestep forward
  document.getElementById('hexToggle').onchange=()=>draw(+slider.value);
  document.getElementById('speed').onchange=function(){{ speed=+this.value; }};
  document.getElementById('play').onclick=function(){{ playing=!playing;
    if(playing && DURATION<=0) playing=false;
    this.textContent = playing?'⏸ pause':'▶ play'; if(playing){{ clock=+slider.value;
      if(clock>=END){{ clock=START; slider.value=clock; draw(clock); }} tick(); }} }};
  document.addEventListener('keydown', e=>{{      // ← / → step one timestep, space toggles play
    if(e.key==='ArrowLeft') step(-DATA.dt);
    else if(e.key==='ArrowRight') step(+DATA.dt);
    else if(e.key===' '){{ e.preventDefault(); document.getElementById('play').click(); }}
  }});
}}
function buildLegend(){{                          // per-USS show/hide (only when >1 operator flew)
  const usses = DATA.usses || [];
  if(usses.length < 2) return;
  const legend = document.getElementById('legend');
  usses.forEach((u, ui) => {{
    const lab = document.createElement('label'); lab.className = 'tog';
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = true;
    cb.onchange = ()=>{{ cb.checked ? hidden.delete(ui) : hidden.add(ui); draw(+slider.value); }};
    const sw = document.createElement('span');
    sw.style.cssText = 'display:inline-block;width:11px;height:11px;border-radius:2px;background:'+DATA.uss_colors[u];
    lab.appendChild(cb); lab.appendChild(sw); lab.appendChild(document.createTextNode(u));
    legend.appendChild(lab);
  }});
}}
function buildLevelLegend(){{                     // flight-level dash key (A* multi-altitude)
  if(LEVELS.length < 2) return;
  const legend = document.getElementById('legend');
  const NAMES = ['solid','dashed','dotted','fine','dash-dot'];
  const tag = document.createElement('span'); tag.style.cssText='opacity:.6'; tag.textContent='levels:';
  legend.appendChild(tag);
  LEVELS.forEach((z,i)=>{{ const s=document.createElement('span'); s.style.cssText='opacity:.85;font-size:11px';
    s.textContent = Math.round(z)+'m '+NAMES[i % NAMES.length]; legend.appendChild(s); }});
}}
</script></body></html>"""


def write_html(result: SimResult, out) -> str:
    """Render the realized run to a standalone HTML scrubber; return the output path."""
    payload = _payload(result)
    n_explicit, n_flights = payload["explicit_box_flights"], len(payload["flights"])
    if n_flights and n_explicit > _FALLBACK_WARN_FRAC * n_flights:
        # Loud, because this is the size lever failing: these flights ship their polygons verbatim, which
        # is what took a dense replay to 78 MB. A planner that stopped reserving the swept centerline —
        # or an archive planned under an older corridor formula — regresses silently otherwise.
        warnings.warn(
            f"replay: {n_explicit}/{n_flights} flights could not have their corridor boxes rebuilt from "
            f"the path (planner={payload['planner']}); storing explicit polygons for those, so {out} "
            f"will be substantially larger",
            RuntimeWarning, stacklevel=2)
    blob = _scene_json(payload).encode()
    # mtime=0 so the same run always produces a byte-identical file (gzip stamps the clock otherwise).
    b64 = base64.b64encode(gzip.compress(blob, compresslevel=9, mtime=0)).decode()
    html = _HTML.format(start=json.dumps(payload["simulation_start_s"]),
                        end=json.dumps(payload["simulation_end_s"]),
                        b64=b64, seg_poly=_SEG_POLY_JS)
    # encoding is explicit: the document declares utf-8 and the transport bar is full of non-ASCII
    # glyphs, so a C/cp1252 locale would otherwise mojibake or raise at the very end of a long run.
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return str(out)
