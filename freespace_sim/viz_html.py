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
   *verifies* sub-pixel fidelity per flight rather than assuming it, so a planner that reserves something
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
import math
import warnings

import numpy as np

from . import metrics, volumes
from .geometry import BoxSpec, CylinderSpec
from .planner import uses_hex_lattice
from .sim import SimResult
from .types import as_terminal
from .viz import box_footprint, flight_color_by_uss, result_uss_hues, uss_swatch_hex

_Q = 10        # minimum position quantum: decimetres. Smaller regions use a finer adaptive quantum so
               # reconstruction remains sub-pixel even at maximum zoom (see _position_q).
_QT = 1000     # time quantum: milliseconds. Deliberately NOT dt-steps: astar_shortcut splices refined
               # segments that need not land on the dt grid, and ms costs nothing after delta+gzip.
               # Streams which exceed Int32 are selectively widened in the browser; ordinary runs retain
               # the smaller typed arrays.
# These mirror the literal canvas/view constants in _HTML; a regression test pins the two sides together.
_CANVAS_PX = 760
_PAD_PX = 20
_MAX_ZOOM = 64
_REBUILD_ERROR_PX = 0.5
_REBUILD_TOL = 0.5           # metres; cap retained for large-region payloads
_REBUILD_TOL_T = 1.0 / _QT   # seconds — one time quantum, the most dequantisation can shift a window
_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1
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


def _rebuild_tolerance_m(cfg) -> float:
    """Largest world-space reconstruction error that remains half a pixel at maximum zoom."""
    fit_px_per_m = (_CANVAS_PX - 2 * _PAD_PX) / max(cfg.region_size_m)
    return min(_REBUILD_TOL, _REBUILD_ERROR_PX / (fit_px_per_m * _MAX_ZOOM))


def _position_q(cfg) -> int:
    """Position units/metre, tightened on small regions to leave room for angular quantisation error."""
    return max(_Q, math.ceil(2.0 / _rebuild_tolerance_m(cfg)))


def _needs_float64(*streams: list[int]) -> bool:
    """Whether an undelta'd stream cannot be represented by a browser ``Int32Array``."""
    return any(v < _INT32_MIN or v > _INT32_MAX for stream in streams for v in stream)


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


def _rebuildable(intent, cfg, quantised_centerline, tolerance_m: float) -> bool:
    """True when the browser's reconstruction stays within ``tolerance_m`` of what was reserved.

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
        and abs(a.shape.center[2] - b.shape.center[2]) <= tolerance_m      # level dash styling
        and all(abs(p - q) <= tolerance_m
                for p, q in zip(_footprint_xy(a.shape), _footprint_xy(b.shape)))
        for a, b in zip(got, want))


def _payload(result: SimResult, clip_to_horizon: bool | None = None) -> dict:
    """Flatten the accepted intents into a compact, JSON-serialisable scene description.

    The clock spans the **realized operation** — :func:`metrics.simulation_window`, i.e. the first
    airspace reservation through the last one to clear — not ``[0, cfg.horizon_s]``. ``horizon_s`` is a
    planner *envelope*, not a schedule: flights are filed from t=0 but depart on a lead of hundreds of
    seconds, and they all land long before the envelope closes. Anchoring on it left the density replays
    with 768 s of nothing at the head and 3329 s at the tail — 57% of the slider scrubbing through an
    empty sky. This is a display choice only; the measurement window is a separate concern that
    :func:`metrics.steady_state_window` owns (issue #25).

    ``clip_to_horizon`` is a compatibility switch for callers of the previous API. ``None`` (the new
    default) uses the realized window; explicit ``True`` restores ``[0, horizon_s]`` and explicit
    ``False`` restores ``[0, max(horizon_s, last activity)]``.
    """
    cfg = result.config
    hues = result_uss_hues(result)
    usses = sorted(hues)
    uss_ix = {u: i for i, u in enumerate(usses)}
    realized_start, realized_end = metrics.simulation_window(result)
    if clip_to_horizon is None:
        play_start, play_end = realized_start, realized_end
    elif clip_to_horizon:
        play_start, play_end = 0.0, float(cfg.horizon_s)
    else:
        play_start, play_end = 0.0, max(float(cfg.horizon_s), realized_end)
    position_q = _position_q(cfg)
    rebuild_tolerance_m = _rebuild_tolerance_m(cfg)

    def q(v) -> int:                       # metres → quantised int
        return round(float(v) * position_q)

    def qt(t) -> int:                      # absolute seconds → quantised int, relative to play_start
        return round((float(t) - play_start) * _QT)

    flights, n_explicit, label_chars = [], 0, 4
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
        if qz:
            # Math.round(z*IQ)+'m' is what JS renders. Publish its maximum character count so the
            # declutter grid grows with valid high/negative flight levels instead of assuming '110m'.
            label_chars = max(label_chars, max(
                len(str(math.floor(z / position_q + 0.5))) + 1 for z in qz))
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
        if _needs_float64(qx, qy, qz):
            rec["P"] = 1                  # only this flight pays for 64-bit position arrays in JS
        if _needs_float64(qtime):
            rec["T"] = 1                  # e.g. a >24.86-day clock; prevents signed Int32 wraparound
        dequantised = [(np.array([x / position_q, y / position_q, z / position_q]),
                        t / _QT + play_start)
                       for x, y, z, t in zip(qx, qy, qz, qtime)]
        if not _rebuildable(intent, cfg, dequantised, rebuild_tolerance_m):
            # explicit polygons for just this flight
            n_explicit += 1
            rec["b"] = [[*(q(c) for corner in box_footprint(v.shape) for c in corner),
                         q(v.shape.center[2]), qt(v.t_start), qt(v.t_end)]
                        for v in vols if isinstance(v.shape, BoxSpec)]
        flights.append(rec)

    # Expose the circumradius whenever the selected planner searches the shared axial lattice, so the
    # replay can overlay the exact grid used by A*, SIPP, or column generation (including wrappers).
    hex_available = uses_hex_lattice(cfg.planner)
    from .planner.hexgrid import circumradius
    # always-active terminal WALLS (permanent no-fly columns) live in the ledger, NOT in any accepted
    # intent's volumes — render them as permanent overlays so a denial's blocker is visible. Each also
    # carries its EXIT-LANE radius: where the hub's reserved lanes begin. Configured overlap can place
    # that edge inside or outside the column; without the ring the replay makes the transition look like
    # unexplained missing corridor geometry.
    terms = {}
    for intent in result.intents:                         # denied-only hubs still own static walls
        for t in (intent.request.origin_terminal, intent.request.dest_terminal):
            if (t := as_terminal(t)) is not None:
                terms[str(t.id)] = t
    # A scenario can register a static hub which no request happens to use. The ledger is authoritative
    # when its descriptor disagrees with request metadata.
    ledger = getattr(result, "ledger", None)
    for _, raw in (getattr(ledger, "_static_terms", []) or []):
        if (term := as_terminal(raw)) is not None:
            terms[str(term.id)] = term
    persisted_radii = {str(k): float(v) for k, v in
                       (getattr(result, "static_exit_radii", {}) or {}).items()
                       if math.isfinite(float(v))}
    walls = []
    for v in _static_walls(result):
        if not isinstance(v.shape, CylinderSpec):
            continue
        tid = None if v.terminal_id is None else str(v.terminal_id)
        term = terms.get(tid)
        er = persisted_radii.get(tid)
        if er is None and term is not None:
            er = volumes.exit_radius(term, cfg)
        walls.append({"cx": float(v.shape.cx), "cy": float(v.shape.cy), "r": float(v.shape.radius),
                      "tid": tid,
                      # volumes.exit_radius is the single source of truth for the lane edge — the same
                      # radius the A* fold, the commit, and TerminalCapacity.exit_clear all root on.
                      "er": er})
    return {
        "v": 2,                              # payload schema version
        "simulation_start_s": play_start,
        "simulation_end_s": play_end,
        "simulation_duration_s": play_end - play_start,
        "horizon": play_end,                 # compatibility alias for older payload consumers
        "dt": cfg.dt_s,
        "region": list(cfg.region_size_m),
        "q": position_q, "qt": _QT,          # quantisation of the per-flight int streams
        "corridor_w": cfg.corridor_width_m,  # the three scalars that let JS rebuild every corridor box
        "corridor_h": cfg.corridor_height_m,
        "t_buffer": cfg.time_buffer_s,
        "flights": flights,
        "explicit_box_flights": n_explicit,  # flights whose boxes could NOT be rebuilt faithfully
                                             # (systemic non-zero is worth investigating)
        "walls": walls,                      # permanent terminal no-fly columns (always-active)
        "usses": usses,                      # index order for each flight's "u"
        "uss_colors": {uid: uss_swatch_hex(uid, hues) for uid in usses},   # legend / per-USS slice
        "hex_available": hex_available,
        "hex_R": circumradius(cfg) if hex_available else 0.0,
        "planner": cfg.planner,
        "flight_levels": [round(float(z), 1) for z in cfg.flight_levels_m],
        "label_chars": label_chars,
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
 amber disc = permanent terminal column (no-fly) · fine ring = where its reserved exit lanes begin<br>
 scroll to zoom · drag to pan · double-click to zoom in · <kbd>0</kbd> to fit</small>
 <canvas id="c" width="760" height="760"></canvas>
 <div id="bar"><button id="play">▶ play</button>
  <button id="back" title="step back one timestep">⏮</button>
  <button id="fwd" title="step forward one timestep">⏭</button>
  <input id="slider" type="range" min="{start}" max="{end}" value="{start}" step="any">
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
function undelta(a, Out=Int32Array){{                     // delta stream → absolute values, unboxed
  const out = new Out(a.length); let v = 0;
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
  initLabelGrid(DATA.label_chars || 4);
  for(const f of DATA.flights){{
    const Pos = f.P ? Float64Array : Int32Array;          // widen only streams that cannot fit signed i32
    f.x=undelta(f.x,Pos); f.y=undelta(f.y,Pos); f.z=undelta(f.z,Pos);
    f.t=undelta(f.t,f.T ? Float64Array : Int32Array);
  }}
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
// Altitude-label size, in SCREEN px — the one knob. The declutter slot is DERIVED from it so the two
// cannot drift apart: bump LABEL_PX alone and the slot grows to match, instead of labels quietly
// starting to overlap again. The payload supplies the widest altitude's character count.
const LABEL_PX = 13;
const LABEL_FONT = LABEL_PX + 'px monospace';
const LABEL_H = LABEL_PX + 4;
let LABEL_W=40, LABEL_COLS=0, LABEL_ROWS=0;
let labelHeads, labelCounts;                                  // fixed grid; nodes are pooled across frames
const labelNodes = [];
let labelCount = 0;
// The altitude readout is drawn in SCREEN pixels and deliberately does NOT scale with zoom — a label
// that grew with the view would swamp the map. What makes it LOOK like it scales is density: at fit a
// dense run puts ~3 drones in every label-sized slot (5.5k into 1.8k, measured) and their labels
// overprint into one illegible block, which then thins out as you zoom in.
//
// So the rule is not "shrink them" but "only draw a label you can attribute". The grid makes that
// neighbor search linear in normal use; the overlap check reaches across cell boundaries, where two
// labels can be only a fraction of a pixel apart despite belonging to different cells.
function labelsOverlap(a,b){{
  return Math.abs(a.X-b.X)<LABEL_W && Math.abs(a.Y-b.Y)<LABEL_H;
}}
function initLabelGrid(chars){{
  ctx.font=LABEL_FONT;
  // Monospace makes character count sufficient, while measureText uses the browser's actual font metric
  // instead of assuming every platform renders a glyph at exactly 0.6em.
  LABEL_W=Math.ceil(ctx.measureText('0'.repeat(Math.max(1,chars))).width)+8;
  LABEL_COLS=Math.floor(cv.width/LABEL_W)+1; LABEL_ROWS=Math.floor(cv.height/LABEL_H)+1;
  labelHeads=new Int32Array(LABEL_COLS*LABEL_ROWS);
  labelCounts=new Uint32Array(LABEL_COLS*LABEL_ROWS);
}}
function resetLabels(){{
  labelHeads.fill(-1); labelCounts.fill(0); labelCount=0;  // retain pooled node objects, avoid frame churn
}}
function noteLabel(z, X, Y){{
  if(X < 0 || X > cv.width || Y < 0 || Y > cv.height) return;   // off-canvas: never drawn
  const ix=Math.floor(X/LABEL_W), iy=Math.floor(Y/LABEL_H), ci=iy*LABEL_COLS+ix;
  let rec=labelNodes[labelCount];
  if(rec){{ rec.z=z; rec.X=X; rec.Y=Y; rec.ok=true; rec.next=-1; }}
  else {{ rec={{z,X,Y,ok:true,next:-1}}; labelNodes.push(rec); }}

  // Every point in the same half-open cell overlaps. By induction, after the second arrives every
  // existing node there is already invalid, so touching the head is sufficient rather than O(n²).
  if(labelCounts[ci]){{ rec.ok=false; labelNodes[labelHeads[ci]].ok=false; }}
  for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++){{
    if((dx===0&&dy===0) || ix+dx<0 || ix+dx>=LABEL_COLS || iy+dy<0 || iy+dy>=LABEL_ROWS) continue;
    const ni=(iy+dy)*LABEL_COLS+(ix+dx), n=labelCounts[ni];
    if(!n) continue;
    if(n===1){{                                           // the sole neighbor may still be drawable
      const other=labelNodes[labelHeads[ni]];
      if(labelsOverlap(rec,other)){{ rec.ok=false; other.ok=false; }}
    }} else if(rec.ok){{                                  // crowded neighbors are already all suppressed
      for(let j=labelHeads[ni];j>=0;j=labelNodes[j].next)
        if(labelsOverlap(rec,labelNodes[j])){{ rec.ok=false; break; }}
    }}
  }}
  rec.next=labelHeads[ci]; labelHeads[ci]=labelCount; labelCounts[ci]++; labelCount++;
}}
function drawLabels(){{                                      // after every flight, so counts are final
  ctx.fillStyle='#c6cedb'; ctx.font=LABEL_FONT;              // a touch brighter at this size
  for(let i=0;i<labelCount;i++){{ const s=labelNodes[i];
    if(s.ok) ctx.fillText(Math.round(s.z)+'m', s.X+7, s.Y-6); }}     // clear of the 4 px drone dot
}}
function exitRingVisible(wl){{
  return Number.isFinite(wl.er) && wl.er>0 && Math.abs(wl.er-wl.r)*VS >= 1.5;
}}
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
    // The exit-lane ring: where this hub's RESERVED lanes begin. Drawn only once it is visibly distinct
    // from the column — at fit the usual gap is corridor_width/2 (0.4 px on a 60 km region), so
    // below that it is 490 arcs a frame drawing nothing you can see.
    if(exitRingVisible(wl)){{
      ctx.save(); ctx.beginPath(); ctx.arc(sx(wl.cx),sy(wl.cy),wl.er*VS,0,2*Math.PI);
      ctx.setLineDash([2,3]); ctx.strokeStyle='#fbbf2466'; ctx.lineWidth=1; ctx.stroke(); ctx.restore();
    }}
  }}
  resetLabels();
  const tq = (t - START)*DATA.qt, lo = tq - TBQ;
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
    }} else {{      // segment i spans [t[i], t[i+1]+buf) — the pad is LEADING-ONLY — so it is live when
                   // t[i] <= tq && t[i+1] > lo. Both bounds are upperBound because the window is
                   // half-open at BOTH ends every frame on a dt-aligned clock: the oldest segment
                   // expires exactly at tq (t[i+1]+buf == tq, i.e. t[i+1] == lo) as the newest one
                   // opens exactly at tq (t[i] == tq). Neither boundary may be decided by epsilon.
      const a = Math.max(0, upperBound(fl.t, lo) - 1);
      const z = Math.min(fl.t.length - 2, upperBound(fl.t, tq) - 1);
      for(let i=a;i<=z;i++){{ on=true; strokePoly(POLY, fl.k, segPoly(fl, i, POLY)); }}
    }}
    for(const cy of fl.c){{ if(cy[3]>tq || tq>=cy[4]) continue; on=true;
      ctx.beginPath(); ctx.arc(sx(cy[0]*IQ),sy(cy[1]*IQ),cy[2]*IQ*VS,0,2*Math.PI);
      ctx.fillStyle=fl.k+'33'; ctx.fill(); ctx.strokeStyle=fl.k; ctx.stroke(); }}
    const p = posAt(fl, tq);
    if(p){{ on=true; ctx.beginPath(); ctx.arc(sx(p[0]),sy(p[1]),4,0,2*Math.PI);
      ctx.fillStyle=fl.k; ctx.fill(); ctx.strokeStyle='#000'; ctx.lineWidth=0.5; ctx.stroke();
      if(LEVELS.length>1) noteLabel(p[2], sx(p[0]), sy(p[1])); }}   // altitude readout (see drawLabels)
    if(on){{                                          // dashed straight origin→dest for active flights
      ctx.save(); ctx.setLineDash([6,5]); ctx.lineWidth=1; ctx.strokeStyle=fl.k+'aa';
      ctx.beginPath(); ctx.moveTo(sx(fl.o[0]*IQ),sy(fl.o[1]*IQ));
      ctx.lineTo(sx(fl.d[0]*IQ),sy(fl.d[1]*IQ)); ctx.stroke(); ctx.restore();
      nActive++;
    }}
  }}
  if(LEVELS.length>1) drawLabels();
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
function wheelDeltaPx(deltaY, mode, pagePx){{             // WheelEvent line/page units → CSS pixels
  return deltaY * (mode===1 ? 16 : mode===2 ? pagePx : 1);
}}
function redraw(){{
  document.getElementById('zoomLbl').textContent = Z.toFixed(Z < 10 ? 1 : 0) + '\u00d7';
  draw(+slider.value);
}}
function wireView(){{
  cv.style.cursor = 'grab';
  cv.addEventListener('wheel', e => {{                     // wheel / trackpad pinch → zoom at cursor
    e.preventDefault();
    const dy = wheelDeltaPx(e.deltaY, e.deltaMode, cv.height);
    setZoom(Z * Math.pow(1.0015, -dy), e.offsetX, e.offsetY);
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
function atEnd(t){{ return t >= END - IQT/2; }}             // tolerate range-input decimal sanitization
function step(d){{ playing=false; document.getElementById('play').textContent='▶ play';
  clock = Math.max(START, Math.min(END, +slider.value + d)); slider.value=clock; draw(clock); }}
// The play position is a FLOAT clock (rather than repeatedly reading the range control), so sub-unit
// advances and fractional bounds survive browser value sanitization. At 1x the realized run plays in
// ~10s (60fps · duration/600 per frame); speed scales that per-frame step (0.25x ⇒ 40s, 8x ⇒ ~1.25s).
function tick(){{ if(!playing || DURATION<=0) return; clock += speed*DURATION/600;
  if(atEnd(clock)){{ clock=END; slider.value=clock; draw(clock); playing=false;
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
    this.textContent = playing?'⏸ pause':'▶ play'; if(playing){{
      if(atEnd(clock)){{ clock=START; slider.value=clock; draw(clock); }} tick(); }} }};
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


def write_html(result: SimResult, out, clip_to_horizon: bool | None = None) -> str:
    """Render a standalone HTML scrubber; return its path.

    With no compatibility argument the replay uses the realized operation. Explicit booleans retain
    the previous API's horizon-clipped (``True``) and horizon-or-tail (``False``) clock modes.
    """
    payload = _payload(result, clip_to_horizon)
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
