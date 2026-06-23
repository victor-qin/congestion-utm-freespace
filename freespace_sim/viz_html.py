"""Self-contained HTML replay — scrub time and watch FCFS deconfliction happen.

Emits a single standalone ``.html`` file (no server, no external assets): the reservation geometry
is serialised to JSON and drawn as projected polygons on a ``<canvas>``, with a play/pause/step/slider
transport. This is the free-space analog of the sibling project's ``viz_html.py`` — corridors and
hover cylinders appear and vanish with their ASTM time windows while drones glide along centerlines.
"""

from __future__ import annotations

import json

from .geometry import BoxSpec, CylinderSpec
from .sim import SimResult
from .viz import box_footprint, flight_color_by_uss, result_uss_hues, uss_swatch_hex


def _payload(result: SimResult) -> dict:
    """Flatten the accepted intents into a compact, JSON-serialisable scene description."""
    hues = result_uss_hues(result)
    flights = []
    for intent in result.accepted:
        r, g, b = flight_color_by_uss(intent.request.uss_id, intent.request.flight_id, hues)
        boxes, cyls = [], []
        for v in intent.volumes or []:
            if isinstance(v.shape, BoxSpec):
                boxes.append({"poly": box_footprint(v.shape).round(1).tolist(),
                              "t0": v.t_start, "t1": v.t_end})
            elif isinstance(v.shape, CylinderSpec):
                cyls.append({"cx": v.shape.cx, "cy": v.shape.cy, "r": v.shape.radius,
                             "t0": v.t_start, "t1": v.t_end})
        path = [[float(p[0]), float(p[1]), float(t)] for p, t in (intent.centerline or [])]
        o, d = intent.request.origin, intent.request.dest
        flights.append({
            "id": intent.request.flight_id,
            "uss": intent.request.uss_id,
            "color": f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}",
            "boxes": boxes, "cyls": cyls, "path": path,
            "o": [float(o[0]), float(o[1])], "d": [float(d[0]), float(d[1])],
        })
    cfg = result.config
    # The hex lattice only exists if an A*-based planner ran (astar / astar_milp / opt_astar). When
    # it did, expose its circumradius so the replay can overlay the exact grid A* searched on.
    hex_available = "astar" in cfg.planner
    from .planner.hexgrid import circumradius
    uss_colors = {uid: uss_swatch_hex(uid, hues) for uid in sorted(hues)}
    # The replay clock must span the LAST volume to clear, not just the demand horizon: return flights
    # are scheduled after their delivery + turnaround, so they fly well past cfg.horizon_s and would be
    # clipped off the right edge of the slider otherwise. For return-free runs this stays == horizon_s.
    play_end = cfg.horizon_s
    for f in flights:
        for seg in f["boxes"]:
            play_end = max(play_end, seg["t1"])
        for seg in f["cyls"]:
            play_end = max(play_end, seg["t1"])
    return {
        "horizon": play_end,
        "dt": cfg.dt_s,
        "region": list(cfg.region_size_m),
        "flights": flights,
        "uss_colors": uss_colors,           # {uss_id: #rrggbb} for the legend / per-USS slice
        "hex_available": hex_available,
        "hex_R": circumradius(cfg) if hex_available else 0.0,
        "planner": cfg.planner,
    }


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
 h3{{margin:6px 0 0}} small{{color:#8b949e}}
</style></head><body><div id="wrap">
 <h3>FCFS strategic deconfliction — free-space replay</h3>
 <small>corridors = trajectory intents · circles = hover reservations · dots = drones · dashed = straight origin→dest</small>
 <canvas id="c" width="760" height="760"></canvas>
 <div id="bar"><button id="play">▶ play</button>
  <button id="back" title="step back one timestep">⏮</button>
  <button id="fwd" title="step forward one timestep">⏭</button>
  <input id="slider" type="range" min="0" max="{horizon}" value="0" step="1">
  <span id="t">t = 0 s</span>
  <label class="tog" for="speed">speed
   <select id="speed" title="playback speed">
    <option value="0.25">0.25&times;</option>
    <option value="0.5">0.5&times;</option>
    <option value="1" selected>1&times;</option>
    <option value="2">2&times;</option>
    <option value="4">4&times;</option>
    <option value="8">8&times;</option>
   </select></label>
  <label class="tog" id="hexWrap"><input type="checkbox" id="hexToggle"> hex grid (A*)</label>
  <span id="legend" style="display:flex;gap:10px;flex-wrap:wrap"></span>
 </div>
</div><script>
const DATA = {data};
const hidden = new Set();                              // USS ids toggled off via the legend
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const [W, H] = DATA.region, PAD = 20, S = (cv.width - 2*PAD) / Math.max(W, H);
const sx = x => PAD + x*S, sy = y => cv.height - PAD - y*S;   // flip y: north is up
function active(o, t){{ return o.t0 <= t && t < o.t1; }}
function posAt(path, t){{
  if(!path.length || t < path[0][2] || t > path[path.length-1][2]) return null;
  for(let i=0;i<path.length-1;i++){{ const a=path[i], b=path[i+1];
    if(a[2]<=t && t<=b[2]){{ const f=(b[2]-a[2])<1e-9?0:(t-a[2])/(b[2]-a[2]);
      return [a[0]+f*(b[0]-a[0]), a[1]+f*(b[1]-a[1])]; }} }}
  return [path[path.length-1][0], path[path.length-1][1]];
}}
function drawHexGrid(){{
  const R = DATA.hex_R; if(!R) return;
  const SQRT3 = Math.sqrt(3);
  ctx.strokeStyle = '#39414f'; ctx.lineWidth = 0.4;          // faint lattice beneath the corridors
  const rMax = Math.ceil(H/(1.5*R)) + 1;
  for(let r=-1; r<=rMax; r++){{
    const qLo = Math.floor(-r/2) - 1, qHi = Math.ceil(W/(SQRT3*R) - r/2) + 1;
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
function draw(t){{
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.strokeStyle='#30363d'; ctx.strokeRect(sx(0),sy(H),W*S,H*S);
  if(document.getElementById('hexToggle').checked) drawHexGrid();
  let nActive=0;
  for(const fl of DATA.flights){{
    if(hidden.has(fl.uss)) continue;                 // per-USS slice (legend toggles)
    let on=false;
    ctx.lineWidth=0.6;
    for(const bx of fl.boxes){{ if(!active(bx,t)) continue; on=true;
      ctx.beginPath(); bx.poly.forEach((p,i)=> i?ctx.lineTo(sx(p[0]),sy(p[1])):ctx.moveTo(sx(p[0]),sy(p[1])));
      ctx.closePath(); ctx.fillStyle=fl.color+'55'; ctx.fill(); ctx.strokeStyle=fl.color; ctx.stroke(); }}
    for(const cy of fl.cyls){{ if(!active(cy,t)) continue; on=true;
      ctx.beginPath(); ctx.arc(sx(cy.cx),sy(cy.cy),cy.r*S,0,2*Math.PI);
      ctx.fillStyle=fl.color+'33'; ctx.fill(); ctx.strokeStyle=fl.color; ctx.stroke(); }}
    const p = posAt(fl.path, t);
    if(p){{ on=true; ctx.beginPath(); ctx.arc(sx(p[0]),sy(p[1]),4,0,2*Math.PI);
      ctx.fillStyle=fl.color; ctx.fill(); ctx.strokeStyle='#000'; ctx.lineWidth=0.5; ctx.stroke(); }}
    if(on){{                                          // dashed straight origin→dest for active flights
      ctx.save(); ctx.setLineDash([6,5]); ctx.lineWidth=1; ctx.strokeStyle=fl.color+'aa';
      ctx.beginPath(); ctx.moveTo(sx(fl.o[0]),sy(fl.o[1])); ctx.lineTo(sx(fl.d[0]),sy(fl.d[1]));
      ctx.stroke(); ctx.restore();
      nActive++;
    }}
  }}
  document.getElementById('t').textContent = 't = '+Math.round(t)+' s  ('+nActive+' active)';
}}
const slider=document.getElementById('slider');
slider.oninput=()=>{{ clock=+slider.value; draw(clock); }};   // scrubbing re-seats the play clock
function step(d){{ playing=false; document.getElementById('play').textContent='▶ play';
  let t=Math.max(0, Math.min(DATA.horizon, +slider.value + d));
  clock=t; slider.value=t; draw(t); }}
document.getElementById('back').onclick=()=>step(-DATA.dt);   // one timestep back
document.getElementById('fwd').onclick=()=>step(+DATA.dt);    // one timestep forward
document.getElementById('hexToggle').onchange=()=>draw(+slider.value);
if(!DATA.hex_available) document.getElementById('hexWrap').style.display='none';
let playing=false, raf=null, speed=1, clock=0;
document.getElementById('speed').onchange=function(){{ speed=+this.value; }};
// the play position is a FLOAT clock, not the slider value — the range input snaps to step=1, so it
// cannot hold sub-unit advances and speeds < 1 would round away. At 1x the full horizon plays in ~10s
// (60fps · horizon/600 per frame); speed scales that per-frame step (0.25x ⇒ 40s, 8x ⇒ ~1.25s).
function tick(){{ if(!playing) return; clock += speed*DATA.horizon/600;
  if(clock>DATA.horizon) clock=0; slider.value=clock; draw(clock); raf=requestAnimationFrame(tick); }}
document.getElementById('play').onclick=function(){{ playing=!playing;
  this.textContent = playing?'⏸ pause':'▶ play'; if(playing){{ clock=+slider.value; tick(); }} }};
// keyboard: ← / → step one timestep, space toggles play
document.addEventListener('keydown', e=>{{
  if(e.key==='ArrowLeft') step(-DATA.dt);
  else if(e.key==='ArrowRight') step(+DATA.dt);
  else if(e.key===' '){{ e.preventDefault(); document.getElementById('play').click(); }}
}});
// legend + per-USS show/hide (only when more than one operator flew)
(function buildLegend(){{
  const usses = Object.keys(DATA.uss_colors || {{}});
  if(usses.length < 2) return;
  const legend = document.getElementById('legend');
  for(const u of usses){{
    const lab = document.createElement('label'); lab.className = 'tog';
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = true;
    cb.onchange = ()=>{{ cb.checked ? hidden.delete(u) : hidden.add(u); draw(+slider.value); }};
    const sw = document.createElement('span');
    sw.style.cssText = 'display:inline-block;width:11px;height:11px;border-radius:2px;background:'+DATA.uss_colors[u];
    lab.appendChild(cb); lab.appendChild(sw); lab.appendChild(document.createTextNode(u));
    legend.appendChild(lab);
  }}
}})();
draw(0);
</script></body></html>"""


def write_html(result: SimResult, out) -> str:
    """Render ``result`` to a standalone HTML scrubber at ``out``; returns the path written."""
    payload = _payload(result)
    html = _HTML.format(horizon=int(payload["horizon"]), data=json.dumps(payload))
    with open(out, "w") as f:
        f.write(html)
    return str(out)
