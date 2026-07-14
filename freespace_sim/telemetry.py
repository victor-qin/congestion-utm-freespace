"""Permanent run telemetry — the non-recoverable congestion streams (issue: run-instrumentation).

`save_run` already archives everything about *accepted* flights (reservations, trajectories, per-flight
outcomes) and is reproducible from `config.json` + `scenario.parquet`. Three things it can NOT recover
post-hoc are captured here, live, by an **observer-only** collector (zero behaviour change — telemetry-off
is byte-identical):

  * **`conflict_filed` / detour-`budget_exceeded` filed volumes** — the rejected corridor A* built then had
    to deny (`_deny` discards it). Answers "track the volumes filed, at least for errors".
  * **conflict culprits** — the committed volume(s) a `conflict_filed` collided with (`ledger.conflicts`).
  * **per-hub terminal metadata** — a run-time snapshot of every placed hub (incl. zero-traffic ones),
    since `save_run` only receives a demand *string*, not the model.

Per-hub **dwell occupancy** is NOT here — it is recoverable post-hoc from `reservations.parquet` (the
ledger is append-only; committed columns are persisted), so :func:`terminal_frame` sweep-lines them.
**Gate attribution** (pad/air/lane) and **kernel byte-exactness parity** are deferred follow-ups (see
`experiments/TELEMETRY_DESIGN.md` §3b.2 / §9); the `gate_reject` field is present but not yet emitted.

See :func:`freespace_sim.runs.save_run` for persistence and `TELEMETRY_DESIGN.md` for the full design.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .geometry import BoxSpec, CylinderSpec
from .types import as_terminal
from .volumes import terminal_radius

if TYPE_CHECKING:
    from .sim import SimResult
    from .volumes import Volume4D


def _vol_row(v: "Volume4D") -> dict:
    """A single `Volume4D`'s analytical geometry + time window as a flat dict — the schema
    ``runs.reservation_frame`` persists (box: center/rot/extents; cylinder: cx/cy/radius/z_lo/z_hi), minus
    the caller-supplied ``flight_id``. Shared so filed volumes round-trip through ``runs._volume_from_row``
    exactly like committed reservations."""
    s = v.shape
    row = {"t_start": v.t_start, "t_end": v.t_end,
           "terminal_id": None if v.terminal_id is None else str(v.terminal_id)}
    if isinstance(s, BoxSpec):
        row.update({"kind": "box", "cx": s.center[0], "cy": s.center[1], "cz": s.center[2],
                    "rot": json.dumps(list(s.rot)), "ext": json.dumps(list(s.extents)),
                    "radius": np.nan, "z_lo": np.nan, "z_hi": np.nan})
    else:
        row.update({"kind": "cyl", "cx": s.cx, "cy": s.cy, "cz": (s.z_lo + s.z_hi) / 2,
                    "rot": "", "ext": "", "radius": s.radius, "z_lo": s.z_lo, "z_hi": s.z_hi})
    return row


@dataclass
class TelemetryCollector:
    """Observer-only capture of the non-recoverable congestion streams. Lives on a planner as ``_tele``
    (set by ``sim.run`` when telemetry is on); its hooks only read + append, never change control flow."""

    enabled: bool = True
    # per-hub metadata snapshot, filled at sim.run setup (the one place the demand model is in scope)
    terminals: dict[Hashable, dict] = field(default_factory=dict)   # tid -> {cx,cy,capacity,radius}
    # gate attribution — DEFERRED (see module docstring); tid -> [pad, air, lane]. Present so the schema is
    # stable; nothing emits into it yet.
    gate_reject: dict[Hashable, list[int]] = field(default_factory=dict)
    conflict_events: list[dict] = field(default_factory=list)       # one row per CULPRIT (blocker)
    filed_volumes: list[dict] = field(default_factory=list)         # the REJECTED corridor's own volumes

    def on_deny(self, flight_id: int, reason: str, volumes, hits=None) -> None:
        """Record a denial that BUILT a corridor: the filed (rejected) volumes always, and the culprit(s)
        for a conflict. Called from the planner's ``_file_deny`` at every corridor-building deny site."""
        for j, v in enumerate(volumes or []):
            self.filed_volumes.append({"flight_id": int(flight_id), "reason": reason,
                                       "vol_idx": j, **_vol_row(v)})
        for fid, vol in (hits or []):
            self.conflict_events.append({
                "flight_id": int(flight_id), "culprit_fid": int(fid),
                "culprit_tid": None if vol.terminal_id is None else str(vol.terminal_id),
                "shape": type(vol.shape).__name__, "t_start": vol.t_start, "t_end": vol.t_end})

    def on_gate_reject(self, tid: Hashable, kind: int) -> None:   # DEFERRED — not yet called
        self.gate_reject.setdefault(tid, [0, 0, 0])[kind] += 1


def build_terminal_snapshot(cfg, demand, events) -> dict:
    """Snapshot every placed hub's metadata (id → cx/cy/capacity/radius) at run time. Prefers the demand
    model's full placed-hub set (so zero-traffic hubs are included); else harvests the hubs that carry a
    flight from the scenario events. Empty for non-hub demands."""
    if demand is not None and hasattr(demand, "terminals"):
        pairs = list(demand.terminals(cfg))
    else:
        seen: dict = {}
        for ev in events:
            rq = ev.request
            for pt, t in ((rq.origin, rq.origin_terminal), (rq.dest, rq.dest_terminal)):
                term = as_terminal(t)
                if term is not None:
                    seen.setdefault(term.id, (pt, term))
        pairs = list(seen.values())
    snap: dict = {}
    for center, term in pairs:
        term = as_terminal(term)
        c = np.asarray(center, float)
        snap[term.id] = {"cx": float(c[0]), "cy": float(c[1]),
                         "capacity": int(term.capacity), "radius": float(terminal_radius(term, cfg))}
    return snap


def _peak_overlap(intervals: list[tuple[float, float]]) -> int:
    """Max number of simultaneously-active ``[t_start, t_end)`` intervals — a sweep-line over start/end
    events (start=+1 before end=-1 at a tie, half-open)."""
    if not intervals:
        return 0
    events: list[tuple[float, int]] = []
    for a, b in intervals:
        events.append((a, 1))
        events.append((b, -1))
    events.sort(key=lambda e: (e[0], e[1]))   # -1 (end) before +1 (start) at equal time ⇒ half-open
    peak = cur = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


def terminal_frame(result: "SimResult") -> pd.DataFrame:
    """Per-hub congestion rollup — one row per placed hub. ``peak_pad_occupancy`` is a sweep-line over the
    accepted terminal-tagged **cylinder** dwells (from the persisted reservations, NOT a live hook);
    departures/arrivals and ground-delay stats come from the accepted flights' terminal membership;
    metadata (pads/radius/center) from the run-time snapshot so zero-traffic hubs still appear."""
    cfg = result.config
    tele = getattr(result, "telemetry", None)
    terms = dict(tele.terminals) if tele else {}
    w, h = cfg.region_size_m

    dwells: dict = defaultdict(list)      # tid -> [(t_start, t_end)]  (both origin + dest columns)
    dep_delays: dict = defaultdict(list)  # tid -> [ground_delay_s of departures]
    arrivals: dict = defaultdict(int)
    uss_of: dict = {}
    for i in result.accepted:
        for v in i.volumes or []:
            if v.terminal_id is not None and isinstance(v.shape, CylinderSpec):
                dwells[v.terminal_id].append((v.t_start, v.t_end))
        ot, dt = as_terminal(i.request.origin_terminal), as_terminal(i.request.dest_terminal)
        if ot is not None:
            dep_delays[ot.id].append(i.ground_delay_s)
            uss_of.setdefault(ot.id, i.request.uss_id)
        if dt is not None:
            arrivals[dt.id] += 1
            uss_of.setdefault(dt.id, i.request.uss_id)

    rows = []
    for tid in sorted(set(terms) | set(dwells) | set(dep_delays) | set(arrivals), key=str):
        meta = terms.get(tid, {})
        cx, cy = meta.get("cx"), meta.get("cy")
        gd = dep_delays.get(tid, [])
        gate = tele.gate_reject.get(tid, [0, 0, 0]) if tele else [0, 0, 0]
        rows.append({
            "tid": str(tid), "type": uss_of.get(tid),
            "cx": cx, "cy": cy, "pads": meta.get("capacity"), "radius": meta.get("radius"),
            "dist_to_edge_m": (min(cx, w - cx, cy, h - cy) if cx is not None else np.nan),
            "n_departures": len(gd), "n_arrivals": arrivals.get(tid, 0),
            "peak_pad_occupancy": _peak_overlap(dwells.get(tid, [])),
            "pad_reject": gate[0], "air_reject": gate[1], "lane_reject": gate[2],   # DEFERRED → 0
            "mean_ground_delay_s": float(np.mean(gd)) if gd else 0.0,
            "max_ground_delay_s": float(np.max(gd)) if gd else 0.0,
        })
    return pd.DataFrame(rows)


def conflict_frame(result: "SimResult") -> pd.DataFrame:
    """One row per culprit of a `conflict_filed`, with ``culprit_kind`` classified: ``static_wall`` for the
    always-active wall sentinel (fid == -1), ``sibling`` if the culprit is the filed flight's own hub, else
    ``foreign``. Empty frame (stable columns) when no conflicts were captured."""
    from .ledger import ReservationLedger

    tele = getattr(result, "telemetry", None)
    events = list(tele.conflict_events) if tele else []
    cols = ["flight_id", "culprit_fid", "culprit_kind", "culprit_tid", "shape", "t_start", "t_end"]
    if not events:
        return pd.DataFrame(columns=cols)
    # a flight's own hub id(s), to tell sibling from foreign
    hub_of: dict = {}
    for i in result.intents:
        ids = {t2.id for t in (i.request.origin_terminal, i.request.dest_terminal)
               if (t2 := as_terminal(t)) is not None}
        hub_of[i.request.flight_id] = ids
    rows = []
    for e in events:
        if e["culprit_fid"] == ReservationLedger.STATIC_WALL_FID:
            kind = "static_wall"
        elif e["culprit_tid"] is not None and e["culprit_tid"] in {str(x) for x in hub_of.get(e["flight_id"], ())}:
            kind = "sibling"
        else:
            kind = "foreign"
        rows.append({**e, "culprit_kind": kind})
    return pd.DataFrame(rows, columns=cols)


def filed_volume_frame(result: "SimResult") -> pd.DataFrame:
    """The rejected-corridor geometry for every built-then-denied flight (error forensics), same geometry
    schema as `reservations.parquet` + ``flight_id``/``reason``/``vol_idx``. Joinable to
    :func:`conflict_frame` on ``flight_id``."""
    tele = getattr(result, "telemetry", None)
    rows = list(tele.filed_volumes) if tele else []
    cols = ["flight_id", "reason", "vol_idx", "kind", "cx", "cy", "cz", "rot", "ext",
            "radius", "z_lo", "z_hi", "terminal_id", "t_start", "t_end"]
    return pd.DataFrame(rows, columns=cols)
