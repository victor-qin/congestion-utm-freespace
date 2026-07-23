"""Run capture — freeze a `SimResult` to a self-contained, replayable folder under ``results/``.

Mirrors the sibling project's run tracking, adapted to continuous free space. Every run writes a
timestamped folder ``results/{ISO}_{label}_{hash}/`` holding **everything needed to reproduce,
analyse, or replay it without re-running the sim**:

    config.json          the exact SimConfig used
    experiment.json      which experiment ran + its args + wall-clock seconds  (← "what was run")
    env.json / git.json  toolchain + commit (best-effort; this package is often used outside git)
    summary.json         headline aggregate
    scenario.parquet     EVERY generated flight request (origin/dest/filing time)  (← the scenario)
    trajectories.parquet what was actually flown — timed centerline waypoints per flight
    reservations.parquet what was reserved in 4D — every corridor box + hover cylinder + window
    flights.parquet      per-flight metrics rows
    replay.html          the standalone scrubbable replay (the "video")

``load_run(folder)`` is the exact reverse: it rebuilds a `SimResult`-shaped object (config + intents
with their volumes and centerlines) so a replay or analysis can start from disk. An append-only
``results/index.parquet`` indexes every run for cross-run queries.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics
from .config import SimConfig
from .geometry import BoxSpec, CylinderSpec
from .sim import SimResult
from .telemetry import _vol_row, conflict_frame, filed_volume_frame, terminal_frame
from .types import DenialReason, FlightRequest, IntentStatus, OperationalIntent, as_terminal, vec
from .volumes import Volume4D

DEFAULT_ROOT = Path("results")
INDEX_FILENAME = "index.parquet"


# --- metadata captures -----------------------------------------------------


def _config_hash(cfg: SimConfig) -> str:
    payload = json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:8]


def _git_info() -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=2)
        if sha.returncode != 0:
            return {"available": False}
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=2)
        return {"available": True, "commit": sha.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"available": False}


def _env_info() -> dict:
    versions = {}
    for mod in ("numpy", "fcl", "pandas", "pulp", "trimesh"):
        try:
            versions[mod] = __import__(mod).__version__
        except (ImportError, AttributeError):
            versions[mod] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "versions": versions}


# --- flight-data frame builders --------------------------------------------


def _term_to_json(t) -> str | None:
    """Serialize a flight's terminal (id, capacity, radius, corridor_overlap) to JSON so hub membership
    round-trips — including for DENIED flights, whose geometry is otherwise unrecoverable. ``None`` for a
    non-hub endpoint. The id round-trips exactly for str/int ids (the common case)."""
    t = as_terminal(t)
    if t is None:
        return None
    tid = t.id if isinstance(t.id, (str, int, float, bool)) else str(t.id)   # JSON-safe (str for exotic ids)
    return json.dumps([tid, t.capacity, t.radius, t.corridor_overlap])


def _term_from_json(s):
    """Inverse of :func:`_term_to_json` → an ``(id, capacity, radius, corridor_overlap)`` tuple
    (as_terminal-friendly), or ``None``. Tolerates a NaN/None cell from parquet."""
    if s is None or (isinstance(s, float) and s != s):
        return None
    return tuple(json.loads(s))


def scenario_frame(result: SimResult) -> pd.DataFrame:
    """Every generated flight request — the scenario, independent of what got accepted. Carries each
    endpoint's terminal (hub) membership so a saved run — including its denied flights — records which hub
    each flight used (round-tripped by :func:`load_run`)."""
    rows = []
    for i in result.intents:
        r = i.request
        o, d = np.asarray(r.origin, float), np.asarray(r.dest, float)
        rows.append({
            "flight_id": r.flight_id, "uss_id": r.uss_id,
            "t_request": r.t_request,
            "t_departure": r.t_departure if r.t_departure is not None else r.t_request,
            "origin_x": o[0], "origin_y": o[1], "origin_z": o[2],
            "dest_x": d[0], "dest_y": d[1], "dest_z": d[2],
            "origin_terminal": _term_to_json(r.origin_terminal),
            "dest_terminal": _term_to_json(r.dest_terminal),
        })
    return pd.DataFrame(rows)


def trajectory_frame(result: SimResult) -> pd.DataFrame:
    """What was actually flown: one row per timed centerline waypoint (v0: flown == reserved)."""
    rows = []
    for i in result.accepted:
        for p, t in i.centerline or []:
            p = np.asarray(p, float)
            rows.append({"flight_id": i.request.flight_id, "t": float(t),
                         "x": p[0], "y": p[1], "z": p[2]})
    return pd.DataFrame(rows, columns=["flight_id", "t", "x", "y", "z"])


def reservation_frame(result: SimResult) -> pd.DataFrame:
    """What was reserved in 4D: one row per Volume4D (full analytical geometry + time window).

    ``rot``/``ext`` are JSON-encoded for boxes; ``radius``/``z_lo``/``z_hi`` carry cylinders. This is
    enough to rebuild the exact `Volume4D` (see :func:`load_run`) and to drive the replay.
    """
    rows = [{"flight_id": i.request.flight_id, **_vol_row(v)}
            for i in result.accepted for v in (i.volumes or [])]
    cols = ["flight_id", "kind", "t_start", "t_end", "cx", "cy", "cz",
            "rot", "ext", "radius", "z_lo", "z_hi", "terminal_id"]
    return pd.DataFrame(rows, columns=cols)


def _ledger_end_frame(result: SimResult) -> pd.DataFrame:
    """The always-active terminal WALLS (``ledger._static_vols``) — the part of the end-of-run ledger that
    ``reservation_frame`` (accepted intents only) doesn't capture. Same geometry schema; empty when the run
    used no always-active walls. ``reservations.parquet`` ∪ this == the full end-of-run ledger (see the
    telemetry design §10)."""
    ledger = getattr(result, "ledger", None)
    vols = list(getattr(ledger, "_static_vols", []) or [])
    rows = [{"wall_idx": j, **_vol_row(v)} for j, v in enumerate(vols)]
    cols = ["wall_idx", "kind", "t_start", "t_end", "cx", "cy", "cz",
            "rot", "ext", "radius", "z_lo", "z_hi", "terminal_id"]
    return pd.DataFrame(rows, columns=cols)


# --- save / load -----------------------------------------------------------


def save_run(
    result: SimResult, *,
    root: Path | str = DEFAULT_ROOT,
    label: str = "run",
    experiment: str | None = None,
    experiment_args: dict | None = None,
    wall_seconds: float | None = None,
    scenario: str | None = None,
    demand: str | None = None,
    write_replay: bool = True,
    index: bool = True,
    clip_replay_to_horizon: bool = True,
    window_frac: float = 0.9,
) -> Path:
    """Write the full self-contained run folder and return its path.

    Captures config/env/git, the experiment identity + args, the scenario, the flown trajectories,
    the reserved 4D volumes, per-flight metrics, and (by default) the standalone replay HTML. Everything
    is parquet + json — deliberately NOT pickle: portable, inspectable, safe to sync to the run store,
    and Python-version-independent. The analytical geometry stored in reservations/ledger_end is enough
    to rebuild every ``Volume4D`` on load (see :func:`load_run` / :func:`_volume_from_row`).

    ``summary.json`` carries the whole-run headline numbers **and** their steady-state twin (metrics
    over the representative density plateau — issue #25) in a nested ``steady_state`` block; ``window_frac``
    tunes the plateau threshold. ``clip_replay_to_horizon`` (default) stops ``replay.html`` at the horizon;
    pass ``False`` to keep post-horizon return flights visible.
    """
    cfg = result.config
    agg = metrics.aggregate_with_steady(result, frac=window_frac)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    folder = Path(root) / f"{stamp}_{label}_{_config_hash(cfg)}"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    (folder / "env.json").write_text(json.dumps(_env_info(), indent=2))
    (folder / "git.json").write_text(json.dumps(_git_info(), indent=2))
    (folder / "experiment.json").write_text(json.dumps({
        "experiment": experiment or label,
        "scenario": scenario,
        "demand": demand,
        "tag": label,
        "args": experiment_args or {},
        "wall_seconds": wall_seconds,
        "timestamp": stamp,
        "planner": cfg.planner,
        "n_requests": len(result.intents),
        "verified": result.verified,
    }, indent=2, default=str))
    (folder / "summary.json").write_text(json.dumps(agg, indent=2))

    scenario_frame(result).to_parquet(folder / "scenario.parquet", index=False)
    trajectory_frame(result).to_parquet(folder / "trajectories.parquet", index=False)
    reservation_frame(result).to_parquet(folder / "reservations.parquet", index=False)
    metrics.flight_frame(result).to_parquet(folder / "flights.parquet", index=False)
    metrics.per_uss_frame(result).to_parquet(folder / "per_uss.parquet", index=False)   # per-operator slice

    # The always-active terminal walls belong to the run regardless of telemetry — the replay overlay and
    # the full end-of-run ledger both need them — so persist them whenever they exist (cheap: one row/hub).
    walls = _ledger_end_frame(result)
    if len(walls):
        walls.to_parquet(folder / "ledger_end.parquet", index=False, compression="zstd")

    if result.telemetry is not None:
        # observer-only congestion telemetry (issue: run instrumentation) — the streams post-hoc can't
        # recover: rejected-corridor geometry + conflict culprits + per-hub metadata.
        terminal_frame(result).to_parquet(folder / "terminal_telemetry.parquet", index=False, compression="zstd")
        conflict_frame(result).to_parquet(folder / "conflict_events.parquet", index=False, compression="zstd")
        filed_volume_frame(result).to_parquet(folder / "filed_volumes.parquet", index=False, compression="zstd")

    if write_replay:
        from . import viz_html
        viz_html.write_html(result, folder / "replay.html", clip_to_horizon=clip_replay_to_horizon)

    if index:
        _append_index(result, folder, Path(root), wall_seconds, scenario=scenario,
                      tag=label, demand=demand, agg=agg)
    return folder


def _append_index(result: SimResult, folder: Path, root: Path, wall_seconds: float | None,
                  *, scenario: str | None = None, tag: str | None = None,
                  demand: str | None = None, agg: dict | None = None) -> None:
    """Append one queryable row per run to ``results/index.parquet``.

    The ``scenario`` / ``tag`` / ``demand`` columns are the join keys cross-run readouts filter on:
    a batch sweep stamps every run with the same ``tag`` so a readout can select exactly its runs.
    ``agg`` may be a precomputed :func:`metrics.aggregate_with_steady` (avoids recomputing it); the
    ``steady_*`` / ``window_*`` columns carry the steady-state twin of the headline metrics so a
    cross-run curve can plot the de-biased trend alongside the whole-run one (issue #25).
    """
    cfg = result.config
    if agg is None:
        agg = metrics.aggregate_with_steady(result)
    steady = agg.get("steady_state", {})
    steady_cols = {f"steady_{k}": steady.get(k) for k in
                   ("mean_total_delay_s", "p50_total_delay_s", "p95_total_delay_s",
                    "throughput_per_h", "denial_rate", "congestion_denial_rate")}
    steady_cols["window_lo"] = steady.get("window_lo")
    steady_cols["window_hi"] = steady.get("window_hi")
    row = {"path": str(folder), "scenario": scenario, "tag": tag, "demand": demand,
           "planner": cfg.planner, "lam_per_hour": cfg.lam_per_hour, "seed": cfg.seed,
           "horizon_s": cfg.horizon_s,
           "region_w": cfg.region_size_m[0], "region_h": cfg.region_size_m[1],
           "wall_seconds": wall_seconds,
           "has_telemetry": result.telemetry is not None,
           **{k: agg[k] for k in ("n_uss", "n_requests", "n_accepted", "n_denied", "denial_rate",
                                  "congestion_denial_rate", "offered_load_per_h", "throughput_per_h",
                                  "mean_total_delay_s", "p95_total_delay_s", "mean_air_detour_m",
                                  "mean_stretch", "mean_cost",
                                  "airspace_utilization", "denial_rate_spread", "mean_delay_spread",
                                  "mean_solve_time_s", "p95_solve_time_s",
                                  "max_solve_time_s", "total_solve_time_s", "verified")},
           **steady_cols}
    path = root / INDEX_FILENAME
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


def load_index(root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """Load the cross-run index (one row per saved run), or an empty frame if none exists yet.

    This is the interface for cross-run readouts (curve, compare): read it, filter by
    ``scenario`` / ``tag`` / ``planner``, and plot — no re-simulation."""
    path = Path(root) / INDEX_FILENAME
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def sweep_dir(label: str, root: Path | str = DEFAULT_ROOT) -> Path:
    """Folder that groups a *run set's* cross-run readout artifacts (curve / histograms / compare).

    A cross-run readout describes a *set* of runs (the ``--tag``/``--scenario`` it filtered on), not a
    single run, so its artifacts don't belong in any one run folder nor loose in the results root —
    they live here, under ``<root>/sweeps/<label>/``. Stable per label, so re-running a readout
    refreshes its artifacts in place instead of scattering timestamped copies."""
    d = Path(root) / "sweeps" / label
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclasses.dataclass
class LoadedRun:
    """`SimResult`-shaped container rebuilt from disk — same surface the viz/metrics layer reads."""

    config: SimConfig
    intents: list[OperationalIntent]
    verified: bool
    static_walls: list = dataclasses.field(default_factory=list)   # always-active walls (from ledger_end)

    @property
    def accepted(self) -> list[OperationalIntent]:
        return [i for i in self.intents if i.accepted]

    @property
    def denied(self) -> list[OperationalIntent]:
        return [i for i in self.intents if i.status is IntentStatus.REJECTED]

    def summary(self) -> dict:
        return metrics.aggregate_with_steady(self)  # type: ignore[arg-type]


def load_run(folder: Path | str) -> LoadedRun:
    """Rebuild a `SimResult`-shaped object from a saved run folder (the reverse of `save_run`).

    Reconstructs each flight's exact `Volume4D` reservation and flown centerline so a replay or
    analysis can run entirely from disk — no re-simulation needed.
    """
    folder = Path(folder)
    cfg_payload = json.loads((folder / "config.json").read_text())
    for k in ("region_size_m", "region_center_latlon"):
        if isinstance(cfg_payload.get(k), list):
            cfg_payload[k] = tuple(cfg_payload[k])
    cfg = SimConfig(**cfg_payload)

    scen = pd.read_parquet(folder / "scenario.parquet")
    traj = pd.read_parquet(folder / "trajectories.parquet")
    flights = pd.read_parquet(folder / "flights.parquet")
    res = pd.read_parquet(folder / "reservations.parquet")

    vols_by_flight: dict[int, list[Volume4D]] = {}
    for fid, grp in res.groupby("flight_id"):
        vols_by_flight[int(fid)] = [_volume_from_row(r) for r in grp.itertuples(index=False)]
    cl_by_flight: dict[int, list] = {}
    for fid, grp in traj.sort_values(["flight_id", "t"]).groupby("flight_id"):
        cl_by_flight[int(fid)] = [(vec(r.x, r.y, r.z), float(r.t))
                                  for r in grp.itertuples(index=False)]

    scen_by_id = {int(r.flight_id): r for r in scen.itertuples(index=False)}
    intents: list[OperationalIntent] = []
    for fr in flights.itertuples(index=False):
        fid = int(fr.flight_id)
        s = scen_by_id[fid]
        t_dep = None if s.t_departure == s.t_request else float(s.t_departure)
        req = FlightRequest(fid, vec(s.origin_x, s.origin_y, s.origin_z),
                            vec(s.dest_x, s.dest_y, s.dest_z), float(s.t_request),
                            t_departure=t_dep, uss_id=str(s.uss_id),
                            origin_terminal=_term_from_json(getattr(s, "origin_terminal", None)),
                            dest_terminal=_term_from_json(getattr(s, "dest_terminal", None)))
        accepted = bool(fr.accepted)
        intents.append(OperationalIntent(
            request=req,
            status=IntentStatus.ACCEPTED if accepted else IntentStatus.REJECTED,
            volumes=vols_by_flight.get(fid) if accepted else None,
            centerline=cl_by_flight.get(fid) if accepted else None,
            ground_delay_s=float(fr.ground_delay_s), air_hold_s=float(fr.air_hold_s),
            air_detour_m=float(fr.air_detour_m), altitude_change_m=float(fr.altitude_change_m),
            cost=float(fr.cost), denial_reason=DenialReason(fr.denial_reason), planner=str(fr.planner),
            solve_time_s=float(fr.solve_time_s),
        ))
    walls = []
    if (folder / "ledger_end.parquet").exists():   # always-active terminal walls → replay overlay
        walls = [_volume_from_row(r)
                 for r in pd.read_parquet(folder / "ledger_end.parquet").itertuples(index=False)]
    return LoadedRun(config=cfg, intents=intents, static_walls=walls, verified=bool(json.loads(
        (folder / "experiment.json").read_text()).get("verified", True)))


def _volume_from_row(r) -> Volume4D:
    if r.kind == "box":
        spec: Any = BoxSpec(center=(r.cx, r.cy, r.cz),
                            rot=tuple(json.loads(r.rot)), extents=tuple(json.loads(r.ext)))
    else:
        spec = CylinderSpec(cx=r.cx, cy=r.cy, radius=r.radius, z_lo=r.z_lo, z_hi=r.z_hi)
    tid = getattr(r, "terminal_id", None)
    if tid is not None and (tid != tid or tid == ""):   # NaN / empty → no terminal
        tid = None
    return Volume4D(spec, float(r.t_start), float(r.t_end), terminal_id=tid)


def save_sweep(rows: list[dict], *, root: Path | str = DEFAULT_ROOT, label: str = "sweep",
               experiment_args: dict | None = None) -> Path:
    """Persist a parameter sweep's aggregate rows as one parquet table + metadata."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    folder = Path(root) / f"{stamp}_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    flat = [{**r, "denials_by_reason": json.dumps(r.get("denials_by_reason", {}))} for r in rows]
    pd.DataFrame(flat).to_parquet(folder / "sweep.parquet", index=False)
    (folder / "env.json").write_text(json.dumps(_env_info(), indent=2))
    (folder / "experiment.json").write_text(json.dumps(
        {"experiment": label, "args": experiment_args or {}, "timestamp": stamp}, indent=2,
        default=str))
    return folder
