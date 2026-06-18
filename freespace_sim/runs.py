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
from .types import DenialReason, FlightRequest, IntentStatus, OperationalIntent, vec
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
    for mod in ("numpy", "fcl", "pandas", "pulp", "casadi", "trimesh"):
        try:
            versions[mod] = __import__(mod).__version__
        except (ImportError, AttributeError):
            versions[mod] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "versions": versions}


# --- flight-data frame builders --------------------------------------------


def scenario_frame(result: SimResult) -> pd.DataFrame:
    """Every generated flight request — the scenario, independent of what got accepted."""
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
    rows = []
    for i in result.accepted:
        for v in i.volumes or []:
            s = v.shape
            row = {"flight_id": i.request.flight_id, "t_start": v.t_start, "t_end": v.t_end}
            if isinstance(s, BoxSpec):
                row.update({"kind": "box", "cx": s.center[0], "cy": s.center[1], "cz": s.center[2],
                            "rot": json.dumps(list(s.rot)), "ext": json.dumps(list(s.extents)),
                            "radius": np.nan, "z_lo": np.nan, "z_hi": np.nan})
            else:
                row.update({"kind": "cyl", "cx": s.cx, "cy": s.cy, "cz": (s.z_lo + s.z_hi) / 2,
                            "rot": "", "ext": "", "radius": s.radius, "z_lo": s.z_lo, "z_hi": s.z_hi})
            rows.append(row)
    cols = ["flight_id", "kind", "t_start", "t_end", "cx", "cy", "cz",
            "rot", "ext", "radius", "z_lo", "z_hi"]
    return pd.DataFrame(rows, columns=cols)


# --- save / load -----------------------------------------------------------


def save_run(
    result: SimResult, *,
    root: Path | str = DEFAULT_ROOT,
    label: str = "run",
    experiment: str | None = None,
    experiment_args: dict | None = None,
    wall_seconds: float | None = None,
    write_replay: bool = True,
    index: bool = True,
) -> Path:
    """Write the full self-contained run folder and return its path.

    Captures config/env/git, the experiment identity + args, the scenario, the flown trajectories,
    the reserved 4D volumes, per-flight metrics, and (by default) the standalone replay HTML.
    """
    cfg = result.config
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    folder = Path(root) / f"{stamp}_{label}_{_config_hash(cfg)}"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    (folder / "env.json").write_text(json.dumps(_env_info(), indent=2))
    (folder / "git.json").write_text(json.dumps(_git_info(), indent=2))
    (folder / "experiment.json").write_text(json.dumps({
        "experiment": experiment or label,
        "args": experiment_args or {},
        "wall_seconds": wall_seconds,
        "timestamp": stamp,
        "planner": cfg.planner,
        "n_requests": len(result.intents),
        "verified": result.verified,
    }, indent=2, default=str))
    (folder / "summary.json").write_text(json.dumps(metrics.aggregate(result), indent=2))

    scenario_frame(result).to_parquet(folder / "scenario.parquet", index=False)
    trajectory_frame(result).to_parquet(folder / "trajectories.parquet", index=False)
    reservation_frame(result).to_parquet(folder / "reservations.parquet", index=False)
    metrics.flight_frame(result).to_parquet(folder / "flights.parquet", index=False)

    if write_replay:
        from . import viz_html
        viz_html.write_html(result, folder / "replay.html")

    if index:
        _append_index(result, folder, Path(root), wall_seconds)
    return folder


def _append_index(result: SimResult, folder: Path, root: Path, wall_seconds: float | None) -> None:
    """Append one queryable row per run to ``results/index.parquet``."""
    agg = metrics.aggregate(result)
    row = {"path": str(folder), "planner": result.config.planner,
           "lam_per_hour": result.config.lam_per_hour, "seed": result.config.seed,
           "wall_seconds": wall_seconds,
           **{k: agg[k] for k in ("n_requests", "n_accepted", "n_denied", "denial_rate",
                                  "mean_total_delay_s", "p95_total_delay_s", "mean_air_detour_m",
                                  "airspace_utilization", "mean_solve_time_s", "p95_solve_time_s",
                                  "max_solve_time_s", "total_solve_time_s", "verified")}}
    path = root / INDEX_FILENAME
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False)


@dataclasses.dataclass
class LoadedRun:
    """`SimResult`-shaped container rebuilt from disk — same surface the viz/metrics layer reads."""

    config: SimConfig
    intents: list[OperationalIntent]
    verified: bool

    @property
    def accepted(self) -> list[OperationalIntent]:
        return [i for i in self.intents if i.accepted]

    @property
    def denied(self) -> list[OperationalIntent]:
        return [i for i in self.intents if i.status is IntentStatus.REJECTED]

    def summary(self) -> dict:
        return metrics.aggregate(self)  # type: ignore[arg-type]


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
                            t_departure=t_dep, uss_id=str(s.uss_id))
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
    return LoadedRun(config=cfg, intents=intents, verified=bool(json.loads(
        (folder / "experiment.json").read_text()).get("verified", True)))


def _volume_from_row(r) -> Volume4D:
    if r.kind == "box":
        spec: Any = BoxSpec(center=(r.cx, r.cy, r.cz),
                            rot=tuple(json.loads(r.rot)), extents=tuple(json.loads(r.ext)))
    else:
        spec = CylinderSpec(cx=r.cx, cy=r.cy, radius=r.radius, z_lo=r.z_lo, z_hi=r.z_hi)
    return Volume4D(spec, float(r.t_start), float(r.t_end))


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
