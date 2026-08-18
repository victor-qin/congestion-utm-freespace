"""Run capture — freeze a `SimResult` to a self-contained, replayable folder under ``results/``.

Mirrors the sibling project's run tracking, adapted to continuous free space. Every run writes a
timestamped folder ``results/{ISO}_{label}_{hash}/`` holding **everything needed to reproduce,
analyse, or replay it without re-running the sim**:

    config.json          the exact SimConfig used
    scenario_spec.json   the resolved post-override ScenarioSpec recipe (when supplied)
    experiment.json      which experiment ran + its args + wall-clock seconds  (← "what was run")
    env.json / git.json  toolchain + commit (best-effort; this package is often used outside git)
    summary.json         headline aggregate
    scenario.parquet     EVERY generated flight request (origin/dest/filing time)  (← the scenario)
    trajectories.parquet what was actually flown — timed centerline waypoints per flight
    reservations.parquet what was reserved in 4D — every corridor box + hover cylinder + window
    flights.parquet      per-flight metrics rows
    index_row.parquet    this run's row of the cross-run index (its own copy — see rebuild_index)
    replay.html          the standalone scrubbable replay (the "video")

``load_run(folder)`` is the exact reverse: it rebuilds a `SimResult`-shaped object (config + intents
with their volumes and centerlines) so a replay or analysis can start from disk. An append-only
``results/index.parquet`` indexes every run for cross-run queries; concurrent writers serialise on a
lock and `rebuild_index` restores it from the per-run rows if one is ever lost anyway.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import metrics, volumes
from .config import SimConfig
from .geometry import BoxSpec, CylinderSpec
from .sim import SimResult
from .telemetry import _vol_row, conflict_frame, filed_volume_frame, terminal_frame
from .types import DenialReason, FlightRequest, IntentStatus, OperationalIntent, as_terminal, vec
from .volumes import Volume4D

DEFAULT_ROOT = Path("results")
INDEX_FILENAME = "index.parquet"
#: Each run's own copy of its index row. The shared index is appended by read-modify-write, so a
#: concurrent writer can drop a row; :func:`rebuild_index` restores it from these.
INDEX_ROW_FILENAME = "index_row.parquet"


# --- metadata captures -----------------------------------------------------


log = logging.getLogger(__name__)


def _config_hash(cfg: SimConfig, scenario_spec: dict | None = None) -> str:
    """Short digest of everything that makes a run a DIFFERENT run.

    ``SimConfig`` alone is not enough. Two scenarios can share a byte-identical SimConfig and still be
    different worlds, because the whole demand recipe — operator mix, per-USS rates, service radii, and
    the scheduling leads the lead arms vary — lives in ``DemandSpec``, which SimConfig never sees. The
    five FAA lead arms all hashed to a246cd5e, so under one ``--tag`` their run folders differed only by
    a second-granularity timestamp and same-second finishers merged into one directory. Fold the
    archived scenario recipe in when there is one.
    """
    payload = {"config": dataclasses.asdict(cfg)}
    if scenario_spec is not None:
        payload["scenario_spec"] = scenario_spec
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob, usedforsecurity=False).hexdigest()[:8]


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


def _opt_int(v) -> int | None:
    """A parquet cell back to ``int | None`` — None for a missing column or a NaN (unlinked) row."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    return int(v)


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
            # Round-trip link (return leg → its outbound). Without it a reloaded run cannot tell which
            # legs were paired, so nothing downstream could re-derive the schedule slip or re-anchor a
            # return post-hoc — the coupled t_departure above is the OUTCOME, not the relationship.
            # pandas has no nullable-int dtype by default, so an unlinked leg stores NaN and
            # load_run reads it back as None.
            "paired_outbound_id": r.paired_outbound_id,
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
    # Static terminals need not appear on a request (a scenario may place an unused hub), so persist
    # the lane edge alongside its wall.  Storing the derived radius also preserves the exact archived
    # geometry if the exit-lane formula changes in a future release.
    terms = {}
    for _, raw in (getattr(ledger, "_static_terms", []) or []):
        if (term := as_terminal(raw)) is not None:
            terms[str(term.id)] = term
    rows = []
    for j, v in enumerate(vols):
        tid = None if v.terminal_id is None else str(v.terminal_id)
        term = terms.get(tid)
        rows.append({"wall_idx": j, **_vol_row(v),
                     "exit_radius": (volumes.exit_radius(term, result.config)
                                     if term is not None else np.nan)})
    cols = ["wall_idx", "kind", "t_start", "t_end", "cx", "cy", "cz",
            "rot", "ext", "radius", "z_lo", "z_hi", "terminal_id", "exit_radius"]
    return pd.DataFrame(rows, columns=cols)


# --- save / load -----------------------------------------------------------


def _json_finite(value):
    """Replace non-finite floats with ``None`` so the payload is real JSON.

    ``json.dumps`` emits bare ``Infinity``/``NaN``, which RFC 8259 does not allow: strict
    parsers reject the file outright, and ``jq`` silently clamps to 1.8e308 — turning "no
    bound was ever computed" into a plausible-looking finite bound. ``null`` says the same
    thing without inviting either failure.
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(v) for v in value]
    return value


def save_run(
    result: SimResult, *,
    root: Path | str = DEFAULT_ROOT,
    label: str = "run",
    experiment: str | None = None,
    experiment_args: dict | None = None,
    scenario_spec: dict | None = None,
    wall_seconds: float | None = None,
    scenario: str | None = None,
    demand: str | None = None,
    write_replay: bool = True,
    index: bool = True,
    window_frac: float = 0.9,
) -> Path:
    """Write the full self-contained run folder and return its path.

    Captures config/env/git, the experiment identity + args, the resolved scenario, the flown trajectories,
    the reserved 4D volumes, per-flight metrics, and (by default) the standalone replay HTML. Everything
    is parquet + json — deliberately NOT pickle: portable, inspectable, safe to sync to the run store,
    and Python-version-independent. The analytical geometry stored in reservations/ledger_end is enough
    to rebuild every ``Volume4D`` on load (see :func:`load_run` / :func:`_volume_from_row`).

    ``summary.json`` carries the whole-run headline numbers **and** their steady-state twin (metrics
    over the representative density plateau — issue #25) in a nested ``steady_state`` block; ``window_frac``
    tunes the plateau threshold. The replay spans the REALIZED operation — first reservation through last
    to clear (:func:`metrics.simulation_window`) — so the post-horizon return tail these scenarios exist
    to produce stays visible, and an early-finishing run no longer scrubs through an empty sky out to
    ``horizon_s``.
    """
    cfg = result.config
    agg = metrics.aggregate_with_steady(result, frac=window_frac)
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    # Seed in the name as well as the hash: a sweep's folders are read by eye far more often than they
    # are parsed, and `_s0/_s1/_s2` is the axis one actually scans for. The hash still carries it.
    base_name = f"{stamp}_{label}_s{cfg.seed}_{_config_hash(cfg, scenario_spec)}"
    # exist_ok=False in a claim loop, NOT exist_ok=True. Two runs landing on one name used to merge
    # their parquet into a single directory — a silent corruption, and the likeliest way to hit it is a
    # Slurm array whose tasks share a tag and finish inside the same second. Suffixing keeps both runs
    # (losing a finished multi-hour run to a raise would be worse) and says so.
    folder = Path(root) / base_name
    for attempt in range(2, 1000):
        try:
            folder.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            folder = Path(root) / f"{base_name}__{attempt}"
            log.warning("run folder %s already exists — writing to %s instead, so the two runs cannot "
                        "interleave parquet into one directory", base_name, folder.name)
    else:
        raise RuntimeError(f"could not find a free run folder for {base_name} after 998 attempts")

    (folder / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    (folder / "env.json").write_text(json.dumps(_env_info(), indent=2))
    (folder / "git.json").write_text(json.dumps(_git_info(), indent=2))
    scenario_description = scenario_spec.get("description") if scenario_spec is not None else None
    if scenario_spec is not None:
        (folder / "scenario_spec.json").write_text(
            json.dumps(scenario_spec, indent=2, default=str)
        )
    (folder / "experiment.json").write_text(json.dumps({
        "experiment": experiment or label,
        "scenario": scenario,
        "scenario_description": scenario_description,
        "demand": demand,
        "tag": label,
        "args": experiment_args or {},
        "wall_seconds": wall_seconds,
        "timestamp": stamp,
        "planner": cfg.planner,
        "n_requests": len(result.intents),
        "simulation_start_s": agg["simulation_start_s"],
        "simulation_end_s": agg["simulation_end_s"],
        "simulation_duration_s": agg["simulation_duration_s"],
        "verified": result.verified,
    }, indent=2, default=str))
    (folder / "summary.json").write_text(json.dumps(agg, indent=2))
    # Whole-schedule solver diagnostics, when the run had a solver (colgen). Without this a
    # column-generation run is indistinguishable on disk from a converged one: it files a
    # complete, feasible accepted set whether it ran to optimality or stopped at iteration 1.
    # `default=str` because the stats carry tuples of ids and occasional non-JSON scalars.
    if getattr(result, "planner_stats", None):
        (folder / "planner_stats.json").write_text(
            json.dumps(_json_finite(result.planner_stats), indent=2, default=str)
        )

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
        viz_html.write_html(result, folder / "replay.html")

    if index:
        row_df = _index_row(result, folder, wall_seconds, scenario=scenario,
                            scenario_description=scenario_description, tag=label, demand=demand, agg=agg)
        # Own copy first, and deliberately OUTSIDE the guard below: it is the only thing
        # `rebuild_index` can put this row back from, so swallowing its failure would lose the run
        # from every readout silently. A folder that refuses a write this late (a filling disk) also
        # casts doubt on the artifacts written above it, which are themselves unguarded — that is a
        # real failure and must still raise.
        row_df.to_parquet(folder / INDEX_ROW_FILENAME, index=False)
        try:
            _append_index(row_df, Path(root))
        except Exception:
            # The shared index is derived data; every fact it holds about this run is now in the
            # folder's own index_row.parquet. Failing here would report a finished multi-hour
            # run as a failed one to any wrapper watching the exit code.
            log.warning("cross-run index update failed — run folder %s is complete and keeps its "
                        "own index_row.parquet; run freespace_sim.runs.rebuild_index() to restore "
                        "index.parquet", folder.name, exc_info=True)
    return folder


def _index_row(result: SimResult, folder: Path, wall_seconds: float | None,
               *, scenario: str | None = None, tag: str | None = None,
               demand: str | None = None, agg: dict | None = None,
               scenario_description: str | None = None) -> pd.DataFrame:
    """Build the one queryable row this run contributes to ``results/index.parquet``.

    The ``scenario`` / ``tag`` / ``demand`` columns are the join keys cross-run readouts filter on:
    a batch sweep stamps every run with the same ``tag`` so a readout can select exactly its runs.
    ``agg`` may be a precomputed :func:`metrics.aggregate_with_steady` (avoids recomputing it); the
    ``steady_*`` / ``window_*`` columns carry the steady-state twin of the headline metrics so a
    cross-run curve can plot the de-biased trend alongside the whole-run one (issue #25).
    """
    cfg = result.config
    if agg is None:
        agg = metrics.aggregate_with_steady(result)
    planner_stats = getattr(result, "planner_stats", None) or {}
    steady = agg.get("steady_state", {})
    steady_cols = {f"steady_{k}": steady.get(k) for k in
                   ("mean_total_delay_s", "p50_total_delay_s", "p95_total_delay_s",
                    "throughput_per_h", "denial_rate", "congestion_denial_rate")}
    steady_cols["window_lo"] = steady.get("window_lo")
    steady_cols["window_hi"] = steady.get("window_hi")
    row = {"path": str(folder), "scenario": scenario,
           # index.parquet is APPENDED to, so one file accumulates rows from different metric
           # definitions. Without this stamp a cross-run mean silently mixes them — most sharply
           # airspace_utilization, whose denominator changed for every scenario in version 2.
           "metrics_version": metrics.METRICS_VERSION,
           "scenario_description": scenario_description, "tag": tag, "demand": demand,
           "planner": cfg.planner, "lam_per_hour": cfg.lam_per_hour, "seed": cfg.seed,
           "horizon_s": cfg.horizon_s,
           "demand_duration_s": cfg.effective_demand_duration_s,
           "simulation_start_s": agg["simulation_start_s"],
           "simulation_end_s": agg["simulation_end_s"],
           "simulation_duration_s": agg["simulation_duration_s"],
           "region_w": cfg.region_size_m[0], "region_h": cfg.region_size_m[1],
           # None for per-flight planners, which have neither. For a whole-schedule solver these
           # are what separates "we have six colgen runs" from "we have six colgen runs, five of
           # which stopped at iteration 1" -- without them that needs opening every folder.
           #
           # They also mark the rows whose `*_solve_time_s` columns are not comparable with the
           # rest: a whole-schedule planner has no per-flight solve, so `colgen` files the SAME
           # amortized share (solve wall / n_flights) on every intent. Its mean is the amortized
           # share, its p95 and max are that share again, and its total is the solve. Against an
           # FCFS run those columns describe a different quantity, so filter on
           # `planner_termination.isna()` before comparing them.
           "planner_termination": planner_stats.get("termination_reason"),
           "planner_iterations": planner_stats.get("iterations"),
           "wall_seconds": wall_seconds,
           "has_telemetry": result.telemetry is not None,
           **{k: agg[k] for k in ("n_uss", "n_requests", "n_accepted", "n_denied", "denial_rate",
                                  "congestion_denial_rate", "offered_load_per_h", "throughput_per_h",
                                  "mean_total_delay_s", "p95_total_delay_s", "mean_air_detour_m",
                                  "mean_lattice_overhead_m", "mean_deconfliction_detour_m",
                                  "mean_stretch", "mean_cost",
                                  "airspace_utilization", "denial_rate_spread", "mean_delay_spread",
                                  "mean_solve_time_s", "p95_solve_time_s",
                                  "max_solve_time_s", "total_solve_time_s", "verified")},
           **steady_cols}
    return pd.DataFrame([row])


def _append_index(row_df: pd.DataFrame, root: Path) -> None:
    """Merge one run's row into the shared ``results/index.parquet`` under the cross-process lock.

    The caller has already written the row to the run's own folder, so anything lost here — a race,
    an unreadable file, a filesystem that will not lock — is recoverable by :func:`rebuild_index`.
    """
    path = root / INDEX_FILENAME
    rebuild = False
    with _index_lock(root):
        try:
            out = pd.concat([pd.read_parquet(path), row_df], ignore_index=True) if path.exists() else row_df
        except Exception as exc:
            # An unreadable index (a torn write from a crash or an unlocked pre-fix appender)
            # otherwise fails EVERY subsequent run against this root at its final step. Sideline
            # it and rebuild from the per-run rows instead; if the sideline itself failed, leave
            # the file for a human and skip the append — this run's row is safe in its folder.
            rebuild = _sideline_corrupt_index(path, exc) is not None
            out = None
        if out is not None:
            out.to_parquet(path, index=False)
    if rebuild:
        # Outside the lock: `rebuild_index` takes it again, and a second flock on the same file
        # in one process blocks. The row written above is on disk, so the rebuild includes it.
        rebuild_index(root)


@contextlib.contextmanager
def _index_lock(root: Path):
    """Serialise the index's read-modify-write (read whole file → concat → rewrite) across processes.

    Without it, two runs finishing together read the same base and the second rewrite drops the
    first's row — the normal case for a Slurm array, whose tasks start together and take about the
    same time, leaving the sweep silently one arm short.

    Best-effort: ``flock`` is absent on Windows and unreliable on some network filesystems, and
    neither should abort a finished run. :func:`rebuild_index` is what makes that safe to accept.
    """
    try:
        import fcntl
    except ImportError:                                  # non-POSIX: no lock; folder rows still land
        yield
        return
    root.mkdir(parents=True, exist_ok=True)
    with open(root / (INDEX_FILENAME + ".lock"), "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX)
        except OSError as exc:                           # filesystem without working flock
            log.warning("could not lock the run index (%s) — concurrent writers may drop rows; "
                        "run freespace_sim.runs.rebuild_index() afterwards to restore them", exc)
            yield
            return
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _sideline_corrupt_index(path: Path, exc: Exception) -> Path | None:
    """Rename an unreadable ``index.parquet`` out of the way, keeping its bytes for salvage.

    Rename rather than delete: rows of runs archived before ``index_row.parquet`` existed live
    only in this file, and the sidelined copy is the single remaining artifact a manual repair
    could recover them from. Returns the quarantine path, or ``None`` if the rename failed too
    (a read-only or misbehaving filesystem), in which case the file is left untouched.
    """
    quarantine = path.with_name(
        f"{path.name}.corrupt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    try:
        path.rename(quarantine)
    except OSError as rename_exc:
        log.warning("the run index %s is unreadable (%s) and could not be sidelined (%s) — "
                    "move it aside manually, then run freespace_sim.runs.rebuild_index()",
                    path, exc, rename_exc)
        return None
    log.warning("the run index %s was unreadable (%s) — moved it to %s and rebuilding from each "
                "run folder's index_row.parquet; rows of runs saved without index_row.parquet "
                "survive only in the sidelined file", path.name, exc, quarantine.name)
    return quarantine


def rebuild_index(root: Path | str = DEFAULT_ROOT) -> pd.DataFrame:
    """Reconstitute ``index.parquet`` from each run folder's own ``index_row.parquet``, and return it.

    A dropped index row is invisible — a cross-run readout just reports one run fewer, with no error
    and no gap to notice. Call this after a batch array (having every task call it means whichever
    finishes last leaves a complete index, whatever the shared filesystem did with the lock).

    Non-destructive and idempotent: index rows whose folder has no copy (runs archived before this
    file existed, or folders since deleted) are KEPT; where both exist the folder's copy wins.
    An index that cannot be read at all is sidelined to ``index.parquet.corrupt-<stamp>`` and the
    rebuild proceeds from the folder rows alone.
    """
    root = Path(root)
    rows = []
    for row_path in sorted(root.glob(f"*/{INDEX_ROW_FILENAME}")):
        try:
            rows.append(pd.read_parquet(row_path))
        except Exception as exc:                         # one half-written row must not sink the rest
            log.warning("skipping unreadable %s: %s", row_path, exc)
    folder_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    with _index_lock(root):
        try:
            existing = load_index(root)
        except Exception as exc:
            # The repair tool must not die on the very file it exists to repair. Orphan rows in
            # an unreadable index are beyond reach either way; sidelining keeps their bytes.
            _sideline_corrupt_index(root / INDEX_FILENAME, exc)
            existing = pd.DataFrame()
        if len(existing) and len(folder_rows):
            orphans = existing[~existing["path"].isin(set(folder_rows["path"]))]
            out = pd.concat([orphans, folder_rows], ignore_index=True)
        else:
            out = folder_rows if len(folder_rows) else existing
        if len(out):
            out = out.sort_values("path", kind="stable", ignore_index=True)   # folder name is an ISO stamp
            out.to_parquet(root / INDEX_FILENAME, index=False)
    return out


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
    # Lane edges persisted with static walls.  Old archives omit this field and remain loadable; their
    # request-associated terminals can still be reconstructed from scenario.parquet.
    static_exit_radii: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def accepted(self) -> list[OperationalIntent]:
        return [i for i in self.intents if i.accepted]

    @property
    def denied(self) -> list[OperationalIntent]:
        return [i for i in self.intents if i.status is IntentStatus.REJECTED]

    def summary(self) -> dict:
        return metrics.aggregate_with_steady(self)  # type: ignore[arg-type]


def load_scenario_spec(folder: Path | str):
    """Rebuild the ``ScenarioSpec`` a run was launched from, or ``None`` if the folder has none.

    Complements :func:`load_run`, which reconstructs the *result*. This reconstructs the *recipe* —
    the resolved post-override world — so a run can be re-executed (at a new seed, planner, or λ)
    from the folder alone instead of by remembering the command line.

    ``None`` is expected, not exceptional: ``scenario_spec.json`` is written only when the caller
    supplies one (``experiments.run`` does; ``analysis/altitude_benchmark.py`` does not), and no run
    archived before the file existed has it.
    """
    from .scenarios.spec import ScenarioSpec

    path = Path(folder) / "scenario_spec.json"
    if not path.exists():
        return None
    return ScenarioSpec.from_json_dict(json.loads(path.read_text()))


def load_run(folder: Path | str) -> LoadedRun:
    """Rebuild a `SimResult`-shaped object from a saved run folder (the reverse of `save_run`).

    Reconstructs each flight's exact `Volume4D` reservation and flown centerline so a replay or
    analysis can run entirely from disk — no re-simulation needed.
    """
    folder = Path(folder)
    cfg_payload = json.loads((folder / "config.json").read_text())
    for k in ("region_size_m", "region_center_latlon", "flight_levels_m"):
        if isinstance(cfg_payload.get(k), list):
            cfg_payload[k] = tuple(cfg_payload[k])
    # Runs archived before the per-second cost normalization stored cost_air_lateral_per_m /
    # cost_altitude_change_per_m directly; both are derived @properties now. Back-convert them into
    # the per-second knobs BEFORE the whitelist drop below — otherwise the old value is silently
    # discarded and the run replays under today's defaults (0.1 instead of 3.0), i.e. a different
    # cost model than it was planned with. Multiplying back by the same speed/climb_rate the property
    # divides by reproduces the archived weight exactly. Stay drift-tolerant like the rest of this
    # function: fall back to the SimConfig default if the companion speed field predates the archive
    # (a bare subscript here would KeyError inside the very block meant to absorb schema drift), and
    # never clobber a per-second value the payload already carries.
    for per_m_key, per_s_key, scale_key in (
        ("cost_air_lateral_per_m", "cost_air_lateral_per_s", "nominal_speed_mps"),
        ("cost_altitude_change_per_m", "cost_altitude_change_per_s", "climb_rate_mps"),
    ):
        if per_m_key in cfg_payload:
            scale = cfg_payload.get(scale_key, getattr(SimConfig, scale_key))
            cfg_payload.setdefault(per_s_key, cfg_payload.pop(per_m_key) * scale)
    # Tolerate schema drift: drop keys that are no longer SimConfig fields (e.g. cruise_level_m / z_min_m /
    # z_max_m — now derived @properties) so runs archived before that change still load.
    _fields = {f.name for f in dataclasses.fields(SimConfig)}
    cfg = SimConfig(**{k: v for k, v in cfg_payload.items() if k in _fields})

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
                            dest_terminal=_term_from_json(getattr(s, "dest_terminal", None)),
                            # getattr + NaN check: runs archived before the column existed have neither
                            # the attribute nor a value, and an unlinked leg stores NaN either way.
                            paired_outbound_id=_opt_int(getattr(s, "paired_outbound_id", None)))
        accepted = bool(fr.accepted)
        intents.append(OperationalIntent(
            request=req,
            status=IntentStatus.ACCEPTED if accepted else IntentStatus.REJECTED,
            volumes=vols_by_flight.get(fid) if accepted else None,
            centerline=cl_by_flight.get(fid) if accepted else None,
            ground_delay_s=float(fr.ground_delay_s), air_hold_s=float(fr.air_hold_s),
            air_detour_m=float(fr.air_detour_m), altitude_change_m=float(fr.altitude_change_m),
            # getattr: runs archived before the lattice split have no such column (0.0 ⇒ the whole
            # detour reads as traffic-attributable, which is what those runs already reported).
            lattice_overhead_m=float(getattr(fr, "lattice_overhead_m", 0.0)),
            cost=float(fr.cost), denial_reason=DenialReason(fr.denial_reason), planner=str(fr.planner),
            solve_time_s=float(fr.solve_time_s),
        ))
    walls = []
    static_exit_radii = {}
    if (folder / "ledger_end.parquet").exists():   # always-active terminal walls → replay overlay
        ledger_end = pd.read_parquet(folder / "ledger_end.parquet")
        for row in ledger_end.itertuples(index=False):
            wall = _volume_from_row(row)
            walls.append(wall)
            er = getattr(row, "exit_radius", np.nan)   # archives written before this column still load
            if wall.terminal_id is not None and pd.notna(er):
                static_exit_radii[str(wall.terminal_id)] = float(er)
    return LoadedRun(config=cfg, intents=intents, static_walls=walls,
                     static_exit_radii=static_exit_radii, verified=bool(json.loads(
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
