"""Metrics — turn a `SimResult` into per-flight rows and aggregate rollups.

Two surfaces:

- ``flight_frame(result)`` → a tidy pandas DataFrame, one row per `OperationalIntent`, carrying the
  cost decomposition (delay / hold / detour / altitude), efficiency (stretch = flown ÷ straight),
  and each flight's reserved **volume-seconds** (its slice of the 4D airspace pie).
- ``aggregate(result)`` → a flat dict of headline numbers for the λ-sweep: acceptance/denial,
  delay & detour distributions, throughput, and **airspace utilization** (reserved volume-seconds
  ÷ the whole region × horizon) — the free-space analog of the sibling project's hex-occupancy.

The congestion story the experiment tells is the relationship between *offered load* (requests/hour)
and these outcomes: as λ rises, the FCFS newcomer is pushed into ever costlier delays/detours until
the budget can't absorb it and denials climb. Keeping `BUDGET_EXCEEDED` denials (real congestion)
separate from `SEARCH_EXHAUSTED` (a planner artifact) keeps that signal honest — see `DenialReason`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import SimConfig
from .geometry import BoxSpec, CylinderSpec
from .sim import SimResult
from .types import DenialReason, OperationalIntent
from .volumes import Volume4D


def shape_volume_m3(shape) -> float:
    """Spatial volume (m³) of a reservation's 3D shape — box L·W·H or cylinder π·r²·h."""
    if isinstance(shape, BoxSpec):
        lx, wy, hz = shape.extents
        return float(lx * wy * hz)
    if isinstance(shape, CylinderSpec):
        return float(math.pi * shape.radius**2 * (shape.z_hi - shape.z_lo))
    raise TypeError(f"unknown shape {type(shape).__name__}")


def reserved_volume_seconds(volumes: list[Volume4D] | None, horizon_s: float) -> float:
    """Sum of (spatial volume × time-window duration) over a flight's reservation, in m³·s.

    The time window is clamped to ``[0, horizon_s]`` so an open-ended hover reservation
    (``t_end`` ~ 1e6 in some fixtures) can't dominate the sum with off-horizon seconds.
    """
    if not volumes:
        return 0.0
    total = 0.0
    for v in volumes:
        dur = min(v.t_end, horizon_s) - max(v.t_start, 0.0)
        if dur > 0.0:
            total += shape_volume_m3(v.shape) * dur
    return total


def _flown_horizontal_m(intent: OperationalIntent) -> float:
    """Horizontal path length actually flown, summed along the (projected) centerline."""
    if not intent.centerline:
        return float("nan")
    pts = np.array([p[0][:2] for p in intent.centerline], float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _straight_horizontal_m(intent: OperationalIntent) -> float:
    """Great-circle-free straight-line horizontal distance origin→dest."""
    o, d = intent.request.origin[:2], intent.request.dest[:2]
    return float(np.linalg.norm(np.asarray(d, float) - np.asarray(o, float)))


def total_delay_s(intent: OperationalIntent, cfg: SimConfig) -> float:
    """Unified congestion lateness (s): the seconds a flight loses *to other traffic*.

    Ground hold + air loiter + detour-time (extra horizontal metres ÷ cruise speed). This is the
    free-space analog of the hex repo's single ``delay`` scalar — excess over the unimpeded flight.
    Mandatory takeoff/landing climb is deliberately excluded: it's a constant every flight pays, not
    a congestion signal. NaN for denied flights (they never arrive). See :func:`flight_row`.
    """
    if not intent.accepted:
        return float("nan")
    return (
        intent.ground_delay_s
        + intent.air_hold_s
        + intent.air_detour_m / cfg.nominal_speed_mps
    )


def nominal_flight_time_s(straight_m: float, cfg: SimConfig) -> float:
    """Unimpeded door-to-door air time (s): straight cruise + the mandatory climb and descent."""
    return straight_m / cfg.nominal_speed_mps + 2.0 * cfg.climb_time_s


def flight_row(intent: OperationalIntent, cfg: SimConfig) -> dict:
    """One tidy record for a single operational intent (accepted or denied)."""
    straight = _straight_horizontal_m(intent)
    flown = _flown_horizontal_m(intent)
    stretch = (flown / straight) if (intent.accepted and straight > 1e-9) else float("nan")
    td = total_delay_s(intent, cfg)
    # delay as a fraction of the actual trip time — bounded [0, 100), comparable across trip lengths
    nominal = nominal_flight_time_s(straight, cfg)
    delay_pct = (100.0 * td / (nominal + td)) if (intent.accepted and nominal + td > 0) else float("nan")
    # trip-time inflation: actual trip time (straight-line flight time + all delay) ÷ the ideal
    # straight-line time. ≥ 1, UNBOUNDED — 1.0 = flew the ideal with no wait, 2.0 = took twice as long.
    # The unbounded complement of delay_pct: trip_time_ratio == 100 / (100 - delay_pct).
    trip_time_ratio = ((nominal + td) / nominal) if (intent.accepted and nominal > 1e-9) else float("nan")
    return {
        "flight_id": intent.request.flight_id,
        "uss_id": intent.request.uss_id,
        "t_request": intent.request.t_request,
        "planner": intent.planner,
        "status": intent.status.value,
        "accepted": intent.accepted,
        "denied": intent.status.name == "REJECTED",
        "denial_reason": intent.denial_reason.value,
        "ground_delay_s": intent.ground_delay_s,
        "air_hold_s": intent.air_hold_s,
        "air_detour_m": intent.air_detour_m,
        # detour expressed as lateness-seconds; ground_delay + air_hold + detour_time == total_delay
        "detour_time_s": (intent.air_detour_m / cfg.nominal_speed_mps) if intent.accepted else float("nan"),
        "altitude_change_m": intent.altitude_change_m,
        "total_delay_s": td,                      # unified congestion lateness (s)
        "delay_pct": delay_pct,                    # ... as % of the flight's total trip time
        "trip_time_ratio": trip_time_ratio,        # (straight-line time + delay) / straight-line time
        "cost": intent.cost,
        "solve_time_s": intent.solve_time_s,   # planner wall time for this flight
        "straight_line_m": straight,
        "flown_m": flown,
        "stretch": stretch,
        "reserved_vol_m3_s": reserved_volume_seconds(intent.volumes, cfg.horizon_s),
    }


def flight_frame(result: SimResult) -> pd.DataFrame:
    """Per-flight metrics table — one row per intent, FCFS order preserved."""
    return pd.DataFrame([flight_row(i, result.config) for i in result.intents])


def _q(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if len(series) else 0.0


def _rollup(df: pd.DataFrame, cfg: SimConfig) -> dict:
    """Group-level rollup of a (sub)frame of flight rows — shared by ``aggregate`` (the whole run)
    and ``per_uss_frame`` (one operator's slice). Denominators (horizon, region capacity) are the
    *run's*, so a per-USS ``airspace_utilization`` reads as that operator's share of the whole sky.
    """
    acc = df[df["accepted"]]
    den = df[df["denied"]]
    horizon_h = cfg.horizon_s / 3600.0
    # Vertical extent of the usable airspace. When the altitude band is collapsed to a single flight
    # level (z_max == z_min), fall back to the corridor slab height — the vertical footprint a flight
    # actually occupies at that level — so utilization stays meaningful instead of dividing by zero.
    vert_extent_m = max(cfg.z_max_m - cfg.z_min_m, cfg.corridor_height_m)
    region_vol_m3 = cfg.region_size_m[0] * cfg.region_size_m[1] * vert_extent_m
    airspace_capacity_m3_s = region_vol_m3 * cfg.horizon_s
    # split real congestion (budget) from the planner's search artifact
    n_budget = int((den["denial_reason"] == DenialReason.BUDGET_EXCEEDED.value).sum()) if len(den) else 0
    return {
        "n_requests": len(df),
        "n_accepted": int(len(acc)),
        "n_denied": int(len(den)),
        "denial_rate": len(den) / max(1, len(df)),
        "congestion_denial_rate": n_budget / max(1, len(df)),  # budget-only (real congestion)
        "offered_load_per_h": len(df) / max(horizon_h, 1e-9),
        "throughput_per_h": len(acc) / max(horizon_h, 1e-9),
        "mean_ground_delay_s": float(acc["ground_delay_s"].mean()) if len(acc) else 0.0,
        "p95_ground_delay_s": _q(acc["ground_delay_s"], 0.95),
        "mean_total_delay_s": float(acc["total_delay_s"].mean()) if len(acc) else 0.0,
        "p50_total_delay_s": _q(acc["total_delay_s"], 0.50),
        "p95_total_delay_s": _q(acc["total_delay_s"], 0.95),
        "mean_delay_pct": float(acc["delay_pct"].mean()) if len(acc) else 0.0,
        "p95_delay_pct": _q(acc["delay_pct"], 0.95),
        "mean_air_detour_m": float(acc["air_detour_m"].mean()) if len(acc) else 0.0,
        "p95_air_detour_m": _q(acc["air_detour_m"], 0.95),
        "mean_stretch": float(acc["stretch"].mean()) if len(acc) else 1.0,
        "mean_cost": float(acc["cost"].mean()) if len(acc) else 0.0,
        # planner runtime over ALL flights (denials included — they often exhaust the search)
        "mean_solve_time_s": float(df["solve_time_s"].mean()) if len(df) else 0.0,
        "p95_solve_time_s": _q(df["solve_time_s"], 0.95),
        "max_solve_time_s": float(df["solve_time_s"].max()) if len(df) else 0.0,
        "total_solve_time_s": float(df["solve_time_s"].sum()),
        "reserved_vol_m3_s": float(df["reserved_vol_m3_s"].sum()),
        "airspace_utilization": float(df["reserved_vol_m3_s"].sum()) / max(airspace_capacity_m3_s, 1e-9),
    }


def _per_uss_table(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    total_acc = int(df["accepted"].sum()) if len(df) else 0
    rows = []
    for uss_id, g in df.groupby("uss_id", sort=True):
        acc = g[g["accepted"]]
        rows.append({
            "uss_id": uss_id,
            **_rollup(g, cfg),
            # per-USS-only: flight length (confirms hub-demand shortening) + share of the throughput
            "mean_straight_line_m": float(acc["straight_line_m"].mean()) if len(acc) else 0.0,
            "share_of_accepted": (len(acc) / total_acc) if total_acc else 0.0,
        })
    return pd.DataFrame(rows)


def per_uss_frame(result: SimResult) -> pd.DataFrame:
    """One metrics row per USS — the per-operator slice of a (multi-)USS run. Each row's counts and
    reserved volume sum to the overall ``aggregate`` totals (see tests)."""
    return _per_uss_table(flight_frame(result), result.config)


def aggregate(result: SimResult) -> dict:
    """Flat headline rollup for one run — the row a λ-sweep collects."""
    cfg = result.config
    df = flight_frame(result)
    den = df[df["denied"]]
    by_reason = den["denial_reason"].value_counts().to_dict() if len(den) else {}

    # cross-USS fairness: does one operator systematically lose under FCFS? (0 when single-USS)
    per_uss = _per_uss_table(df, cfg)
    n_uss = int(len(per_uss))
    if n_uss > 1:
        denial_rate_spread = float(per_uss["denial_rate"].max() - per_uss["denial_rate"].min())
        mean_delay_spread = float(per_uss["mean_total_delay_s"].max() - per_uss["mean_total_delay_s"].min())
    else:
        denial_rate_spread = mean_delay_spread = 0.0

    return {
        "lam_per_hour": cfg.lam_per_hour,
        "seed": cfg.seed,
        "planner": cfg.planner,
        **_rollup(df, cfg),
        "denials_by_reason": by_reason,
        "n_uss": n_uss,
        "denial_rate_spread": denial_rate_spread,
        "mean_delay_spread": mean_delay_spread,
        "verified": result.verified,
    }
