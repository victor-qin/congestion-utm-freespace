"""Metrics — turn a `SimResult` into per-flight rows and aggregate rollups.

Two surfaces:

- ``flight_frame(result)`` → a tidy pandas DataFrame, one row per `OperationalIntent`. It carries each
  flight's congestion in BOTH currencies — the planner's **cost** split per lever (the ``*_cost``
  columns, which reconcile exactly to ``cost`` == :func:`cost.trajectory_cost`) and the **real seconds**
  each lever costs (``ground_delay_s`` / ``air_hold_s`` / ``detour_time_s``, plus altitude's two
  readings) — so you can read what the optimiser *paid* AND how that pay translates into time. The two
  currencies diverge wherever a cost weight isn't 1 s/unit (a hover-second costs 3×; a detour-metre is
  ~1/30 s); altitude has two honest time readings (physical vs cost-equivalent) and we record both. Plus
  efficiency (stretch = flown ÷ straight) and reserved **volume-seconds** (its slice of the 4D airspace
  pie). See :func:`cost_breakdown` / :func:`delay_breakdown_s`.
- ``aggregate(result)`` → a flat dict of headline numbers for the λ-sweep: acceptance/denial,
  delay & detour distributions, throughput, and **airspace utilization** (reserved volume-seconds
  ÷ the whole region × horizon) — the free-space analog of the sibling project's hex-occupancy.
  ``aggregate`` also accepts a ``window`` to measure only a slice of the run; :func:`aggregate_with_steady`
  reports the whole-run numbers next to their **steady-state** twin (metrics over the representative
  density plateau, dropping the ramp-up/ramp-down tails — issue #25).

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


def reserved_volume_seconds(volumes: list[Volume4D] | None, t_lo: float, t_hi: float) -> float:
    """Sum of (spatial volume × time-window duration) over a flight's reservation, in m³·s.

    Each volume's window is clamped to ``[t_lo, t_hi]`` so (a) an open-ended hover reservation
    (``t_end`` ~ 1e6 in some fixtures) can't dominate the sum with off-window seconds, and (b) a
    steady-state window measures only the volume-seconds a flight spends inside it. Pass
    ``(0.0, horizon_s)`` for the whole-run figure.
    """
    if not volumes:
        return 0.0
    total = 0.0
    for v in volumes:
        dur = min(v.t_end, t_hi) - max(v.t_start, t_lo)
        if dur > 0.0:
            total += shape_volume_m3(v.shape) * dur
    return total


# --- steady-state measurement window ----------------------------------------------------------------
# A run's airborne density is a trapezoid: it ramps up as the sky fills from empty, plateaus, then ramps
# down as the last flights land (and, with return traffic, past the horizon). Metrics taken over the
# whole run are diluted by the low-density ramps — a flight filed at t≈0 waits on almost-empty airspace,
# one filed near the horizon flies into a thinning tail. These helpers find the plateau so delay /
# throughput / denial can be measured where the airspace density is representative. This is the
# principled replacement for the removed ``clip_returns_to_horizon`` demand hack (issue #25): run the
# natural demand (tails and all), but *measure* only the representative window.


def _airborne_interval(intent: OperationalIntent) -> tuple[float, float] | None:
    """The [takeoff, land] span over which a flight occupies airspace — its reservation's time
    envelope (earliest ``t_start`` → latest ``t_end``), or the centerline span if volumes are absent.
    ``None`` when neither is present (a denied flight)."""
    if intent.volumes:
        return (min(v.t_start for v in intent.volumes), max(v.t_end for v in intent.volumes))
    if intent.centerline:
        return (intent.centerline[0][1], intent.centerline[-1][1])
    return None


def density_timeseries(result: SimResult, dt: float | None = None, kind: str = "count",
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Airborne density over time on a uniform ``dt`` grid spanning ``[0, T]``.

    ``kind="count"`` → concurrent airborne flights (each accepted flight contributes +1 over its
    :func:`_airborne_interval`); ``kind="volume"`` → active reserved spatial volume in m³ (each
    ``Volume4D`` contributes its ``shape_volume_m3`` over ``[t_start, t_end)``) — smoother, the
    instantaneous-rate analog of :func:`reserved_volume_seconds`. Built with a difference array, so it
    is O(flights + grid). Returns ``(t_grid, density)``; ``(array([0.]), array([0.]))`` if nothing flew.
    """
    cfg = result.config
    dt = cfg.dt_s if dt is None else dt
    acc = result.accepted
    if not acc:
        return np.array([0.0]), np.array([0.0])
    if kind == "count":
        contribs = [(*iv, 1.0) for i in acc if (iv := _airborne_interval(i)) is not None]
    elif kind == "volume":
        contribs = [(v.t_start, v.t_end, shape_volume_m3(v.shape))
                    for i in acc for v in (i.volumes or [])]
    else:
        raise ValueError(f"kind must be 'count' or 'volume', got {kind!r}")
    if not contribs:
        return np.array([0.0]), np.array([0.0])
    # Cap the grid at a generous multiple of the horizon. Real accepted reservations end shortly after
    # the last landing, but a hand-built fixture with an open-ended (``t_end`` ~ 1e6) volume must not
    # blow up the grid; 4× horizon is far beyond any real ramp-down (a trip is a small fraction of H).
    t_max = min(max(hi for _, hi, _ in contribs), 4.0 * cfg.horizon_s)
    n = int(math.ceil(t_max / dt)) + 1
    delta = np.zeros(n + 1)
    for lo, hi, w in contribs:
        a = min(max(int(math.floor(lo / dt)), 0), n)
        b = min(max(int(math.ceil(hi / dt)), 0), n)
        delta[a] += w
        delta[b] -= w
    return np.arange(n) * dt, np.cumsum(delta)[:n]


def _widest_hot_run(hot: np.ndarray) -> tuple[int, int] | None:
    """Indices ``(i0, i1)`` inclusive of the widest contiguous run of ``True`` in ``hot`` (earliest on
    ties), or ``None`` if there is no ``True`` element."""
    best: tuple[int, int] | None = None
    n = len(hot)
    i = 0
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j + 1
        else:
            i += 1
    return best


def steady_state_window(result: SimResult, frac: float = 0.9, dt: float | None = None,
                        smooth_s: float | None = None) -> tuple[float, float]:
    """The widest contiguous interval whose airborne density ≥ ``frac × peak`` — the representative
    plateau, trimming the ramp-up and ramp-down tails automatically (adapting to whatever λ / horizon /
    trip-length mix the run produced).

    ``smooth_s`` moving-averages the (jagged integer) count density before thresholding — essential so
    the threshold tracks the *plateau* level, not a transient concurrency spike (a raw count density
    spikes well above its plateau, and ``0.9 × peak`` then latches onto a few bins around that spike).
    ``None`` (default) adapts the smoothing width to the median trip duration — the scale of the ramp
    itself; ``0`` disables it (raw density, for controlled inputs). Falls back to ``(0, horizon_s)`` when
    nothing flew or no plateau is detectable, so the window is always safe to feed to :func:`aggregate`
    (there, steady == whole-run)."""
    cfg = result.config
    if not result.accepted:
        return (0.0, cfg.horizon_s)
    dt = cfg.dt_s if dt is None else dt
    t, d = density_timeseries(result, dt)
    if d.size == 0 or float(d.max()) <= 0.0:
        return (0.0, cfg.horizon_s)
    if smooth_s is None:   # adapt to the median airborne span (≈ the trip duration = the ramp width)
        widths = [iv[1] - iv[0] for i in result.accepted if (iv := _airborne_interval(i)) is not None]
        smooth_s = float(np.median(widths)) if widths else 0.0
    if smooth_s and smooth_s > dt:
        k = max(1, int(round(smooth_s / dt)))
        d = np.convolve(d, np.ones(k) / k, mode="same")
    run = _widest_hot_run(d >= frac * float(d.max()))
    if run is None:
        return (0.0, cfg.horizon_s)
    return (float(t[run[0]]), float(t[run[1]]))


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


def _unimpeded_cruise_z(cfg: SimConfig) -> float:
    """The altitude the run's planner cruises at when UNIMPEDED. The A* family deconflicts by altitude on
    the discrete ladder, so its unimpeded cruise is the lowest flight level; the continuous single-plane
    planners (straight / rrt / milp / opt / …) are pinned to ``cruise_level_m`` (no altitude lever).

    Keyed on ``cfg.planner`` — the run's registry name ('astar' / 'opt_astar' / 'astar_milp' all contain
    'astar'; 'straight' / 'opt' / 'milp' don't) — NOT ``intent.planner``, which a refiner relabels to its
    own stage (``opt_astar`` stamps 'opt', ``astar_milp`` stamps 'milp'), dropping the A* origin. So a
    single-plane run reads ZERO excess altitude (its cruise IS its baseline) while a traffic-forced A*
    climb above the floor reads positive excess (real congestion)."""
    return cfg.cruise_level_m if "astar" not in cfg.planner else cfg.flight_levels_m[0]


def total_delay_s(intent: OperationalIntent, cfg: SimConfig) -> float:
    """Unified congestion lateness (s): the seconds a flight loses *to other traffic*, across ALL four
    levers — ground hold + air loiter + detour-time + a traffic-forced vertical climb (excess altitude ÷
    climb rate). Each is excess over the unimpeded flight (a straight dash at the planner's own cruise
    altitude), so the mandatory takeoff/landing climb is excluded but a climb *forced by congestion* is
    NOT (that is exactly the vertical lever this project adds). This is the time-space twin of
    ``congestion_cost``. NaN for denied flights (they never arrive). See :func:`flight_row`.
    """
    if not intent.accepted:
        return float("nan")
    excess_m = max(0.0, intent.altitude_change_m - nominal_altitude_change_m(cfg))
    return (
        intent.ground_delay_s
        + intent.air_hold_s
        + intent.air_detour_m / cfg.nominal_speed_mps
        + excess_m / cfg.climb_rate_mps
    )


def nominal_flight_time_s(straight_m: float, cfg: SimConfig) -> float:
    """Unimpeded door-to-door air time (s): straight cruise + the mandatory climb and descent to the
    run's cruise altitude (:func:`_unimpeded_cruise_z` — the ladder floor for A*, ``cruise_level_m`` for
    single-plane planners), so the time nominal agrees with :func:`nominal_altitude_change_m` and
    ``delay_pct`` is measured against the true unimpeded trip."""
    return straight_m / cfg.nominal_speed_mps + 2.0 * cfg.climb_time_to(_unimpeded_cruise_z(cfg))


def nominal_altitude_change_m(cfg: SimConfig) -> float:
    """Mandatory vertical travel of an unimpeded flight: climb to the flight's own cruise altitude
    (:func:`_unimpeded_cruise_z`) and descend back (climb + descent ⇒ the factor of 2). This is the
    reference ``excess_altitude_m`` measures against, so a single-plane planner (cruise == baseline) reads
    zero excess and only a traffic-forced A* climb above its ladder floor reads positive excess."""
    return 2.0 * (_unimpeded_cruise_z(cfg) - cfg.ground_level_m)


_COST_LEVERS = ("ground_delay_cost", "air_hold_cost", "air_detour_cost", "altitude_cost")


def cost_breakdown(intent: OperationalIntent, cfg: SimConfig) -> dict[str, float]:
    """Exact split of the planner objective ``intent.cost`` into its four levers, in COST units. The
    terms reconcile to :func:`cost.trajectory_cost` (== ``intent.cost`` for an accepted flight).

    Mind the asymmetry the cost model bakes in: ground / hold / detour are charged as pure EXCESS over
    an unimpeded flight (zero baseline), but ``altitude_cost`` is the FULL climb+descent — every flight
    pays its mandatory :func:`nominal_altitude_change_m` whether or not traffic forced a higher cruise.
    The congestion-attributable slice of altitude lives in :func:`delay_breakdown_s` (``excess_*``). NaN
    for a denied flight (it never flew — a real 0 would bias the means)."""
    if not intent.accepted:
        return dict.fromkeys(_COST_LEVERS, float("nan"))
    return {
        "ground_delay_cost": cfg.cost_ground_delay_per_s * intent.ground_delay_s,
        "air_hold_cost": cfg.cost_air_hold_per_s * intent.air_hold_s,
        "air_detour_cost": cfg.cost_air_lateral_per_m * intent.air_detour_m,
        "altitude_cost": cfg.cost_altitude_change_per_m * intent.altitude_change_m,
    }


_DELAY_LEVERS = ("ground_delay_s", "air_hold_s", "detour_time_s",
                 "excess_altitude_m", "altitude_delay_phys_s", "altitude_delay_costeq_s")


def delay_breakdown_s(intent: OperationalIntent, cfg: SimConfig) -> dict[str, float]:
    """Time-space twin of :func:`cost_breakdown`: the real SECONDS each lever costs the flight, plus the
    congestion-driven vertical travel translated into time two honest ways.

    Cost and time diverge wherever a weight isn't 1 s/unit — an air-hold second *costs* ``c_air_hold``
    (3×) but *is* one real second; a detour metre costs ``c_lat`` but is ``1/speed`` s. Altitude has two
    readings of "what a climbed metre is worth in seconds", and we record BOTH:

      * ``altitude_delay_phys_s`` — physical: ``excess_m / climb_rate`` (the extra airborne seconds).
      * ``altitude_delay_costeq_s`` — cost-equivalent: ``excess_m · c_alt / c_ground`` (the ground-delay
        seconds the OPTIMISER treats the climb as worth — same currency as the other ``*_cost`` levers
        divided back into time).

    The two differ by ``c_alt · climb_rate / c_ground`` (12× at defaults) — that gap *is* the cost-vs-time
    story this surface exists to tell. ``excess_m`` is altitude above :func:`nominal_altitude_change_m`
    (the traffic-forced climb, measured against the flight's own planner baseline). NaN for a denied flight."""
    if not intent.accepted:
        return dict.fromkeys(_DELAY_LEVERS, float("nan"))
    excess_m = max(0.0, intent.altitude_change_m - nominal_altitude_change_m(cfg))
    return {
        "ground_delay_s": intent.ground_delay_s,
        "air_hold_s": intent.air_hold_s,
        "detour_time_s": intent.air_detour_m / cfg.nominal_speed_mps,
        "excess_altitude_m": excess_m,
        "altitude_delay_phys_s": excess_m / cfg.climb_rate_mps,
        "altitude_delay_costeq_s": (excess_m * cfg.cost_altitude_change_per_m
                                    / cfg.cost_ground_delay_per_s),
    }


def flight_row(intent: OperationalIntent, cfg: SimConfig,
               window: tuple[float, float] | None = None) -> dict:
    """One tidy record for a single operational intent (accepted or denied).

    ``window=(t_lo, t_hi)`` clamps this row's reserved volume-seconds to the measurement window
    (default ``[0, horizon_s]``); it does not otherwise change the row (membership filtering by filing
    time is :func:`flight_frame`'s job)."""
    res_lo, res_hi = (0.0, cfg.horizon_s) if window is None else window
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
    # the two parallel decompositions: COST (what the planner paid, reconciles to `cost`) and TIME (real
    # seconds, with altitude read both physically and as a cost-equivalent). See the module docstring.
    cb = cost_breakdown(intent, cfg)
    db = delay_breakdown_s(intent, cfg)
    # cost-space twin of total_delay_s: the four congestion levers above an unimpeded straight flight at
    # the planner's own cruise altitude — ground/hold/detour (already excess) + altitude's EXCESS only
    # (the mandatory climb is not congestion). Same four levers total_delay_s sums in TIME; differs from
    # `cost`, which also carries the mandatory baseline-altitude cost.
    congestion_cost = ((cb["ground_delay_cost"] + cb["air_hold_cost"] + cb["air_detour_cost"]
                        + cfg.cost_altitude_change_per_m * db["excess_altitude_m"])
                       if intent.accepted else float("nan"))
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
        # detour as lateness-seconds; ground_delay_s + air_hold_s + detour_time_s + altitude_delay_phys_s
        # == total_delay_s (the four time-space congestion levers). Reuse db so the formula lives once.
        "detour_time_s": db["detour_time_s"],
        "altitude_change_m": intent.altitude_change_m,
        # congestion-driven vertical travel (above the flight's own cruise baseline) + its two time readings
        "excess_altitude_m": db["excess_altitude_m"],
        "altitude_delay_phys_s": db["altitude_delay_phys_s"],      # physical: excess_m / climb_rate
        "altitude_delay_costeq_s": db["altitude_delay_costeq_s"],  # cost-equivalent: excess_m·c_alt/c_gd
        "total_delay_s": td,                      # unified congestion lateness (s): all four levers
        "delay_pct": delay_pct,                    # ... as % of the flight's total trip time
        "trip_time_ratio": trip_time_ratio,        # (straight-line time + delay) / straight-line time
        # per-lever COST split (units of the planner objective); the four reconcile to `cost`
        **cb,
        "congestion_cost": congestion_cost,        # cost-space twin of total_delay_s (excl. mandatory climb)
        "cost": intent.cost,
        "solve_time_s": intent.solve_time_s,   # planner wall time for this flight
        "straight_line_m": straight,
        "flown_m": flown,
        "stretch": stretch,
        "reserved_vol_m3_s": reserved_volume_seconds(intent.volumes, res_lo, res_hi),
    }


def flight_frame(result: SimResult, window: tuple[float, float] | None = None) -> pd.DataFrame:
    """Per-flight metrics table — one row per intent, FCFS order preserved.

    ``window=(t_lo, t_hi)`` restricts the table to flights *filed* in ``[t_lo, t_hi)`` (filing-time
    membership — a flight's delay is fixed at entry, and this drops the ramp tails, incl. return flights
    filed past the horizon) and clamps each row's reserved volume-seconds to the window. ``None``
    (default) is the whole run: every intent, volume clamped to ``[0, horizon_s]`` — identical to the
    persisted ``flights.parquet``."""
    df = pd.DataFrame([flight_row(i, result.config, window) for i in result.intents])
    if window is not None and len(df):
        lo, hi = window
        df = df[(df["t_request"] >= lo) & (df["t_request"] < hi)].reset_index(drop=True)
    return df


def _q(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if len(series) else 0.0


def _mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def _rollup(df: pd.DataFrame, cfg: SimConfig, dur_s: float | None = None) -> dict:
    """Group-level rollup of a (sub)frame of flight rows — shared by ``aggregate`` (the whole run)
    and ``per_uss_frame`` (one operator's slice). Denominators (horizon, region capacity) are the
    *run's*, so a per-USS ``airspace_utilization`` reads as that operator's share of the whole sky.

    ``dur_s`` (the measurement window's duration) overrides the horizon in the rate and capacity
    denominators, so a windowed rollup divides throughput/offered-load by the window — not the full
    horizon — and its ``airspace_utilization`` is volume-seconds ÷ region × window (see :func:`aggregate`).
    """
    acc = df[df["accepted"]]
    den = df[df["denied"]]
    dur_s = cfg.horizon_s if dur_s is None else dur_s
    horizon_h = dur_s / 3600.0
    # Vertical extent of the usable airspace. With discrete flight levels (n_levels > 1) the usable
    # tube is the regulated band [ground, airspace_ceiling]. Otherwise the continuous band is collapsed
    # to a single plane (z_max == z_min), so fall back to the corridor slab height — the vertical
    # footprint a flight occupies at that level — so utilization stays meaningful (not a divide-by-zero).
    vert_extent_m = ((cfg.airspace_ceiling_m - cfg.ground_level_m) if cfg.n_levels > 1
                     else max(cfg.z_max_m - cfg.z_min_m, cfg.corridor_height_m))
    region_vol_m3 = cfg.region_size_m[0] * cfg.region_size_m[1] * vert_extent_m
    airspace_capacity_m3_s = region_vol_m3 * dur_s
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
        # COST decomposition by lever (planner-objective units) — "where did the congestion cost go?".
        # The four reconcile to mean_cost EXACTLY (altitude_cost is the full climb+descent); it is
        # mean_congestion_cost that equals mean_cost minus the mandatory baseline-altitude cost.
        "mean_ground_delay_cost": _mean(acc["ground_delay_cost"]),
        "mean_air_hold_cost": _mean(acc["air_hold_cost"]),
        "mean_air_detour_cost": _mean(acc["air_detour_cost"]),
        "mean_altitude_cost": _mean(acc["altitude_cost"]),
        "mean_congestion_cost": _mean(acc["congestion_cost"]),   # cost-space twin of mean_total_delay_s
        # vertical deconfliction: traffic-forced climb above the floor, in metres + both time readings
        "mean_excess_altitude_m": _mean(acc["excess_altitude_m"]),
        "p95_excess_altitude_m": _q(acc["excess_altitude_m"], 0.95),
        "mean_altitude_delay_phys_s": _mean(acc["altitude_delay_phys_s"]),     # physical seconds
        "mean_altitude_delay_costeq_s": _mean(acc["altitude_delay_costeq_s"]),  # cost-equivalent seconds
        # planner runtime over ALL flights (denials included — they often exhaust the search)
        "mean_solve_time_s": float(df["solve_time_s"].mean()) if len(df) else 0.0,
        "p95_solve_time_s": _q(df["solve_time_s"], 0.95),
        "max_solve_time_s": float(df["solve_time_s"].max()) if len(df) else 0.0,
        "total_solve_time_s": float(df["solve_time_s"].sum()),
        "reserved_vol_m3_s": float(df["reserved_vol_m3_s"].sum()),
        "airspace_utilization": float(df["reserved_vol_m3_s"].sum()) / max(airspace_capacity_m3_s, 1e-9),
    }


def _per_uss_table(df: pd.DataFrame, cfg: SimConfig, dur_s: float | None = None) -> pd.DataFrame:
    total_acc = int(df["accepted"].sum()) if len(df) else 0
    rows = []
    for uss_id, g in df.groupby("uss_id", sort=True):
        acc = g[g["accepted"]]
        rows.append({
            "uss_id": uss_id,
            **_rollup(g, cfg, dur_s=dur_s),
            # per-USS-only: flight length (confirms hub-demand shortening) + share of the throughput
            "mean_straight_line_m": float(acc["straight_line_m"].mean()) if len(acc) else 0.0,
            "share_of_accepted": (len(acc) / total_acc) if total_acc else 0.0,
        })
    return pd.DataFrame(rows)


def per_uss_frame(result: SimResult, window: tuple[float, float] | None = None) -> pd.DataFrame:
    """One metrics row per USS — the per-operator slice of a (multi-)USS run. Each row's counts and
    reserved volume sum to the overall ``aggregate`` totals (see tests). ``window`` restricts to flights
    filed in ``[t_lo, t_hi)`` and uses the window duration for the rate/capacity denominators."""
    cfg = result.config
    lo, hi = (0.0, cfg.horizon_s) if window is None else window
    return _per_uss_table(flight_frame(result, window), cfg, dur_s=hi - lo)


def aggregate(result: SimResult, window: tuple[float, float] | None = None) -> dict:
    """Flat headline rollup for one run — the row a λ-sweep collects.

    ``window=(t_lo, t_hi)`` measures only flights filed in that interval, with rate/capacity
    denominators using the window duration and ``window_lo``/``window_hi`` added for provenance. ``None``
    (default) is the whole run — every field identical to before this option existed. Use
    :func:`aggregate_with_steady` to report the whole-run numbers next to their steady-state twin."""
    cfg = result.config
    lo, hi = (0.0, cfg.horizon_s) if window is None else window
    dur_s = hi - lo
    df = flight_frame(result, window)
    den = df[df["denied"]] if len(df) else df
    by_reason = den["denial_reason"].value_counts().to_dict() if len(den) else {}

    # cross-USS fairness: does one operator systematically lose under FCFS? (0 when single-USS)
    per_uss = _per_uss_table(df, cfg, dur_s=dur_s)
    n_uss = int(len(per_uss))
    if n_uss > 1:
        denial_rate_spread = float(per_uss["denial_rate"].max() - per_uss["denial_rate"].min())
        mean_delay_spread = float(per_uss["mean_total_delay_s"].max() - per_uss["mean_total_delay_s"].min())
    else:
        denial_rate_spread = mean_delay_spread = 0.0

    out = {
        "lam_per_hour": cfg.lam_per_hour,
        "seed": cfg.seed,
        "planner": cfg.planner,
        **_rollup(df, cfg, dur_s=dur_s),
        "denials_by_reason": by_reason,
        "n_uss": n_uss,
        "denial_rate_spread": denial_rate_spread,
        "mean_delay_spread": mean_delay_spread,
        "verified": result.verified,
    }
    if window is not None:
        out["window_lo"], out["window_hi"] = float(lo), float(hi)
    return out


def aggregate_with_steady(result: SimResult, frac: float = 0.9, smooth_s: float | None = None,
                          dt: float | None = None) -> dict:
    """The whole-run :func:`aggregate` **plus** a nested ``"steady_state"`` block holding the same
    rollup measured over :func:`steady_state_window` (the representative density plateau) — the two
    views reported side by side (issue #25). The block carries ``window_lo``/``window_hi`` so a windowed
    number is self-describing; it drops the run-identity keys (lam/seed/planner/n_uss/verified) the two
    views share. When no plateau is detectable (small / low-λ runs) the window is the whole horizon, so
    the steady block simply equals the whole-run numbers.

    ``smooth_s`` is forwarded to :func:`steady_state_window` (``None`` → adapt the smoothing width to
    the median trip duration, so the window tracks the plateau, not a transient concurrency spike)."""
    win = steady_state_window(result, frac=frac, dt=dt, smooth_s=smooth_s)
    out = aggregate(result)
    steady = aggregate(result, window=win)
    for k in ("lam_per_hour", "seed", "planner", "n_uss", "verified"):
        steady.pop(k, None)
    out["steady_state"] = steady
    return out
