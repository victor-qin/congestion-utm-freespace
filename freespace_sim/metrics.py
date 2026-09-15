"""Metrics — turn a `SimResult` into per-flight rows and aggregate rollups.

Two surfaces:

- ``flight_frame(result)`` → a tidy pandas DataFrame, one row per `OperationalIntent`, carrying
  each flight's congestion in BOTH currencies: the planner's **cost** split per lever (the
  ``*_cost`` columns, which reconcile exactly to ``cost`` == :func:`cost.trajectory_cost`) and the
  **real seconds** each lever costs (``ground_delay_s`` / ``air_hold_s`` / ``detour_time_s`` plus
  altitude's two readings, physical and cost-equivalent). The two diverge wherever a cost weight
  isn't 1 s/unit (a hover-second costs 3×; a detour-metre ~1/30 s), so you can read what the
  optimiser *paid* and how that pay becomes time. Plus efficiency (stretch = flown ÷ straight) and
  reserved **volume-seconds**. See :func:`cost_breakdown` / :func:`delay_breakdown_s`.
- ``aggregate(result)`` → a flat dict of headline numbers for the λ-sweep: acceptance/denial,
  delay & detour distributions, throughput, and **airspace utilization** (reserved volume-seconds ÷
  the whole region × realized duration) — the free-space analog of the sibling project's
  hex-occupancy. A ``window`` measures only a slice of the run; :func:`aggregate_with_steady`
  reports the whole-run numbers beside their **steady-state** twin (metrics over the representative
  density plateau, ramp-up/ramp-down tails dropped).

The congestion story the experiment tells is *offered load* (requests/hour) vs these outcomes: as λ
rises, the FCFS newcomer is pushed into ever costlier delays/detours until the budget can't absorb
it and denials climb. Keeping `BUDGET_EXCEEDED` denials (real congestion) separate from
`SEARCH_EXHAUSTED` (a planner artifact) keeps that signal honest — see `DenialReason`.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from .config import SimConfig
from .geometry import BoxSpec, CylinderSpec
from .planner import uses_hex_lattice
from .sim import SimResult
from .types import DenialReason, OperationalIntent
from .volumes import Volume4D, enroute_flown_m, enroute_reference_m


# Bump whenever a persisted metric changes MEANING (not when one is added). `runs._index_row` stamps
# it on every row, so `results/index.parquet` (appended to across versions of this file) can never
# silently average numbers computed under incomparable definitions.
METRICS_VERSION = 3

# The realized run may legitimately exceed horizon_s (post-horizon return tail) but not by orders
# of magnitude — the density family's tail runs ~1.3x its envelope. Generous on purpose: this only
# has to catch a sentinel timestamp (1e12) that leaked into an accepted reservation.
_MAX_PLAUSIBLE_RUN_FACTOR = 100.0


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

    Each volume's window is clamped to ``[t_lo, t_hi]`` so a windowed view measures only the
    volume-seconds a flight spends inside it. Full-run callers pass :func:`simulation_window`, which
    spans every accepted reservation without clipping.

    Parameters
    ------------
    - volumes (list[Volume4D] | None): the flight's reservation; ``None`` or empty ⇒ 0.0.
    - t_lo (float): window start; each volume's ``t_start`` is clamped up to it.
    - t_hi (float): window end; each volume's ``t_end`` is clamped down to it.

    Return
    --------
    - output (float): reserved volume-seconds (m³·s), over volumes with positive clamped duration.
    """
    if not volumes:
        return 0.0
    total = 0.0
    for v in volumes:
        dur = min(v.t_end, t_hi) - max(v.t_start, t_lo)
        if dur > 0.0:
            total += shape_volume_m3(v.shape) * dur
    return total


# --- steady-state measurement window ---
# A run's airborne density is a trapezoid: it ramps up as the sky fills from empty, plateaus, then
# ramps down as the last flights land (and, with return traffic, past the horizon). Metrics over the
# whole run are diluted by the low-density ramps — a flight filed at t≈0 waits on almost-empty
# airspace, one near the horizon flies into a thinning tail. These helpers find the plateau so
# delay / throughput / denial are measured where the density is representative: run the natural
# demand (tails and all), but *measure* only that window.


def _airborne_interval(intent: OperationalIntent) -> tuple[float, float] | None:
    """The [takeoff, land] span over which a flight occupies airspace: its reservation's time
    envelope (earliest ``t_start`` → latest ``t_end``), or the centerline span if volumes are
    absent. ``None`` when neither is present (a denied flight)."""
    if intent.volumes:
        return (min(v.t_start for v in intent.volumes), max(v.t_end for v in intent.volumes))
    if intent.centerline:
        return (intent.centerline[0][1], intent.centerline[-1][1])
    return None


def occupancy_time_s(intent: OperationalIntent) -> float:
    """When this flight took (or wanted to take) the airspace — the clock windows are measured on.

    NOT ``t_request``: filing time and airborne time are the same clock only when flights depart on
    filing. Under ``timing_mode="departure"`` they are separated by the whole scheduling lead, so
    selecting a cohort by ``t_request`` against a window derived from airborne density can miss it
    entirely.

    Parameters
    ------------
    - intent (OperationalIntent): the flight; accepted or denied.

    Return
    --------
    - output (float): accepted ⇒ the start of its :func:`_airborne_interval` (actual takeoff);
      denied ⇒ its desired ``t_departure`` (it never flew, but is the traffic the window
      characterises, so dropping denials would make ``denial_rate`` structurally 0 in every window).
      Known bias: an accepted takeoff is pushed later by ground delay, a denied stand-in isn't,
      so denials skew tens of seconds early against windows hundreds of seconds wide — the honest
      cost of the only clock both cohorts share.
    """
    iv = _airborne_interval(intent)
    return float(iv[0]) if iv is not None else float(intent.request.t_departure)


def simulation_window(result: SimResult) -> tuple[float, float]:
    """Realized run bounds: first accepted flight activity through the final landing reservation.

    Flight reservations include takeoff and landing columns, so their earliest/latest timestamps are
    the operational bounds the replay and full-run utilization need. A centerline is the fallback
    for synthetic/legacy intents without volumes.

    Parameters
    ------------
    - result (SimResult): the run; only its accepted intents are scanned.

    Return
    --------
    - output (tuple[float, float]): ``(start, end)`` over every accepted reservation, or ``(0.0,
      0.0)`` when nothing was accepted.
    """
    intervals = [
        interval
        for intent in result.accepted
        if (interval := _airborne_interval(intent)) is not None
    ]
    if not intervals:
        return (0.0, 0.0)
    return (
        float(min(lo for lo, _ in intervals)),
        float(max(hi for _, hi in intervals)),
    )


def density_timeseries(result: SimResult, dt: float | None = None, kind: str = "count",
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Airborne density over time on a uniform grid spanning every accepted flight.

    Built with a difference array, so it is O(flights + grid).

    Parameters
    ------------
    - result (SimResult): the run; its accepted intents supply the intervals/volumes.
    - dt (float | None): grid step; ``None`` ⇒ ``cfg.dt_s``. Coarsened up if the span would need too
      many bins (see the body), never truncated.
    - kind (str): ``"count"`` ⇒ concurrent airborne flights (+1 per :func:`_airborne_interval`);
      ``"volume"`` ⇒ active reserved spatial volume in m³ (each ``Volume4D``'s ``shape_volume_m3``
      over ``[t_start, t_end)``) — smoother, the instantaneous-rate analog of
      :func:`reserved_volume_seconds`. Any other value raises ``ValueError``.

    Return
    --------
    - output (tuple[np.ndarray, np.ndarray]): ``(t_grid, density)``, or ``(array([0.]),
      array([0.]))`` when nothing flew.
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
    t_min = min(lo for lo, _, _ in contribs)
    t_max = max(hi for _, hi, _ in contribs)
    span_s = max(0.0, t_max - t_min)
    # Never truncate the time range. For pathological synthetic open-ended reservations, coarsen the
    # grid instead of allocating millions of bins; real runs retain cfg.dt_s resolution.
    max_bins = 250_000
    grid_dt = max(dt, span_s / max_bins) if span_s > 0.0 else dt
    n = int(math.ceil(span_s / grid_dt)) + 1
    delta = np.zeros(n + 1)
    for lo, hi, w in contribs:
        a = min(max(int(math.floor((lo - t_min) / grid_dt)), 0), n)
        b = min(max(int(math.ceil((hi - t_min) / grid_dt)), 0), n)
        delta[a] += w
        delta[b] -= w
    return t_min + np.arange(n) * grid_dt, np.cumsum(delta)[:n]


def _widest_hot_run(hot: np.ndarray) -> tuple[int, int] | None:
    """Indices ``(i0, i1)`` (inclusive) of the widest contiguous run of ``True`` in ``hot``
    (earliest on ties), or ``None`` if there is no ``True`` element."""
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
    plateau, trimming the ramp-up/ramp-down tails automatically (adapting to whatever λ / horizon /
    trip-length mix the run produced). Falls back to :func:`simulation_window` when no plateau is
    detectable.

    Parameters
    ------------
    - result (SimResult): the run; empty/degenerate accepted sets fall back to the realized window.
    - frac (float): plateau threshold as a fraction of peak density (0.9 ⇒ ``≥ 0.9 × peak``).
    - dt (float | None): grid step for :func:`density_timeseries`; ``None`` ⇒ ``cfg.dt_s``.
    - smooth_s (float | None): moving-average width applied to the (jagged integer) count density
      before thresholding — essential so the threshold tracks the *plateau*, not a transient spike
      (a raw count spikes above its plateau, and ``frac × peak`` then latches onto a few bins around
      it). ``None`` (default) adapts it to the median trip duration (the scale of the ramp); ``0``
      disables it (raw density, for controlled inputs).

    Return
    --------
    - output (tuple[float, float]): ``(start, end)`` of the plateau, or the realized window.
    """
    cfg = result.config
    # simulation_window walks every accepted intent's volumes (millions on a density run), so pay
    # for it only on the fallback paths that return it — not before the plateau search that usually
    # discards it.
    if not result.accepted:
        return simulation_window(result)
    dt = cfg.dt_s if dt is None else dt
    t, d = density_timeseries(result, dt)
    if d.size == 0 or float(d.max()) <= 0.0:
        return simulation_window(result)
    if smooth_s is None:   # adapt to the median airborne span (≈ the trip duration = the ramp width)
        widths = [iv[1] - iv[0] for i in result.accepted if (iv := _airborne_interval(i)) is not None]
        smooth_s = float(np.median(widths)) if widths else 0.0
    # Measure the kernel against the grid we actually got back, not the one we asked for:
    # density_timeseries coarsens to max(dt, span/250k), so on a coarsened grid `smooth_s / dt`
    # over-widens the kernel by grid_dt/dt and mislocates the plateau.
    grid_dt = float(t[1] - t[0]) if t.size > 1 else dt
    if smooth_s and smooth_s > grid_dt:
        k = max(1, int(round(smooth_s / grid_dt)))
        d = np.convolve(d, np.ones(k) / k, mode="same")
    run = _widest_hot_run(d >= frac * float(d.max()))
    if run is None:
        return simulation_window(result)
    return (float(t[run[0]]), float(t[run[1]]))


def _flown_horizontal_m(intent: OperationalIntent, cfg: SimConfig) -> float:
    """Horizontal length actually flown EN ROUTE — the centerline, rooted at both column edges.

    Run through the same :func:`volumes.fold_corners_to_columns` every reservation uses, so A*
    (whose path already starts on a boundary lane cell) and the continuous planners (whose warm
    candidates may still run centre → centre) are measured on one ruler. Without it a planner that
    flies through its own column is charged for terminal airspace the lane → lane baseline excludes,
    and its stretch reads high for no en-route reason. Pairs with :func:`_straight_horizontal_m`:
    both span exit lane → exit lane, so their difference is en-route detour and nothing else.

    Parameters
    ------------
    - intent (OperationalIntent): the flight; its ``centerline`` and O-D/terminal request fields.
    - cfg (SimConfig): geometry for the terminal fold and reference.

    Return
    --------
    - output (float): en-route flown horizontal length (m), or ``nan`` when there is no centerline.
    """
    if not intent.centerline:
        return float("nan")
    return enroute_flown_m([p[0] for p in intent.centerline],
                           intent.request.origin, intent.request.dest,
                           intent.request.origin_terminal, intent.request.dest_terminal, cfg)


def _straight_horizontal_m(intent: OperationalIntent, cfg: SimConfig) -> float:
    """The straight-line reference: exit lane → exit lane (:func:`volumes.enroute_reference_m`)."""
    return enroute_reference_m(intent.request.origin, intent.request.dest,
                               intent.request.origin_terminal, intent.request.dest_terminal, cfg)


def _unimpeded_cruise_z(cfg: SimConfig) -> float:
    """The altitude the run's planner cruises at when UNIMPEDED.

    Hex-lattice families deconflict by altitude on the discrete ladder, so their unimpeded cruise is
    the lowest flight level; the MILP cruises the continuous band ``[z_min_m, z_max_m]``, so its
    unimpeded cruise is the band floor; only the truly single-plane planners (straight / decoupled)
    are pinned to ``cruise_level_m`` (no altitude lever).

    Keyed on ``cfg.planner`` (the run's registry name), NOT ``intent.planner``, which a refiner
    relabels to its own stage (``astar_milp`` stamps 'milp'), dropping the A* origin — so a
    single-plane run reads ZERO excess altitude (its cruise IS its baseline) while a traffic-forced
    climb above the floor reads positive excess (real congestion).

    Parameters
    ------------
    - cfg (SimConfig): read for ``planner`` and the derived flight levels / z-band.

    Return
    --------
    - output (float): the unimpeded cruise altitude (m) for this run's planner.
    """
    if uses_hex_lattice(cfg.planner):
        return cfg.flight_levels_m[0]
    if "milp" in cfg.planner:
        return cfg.z_min_m           # MILP band floor (astar_milp takes the astar branch — same value)
    return cfg.cruise_level_m        # straight / decoupled: single-plane


def total_delay_s(intent: OperationalIntent, cfg: SimConfig) -> float:
    """Unified congestion lateness (s): the seconds a flight loses *to other traffic*, across ALL
    four levers — ground hold + air loiter + detour-time + a traffic-forced vertical climb (excess
    altitude ÷ climb rate). Each is excess over the unimpeded flight (a straight dash at the
    planner's own cruise altitude), so the mandatory takeoff/landing climb is excluded but a climb
    *forced by congestion* is NOT (exactly the vertical lever this project adds). Time-space twin of
    ``congestion_cost``.

    Caveat for the A* family: the detour term is built from ``air_detour_m``, measured against the
    Euclidean straight line — unreachable on a 6-direction lattice — so for A* this figure also
    carries the hex-quantization share, and *that* part is not lost "to other traffic";
    ``flight_row``'s ``lattice_overhead_m`` / ``deconfliction_detour_m`` split it out.

    Hub flights measure the EN-ROUTE segment only: ``air_detour_m`` spans exit lane → exit lane on
    both sides, so the unreserved hub-column legs are in neither the flown length nor the baseline.
    Flying inside a terminal is terminal operations (it consumes that hub's column + pad gate), not
    en-route distance. Ground delay IS counted — waiting on the pad is time lost to whoever owns the
    airspace.

    Parameters
    ------------
    - intent (OperationalIntent): the flight; delay/hold/detour/altitude fields and ``accepted``.
    - cfg (SimConfig): cost/kinematics used to convert metres to seconds.

    Return
    --------
    - output (float): total congestion lateness (s); ``nan`` for a denied flight (never arrives).
    """
    if not intent.accepted:
        return float("nan")
    excess_m = max(0.0, intent.altitude_change_m - nominal_altitude_change_m(cfg))
    lattice_s, traffic_s = _detour_seconds(intent, cfg)
    return (
        intent.ground_delay_s
        + intent.air_hold_s
        + (lattice_s + traffic_s)
        + excess_m / cfg.climb_rate_mps
    )


def _detour_seconds(intent: OperationalIntent, cfg: SimConfig) -> tuple[float, float]:
    """The exact two-way split of ``air_detour_m`` in seconds; the detour time is their sum, by
    construction rather than as a third returned value.

    A straight split of ``air_detour_m`` — no terminal fold to net out any more, because
    ``air_detour_m`` is measured lane → lane on both sides. For A* ``detour_traffic_s`` is
    a real measurement (whole hex steps beyond the lattice geodesic) and nothing adjusts it.

    Parameters
    ------------
    - intent (OperationalIntent): the flight; ``air_detour_m`` and ``lattice_overhead_m``.
    - cfg (SimConfig): ``nominal_speed_mps`` converts the split metres to seconds.

    Return
    --------
    - output (tuple[float, float]): ``(lattice_s, traffic_s)``; ``lattice_s`` is the quantization
      share (clamped never to exceed ``air_detour_m``), ``traffic_s`` the remainder.
    """
    lattice_m = min(intent.lattice_overhead_m, intent.air_detour_m)   # never exceed what it splits
    lattice_s = lattice_m / cfg.nominal_speed_mps
    traffic_s = (intent.air_detour_m - lattice_m) / cfg.nominal_speed_mps
    return lattice_s, traffic_s


def nominal_flight_time_s(straight_m: float, cfg: SimConfig) -> float:
    """Unimpeded EN-ROUTE air time (s): straight cruise over ``straight_m`` + mandatory climb and
    descent to the run's cruise altitude (:func:`_unimpeded_cruise_z` — the ladder floor for A*,
    ``cruise_level_m`` for single-plane planners), so this nominal agrees with
    :func:`nominal_altitude_change_m`.

    NOT door-to-door: the caller passes the lane → lane reference, so time flying inside a terminal
    column is in neither this nominal nor ``total_delay_s`` — ``delay_pct`` and ``trip_time_ratio``
    are en-route ratios, consistent with every other en-route metric.

    Parameters
    ------------
    - straight_m (float): the lane → lane straight-line reference length (m).
    - cfg (SimConfig): cruise speed and the run's climb time to :func:`_unimpeded_cruise_z`.

    Return
    --------
    - output (float): unimpeded en-route air time (s) — cruise plus the mandatory climb and descent.
    """
    return straight_m / cfg.nominal_speed_mps + 2.0 * cfg.climb_time_to(_unimpeded_cruise_z(cfg))


def nominal_altitude_change_m(cfg: SimConfig) -> float:
    """Mandatory vertical travel of an unimpeded flight: climb to the flight's own cruise altitude
    (:func:`_unimpeded_cruise_z`) and descend back (climb + descent ⇒ the factor of 2).

    The reference ``excess_altitude_m`` measures against, so a single-plane planner (cruise ==
    baseline) reads zero excess and only a traffic-forced A* climb above its ladder floor reads
    positive excess.

    Parameters
    ------------
    - cfg (SimConfig): read for ground level and the run's :func:`_unimpeded_cruise_z`.

    Return
    --------
    - output (float): mandatory climb + descent travel (m) for an unimpeded flight.
    """
    return 2.0 * (_unimpeded_cruise_z(cfg) - cfg.ground_level_m)


_COST_LEVERS = ("ground_delay_cost", "air_hold_cost", "air_detour_cost", "altitude_cost")


def cost_breakdown(intent: OperationalIntent, cfg: SimConfig) -> dict[str, float]:
    """Exact split of the planner objective ``intent.cost`` into its four levers, in COST units.

    The terms reconcile to :func:`cost.trajectory_cost` (== ``intent.cost`` for an accepted flight).
    Mind the asymmetry the cost model bakes in: ground/hold/detour are charged as pure EXCESS over
    an unimpeded flight (zero baseline), but ``altitude_cost`` is the FULL climb+descent — every
    flight pays its mandatory :func:`nominal_altitude_change_m` whether or not traffic forced a
    higher cruise. The congestion-attributable slice of altitude lives in :func:`delay_breakdown_s`
    (``excess_*``).

    Parameters
    ------------
    - intent (OperationalIntent): the flight; delay/hold/detour/altitude fields and ``accepted``.
    - cfg (SimConfig): the per-second / per-metre cost weights.

    Return
    --------
    - output (dict[str, float]): ``ground_delay_cost`` / ``air_hold_cost`` / ``air_detour_cost``
      / ``altitude_cost``; all ``nan`` for a denied flight (a real 0 would bias the means).
    """
    if not intent.accepted:
        return dict.fromkeys(_COST_LEVERS, float("nan"))
    return {
        "ground_delay_cost": cfg.cost_ground_delay_per_s * intent.ground_delay_s,
        "air_hold_cost": cfg.cost_air_hold_per_s * intent.air_hold_s,
        "air_detour_cost": cfg.cost_air_lateral_per_m * intent.air_detour_m,
        "altitude_cost": cfg.cost_altitude_change_per_m * intent.altitude_change_m,
    }


_DELAY_LEVERS = ("ground_delay_s", "air_hold_s", "detour_time_s",
                 "detour_lattice_s", "detour_traffic_s",
                 "excess_altitude_m", "altitude_delay_phys_s", "altitude_delay_costeq_s")


def delay_breakdown_s(intent: OperationalIntent, cfg: SimConfig) -> dict[str, float]:
    """Time-space twin of :func:`cost_breakdown`: the real SECONDS each lever costs the flight, plus
    the congestion-driven vertical travel translated into time two honest ways.

    Cost and time diverge where a weight isn't 1 s/unit — an air-hold second *costs* ``c_air_hold``
    (3×) but *is* one real second; a detour metre costs ``c_lat`` but is ``1/speed`` s. Altitude has
    two readings of "what a climbed metre is worth in seconds", and we record BOTH: physical
    (``excess_m / climb_rate``, extra airborne seconds) and cost-equivalent (``excess_m · c_alt /
    c_ground``, the ground-delay seconds the OPTIMISER treats the climb as worth). The two differ by
    ``c_alt · climb_rate / c_ground`` (12× at defaults) — that gap *is* the cost-vs-time story this
    surface exists to tell. ``excess_m`` is altitude above :func:`nominal_altitude_change_m` (the
    traffic-forced climb, against the flight's own planner baseline).

    ``detour_time_s`` splits further into ``detour_lattice_s`` + ``detour_traffic_s`` (they always
    re-sum to it), so a chart can separate hex quantization from a traffic-forced berth;
    ``detour_lattice_s`` is 0 for the continuous planners.

    Parameters
    ------------
    - intent (OperationalIntent): the flight; delay/hold/detour/altitude fields and ``accepted``.
    - cfg (SimConfig): kinematics and cost weights used for the time conversions.

    Return
    --------
    - output (dict[str, float]): the ``_DELAY_LEVERS`` keys (ground/hold/detour split + excess
      altitude and its two time readings); all ``nan`` for a denied flight.
    """
    if not intent.accepted:
        return dict.fromkeys(_DELAY_LEVERS, float("nan"))
    excess_m = max(0.0, intent.altitude_change_m - nominal_altitude_change_m(cfg))
    lattice_s, traffic_s = _detour_seconds(intent, cfg)
    return {
        "ground_delay_s": intent.ground_delay_s,
        "air_hold_s": intent.air_hold_s,
        "detour_time_s": lattice_s + traffic_s,
        "detour_lattice_s": lattice_s,
        "detour_traffic_s": traffic_s,
        "excess_altitude_m": excess_m,
        "altitude_delay_phys_s": excess_m / cfg.climb_rate_mps,
        "altitude_delay_costeq_s": (excess_m * cfg.cost_altitude_change_per_m
                                    / cfg.cost_ground_delay_per_s),
    }


def flight_row(intent: OperationalIntent, cfg: SimConfig,
               window: tuple[float, float] | None = None) -> dict:
    """One tidy record for a single operational intent (accepted or denied).

    ``window=None`` means *unclamped*, NOT ``[0, horizon_s]``: a single intent has no run bounds,
    so the honest default is to invent none. (:func:`flight_frame` passes :func:`simulation_window`
    for the whole-run view; membership filtering is its job, not this row's.)

    Parameters
    ------------
    - intent (OperationalIntent): the flight to describe.
    - cfg (SimConfig): geometry / cost / kinematics the derived columns need.
    - window (tuple[float, float] | None): clamps only this row's reserved volume-seconds to
      ``[t_lo, t_hi]``; ``None`` ⇒ unclamped. Nothing else in the row depends on it.

    Return
    --------
    - output (dict): one column per metric — ids, status, the per-lever cost and time splits,
      stretch, reserved volume-seconds, and ``t_occupancy``. Congestion columns are ``nan`` when
      denied.
    """
    res_lo, res_hi = (-math.inf, math.inf) if window is None else window
    straight = _straight_horizontal_m(intent, cfg)
    flown = _flown_horizontal_m(intent, cfg)
    stretch = (flown / straight) if (intent.accepted and straight > 1e-9) else float("nan")
    td = total_delay_s(intent, cfg)
    db = delay_breakdown_s(intent, cfg)
    # delay as a fraction of the actual trip time — bounded [0, 100), comparable across trip lengths
    nominal = nominal_flight_time_s(straight, cfg)
    delay_pct = (100.0 * td / (nominal + td)) if (intent.accepted and nominal + td > 0) else float("nan")
    # trip-time inflation: trip time (straight-line time + all delay) ÷ ideal straight-line time.
    # ≥ 1, UNBOUNDED — 1.0 = flew the ideal with no wait, 2.0 = twice as long. The unbounded
    # complement of delay_pct: trip_time_ratio == 100 / (100 - delay_pct).
    trip_time_ratio = ((nominal + td) / nominal) if (intent.accepted and nominal > 1e-9) else float("nan")
    # the two parallel decompositions: COST (what the planner paid, reconciles to `cost`) and TIME
    # (real seconds, altitude read physically and cost-equivalent). See the module docstring.
    cb = cost_breakdown(intent, cfg)
    # air_detour_m's lattice/traffic split in METRES. Clamp as delay_breakdown_s does for seconds
    # ("never exceed what it splits"): the ShortcutRefiner can leave lattice_overhead_m a hair above
    # air_detour_m, without this the two metre columns (and the rollup means built off them) would
    # overshoot air_detour_m while the seconds (taken from db) stayed reconciled — the two surfaces
    # would silently disagree.
    lattice_m = min(intent.lattice_overhead_m, intent.air_detour_m)
    # cost-space twin of total_delay_s: the four congestion levers above an unimpeded flight
    # at the planner's own cruise altitude — ground/hold/detour (already excess) + altitude's EXCESS
    # only (mandatory climb is not congestion). Same four levers total_delay_s sums in TIME; differs
    # from `cost`, which also carries the mandatory baseline-altitude cost.
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
        # A*-only split of air_detour_m. Its Euclidean baseline is unreachable on a 6-direction
        # lattice, so for A* it books pure geometry as congestion; the split separates that
        # quantization from the traffic-attributable berth. air_detour_m is NOT planner-neutral,
        # deliberately: it charges A* for the staircase and the endpoint/lane snap onto hex centres,
        # both real metres A* makes the drone fly, so they belong on A*'s bill (this is why a hex
        # planner reads worse than a continuous one here — the honest comparison, not a bias to
        # correct). For "how hard is traffic pushing flights sideways?" read deconfliction_detour_m:
        # its traffic share is derived from hex step counts, independently of air_detour_m, so the
        # snap and staircase both land in lattice_overhead_m and the traffic number is unchanged by
        # them. Exact for the A* family; continuous planners have lattice_overhead_m == 0, but their
        # air_detour_m is NOT 0 (a MILP books its knot discretization there), so for milp read
        # deconfliction_detour_m ± that knot noise.
        "lattice_overhead_m": lattice_m,
        "deconfliction_detour_m": intent.air_detour_m - lattice_m,
        # detour as lateness-seconds; ground_delay_s + air_hold_s + detour_time_s +
        # altitude_delay_phys_s == total_delay_s (the four time-space levers). Reuse db, lives once.
        "detour_time_s": db["detour_time_s"],
        # ... and detour_time_s itself splits exactly into these two (see delay_breakdown_s), so
        # viz.delay_sources can stack a five-band decomposition that still reconciles to the total.
        "detour_lattice_s": db["detour_lattice_s"],
        "detour_traffic_s": db["detour_traffic_s"],
        "altitude_change_m": intent.altitude_change_m,
        # traffic-forced vertical travel above the flight's cruise baseline + its two time readings
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
        # Persisted so a readout can reproduce flight_frame's window membership: it is a DIFFERENT
        # clock from t_request under timing_mode="departure". See occupancy_time_s.
        "t_occupancy": occupancy_time_s(intent),
    }


def flight_frame(result: SimResult, window: tuple[float, float] | None = None) -> pd.DataFrame:
    """Per-flight metrics table — one row per intent, FCFS order preserved.

    Membership is on the OCCUPANCY clock, not the filing clock, because every window this is called
    with (``steady_state_window``, ``simulation_window``) is itself derived from airborne density.
    Mixing the two silently produced empty cohorts under ``timing_mode="departure"``.

    Parameters
    ------------
    - result (SimResult): the run; ``result.intents`` become the rows.
    - window (tuple[float, float] | None): restrict to flights that took the airspace in
      ``[t_lo, t_hi)`` (the occupancy clock; see :func:`occupancy_time_s`) and clamp each row's
      reserved volume-seconds to it. ``None`` (default) is the whole run: every intent, volume
      measured across the complete realized :func:`simulation_window`, identical to the persisted
      ``flights.parquet``.

    Return
    --------
    - output (pd.DataFrame): one row per intent (filtered to the window when one is given).
    """
    measurement_window = simulation_window(result) if window is None else window
    df = pd.DataFrame([
        flight_row(intent, result.config, measurement_window)
        for intent in result.intents
    ])
    if window is not None and len(df):
        lo, hi = window
        df = df[(df["t_occupancy"] >= lo) & (df["t_occupancy"] < hi)].reset_index(drop=True)
    return df


def _q(series: pd.Series, q: float) -> float:
    """The ``q`` quantile of ``series`` (0.0 for an empty series)."""
    return float(series.quantile(q)) if len(series) else 0.0


def _mean(series: pd.Series) -> float:
    """The mean of ``series`` (0.0 for an empty series)."""
    return float(series.mean()) if len(series) else 0.0


def _rollup(
    df: pd.DataFrame,
    cfg: SimConfig,
    *,
    dur_s: float,
    rate_dur_s: float,
) -> dict:
    """Group-level rollup of a (sub)frame of flight rows — shared by ``aggregate`` (the whole run)
    and ``per_uss_frame`` (one operator's slice). Denominators (duration, region capacity) are the
    *run's*, so a per-USS ``airspace_utilization`` reads as that operator's share of the whole sky.

    Parameters
    ------------
    - df (pd.DataFrame): the (sub)frame of :func:`flight_row` records to summarize.
    - cfg (SimConfig): region geometry and flight-level band for the capacity denominator.
    - dur_s (float): the reserved-volume CAPACITY denominator (realized-run seconds).
    - rate_dur_s (float): the offered-load / throughput RATE denominator (demand-window seconds).
      Both are REQUIRED with no default: the only sensible fallback (``cfg.horizon_s``) is the
      denominator this split replaced, so a caller that silently got it would report
      horizon-normalized rates inside a realized-run summary. Build them with :func:`_denominators`.
      They differ only for a whole-run measurement of a scenario whose demand window is shorter than
      its planner envelope (the density family) — reporting rates over the active demand window
      while normalizing utilization over the realized run.

    Return
    --------
    - output (dict): headline counts, denial/rate/delay/detour/cost/altitude aggregates, and
      ``airspace_utilization`` for this frame.
    """
    acc = df[df["accepted"]]
    den = df[df["denied"]]
    rate_h = rate_dur_s / 3600.0
    # Vertical extent of the usable airspace. With discrete flight levels (n_levels > 1) the usable
    # tube is the regulated band [ground, airspace_ceiling]. Otherwise the continuous band collapses
    # to a single plane (z_max == z_min), so fall back to the corridor slab height (the vertical
    # footprint a flight occupies) so utilization stays meaningful, not a divide-by-zero.
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
        "offered_load_per_h": len(df) / max(rate_h, 1e-9),
        "throughput_per_h": len(acc) / max(rate_h, 1e-9),
        "mean_ground_delay_s": float(acc["ground_delay_s"].mean()) if len(acc) else 0.0,
        "p95_ground_delay_s": _q(acc["ground_delay_s"], 0.95),
        "mean_total_delay_s": float(acc["total_delay_s"].mean()) if len(acc) else 0.0,
        "p50_total_delay_s": _q(acc["total_delay_s"], 0.50),
        "p95_total_delay_s": _q(acc["total_delay_s"], 0.95),
        "mean_delay_pct": float(acc["delay_pct"].mean()) if len(acc) else 0.0,
        "p95_delay_pct": _q(acc["delay_pct"], 0.95),
        "mean_air_detour_m": float(acc["air_detour_m"].mean()) if len(acc) else 0.0,
        "p95_air_detour_m": _q(acc["air_detour_m"], 0.95),
        # A*-only split of mean_air_detour_m: hex quantization vs. traffic-attributable berth. Read
        # mean_deconfliction_detour_m, not mean_air_detour_m, when asking "how far did traffic push
        # flights sideways?" — on a lattice the latter is dominated by geometry at low congestion.
        "mean_lattice_overhead_m": float(acc["lattice_overhead_m"].mean()) if len(acc) else 0.0,
        "mean_deconfliction_detour_m": (float(acc["deconfliction_detour_m"].mean())
                                        if len(acc) else 0.0),
        "p95_deconfliction_detour_m": _q(acc["deconfliction_detour_m"], 0.95),
        "mean_stretch": float(acc["stretch"].mean()) if len(acc) else 1.0,
        "mean_cost": float(acc["cost"].mean()) if len(acc) else 0.0,
        # COST decomposition (planner-objective units) — "where did the congestion cost go?". The
        # four reconcile to mean_cost EXACTLY (altitude_cost is the full climb+descent); it is
        # mean_congestion_cost that equals mean_cost minus the mandatory baseline-altitude cost.
        "mean_ground_delay_cost": _mean(acc["ground_delay_cost"]),
        "mean_air_hold_cost": _mean(acc["air_hold_cost"]),
        "mean_air_detour_cost": _mean(acc["air_detour_cost"]),
        "mean_altitude_cost": _mean(acc["altitude_cost"]),
        "mean_congestion_cost": _mean(acc["congestion_cost"]),   # cost-space twin of mean_total_delay_s
        # vertical deconfliction: traffic-forced climb above floor, in metres + two time readings
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


def _denominators(result: SimResult, window: tuple[float, float] | None):
    """The measurement bounds and the two rollup denominators, derived in ONE place.

    ``aggregate`` and ``per_uss_frame`` both need this pair; keeping one spelling stops the two from
    drifting apart and silently normalizing an aggregate's totals differently from the per-USS rows
    embedded in it.

    Parameters
    ------------
    - result (SimResult): the run; only scanned via :func:`simulation_window` on the whole-run path.
    - window (tuple[float, float] | None): ``None`` ⇒ whole run (bounds from
      :func:`simulation_window`, capacity over the realized run and rates over the active demand
      window — they differ only when demand is shorter than the planner envelope, the density
      family). A window uses itself for both denominators.

    Return
    --------
    - output (tuple[float, float, float, float]): ``(lo, hi, dur_s, rate_dur_s)`` — the bounds, the
      capacity denominator, and the rate denominator.
    """
    if window is None:
        sim_lo, sim_hi = simulation_window(result)
        dur = sim_hi - sim_lo
        # The realized run is unbounded by horizon_s by design — the return tail legitimately runs
        # past it — but it is bounded by PHYSICS: a run cannot last orders of magnitude longer than
        # its planner envelope. A wall-style sentinel (_WALL_T_END_S = 1e12) reaching an accepted
        # intent's .volumes would divide utilization by ~1e9 with no error, so warn instead of
        # silently publishing it. Not a clamp: clamping would re-truncate the return tail.
        if dur > _MAX_PLAUSIBLE_RUN_FACTOR * result.config.horizon_s:
            warnings.warn(
                f"realized run duration {dur:.0f}s exceeds {_MAX_PLAUSIBLE_RUN_FACTOR}x horizon_s "
                f"({result.config.horizon_s:.0f}s) — a sentinel/open-ended reservation has probably "
                f"reached an accepted intent; airspace_utilization is meaningless for this run",
                RuntimeWarning, stacklevel=2)
        return sim_lo, sim_hi, dur, result.config.effective_demand_duration_s
    lo, hi = window
    return lo, hi, hi - lo, hi - lo


def _per_uss_table(
    df: pd.DataFrame,
    cfg: SimConfig,
    *,
    dur_s: float,
    rate_dur_s: float,
) -> pd.DataFrame:
    """One :func:`_rollup` row per USS, plus per-operator-only columns.

    The rollup denominators are the run's (passed through), so each USS row reads as that operator's
    share of the whole sky. ``share_of_accepted`` is against the frame's total accepted.

    Parameters
    ------------
    - df (pd.DataFrame): the flight rows to group by ``uss_id``.
    - cfg (SimConfig): forwarded to :func:`_rollup`.
    - dur_s (float): capacity denominator, forwarded to :func:`_rollup`.
    - rate_dur_s (float): rate denominator, forwarded to :func:`_rollup`.

    Return
    --------
    - output (pd.DataFrame): one row per USS (sorted by id), empty when ``df`` is empty.
    """
    total_acc = int(df["accepted"].sum()) if len(df) else 0
    rows = []
    for uss_id, g in df.groupby("uss_id", sort=True):
        acc = g[g["accepted"]]
        rows.append({
            "uss_id": uss_id,
            **_rollup(g, cfg, dur_s=dur_s, rate_dur_s=rate_dur_s),
            # per-USS-only: flight length (confirms hub-demand shortening) + share of the throughput
            "mean_straight_line_m": float(acc["straight_line_m"].mean()) if len(acc) else 0.0,
            "share_of_accepted": (len(acc) / total_acc) if total_acc else 0.0,
        })
    return pd.DataFrame(rows)


def per_uss_frame(result: SimResult, window: tuple[float, float] | None = None) -> pd.DataFrame:
    """One metrics row per USS — the per-operator slice of a (multi-)USS run.

    Each row's counts and reserved volume sum to the overall ``aggregate`` totals (see tests).

    Parameters
    ------------
    - result (SimResult): the run.
    - window (tuple[float, float] | None): restrict to flights that took the airspace in
      ``[t_lo, t_hi)`` (occupancy clock; see :func:`occupancy_time_s`), using the window duration
      for both denominators. ``None`` ⇒ rates use the active demand duration and utilization the
      realized simulation duration.

    Return
    --------
    - output (pd.DataFrame): one row per USS.
    """
    _lo, _hi, dur_s, rate_dur_s = _denominators(result, window)
    return _per_uss_table(
        flight_frame(result, window),
        result.config,
        dur_s=dur_s,
        rate_dur_s=rate_dur_s,
    )


def aggregate(result: SimResult, window: tuple[float, float] | None = None) -> dict:
    """Flat headline rollup for one run — the row a λ-sweep collects.

    Use :func:`aggregate_with_steady` to report the whole-run view next to its steady-state twin.

    Parameters
    ------------
    - result (SimResult): the run to summarize.
    - window (tuple[float, float] | None): measure only flights that took the airspace in that
      interval (the occupancy clock, matching :func:`flight_frame`), with denominators using the
      window duration and ``window_lo`` / ``window_hi`` added for provenance. ``None`` (default)
      measures the complete realized run, first activity through final landing.

    Return
    --------
    - output (dict): the :func:`_rollup` headline keys plus run identity, denials-by-reason,
      cross-USS fairness spreads, and either the ``simulation_*`` bounds (whole run) or
      ``window_lo`` / ``window_hi`` (windowed).
    """
    cfg = result.config
    # A windowed call must NOT pay for simulation_window: a full scan of every accepted flight's
    # volumes (millions on a density run), and aggregate_with_steady pops the simulation_* keys back
    # off the steady block. _denominators only derives it when needed.
    sim_lo, sim_hi, dur_s, rate_dur_s = _denominators(result, window)
    df = flight_frame(result, window)
    den = df[df["denied"]] if len(df) else df
    by_reason = den["denial_reason"].value_counts().to_dict() if len(den) else {}

    # cross-USS fairness: does one operator systematically lose under FCFS? (0 when single-USS)
    per_uss = _per_uss_table(df, cfg, dur_s=dur_s, rate_dur_s=rate_dur_s)
    n_uss = int(len(per_uss))
    if n_uss > 1:
        denial_rate_spread = float(per_uss["denial_rate"].max() - per_uss["denial_rate"].min())
        mean_delay_spread = float(per_uss["mean_total_delay_s"].max() - per_uss["mean_total_delay_s"].min())
    else:
        denial_rate_spread = mean_delay_spread = 0.0

    out = {
        "lam_per_hour": cfg.lam_per_hour,
        "demand_duration_s": cfg.effective_demand_duration_s,
        "seed": cfg.seed,
        "planner": cfg.planner,
        **_rollup(df, cfg, dur_s=dur_s, rate_dur_s=rate_dur_s),
        "denials_by_reason": by_reason,
        "n_uss": n_uss,
        "denial_rate_spread": denial_rate_spread,
        "mean_delay_spread": mean_delay_spread,
        "verified": result.verified,
    }
    if window is None:
        # Only the whole-run view can honestly name these: for a windowed call `_denominators`
        # returns the WINDOW, so emitting them here would publish the window under a key that says
        # "simulation". aggregate_with_steady pops them off the steady block anyway, so the windowed
        # path loses nothing — and skips the full simulation_window scan it would otherwise pay for.
        out["simulation_start_s"] = sim_lo
        out["simulation_end_s"] = sim_hi
        out["simulation_duration_s"] = sim_hi - sim_lo
    else:
        out["window_lo"], out["window_hi"] = float(window[0]), float(window[1])
    return out


def aggregate_with_steady(result: SimResult, frac: float = 0.9, smooth_s: float | None = None,
                          dt: float | None = None) -> dict:
    """The whole-run :func:`aggregate` plus a nested ``"steady_state"`` block: the same rollup
    measured over :func:`steady_state_window` (the representative density plateau), the two views
    reported side by side.

    The steady block carries ``window_lo`` / ``window_hi`` so a windowed number is self-describing,
    and drops the run-identity keys (lam/seed/planner/n_uss/verified) the two views share. When no
    plateau is detectable (small / low-λ runs) the window is the realized run, whose rate fields can
    still differ from full-cohort rates when the configured demand duration differs from that run.

    Parameters
    ------------
    - result (SimResult): the run to summarize.
    - frac (float): plateau threshold, forwarded to :func:`steady_state_window`.
    - smooth_s (float | None): smoothing width for the plateau search; ``None`` adapts it to the
      median trip duration, so the window tracks the plateau, not a transient concurrency spike.
    - dt (float | None): density-grid step, forwarded to :func:`steady_state_window`.

    Return
    --------
    - output (dict): the whole-run :func:`aggregate` with an added ``"steady_state"`` sub-dict.
    """
    win = steady_state_window(result, frac=frac, dt=dt, smooth_s=smooth_s)
    out = aggregate(result)
    # The steady twin means something only if the window actually selected flights. Two ways it can
    # come up empty and republish a block of structural zeros — an all-denied run would then archive
    # steady_denial_rate 0.0, byte-identical to a flawless one:
    #   - a zero-width window (nothing accepted ⇒ simulation_window is (0, 0)); and
    #   - a positive window no takeoff's occupancy clock falls inside (a low-λ plateau spanned end
    #     to end by aircraft that departed earlier) ⇒ n_requests == 0 despite hi > lo.
    # Either way, fall back to the whole-run view. (dict(out) is a shallow copy; only the two scalar
    # window keys are reassigned below.)
    steady = aggregate(result, window=win) if win[1] > win[0] else dict(out)
    if steady.get("n_requests", 0) == 0:
        steady = dict(out)
    steady["window_lo"], steady["window_hi"] = float(win[0]), float(win[1])
    for k in (
        "lam_per_hour",
        "demand_duration_s",
        "simulation_start_s",
        "simulation_end_s",
        "simulation_duration_s",
        "seed",
        "planner",
        "n_uss",
        "verified",
    ):
        steady.pop(k, None)
    out["steady_state"] = steady
    return out
