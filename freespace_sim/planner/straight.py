"""Straight-line + time-shift planner (the cheap temporal tier).

Geometry is fixed (the direct path at cruise altitude); the only lever is *when* to depart. The fix
for a conflict is the **jump-to-gap time-shift**: ask the ledger which committed volumes block us and
slide the whole schedule forward to when the earliest blocker clears — not a blind dt-by-dt scan.

It can only ever produce a ground delay. When no time gap exists within ``max_ground_delay_s`` it
returns REJECTED(BUDGET_EXCEEDED) — the cue for a spatial planner (A*) to take over.

`plan_timeshift` is factored out with a speed factor so the decoupled planner can reuse the exact
same time search at different cruise speeds.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from ..cost import trajectory_cost
from ..ledger import ReservationLedger
from ..types import (
    DenialReason,
    FlightRequest,
    IntentStatus,
    OperationalIntent,
    TimedPoint,
    Vec,
    vec,
)
from ..volumes import Volume4D, build_corridor, hover_reservation
from .itinerary import reject_itinerary

_EPS = 1e-6


def cruise_centerline(
    origin: Vec, dest: Vec, t_depart: float, cfg: SimConfig, speed: float | None = None
) -> list[TimedPoint]:
    """Timed waypoints along the level cruise leg, spaced one corridor segment apart.

    Cruise begins after the climb (``t_start = t_depart + climb_time_s``) at ``cruise_level_m``; a
    degenerate origin==dest leg returns a two-point stationary pair.

    Parameters
    ------------
    - origin (Vec): origin position (xy used).
    - dest (Vec): destination position (xy used).
    - t_depart (float): filed departure time (s); the climb is added before cruise.
    - cfg (SimConfig): supplies cruise level, climb time, and ``corridor_segment_len_m``.
    - speed (float | None): cruise speed (m/s); ``None`` ⇒ ``cfg.nominal_speed_mps``.

    Return
    --------
    - output (list[TimedPoint]): waypoints ``corridor_segment_len_m`` apart, timed at ``speed``.
    """
    speed = speed if speed is not None else cfg.nominal_speed_mps
    o = np.asarray(origin, float)[:2]
    d = np.asarray(dest, float)[:2]
    z = cfg.cruise_level_m
    t_start = t_depart + cfg.climb_time_s          # cruise begins after the climb
    distance = float(np.linalg.norm(d - o))
    if distance < _EPS:                            # origin == dest: degenerate two-point leg
        p = vec(o[0], o[1], z)
        return [(p, t_start), (p, t_start + cfg.dt_s)]
    seg = cfg.corridor_segment_len_m
    n = max(1, int(np.ceil(distance / seg)))
    pts: list[TimedPoint] = []
    for k in range(n + 1):
        s = min(k * seg, distance)
        frac = s / distance
        xy = o + frac * (d - o)
        pts.append((vec(xy[0], xy[1], z), t_start + s / speed))
    return pts


def build_reservation(
    origin: Vec, dest: Vec, t_depart: float, cfg: SimConfig, speed: float | None = None
) -> tuple[list[Volume4D], list[TimedPoint]]:
    """Assemble the straight-cruise reservation: hover@origin + corridor + hover@dest.

    Parameters
    ------------
    - origin (Vec): origin position.
    - dest (Vec): destination position.
    - t_depart (float): filed departure time (s).
    - cfg (SimConfig): supplies geometry, speeds, and timing.
    - speed (float | None): cruise speed (m/s); ``None`` ⇒ ``cfg.nominal_speed_mps``.

    Return
    --------
    - output (tuple[list[Volume4D], list[TimedPoint]]): the reservation volumes (origin hover,
      corridor boxes, dest hover) and the timed cruise centreline.
    """
    cl = cruise_centerline(origin, dest, t_depart, cfg, speed=speed)
    t_arrive = cl[-1][1]
    volumes = [hover_reservation(origin, t_depart, cfg)]
    volumes += build_corridor(cl, cfg)
    volumes.append(hover_reservation(dest, t_arrive, cfg))
    return volumes, cl


def plan_timeshift(
    req: FlightRequest,
    ledger: ReservationLedger,
    cfg: SimConfig,
    *,
    speed_factor: float = 1.0,
    planner_name: str = "straight",
) -> OperationalIntent:
    """Jump-to-gap time-shift search at a fixed cruise speed (``speed_factor`` × nominal).

    Slides the whole schedule forward to the earliest conflict-free time: on a conflict it jumps
    past the earliest blocker's ``t_end`` (never a blind ``dt`` scan), so each retry makes strict
    progress. Returns REJECTED(BUDGET_EXCEEDED) when no gap opens within ``max_ground_delay_s``.

    Parameters
    ------------
    - req (FlightRequest): the flight to plan; departs at ``t_departure`` (else ``t_request``).
    - ledger (ReservationLedger): committed reservations to deconflict against.
    - cfg (SimConfig): supplies speed, timing, and ``max_ground_delay_s``.
    - speed_factor (float): cruise speed as a fraction of ``nominal_speed_mps``.
    - planner_name (str): name stamped on the returned intent.

    Return
    --------
    - output (OperationalIntent): ACCEPTED with the conflict-free volumes and ground delay, or
      REJECTED(BUDGET_EXCEEDED) when no gap fits the ground-delay budget.
    """
    reject_itinerary(req, planner_name)
    speed = cfg.nominal_speed_mps * speed_factor
    base = req.t_departure if req.t_departure is not None else req.t_request
    delay = 0.0
    while delay <= cfg.max_ground_delay_s:
        volumes, cl = build_reservation(req.origin, req.dest, base + delay, cfg, speed=speed)
        hits = ledger.conflicts(volumes)
        if not hits:
            intent = OperationalIntent(
                request=req,
                status=IntentStatus.ACCEPTED,
                volumes=volumes,
                centerline=cl,
                ground_delay_s=delay,
                altitude_change_m=2.0 * (cfg.cruise_level_m - cfg.ground_level_m),
                planner=planner_name,
            )
            intent.cost = trajectory_cost(intent, cfg)
            return intent
        # Jump the whole schedule past the earliest blocker, ensuring strict progress.
        earliest_clear = min(cv.t_end for _, cv in hits)
        delay = max((earliest_clear + _EPS) - base, delay + cfg.dt_s)
    return OperationalIntent(
        request=req,
        status=IntentStatus.REJECTED,
        denial_reason=DenialReason.BUDGET_EXCEEDED,
        planner=planner_name,
    )


class StraightLineTimeShift:
    """The cheap temporal tier: a straight cruise deconflicted by ground delay alone."""

    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent:
        """Deconflict ``req`` by ground delay only, at nominal cruise speed."""
        return plan_timeshift(req, ledger, cfg, speed_factor=1.0, planner_name="straight")
