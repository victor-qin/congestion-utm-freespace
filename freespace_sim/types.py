"""Core data types — the continuous-space analogues of the sibling project's grid types.

A drone is never modelled as a persistent object; the simulation is coordination-mechanism
focused. Demand is a `FlightRequest`, the reserved plan is an `OperationalIntent`, and what was
flown is a `FlightLog`. (Mirrors `congestion_sim/types.py`, but positions are continuous 3D
vectors instead of H3 cells.)
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:  # avoid a runtime import cycle (volumes imports geometry imports nothing here)
    from .volumes import Volume4D

# A continuous position in the local ENU frame, metres: array([x, y, z]).
Vec = np.ndarray
# A time-stamped waypoint along a centreline: (position, wall-clock seconds).
TimedPoint = tuple[Vec, float]


def vec(x: float, y: float, z: float = 0.0) -> Vec:
    """Build a 3D position vector (metres)."""
    return np.array([x, y, z], dtype=float)


class IntentStatus(Enum):
    """ASTM F3548-21 §4.4 operational-intent states.

    v0 (strategic only) uses REJECTED / ACCEPTED / ENDED. The off-nominal states are carried so
    the future BlueSky tactical layer can drive conformance transitions without a type change.
    """

    REJECTED = "rejected"      # no conflict-free plan within budget (denied)
    ACCEPTED = "accepted"      # committed in the ledger (nominal)
    ACTIVATED = "activated"    # in flight (nominal)
    NONCONFORMING = "nonconforming"  # off-nominal (tactical layer)
    CONTINGENT = "contingent"        # off-nominal (tactical layer)
    ENDED = "ended"            # completed / removed


class DenialReason(Enum):
    """Why a request was denied — keeps real congestion separate from compute artifacts.

    BUDGET_EXCEEDED is *physics*: no plan exists within the operator's budgets (delay/detour) — the
    congestion signal the experiment measures. It covers a path that busts the delay/detour budget AND
    an exhaustive planner (A*) emptying its queue with no feasible plan inside the horizon (a saturated
    hub with no launch/landing slot). SEARCH_EXHAUSTED is a *possible artifact*: the planner stopped at
    its compute cap before exhausting the space; a higher cap might have found a path. Reporting them
    separately lets the headline denial-rate count real congestion and audit the artifact's size.
    """

    NONE = "none"
    BUDGET_EXCEEDED = "budget_exceeded"      # no feasible plan within max_ground_delay_s / max_detour_factor
                                             #   (incl. A* exhausting its bounded horizon — real congestion)
    SEARCH_EXHAUSTED = "search_exhausted"    # stopped at the compute cap: A* max_expansions
    CONFLICT_AT_COMMIT = "conflict_at_commit"  # lost a commit-time race (multi-USS, future)
    CONFLICT_FILED = "conflict_filed"  # filing has a conflict (multi-USS, future)


class Terminal(NamedTuple):
    """A multi-pad vertiport endpoint a flight uses (origin for a takeoff, dest for a landing).

    Vertiport infrastructure travels with the terminal, not in global config:
    - ``radius`` — the shared terminal column size; ``None`` ⇒ ``cfg.terminal_radius_m`` (90 m default),
      wide enough that divergent same-hub exit lanes don't crowd at the edge when flush.
    - ``corridor_overlap`` — how far the reserved exit lane overlaps INTO the column (inner edge =
      ``R − overlap``). ``None``/``0`` (default) ⇒ the lane starts FLUSH with the column edge; the
      column-involved exemption (``conflict.volumes_conflict``) keeps the tagged exit-lane box
      conflict-free with same-hub columns, while two same-hub corridors still contend. ``> 0`` penetrates
      the column; ``< 0`` leaves a clearance gap outside it. See ``volumes.exit_radius``.

    Both are set when hubs are created (the demand model), so a big-box hub and a small pad can differ
    and a non-hub flight simply has no terminal. ``capacity`` is the pad count N (Phase B).
    """

    id: Hashable
    capacity: int = 1
    radius: float | None = None
    corridor_overlap: float | None = None


def as_terminal(t) -> "Terminal | None":
    """Normalize a terminal descriptor: ``None``, a :class:`Terminal`, or a plain
    ``(id, capacity[, radius[, corridor_overlap]])`` tuple → a :class:`Terminal` (or ``None``)."""
    if t is None or isinstance(t, Terminal):
        return t
    return Terminal(*t)


@dataclass
class FlightRequest:
    """Pure demand: who wants to fly from where to where, and when they filed.

    FCFS order is defined by ``(t_request, flight_id)``. Positions are continuous 3D vectors;
    origin/dest are typically at ground level (z = 0).
    """

    flight_id: int
    origin: Vec
    dest: Vec
    t_request: float                 # filing time → FCFS order
    t_departure: float | None = None  # desired departure (None = depart at t_request)
    uss_id: str = "default"
    # multi-pad vertiport endpoints (:class:`Terminal`) when origin/dest is a shared-terminal hub.
    # None (default) → ordinary single pad. The hub *centre* is ``origin``/``dest``; the Terminal drives
    # the shared-terminal exemption + pad capacity + column size. A delivery sets origin_terminal; a
    # return sets dest_terminal. Plain ``(id, capacity)`` tuples are accepted (normalized by builders).
    origin_terminal: "Terminal | None" = None
    dest_terminal: "Terminal | None" = None
    # ``origin -> dest -> origin`` flown as ONE flight (:class:`~planner.itinerary.ItineraryPlanner`).
    # The return leg departs from the arrival the outbound ACTUALLY achieved, so it cannot precede it.
    return_to_origin: bool = False
    # Ground time at ``dest`` between the legs: the delivery itself. ``None`` inherits
    # ``SimConfig.turnaround_s``, which is the one owner of the number. Excludes the descent and
    # climb that bracket it (``volumes.column_dwell_s``), so it cannot budget a dwell physics
    # contradicts.
    turnaround_s: "float | None" = None

    def __post_init__(self):
        """Default and validate ``t_departure`` after construction.

        Parameters
        ------------
        - none: reads and, if unset, fills ``t_departure`` on ``self``.

        Return
        --------
        - output (None): sets ``t_departure = t_request`` when it is ``None``; raises ``ValueError``
          if a given ``t_departure`` precedes ``t_request``.
        """
        # Single source of truth for the file/departure relationship: a flight departs no earlier than
        # it is filed. ``None`` means "depart as soon as filed". Enforced here, not per-planner, so every
        # consumer can rely on ``t_departure`` being set and ``>= t_request`` — A*'s ``ceil(t_depart/dt)``
        # discretization and the request-clock eviction watermark both depend on it.
        if self.t_departure is None:
            self.t_departure = self.t_request
        elif self.t_departure < self.t_request:
            raise ValueError(
                f"t_departure ({self.t_departure}) < t_request ({self.t_request}): "
                "a flight cannot depart before it is filed"
            )
        if self.turnaround_s is not None and self.turnaround_s < 0.0:
            raise ValueError(f"flight {self.flight_id}: turnaround_s must be >= 0")

    def sort_key(self) -> tuple[float, int]:
        """FCFS sort key: ``(t_request, flight_id)``."""
        return (self.t_request, self.flight_id)


@dataclass
class OperationalIntent:
    """
    The reserved plan for one flight (ASTM operational intent).

    Attributes
    ------------
    - request (FlightRequest): the request this answers; for a round trip, the whole itinerary
    - status (IntentStatus): lifecycle state; ACCEPTED and ACTIVATED both count as `accepted`
    - volumes (list[Volume4D] | None): the full reservation — origin column, corridor boxes,
      destination column, and for a round trip the return leg with the pad-hold ground box between
      the two. Empty or None when denied.
    - centerline (list[TimedPoint] | None): the timed polyline the corridor was built around, which
      is also the flown path in v0
    - ground_delay_s (float): time held on the pad before departure
    - air_hold_s (float): time loitering or hovering mid-route
    - air_detour_m (float): en-route detour, flown minus reference, both measured exit lane to exit
      lane (`volumes.enroute_flown_m` / `enroute_reference_m`). Never derive it from a hub-centre
      baseline, which reintroduces stretch < 1; `metrics.flight_row` explains why A* books its
      endpoint snap here and continuous planners do not.
    - lattice_overhead_m (float): the A*-only share of `air_detour_m` forced by the hex lattice
      rather than by traffic, up to 2/√3 − 1 ≈ 15.5% of an unimpeded flight. Subtract it for the
      traffic-attributable detour. 0.0 for continuous planners; ShortcutRefiner reduces it.
    - altitude_change_m (float): total vertical travel, climb plus descent
    - cost (float): weighted sum of ground delay, air hold, detour and altitude change
      (`cost.trajectory_cost`), the one number every planner minimizes
    - denial_reason (DenialReason): why it was denied; defaulted from `status` in `__post_init__`
    - leg_starts (tuple[int, ...]): centerline indices where each later leg begins; empty for a
      one-way flight. The segment joining two legs is an unflown ground dwell, so split with
      `leg_slices` before building one corridor box per segment.
    - planner (str): name of the planner that produced this intent
    - solve_time_s (float): wall time spent in this flight's `plan()` call

    """

    request: FlightRequest
    status: IntentStatus
    volumes: list[Volume4D] | None = None
    centerline: list[TimedPoint] | None = None
    ground_delay_s: float = 0.0
    air_hold_s: float = 0.0
    air_detour_m: float = 0.0
    lattice_overhead_m: float = 0.0
    altitude_change_m: float = 0.0
    cost: float = 0.0
    denial_reason: "DenialReason" = field(default=None)  # type: ignore[assignment]
    leg_starts: tuple[int, ...] = ()
    planner: str = ""
    solve_time_s: float = 0.0

    def __post_init__(self):
        """Default ``denial_reason`` from ``status`` when it was not given explicitly.

        Parameters
        ------------
        - none: reads ``status`` / ``denial_reason`` on ``self``.

        Return
        --------
        - output (None): leaves an explicit ``denial_reason`` untouched; otherwise sets ``NONE`` for
          a non-rejected intent and ``BUDGET_EXCEEDED`` for a rejected one.
        """
        if self.denial_reason is None:
            self.denial_reason = (
                DenialReason.NONE if self.status is not IntentStatus.REJECTED
                else DenialReason.BUDGET_EXCEEDED
            )

    def leg_slices(self, seq=None) -> list:
        """
        ``seq`` cut at :attr:`leg_starts` — one slice per flown leg, in flown order.

        The one owner of "split an itinerary into legs", so metrics, the viz payload and any future
        consumer cannot drift apart on where a leg ends. Slices are returned UNFILTERED, including
        empty ones, because dropping a short leg silently renumbers the rest — a caller that cannot
        use a run shorter than two points must skip it itself and say so.

        Parameters
        ------------
        - seq (Sequence | None): the per-waypoint sequence to cut, indexed like ``centerline``;
          ``None`` uses this intent's own ``centerline``

        Return
        --------
        - legs (list): ``len(leg_starts) + 1`` slices of ``seq``, covering it exactly once
        """
        if seq is None:
            seq = self.centerline or []
        bounds = [0, *self.leg_starts, len(seq)]
        return [seq[a:b] for a, b in zip(bounds, bounds[1:])]

    @property
    def accepted(self) -> bool:
        """True when the intent is committed or in flight (ACCEPTED or ACTIVATED)."""
        return self.status in (IntentStatus.ACCEPTED, IntentStatus.ACTIVATED)


@dataclass
class FlightLog:
    """What was actually flown. v0 = perfect conformance (trajectory == reserved centerline)."""

    flight_id: int
    trajectory: list[TimedPoint] = field(default_factory=list)
    conformed: bool = True
