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
    congestion signal the experiment measures. SEARCH_EXHAUSTED is a *possible artifact*: the
    bounded planner gave up; a higher sample cap might have found a path. Reporting them separately
    lets the headline denial-rate count real congestion and lets us audit the artifact's size.
    """

    NONE = "none"
    BUDGET_EXCEEDED = "budget_exceeded"      # no plan within max_ground_delay_s / max_detour_factor
    SEARCH_EXHAUSTED = "search_exhausted"    # hit the RRT* sample cap (compute-bounded)
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

    def __post_init__(self):
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

    def sort_key(self) -> tuple[float, int]:
        return (self.t_request, self.flight_id)


@dataclass
class OperationalIntent:
    """The reserved plan for one flight (ASTM operational intent).

    ``volumes`` is the full reservation: hover cylinder @origin + corridor boxes + hover cylinder
    @dest. ``centerline`` is the timed polyline the corridor was built around (also the v0 flown
    path). Cost decomposes into the knobs that drive FCFS trade-offs.
    """

    request: FlightRequest
    status: IntentStatus
    volumes: list[Volume4D] | None = None
    centerline: list[TimedPoint] | None = None
    ground_delay_s: float = 0.0       # time held on the pad before departure
    air_hold_s: float = 0.0           # time loitering/hovering mid-route
    air_detour_m: float = 0.0         # flown horizontal length − straight-line length
    altitude_change_m: float = 0.0    # total vertical travel (climb + descent)
    cost: float = 0.0
    denial_reason: "DenialReason" = field(default=None)  # type: ignore[assignment]
    planner: str = ""                 # which planner produced this intent
    solve_time_s: float = 0.0         # wall time the planner spent on this flight's plan() call

    def __post_init__(self):
        if self.denial_reason is None:
            self.denial_reason = (
                DenialReason.NONE if self.status is not IntentStatus.REJECTED
                else DenialReason.BUDGET_EXCEEDED
            )

    @property
    def accepted(self) -> bool:
        return self.status in (IntentStatus.ACCEPTED, IntentStatus.ACTIVATED)


@dataclass
class FlightLog:
    """What was actually flown. v0 = perfect conformance (trajectory == reserved centerline)."""

    flight_id: int
    trajectory: list[TimedPoint] = field(default_factory=list)
    conformed: bool = True
