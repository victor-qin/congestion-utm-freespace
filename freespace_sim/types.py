"""Core data types — the continuous-space analogues of the sibling project's grid types.

A drone is never modelled as a persistent object; the simulation is coordination-mechanism
focused. Demand is a `FlightRequest`, the reserved plan is an `OperationalIntent`, and what was
flown is a `FlightLog`. (Mirrors `congestion_sim/types.py`, but positions are continuous 3D
vectors instead of H3 cells.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

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
