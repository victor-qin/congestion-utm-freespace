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
    """Build a 3D position vector in metres.

    Parameters
    ------------
    - x (float): Local ENU x-coordinate (East) in metres.
    - y (float): Local ENU y-coordinate (North) in metres.
    - z (float): Local ENU z-coordinate (Up) in metres. Defaults to 0.0.

    Return
    --------
    - position (Vec): 3-element float array representing the 3D position.
    """
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
    """Multi-pad vertiport terminal endpoint used by a flight at origin or destination.

    Parameters
    ------------
    - id (Hashable): Unique identifier for the vertiport terminal.
    - capacity (int): Number of concurrent landing/takeoff pads N. Defaults to 1.
    - radius (float | None): Shared terminal column radius in metres, or None to use default. Defaults to None.
    - corridor_overlap (float | None): Overlap distance into terminal column in metres. Defaults to None.
    """

    id: Hashable
    capacity: int = 1
    radius: float | None = None
    corridor_overlap: float | None = None


def as_terminal(t) -> "Terminal | None":
    """Normalize a terminal descriptor tuple or instance into a Terminal object.

    Parameters
    ------------
    - t (Terminal | tuple | None): Terminal instance or (id, capacity[, radius[, corridor_overlap]]) tuple.

    Return
    --------
    - terminal (Terminal | None): Normalized Terminal object or None.
    """
    if t is None or isinstance(t, Terminal):
        return t
    return Terminal(*t)


@dataclass
class FlightRequest:
    """Demand request representing a flight intent from origin to destination.

    Parameters
    ------------
    - flight_id (int): Unique identifier for the flight.
    - origin (Vec): 3D origin position vector [x, y, z] in local ENU metres.
    - dest (Vec): 3D destination position vector [x, y, z] in local ENU metres.
    - t_request (float): Time the flight was filed/requested (seconds).
    - t_departure (float | None): Desired departure time (seconds). Defaults to None (depart at t_request).
    - uss_id (str): UAS Service Supplier identifier. Defaults to "default".
    - origin_terminal (Terminal | None): Origin vertiport terminal specification if applicable. Defaults to None.
    - dest_terminal (Terminal | None): Destination vertiport terminal specification if applicable. Defaults to None.
    - paired_outbound_id (int | None): Flight ID of outbound leg for round-trip return flights. Defaults to None.
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
    # Round-trip link: on a RETURN leg, the flight_id of the outbound whose arrival this leg waits on.
    # Demand models anchor the return's desired departure to a NOMINAL estimate of that arrival, which
    # ignores whatever ground delay / hold / detour the outbound actually took — so under congestion a
    # return can want to depart before its aircraft has landed. Naming the dependency explicitly (rather
    # than leaving it an implicit flight_id + 1 convention) lets ``sim.run(return_anchor="realized")``
    # re-anchor the return to the arrival its outbound actually achieved.
    paired_outbound_id: "int | None" = None

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

    def sort_key(self) -> tuple[float, int]:
        """FCFS sort key: ``(t_request, flight_id)``."""
        return (self.t_request, self.flight_id)


@dataclass
class OperationalIntent:
    """The reserved plan for one flight under ASTM F3548-21 operational intent semantics.

    Parameters
    ------------
    - request (FlightRequest): The underlying demand request.
    - status (IntentStatus): Operational intent state (ACCEPTED, REJECTED, etc.).
    - volumes (list[Volume4D] | None): 4D reserved space-time volumes. Defaults to None.
    - centerline (list[TimedPoint] | None): Timed waypoint polyline path. Defaults to None.
    - ground_delay_s (float): Delay held on the pad before departure (seconds). Defaults to 0.0.
    - air_hold_s (float): Airborne holding or hovering time (seconds). Defaults to 0.0.
    - air_detour_m (float): En-route lateral detour distance (metres). Defaults to 0.0.
    - lattice_overhead_m (float): Geometric overhead attributed to lattice discretization (metres). Defaults to 0.0.
    - altitude_change_m (float): Total vertical climb and descent distance (metres). Defaults to 0.0.
    - cost (float): Total weighted cost according to the cost model. Defaults to 0.0.
    - denial_reason (DenialReason): Root cause classification if rejected. Defaults to None.
    - planner (str): Identifier of the planner that generated this intent. Defaults to "".
    - solve_time_s (float): Planner compute wall-clock time in seconds. Defaults to 0.0.
    """

    request: FlightRequest
    status: IntentStatus
    volumes: list[Volume4D] | None = None
    centerline: list[TimedPoint] | None = None
    ground_delay_s: float = 0.0       # time held on the pad before departure
    air_hold_s: float = 0.0           # time loitering/hovering mid-route
    # EN-ROUTE detour: flown minus reference, BOTH measured exit lane -> exit lane via
    # volumes.enroute_flown_m / enroute_reference_m — never hub centre -> hub centre. Terminal-column
    # flying is in neither side (capacity-only); a terminal-free endpoint extends to the true
    # origin/dest, so A* pays its endpoint snap here while continuous planners don't (deliberate —
    # see metrics.flight_row). Any producer setting this from a hand-rolled centre->centre baseline
    # reintroduces the stretch<1 bug.
    air_detour_m: float = 0.0
    # A*-ONLY diagnostic: the share of ``air_detour_m`` forced by the hex lattice rather than by
    # traffic. A* moves on 6 axial directions, so the Euclidean (lane -> lane) straight line
    # ``air_detour_m`` measures against is UNREACHABLE — a wholly unimpeded flight still books up to
    # 2/√3 − 1 ≈ 15.5% of pure geometry as if it were congestion (worst case at 30° off-axis, zero
    # on-axis). Subtract this to read the traffic-attributable detour. 0.0 for the continuous
    # planners (milp / straight), which have no lattice; reduced by ShortcutRefiner, which
    # collapses the staircase.
    lattice_overhead_m: float = 0.0
    altitude_change_m: float = 0.0    # total vertical travel (climb + descent)
    cost: float = 0.0
    denial_reason: "DenialReason" = field(default=None)  # type: ignore[assignment]
    planner: str = ""                 # which planner produced this intent
    solve_time_s: float = 0.0         # wall time the planner spent on this flight's plan() call

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

    @property
    def accepted(self) -> bool:
        """True when the intent is committed or in flight (ACCEPTED or ACTIVATED)."""
        return self.status in (IntentStatus.ACCEPTED, IntentStatus.ACTIVATED)


@dataclass
class FlightLog:
    """Record of trajectory flown by a flight and conformance verification.

    Parameters
    ------------
    - flight_id (int): Unique identifier of the flight.
    - trajectory (list[TimedPoint]): List of timed 3D waypoints actually flown.
    - conformed (bool): True if flight maintained conformance to operational intent. Defaults to True.
    """

    flight_id: int
    trajectory: list[TimedPoint] = field(default_factory=list)
    conformed: bool = True
