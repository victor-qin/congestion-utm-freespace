"""Scenario assembly — lift a flat request list into FCFS-ordered demand events.

Mirrors the sibling project: the simulator consumes time-ordered events; FCFS order is
``(t_request, flight_id)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import FlightRequest


@dataclass
class DemandEvent:
    """One demand event: a flight ``request`` becomes plannable at simulation time ``t`` (s)."""

    t: float
    request: FlightRequest


@dataclass
class Scenario:
    """A run's demand as time-ordered events plus the set of participating USS ids."""

    events: list[DemandEvent]
    uss_ids: list[str]


def scenario_from_requests(requests: list[FlightRequest]) -> Scenario:
    """Lift a flat request list into a FCFS-ordered ``Scenario``.

    Events are sorted by ``(t_request, flight_id)`` — the FCFS tie-break — and the USS ids are
    the sorted unique suppliers, falling back to ``["default"]`` when no request names one.

    Parameters
    ------------
    - requests (list[FlightRequest]): the flat, unordered flight requests.

    Return
    --------
    - output (Scenario): events in FCFS order plus the sorted unique USS ids.
    """
    events = [DemandEvent(r.t_request, r) for r in requests]
    events.sort(key=lambda e: (e.t, e.request.flight_id))
    uss_ids = sorted({r.uss_id for r in requests}) or ["default"]
    return Scenario(events=events, uss_ids=uss_ids)
