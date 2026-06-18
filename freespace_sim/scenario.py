"""Scenario assembly — lift a flat request list into FCFS-ordered demand events.

Mirrors the sibling project: the simulator consumes time-ordered events; FCFS order is
``(t_request, flight_id)``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import FlightRequest


@dataclass
class DemandEvent:
    t: float
    request: FlightRequest


@dataclass
class Scenario:
    events: list[DemandEvent]
    uss_ids: list[str]


def scenario_from_requests(requests: list[FlightRequest]) -> Scenario:
    events = [DemandEvent(r.t_request, r) for r in requests]
    events.sort(key=lambda e: (e.t, e.request.flight_id))
    uss_ids = sorted({r.uss_id for r in requests}) or ["default"]
    return Scenario(events=events, uss_ids=uss_ids)
