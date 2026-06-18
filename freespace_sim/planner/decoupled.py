"""Decoupled planner — fixed straight path, search the schedule (departure delay × cruise speed).

Isolates the *temporal* deconfliction story: keep the geometric path, and slot it into free time
windows by choosing a departure delay (the shared jump-to-gap search) at a few cruise speeds.
Deterministic and fast; the trade-off is that it never bends in space, so it denies where only a
spatial detour would work (that's RRT*'s job).
"""

from __future__ import annotations

from ..config import SimConfig
from ..ledger import ReservationLedger
from ..types import FlightRequest, OperationalIntent
from .straight import plan_timeshift


class DecoupledPlanner:
    speed_factors: tuple[float, ...] = (1.0, 0.75, 0.5)

    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent:
        best: OperationalIntent | None = None
        denied: OperationalIntent | None = None
        for sf in self.speed_factors:
            intent = plan_timeshift(req, ledger, cfg, speed_factor=sf, planner_name="decoupled")
            if intent.accepted:
                if best is None or intent.cost < best.cost:
                    best = intent
            else:
                denied = intent
        return best if best is not None else denied
