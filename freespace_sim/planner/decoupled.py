"""Decoupled planner — fixed straight path, search the schedule (departure delay × cruise speed).

Isolates the *temporal* deconfliction story: keep the geometric path, and slot it into free time
windows by choosing a departure delay (the shared jump-to-gap search) at a few cruise speeds.
Deterministic and fast; the trade-off is that it never bends in space, so it denies where only a
spatial detour would work (that's A*'s job).
"""

from __future__ import annotations

from ..config import SimConfig
from ..ledger import ReservationLedger
from ..types import FlightRequest, OperationalIntent
from .straight import plan_timeshift


class DecoupledPlanner:
    """Fixed straight path, searched over departure delay × a few cruise speeds.

    Isolates temporal deconfliction: never bends in space, so it denies where only a spatial detour
    would work.
    """

    speed_factors: tuple[float, ...] = (1.0, 0.75, 0.5)

    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent:
        """Return the cheapest conflict-free straight schedule over the configured cruise speeds.

        Runs the shared jump-to-gap time-shift at each speed factor and keeps the lowest-cost
        ACCEPTED intent; if every speed is denied, returns the last denial.

        Parameters
        ------------
        - req (FlightRequest): the flight to plan.
        - ledger (ReservationLedger): committed reservations to deconflict against.
        - cfg (SimConfig): supplies speeds, timing, and the ground-delay budget.

        Return
        --------
        - output (OperationalIntent): the cheapest ACCEPTED intent across speed factors, or the last
          REJECTED intent when none is feasible.
        """
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
