"""Planner protocol + factory.

Every planner takes a flight request and the live ledger and returns an `OperationalIntent` that is
ACCEPTED (with the exact `volumes` it conflict-checked) or REJECTED (no conflict-free plan within
budget). The geometry a planner checks is the geometry it commits — see
`volumes.corridor_segment_volume`.
"""

from __future__ import annotations

from typing import Protocol

from ..config import SimConfig
from ..ledger import ReservationLedger
from ..types import FlightRequest, OperationalIntent


class Planner(Protocol):
    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent: ...


class LazyPlanner:
    """Lazy escalation: deterministic straight-line time-shift first; RRT* only if that denies.

    Ground delay is owned by the cheap tier and searched exhaustively, so RRT* is invoked only when
    no pure-time solution exists and you must bend in space — the blocked minority pays its cost.
    """

    def __init__(self):
        from .rrt import SpaceTimeRRTStar
        from .straight import StraightLineTimeShift

        self.straight = StraightLineTimeShift()
        self.rrt = SpaceTimeRRTStar()

    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent:
        intent = self.straight.plan(req, ledger, cfg)
        if not intent.accepted:
            intent = self.rrt.plan(req, ledger, cfg)
        intent.planner = "lazy"
        return intent


def get_planner(name: str) -> Planner:
    """Resolve a planner by name."""
    if name == "straight":
        from .straight import StraightLineTimeShift

        return StraightLineTimeShift()
    if name == "rrt":
        from .rrt import SpaceTimeRRTStar

        return SpaceTimeRRTStar()
    if name == "decoupled":
        from .decoupled import DecoupledPlanner

        return DecoupledPlanner()
    if name == "lazy":
        return LazyPlanner()
    if name == "opt":
        from .opt import NLPOptPlanner

        return NLPOptPlanner()
    if name == "milp":
        from .milp import MILPOptPlanner

        return MILPOptPlanner()
    if name == "astar":
        from .astar import AStarPlanner

        return AStarPlanner()
    if name == "opt_astar":
        from .astar import AStarPlanner
        from .opt import NLPOptPlanner

        return NLPOptPlanner(warm_planner=AStarPlanner())   # A* homotopy + NLP continuous polish
    if name == "astar_milp":
        from .astar import AStarPlanner
        from .milp import MILPOptPlanner

        # A* picks the homotopy (which side) + the delay; the MILP is LOCKED to that homotopy and
        # tightens the geometry within it (its binaries are pinned → a fast LP, not a fresh search).
        return MILPOptPlanner(warm_planner=AStarPlanner(), optimize_delay=False, lock_homotopy=True)
    raise ValueError(f"unknown planner: {name!r}")
