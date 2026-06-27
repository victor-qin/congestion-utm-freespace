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
    if name == "sipp":
        from .sipp import SIPPPlanner

        # Safe Interval Path Planning: same cost model + terminal gating + output as A*, but the air
        # search collapses the per-step axis into safe intervals (cost-aware Pareto). Drop-in for astar.
        # Compiled by default: a numba air-cruise kernel for non-terminal flights, auto-falling back to
        # the pure-Python reference (terminals, out-of-box strays, or when numba is unavailable).
        return SIPPPlanner(compiled=True)
    if name == "sipp_ref":
        from .sipp import SIPPPlanner

        return SIPPPlanner(compiled=False)   # pure-Python reference oracle (A/B + the fallback path)
    if name == "sipp_shortcut":
        from .shortcut import ShortcutRefiner
        from .sipp import SIPPPlanner

        return ShortcutRefiner(SIPPPlanner(compiled=True), label="sipp_sc")
    if name == "opt_astar":
        from .astar import AStarPlanner
        from .opt import NLPOptPlanner

        return NLPOptPlanner(warm_planner=AStarPlanner())   # A* homotopy + NLP continuous polish
    if name == "astar_milp":
        return _astar_milp()
    if name == "astar_shortcut":
        from .astar import AStarPlanner
        from .shortcut import ShortcutRefiner

        # A* → greedy shortcut: a solver-free alternative to the MILP refine (tightens the staircase
        # against the REAL committed obstacles, not A*'s conservative raster).
        return ShortcutRefiner(AStarPlanner(), label="astar_sc")
    if name == "astar_milp_shortcut":
        from .astar import AStarPlanner
        from .milp import MILPOptPlanner
        from .shortcut import ShortcutRefiner

        # The full sandwich A* → shortcut → MILP → shortcut: the PRE-shortcut tightens the warm
        # reference so the MILP locks more binaries and certifies its gap fast (often before the time
        # cap); the MILP does the optimal continuous refinement within that homotopy; the POST-shortcut
        # crosses any residual lock slack and strips the resample bloat. Tightest *and* fastest.
        milp = MILPOptPlanner(
            warm_planner=ShortcutRefiner(AStarPlanner()), optimize_delay=False, lock_homotopy=True)
        return ShortcutRefiner(milp, label="astar_milp_sc")
    raise ValueError(f"unknown planner: {name!r}")


def _astar_milp() -> Planner:
    """A* picks the homotopy (which side) + the delay; the MILP is LOCKED to that homotopy and
    tightens the geometry within it (its binaries are pinned → a fast LP, not a fresh search)."""
    from .astar import AStarPlanner
    from .milp import MILPOptPlanner

    return MILPOptPlanner(warm_planner=AStarPlanner(), optimize_delay=False, lock_homotopy=True)
