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


#: Planners that solve every flight at once instead of one-at-a-time under FCFS, so `sim.run` routes
#: them to `colgen.run_batch` and they never enter the per-flight loop (nor its round-trip coupling).
#: A NAME list for callers that only hold `cfg.planner` and want to fail early with a readable
#: message; `sim.run` itself tests the authoritative `plans_whole_schedule` marker on the built
#: instance, so a new batch planner missing from here still gets refused — just less prettily.
WHOLE_SCHEDULE_PLANNERS = ("colgen",)


def get_planner(name: str, params=None) -> Planner:
    """Resolve a planner by name.

    ``params`` is a planner-specific configuration object. Only ``colgen`` accepts one today
    (a :class:`~.colgen.ColGenParams`), so passing one for any other planner raises rather than
    being silently dropped — a dropped solver budget looks like a converged run, not an error.
    """
    if params is not None and name != "colgen":
        raise ValueError(f"planner {name!r} takes no params object (got {type(params).__name__})")
    if name == "straight":
        from .straight import StraightLineTimeShift

        return StraightLineTimeShift()
    if name == "decoupled":
        from .decoupled import DecoupledPlanner

        return DecoupledPlanner()
    if name == "milp":
        from .milp import MILPOptPlanner

        return MILPOptPlanner()
    if name == "astar":
        from .astar import AStarPlanner

        return AStarPlanner(compiled=True)               # numba kernel + auto-fallback to the reference
    if name == "astar_ref":
        from .astar import AStarPlanner

        return AStarPlanner(compiled=False)              # pure-Python reference oracle (A/B + fallback)
    if name == "astar_milp":
        return _astar_milp()
    if name == "astar_shortcut":
        from .astar import AStarPlanner
        from .shortcut import ShortcutRefiner

        # A* → greedy shortcut: a solver-free alternative to the MILP refine (tightens the staircase
        # against the REAL committed obstacles, not A*'s conservative raster).
        return ShortcutRefiner(AStarPlanner(), label="astar_sc")
    if name == "astar_heading_shortcut":
        from .astar import AStarPlanner
        from .shortcut import ShortcutRefiner

        # Byte-equivalent A/B arm for the legacy ordering. Same-heading knots bypass their candidate
        # rebuild only when the reservation subsegment partition is exactly unchanged. Keep the legacy
        # label intentionally: accepted intents should differ only in solve_time, never in plan bytes.
        return ShortcutRefiner(
            AStarPlanner(), label="astar_sc", strategy="single_knot_heading")
    if name == "astar_batched_shortcut":
        from .astar import AStarPlanner
        from .shortcut import ShortcutRefiner

        # Experimental A/B arm: seed shortcuts at genuine 3D turns, then batch the maximal straight
        # runs on either side. The established astar_shortcut remains unchanged for comparison.
        return ShortcutRefiner(
            AStarPlanner(), label="astar_batched_sc", strategy="batched_turns")
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
    if name == "colgen":
        from .colgen import ColumnGenerationPlanner

        # Whole-schedule, not per-flight: `plan` refuses the FCFS protocol and `sim.run`
        # routes this planner to `colgen.run_batch` instead. See `plans_whole_schedule`.
        return ColumnGenerationPlanner(params)
    raise ValueError(f"unknown planner: {name!r}")


def _astar_milp() -> Planner:
    """A* picks the homotopy (which side) + the delay; the MILP is LOCKED to that homotopy and
    tightens the geometry within it (its binaries are pinned → a fast LP, not a fresh search)."""
    from .astar import AStarPlanner
    from .milp import MILPOptPlanner

    return MILPOptPlanner(warm_planner=AStarPlanner(), optimize_delay=False, lock_homotopy=True)
