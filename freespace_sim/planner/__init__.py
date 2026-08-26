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

    # Optional, duck-typed members a planner MAY also expose (the house pattern for markers, as with
    # ``plans_whole_schedule`` / ``plans_terminal_airspace``) — deliberately not declared above, so
    # planners that have no use for them stay structurally conformant:
    #
    #   capacity_authority(ledger) -> TerminalCapacity | None
    #       The pad-capacity authority this planner has ALREADY brought current for ``ledger``, or
    #       None if it holds none bound to that ledger. Lets a post-pass reuse the authority the
    #       inner plan just built instead of paying a second ledger subscription + index. A*
    #       and the MILP implement it; ``shortcut._terminal_capacity_for`` is the consumer.


def iter_planner_chain(planner):
    """Every planner reachable from ``planner`` through the ``inner``/``warm_planner`` wrapper chain,
    ``planner`` itself first. Deduped by identity, so a diamond (``astar_milp_shortcut`` wraps a MILP
    that is warm-started by a *different* ShortcutRefiner) visits each node once.

    ONE definition of "walk the wrapper chain": ``sim`` uses it to find where to attach telemetry and
    whether the committed corridor is wall-aware, ``parallel`` to reach the A* instances inside a
    worker's planner, and ``shortcut`` to find a capacity authority. Those four had drifted into four
    identical copies (``parallel``'s even documented itself as one), which is a silent-divergence
    risk: add a fifth wrapper attribute, miss one copy, and that caller quietly sees no planners
    rather than raising. Order is load-bearing — ``_terminal_capacity_for`` returns the FIRST match —
    so this reproduces the copies exactly: LIFO, ``warm_planner`` visited before ``inner``.
    """
    seen: set[int] = set()
    stack = [planner]
    while stack:
        p = stack.pop()
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        yield p
        stack.extend((getattr(p, "inner", None), getattr(p, "warm_planner", None)))


#: Planners that solve every flight at once, so `sim.run` routes them to `colgen.run_batch` and they
#: never enter the per-flight FCFS loop (nor its round-trip coupling). Names, for callers holding only
#: `cfg.planner` that want to fail early with a readable message; `sim.run` tests the authoritative
#: `plans_whole_schedule` marker, so a batch planner missing here is still refused, just less prettily.
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
    if name == "sipp":
        from .sipp import SIPPPlanner

        # Cost-aware Safe Interval Path Planning: same cost model, terminal gating and output contract as
        # A*, but the air search collapses the per-step axis into safe intervals (Pareto over
        # (arrival, cost)). Compiled by default, auto-falling back to A* when the kernel bails.
        return SIPPPlanner(compiled=True)
    if name == "sipp_ref":
        from .sipp import SIPPPlanner

        return SIPPPlanner(compiled=False)               # pure-Python SIPP oracle (A/B + the fallback path)
    if name == "sipp_shortcut":
        from .shortcut import ShortcutRefiner
        from .sipp import SIPPPlanner

        return ShortcutRefiner(SIPPPlanner(compiled=True), label="sipp_sc")
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
