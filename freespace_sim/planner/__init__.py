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
    """Yield every planner reachable through the ``inner``/``warm_planner`` wrapper chain.

    One definition of "walk the wrapper chain", shared by ``sim`` (attach telemetry, test whether
    the committed corridor is wall-aware), ``parallel`` (reach the A* instances inside a worker's
    planner), and ``shortcut`` (find a capacity authority). Centralised so a newly added wrapper
    attribute cannot be missed in one copy and make that caller silently see no planners. Order is
    load-bearing — ``_terminal_capacity_for`` returns the FIRST match — so the walk is LIFO, with
    ``warm_planner`` visited before ``inner``.

    Parameters
    ------------
    - planner (Planner): the head of the wrapper chain; yielded first.

    Return
    --------
    - output (Iterator[Planner]): each reachable planner once, deduped by identity, so a diamond
      (``astar_milp_shortcut`` wraps a MILP warm-started by a *different* ShortcutRefiner) visits
      each node once.
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


def uses_hex_lattice(name: str) -> bool:
    """Whether a planner registry name searches the shared axial hex lattice.

    Keep this classification next to :func:`get_planner` so metrics and replay integrations do not
    each grow their own incomplete string check when a lattice-planner family is registered.
    """
    return any(name == family or name.startswith(f"{family}_")
               for family in ("astar", "sipp", "colgen"))


def get_planner(name: str, params=None) -> Planner:
    """Resolve a planner registry name to a fresh planner instance, wrapped for round trips.

    Only ``colgen`` accepts a ``params`` object today, so passing one for any other name raises
    rather than being silently dropped — a dropped solver budget looks like a converged run, not an
    error.

    Every per-flight planner is wrapped in :class:`~.itinerary.ItineraryPlanner`, which composes a
    ``return_to_origin`` request into one intent covering both legs and the pad between them. The
    wrapper forwards one-way requests untouched, so this is invisible to every flight that is not a
    round trip. Whole-schedule planners are NOT wrapped: they receive the schedule, not a request.
    Reach an inner planner through :func:`iter_planner_chain`, never by assuming a wrapper depth.

    Parameters
    ------------
    - name (str): registry name (e.g. ``"astar"``, ``"astar_shortcut"``, ``"colgen"``).
    - params: planner config (a :class:`~.colgen.ColGenParams` for ``colgen``), else ``None``.

    Return
    --------
    - output (Planner): the resolved planner; raises ``ValueError`` on an unknown name or on a
      ``params`` object passed for a non-``colgen`` planner.
    """
    inner = _get_planner(name, params)
    if name in WHOLE_SCHEDULE_PLANNERS:
        return inner
    from .itinerary import ItineraryPlanner

    return ItineraryPlanner(inner)


def _get_planner(name: str, params=None) -> Planner:
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
