"""Post-run invariant — the core ASTM strategic-deconfliction property.

No two committed intents from *different* flights may conflict (a flight's own consecutive corridor
boxes are allowed to overlap — ASTM contiguity). We re-derive this independently of the live ledger
by replaying accepted intents in FCFS order into a fresh ledger and asserting each one is clear
against everything committed before it. That checks every inter-flight pair exactly once and will
catch any bug in a planner's build-then-check discipline.

One documented exception: volumes sharing a ``terminal_id`` (a multi-pad vertiport's shared terminal
airspace) are mutually transparent — this is enforced uniformly inside ``conflict.volumes_conflict``,
which both the live ledger and this replay route through, so no special-casing is needed here.

Under ``cfg.terminal_airspace_always_active`` the permanent terminal walls are ledger volumes too, so
passing ``static_terminals`` registers them into the replay ledger and this check now also catches a
committed corridor that crosses a walled (foreign) terminal — a property it was structurally blind to when
the walls lived off-ledger. A static-wall hit reports the partner id as ``-1`` (there is no owning flight).
"""

from __future__ import annotations

from .config import SimConfig
from .ledger import ReservationLedger
from .types import OperationalIntent


def find_interflight_conflict(
    intents: list[OperationalIntent], cfg: SimConfig, static_terminals=()
) -> tuple[int, int] | None:
    """Return the first ``(flight_id, other_flight_id)`` pair that conflicts, or None if clean. A conflict
    with an always-active terminal wall (``static_terminals``: ``(center, term)`` pairs, filed permanently
    into the replay ledger before the intents) surfaces as ``(flight_id, -1)`` — the ``-1`` marks a static
    wall, not a real partner flight (the ledger's documented sentinel)."""
    led = ReservationLedger(cfg)
    for center, term in static_terminals:
        led.register_static_terminal(center, term)
    for intent in intents:
        if not intent.accepted or not intent.volumes:
            continue
        hits = led.conflicts(intent.volumes)
        if hits:
            return (intent.request.flight_id, hits[0][0])
        led.commit(intent.request.flight_id, intent.volumes)
    return None


def realized_takeoff_s(intent: OperationalIntent) -> float | None:
    """When an accepted flight's aircraft actually leaves the ground — the takeoff column's
    ``t_start``, hence the smallest time the intent holds. ``None`` if nothing flew.

    The mirror of ``sim.realized_release_s`` (the landing column's ``t_end``). NOT
    ``centerline[0][1]``, for the same reason that one is not ``centerline[-1]``: under fixed exit
    lanes the corridor begins at the column's EDGE, so the first waypoint FOLLOWS liftoff by the
    climb-plus-egress dwell, and measuring there would understate the hold by exactly the window
    this check is about.
    """
    if not intent.accepted or not intent.volumes:
        return None
    return float(min(v.t_start for v in intent.volumes))


def find_paired_precedence_violation(
    intents: list[OperationalIntent], cfg: SimConfig, turnaround_s: float = 0.0, tol: float = 1e-6
) -> tuple[int, int, float] | None:
    """First ``(return_id, outbound_id, shortfall_s)`` where a paired return departs before its own
    aircraft is back on the pad, or None if every round trip is flyable.

    **This is not a conflict, which is why nothing else catches it.** The two legs of a round trip
    are separate flights with independently timed reservations, and a return that lifts off before
    its outbound lands simply holds a *disjoint* window at the same pad — no 4D overlap, so the
    ledger accepts it and :func:`find_interflight_conflict` reports the schedule clean. What is
    violated is precedence, not separation: one aircraft cannot depart a place it has not reached.

    It is reachable because ``t_departure`` for a paired return is a NOMINAL estimate fixed at
    demand-generation time (``t_dep_outbound + est_trip_s + turnaround_s``). When the outbound's real
    trip runs longer than the estimate — any ground hold or detour — the return's filed departure
    precedes the real arrival and, under ``return_anchor="nominal"``, nothing re-anchors it.
    ``return_anchor="realized"`` re-times each return off its outbound's measured release and is the
    supported fix; this check is what makes its absence visible.

    ``turnaround_s`` is the ground time the demand model budgeted between the legs — the same value
    the LNS paired-return guard uses, so the two cannot disagree about what "available" means.
    """
    from .sim import realized_release_s          # sim imports THIS module; keep one owner of the

    by = {i.request.flight_id: i for i in intents}       # release definition rather than a copy
    for intent in intents:
        outbound_id = intent.request.paired_outbound_id
        if outbound_id is None or not intent.accepted:
            continue
        outbound = by.get(outbound_id)
        if outbound is None or not outbound.accepted:
            continue                                     # denied outbound: no aircraft to wait for
        release = realized_release_s(outbound)
        takeoff = realized_takeoff_s(intent)
        if release is None or takeoff is None:
            continue
        available = release + turnaround_s
        if takeoff < available - tol:
            return (intent.request.flight_id, outbound_id, available - takeoff)
    return None


def count_paired_precedence_violations(
    intents: list[OperationalIntent], cfg: SimConfig, turnaround_s: float = 0.0, tol: float = 1e-6
) -> tuple[int, float]:
    """``(n_violations, total_shortfall_s)`` — the reporting counterpart of
    :func:`find_paired_precedence_violation`, for callers that want the scale rather than an example."""
    from .sim import realized_release_s

    by = {i.request.flight_id: i for i in intents}
    n, total = 0, 0.0
    for intent in intents:
        outbound_id = intent.request.paired_outbound_id
        if outbound_id is None or not intent.accepted:
            continue
        outbound = by.get(outbound_id)
        if outbound is None or not outbound.accepted:
            continue
        release = realized_release_s(outbound)
        takeoff = realized_takeoff_s(intent)
        if release is None or takeoff is None:
            continue
        short = release + turnaround_s - takeoff
        if short > tol:
            n += 1
            total += short
    return n, total


def assert_no_paired_precedence_violation(intents: list[OperationalIntent], cfg: SimConfig,
                                          turnaround_s: float = 0.0) -> None:
    bad = find_paired_precedence_violation(intents, cfg, turnaround_s=turnaround_s)
    if bad is not None:
        ret, out, short = bad
        raise AssertionError(
            f"paired return {ret} departs {short:.1f}s before outbound {out} releases its pad — the "
            f"aircraft has not arrived. Use return_anchor='realized' to re-anchor returns off the "
            f"measured arrival.")


def assert_no_interflight_conflict(intents: list[OperationalIntent], cfg: SimConfig,
                                   static_terminals=()) -> None:
    bad = find_interflight_conflict(intents, cfg, static_terminals=static_terminals)
    assert bad is None, f"inter-flight 4D conflict between flights {bad[0]} and {bad[1]}"
