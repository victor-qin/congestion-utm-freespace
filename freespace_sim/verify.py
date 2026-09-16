"""Post-run invariant — the core ASTM strategic-deconfliction property.

No two committed intents from different flights may conflict (a flight's own consecutive corridor
boxes are allowed to overlap — ASTM contiguity). We re-derive this independently of the live ledger
by replaying accepted intents in FCFS order into a fresh ledger and asserting each one is clear
against everything committed before it. That checks every inter-flight pair exactly once and will
catch any bug in a planner's build-then-check discipline.

One documented exception: volumes sharing a ``terminal_id`` (a multi-pad vertiport's shared
terminal airspace) are mutually transparent — this is enforced uniformly inside
``conflict.volumes_conflict``, which both the live ledger and this replay route through, so no
special-casing is needed here.

Under ``cfg.terminal_airspace_always_active`` the permanent terminal walls are ledger volumes
too, so passing ``static_terminals`` registers them into the replay ledger and this check now
also catches a committed corridor that crosses a walled (foreign) terminal — a property it was
structurally blind to when the walls lived off-ledger. A static-wall hit reports the partner id
as ``-1`` (there is no owning flight).
"""

from __future__ import annotations

from .config import SimConfig
from .ledger import ReservationLedger
from .types import OperationalIntent


def find_interflight_conflict(
    intents: list[OperationalIntent], cfg: SimConfig, static_terminals=()
) -> tuple[int, int] | None:
    """Return the first ``(flight_id, other_flight_id)`` pair that conflicts, or None if clean.

    Replays accepted intents in FCFS order into a fresh ledger, asserting each is clear against
    everything committed before it. A conflict with an always-active terminal wall surfaces as
    ``(flight_id, -1)`` — the ``-1`` marks a static wall, not a real partner flight (the ledger's
    documented sentinel).

    Parameters
    ------------
    - intents (list[OperationalIntent]): accepted intents to replay in FCFS order; others are
      skipped.
    - cfg (SimConfig): configures the fresh replay ledger.
    - static_terminals: ``(center, term)`` pairs filed permanently into the replay ledger before
      the intents, so a corridor crossing a walled terminal is caught too.

    Return
    --------
    - output (tuple[int, int] | None): the first conflicting ``(flight_id, other_flight_id)``, or
      None if every accepted intent is clear.
    """
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
