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


def assert_no_interflight_conflict(intents: list[OperationalIntent], cfg: SimConfig,
                                   static_terminals=()) -> None:
    bad = find_interflight_conflict(intents, cfg, static_terminals=static_terminals)
    assert bad is None, f"inter-flight 4D conflict between flights {bad[0]} and {bad[1]}"
