"""Post-run invariant — the core ASTM strategic-deconfliction property.

No two committed intents from *different* flights may conflict (a flight's own consecutive corridor
boxes are allowed to overlap — ASTM contiguity). We re-derive this independently of the live ledger
by replaying accepted intents in FCFS order into a fresh ledger and asserting each one is clear
against everything committed before it. That checks every inter-flight pair exactly once and will
catch any bug in a planner's build-then-check discipline.

One documented exception: volumes sharing a ``terminal_id`` (a multi-pad vertiport's shared terminal
airspace) are mutually transparent — this is enforced uniformly inside ``conflict.volumes_conflict``,
which both the live ledger and this replay route through, so no special-casing is needed here.
"""

from __future__ import annotations

from .config import SimConfig
from .ledger import ReservationLedger
from .types import OperationalIntent


def find_interflight_conflict(
    intents: list[OperationalIntent], cfg: SimConfig
) -> tuple[int, int] | None:
    """Return the first ``(flight_id, other_flight_id)`` pair that conflicts, or None if clean."""
    led = ReservationLedger(cfg)
    for intent in intents:
        if not intent.accepted or not intent.volumes:
            continue
        hits = led.conflicts(intent.volumes)
        if hits:
            return (intent.request.flight_id, hits[0][0])
        led.commit(intent.request.flight_id, intent.volumes)
    return None


def assert_no_interflight_conflict(intents: list[OperationalIntent], cfg: SimConfig) -> None:
    bad = find_interflight_conflict(intents, cfg)
    assert bad is None, f"inter-flight 4D conflict between flights {bad[0]} and {bad[1]}"
