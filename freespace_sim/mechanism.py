"""Commit policy — the swappable airspace-allocation mechanism.

v0 is FCFS (ASTM §4.2.5: "the first-planned operation is given priority"). The `Mechanism`
protocol leaves room for auctions / priority / negotiation later without touching planner, ledger,
or sim — exactly as the sibling project structured it.
"""

from __future__ import annotations

from typing import Protocol

from .ledger import ReservationLedger
from .types import DenialReason, IntentStatus, OperationalIntent


class Mechanism(Protocol):
    def commit(self, ledger: ReservationLedger, intent: OperationalIntent) -> bool: ...


class FCFSMechanism:
    """Accept the first conflict-free plan; the committed flight becomes an obstacle for later ones.

    The planner has already searched for a conflict-free reservation, but we re-check at commit time
    so the mechanism is the *single* authority on what enters the ledger. (In single-threaded v0 the
    re-check never fails; it's the hook for multi-USS races in a later phase.)
    """

    def commit(self, ledger: ReservationLedger, intent: OperationalIntent) -> bool:
        if not intent.accepted or not intent.volumes:
            return False
        if ledger.any_conflict(intent.volumes):
            intent.status = IntentStatus.REJECTED
            intent.denial_reason = DenialReason.CONFLICT_AT_COMMIT
            return False
        ledger.commit(intent.request.flight_id, intent.volumes)
        return True
