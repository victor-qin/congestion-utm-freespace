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
    """The swappable airspace-allocation policy: decide whether an intent enters the ledger.

    Implementations (FCFS today; auctions / priority / negotiation later) are the single authority
    on what commits, so swapping the policy leaves planner, ledger, and sim untouched.
    """

    def commit(self, ledger: ReservationLedger, intent: OperationalIntent) -> bool:
        """Decide whether ``intent`` may enter ``ledger``; return True iff it commits."""
        ...


class FCFSMechanism:
    """Accept the first conflict-free plan; the committed flight becomes an obstacle for later ones.

    The planner has already searched for a conflict-free reservation, but we re-check at commit time
    so the mechanism is the single authority on what enters the ledger. (In single-threaded v0 the
    re-check never fails; it's the hook for multi-USS races in a later phase.)
    """

    def commit(self, ledger: ReservationLedger, intent: OperationalIntent) -> bool:
        """Commit ``intent`` iff it is accepted and still conflict-free against the ledger.

        Re-checks conflicts at commit time (the mechanism, not the planner, is the authority). On a
        conflict the intent is marked ``REJECTED`` / ``CONFLICT_AT_COMMIT`` and nothing is written.

        Parameters
        ------------
        - ledger (ReservationLedger): the shared ledger; mutated only on a successful commit.
        - intent (OperationalIntent): the plan to commit; its status and denial_reason are set on a
          commit-time conflict.

        Return
        --------
        - output (bool): True iff the intent's volumes were written to the ledger.
        """
        if not intent.accepted or not intent.volumes:
            return False
        if ledger.any_conflict(intent.volumes):
            intent.status = IntentStatus.REJECTED
            intent.denial_reason = DenialReason.CONFLICT_AT_COMMIT
            return False
        ledger.commit(intent.request.flight_id, intent.volumes)
        return True
