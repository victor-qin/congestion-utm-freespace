"""Discovery & Synchronization Service — the shared registry (ASTM §3.2.17).

In this single-region, single-thread model the DSS simply owns the one ledger and the commit
mechanism. It exists as its own object so the USS/DSS split mirrors ASTM (and the sibling project),
which matters once multiple USSs coordinate through it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ledger import ReservationLedger
from .mechanism import Mechanism
from .types import OperationalIntent


@dataclass
class DSS:
    ledger: ReservationLedger
    mechanism: Mechanism

    def commit(self, intent: OperationalIntent) -> bool:
        """Run the commit policy against the shared ledger."""
        return self.mechanism.commit(self.ledger, intent)
