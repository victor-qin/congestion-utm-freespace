"""UAS Service Supplier — the operator-side controller (ASTM §3.2.44).

A USS owns an identity and a planner; it handles a flight request by planning a conflict-free
reservation and submitting it to the DSS for commit. World state lives in the DSS/ledger, not here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import SimConfig
from .dss import DSS
from .planner import Planner
from .types import FlightRequest, OperationalIntent


@dataclass
class USS:
    uss_id: str
    dss: DSS
    cfg: SimConfig
    planner: Planner

    def handle_request(self, req: FlightRequest) -> OperationalIntent:
        t0 = time.monotonic()
        intent = self.planner.plan(req, self.dss.ledger, self.cfg)
        intent.solve_time_s = time.monotonic() - t0   # planner time only, before commit
        self.dss.commit(intent)   # ACCEPTED → committed; conflict at commit → REJECTED
        return intent
