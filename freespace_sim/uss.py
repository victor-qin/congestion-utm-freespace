"""UAS Service Supplier — the operator-side controller (ASTM §3.2.44).

A USS owns an identity and a planner; it handles a flight request by planning a conflict-free
reservation and submitting it to the DSS for commit. World state lives in the DSS/ledger, not here.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

from .config import SimConfig
from .dss import DSS
from .planner import Planner
from .types import FlightRequest, OperationalIntent


def _warn_if_terminal_dropped(req: FlightRequest, intent: OperationalIntent) -> None:
    """Make it *loud* when a planner ignores multi-pad terminal airspace.

    Only A*-based geometry tags a hub flight's terminal column; the refiners (shortcut/MILP) and
    the non-A* planners (straight) rebuild corridors and drop the tag, which silently disables the
    shared-terminal exemption and pad capacity. If an accepted flight asked for a terminal but its
    committed volumes don't carry it, warn — better an obvious RuntimeWarning than a wrong result.
    """
    if not intent.accepted:
        return
    expected = [t[0] for t in (req.origin_terminal, req.dest_terminal) if t is not None]
    if not expected:
        return
    have = {v.terminal_id for v in (intent.volumes or [])}
    if any(t not in have for t in expected):
        warnings.warn(
            f"planner {intent.planner!r} did not tag terminal airspace for a multi-pad hub flight: "
            "the shared-terminal exemption and pad capacity are NOT applied. Use an A*-based planner "
            "('astar') for hub scenarios.",
            RuntimeWarning, stacklevel=2,
        )


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
        _warn_if_terminal_dropped(req, intent)   # apparent failure if terminal airspace was ignored
        self.dss.commit(intent)   # ACCEPTED → committed; conflict at commit → REJECTED
        return intent
