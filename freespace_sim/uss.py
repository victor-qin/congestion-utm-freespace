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
    """Warn loudly when an accepted plan dropped the terminal tag it asked for.

    A*-based geometry and the terminal-aware MILP family tag a hub flight's terminal column; a
    planner that rebuilds corridors without threading the terminal (e.g. ``straight``) drops the
    tag, silently disabling the shared-terminal exemption and pad capacity. A visible
    ``RuntimeWarning`` is preferred to a quietly wrong result.

    Parameters
    ------------
    - req (FlightRequest): the request, whose ``origin_terminal`` / ``dest_terminal`` name any
      terminals the committed volumes must carry.
    - intent (OperationalIntent): the planned intent; only accepted intents are checked.

    Return
    --------
    - output (None): emits a ``RuntimeWarning`` when an expected terminal tag is missing; returns
      nothing and mutates nothing.
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
    """UAS Service Supplier: owns an identity and a planner, and submits plans to the DSS.

    World state lives in the DSS/ledger, not here; a USS only plans a conflict-free reservation for
    a request and hands it to the DSS to commit.
    """

    uss_id: str
    dss: DSS
    cfg: SimConfig
    planner: Planner

    def handle_request(self, req: FlightRequest) -> OperationalIntent:
        """Plan a conflict-free reservation for ``req`` and submit it to the DSS to commit.

        Records planner wall time on the intent (``solve_time_s``, before commit) and warns if the
        planner dropped requested terminal airspace. The DSS commit flips an accepted intent to
        committed, or to rejected on a commit-time conflict.

        Parameters
        ------------
        - req (FlightRequest): the flight request to plan and commit.

        Return
        --------
        - output (OperationalIntent): the planned intent, with ``solve_time_s`` set and its status
          resolved by the DSS commit.
        """
        t0 = time.monotonic()
        intent = self.planner.plan(req, self.dss.ledger, self.cfg)
        intent.solve_time_s = time.monotonic() - t0   # planner time only, before commit
        _warn_if_terminal_dropped(req, intent)   # apparent failure if terminal airspace was ignored
        self.dss.commit(intent)   # ACCEPTED → committed; conflict at commit → REJECTED
        return intent
