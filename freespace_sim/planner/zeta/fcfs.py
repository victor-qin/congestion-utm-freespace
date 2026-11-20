"""File one-way requests through the standalone Zeta planner and audit the row loads."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import replace
from typing import Any

from ...config import SimConfig
from ...dss import DSS
from ...ledger import ReservationLedger
from ...mechanism import FCFSMechanism
from ...types import DenialReason, FlightRequest, OperationalIntent
from ...uss import USS
from .. import get_planner
from ..route_domain import RowIndex
from .geometry import reservation_claims
from .model import Route
from .planner import ZetaPlannerParams


def run_fcfs(
    cfg: SimConfig,
    requests: list[FlightRequest],
    terms: tuple,
    params: ZetaPlannerParams,
) -> tuple[dict[int, Route], dict[int, OperationalIntent], dict[str, Any]]:
    """File each request in simulator order and audit the resulting buffered row loads.

    Parameters
    ------------
    - cfg (SimConfig): shared physical and timing rules; the planner name is set to zeta.
    - requests (list[FlightRequest]): one-way requests with unique flight ids.
    - terms (tuple): permanent ``(center, Terminal)`` catalogue registered before filing.
    - params (ZetaPlannerParams): route domain, objective and budget for each leg.

    Return
    --------
    - output (tuple[dict[int, Route], dict[int, OperationalIntent], dict[str, Any]]):
      accepted routes, filed intents, and per-leg search and row-audit evidence. Raises
      ValueError for duplicate ids or round trips, and RuntimeError when a checked plan cannot
      be filed or its search and filed claims disagree.
    """
    started = time.monotonic()
    if len({request.flight_id for request in requests}) != len(requests):
        raise ValueError("flight ids must be unique")
    if any(request.return_to_origin for request in requests):
        raise ValueError("the Zeta comparison expects one-way requests")
    cfg = replace(cfg, planner="zeta")
    ledger = ReservationLedger(cfg)
    row_index = RowIndex()
    for center, terminal in terms:
        ledger.register_static_terminal(center, terminal)
        row_index.register_terminal(terminal)
    protected = frozenset(terminal.id for _, terminal in terms)
    dss = DSS(ledger=ledger, mechanism=FCFSMechanism())
    usses = {
        uid: USS(uid, dss, cfg, get_planner("zeta", params))
        for uid in {request.uss_id for request in requests}
    }
    routes: dict[int, Route] = {}
    intents: dict[int, OperationalIntent] = {}
    records = []
    for req in sorted(requests, key=FlightRequest.sort_key):
        uss = usses[req.uss_id]
        intent = uss.handle_request(req)
        result = uss.planner.last_result
        if intent.denial_reason is DenialReason.CONFLICT_AT_COMMIT:
            raise RuntimeError(f"Zeta checked flight {req.flight_id} but FCFS could not file it")
        if intent.accepted:
            assert result is not None and result.route is not None
            intents[req.flight_id] = intent
            routes[req.flight_id] = result.route
        records.append({
            "flight_id": req.flight_id,
            "accepted": intent.accepted,
            "denial_reason": intent.denial_reason.value,
            "reason": "graph_infeasible" if result is None else result.reason,
            "complete": result is None or result.complete,
            "route_cost_lower_bound": None if result is None else result.lower_bound,
            # Already subtracted from the lower bound; kept so "merge_uncertainty" has a size.
            "merge_error_bound": 0.0 if result is None else result.merge_error_bound,
            "expanded": 0 if result is None else result.expanded,
            "generated": 0 if result is None else result.generated,
            "search_wall_s": 0.0 if result is None else result.wall_s,
            "plan_wall_s": intent.solve_time_s,
        })
    elapsed = time.monotonic() - started
    loads: Counter = Counter()
    for flight_id, intent in intents.items():
        claims = reservation_claims(
            intent.volumes, cfg, row_index=row_index, protected_terminal_ids=protected,
        )
        if claims != routes[flight_id].claims:
            raise RuntimeError(f"Zeta search and filing claims disagree for flight {flight_id}")
        loads.update(claims)
    stats = {
        "policy": "FCFS",
        "certificate_scope": "per leg with earlier filings fixed",
        "wall_budget_s_per_leg": params.time_limit_s,
        "row_rule": "complete simulator box/cylinder coverage and terminal pad rows",
        "elapsed_s": elapsed,
        "pricing_wall_s": math.fsum(record["search_wall_s"] for record in records),
        "violated_rows": sum(load > row_index.cap(row) for row, load in loads.items()),
        "unique_rows": len(loads),
        "nonzeros": sum(loads.values()),
        "searches_complete": sum(record["complete"] for record in records),
        "search_records": records,
    }
    return routes, intents, stats
