"""Whole-schedule integration for the column-generation planner."""

from __future__ import annotations

import logging
import time

from ...config import SimConfig
from ...dss import DSS
from ...ledger import ReservationLedger
from ...scenario import Scenario
from ...types import (
    DenialReason,
    FlightRequest,
    IntentStatus,
    OperationalIntent,
    as_terminal,
)
from ...uss import _warn_if_terminal_dropped
from .params import ColGenParams
from .solver import ColGenSolver
from .translate import column_to_intent

log = logging.getLogger(__name__)


class ColumnGenerationPlanner:
    """Factory-facing marker for a planner that solves the whole schedule at once."""

    plans_terminal_airspace = True
    plans_whole_schedule = True

    def __init__(self, params: ColGenParams | None = None) -> None:
        self.params = params if params is not None else ColGenParams()

    def plan(
        self,
        req: FlightRequest,
        ledger: ReservationLedger,
        cfg: SimConfig,
    ) -> OperationalIntent:
        """Refuse the per-flight protocol; :func:`run_batch` is the execution path."""
        raise RuntimeError("colgen is a whole-schedule planner; run it through sim.run batch mode")


def run_batch(
    scenario: Scenario,
    cfg: SimConfig,
    ledger: ReservationLedger,
    dss: DSS,
    static_terms,
    status,
    report,
    collector,
    *,
    params: ColGenParams | None = None,
) -> list[OperationalIntent]:
    """Solve once, then file every result through the normal DSS in FCFS order.

    ``collector`` is accepted for parity with the parallel runner.  Column generation
    has no A* telemetry hooks, so its conflict/filed streams intentionally remain empty.
    """
    if getattr(dss, "ledger", ledger) is not ledger:
        raise ValueError("colgen batch requires dss and run_batch to share one ledger")
    if ledger.n_volumes:
        raise ValueError(
            "colgen batch requires an empty dynamic ledger; pre-existing reservations "
            "must first be represented as fixed row claims"
        )
    del collector

    events = tuple(scenario.events)
    requests = [event.request for event in events]
    if not cfg.terminal_airspace_always_active and any(
        as_terminal(request.origin_terminal) is not None
        or as_terminal(request.dest_terminal) is not None
        for request in requests
    ):
        raise NotImplementedError(
            "colgen terminal endpoints require terminal_airspace_always_active=True; "
            "without permanent walls, transient foreign terminal geometry is not represented "
            "by the batch capacity rows"
        )

    batch_params = params if params is not None else ColGenParams()
    solve_started = time.monotonic()
    result = ColGenSolver().solve(requests, cfg, static_terms, batch_params)
    solve_elapsed = time.monotonic() - solve_started
    solve_share = solve_elapsed / len(events) if events else 0.0
    stats = result.stats
    log.info(
        "colgen solver: backend=%s termination=%s iterations=%s objective_delay_s=%s "
        "lp_gap=%s ip_gap=%s selected=%s/%d search_exhausted=%s columns=%s rows=%s "
        "elapsed_s=%.3f",
        stats.get("backend", "unknown"),
        stats.get("termination_reason", "unknown"),
        stats.get("iterations", "unknown"),
        stats.get("objective", "unknown"),
        stats.get("lp_gap", "unknown"),
        stats.get("ip_gap", "unknown"),
        stats.get("selected_flights", len(result.columns)),
        len(requests),
        len(stats.get("search_exhausted_flight_ids", ())),
        stats.get("n_columns", "unknown"),
        stats.get("n_materialized_rows", "unknown"),
        solve_elapsed,
    )

    intents: list[OperationalIntent] = []
    search_exhausted = frozenset(result.stats.get("search_exhausted_flight_ids", ()))
    total = len(events)
    for done, event in enumerate(events, 1):
        request = event.request
        filing_started = time.monotonic()
        column = result.columns.get(request.flight_id)
        if column is None:
            intent = OperationalIntent(
                request=request,
                status=IntentStatus.REJECTED,
                denial_reason=(
                    DenialReason.SEARCH_EXHAUSTED
                    if request.flight_id in search_exhausted
                    else DenialReason.BUDGET_EXCEEDED
                ),
                planner="colgen",
                solve_time_s=solve_share,
            )
        else:
            intent = column_to_intent(column, request, cfg, solve_share_s=solve_share)

        _warn_if_terminal_dropped(request, intent)
        was_accepted = intent.accepted
        committed = dss.commit(intent)
        intent.solve_time_s += time.monotonic() - filing_started
        if was_accepted and not committed:
            log.error(
                "colgen filing denial -- covering bug: flight_id=%s",
                request.flight_id,
            )

        intents.append(intent)
        status(done, request, intent)
        if report:
            report(done, total, intent)

    return intents
