"""Whole-schedule integration for the column-generation planner."""

from __future__ import annotations

import logging
import os
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
from .pricing_pool import ParallelPricingConfig
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


# Below this many flights the pool costs more than it saves: a cold worker is ~0.36s
# (import + warm_kernel) against per-flight pricing that is often well under a second.
_PARALLEL_MIN_FLIGHTS = 32


def _default_pricing_pool(n_flights: int) -> ParallelPricingConfig | None:
    """Fan pricing across processes by default -- it is a pure performance knob.

    The sweep's subproblems are independent given the iteration's duals, and
    ``price_sweep`` reproduces the sequential loop's timeout prefix, so the accepted
    columns and the objective are unchanged.  Processes rather than threads because
    ``price_flight`` is mostly Python around the compiled kernel (GIL-free fraction
    measured at 19%), and because a worker that exits returns its arena to the OS --
    which matters when the label pool is allocated and freed per flight.

    ``chunksize=4`` amortises per-task dispatch now that the cost objective made pricing
    ~29x cheaper: measured on density_faa first-500, worker efficiency 46.7% -> 80.3% and
    sweep makespan 72.3s -> 41.6s, at roughly 2x peak memory.
    """

    if n_flights < _PARALLEL_MIN_FLIGHTS:
        return None
    workers = max(2, min(8, (os.cpu_count() or 4) - 2))
    return ParallelPricingConfig(n_workers=workers, chunksize=4)


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
    parallel=None,
) -> list[OperationalIntent]:
    """Solve once, then file every result through the normal DSS in FCFS order.

    ``collector`` is accepted for parity with the parallel runner.  Column generation
    has no A* telemetry hooks, so its conflict/filed streams intentionally remain empty.

    ``parallel`` is a :class:`~.pricing_pool.ParallelPricingConfig` fanning the per-iteration
    pricing sweep across processes.  It is a pure performance knob -- the subproblems are
    independent given the iteration's duals and the pool reproduces the sequential loop's
    timeout prefix -- so the accepted columns and the objective are unchanged.  ``None``
    keeps the sequential sweep.  Note ``sim.run``'s own ``parallel`` argument is a
    different mechanism entirely (the A* speculative runner) and rejects batch planners,
    so this is the only route by which whole-schedule planning reaches the pool.
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
    if parallel is None:
        parallel = _default_pricing_pool(len(requests))
    solve_started = time.monotonic()
    result = ColGenSolver().solve(
        requests, cfg, static_terms, batch_params, parallel=parallel
    )
    solve_elapsed = time.monotonic() - solve_started
    solve_share = solve_elapsed / len(events) if events else 0.0
    stats = result.stats
    log.info(
        "colgen solver: backend=%s termination=%s iterations=%s objective_delay_s=%s "
        "lp_gap=%s ip_gap=%s selected=%s/%d search_exhausted=%s columns=%s rows=%s "
        "stage=%s graphs=%s seeds=%s graph_s=%s seed_s=%s master_s=%s "
        "arc_nodes=%s arc_checks=%s cache_hits=%s wall_queries=%s wall_candidates=%s "
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
        stats.get("preprocessing_stage", "unknown"),
        stats.get("graphs_built", "unknown"),
        stats.get("seeds_completed", "unknown"),
        stats.get("graph_build_elapsed_s", "unknown"),
        stats.get("seed_elapsed_s", "unknown"),
        stats.get("time_to_master_s", "unknown"),
        stats.get("arc_expanded_nodes", "unknown"),
        stats.get("arc_checks", "unknown"),
        stats.get("arc_cache_hits", "unknown"),
        stats.get("wall_index_queries", "unknown"),
        stats.get("wall_index_candidates", "unknown"),
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
