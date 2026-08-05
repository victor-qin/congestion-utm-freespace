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

    ``max_tasks_per_child=16`` is stated rather than inherited: the 4-flight default times
    chunksize 4 is what these numbers were actually measured under, back when the two
    knobs multiplied.  Now that they no longer do, the value has to be written down to
    keep the tuning, and 16 flights per worker is a deliberate residue/respawn trade --
    a restart costs ~0.36s of import + warm_kernel and returns the arena to the OS.
    """

    if n_flights < _PARALLEL_MIN_FLIGHTS:
        return None
    workers = max(2, min(8, (os.cpu_count() or 4) - 2))
    return ParallelPricingConfig(n_workers=workers, chunksize=4, max_tasks_per_child=16)


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
    on_iteration=None,
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

    ``on_iteration`` is forwarded to the solver, which calls it once per column-generation
    iteration.  It is forwarded rather than dropped because ``run_batch`` is the production
    entry point: without this the per-iteration telemetry existed but was reachable only by
    calling :meth:`ColGenSolver.solve` directly, so every real run was blind to it.
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
        requests, cfg, static_terms, batch_params,
        parallel=parallel, on_iteration=on_iteration,
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
    # A second line rather than a longer first one: the line above is a stable format
    # that callers grep, and these are answers to different questions -- "did the fan-out
    # happen and did it pay" and "how converged is this really".  Both were unanswerable
    # from a production log before, which is how a full-scenario run could be reported as
    # parallel without any evidence in its own output that it was.
    sweep_s = stats.get("parallel_sweep_wall_s") or 0.0
    task_s = stats.get("parallel_task_wall_total_s") or 0.0
    workers = stats.get("parallel_workers") or 0
    log.info(
        "colgen detail: gap_metric=%s lp_gap_revenue=%s lp_gap_cost=%s ip_gap_revenue=%s "
        "greedy_s=%s workers=%s worker_procs=%s sweep_s=%.3f task_s=%.3f straggler_s=%.3f "
        "worker_efficiency=%s worker_peak_rss_mb=%.1f tasks_discarded=%s",
        stats.get("gap_metric", "unknown"),
        stats.get("lp_gap_revenue", "unknown"),
        # The cost-scale gap is logged beside the revenue one deliberately: with M an
        # artificial big-M the revenue gap can read as converged while this one is still
        # enormous, and only printing the first hides that entirely.
        stats.get("lp_gap_cost", "unknown"),
        stats.get("ip_gap_revenue", "unknown"),
        stats.get("initial_greedy_elapsed_s", "unknown"),
        workers,
        stats.get("parallel_worker_processes", 0),
        sweep_s,
        task_s,
        stats.get("parallel_task_wall_max_s") or 0.0,
        f"{task_s / (sweep_s * workers):.1%}" if sweep_s and workers else "n/a",
        (stats.get("parallel_worker_peak_rss_bytes") or 0) / 1e6,
        stats.get("parallel_tasks_discarded", 0),
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
