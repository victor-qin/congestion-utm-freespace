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


def _log_iteration(state: dict) -> None:
    """One INFO line per column-generation iteration.

    Both gap scales, because which one is the gate is a parameter and the other is the
    honest magnitude; `columns_added` because a sweep that stops adding columns is the
    signal that the pool has converged, whatever the bound says.
    """

    log.info(
        "colgen iter %s: lp=%.6g gap_revenue=%.3g gap_cost=%.3g columns=%s (+%s) "
        "rc_sum=%.6g rc_n+=%s sweep_s=%.1f",
        state.get("iteration", "?"),
        state.get("lp_objective", float("nan")),
        state.get("lp_gap_revenue", float("nan")),
        state.get("lp_gap_cost", float("nan")),
        state.get("columns", "?"),
        state.get("columns_added", "?"),
        state.get("rc_sum", float("nan")),
        state.get("rc_n_positive", "?"),
        state.get("sweep_s", float("nan")),
    )


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
    on_iteration=None,
) -> tuple[list[OperationalIntent], dict]:
    """Solve once, then file every result through the normal DSS in FCFS order.

    Returns ``(intents, stats)``.  The stats are returned rather than only logged because
    the intents alone cannot answer "did this solve converge" -- a run that stopped at
    iteration 1 files a complete, feasible, ordinary-looking accepted set.  ``sim.run``
    carries them onto :class:`~freespace_sim.sim.SimResult` so they reach the run folder.

    ``collector`` is accepted for parity with the parallel runner.  Column generation
    has no A* telemetry hooks, so its conflict/filed streams intentionally remain empty.

    Pricing runs sequentially.  ``sim.run``'s own ``parallel`` argument is a different
    mechanism entirely -- the A* speculative runner -- and rejects whole-schedule
    planners, so a colgen run is single-process by construction.

    ``on_iteration`` is forwarded to the solver, which calls it once per column-generation
    iteration.  A caller that supplies nothing gets :func:`_log_iteration`, because pricing
    is minutes per sweep at scenario scale and the alternative is a production entry point
    that prints its banner and then says nothing for the rest of the solve.
    """
    if cfg.n_levels != 1:
        # Also guarded inside `build_flight_graph`, but that fires per flight from four frames
        # down. Selecting a planner is a whole-run decision, so refuse it here where the message
        # can name the knob -- the shipped default ladder is three levels, which makes this the
        # first thing anyone pointing colgen at an existing scenario hits.
        raise NotImplementedError(
            f"colgen v1 plans on a single flight level, but this run has {cfg.n_levels} "
            "(flight_levels_m). Pin the scenario to one level (e.g. flight_levels_m=(100.0,)) "
            "or choose a multi-level planner"
        )
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
    result = ColGenSolver().solve(
        requests, cfg, static_terms, batch_params,
        on_iteration=on_iteration if on_iteration is not None else _log_iteration,
    )
    solve_elapsed = time.monotonic() - solve_started
    solve_share = solve_elapsed / len(events) if events else 0.0
    stats = result.stats
    log.info(
        "colgen solver: backend=%s termination=%s iterations=%s objective=%s:%s "
        "lp_gap=%s ip_gap=%s selected=%s/%d search_exhausted=%s columns=%s rows=%s "
        "stage=%s graphs=%s seeds=%s graph_s=%s seed_s=%s master_s=%s "
        "arc_nodes=%s arc_checks=%s cache_hits=%s wall_queries=%s wall_candidates=%s "
        "elapsed_s=%.3f",
        stats.get("backend", "unknown"),
        stats.get("termination_reason", "unknown"),
        stats.get("iterations", "unknown"),
        # Named, because `objective` is in the cost model's currency: at
        # objective="total_cost" it is weighted cost units, not seconds.
        stats.get("objective_name", "unknown"),
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
    # that callers grep, and these are answers to a different question -- "how converged
    # is this really, and where did the wall go".  Both were unanswerable from a
    # production log before.
    log.info(
        "colgen detail: gap_metric=%s lp_gap_revenue=%s lp_gap_cost=%s ip_gap_revenue=%s "
        "greedy_s=%s pricing_wall_s=%s",
        stats.get("gap_metric", "unknown"),
        stats.get("lp_gap_revenue", "unknown"),
        # The cost-scale gap is logged beside the revenue one deliberately: with M an
        # artificial big-M the revenue gap can read as converged while this one is still
        # enormous, and only printing the first hides that entirely.
        stats.get("lp_gap_cost", "unknown"),
        stats.get("ip_gap_revenue", "unknown"),
        stats.get("initial_greedy_elapsed_s", "unknown"),
        stats.get("pricing_wall_s", "unknown"),
    )
    # A budget-terminated solve still returns a full, feasible schedule -- so it looks
    # exactly like a converged one in the results, and nothing downstream can tell them
    # apart.  Say it once, loudly, at the point the run can still be re-launched with a
    # bigger budget.  Pricing is the whole cost, so on the reference DP this is the
    # ordinary outcome at scenario scale rather than an exotic failure.
    if stats.get("termination_reason") == "time_limit":
        log.warning(
            "colgen stopped on its time limit (%.0fs) after %s iteration(s) -- this is the "
            "best schedule found within the budget, NOT a converged column-generation "
            "solution. Raise --colgen-time-limit to converge.",
            batch_params.time_limit_s,
            stats.get("iterations", "?"),
        )
    # The third truncated exit, and until now the silent one.  `solver` treats it exactly
    # like the branch above -- every denial becomes SEARCH_EXHAUSTED and the schedule is
    # uncertified -- and the shipped cap of 30 makes it reachable, so leaving it
    # unannounced is the same failure that branch exists to prevent.
    elif stats.get("termination_reason") == "iteration_limit":
        log.warning(
            "colgen stopped at its iteration cap (%s) -- this is the best schedule found "
            "within that many column-generation rounds, NOT a converged solution. Raise "
            "--colgen-max-iterations to continue.",
            batch_params.max_iterations,
        )
    # A different fact with a different remedy, which is why it is not folded into either
    # message above: the generation loop finished, and only the final integer master failed
    # to PROVE its selection optimal over the pool it was handed.  The schedule is feasible
    # and may well be optimal; it is simply uncertified, so no absent flight can be reported
    # as a physical denial.  Note the backend is asked for a much tighter gap than `ip_gap`
    # names -- it is converted to the master's revenue scale by dividing by n*M -- so at a
    # large M this is the expected outcome rather than a rare one.
    elif stats.get("termination_reason") == "ip_not_proven":
        log.warning(
            "colgen's generation loop converged (%s iterations) but the final integer "
            "master returned status=%s without proving optimality over its %s columns. "
            "The schedule is feasible; its optimality is NOT certified, and every denial "
            "is reported as search-exhausted rather than infeasible. Raising "
            "--colgen-time-limit gives the IP more room; loosening ip_gap (currently %g, "
            "handed to the backend divided by n*M) asks it to prove less.",
            stats.get("iterations", "?"),
            stats.get("ip_status", "unknown"),
            stats.get("n_columns", "?"),
            batch_params.ip_gap,
        )
    # The two gap scales can disagree by orders of magnitude, and only one of them is the
    # gate.  Under `gap_metric="revenue"` the denominator carries n*M, so with M an
    # artificial big-M the gate can close on a pool that is barely past the greedy start --
    # measured on colgen_test: Gurobi's duals close it at ITERATION 1 where HiGHS's, on the
    # identical problem, leave it at 0.194.  Both bounds are valid; they are different
    # optimal dual vertices of a degenerate master, and the gate keys on how tight the one
    # the backend happened to return is.  A converged LP also says nothing about the
    # integrality gap over the pool it converged on.  So when the two scales disagree, say
    # so rather than let "lp_gap" read as "solved".
    lp_gap_cost = stats.get("lp_gap_cost")
    if (
        stats.get("termination_reason") in {"lp_gap", "heuristic_gap"}
        and batch_params.gap_metric == "revenue"
        and isinstance(lp_gap_cost, float)
        and lp_gap_cost > batch_params.lp_gap
    ):
        log.warning(
            "colgen stopped on the revenue-scale %s at iteration %s, but the cost-scale LP "
            "gap is still %.3g (threshold %g) -- the revenue scale is normalised by an "
            "objective containing n*M (M=%g), so it closes early on a degenerate master. "
            "Re-run with --colgen-gap-metric cost to price against total cost instead.",
            stats.get("termination_reason"),
            stats.get("iterations", "?"),
            lp_gap_cost,
            batch_params.lp_gap,
            batch_params.M,
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

    return intents, stats
