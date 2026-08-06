"""Whole-schedule column-generation orchestration.

This module deliberately contains no capacity-row geometry and no optimization
backend details.  It wires the three owners together instead: ``network``
certifies canonical claims, ``pricing`` searches one flight's finite DAG, and
``master`` solves the restricted master problem.  Keeping that boundary sharp
also makes the final repair pass use exactly the same column universe and row
semantics as the LP/IP loop.
"""

from __future__ import annotations

import collections
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...config import SimConfig
from ...types import FlightRequest, as_terminal
from .master import BackendTimeout, RestrictedMaster
from .objective import DELAY_MODEL, CostModel, cost_model
from .network import (
    FlightGraph,
    RowIndex,
    RowKey,
    StaticTerminalCatalog,
    build_flight_graph,
    column_claims,
)
from .params import ColGenParams
from .pricing import (
    DualView,
    PricingTimeout,
    find_feasible_column,
    price_flight,
    seed_column,
)
from .translate import Column

_FEASIBILITY_TOL = 1e-7
_REDUCED_COST_TOL = 1e-9
_OBJECTIVE_TOL = 1e-8


@dataclass(frozen=True, slots=True)
class ColGenResult:
    """Selected columns and diagnostics for one whole-schedule solve.

    A flight absent from :attr:`columns` is either an exhaustive optimizer
    denial or a compute-cap artifact.  ``budget_denied_flight_ids`` and
    ``search_exhausted_flight_ids`` partition those cases for ``batch.run_batch``,
    which turns the distinction into a :class:`DenialReason`.
    """

    columns: dict[int, Column]
    stats: dict[str, Any]


def _fixed_loads(fixed_claims: Sequence[frozenset[RowKey]]) -> dict[RowKey, int]:
    """Aggregate already committed claims without losing repeated occupancy."""

    loads: Counter[RowKey] = Counter()
    for claim_set in fixed_claims:
        for raw_key in frozenset(claim_set):
            key = raw_key if isinstance(raw_key, RowKey) else RowKey(raw_key)
            loads[key] += 1
    return dict(loads)


def _coverage_diagnostics(master, x, rc_by_flight, benefit) -> dict:
    """Cross-reference LP coverage against the flights whose reduced cost is ~M.

    A flight the LP leaves uncovered has a slack cover constraint, so complementary
    slackness sets its dual to zero and pricing reports ``rc = M - cost - dual_cost``,
    i.e. roughly the whole benefit.  Those terms then dominate the Lagrangian bound
    ``LP + sum(max(0, rc))``.  This measures the overlap directly rather than inferring
    it, and reports the largest column cost -- the quantity that actually bounds how
    small ``M`` is allowed to be.
    """

    coverage: dict[int, float] = collections.defaultdict(float)
    max_cost = 0.0
    for value, column in zip(np.asarray(x, dtype=float), master.columns, strict=False):
        coverage[column.flight_id] += float(value)
        if column.delay_s > max_cost:
            max_cost = float(column.delay_s)
    uncovered = {f for f, v in coverage.items() if v < 1.0 - 1e-6}
    contaminated = {f for f, rc in rc_by_flight.items() if rc > 0.5 * float(benefit)}
    return {
        "max_column_cost": max_cost,
        "n_uncovered": len(uncovered),
        "n_rc_near_M": len(contaminated),
        "n_overlap": len(uncovered & contaminated),
    }


def _canonical_column(column: Column, graph: FlightGraph, cfg: SimConfig) -> Column:
    """Re-run the one authoritative geometry/claim gate for a solver column."""

    claims = column_claims(column, graph, cfg)
    if column.claims == claims:
        return column
    return replace(column, claims=claims)


def _shift_claims(claims: Sequence[RowKey] | frozenset[RowKey], steps: int) -> frozenset[RowKey]:
    """Translate every canonical capacity claim by whole lattice periods."""

    if steps == 0:
        return frozenset(claims)
    shifted: set[RowKey] = set()
    for row in claims:
        if row.kind == "cell":
            q, r = row.cell_coord
            shifted.add(RowKey.cell(q, r, row.level, row.step + steps))
        else:
            shifted.add(RowKey.term(row.terminal_id, row.step + steps))
    return frozenset(shifted)


def _shift_column(
    column: Column, departure_step: int, cfg: SimConfig, model: CostModel = DELAY_MODEL
) -> Column:
    """Return the same certified spatial route at a later integer departure."""

    delta = departure_step - column.departure_step
    if delta < 0:
        raise ValueError("a seed column can only be shifted later")
    return replace(
        column,
        departure_step=departure_step,
        # A pure clock translation adds GROUND delay only; the route, and so the air
        # term, is invariant.  See colgen.objective.
        delay_s=column.delay_s + model.ground_weight * (delta * cfg.dt_s),
        claims=_shift_claims(column.claims, delta),
    )


def _initial_feasible_selection(
    seeds: Mapping[int, Column],
    graphs: Mapping[int, FlightGraph],
    fixed_loads: Mapping[RowKey, int],
    row_index: RowIndex,
    cfg: SimConfig,
    *,
    deadline: float | None = None,
    model: CostModel = DELAY_MODEL,
) -> dict[int, Column]:
    """Greedily ground-shift seed paths into a globally feasible RMP incumbent.

    This is an acceleration, not a restriction: nominal seeds remain in the
    master, and pricing can add every route in the original universe.  Seeding
    the RMP with a real all-flight incumbent prevents a time-capped solve from
    discarding known coverage and then repeating the same search in repair.
    """

    selected: dict[int, Column] = {}
    loads: Counter[RowKey] = Counter(fixed_loads)
    order = sorted(
        (graphs[flight_id] for flight_id in seeds),
        key=lambda graph: (graph.request.t_departure, graph.request.flight_id),
    )
    for graph in order:
        if deadline is not None and time.monotonic() >= deadline:
            break
        seed = seeds[graph.request.flight_id]
        origin_lane_steps = (
            0
            if seed.origin_lane_idx is None
            else graph.origin_lanes[seed.origin_lane_idx].steps
        )
        path_latest_departure = (
            graph.max_step
            - graph.takeoff_steps[seed.level]
            - origin_lane_steps
            - (len(seed.cell_path) - 1)
        )
        latest_departure = min(graph.latest_departure_step, path_latest_departure)
        for departure_step in range(seed.departure_step, latest_departure + 1):
            if deadline is not None and time.monotonic() >= deadline:
                return dict(sorted(selected.items()))
            candidate = _shift_column(seed, departure_step, cfg, model)
            if any(loads[row] + 1 > row_index.cap(row) for row in candidate.claims):
                continue
            # Time translation is exact for this fixed path.  Re-run the
            # authoritative gate once for the chosen shift as a defensive
            # assertion against future claim-window changes.
            candidate = _canonical_column(candidate, graph, cfg)
            if any(loads[row] + 1 > row_index.cap(row) for row in candidate.claims):
                continue
            selected[graph.request.flight_id] = candidate
            loads.update(candidate.claims)
            break
    return dict(sorted(selected.items()))


def _greedy_feasible_selection(
    graphs: Mapping[int, FlightGraph],
    fixed_loads: Mapping[RowKey, int],
    row_index: RowIndex,
    cfg: SimConfig,
    params: ColGenParams,
    *,
    deadline: float,
    initial: Mapping[int, Column] | None = None,
) -> tuple[dict[int, Column], bool]:
    """Improve a complete seed incumbent with lazy best-first local searches.

    The shifted-seed schedule above is a cheap fallback, but it cannot route
    around a busy cell.  This post-first-LP pass removes one flight at a time
    from the incumbent, asks the pricing topology for a lower-delay replacement
    avoiding every other selected column, and restores the old column when no
    improvement is found.  A timeout therefore keeps a complete feasible
    schedule instead of discarding a partially built prefix.  It remains only
    an incumbent heuristic and never contributes to a global bound.
    """

    model = cost_model(cfg, params)
    selected: dict[int, Column] = dict(initial or {})
    loads = _loads_for(selected, fixed_loads)
    saturated = {
        row for row, load in loads.items() if load >= row_index.cap(row)
    }

    def remove_claims(claims: frozenset[RowKey]) -> None:
        for row in claims:
            old_load = loads[row]
            new_load = old_load - 1
            if new_load:
                loads[row] = new_load
            else:
                del loads[row]
            if old_load >= row_index.cap(row) and new_load < row_index.cap(row):
                saturated.discard(row)

    def add_claims(claims: frozenset[RowKey]) -> None:
        for row in claims:
            old_load = loads[row]
            new_load = old_load + 1
            loads[row] = new_load
            if old_load < row_index.cap(row) <= new_load:
                saturated.add(row)

    order = sorted(
        (
            graph
            for graph in graphs.values()
            if graph.request.flight_id not in selected
            or selected[graph.request.flight_id].delay_s > _OBJECTIVE_TOL
        ),
        key=lambda graph: (
            -selected[graph.request.flight_id].delay_s
            if graph.request.flight_id in selected
            else -math.inf,
            graph.request.t_departure,
            graph.request.flight_id,
        ),
    )
    candidate_limit = max(64, params.n_heuristic_tries * 16)
    completed = len(order) <= candidate_limit
    order = order[:candidate_limit]
    for graph_index, graph in enumerate(order):
        if time.monotonic() >= deadline:
            return dict(sorted(selected.items())), False
        flight_id = graph.request.flight_id
        incumbent = selected.pop(flight_id, None)
        if incumbent is not None:
            remove_claims(incumbent.claims)

        now = time.monotonic()
        attempts_remaining = len(order) - graph_index
        # Reserve an equal share for every remaining flight.  Easy A* searches
        # return early and donate their unused share; a hard flight times out
        # locally instead of preventing the rest of the sweep from being tried.
        flight_deadline = min(deadline, now + (deadline - now) / attempts_remaining)
        try:
            candidate = find_feasible_column(
                graph,
                cfg,
                forbidden_rows=saturated,
                improve_below_delay_s=(
                    None if incumbent is None else incumbent.delay_s
                ),
                deadline=flight_deadline,
                model=model,
            )
        except PricingTimeout:
            completed = False
            if incumbent is not None:
                selected[flight_id] = incumbent
                add_claims(incumbent.claims)
            continue
        if candidate is None or (
            incumbent is not None
            and not _better_selection(
                {flight_id: candidate},
                {flight_id: incumbent},
                benefit=0.0,
            )
        ):
            if incumbent is not None:
                selected[flight_id] = incumbent
                add_claims(incumbent.claims)
            continue
        candidate = _canonical_column(candidate, graph, cfg)
        if any(loads[row] + 1 > row_index.cap(row) for row in candidate.claims):
            raise RuntimeError(
                f"greedy pricing returned an infeasible column for flight "
                f"{graph.request.flight_id}"
            )
        selected[flight_id] = candidate
        add_claims(candidate.claims)
    return dict(sorted(selected.items())), completed


def _column_key(column: Column) -> tuple[Any, ...]:
    """A backend-independent tie-break key for reproducible incumbents."""

    return (
        column.departure_step,
        column.level,
        -1 if column.origin_lane_idx is None else column.origin_lane_idx,
        -1 if column.dest_lane_idx is None else column.dest_lane_idx,
        column.cell_path,
    )


def _selection_key(selection: Mapping[int, Column]) -> tuple[Any, ...]:
    return tuple((flight_id, _column_key(selection[flight_id])) for flight_id in sorted(selection))


def _selection_objective(selection: Mapping[int, Column], benefit: float) -> float:
    # ``delay_s`` already carries whatever the objective weights made it -- see
    # colgen.objective -- so this stays a plain difference.
    return math.fsum(benefit - column.delay_s for column in selection.values())


def _better_selection(
    candidate: Mapping[int, Column],
    incumbent: Mapping[int, Column],
    benefit: float,
) -> bool:
    """Compare maximize-sense incumbents, with a deterministic exact-tie rule."""

    candidate_obj = _selection_objective(candidate, benefit)
    incumbent_obj = _selection_objective(incumbent, benefit)
    if candidate_obj > incumbent_obj + _OBJECTIVE_TOL:
        return True
    if incumbent_obj > candidate_obj + _OBJECTIVE_TOL:
        return False
    return _selection_key(candidate) < _selection_key(incumbent)


def _relative_revenue_gap(upper_bound: float, rmp_value: float) -> float:
    """The paper's gap, equations (10) and (11): ``(UB - RMP) / RMP``.

    Measured on the maximize objective, whose scale includes ``n * M``.  That
    normalization is the whole difference from :func:`_relative_cost_gap`: the same
    absolute slack of ~90,000 units reads as 0.009% here (against a revenue of ~1e9) and
    as 63% there (against a total cost of ~141,000).  Both describe the identical
    solution; the paper's thresholds -- 0.01% for the LP, 0.1% for the heuristic -- are
    calibrated against this one.
    """

    if not math.isfinite(upper_bound) or not math.isfinite(rmp_value):
        return math.inf
    # Test the numerator before the scale.  A bound that coincides with the RMP has
    # closed, and that is true at any scale including zero -- which is reachable here,
    # because a one-flight master whose only column exactly consumes its benefit prices
    # out at objective 0.  Dividing first would report that tight bound as `inf`.
    absolute = max(0.0, upper_bound - rmp_value)
    if absolute <= 0.0:
        return 0.0
    denominator = abs(rmp_value)
    if denominator <= 0.0:
        return math.inf
    return absolute / denominator


def _relative_cost_gap(cost_upper_bound: float, cost_lower_bound: float) -> float:
    """Return a relative minimization gap in transformed delay currency.

    Master revenue includes the large per-flight benefit ``M``.  Normalizing a
    gap by that revenue would make a few seconds of delay look converged.  The
    equivalent minimization objective charges delay to selected flights and
    ``M`` to an omitted flight; its scale is the one the configured gaps mean.
    """

    if not math.isfinite(cost_upper_bound) or not math.isfinite(cost_lower_bound):
        return math.inf
    return max(0.0, cost_upper_bound - cost_lower_bound) / max(1.0, abs(cost_upper_bound))


def _loads_for(
    selection: Mapping[int, Column], fixed_loads: Mapping[RowKey, int]
) -> Counter[RowKey]:
    loads: Counter[RowKey] = Counter(fixed_loads)
    for column in selection.values():
        loads.update(column.claims)
    return loads


def _assert_claim_feasible(
    selection: Mapping[int, Column],
    fixed_loads: Mapping[RowKey, int],
    row_index: RowIndex,
) -> Counter[RowKey]:
    """Independently referee an integer selection against all canonical claims."""

    loads = _loads_for(selection, fixed_loads)
    for row, load in loads.items():
        capacity = row_index.cap(row)
        if load > capacity:
            raise RuntimeError(
                f"colgen returned row-infeasible selection at {tuple(row)!r}: "
                f"load {load} > capacity {capacity}"
            )
    return loads


def _backend_name(master: RestrictedMaster) -> str:
    explicit = getattr(master, "backend_name", None)
    if explicit is not None:
        return str(explicit)
    backend = getattr(master, "backend", None)
    if backend is None:
        backend = getattr(master, "_backend", None)
    if backend is None:
        return "unknown"
    return type(backend).__name__.removesuffix("Backend").lower()


def _pre_master_timeout_result(
    flight_ids: Sequence[int],
    started: float,
    params: ColGenParams,
    *,
    stage: str = "startup",
    graphs: Mapping[int, FlightGraph] | None = None,
    catalog: StaticTerminalCatalog | None = None,
    graph_build_elapsed_s: float = 0.0,
    seed_elapsed_s: float = 0.0,
    seeds_completed: int = 0,
    seed_flights_processed: int = 0,
    master: RestrictedMaster | None = None,
    time_to_master_s: float = 0.0,
    seedless_flight_ids: Sequence[int] = (),
) -> ColGenResult:
    """Return an explicit compute-cap verdict before a usable master exists."""

    elapsed_s = time.monotonic() - started
    denied = tuple(flight_ids)
    heuristic_cost = len(denied) * params.M
    graph_values = tuple((graphs or {}).values())
    arc_stats: Counter[str] = Counter()
    for graph in graph_values:
        arc_stats.update(graph.arc_cache_stats)
    wall_stats = {} if catalog is None else catalog.wall_index.stats
    return ColGenResult(
        columns={},
        stats={
            "backend": "none" if master is None else _backend_name(master),
            "iterations": 0,
            "termination_reason": "time_limit",
            "lp_objectives": (),
            "lower_bounds": (),
            "upper_bounds": (),
            "cost_upper_bounds": (),
            "cost_lower_bounds": (),
            "lp_gaps": (),
            "heuristic_objectives": (),
            "heuristic_costs": (),
            "final_lp_objective": -math.inf,
            "upper_bound": math.inf,
            "cost_upper_bound": math.inf,
            "cost_lower_bound": -math.inf,
            "lp_gap": math.inf,
            "heuristic_objective": 0.0,
            "heuristic_cost": heuristic_cost,
            # The full path's gap and timing keys, at the values a pre-master timeout
            # actually implies.  `batch.run_batch` reads five of these by name; a missing
            # one logged as "unknown" on exactly the run whose cost you most want to see.
            "gap_metric": params.gap_metric,
            "lp_gap_revenue": math.inf,
            "lp_gap_cost": math.inf,
            "ip_gap_revenue": None,
            "pricing_wall_s": 0.0,
            "seeded_columns": 0,
            "ip_elapsed_s": 0.0,
            "ip_objective": None,
            "ip_upper_bound": None,
            "ip_cost_lower_bound": None,
            "ip_cost_upper_bound": None,
            "ip_gap": None,
            "ip_gap_met": None,
            "ip_status": "time_limit_skipped",
            "ip_optimal": None,
            "ip_skipped": True,
            "objective": 0.0,
            "objective_name": params.objective,
            "master_objective": 0.0,
            "selected_flights": 0,
            "denied_flight_ids": denied,
            "budget_denied_flight_ids": (),
            "search_exhausted_flight_ids": denied,
            "repair_added": 0,
            "seedless_flight_ids": tuple(sorted(seedless_flight_ids)),
            "initial_heuristic_strategy": "time_limit",
            "initial_heuristic_flights": 0,
            "initial_heuristic_delay_s": 0.0,
            "initial_greedy_completed": False,
            "initial_greedy_flights": 0,
            "initial_greedy_elapsed_s": 0.0,
            "initial_seed_columns": seeds_completed,
            "graphs_built": len(graph_values),
            "seeds_completed": seeds_completed,
            "seed_flights_processed": seed_flights_processed,
            "preprocessing_stage": stage,
            "graph_build_elapsed_s": graph_build_elapsed_s,
            "seed_elapsed_s": seed_elapsed_s,
            "time_to_master_s": time_to_master_s,
            "static_terminal_count": 0 if catalog is None else len(catalog.entries),
            "static_excluded_cells": 0 if catalog is None else catalog.excluded_cell_count,
            "corridor_domains_materialized": sum(
                bool(getattr(graph.corridor_cells, "is_materialized", True))
                for graph in graph_values
            ),
            "arc_expanded_nodes": arc_stats["expanded_nodes"],
            "arc_checks": arc_stats["arc_checks"],
            "arc_cache_hits": arc_stats["cache_hits"],
            "arc_allowed": arc_stats["allowed_arcs"],
            "arc_blocked": arc_stats["blocked_arcs"],
            "wall_index_queries": int(wall_stats.get("queries", 0)),
            "wall_index_candidates": int(wall_stats.get("candidates", 0)),
            "pricing_flights_completed": 0,
            "pricing_sweeps_completed": 0,
            "pricing_timeout_flight_id": None,
            "n_columns": 0 if master is None else len(master.columns),
            "n_materialized_rows": 0,
            "lazy_rows_added": 0,
            "lazy_row_rounds": 0,
            "time_limit_overrun_s": max(0.0, elapsed_s - params.time_limit_s),
            "elapsed_s": elapsed_s,
        },
    )


class ColGenSolver:
    """Solve all requests together by deterministic delayed-row column generation."""

    def solve(
        self,
        requests: Sequence[FlightRequest],
        cfg: SimConfig,
        static_terms,
        params: ColGenParams,
        *,
        fixed_claims: Sequence[frozenset[RowKey]] = (),
        on_iteration=None,
        seed_columns: Mapping[int, Sequence[Column]] | None = None,
    ) -> ColGenResult:
        """Run the column-generation loop to convergence, a bound, or a time limit.

        Pricing is a sequential sweep over the flights: each subproblem is independent
        given the iteration's duals, so the loop order affects only which columns a
        timed-out sweep managed to reach, never their value.

        ``on_iteration`` is called once per column-generation iteration with a dict of
        that iteration's master state (LP objective, global upper bound, gaps, column
        counts).  Without it a long solve is opaque until it returns: the bound and the
        gap are computed every iteration but only surface in the final stats, so a run
        that is killed -- or one you simply want to watch -- discards them.  Nothing else
        in this package logs, so without a callback a solve is silent until `run_batch`
        summarises it.
        """
        started = time.monotonic()
        pricing_wall_s = 0.0
        deadline = started + params.time_limit_s
        # Leave a small tail for the final restricted-master IP.  An incomplete
        # pricing sweep cannot certify a global bound, but every completed
        # column is still valid and can improve that final incumbent.
        ip_reserve_s = min(5.0, 0.05 * params.time_limit_s)
        pricing_deadline = deadline - ip_reserve_s
        ordered_requests = tuple(sorted(requests, key=lambda request: request.flight_id))
        flight_ids = tuple(request.flight_id for request in ordered_requests)
        if len(set(flight_ids)) != len(flight_ids):
            raise ValueError("colgen requests must have unique flight_id values")

        if not ordered_requests:
            return ColGenResult(
                columns={},
                stats={
                    "backend": "none",
                    "iterations": 0,
                    "termination_reason": "empty",
                    "lp_objectives": (),
                    "lower_bounds": (),
                    "upper_bounds": (),
                    "cost_upper_bounds": (),
                    "cost_lower_bounds": (),
                    "lp_gaps": (),
                    "heuristic_objectives": (),
                    "heuristic_costs": (),
                    "final_lp_objective": 0.0,
                    "upper_bound": 0.0,
                    "cost_upper_bound": 0.0,
                    "cost_lower_bound": 0.0,
                    "lp_gap": 0.0,
                    "heuristic_objective": 0.0,
                    "heuristic_cost": 0.0,
                    "gap_metric": params.gap_metric,
                    "lp_gap_revenue": 0.0,
                    "lp_gap_cost": 0.0,
                    "ip_gap_revenue": None,
                    "pricing_wall_s": 0.0,
                    "seeded_columns": 0,
                    "ip_elapsed_s": 0.0,
                    "ip_objective": None,
                    "ip_upper_bound": None,
                    "ip_cost_lower_bound": None,
                    "ip_cost_upper_bound": None,
                    "ip_gap": None,
                    "ip_gap_met": None,
                    "ip_status": "skipped",
                    "ip_optimal": None,
                    "ip_skipped": True,
                    "objective": 0.0,
                    "objective_name": params.objective,
                    "master_objective": 0.0,
                    "selected_flights": 0,
                    "denied_flight_ids": (),
                    "repair_added": 0,
                    "seedless_flight_ids": (),
                    "budget_denied_flight_ids": (),
                    "search_exhausted_flight_ids": (),
                    "initial_heuristic_strategy": "empty",
                    "initial_heuristic_flights": 0,
                    "initial_heuristic_delay_s": 0.0,
                    "initial_greedy_completed": True,
                    "initial_greedy_flights": 0,
                    "initial_greedy_elapsed_s": 0.0,
                    "initial_seed_columns": 0,
                    "graphs_built": 0,
                    "seeds_completed": 0,
                    "seed_flights_processed": 0,
                    "preprocessing_stage": "empty",
                    "graph_build_elapsed_s": 0.0,
                    "seed_elapsed_s": 0.0,
                    "time_to_master_s": 0.0,
                    "static_terminal_count": 0,
                    "static_excluded_cells": 0,
                    "corridor_domains_materialized": 0,
                    "arc_expanded_nodes": 0,
                    "arc_checks": 0,
                    "arc_cache_hits": 0,
                    "arc_allowed": 0,
                    "arc_blocked": 0,
                    "wall_index_queries": 0,
                    "wall_index_candidates": 0,
                    "pricing_flights_completed": 0,
                    "pricing_sweeps_completed": 0,
                    "pricing_timeout_flight_id": None,
                    "n_columns": 0,
                    "n_materialized_rows": 0,
                    "lazy_rows_added": 0,
                    "lazy_row_rounds": 0,
                    "time_limit_overrun_s": 0.0,
                    "elapsed_s": time.monotonic() - started,
                },
            )

        # ``static_terms`` is commonly a generator owned by the simulation.  Every
        # graph must see the identical wall catalogue, so snapshot it once.
        graph_build_started = time.monotonic()
        static_catalog = StaticTerminalCatalog(static_terms, cfg)
        static_term_snapshot = static_catalog.entries
        graphs: dict[int, FlightGraph] = {}
        for request in ordered_requests:
            if time.monotonic() >= pricing_deadline:
                return _pre_master_timeout_result(
                    flight_ids,
                    started,
                    params,
                    stage="graph_build",
                    graphs=graphs,
                    catalog=static_catalog,
                    graph_build_elapsed_s=time.monotonic() - graph_build_started,
                )
            graphs[request.flight_id] = build_flight_graph(
                request,
                cfg,
                static_catalog,
                params,
            )
        graph_build_elapsed_s = time.monotonic() - graph_build_started
        if time.monotonic() >= pricing_deadline:
            return _pre_master_timeout_result(
                flight_ids,
                started,
                params,
                stage="graph_build",
                graphs=graphs,
                catalog=static_catalog,
                graph_build_elapsed_s=graph_build_elapsed_s,
            )

        # One objective for the whole solve, threaded into seeding, the greedy
        # heuristic and pricing.  See colgen.objective.
        model = cost_model(cfg, params)

        row_index = RowIndex()
        for graph in graphs.values():
            for terminal_id, capacity in graph.terminal_capacities.items():
                row_index.register_terminal(terminal_id, capacity)
        # Register static metadata too.  This matters to the rolling-horizon seam
        # when a fixed terminal claim belongs to a flight outside this batch.
        for _center, raw_terminal in static_term_snapshot:
            terminal = as_terminal(raw_terminal)
            if terminal is not None:
                row_index.register_terminal(terminal)

        committed_loads = _fixed_loads(fixed_claims)
        for row, load in committed_loads.items():
            if load > row_index.cap(row):
                raise ValueError(
                    f"fixed claims already exceed capacity at {tuple(row)!r}: "
                    f"load {load} > capacity {row_index.cap(row)}"
                )

        master = RestrictedMaster(
            flight_ids,
            row_index,
            params,
            seed=cfg.seed,
            fixed_loads=committed_loads,
        )
        time_to_master_s = time.monotonic() - started
        seedless_flights: set[int] = set()
        seeds: dict[int, Column] = {}
        seed_started = time.monotonic()
        for flight_id in flight_ids:
            try:
                seed = seed_column(
                    graphs[flight_id], cfg, deadline=pricing_deadline, model=model
                )
            except PricingTimeout:
                return _pre_master_timeout_result(
                    flight_ids,
                    started,
                    params,
                    stage="seeding",
                    graphs=graphs,
                    catalog=static_catalog,
                    graph_build_elapsed_s=graph_build_elapsed_s,
                    seed_elapsed_s=time.monotonic() - seed_started,
                    seeds_completed=len(seeds),
                    seed_flights_processed=len(seeds) + len(seedless_flights),
                    master=master,
                    time_to_master_s=time_to_master_s,
                    seedless_flight_ids=tuple(seedless_flights),
                )
            except ValueError:
                # A disconnected/static-blocked graph is a legitimate
                # optimizer-verdict denial, not a batch-wide solver failure.
                # Its <=1 flight row can remain empty while other flights run.
                seedless_flights.add(flight_id)
                continue
            seed = _canonical_column(seed, graphs[flight_id], cfg)
            seeds[flight_id] = seed
            master.add_column(seed)
        seed_elapsed_s = time.monotonic() - seed_started

        shifted_seed_heuristic = _initial_feasible_selection(
            seeds,
            graphs,
            committed_loads,
            row_index,
            cfg,
            deadline=pricing_deadline,
            model=model,
        )
        # Keep initialization deliberately small: one certified shortest seed
        # per flight plus at most one time-shifted seed selected by the cheap
        # claim-feasible pass above.  Route alternatives belong to reduced-cost
        # pricing after the first LP, not to an eager all-flight prepass.
        greedy_heuristic: dict[int, Column] = {}
        greedy_completed = False
        greedy_elapsed_s = 0.0
        initial_heuristic = dict(shifted_seed_heuristic)
        initial_heuristic_strategy = "shifted_seeds"
        best_heuristic = dict(initial_heuristic)
        for column in initial_heuristic.values():
            master.add_column(column)
        # Optional warm start.  The policy above is a deliberate bet -- that route
        # alternatives are cheaper to discover by reduced-cost pricing than to enumerate
        # up front -- and `seed_columns` is how that bet gets tested rather than assumed.
        # It is a pool-contents knob only: every column still goes through the same
        # canonical claim gate, so it cannot introduce a trajectory pricing could not
        # have produced, and it changes which optimum is reached only by the same
        # tie-breaking that column order already governs.
        seeded_columns = 0
        for flight_id, extras in (seed_columns or {}).items():
            if flight_id not in graphs:
                raise KeyError(f"seed_columns names flight {flight_id}, which is not in this batch")
            for column in extras:
                master.add_column(_canonical_column(column, graphs[flight_id], cfg))
                seeded_columns += 1
        if best_heuristic:
            master.set_heuristic(best_heuristic)

        rng = np.random.default_rng(cfg.seed)
        total_benefit = len(flight_ids) * params.M
        lp_objectives: list[float] = []
        upper_bounds: list[float] = []
        cost_upper_bounds: list[float] = []
        cost_lower_bounds: list[float] = []
        lp_gaps: list[float] = []
        # Previous iteration's capacity duals, kept only to measure dual movement.
        # Tailing-off with a frozen bound has two very different causes -- duals
        # converging slowly, or duals oscillating -- and they call for opposite fixes
        # (more iterations vs. stabilization).  ||pi_k - pi_{k-1}|| separates them.
        prev_capacity_duals: dict | None = None
        heuristic_objectives: list[float] = []
        heuristic_costs: list[float] = []
        lazy_rows_added = 0
        lazy_row_rounds = 0
        termination_reason = "iteration_limit"
        last_upper_bound = math.inf
        last_lp_objective = -math.inf
        last_x: np.ndarray | None = None
        iterations = 0
        pricing_flights_completed = 0
        pricing_sweeps_completed = 0
        pricing_timeout_flight_id: int | None = None

        for iteration in range(params.max_iterations):
            # Per-iteration stage timings.  The master block was one unattributed lump in
            # the serial tail, and "the LP is slow" is only one of four candidates in it:
            # the LP itself, the lazy-row re-solve loop around it, `_canonical_column`
            # (run once per priced column, in the parent), and `add_column`.
            stage_s: dict[str, float] = collections.defaultdict(float)
            stage_n: dict[str, int] = collections.defaultdict(int)

            def _timed(key, fn, *a, **kw):
                t0 = time.perf_counter()
                try:
                    return fn(*a, **kw)
                finally:
                    stage_s[key] += time.perf_counter() - t0
                    stage_n[key] += 1

            if time.monotonic() >= pricing_deadline:
                termination_reason = "time_limit"
                break

            # A bound is meaningful only for the full claim-feasible relaxation.
            # Materialize violated rows and re-solve until the current LP is clean.
            lp_complete = False
            while True:
                remaining_s = pricing_deadline - time.monotonic()
                if remaining_s <= 0.0:
                    break
                master.backend.time_limit_s = max(1e-6, remaining_s)
                try:
                    lp_objective, capacity_duals, x = _timed("solve_lp", master.solve_lp)
                except BackendTimeout:
                    break
                added_rows = _timed(
                    "add_violated_rows", master.add_violated_rows, x, _FEASIBILITY_TOL
                )
                if not added_rows:
                    lp_complete = True
                    break
                lazy_rows_added += added_rows
                lazy_row_rounds += 1
            if not lp_complete:
                termination_reason = "time_limit"
                break

            iterations = iteration + 1
            last_lp_objective = float(lp_objective)
            last_x = np.asarray(x, dtype=float)
            columns_at_lp = len(master.columns)

            heuristic = _timed(
                "round_heuristic", master.round_heuristic, last_x, rng, params.n_heuristic_tries
            )
            heuristic = _timed(
                "canonical_heuristic",
                lambda h: {
                    flight_id: _canonical_column(column, graphs[flight_id], cfg)
                    for flight_id, column in h.items()
                },
                heuristic,
            )
            _assert_claim_feasible(heuristic, committed_loads, row_index)
            if not best_heuristic or _better_selection(heuristic, best_heuristic, params.M):
                best_heuristic = dict(heuristic)
                master.set_heuristic(best_heuristic)

            if iteration == 0:
                # The first LP has now established the real column-generation
                # cycle.  Build at most one route-aware incumbent column per
                # flight using the same lazy topology, bounded independently so
                # formal reduced-cost pricing retains most of the solve budget.
                greedy_started = time.monotonic()
                greedy_budget_s = min(60.0, 0.55 * params.time_limit_s)
                greedy_deadline = min(pricing_deadline, greedy_started + greedy_budget_s)
                greedy_heuristic, greedy_completed = _greedy_feasible_selection(
                    graphs,
                    committed_loads,
                    row_index,
                    cfg,
                    params,
                    deadline=greedy_deadline,
                    initial=best_heuristic,
                )
                greedy_elapsed_s = time.monotonic() - greedy_started
                for column in greedy_heuristic.values():
                    _timed("add_column", master.add_column, column)
                if greedy_heuristic and _better_selection(
                    greedy_heuristic,
                    best_heuristic,
                    params.M,
                ):
                    best_heuristic = dict(greedy_heuristic)
                    master.set_heuristic(best_heuristic)

            best_reduced_costs: list[float] = []
            rc_by_flight: dict[int, float] = {}
            priced_columns: list[Column] = []
            pricing_order = sorted(
                flight_ids,
                key=lambda flight_id: (
                    flight_id in best_heuristic,
                    -best_heuristic[flight_id].delay_s
                    if flight_id in best_heuristic
                    else -math.inf,
                    flight_id,
                ),
            )
            pricing_complete = True
            dual_view = DualView(capacity_duals, cfg)
            flight_duals = master.flight_duals
            # Per-iteration sweep cost, which is the whole cost of a solve.  A total
            # cannot answer "is pricing getting cheaper as the cutoffs tighten, or is
            # iteration 80 as expensive as iteration 1" -- and that question decides
            # whether more iterations are affordable.
            sweep_started = time.perf_counter()
            for flight_id in pricing_order:
                try:
                    reduced_cost, column = price_flight(
                        graphs[flight_id],
                        dual_view,
                        flight_duals[flight_id],
                        cfg,
                        params,
                        known_column=best_heuristic.get(flight_id),
                        deadline=pricing_deadline,
                    )
                except PricingTimeout:
                    pricing_complete = False
                    pricing_timeout_flight_id = flight_id
                    break
                pricing_flights_completed += 1
                best_reduced_costs.append(float(reduced_cost))
                rc_by_flight[flight_id] = float(reduced_cost)
                if column is not None and reduced_cost > _REDUCED_COST_TOL:
                    priced_columns.append(
                        _timed(
                            "canonical_priced", _canonical_column,
                            column, graphs[flight_id], cfg,
                        )
                    )

            iteration_sweep_s = time.perf_counter() - sweep_started
            pricing_wall_s += iteration_sweep_s

            before_pricing = len(master.columns)
            for column in sorted(
                priced_columns, key=lambda item: (item.flight_id, _column_key(item))
            ):
                _timed("add_column", master.add_column, column)
            if not pricing_complete:
                termination_reason = "time_limit"
                break
            pricing_sweeps_completed += 1

            raw_upper_bound = master.upper_bound(last_lp_objective, best_reduced_costs)
            # Each iteration's value is a valid global upper bound.  Their
            # running minimum remains valid and removes harmless solver jitter.
            last_upper_bound = min(last_upper_bound, max(last_lp_objective, raw_upper_bound))
            # Transform maximize revenue back to the user objective before
            # measuring gaps.  Revenue-scale gaps are diluted by n*M and can
            # incorrectly certify a trajectory that is tens of seconds worse.
            cost_upper_bound = total_benefit - last_lp_objective
            cost_lower_bound = total_benefit - last_upper_bound
            heuristic_objective = _selection_objective(best_heuristic, params.M)
            heuristic_cost = total_benefit - heuristic_objective
            # Both scales are computed every iteration so neither is lost; only which one
            # the thresholds are applied to depends on params.gap_metric.
            lp_gap_cost = _relative_cost_gap(cost_upper_bound, cost_lower_bound)
            heuristic_gap_cost = _relative_cost_gap(heuristic_cost, cost_lower_bound)
            lp_gap_revenue = _relative_revenue_gap(last_upper_bound, last_lp_objective)
            heuristic_gap_revenue = _relative_revenue_gap(
                last_upper_bound, heuristic_objective
            )
            if params.gap_metric == "revenue":
                lp_gap, heuristic_gap = lp_gap_revenue, heuristic_gap_revenue
            else:
                lp_gap, heuristic_gap = lp_gap_cost, heuristic_gap_cost

            lp_objectives.append(last_lp_objective)
            upper_bounds.append(last_upper_bound)
            cost_upper_bounds.append(cost_upper_bound)
            cost_lower_bounds.append(cost_lower_bound)
            lp_gaps.append(lp_gap)
            heuristic_objectives.append(heuristic_objective)
            heuristic_costs.append(heuristic_cost)

            if on_iteration is not None:
                # Reduced-cost spread.  The bound is LP + sum(max(0, rc)), so whether the
                # gap is a few pathological flights or all of them uniformly decides
                # whether this is targetable or structural -- and the values are already
                # computed here and otherwise discarded.
                positives = sorted(v for v in best_reduced_costs if v > 0.0)
                def _pct(frac: float) -> float:
                    if not positives:
                        return 0.0
                    return positives[min(len(positives) - 1, int(frac * len(positives)))]

                # Dual movement over the union of both iterations' support.
                dual_l2 = dual_linf = float("nan")
                if prev_capacity_duals is not None:
                    keys = set(capacity_duals) | set(prev_capacity_duals)
                    diffs = [
                        capacity_duals.get(k, 0.0) - prev_capacity_duals.get(k, 0.0)
                        for k in keys
                    ]
                    dual_l2 = math.sqrt(math.fsum(d * d for d in diffs))
                    dual_linf = max((abs(d) for d in diffs), default=0.0)

                on_iteration({
                    "iteration": iterations,
                    "lp_objective": last_lp_objective,
                    "upper_bound": last_upper_bound,
                    "raw_upper_bound": raw_upper_bound,
                    "cost_upper_bound": cost_upper_bound,
                    "cost_lower_bound": cost_lower_bound,
                    "lp_gap": lp_gap,
                    "lp_gap_revenue": lp_gap_revenue,
                    "lp_gap_cost": lp_gap_cost,
                    "heuristic_gap_revenue": heuristic_gap_revenue,
                    "heuristic_gap_cost": heuristic_gap_cost,
                    "heuristic_cost": heuristic_cost,
                    "heuristic_gap": heuristic_gap,
                    # The live master, for analysis that has to ask it something the
                    # scalars above cannot answer -- solving the IP at THIS iteration
                    # being the motivating case, since the difference between the
                    # rounding heuristic and the true restricted IP is exactly what
                    # `heuristic_cost` alone cannot show.  Callers that mutate it own the
                    # consequences; the solver reads back only `last_ip_*`, which its own
                    # final solve overwrites.
                    "master": master,
                    # This iteration's LP solution, positionally aligned with
                    # ``master.columns``.  Needed to tell a column that was ADDED from one
                    # the LP actually USES: pricing's acceptance test is a reduced-cost
                    # sign, which says a column improves the LP basis, not that it ends up
                    # with x > 0.  Without x those two are indistinguishable from outside.
                    "lp_x": last_x,
                    # This iteration's capacity duals, by row.  `dual_nonzero` counts them
                    # but cannot say WHICH rows are expensive, and that is the question
                    # behind "why do the new columns conflict": pricing steers every
                    # flight away from the same few hot rows at once, so their proposals
                    # collide on the alternatives.
                    "capacity_duals": capacity_duals,
                    # This iteration's pricing cost -- the wall the solver spent in the
                    # sweep, which is very nearly the wall of the whole iteration.
                    "sweep_s": iteration_sweep_s,
                    "columns": len(master.columns),
                    "columns_added": len(priced_columns),
                    "rc_sum": math.fsum(positives),
                    "rc_n_positive": len(positives),
                    "rc_max": positives[-1] if positives else 0.0,
                    "rc_p50": _pct(0.5),
                    "rc_p90": _pct(0.9),
                    "dual_l2": dual_l2,
                    "dual_linf": dual_linf,
                    "dual_nonzero": sum(1 for v in capacity_duals.values() if v != 0.0),
                    "elapsed_s": time.monotonic() - started,
                    **_coverage_diagnostics(master, last_x, rc_by_flight, params.M),
                    "stage_s": dict(stage_s),
                    "stage_n": dict(stage_n),
                    "lazy_rows_added": lazy_rows_added,
                    "lazy_row_rounds": lazy_row_rounds,
                })
            prev_capacity_duals = dict(capacity_duals)

            new_columns_since_lp = len(master.columns) > columns_at_lp
            # Section 4.2.3 stops as soon as the LP gap is under threshold, full stop --
            # the bound `RMP + sum(max(0, phi_max))` is valid however many columns were
            # just banked, and further columns can only move the RMP toward it.  The
            # `not new_columns_since_lp` guard is a stricter local rule that makes the
            # criterion effectively unreachable while pricing is still productive, so it
            # applies only to the cost-scale metric this repo used before.
            gate = params.gap_metric == "revenue" or not new_columns_since_lp
            if lp_gap <= params.lp_gap and gate:
                termination_reason = "lp_gap"
                break
            if best_heuristic and heuristic_gap <= params.ip_gap and gate:
                termination_reason = "heuristic_gap"
                break
            if not priced_columns and not new_columns_since_lp:
                termination_reason = "no_improving_columns"
                break

            if len(master.columns) == before_pricing and not new_columns_since_lp:
                termination_reason = "no_new_columns"
                break
        else:
            termination_reason = "iteration_limit"

        # The heuristic is a genuine incumbent on both backends and a MIP start
        # on Gurobi.  It may prove the requested IP gap before a final MILP is
        # useful; otherwise let the backend solve and independently compare.
        incumbent = dict(best_heuristic)
        heuristic_objective = _selection_objective(incumbent, params.M)
        heuristic_cost = total_benefit - heuristic_objective
        final_cost_lower_bound = total_benefit - last_upper_bound
        heuristic_gap = _relative_cost_gap(heuristic_cost, final_cost_lower_bound)
        ip_skipped = bool(incumbent) and heuristic_gap <= params.ip_gap
        ip_objective: float | None = None
        ip_upper_bound: float | None = None
        ip_cost_lower_bound: float | None = None
        ip_cost_upper_bound: float | None = None
        ip_gap: float | None = None
        ip_gap_revenue: float | None = None
        ip_gap_met: bool | None = None
        ip_elapsed_s = 0.0
        ip_status = "skipped"
        ip_optimal: bool | None = None
        if not ip_skipped and time.monotonic() >= deadline:
            ip_skipped = True
            ip_status = "time_limit_skipped"
            termination_reason = "time_limit"
        elif not ip_skipped:
            master.backend.time_limit_s = max(1e-6, deadline - time.monotonic())
            master.set_heuristic(incumbent)
            # Timed because it was the one unattributed block left in the solve.  Worth
            # knowing precisely: on a 1,138-column 100-flight pool the whole solve took
            # 643s and the IP was under a second of it, so "the IP is slow" is a
            # hypothesis that needs a number before anyone acts on it.
            ip_started = time.monotonic()
            ip_selection = master.solve_ip(deadline=deadline)
            ip_elapsed_s = time.monotonic() - ip_started
            ip_selection = {
                flight_id: _canonical_column(column, graphs[flight_id], cfg)
                for flight_id, column in ip_selection.items()
            }
            _assert_claim_feasible(ip_selection, committed_loads, row_index)
            ip_objective = _selection_objective(ip_selection, params.M)
            ip_upper_bound = master.last_ip_bound
            ip_cost_upper_bound = total_benefit - ip_objective
            # Equation (11): the IP gap is measured against the LP UPPER BOUND -- a bound
            # on the full master problem -- not against the restricted IP's own bound,
            # which only certifies the columns already in the pool.
            ip_gap_revenue = _relative_revenue_gap(last_upper_bound, ip_objective)
            if ip_upper_bound is not None:
                ip_cost_lower_bound = total_benefit - ip_upper_bound
                ip_gap_cost = _relative_cost_gap(ip_cost_upper_bound, ip_cost_lower_bound)
            else:
                ip_gap_cost = None
            if params.gap_metric == "revenue":
                ip_gap = ip_gap_revenue
                ip_gap_met = ip_gap <= params.ip_gap
            elif ip_gap_cost is not None:
                ip_gap = ip_gap_cost
                ip_gap_met = ip_gap <= params.ip_gap
            ip_status = master.last_ip_status or "unknown"
            ip_optimal = master.last_ip_optimal
            if ip_optimal is False and termination_reason not in {
                "time_limit", "iteration_limit"
            }:
                # The restricted IP could not PROVE its selection optimal over the pool it
                # was given.  That is a different fact from the CG loop running out of
                # budget: the loop may have converged on `lp_gap` with only the final MILP
                # left uncertified, and only the second is fixed by raising the time limit.
                # Both mean an absent flight is unproven rather than physically impossible,
                # so the denial partition below treats them alike -- but the reason a run
                # reports is read by a person, and "time_limit" sent them to the wrong knob.
                #
                # The IP can also fail to prove because it ran out of WALL CLOCK -- it gets
                # only the small `ip_reserve_s` tail -- and that really is a budget
                # exhaustion, whatever the loop did before it.  Distinguish the two here
                # rather than calling both uncertified, or a sweep filtering runs on
                # `planner_termination == "time_limit"` silently drops them.
                termination_reason = (
                    "time_limit" if time.monotonic() >= deadline else "ip_not_proven"
                )
            if not incumbent or _better_selection(ip_selection, incumbent, params.M):
                incumbent = dict(ip_selection)

        # A ≤1 flight row keeps every RMP feasible.  Recover omitted flights in
        # planned-departure order by pricing with already saturated rows removed
        # from their DAG.  Failure here is the optimizer's budget denial verdict.
        incumbent = {
            flight_id: _canonical_column(column, graphs[flight_id], cfg)
            for flight_id, column in incumbent.items()
        }
        loads = _assert_claim_feasible(incumbent, committed_loads, row_index)
        repair_added = 0
        search_exhausted_flights: set[int] = set()
        repair_order = sorted(
            (graphs[flight_id] for flight_id in flight_ids if flight_id not in incumbent),
            key=lambda graph: (graph.request.t_departure, graph.request.flight_id),
        )
        saturated = {
            row for row, load in loads.items() if load >= row_index.cap(row)
        }
        for repair_index, graph in enumerate(repair_order):
            if time.monotonic() >= deadline:
                termination_reason = "time_limit"
                search_exhausted_flights.update(
                    pending.request.flight_id for pending in repair_order[repair_index:]
                )
                break
            try:
                _reduced_cost, repaired = price_flight(
                    graph,
                    {},
                    0.0,
                    cfg,
                    params,
                    forbidden_rows=saturated,
                    require_improving=False,
                    deadline=deadline,
                )
            except PricingTimeout:
                termination_reason = "time_limit"
                search_exhausted_flights.update(
                    pending.request.flight_id for pending in repair_order[repair_index:]
                )
                break
            if repaired is None:
                continue
            repaired = _canonical_column(repaired, graph, cfg)
            if any(loads[row] + 1 > row_index.cap(row) for row in repaired.claims):
                raise RuntimeError(
                    f"repair pricing returned an infeasible column for flight "
                    f"{graph.request.flight_id}"
                )
            incumbent[graph.request.flight_id] = repaired
            for row in repaired.claims:
                old_load = loads[row]
                new_load = old_load + 1
                loads[row] = new_load
                if old_load < row_index.cap(row) <= new_load:
                    saturated.add(row)
            repair_added += 1

        incumbent = dict(sorted(incumbent.items()))
        _assert_claim_feasible(incumbent, committed_loads, row_index)
        denied = tuple(flight_id for flight_id in flight_ids if flight_id not in incumbent)
        # Every truncated exit, by whatever mechanism: a flight the solve did not place is
        # then a compute-cap artifact, not the optimizer's verdict that no plan exists.
        if termination_reason in {"time_limit", "iteration_limit", "ip_not_proven"}:
            search_exhausted_flights.update(denied)
        search_exhausted = tuple(
            flight_id for flight_id in denied if flight_id in search_exhausted_flights
        )
        budget_denied = tuple(
            flight_id for flight_id in denied if flight_id not in search_exhausted_flights
        )
        # NOT seconds under every objective: `Column.delay_s` carries whatever currency the
        # cost model priced in, so at objective="total_cost" this is weighted cost units
        # (1*ground + 3*air).  Reported beside `objective_name` for that reason -- the run
        # folder's own delay metrics, computed from the filed intents, are the seconds.
        objective_value = math.fsum(column.delay_s for column in incumbent.values())
        master_objective = _selection_objective(incumbent, params.M)
        elapsed_s = time.monotonic() - started
        arc_stats: Counter[str] = Counter()
        for graph in graphs.values():
            arc_stats.update(graph.arc_cache_stats)
        corridor_domains_materialized = sum(
            bool(getattr(graph.corridor_cells, "is_materialized", True))
            for graph in graphs.values()
        )

        materialized_rows = getattr(master, "materialized_rows", ())
        stats: dict[str, Any] = {
            "backend": _backend_name(master),
            "iterations": iterations,
            "termination_reason": termination_reason,
            "lp_objectives": tuple(lp_objectives),
            "lower_bounds": tuple(lp_objectives),
            "upper_bounds": tuple(upper_bounds),
            "cost_upper_bounds": tuple(cost_upper_bounds),
            "cost_lower_bounds": tuple(cost_lower_bounds),
            "lp_gaps": tuple(lp_gaps),
            "heuristic_objectives": tuple(heuristic_objectives),
            "heuristic_costs": tuple(heuristic_costs),
            "final_lp_objective": last_lp_objective,
            "upper_bound": last_upper_bound,
            "cost_upper_bound": total_benefit - last_lp_objective,
            "cost_lower_bound": final_cost_lower_bound,
            "lp_gap": (
                (
                    _relative_revenue_gap(last_upper_bound, last_lp_objective)
                    if params.gap_metric == "revenue"
                    else _relative_cost_gap(
                        total_benefit - last_lp_objective, final_cost_lower_bound
                    )
                )
                if math.isfinite(last_lp_objective)
                else math.inf
            ),
            "lp_gap_revenue": (
                _relative_revenue_gap(last_upper_bound, last_lp_objective)
                if math.isfinite(last_lp_objective)
                else math.inf
            ),
            "lp_gap_cost": (
                _relative_cost_gap(
                    total_benefit - last_lp_objective, final_cost_lower_bound
                )
                if math.isfinite(last_lp_objective)
                else math.inf
            ),
            "ip_gap_revenue": ip_gap_revenue,
            "gap_metric": params.gap_metric,
            "heuristic_objective": heuristic_objective,
            "heuristic_cost": heuristic_cost,
            # Native IP diagnostics describe the final restricted master.
            # ``cost_lower_bound`` above remains the global pricing bound.
            "ip_objective": ip_objective,
            "ip_upper_bound": ip_upper_bound,
            "ip_cost_lower_bound": ip_cost_lower_bound,
            "ip_cost_upper_bound": ip_cost_upper_bound,
            "ip_gap": ip_gap,
            "ip_gap_met": ip_gap_met,
            "ip_status": ip_status,
            "ip_optimal": ip_optimal,
            "ip_skipped": ip_skipped,
            "ip_elapsed_s": ip_elapsed_s,
            # ``objective`` is the user-facing minimization objective.  The
            # maximize-sense master value is retained under an explicit name.
            "objective": objective_value,
            "objective_name": params.objective,
            "master_objective": master_objective,
            "selected_flights": len(incumbent),
            "denied_flight_ids": denied,
            "budget_denied_flight_ids": budget_denied,
            "search_exhausted_flight_ids": search_exhausted,
            "repair_added": repair_added,
            "seedless_flight_ids": tuple(sorted(seedless_flights)),
            "initial_heuristic_strategy": initial_heuristic_strategy,
            "initial_heuristic_flights": len(initial_heuristic),
            "initial_heuristic_delay_s": math.fsum(
                column.delay_s for column in initial_heuristic.values()
            ),
            "initial_greedy_completed": greedy_completed,
            "initial_greedy_flights": len(greedy_heuristic),
            "initial_greedy_elapsed_s": greedy_elapsed_s,
            "initial_seed_columns": len(seeds),
            "graphs_built": len(graphs),
            "seeds_completed": len(seeds),
            "seed_flights_processed": len(flight_ids),
            "preprocessing_stage": "complete",
            "graph_build_elapsed_s": graph_build_elapsed_s,
            "seed_elapsed_s": seed_elapsed_s,
            "time_to_master_s": time_to_master_s,
            "corridor_domains_materialized": corridor_domains_materialized,
            "static_terminal_count": len(static_catalog.entries),
            "static_excluded_cells": static_catalog.excluded_cell_count,
            "arc_expanded_nodes": arc_stats["expanded_nodes"],
            "arc_checks": arc_stats["arc_checks"],
            "arc_cache_hits": arc_stats["cache_hits"],
            "arc_allowed": arc_stats["allowed_arcs"],
            "arc_blocked": arc_stats["blocked_arcs"],
            "wall_index_queries": static_catalog.wall_index.stats["queries"],
            "wall_index_candidates": static_catalog.wall_index.stats["candidates"],
            "pricing_flights_completed": pricing_flights_completed,
            "pricing_sweeps_completed": pricing_sweeps_completed,
            "pricing_timeout_flight_id": pricing_timeout_flight_id,
            # Total pricing wall across every sweep.  Pricing is where a solve spends
            # its time, so this against `elapsed_s` says how much of the run was the
            # subproblem and how much was the master.
            "pricing_wall_s": pricing_wall_s,
            "n_columns": len(master.columns),
            "seeded_columns": seeded_columns,
            "n_materialized_rows": len(materialized_rows),
            "lazy_rows_added": lazy_rows_added,
            "lazy_row_rounds": lazy_row_rounds,
            "time_limit_overrun_s": max(0.0, elapsed_s - params.time_limit_s),
            "elapsed_s": elapsed_s,
        }
        return ColGenResult(columns=incumbent, stats=stats)


__all__ = ["ColGenResult", "ColGenSolver"]
