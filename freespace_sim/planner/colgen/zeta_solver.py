"""Explicit experimental CG driver for simulator-backed any-angle pricing.

The production planner registry and DP solver are deliberately independent of this
entry point. Capacity mathematics remains owned by ``RestrictedMaster``.
"""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np

from ...config import SimConfig
from ...types import FlightRequest, OperationalIntent, Terminal, as_terminal
from ...verify import find_interflight_conflict
from .master import BackendTimeout, RestrictedMaster
from .network import FlightGraphInfeasible, RowIndex, RowKey, StaticTerminalCatalog, build_flight_graph
from .objective import cost_model
from .params import ColGenParams
from .translate import Column
from .zeta import ZetaGraph, ZetaOptions, price_zeta
from .zeta_pool import ZetaPricingPool, ZetaSeedResult, build_seed_columns


@dataclass(frozen=True)
class ZetaSolveResult:
    """Selected experimental routes, independently audited intents and solve evidence.

    Attributes
    ------------
    - columns (dict[int, Column]): one selected route per served flight.
    - intents (dict[int, OperationalIntent]): matching simulator reservations.
    - stats (dict[str, Any]): timings, global bounds and completion diagnostics.
    - master (RestrictedMaster): complete generated pool for bounded analysis.
    """

    columns: dict[int, Column]
    intents: dict[int, OperationalIntent]
    stats: dict[str, Any]
    master: RestrictedMaster


def _terminal_catalog(
    requests: Sequence[FlightRequest],
    static_terms: Iterable[tuple[Sequence[float], Terminal]],
    cfg: SimConfig,
) -> StaticTerminalCatalog:
    """Include every endpoint terminal in permanent protection, rejecting disagreement.

    Parameters
    ------------
    - requests (Sequence[FlightRequest]): flights whose endpoint metadata is authoritative.
    - static_terms (Iterable[tuple[Sequence[float], Terminal]]): additional permanent terminals.
    - cfg (SimConfig): geometry shared with the route builders.

    Return
    --------
    - output (StaticTerminalCatalog): frozen complete permanent-terminal catalogue.
    """
    entries: dict[Hashable, tuple[tuple[float, ...], Terminal]] = {}
    supplied = list(static_terms)
    for request in requests:
        for point, raw in (
            (request.origin, request.origin_terminal),
            (request.dest, request.dest_terminal),
        ):
            terminal = as_terminal(raw)
            if terminal is not None:
                supplied.append((point, terminal))
    for point, raw in supplied:
        terminal = as_terminal(raw)
        if terminal is None:
            raise ValueError("permanent terminal metadata cannot be None")
        xyz = tuple(float(v) for v in point)
        xyz = (*xyz, cfg.ground_level_m) if len(xyz) == 2 else xyz
        entry = xyz, terminal
        previous = entries.get(terminal.id)
        if previous is not None and previous != entry:
            raise ValueError(f"inconsistent geometry or metadata for terminal {terminal.id!r}")
        entries[terminal.id] = entry
    return StaticTerminalCatalog(entries.values(), cfg)


def _keep_incumbent(
    master: RestrictedMaster,
    incumbent: dict[int, Column],
    candidate: dict[int, Column],
) -> dict[int, Column]:
    """Retain only a feasible objective improvement and record its warm start."""
    if master.objective_of(candidate) > master.objective_of(incumbent) + 1e-9:
        incumbent = candidate
    master.set_heuristic(incumbent)
    return incumbent


def _spatial_key(column: Column) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Ignore departure shifts and redundant forward-collinear waypoint knots."""
    corners: list[tuple[int, int]] = []
    for point in column.cell_path:
        while len(corners) >= 2:
            first, last = corners[-2:]
            u = last[0] - first[0], last[1] - first[1]
            v = point[0] - last[0], point[1] - last[1]
            if u[0] * v[1] != u[1] * v[0] or u[0] * v[0] + u[1] * v[1] <= 0:
                break
            corners.pop()
        corners.append(point)
    return column.flight_id, tuple(corners)


def solve_zeta(
    requests: Sequence[FlightRequest],
    cfg: SimConfig,
    static_terms: Iterable[tuple[Sequence[float], Terminal]],
    params: ColGenParams,
    *,
    options: ZetaOptions = ZetaOptions(),
    fixed_claims: Sequence[frozenset[RowKey]] = (),
    seed_columns: Mapping[int, Sequence[Column]] | None = None,
) -> ZetaSolveResult:
    """Run bounded live column generation with exact simulator-backed Zeta pricing.

    Parameters
    ------------
    - requests (Sequence[FlightRequest]): flights optimized jointly; ids must be unique.
    - cfg (SimConfig): unchanged simulator speed, geometry, terminal and departure rules.
    - static_terms (Iterable[tuple[Sequence[float], Terminal]]): permanent terminal walls.
    - params (ColGenParams): objective, departure ladder, iteration and solver budgets.
    - options (ZetaOptions): declared finite waypoint connections and pricing work caps.
    - fixed_claims (Sequence[frozenset[RowKey]]): prior reservations converted with the same
      full-volume claim semantics; their capacities remain unavailable to this batch.
    - seed_columns (Mapping[int, Sequence[Column]] | None): optional routes, recanonicalized
      before entering the pool. The prototype does not accept legacy DP claim proxies.

    Return
    --------
    - output (ZetaSolveResult): selected audited routes and evidence, including explicit
      incomplete-search status and a safe global bound even when a work cap is reached.
    """
    started = time.monotonic()
    ordered = tuple(sorted(requests, key=lambda request: request.flight_id))
    flight_ids = tuple(request.flight_id for request in ordered)
    if len(set(flight_ids)) != len(flight_ids):
        raise ValueError("flight ids must be unique")
    if not params.seed_nominal_routes and not any((seed_columns or {}).values()):
        raise ValueError("seed_nominal_routes=False requires supplied columns")
    deadline = started + params.time_limit_s
    reserve = params.ip_reserve_s
    if reserve is None:
        reserve = min(5.0, 0.05 * params.time_limit_s)
    pricing_deadline = deadline - reserve
    model = cost_model(cfg, params)
    catalog = _terminal_catalog(ordered, static_terms, cfg)
    row_index = RowIndex()
    for _point, terminal in catalog.entries:
        row_index.register_terminal(terminal)
    fixed_loads = Counter(row for claims in fixed_claims for row in frozenset(claims))
    master = RestrictedMaster(flight_ids, row_index, params, fixed_loads=fixed_loads)
    forbidden = frozenset(row for row, load in fixed_loads.items() if load >= row_index.cap(row))
    graphs: dict[int, ZetaGraph] = {}
    infeasible: dict[int, str] = {}
    graph_started = time.monotonic()
    for request in ordered:
        if time.monotonic() >= pricing_deadline:
            break
        try:
            fg = build_flight_graph(request, cfg, catalog, params)
            graphs[request.flight_id] = ZetaGraph(fg, cfg, row_index, options)
        except FlightGraphInfeasible as exc:
            infeasible[request.flight_id] = str(exc)
    graph_s = time.monotonic() - graph_started
    pool_by_flight: dict[int, list[Column]] = {flight_id: [] for flight_id in flight_ids}
    seen: set[Column] = set()
    provided_primary: dict[int, Column] = {}
    seed_count = 0
    seed_failure: str | None = None
    seed_records: list[dict[str, Any]] = []
    seed_task_s = seed_setup_s = 0.0
    rng = np.random.default_rng(cfg.seed)
    history: list[dict[str, Any]] = []
    lp_s = pricing_s = insertion_s = 0.0
    global_upper = len(flight_ids) * params.M
    lp_objective: float | None = None
    complete = False
    pricing_bound_certified = False
    termination = "max_iterations"
    total_expanded = total_generated = capped_searches = 0
    early_return_searches = no_improvement_searches = 0
    edge_evaluations = successor_pops = 0
    history_pruned = history_expired_rows = history_comparisons = 0
    equivalent_histories_merged = 0
    shorter_histories_merged = 0
    max_merge_error_bound = 0.0
    lattice_bound_calls = 0
    bound_build_s = 0.0
    bound_lookups = bound_pruned = 0
    lp_current = False
    workers = ZetaPricingPool(graphs, params) if params.n_pricing_workers and graphs else None
    try:
        seed_started = time.monotonic()
        if workers is not None:
            seed_sweep = workers.run_seed_sweep(seed_columns or {}, forbidden, pricing_deadline)
            seed_results = seed_sweep.results
            seed_failure = seed_sweep.failure
            seed_records = list(seed_sweep.flight_records)
            seed_task_s = seed_sweep.task_s
            seed_setup_s = seed_sweep.setup_s
        else:
            serial_seeds: list[tuple[int, ZetaSeedResult]] = []
            for flight_id, graph in graphs.items():
                task_started = time.monotonic()
                if task_started >= pricing_deadline:
                    seed_failure = "deadline"
                    break
                seed = build_seed_columns(
                    graph, params, (seed_columns or {}).get(flight_id, ()),
                    forbidden, pricing_deadline,
                )
                task_s = time.monotonic() - task_started
                seed_task_s += task_s
                serial_seeds.append((flight_id, seed))
                seed_records.append({
                    "flight_id": flight_id, "worker": None, "pid": os.getpid(),
                    "task_s": task_s, "columns": len(seed.columns),
                    "complete": seed.complete, "reason": seed.reason,
                })
                if not seed.complete:
                    seed_failure = seed.reason
                    break
            seed_results = tuple(serial_seeds)
        for flight_id, seed in seed_results:
            for column in seed.columns:
                if column in seen:
                    continue
                master.add_column(column)
                seen.add(column)
                pool_by_flight[flight_id].append(column)
                seed_count += 1
            if seed.provided_primary is not None:
                provided_primary[flight_id] = seed.provided_primary
        seed_s = time.monotonic() - seed_started
        seed_complete = (seed_failure is None and len(seed_results) == len(graphs)
                         and len(graphs) + len(infeasible) == len(flight_ids)
                         and all(seed.complete for _, seed in seed_results))
        seed_spatial_keys = {_spatial_key(column) for column in master.columns}
        incumbent = master.complete_selection({})
        master.set_heuristic(incumbent)
        if provided_primary and master.is_claim_feasible(provided_primary):
            incumbent = _keep_incumbent(master, incumbent, master.complete_selection(provided_primary))
        if seed_failure is not None:
            termination = "seed_" + seed_failure
        for iteration in range(1, params.max_iterations + 1):
            if seed_failure is not None:
                break
            if time.monotonic() >= pricing_deadline:
                termination = "time_limit"
                break
            lp_started = time.monotonic()
            master.materialize_bindable_rows()
            master.backend.time_limit_s = max(1e-6, pricing_deadline - time.monotonic())
            try:
                lp_objective, duals, x = master.solve_lp()
            except BackendTimeout:
                termination = "lp_time_limit"
                lp_s += time.monotonic() - lp_started
                break
            lp_s += time.monotonic() - lp_started
            lp_current = True
            incumbent = _keep_incumbent(master, incumbent, master.round_heuristic(x, rng))
            proposals: list[Column] = []
            bounds: list[float] = []
            statuses: dict[int, str] = {}
            sweep_complete = len(graphs) + len(infeasible) == len(flight_ids)
            sweep_started = time.monotonic()
            sweep_failure = None
            flight_pricing: list[dict[str, Any]] = []
            if workers is not None:
                sweep = workers.run_sweep(
                    duals, master.flight_duals, pool_by_flight, forbidden, pricing_deadline
                )
                results = sweep.results
                flight_pricing = list(sweep.flight_records)
                sweep_failure = sweep.termination_reason
                sweep_complete = sweep_complete and sweep.complete
            else:
                serial_results = []
                for flight_id, graph in graphs.items():
                    if time.monotonic() >= pricing_deadline:
                        sweep_complete = False
                        sweep_failure = "deadline"
                        break
                    task_started = time.monotonic()
                    result = price_zeta(
                        graph,
                        duals,
                        master.flight_duals[flight_id],
                        params,
                        known_columns=pool_by_flight[flight_id],
                        forbidden_rows=forbidden,
                        deadline=pricing_deadline,
                    )
                    serial_results.append((flight_id, result))
                    flight_pricing.append({
                        "flight_id": flight_id, "worker": None, "pid": os.getpid(),
                        "task_s": time.monotonic() - task_started,
                        "expanded": result.expanded, "complete": result.complete,
                        "history_pruned": result.history_pruned,
                        "history_expired_rows": result.history_expired_rows,
                        "history_comparisons": result.history_comparisons,
                        "bound_build_s": result.bound_build_s,
                        "bound_table_states": result.bound_table_states,
                        "bound_lookups": result.bound_lookups,
                        "bound_pruned": result.bound_pruned,
                    })
                results = tuple(serial_results)
            priced = len(results)
            for flight_id, result in results:
                graph = graphs[flight_id]
                bounds.append(float(result.upper_bound))
                statuses[flight_id] = result.reason
                sweep_complete = sweep_complete and result.complete
                total_expanded += result.expanded
                total_generated += result.generated
                edge_evaluations += result.edge_evaluations
                successor_pops += result.successor_pops
                history_pruned += result.history_pruned
                history_expired_rows += result.history_expired_rows
                history_comparisons += result.history_comparisons
                equivalent_histories_merged += result.equivalent_histories_merged
                shorter_histories_merged += result.shorter_histories_merged
                max_merge_error_bound = max(max_merge_error_bound, result.merge_error_bound)
                lattice_bound_calls += result.lattice_bound_calls
                bound_build_s += result.bound_build_s
                bound_lookups += result.bound_lookups
                bound_pruned += result.bound_pruned
                early_return_searches += int(result.reason == "improving_column")
                no_improvement_searches += int(result.reason == "no_improvement")
                capped_searches += int(not result.complete and result.reason != "improving_column")
                if result.column is not None and result.reduced_cost > params.epsilon:
                    column = graph.canonical_column(result.column, model)
                    if column is None or not column.claims.isdisjoint(forbidden):
                        raise RuntimeError("pricing returned an invalid candidate")
                    rc = model.reduced_cost(
                        benefit=params.M,
                        cost=column.delay_s,
                        dual_cost=math.fsum(duals.get(row, 0.0) for row in column.claims),
                        pi_f=master.flight_duals[flight_id],
                    )
                    if abs(rc - result.reduced_cost) > 1e-6:
                        raise RuntimeError("pricing and canonical reduced costs disagree")
                    if rc > params.epsilon and column not in seen:
                        proposals.append(column)
            sweep_s = time.monotonic() - sweep_started
            pricing_s += sweep_s
            all_priced = priced == len(graphs) and len(graphs) + len(infeasible) == len(flight_ids)
            if all_priced and all(not math.isnan(value) and value != math.inf for value in bounds):
                corrected = lp_objective + math.fsum(max(0.0, value) for value in bounds)
                global_upper = min(global_upper, corrected)
                pricing_bound_certified = True
            insert_started = time.monotonic()
            for column in proposals:
                master.add_column(column)
                seen.add(column)
                pool_by_flight[column.flight_id].append(column)
            insertion_s += time.monotonic() - insert_started
            lp_current = not proposals
            history.append({
                "iteration": iteration,
                "lp_cost": len(flight_ids) * params.M - lp_objective,
                "global_cost_lower_bound": len(flight_ids) * params.M - global_upper,
                "pricing_s": sweep_s,
                "new_columns": len(proposals),
                "max_reduced_cost_upper_bound": max(bounds, default=0.0),
                "pricing_complete": sweep_complete,
                "early_return_searches": sum(result.reason == "improving_column"
                                             for _, result in results),
                "no_improvement_searches": sum(result.reason == "no_improvement"
                                               for _, result in results),
                "search_statuses": statuses,
                "flight_pricing": flight_pricing,
                "sweep_failure": sweep_failure,
                "history_pruned": sum(result.history_pruned for _, result in results),
                "history_expired_rows": sum(result.history_expired_rows for _, result in results),
                "history_comparisons": sum(result.history_comparisons for _, result in results),
                "bound_build_s": sum(result.bound_build_s for _, result in results),
                "bound_table_states": sum(result.bound_table_states for _, result in results),
                "bound_lookups": sum(result.bound_lookups for _, result in results),
                "bound_pruned": sum(result.bound_pruned for _, result in results),
            })
            if sweep_failure is not None:
                termination = "pricing_" + sweep_failure
                break
            if not sweep_complete:
                if not proposals:
                    termination = "pricing_limit"
                    break
                # Useful incomplete proposals can change the duals, but cannot prove convergence.
                continue
            if not proposals:
                if max(bounds, default=0.0) > params.epsilon + 1e-7:
                    raise RuntimeError("complete pricing stalled with an improving upper bound")
                complete = True
                termination = "converged"
                break
    finally:
        if workers is not None:
            workers.close()
    if not lp_current and time.monotonic() < pricing_deadline:
        lp_started = time.monotonic()
        master.materialize_bindable_rows()
        master.backend.time_limit_s = max(1e-6, pricing_deadline - time.monotonic())
        try:
            lp_objective, _duals, x = master.solve_lp()
            incumbent = _keep_incumbent(master, incumbent, master.round_heuristic(x, rng))
        except BackendTimeout:
            pass
        lp_s += time.monotonic() - lp_started
    final_started = time.monotonic()
    master.backend.time_limit_s = params.time_limit_s
    if time.monotonic() < deadline:
        incumbent = master.solve_ip(
            incumbent, deadline=deadline, budget_s=params.ip_time_limit_s, eager=True
        )
    final_ip_s = time.monotonic() - final_started
    audit_started = time.monotonic()
    if not master.is_claim_feasible(incumbent):
        raise RuntimeError("selected Zeta schedule overloads a reservation row")
    intents = {flight_id: graphs[flight_id].intent(column) for flight_id, column in incumbent.items()}
    if any(not intent.accepted for intent in intents.values()):
        raise RuntimeError("selected Zeta route failed final translation")
    conflict = find_interflight_conflict(list(intents.values()), cfg, catalog.entries)
    if conflict is not None:
        raise RuntimeError(f"selected Zeta reservations collide: {conflict}")
    audit_s = time.monotonic() - audit_started
    cost = len(flight_ids) * params.M - master.objective_of(incumbent)
    lower = max(0.0, len(flight_ids) * params.M - global_upper)
    if lower > cost + 1e-5:
        raise RuntimeError("global pricing bound exceeds the feasible schedule cost")
    ip_bound = master.last_ip_bound
    stats: dict[str, Any] = {
        "termination_reason": termination,
        "pricing_search_complete": complete,
        "pricing_bound_certified": pricing_bound_certified,
        "global_bound_valid": True,
        "global_cost_lower_bound": lower,
        "global_integer_gap": max(0.0, cost - lower) / max(1.0, abs(cost)),
        "lp_cost": None if lp_objective is None else len(flight_ids) * params.M - lp_objective,
        "objective_cost": cost,
        "served": len(incumbent),
        "denied": len(flight_ids) - len(incumbent),
        "iterations": len(history),
        "columns": len(master.columns),
        "seed_columns": seed_count,
        "new_spatial_routes": len({_spatial_key(column) for column in master.columns} - seed_spatial_keys),
        "selected_new_spatial_routes": sum(_spatial_key(column) not in seed_spatial_keys for column in incumbent.values()),
        "unique_rows": len({row for column in master.columns for row in column.claims}),
        "nonzeros": sum(len(column.claims) for column in master.columns),
        "search_expanded": total_expanded,
        "search_generated": total_generated,
        "edge_evaluations": edge_evaluations,
        "successor_pops": successor_pops,
        "history_pruned": history_pruned,
        "equivalent_histories_merged": equivalent_histories_merged,
        "shorter_histories_merged": shorter_histories_merged,
        "max_merge_error_bound": max_merge_error_bound,
        "lattice_bound_calls": lattice_bound_calls,
        "history_expired_rows": history_expired_rows,
        "history_comparisons": history_comparisons,
        "bound_build_s": bound_build_s,
        "bound_lookups": bound_lookups,
        "bound_pruned": bound_pruned,
        "capped_searches": capped_searches,
        "early_return_searches": early_return_searches,
        "no_improvement_searches": no_improvement_searches,
        "graph_infeasible": infeasible,
        "graph_build_wall_s": graph_s,
        "seed_wall_s": seed_s,
        "seed_complete": seed_complete,
        "seed_failure": seed_failure,
        "seed_flights_returned": len(seed_results),
        "seed_task_s": seed_task_s,
        "seed_pool_setup_s": seed_setup_s,
        "seed_flight_records": seed_records,
        "lp_wall_s": lp_s,
        "pricing_wall_s": pricing_s,
        "pricing_workers": 0 if workers is None else min(params.n_pricing_workers, len(graphs)),
        "pricing_pool_setup_s": 0.0 if workers is None else workers.setup_s,
        "pricing_pool_close_s": 0.0 if workers is None else workers.close_s,
        "pricing_worker_peak_rss_bytes": {} if workers is None else workers.worker_peak_rss_bytes,
        "insertion_wall_s": insertion_s,
        "final_ip_wall_s": final_ip_s,
        "audit_wall_s": audit_s,
        "elapsed_s": time.monotonic() - started,
        "ip_status": master.last_ip_status,
        "ip_optimal": master.last_ip_optimal,
        "restricted_ip_cost_lower_bound": (
            None if ip_bound is None or not math.isfinite(ip_bound)
            else len(flight_ids) * params.M - ip_bound
        ),
        "collision_pair": conflict,
        "violated_rows": 0,
        "seed_rule": "all base-step lane-pair shortest variants; primary departure ladder",
        "timing_rule": ("nearest speed-feasible arrival grid; nominal fallback"
                        if options.arrival_snap_s else "nominal-speed horizontal segments"),
        "arrival_snap_s": options.arrival_snap_s,
        "speed_fraction": options.speed_fraction,
        "expire_history": options.expire_history,
        "history_dominance": options.history_dominance,
        "merge_equivalent_histories": options.merge_equivalent_histories,
        "merge_shorter_histories": options.merge_shorter_histories,
        "forward_cruise": options.forward_cruise,
        "pricing_traversal": options.traversal,
        "native_state_index": options.native_state_index,
        "lattice_distance_bound": options.lattice_distance_bound,
        "early_return": options.early_return,
        "row_rule": "complete simulator box/cylinder coverage and terminal pad rows",
        "connection_radius_m": options.connection_radius_m,
        "connection_mode": options.connection_mode,
        "certificate_scope": (
            ("forward-only cruise; " if options.forward_cruise else "") +
            ("configured connection graph union fixed supplied seed pool"
             if options.seed_validation_radius_m is not None else
             "full configured radius" if options.connection_mode == "full" else
             "restricted connection graph only")
        ),
        "seed_validation_radius_m": options.seed_validation_radius_m,
        "pricing_backend": options.backend,
        "compiled_loop": options.compiled_loop,
        "lazy_successors": options.lazy_successors and options.traversal != "forward_dp",
        "requested_lazy_successors": options.lazy_successors,
        "history": history,
    }
    return ZetaSolveResult(incumbent, intents, stats, master)
