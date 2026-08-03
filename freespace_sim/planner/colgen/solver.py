"""Whole-schedule column-generation orchestration.

This module deliberately contains no capacity-row geometry and no optimization
backend details.  It wires the three owners together instead: ``network``
certifies canonical claims, ``pricing`` searches one flight's finite DAG, and
``master`` solves the restricted master problem.  Keeping that boundary sharp
also makes the final repair pass use exactly the same column universe and row
semantics as the LP/IP loop.
"""

from __future__ import annotations

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
from .network import FlightGraph, RowIndex, RowKey, build_flight_graph, column_claims
from .params import ColGenParams
from .pricing import PricingTimeout, price_flight, seed_column
from .translate import Column

_FEASIBILITY_TOL = 1e-7
_REDUCED_COST_TOL = 1e-9
_OBJECTIVE_TOL = 1e-8


@dataclass(frozen=True, slots=True)
class ColGenResult:
    """Selected columns and diagnostics for one whole-schedule solve.

    A flight absent from :attr:`columns` is either an exhaustive optimizer
    denial or a compute-cap artifact.  ``budget_denied_flight_ids`` and
    ``search_exhausted_flight_ids`` partition those cases for Phase 3 filing.
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


def _shift_column(column: Column, departure_step: int, cfg: SimConfig) -> Column:
    """Return the same certified spatial route at a later integer departure."""

    delta = departure_step - column.departure_step
    if delta < 0:
        raise ValueError("a seed column can only be shifted later")
    return replace(
        column,
        departure_step=departure_step,
        delay_s=column.delay_s + delta * cfg.dt_s,
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
            candidate = _shift_column(seed, departure_step, cfg)
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
) -> tuple[dict[int, Column], bool]:
    """Build a route-aware FCFS incumbent with exact forbidden-row pricing.

    The shifted-seed schedule above is a cheap fallback, but it cannot route
    around a busy cell.  This pass asks the same exact pricing DAG for the
    minimum-delay column avoiding rows already saturated by earlier flights.
    It is only an incumbent heuristic: timeout leaves a valid partial schedule
    and never changes the master universe or any global bound.
    """

    selected: dict[int, Column] = {}
    loads: Counter[RowKey] = Counter(fixed_loads)
    order = sorted(
        graphs.values(),
        key=lambda graph: (graph.request.t_departure, graph.request.flight_id),
    )
    for graph in order:
        if time.monotonic() >= deadline:
            return dict(sorted(selected.items())), False
        saturated = frozenset(row for row, load in loads.items() if load >= row_index.cap(row))
        try:
            _reduced_cost, candidate = price_flight(
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
            return dict(sorted(selected.items())), False
        if candidate is None:
            continue
        candidate = _canonical_column(candidate, graph, cfg)
        if any(loads[row] + 1 > row_index.cap(row) for row in candidate.claims):
            raise RuntimeError(
                f"greedy pricing returned an infeasible column for flight "
                f"{graph.request.flight_id}"
            )
        selected[graph.request.flight_id] = candidate
        loads.update(candidate.claims)
    return dict(sorted(selected.items())), True


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
) -> ColGenResult:
    """Return an explicit compute-cap verdict before a usable master exists."""

    elapsed_s = time.monotonic() - started
    denied = tuple(flight_ids)
    heuristic_cost = len(denied) * params.M
    return ColGenResult(
        columns={},
        stats={
            "backend": "none",
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
            "total_delay_s": 0.0,
            "master_objective": 0.0,
            "selected_flights": 0,
            "denied_flight_ids": denied,
            "budget_denied_flight_ids": (),
            "search_exhausted_flight_ids": denied,
            "repair_added": 0,
            "seedless_flight_ids": (),
            "initial_heuristic_strategy": "time_limit",
            "initial_heuristic_flights": 0,
            "initial_heuristic_delay_s": 0.0,
            "initial_greedy_completed": False,
            "initial_greedy_flights": 0,
            "initial_greedy_elapsed_s": 0.0,
            "pricing_flights_completed": 0,
            "pricing_sweeps_completed": 0,
            "pricing_timeout_flight_id": None,
            "n_columns": 0,
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
    ) -> ColGenResult:
        started = time.monotonic()
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
                    "total_delay_s": 0.0,
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
        static_term_snapshot = tuple(static_terms)
        graphs: dict[int, FlightGraph] = {}
        for request in ordered_requests:
            if time.monotonic() >= pricing_deadline:
                return _pre_master_timeout_result(flight_ids, started, params)
            graphs[request.flight_id] = build_flight_graph(
                request,
                cfg,
                static_term_snapshot,
                params,
            )
        if time.monotonic() >= pricing_deadline:
            return _pre_master_timeout_result(flight_ids, started, params)

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
        seedless_flights: set[int] = set()
        seeds: dict[int, Column] = {}
        for flight_id in flight_ids:
            try:
                seed = seed_column(graphs[flight_id], cfg, deadline=pricing_deadline)
            except PricingTimeout:
                return _pre_master_timeout_result(flight_ids, started, params)
            except ValueError:
                # A disconnected/static-blocked graph is a legitimate
                # optimizer-verdict denial, not a batch-wide solver failure.
                # Its <=1 flight row can remain empty while other flights run.
                seedless_flights.add(flight_id)
                continue
            seed = _canonical_column(seed, graphs[flight_id], cfg)
            seeds[flight_id] = seed
            master.add_column(seed)

        shifted_seed_heuristic = _initial_feasible_selection(
            seeds,
            graphs,
            committed_loads,
            row_index,
            cfg,
            deadline=pricing_deadline,
        )
        greedy_started = time.monotonic()
        greedy_deadline = min(
            pricing_deadline,
            started + min(60.0, 0.55 * params.time_limit_s),
        )
        greedy_heuristic, greedy_completed = _greedy_feasible_selection(
            graphs,
            committed_loads,
            row_index,
            cfg,
            params,
            deadline=greedy_deadline,
        )
        greedy_elapsed_s = time.monotonic() - greedy_started
        initial_heuristic = dict(shifted_seed_heuristic)
        initial_heuristic_strategy = "shifted_seeds"
        if _better_selection(greedy_heuristic, initial_heuristic, params.M):
            initial_heuristic = dict(greedy_heuristic)
            initial_heuristic_strategy = "greedy_pricing"
        best_heuristic = dict(initial_heuristic)
        for column in initial_heuristic.values():
            master.add_column(column)
        if best_heuristic:
            master.set_heuristic(best_heuristic)

        rng = np.random.default_rng(cfg.seed)
        total_benefit = len(flight_ids) * params.M
        lp_objectives: list[float] = []
        upper_bounds: list[float] = []
        cost_upper_bounds: list[float] = []
        cost_lower_bounds: list[float] = []
        lp_gaps: list[float] = []
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
                    lp_objective, capacity_duals, x = master.solve_lp()
                except BackendTimeout:
                    break
                added_rows = master.add_violated_rows(x, _FEASIBILITY_TOL)
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

            heuristic = master.round_heuristic(
                last_x,
                rng,
                params.n_heuristic_tries,
            )
            heuristic = {
                flight_id: _canonical_column(column, graphs[flight_id], cfg)
                for flight_id, column in heuristic.items()
            }
            _assert_claim_feasible(heuristic, committed_loads, row_index)
            if not best_heuristic or _better_selection(heuristic, best_heuristic, params.M):
                best_heuristic = dict(heuristic)
                master.set_heuristic(best_heuristic)

            best_reduced_costs: list[float] = []
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
            for flight_id in pricing_order:
                try:
                    reduced_cost, column = price_flight(
                        graphs[flight_id],
                        capacity_duals,
                        master.flight_duals[flight_id],
                        cfg,
                        params,
                        deadline=pricing_deadline,
                    )
                except PricingTimeout:
                    pricing_complete = False
                    pricing_timeout_flight_id = flight_id
                    break
                pricing_flights_completed += 1
                best_reduced_costs.append(float(reduced_cost))
                if column is not None and reduced_cost > _REDUCED_COST_TOL:
                    priced_columns.append(_canonical_column(column, graphs[flight_id], cfg))

            before = len(master.columns)
            for column in sorted(
                priced_columns, key=lambda item: (item.flight_id, _column_key(item))
            ):
                master.add_column(column)
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
            lp_gap = _relative_cost_gap(cost_upper_bound, cost_lower_bound)
            heuristic_objective = _selection_objective(best_heuristic, params.M)
            heuristic_cost = total_benefit - heuristic_objective
            heuristic_gap = _relative_cost_gap(heuristic_cost, cost_lower_bound)

            lp_objectives.append(last_lp_objective)
            upper_bounds.append(last_upper_bound)
            cost_upper_bounds.append(cost_upper_bound)
            cost_lower_bounds.append(cost_lower_bound)
            lp_gaps.append(lp_gap)
            heuristic_objectives.append(heuristic_objective)
            heuristic_costs.append(heuristic_cost)

            if lp_gap <= params.lp_gap:
                termination_reason = "lp_gap"
                break
            if best_heuristic and heuristic_gap <= params.ip_gap:
                termination_reason = "heuristic_gap"
                break
            if not priced_columns:
                termination_reason = "no_improving_columns"
                break

            if len(master.columns) == before:
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
        ip_gap_met: bool | None = None
        ip_status = "skipped"
        ip_optimal: bool | None = None
        if not ip_skipped and time.monotonic() >= deadline:
            ip_skipped = True
            ip_status = "time_limit_skipped"
            termination_reason = "time_limit"
        elif not ip_skipped:
            master.backend.time_limit_s = max(1e-6, deadline - time.monotonic())
            master.set_heuristic(incumbent)
            ip_selection = master.solve_ip(deadline=deadline)
            ip_selection = {
                flight_id: _canonical_column(column, graphs[flight_id], cfg)
                for flight_id, column in ip_selection.items()
            }
            _assert_claim_feasible(ip_selection, committed_loads, row_index)
            ip_objective = _selection_objective(ip_selection, params.M)
            ip_upper_bound = master.last_ip_bound
            ip_cost_upper_bound = total_benefit - ip_objective
            if ip_upper_bound is not None:
                ip_cost_lower_bound = total_benefit - ip_upper_bound
                ip_gap = _relative_cost_gap(ip_cost_upper_bound, ip_cost_lower_bound)
                ip_gap_met = ip_gap <= params.ip_gap
            ip_status = master.last_ip_status or "unknown"
            ip_optimal = master.last_ip_optimal
            if ip_optimal is False:
                termination_reason = "time_limit"
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
        for repair_index, graph in enumerate(repair_order):
            if time.monotonic() >= deadline:
                termination_reason = "time_limit"
                search_exhausted_flights.update(
                    pending.request.flight_id for pending in repair_order[repair_index:]
                )
                break
            saturated = frozenset(row for row, load in loads.items() if load >= row_index.cap(row))
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
            loads.update(repaired.claims)
            repair_added += 1

        incumbent = dict(sorted(incumbent.items()))
        _assert_claim_feasible(incumbent, committed_loads, row_index)
        denied = tuple(flight_id for flight_id in flight_ids if flight_id not in incumbent)
        if termination_reason in {"time_limit", "iteration_limit"}:
            search_exhausted_flights.update(denied)
        search_exhausted = tuple(
            flight_id for flight_id in denied if flight_id in search_exhausted_flights
        )
        budget_denied = tuple(
            flight_id for flight_id in denied if flight_id not in search_exhausted_flights
        )
        total_delay_s = math.fsum(column.delay_s for column in incumbent.values())
        master_objective = _selection_objective(incumbent, params.M)
        elapsed_s = time.monotonic() - started

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
                _relative_cost_gap(
                    total_benefit - last_lp_objective,
                    final_cost_lower_bound,
                )
                if math.isfinite(last_lp_objective)
                else math.inf
            ),
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
            # ``objective`` is the user-facing minimization objective.  The
            # maximize-sense master value is retained under an explicit name.
            "objective": total_delay_s,
            "total_delay_s": total_delay_s,
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
            "pricing_flights_completed": pricing_flights_completed,
            "pricing_sweeps_completed": pricing_sweeps_completed,
            "pricing_timeout_flight_id": pricing_timeout_flight_id,
            "n_columns": len(master.columns),
            "n_materialized_rows": len(materialized_rows),
            "lazy_rows_added": lazy_rows_added,
            "lazy_row_rounds": lazy_row_rounds,
            "time_limit_overrun_s": max(0.0, elapsed_s - params.time_limit_s),
            "elapsed_s": elapsed_s,
        }
        return ColGenResult(columns=incumbent, stats=stats)


__all__ = ["ColGenResult", "ColGenSolver"]
