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
from .master import RestrictedMaster
from .network import FlightGraph, RowIndex, RowKey, build_flight_graph, column_claims
from .params import ColGenParams
from .pricing import price_flight, seed_column
from .translate import Column

_FEASIBILITY_TOL = 1e-7
_REDUCED_COST_TOL = 1e-9
_OBJECTIVE_TOL = 1e-8


@dataclass(frozen=True, slots=True)
class ColGenResult:
    """Selected columns and diagnostics for one whole-schedule solve.

    A flight absent from :attr:`columns` is an optimizer-verdict denial.  Phase
    3 translates that absence to ``BUDGET_EXCEEDED``; Phase 2 intentionally
    does not manufacture an ``OperationalIntent`` denial here.
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
                    "n_columns": 0,
                    "n_materialized_rows": 0,
                    "lazy_rows_added": 0,
                    "lazy_row_rounds": 0,
                    "elapsed_s": time.monotonic() - started,
                },
            )

        # ``static_terms`` is commonly a generator owned by the simulation.  Every
        # graph must see the identical wall catalogue, so snapshot it once.
        static_term_snapshot = tuple(static_terms)
        graphs = {
            request.flight_id: build_flight_graph(request, cfg, static_term_snapshot, params)
            for request in ordered_requests
        }

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
        for flight_id in flight_ids:
            try:
                seed = seed_column(graphs[flight_id], cfg)
            except ValueError:
                # A disconnected/static-blocked graph is a legitimate
                # optimizer-verdict denial, not a batch-wide solver failure.
                # Its <=1 flight row can remain empty while other flights run.
                seedless_flights.add(flight_id)
                continue
            master.add_column(_canonical_column(seed, graphs[flight_id], cfg))

        rng = np.random.default_rng(cfg.seed)
        total_benefit = len(flight_ids) * params.M
        best_heuristic: dict[int, Column] = {}
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

        for iteration in range(params.max_iterations):
            if iteration > 0 and time.monotonic() - started >= params.time_limit_s:
                termination_reason = "time_limit"
                break

            # A bound is meaningful only for the full claim-feasible relaxation.
            # Materialize violated rows and re-solve until the current LP is clean.
            while True:
                lp_objective, capacity_duals, x = master.solve_lp()
                added_rows = master.add_violated_rows(x, _FEASIBILITY_TOL)
                if not added_rows:
                    break
                lazy_rows_added += added_rows
                lazy_row_rounds += 1

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
            for flight_id in flight_ids:
                reduced_cost, column = price_flight(
                    graphs[flight_id],
                    capacity_duals,
                    master.flight_duals[flight_id],
                    cfg,
                    params,
                )
                best_reduced_costs.append(float(reduced_cost))
                if column is not None and reduced_cost > _REDUCED_COST_TOL:
                    priced_columns.append(_canonical_column(column, graphs[flight_id], cfg))

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

            before = len(master.columns)
            for column in sorted(
                priced_columns, key=lambda item: (item.flight_id, _column_key(item))
            ):
                master.add_column(column)
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
        if not ip_skipped:
            master.set_heuristic(incumbent)
            ip_selection = master.solve_ip()
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
        repair_order = sorted(
            (graphs[flight_id] for flight_id in flight_ids if flight_id not in incumbent),
            key=lambda graph: (graph.request.t_departure, graph.request.flight_id),
        )
        for graph in repair_order:
            saturated = frozenset(row for row, load in loads.items() if load >= row_index.cap(row))
            _reduced_cost, repaired = price_flight(
                graph,
                {},
                0.0,
                cfg,
                params,
                forbidden_rows=saturated,
                require_improving=False,
            )
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
        total_delay_s = math.fsum(column.delay_s for column in incumbent.values())
        master_objective = _selection_objective(incumbent, params.M)

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
            "repair_added": repair_added,
            "seedless_flight_ids": tuple(sorted(seedless_flights)),
            "n_columns": len(master.columns),
            "n_materialized_rows": len(materialized_rows),
            "lazy_rows_added": lazy_rows_added,
            "lazy_row_rounds": lazy_row_rounds,
            "elapsed_s": time.monotonic() - started,
        }
        return ColGenResult(columns=incumbent, stats=stats)


__all__ = ["ColGenResult", "ColGenSolver"]
