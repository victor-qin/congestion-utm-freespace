"""Column-generation adapter to the shared Zeta route search."""
from __future__ import annotations
from collections.abc import Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass, fields, replace
from functools import partial
import math
import time
from ...types import OperationalIntent
from ..zeta.model import Route, SearchSettings
from ..zeta.search import ZetaGraph as RouteGraph, ZetaOptions as ZetaOptions, search
from .network import RowKey
from .objective import CostModel, cost_model
from .params import ColGenParams
from .translate import Column

Cell = tuple[int, int]
Point = tuple[float, float, float]

@dataclass(frozen=True, slots=True)
class ZetaPricingResult:
    """A feasible route, a reduced-cost upper bound, and a pricing certificate.

    ``complete`` includes ``no_improvement``: no route can improve the master above
    its tolerance, even when the returned route is not the exact pricing optimum.

    Attributes
    ------------
    - equivalent_histories_merged (int): guarded equal-price histories represented by another prefix.
    - merge_error_bound (float): whole-route numerical allowance included in upper_bound.
    - shorter_histories_merged (int): subset of guarded merges comparing both lengths.
    """

    column: Column | None
    reduced_cost: float
    upper_bound: float
    complete: bool
    reason: str
    expanded: int
    generated: int
    wall_s: float
    edge_evaluations: int = 0
    successor_pops: int = 0
    history_pruned: int = 0
    history_expired_rows: int = 0
    history_comparisons: int = 0
    bound_build_s: float = 0.0
    bound_table_states: int = 0
    bound_lookups: int = 0
    bound_pruned: int = 0
    timed_cache_hits: int = 0
    timed_cache_misses: int = 0
    timed_cache_peak_bytes: int = 0
    shared_leg_cache_hits: int = 0
    shared_leg_cache_misses: int = 0
    shared_footprint_cache_hits: int = 0
    shared_footprint_cache_misses: int = 0
    persistent_timing_cache_hits: int = 0
    persistent_timing_cache_misses: int = 0
    strong_history_pruned: int = 0
    history_group_reject_skips: int = 0
    history_group_replace_skips: int = 0
    route_coverage_hits: int = 0
    route_coverage_misses: int = 0
    equivalent_histories_merged: int = 0
    merge_error_bound: float = 0.0
    lattice_bound_calls: int = 0
    shorter_histories_merged: int = 0
    topology_setup_s: float = 0.0
    incumbent_setup_s: float = 0.0
    dp_waypoints: int = 0
    dp_peak_pending: int = 0



def _route(column: Column) -> Route:
    """Convert a master column to a route without trusting its coefficients."""
    return Route(column.flight_id, column.departure_step, column.level, column.origin_lane_idx,
                 column.dest_lane_idx, column.cell_path, column.delay_s, column.claims)

def _column(route: Route | None) -> Column | None:
    """Attach a canonical route to the master's column value type."""
    return None if route is None else Column(
        route.flight_id, route.departure_step, route.level, route.origin_lane_idx,
        route.dest_lane_idx, route.cell_path, route.physical_cost, route.claims)

class ZetaGraph(RouteGraph):
    """CG compatibility facade over the shared graph and its persistent caches.

    Attributes
    ------------
    - options (ZetaOptions): unchanged search domain and optimization switches.
    """

    def intent(self, column: Column | Route) -> OperationalIntent:
        """Keep CG reporting labels while using the shared reservation builder."""
        return replace(super().intent(column), planner="colgen")

    def canonical_column(self, column: Column, model: CostModel) -> Column | None:
        """Validate master coefficients through the shared route authority."""
        return _column(self.canonical_route(_route(column), model))

    def seed_columns(self, params: ColGenParams, deadline: float | None = None) -> tuple[Column, ...]:
        """Adapt shared nominal routes to master columns."""
        return tuple(_column(route) for route in super().seed_routes(cost_model(self.cfg, params), deadline))

def _improves(cost: float, dual_cost: float, *, model: CostModel, params: ColGenParams, pi_f: float) -> bool:
    """Preserve the master's separate floating-point reduced-cost guard."""
    return model.reduced_cost(benefit=params.M, cost=cost, dual_cost=dual_cost, pi_f=pi_f) > params.epsilon

def price_zeta(
    graph: RouteGraph, duals: Mapping[RowKey, float], pi_f: float, params: ColGenParams,
    *, known_columns: Sequence[Column] = (), forbidden_rows: AbstractSet[RowKey] = frozenset(),
    deadline: float | None = None,
) -> ZetaPricingResult:
    """Translate a frozen master objective into shared search and back to reduced costs.

    Parameters
    ------------
    - graph (RouteGraph): reusable route graph and geometry caches.
    - duals (Mapping[RowKey, float]): signed current reservation prices.
    - pi_f (float): finite request dual.
    - params (ColGenParams): objective and master improvement tolerance.
    - known_columns (Sequence[Column]): supplied routes, revalidated by search.
    - forbidden_rows (AbstractSet[RowKey]): saturated reservation rows.
    - deadline (float | None): absolute cutoff covering setup and search.

    Return
    --------
    - output (ZetaPricingResult): master column, reduced-cost bound and complete work evidence.
    """
    started = time.perf_counter()
    if not math.isfinite(pi_f):
        raise ValueError("flight dual must be finite")
    model = cost_model(graph.cfg, params)
    settings = SearchSettings(model, params.epsilon, params.M - pi_f,
                              partial(_improves, model=model, params=params, pi_f=pi_f))
    result = search(graph, duals, settings, known_routes=tuple(_route(c) for c in known_columns),
                      forbidden_rows=forbidden_rows, deadline=deadline)
    margin = params.M - pi_f
    rc = margin - result.priced_cost
    upper = margin - result.raw_lower_bound
    if result.merge_error_bound > 0.0:
        upper = math.nextafter(max(rc, upper) + result.merge_error_bound, math.inf)
    counters = {field.name: getattr(result, field.name) for field in fields(result)
                if field.name not in {"route", "priced_cost", "lower_bound", "raw_lower_bound"}}
    column = _column(result.route)
    counters["wall_s"] = time.perf_counter() - started
    return ZetaPricingResult(column, rc, max(rc, upper), **counters)
