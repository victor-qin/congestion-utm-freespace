"""Cheap column discovery with certificates supplied only by the full graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from copy import copy
from dataclasses import fields, replace
import math
import time

from ..route_domain import RowKey
from .model import SearchSettings
from .model import Route
from .search import ZetaGraph, SearchResult, search


def price_hybrid(
    graph: ZetaGraph, duals: Mapping[RowKey, float], settings: SearchSettings,
    *, known_routes: Sequence[Route] = (), forbidden_rows: AbstractSet[RowKey] = frozenset(),
    deadline: float | None = None,
) -> SearchResult:
    """Discover columns cheaply, then use the remaining budget for full-domain pricing.

    Parameters
    ------------
    - graph (ZetaGraph): full-domain graph, retaining caches across calls.
    - duals (Mapping[RowKey, float]): current signed reservation prices.
    - settings (SearchSettings): physical objective and optional discovery threshold.
    - known_routes (Sequence[Route]): current master columns, excluded from cheap returns.
    - forbidden_rows (AbstractSet[RowKey]): rows unavailable to every stage.
    - deadline (float | None): shared absolute cutoff, including both stages and setup.

    Return
    --------
    - output (SearchResult): aggregated work and a full-domain upper bound; restricted
      completion alone never certifies convergence. Deferred full searches use infinity.
    """
    started = time.perf_counter()
    options = graph.options
    cheap_options = replace(
        options, connection_mode=options.cheap_connection_mode, cheap_connection_mode=None,
        max_expansions=min(options.cheap_max_expansions,
                           options.max_expansions or options.cheap_max_expansions),
        max_labels=min(options.cheap_max_labels, options.max_labels or options.cheap_max_labels),
        early_return=True,
    )
    cheap_graph = getattr(graph, "_cheap_graph", None)
    if cheap_graph is None or cheap_graph.options != cheap_options:
        cheap_graph = ZetaGraph(graph.fg, graph.cfg, graph.row_index, cheap_options,
                                endpoint_validator=graph.endpoint_validator)
        graph._cheap_graph = cheap_graph
    cheap = search(cheap_graph, duals, settings, known_routes=known_routes,
                       forbidden_rows=forbidden_rows, deadline=deadline)
    if deadline is not None and time.perf_counter() >= deadline:
        return replace(cheap, lower_bound=-math.inf, raw_lower_bound=-math.inf, complete=False, reason="deadline",
                       wall_s=time.perf_counter() - started, cheap_expanded=cheap.expanded,
                       cheap_generated=cheap.generated, cheap_wall_s=cheap.wall_s)
    model = settings.model
    candidate = (None if cheap.route is None
                 else graph.canonical_route(cheap.route, model))
    if cheap.route is not None and candidate != cheap.route:
        raise RuntimeError("cheap pricing changed the full-domain reservation or objective")
    known = {graph.canonical_route(column, model) for column in known_routes}
    new_improvement = (settings.target_cost is not None and candidate is not None
                       and candidate not in known
                       and settings.target_cost - cheap.priced_cost > settings.epsilon
                       and settings.accepts(candidate.physical_cost, math.fsum(
                           duals.get(row, 0.0) for row in candidate.claims)))
    remaining_expansions = (None if options.max_expansions is None
                            else options.max_expansions - cheap.expanded)
    remaining_labels = (None if options.max_labels is None
                        else options.max_labels - cheap.generated)
    # Existing caps are checked between expansions, after all children of the last edge.
    exhausted = ((remaining_expansions is not None and remaining_expansions <= 0)
                 or (remaining_labels is not None and remaining_labels <= 0))
    if new_improvement or exhausted:
        # A restricted frontier cannot bound omitted full-graph connections.
        return replace(
            cheap, lower_bound=-math.inf, raw_lower_bound=-math.inf, complete=False,
            reason="improving_column" if new_improvement else "combined_budget",
            wall_s=time.perf_counter() - started,
            cheap_expanded=cheap.expanded, cheap_generated=cheap.generated,
            cheap_wall_s=cheap.wall_s, cheap_columns_returned=int(new_improvement),
        )
    full_graph = getattr(graph, "_full_graph", None)
    if full_graph is None:
        full_graph = copy(graph)
        graph._full_graph = full_graph
    full_graph.options = replace(
        options, cheap_connection_mode=None,
        max_expansions=remaining_expansions, max_labels=remaining_labels,
    )
    incumbents = known_routes if candidate is None else (*known_routes, candidate)
    full = search(full_graph, duals, settings, known_routes=incumbents,
                      forbidden_rows=forbidden_rows, deadline=deadline)
    # Keep the view: native pricing also assigns new geometry and coverage caches to it.
    counters = {
        field.name: getattr(cheap, field.name) + getattr(full, field.name)
        for field in fields(full)
        if field.name not in {
            "route", "priced_cost", "lower_bound", "raw_lower_bound", "complete", "reason", "wall_s",
            "timed_cache_peak_bytes", "cheap_expanded", "cheap_generated", "cheap_wall_s",
            "full_wall_s", "full_calls", "cheap_columns_returned",
            "merge_error_bound",
        }
    }
    if candidate is not None and cheap.priced_cost < full.priced_cost:
        full = replace(full, route=candidate, priced_cost=cheap.priced_cost,
                       lower_bound=min(full.lower_bound, cheap.priced_cost),
                       raw_lower_bound=min(full.raw_lower_bound, cheap.priced_cost))
    return replace(
        full, **counters, wall_s=time.perf_counter() - started,
        timed_cache_peak_bytes=max(cheap.timed_cache_peak_bytes, full.timed_cache_peak_bytes),
        cheap_expanded=cheap.expanded, cheap_generated=cheap.generated,
        cheap_wall_s=cheap.wall_s, full_wall_s=full.wall_s, full_calls=1,
    )
