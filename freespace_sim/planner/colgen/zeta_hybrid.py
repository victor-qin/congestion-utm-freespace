"""Cheap column discovery with certificates supplied only by the full graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
from copy import copy
from dataclasses import fields, replace
import math
import time

from .network import RowKey
from .objective import cost_model
from .params import ColGenParams
from .translate import Column
from .zeta import ZetaGraph, ZetaPricingResult, price_zeta


def price_hybrid(
    graph: ZetaGraph, duals: Mapping[RowKey, float], pi_f: float, params: ColGenParams,
    *, known_columns: Sequence[Column] = (), forbidden_rows: AbstractSet[RowKey] = frozenset(),
    deadline: float | None = None,
) -> ZetaPricingResult:
    """Discover columns cheaply, then use the remaining budget for full-domain pricing.

    Parameters
    ------------
    - graph (ZetaGraph): full-domain graph, retaining caches across calls.
    - duals (Mapping[RowKey, float]): current signed reservation prices.
    - pi_f (float): current flight dual.
    - params (ColGenParams): unchanged objective and improvement tolerance.
    - known_columns (Sequence[Column]): current master columns, excluded from cheap returns.
    - forbidden_rows (AbstractSet[RowKey]): rows unavailable to every stage.
    - deadline (float | None): shared absolute cutoff, including both stages and setup.

    Return
    --------
    - output (ZetaPricingResult): aggregated work and a full-domain upper bound; restricted
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
        cheap_graph = ZetaGraph(graph.fg, graph.cfg, graph.row_index, cheap_options)
        graph._cheap_graph = cheap_graph
    cheap = price_zeta(cheap_graph, duals, pi_f, params, known_columns=known_columns,
                       forbidden_rows=forbidden_rows, deadline=deadline)
    if deadline is not None and time.perf_counter() >= deadline:
        return replace(cheap, upper_bound=math.inf, complete=False, reason="deadline",
                       wall_s=time.perf_counter() - started, cheap_expanded=cheap.expanded,
                       cheap_generated=cheap.generated, cheap_wall_s=cheap.wall_s)
    model = cost_model(graph.cfg, params)
    candidate = (None if cheap.column is None
                 else graph.canonical_column(cheap.column, model))
    if cheap.column is not None and candidate != cheap.column:
        raise RuntimeError("cheap pricing changed the full-domain reservation or objective")
    known = {graph.canonical_column(column, model) for column in known_columns}
    new_improvement = (candidate is not None and candidate not in known
                       and cheap.reduced_cost > params.epsilon
                       and model.reduced_cost(
                           benefit=params.M, cost=candidate.delay_s,
                           dual_cost=math.fsum(duals.get(row, 0.0) for row in candidate.claims),
                           pi_f=pi_f,
                       ) > params.epsilon)
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
            cheap, upper_bound=math.inf, complete=False,
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
    incumbents = known_columns if candidate is None else (*known_columns, candidate)
    full = price_zeta(full_graph, duals, pi_f, params, known_columns=incumbents,
                      forbidden_rows=forbidden_rows, deadline=deadline)
    # Keep the view: native pricing also assigns new geometry and coverage caches to it.
    counters = {
        field.name: getattr(cheap, field.name) + getattr(full, field.name)
        for field in fields(full)
        if field.name not in {
            "column", "reduced_cost", "upper_bound", "complete", "reason", "wall_s",
            "timed_cache_peak_bytes", "cheap_expanded", "cheap_generated", "cheap_wall_s",
            "full_wall_s", "full_calls", "cheap_columns_returned",
            "merge_error_bound",
        }
    }
    if candidate is not None and cheap.reduced_cost > full.reduced_cost:
        full = replace(full, column=candidate, reduced_cost=cheap.reduced_cost,
                       upper_bound=max(full.upper_bound, cheap.reduced_cost))
    return replace(
        full, **counters, wall_s=time.perf_counter() - started,
        timed_cache_peak_bytes=max(cheap.timed_cache_peak_bytes, full.timed_cache_peak_bytes),
        cheap_expanded=cheap.expanded, cheap_generated=cheap.generated,
        cheap_wall_s=cheap.wall_s, full_wall_s=full.wall_s, full_calls=1,
    )
