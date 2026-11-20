"""Compact parent-pointer search with compiled sparse reservation-price expansion.

Geometry and final columns remain owned by the simulator. Numeric successor scans,
label writes, expiry and reservation unions are compiled. Best-first uses a Python
priority queue; forward DP uses waypoint buckets. Timed footprints use exact clocks
and a fresh row table each pricing call.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
import heapq
import itertools
import math
import time
from typing import AbstractSet, TYPE_CHECKING

if TYPE_CHECKING:
    import cython

import numpy as np

from ..route_domain import RowKey
from .model import SearchSettings
from .model import Route
from .search import Cell, ZetaGraph, SearchResult
from .geometry import volume_spatial_cells
from .history import PriceHistory as PythonPriceHistory
from .history_guard import HistoryGuard
from .price_floor import (
    prepare_price_floor, price_floor, snap_certificate, snap_prices, snapped_lower_bound,
)
from .kernel import (
    apply_timed_edge, build_successor_scan, evaluate_edge, partition_expired,
    prepare_timed_edge, prepare_windowed_edge, write_label,
)
from .timing import edge_end_times
from .forward import ForwardFrontier, ForwardLimit, ForwardTopology


class _Labels:
    """Growing numeric prefix storage; paths and reservation unions are never copied here."""

    def __init__(self, compiled: bool = False, native: bool = False) -> None:
        """Allocate a small initial arena that grows only when surviving labels need it."""
        self.ints = np.empty((7, 1024), dtype=np.int64)
        self.floats = np.empty((4, 1024), dtype=np.float64)
        self.int_view = memoryview(self.ints)
        self.float_view = memoryview(self.floats)
        self.retired: list[int] = []
        self.floors: list[float] = []
        self.size = 0
        self.compiled = compiled
        self.native = native

    def add(
        self, node: cython.Py_ssize_t, parent: cython.Py_ssize_t,
        context: cython.Py_ssize_t, paid: cython.Py_ssize_t, sequence: cython.Py_ssize_t,
        terminal: bool, first: bool, stamp: float, length: float, positive: float,
        canonical_length: float, retired: object, floor: float = -math.inf,
    ) -> cython.Py_ssize_t:
        """Store one surviving prefix while retaining its arbitrary-size signed charge.

        Parameters
        ------------
        - node (Py_ssize_t): current waypoint index.
        - parent (Py_ssize_t): previous label index, or -1 at a root.
        - context (Py_ssize_t): departure and endpoint-choice context index.
        - paid (Py_ssize_t): interned live reservation-set index.
        - sequence (Py_ssize_t): exact segment-length sequence index.
        - terminal (bool): whether the destination endpoint role was used.
        - first (bool): whether the first horizontal segment is still pending.
        - stamp (float): exact prefix arrival timestamp.
        - length (float): sequentially accumulated physical length.
        - positive (float): accumulated nonnegative reservation charge.
        - canonical_length (float): separately accumulated builder length.
        - retired (object): Python integer charge; must never be narrowed to a C integer.
        - floor (float): inherited completion-cost lower bound.

        Return
        --------
        - output (Py_ssize_t): stable index of the stored prefix.
        """
        index: cython.Py_ssize_t = self.size
        if index == self.ints.shape[1]:
            ints = np.empty((7, index * 2), dtype=np.int64)
            floats = np.empty((4, index * 2), dtype=np.float64)
            ints[:, :index], floats[:, :index] = self.ints, self.floats
            self.ints, self.floats = ints, floats
            self.int_view = memoryview(self.ints)
            self.float_view = memoryview(self.floats)
        if self.native:
            integer_view: cython.longlong[:, ::1] = self.int_view
            float_view: cython.double[:, ::1] = self.float_view
            integer_view[0, index] = node
            integer_view[1, index] = parent
            integer_view[2, index] = context
            integer_view[3, index] = paid
            integer_view[4, index] = sequence
            integer_view[5, index] = terminal
            integer_view[6, index] = first
            float_view[0, index] = stamp
            float_view[1, index] = length
            float_view[2, index] = positive
            float_view[3, index] = canonical_length
        elif self.compiled:
            write_label(self.ints, self.floats, index, node, parent, context, paid,
                        sequence, terminal, first, stamp, length, positive, canonical_length)
        else:
            self.ints[:, index] = node, parent, context, paid, sequence, terminal, first
            self.floats[:, index] = stamp, length, positive, canonical_length
        self.retired.append(retired)
        self.floors.append(floor)
        self.size += 1
        return index

    def path(self, label: cython.Py_ssize_t, cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
        """Reconstruct a physical path only when canonical sink validation requires it."""
        path = []
        integer_view: cython.longlong[:, ::1] = self.int_view
        index: cython.Py_ssize_t = label
        while index >= 0:
            path.append(cells[integer_view[0, index]])
            index = integer_view[1, index]
        return tuple(reversed(path))


class _RowTable:
    """Sparse exact cell/time prices and forbidden rows, grouped by spatial cell."""

    def __init__(
        self, prices: Mapping[RowKey, float], forbidden: AbstractSet[RowKey],
    ) -> None:
        """Pack nonzero prices and all hard exclusions into stable integer row ids."""
        self.rows = tuple(prices) + tuple(row for row in forbidden if row not in prices)
        self.ids = {row: index for index, row in enumerate(self.rows)}
        self.blocked = np.array([row in forbidden for row in self.rows], dtype=np.bool_)
        self.positive = np.array([max(0.0, prices.get(row, 0.0)) for row in self.rows])
        groups: defaultdict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
        for index, row in enumerate(self.rows):
            if row.kind == "cell":
                groups[row[1:4]].append((row.step, index))
        self.spaces = {space: index for index, space in enumerate(groups)}
        offsets, steps, ids = [0], [], []
        for values in groups.values():
            for step, row in sorted(values):
                steps.append(step)
                ids.append(row)
            offsets.append(len(steps))
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.steps = np.asarray(steps, dtype=np.int64)
        self.row_ids = np.asarray(ids, dtype=np.int64)
        self.marks = np.zeros(len(self.rows), dtype=np.int64)
        self.generation = 0


def search_compact(
    graph: ZetaGraph, duals: Mapping[RowKey, float], settings: SearchSettings,
    *, known_routes: Sequence[Route] = (), forbidden_rows: AbstractSet[RowKey] = frozenset(),
    deadline: float | None = None, _native_loop: bool = False,
    _compiled_history_type: type | None = None,
    _compiled_coverage_type: type | None = None,
    _compiled_state_type: type | None = None,
) -> SearchResult:
    """Price the declared universe with compact prefixes and compiled exact row unions.

    Parameters
    ------------
    - graph (ZetaGraph): simulator-owned finite route universe and geometry authority.
    - duals (Mapping[RowKey, float]): frozen finite prices; signed values remain unchanged.
    - settings (SearchSettings): objective, tolerance and optional target cost.
    - known_routes (Sequence[Route]): candidate incumbents, independently recanonicalized.
    - forbidden_rows (AbstractSet[RowKey]): unavailable capacity rows, including zero-price rows.
    - deadline (float | None): monotonic cutoff; incomplete work retains a valid frontier bound.
    - _native_loop (bool): use direct typed-buffer label writes in the compiled extension.
    - _compiled_history_type (type | None): exact history helper built into the same native
      binary; selected only by the explicit compiled_history option.
    - _compiled_coverage_type (type | None): persistent coverage helper built into the same
      binary; selected only by the explicit compiled_route_coverage option.
    - _compiled_state_type (type | None): native exact state-key index, independent of
      traversal or connection geometry; selected by native_state_index.

    Return
    --------
    - output (SearchResult): feasible best route, safe reduced-cost upper bound and status.
      Geometry preparation and first-use compilation are included in its wall time.
    """
    started = time.perf_counter()
    if (deadline is not None and not math.isfinite(deadline)):
        raise ValueError("flight dual and deadline must be finite")
    prices = {RowKey(row): float(value) for row, value in duals.items() if value != 0.0}
    if any(not math.isfinite(value) for value in prices.values()):
        raise ValueError("row prices must be finite")
    forbidden_rows = frozenset(RowKey(row) for row in forbidden_rows)
    # Labels, floors and histories see only the search vector; incumbents use both.
    exact_prices, snap = prices, None
    if graph.options.price_snap:
        prices, snap = snap_prices(exact_prices, settings.epsilon)
        if not snap.rows:
            prices, snap = exact_prices, None
    negative, price_error = prepare_price_floor(prices.values())
    cfg, fg = graph.cfg, graph.fg
    model = settings.model
    best: float = math.inf
    reported, reported_cost, reported_improves = None, math.inf, False
    exact_claim_prices: dict[frozenset[RowKey], float] = {}
    pricing_margin: float = (settings.target_cost or 0.0)
    improvement_tolerance: float = settings.epsilon
    early_return: bool = graph.options.early_return and settings.target_cost is not None
    # Prefixes at or above ``limit`` are discarded: the incumbent, or also the pricing target
    # under target_cutoff. Withheld negative prices can lower a route by at most their total,
    # so the snapped-space target is raised by it; finish() re-derives the full-price bound.
    cut: float = (pricing_margin - (0.0 if snap is None else snap.negative_mass)
                  if early_return and graph.options.target_cutoff else math.inf)
    limit: float = cut
    incumbent = None
    known_canonical: set[Route] = set()
    claim_prices: dict[frozenset[RowKey], float] = {}
    if graph.options.compiled_route_coverage:
        if _native_loop and _compiled_coverage_type is not None:
            if not isinstance(graph.route_coverage, _compiled_coverage_type):
                graph.route_coverage = _compiled_coverage_type(graph.route_coverage.max_routes)
        elif deadline is None or time.perf_counter() < deadline:
            raise RuntimeError("compiled_route_coverage requires the prepared native coverage helper")
        # Expired native loading returns the ordinary interrupted-setup bound below;
        # the existing deadline checks prohibit a search with an uncompiled helper.
    route_prices = graph.route_coverage.bind(prices) if graph.options.route_coverage_cache else None
    early_candidate_ready = False
    expanded = generated = edge_evaluations = successor_pops = 0
    tol: float = min(1e-7, settings.epsilon * 0.1)
    global_floor = price_floor(0.0, 0.0, negative, price_error, tol)
    forward_dp = graph.options.traversal == "forward_dp"
    lazy = graph.options.lazy_successors and not forward_dp
    frontier = topology = None
    active_floor = global_floor
    topology_setup_s = incumbent_setup_s = 0.0
    heap: list[tuple[float, int, int, int, int]] = []
    serial = itertools.count()
    compiled_search = graph.options.compiled_search
    labels: _Labels = _Labels(compiled_search, _native_loop)
    # The native loader appends this class to the same module as the search loop.
    state_index: _StateIndex = None  # noqa: F821
    if graph.options.native_state_index:
        if _native_loop and _compiled_state_type is not None:
            state_index = _compiled_state_type()
        elif deadline is None or time.perf_counter() < deadline:
            raise RuntimeError("native_state_index requires the prepared native state helper")
    cells = tuple(sorted(graph.cells | {cell for _index, cell, _steps in graph.origins}
                         | {cell for _index, cell in graph.destinations}))
    cell_ids = {cell: index for index, cell in enumerate(cells)}
    turns = graph.turns
    # Heading classes widen the state key as node * count + class; one class leaves it unchanged.
    heading_count: cython.Py_ssize_t = 1 if turns is None else turns.count
    turn_rejected = 0
    node_marks = bytearray(len(cells))
    state_marks = bytearray(len(cells) * heading_count)
    table: _RowTable = _RowTable(prices, forbidden_rows)
    history_type = PythonPriceHistory
    if graph.options.compiled_history:
        if _native_loop and _compiled_history_type is not None:
            history_type = _compiled_history_type
        elif deadline is None or time.perf_counter() < deadline:
            raise RuntimeError("compiled_history requires the prepared native history helper")
        # An expired build deadline uses the existing interrupted-setup bound; the
        # deadline checks below prevent any search with this uncompiled helper.
    history = (
        history_type({table.ids[row]: price for row, price in prices.items()},
                     group_skips=graph.options.history_group_skips,
                     merge_equivalent=graph.options.merge_equivalent_histories,
                     merge_shorter=graph.options.merge_shorter_histories)
        if graph.options.expire_history or graph.options.history_dominance else None
    )
    history_guard = HistoryGuard(graph, model, prices) if graph.options.strong_history_dominance else None
    # A guarded merge leaves at most its endpoint margin unproven. Keeping that band above
    # the cut lets exhausting below it certify whenever the uncut search would.
    if math.isfinite(cut) and history_guard is not None and graph.options.history_dominance:
        cut += max((m for oi, *_ in graph.origins for di, _ in graph.destinations
                    if (m := history_guard.margin(oi, di)) is not None), default=0.0)
        limit = cut
    merge_shorter = (graph.options.merge_shorter_histories
                     and math.isfinite(model.air_weight) and model.air_weight >= 0.0)
    row_ends = np.asarray([(row.step + 1) * cfg.dt_s for row in table.rows])
    expired_rows = 0
    completion_bounds = None
    lattice_bound = None
    bound_build_s = 0.0
    bound_lookups = bound_pruned = 0
    contexts: list[tuple[int, int | None, int | None, Cell, float, float | None]] = []
    paid_arrays: list[np.ndarray] = []
    paid_ids: dict[bytes, int] = {}
    sequence_ids: dict[tuple[int, float], int] = {}
    seen: set[tuple[object, ...]] = set()
    target_cache: dict[tuple[int, int, int], tuple[tuple[float, int, float, float], ...]] = {}
    target_rejections: dict[tuple[int, int, int], int] = {}
    target_arrays: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    scans: dict[int, tuple] = {}
    edge_cache: dict[tuple[int, int, bool, bool], tuple | None] = {}
    timed_cache: OrderedDict[tuple[int, int, bool, bool, float], tuple] = OrderedDict()
    cache_budget = graph.options.timed_cache_bytes
    persistent_timing = graph.options.persistent_timing_cache
    if graph.options.leg_cache or persistent_timing:
        from .leg_cache import get_footprint, get_timing
    cache_size = cache_peak = cache_hits = cache_misses = 0
    sequence_cache: dict[tuple[int, int, int, bool, bool], int] = {}
    # Spatial geometry is independent of the current dual vector and survives LP rounds.
    if not hasattr(graph, "_compact_geometry"):
        graph._compact_geometry = {}
    geometry = graph._compact_geometry

    def finish(complete: bool, reason: str, floor: float) -> SearchResult:
        """Keep every unexplored scan represented in a minimization frontier bound."""
        route, cost = (incumbent, best) if snap is None else (reported, reported_cost)
        raw = snapped_lower_bound(min(best, floor), snap)
        rc = pricing_margin - cost
        upper = pricing_margin - raw
        uncertainty = 0.0 if history is None else history.merge_error_bound
        if uncertainty > 0.0:
            upper = math.nextafter(max(rc, upper) + uncertainty, math.inf)
            # A bound, not an incumbent: target_cutoff can finish with no route found.
            if complete and math.isfinite(raw):
                complete = early_return and upper <= improvement_tolerance
                reason = "no_improvement" if complete else "merge_uncertainty"
        if snap is not None:
            complete, reason = snap_certificate(
                complete, reason, raw, cost, upper, settings.target_cost,
                improvement_tolerance, tol, reported_improves)
        return SearchResult(route, cost,
                                 (math.nextafter(raw - uncertainty, -math.inf)
                                  if uncertainty > 0 else raw), complete, reason,
                                 expanded, generated, time.perf_counter() - started,
                                 edge_evaluations, successor_pops,
                                 0 if history is None else history.pruned, expired_rows,
                                 0 if history is None else history.comparisons,
                                 bound_build_s,
                                 0 if completion_bounds is None else completion_bounds.table_states,
                                 bound_lookups, bound_pruned,
                                 cache_hits, cache_misses, cache_peak,
                                 strong_history_pruned=0 if history is None else history.strong_pruned,
                                 history_group_reject_skips=0 if history is None else history.group_reject_skips,
                                 history_group_replace_skips=0 if history is None else history.group_replace_skips,
                                 route_coverage_hits=0 if route_prices is None else route_prices.hits,
                                 route_coverage_misses=0 if route_prices is None else route_prices.misses,
                                 equivalent_histories_merged=0 if history is None else history.merged,
                                 merge_error_bound=uncertainty,
                                 lattice_bound_calls=int(lattice_bound is not None),
                                 shorter_histories_merged=0 if history is None else history.shorter_merged,
                                 topology_setup_s=topology_setup_s,
                                 incumbent_setup_s=incumbent_setup_s,
                                 dp_waypoints=0 if frontier is None else frontier.visited,
                                 dp_peak_pending=0 if frontier is None else frontier.peak_pending, raw_lower_bound=raw,
                                 turn_rejected=turn_rejected, waypoint_states=node_marks.count(1),
                                 heading_states=state_marks.count(1),
                                 snapped_rows=0 if snap is None else snap.rows,
                                 snapped_positive_mass=0.0 if snap is None else snap.positive_mass,
                                 snapped_negative_mass=0.0 if snap is None else snap.negative_mass)

    def accept(column: Route, *, known: bool = False) -> None:
        """Use only canonical full-volume coefficients to update the feasible cutoff."""
        nonlocal best, incumbent, early_candidate_ready, limit
        nonlocal reported, reported_cost, reported_improves
        canonical = graph.canonical_route(column, model)
        if canonical is None or not canonical.claims.isdisjoint(forbidden_rows):
            return
        if early_return and known:
            known_canonical.add(canonical)
        # Equal reservation unions have the same price only within this frozen-dual call.
        dual_cost = claim_prices.get(canonical.claims)
        if dual_cost is None:
            dual_cost = (route_prices.price(canonical.claims) if route_prices is not None
                         else math.fsum(prices.get(row, 0.0) for row in canonical.claims))
            claim_prices[canonical.claims] = dual_cost
        value = canonical.physical_cost + dual_cost
        if snap is not None:
            # The search cutoff stays snapped; reporting and early exits use full prices.
            exact_cost = exact_claim_prices.get(canonical.claims)
            if exact_cost is None:
                exact_cost = math.fsum(exact_prices.get(row, 0.0) for row in canonical.claims)
                exact_claim_prices[canonical.claims] = exact_cost
            exact_value = canonical.physical_cost + exact_cost
            if exact_value < reported_cost:
                reported, reported_cost = canonical, exact_value
                reported_improves = (pricing_margin - exact_value > settings.epsilon
                                     and settings.accepts(canonical.physical_cost, exact_cost))
                early_candidate_ready = (early_return and reported_improves
                                         and canonical not in known_canonical)
            if value < best:
                best, incumbent = value, canonical
                limit = min(best, cut)
            return
        if value < best:
            best, incumbent = value, canonical
            limit = min(best, cut)
            # Both associations must pass epsilon, or the driver could reject an early exit.
            early_candidate_ready = (early_return
                                     and canonical not in known_canonical
                                     and (settings.target_cost or 0.0) - best > settings.epsilon
                                     and settings.accepts(canonical.physical_cost, dual_cost))

    def bound(context: int, length: float, remaining: float, positive: float) -> float:
        """Use the reference's admissible geometric and all-negative-credit relaxation."""
        _dep, _oi, _di, _goal, ground, fold = contexts[context]
        cost: float = ground
        if fold is not None and graph.reference > 1e-9:
            detour: float = max(0.0, length + remaining + fold - graph.reference)
            cost += model.air_weight * detour / cfg.nominal_speed_mps
        return price_floor(cost, positive, negative, price_error, tol)

    def intern_paid(paid: np.ndarray) -> int:
        """Intern immutable sorted row-id sets, with byte equality rather than hash-only equality."""
        key = paid.tobytes()
        index = paid_ids.get(key)
        if index is None:
            index = len(paid_arrays)
            paid_ids[key] = index
            paid_arrays.append(paid)
        return index

    def enqueue(
        node: int, parent: int, context: int, paid: np.ndarray, sequence: int,
        terminal: bool, first: bool, stamp: float, length: float, positive: float, floor: float,
        canonical_length: float = 0.0, retired: int = 0, heading: int = 0,
    ) -> None:
        """Expire paid rows and apply the reference's exact future-state comparisons."""
        nonlocal generated, expired_rows, bound_lookups, bound_pruned
        if graph.options.expire_history:
            cutoff = stamp - max(0.0, cfg.time_buffer_s)
            if compiled_search:
                paid, expired = partition_expired(paid, row_ends, cutoff)
                expired_rows += len(expired)
                retired += history.charge(expired.tolist())
            else:
                expired = row_ends[paid] <= cutoff
                expired_rows += int(np.count_nonzero(expired))
                retired += history.charge(paid[expired].tolist())
                paid = paid[~expired]
        if completion_bounds is not None:
            bound_lookups += 1
            floor = max(floor, completion_bounds.floor(
                cells[node], contexts[context][3], stamp, length, positive,
                contexts[context][4], negative, contexts[context][5],
                (table.rows[int(row)] for row in paid), terminal,
            ))
            if floor >= limit:
                bound_pruned += 1
                return
        key = None
        paid_id = intern_paid(paid) if state_index is not None else None
        state_node = node * heading_count + heading
        if contexts[context][5] is not None or graph.reference <= 1e-9:
            margin = None if history_guard is None else history_guard.margin(
                contexts[context][1], contexts[context][2],
            )
            if state_index is not None:
                key, broad_key, length_key = state_index.keys(
                    state_node, context, stamp, length, canonical_length, terminal, first,
                    sequence, margin is not None, merge_shorter,
                )
            else:
                key = state_node, context, stamp, length, canonical_length, terminal, first, sequence
                broad_key = key[:-1] if margin is not None else None
                length_key = key[:3] + key[5:7] if merge_shorter else None
            if graph.options.history_dominance:
                history_paid = paid.tolist() if paid_id is None else paid
                if margin is None:
                    accepted = history.accept(key, history_paid, retired, labels.size,
                                              paid_id=paid_id)
                else:
                    accepted = history.accept(key, history_paid, retired, labels.size,
                                              broad_key=broad_key, margin=margin,
                                              length_key=length_key,
                                              lengths=(length, canonical_length) if merge_shorter else None,
                                              paid_id=paid_id)
                if not accepted:
                    return
        if paid_id is None:
            paid_id = intern_paid(paid)
        if key is not None and not graph.options.history_dominance and graph.options.merge_identical_states:
            exact = ((key, paid_id, retired) if state_index is not None
                     else key + (paid_id, retired))
            if exact in seen:
                return
            seen.add(exact)
        label = labels.add(node, parent, context, paid_id, sequence, terminal, first,
                           stamp, length, positive, canonical_length, retired,
                           floor if completion_bounds is not None else -math.inf)
        if forward_dp:
            frontier.add(node, label, floor)
        else:
            heapq.heappush(heap, (floor, next(serial), 0, label, 0))
        generated += 1
        node_marks[node] = 1
        state_marks[state_node] = 1
        if forward_dp and graph.options.max_labels is not None and generated >= graph.options.max_labels:
            raise ForwardLimit("search_cap")

    def heading_of(label: int) -> int:
        """Recover a stored prefix's incoming heading class from its parent waypoint."""
        if turns is None:
            return 0
        integer_view: cython.longlong[:, ::1] = labels.int_view
        index: cython.Py_ssize_t = label
        parent: cython.Py_ssize_t = integer_view[1, index]
        if parent < 0:
            return 0
        return turns.heading(cells[integer_view[0, parent]], cells[integer_view[0, index]])

    def targets(label: int) -> tuple[tuple[float, int, float, float], ...]:
        """Cache geometry-only successor ordering per node, destination and heading class."""
        nonlocal turn_rejected
        integer_view: cython.longlong[:, ::1] = labels.int_view
        index: cython.Py_ssize_t = label
        node, context = integer_view[0, index], integer_view[2, index]
        goal = contexts[context][3]
        heading = heading_of(label)
        key = node, cell_ids[goal], heading
        if key not in target_cache:
            entries = []
            reachable = None if topology is None else topology.reaches(cell_ids[goal])
            neighbors = tuple(graph.neighbors(cells[node]) if topology is None else
                              (cells[int(i)] for i in topology.targets[
                                  topology.offsets[node]:topology.offsets[node + 1]]
                               if reachable[i]))
            if heading:
                # Legality depends on the incoming link; geometry below stays heading-free.
                allowed = turns.allowed[heading]
                origin = cells[node]
                legal = tuple(target for target in neighbors
                              if (target[0] - origin[0], target[1] - origin[1]) in allowed)
                target_rejections[key] = len(neighbors) - len(legal)
                neighbors = legal
            for target in neighbors:
                distance = math.dist(graph.point(cells[node]), graph.point(target))
                remaining = (math.dist(graph.point(target), graph.point(goal))
                             if lattice_bound is None else lattice_bound.remaining(target, goal))
                entries.append((distance + remaining, cell_ids[target], distance, remaining))
            target_cache[key] = tuple(entries)
        if heading:
            turn_rejected += target_rejections[key]
        return target_cache[key]

    def schedule(label: int, start: int) -> None:
        """Queue the first surviving deferred successor; every later one has no lower bound."""
        integer_view: cython.longlong[:, ::1] = labels.int_view
        float_view: cython.double[:, ::1] = labels.float_view
        index: cython.Py_ssize_t = label
        context = integer_view[2, index]
        length: float = float_view[1, index]
        positive: float = float_view[2, index]
        if compiled_search:
            if start == 0:
                entries = targets(label)
                key = integer_view[0, index], cell_ids[contexts[context][3]], heading_of(label)
                if key not in target_arrays:
                    target_arrays[key] = (np.array([entry[2] for entry in entries]),
                                          np.array([entry[3] for entry in entries]))
                ground, fold = contexts[context][4:6]
                indices, floors = build_successor_scan(
                    *target_arrays[key], length, positive, labels.floors[label], ground,
                    math.nan if fold is None else fold, graph.reference, model.air_weight,
                    cfg.nominal_speed_mps, negative, tol, graph.max_length, limit, price_error,
                )
                scans[label] = entries, indices, floors
            _entries, _indices, floors = scans[label]
            if start < len(floors) and floors[start] < limit:
                heapq.heappush(heap, (float(floors[start]), next(serial), 1, label, start))
            else:
                del scans[label]
            return
        if start == 0:
            candidates = []
            for entry in targets(label):
                _order, _target, distance, remaining = entry
                lower = length + distance
                if lower + remaining <= graph.max_length + 1e-7:
                    floor = max(labels.floors[label], bound(context, lower, remaining, positive))
                    if floor < limit:
                        candidates.append((floor, entry))
            # Sort actual floating bounds rather than algebraically equivalent geometric sums.
            scans[label] = tuple(sorted(candidates, key=lambda candidate: candidate[0]))
        entries = scans[label]
        for index in range(start, len(entries)):
            floor, _entry = entries[index]
            if floor < limit:
                heapq.heappush(heap, (floor, next(serial), 1, label, index))
                return
        del scans[label]

    def prepared_edge(node: int, target: int, first: bool, terminal: bool) -> tuple | None:
        """Prepare only requested static edges and only spatial groups relevant to this call."""
        nonlocal edge_evaluations
        edge_evaluations += 1
        key = node, target, first, terminal
        if key in edge_cache:
            return edge_cache[key]
        if key not in geometry:
            parts = graph.parts(cells[node], cells[target], first, terminal)
            geometry[key] = None if parts is None else (
                np.array([part.duration for part in parts]),
                np.array([part.length for part in parts]),
                tuple((get_footprint(part.shape, cfg) if graph.options.leg_cache
                       else volume_spatial_cells(part.shape, cfg)) for part in parts),
            )
        static = geometry[key]
        if static is None:
            edge_cache[key] = None
            return None
        durations, lengths, covered = static
        offsets, spaces = [0], []
        for part_cells in covered:
            spaces.extend(table.spaces[cell] for cell in part_cells if cell in table.spaces)
            offsets.append(len(spaces))
        # Preserve Python's sum semantics for the canonical length key, once per edge.
        result = (durations, lengths, np.asarray(offsets, dtype=np.int64),
                  np.asarray(spaces, dtype=np.int64), sum(float(value) for value in lengths),
                  tuple(float(value) for value in durations) if persistent_timing else None)
        edge_cache[key] = result
        return result

    def expand(label: int, entry: tuple[float, int, float, float]) -> None:
        """Evaluate both legal endpoint roles without losing their unopened alternatives."""
        nonlocal cache_size, cache_peak, cache_hits, cache_misses
        if forward_dp and deadline is not None and time.perf_counter() >= deadline:
            raise ForwardLimit("deadline")
        integer_view: cython.longlong[:, ::1] = labels.int_view
        float_view: cython.double[:, ::1] = labels.float_view
        index: cython.Py_ssize_t = label
        node = integer_view[0, index]
        context = integer_view[2, index]
        parent_paid = paid_arrays[integer_view[3, index]]
        sequence = integer_view[4, index]
        first: bool = bool(integer_view[6, index])
        stamp: float = float_view[0, index]
        length: float = float_view[1, index]
        positive: float = float_view[2, index]
        canonical_length: float = float_view[3, index]
        _order, target, distance, remaining = entry
        lower = length + distance
        if lower + remaining > graph.max_length + 1e-7:
            return
        if max(labels.floors[label], bound(context, lower, remaining, positive)) >= limit:
            return
        goal = contexts[context][3]
        roles = (False,)
        if cells[target] == goal:
            roles = (True, False)
        for terminal in roles:
            data = prepared_edge(node, target, first, terminal)
            if data is None:
                continue
            durations, lengths, offsets, spaces, canonical_increment, duration_key = data
            absolute_times = graph.options.arrival_snap_s > 0.0
            timed_key = node, target, first, terminal, stamp
            cached = timed_cache.get(timed_key) if cache_budget else None
            if cached is not None:
                cache_hits += 1
                child_stamp, encountered, blocked = cached
            else:
                table.generation += 1
                if persistent_timing:
                    ends, firsts, stops = get_timing(
                        stamp, duration_key, graph.options.arrival_snap_s,
                        graph.options.speed_fraction, cfg.dt_s, cfg.time_buffer_s,
                    )
                    cached = prepare_windowed_edge(
                        stamp, ends, firsts, stops, offsets, spaces, table.offsets,
                        table.steps, table.row_ids, table.blocked, table.marks, table.generation,
                    )
                    child_stamp, encountered, blocked = cached
                else:
                    times = np.asarray(edge_end_times(
                        stamp, durations, graph.options.arrival_snap_s, graph.options.speed_fraction,
                    )) if absolute_times else durations
                if cache_budget and not persistent_timing:
                    cached = prepare_timed_edge(
                        stamp, times, offsets, spaces, table.offsets, table.steps,
                        table.row_ids, table.blocked, table.marks, table.generation,
                        cfg.dt_s, cfg.time_buffer_s, absolute_times,
                    )
                    child_stamp, encountered, blocked = cached
                if cache_budget:
                    cache_misses += 1
                    # Account for array payload plus estimated Python entry overhead.
                    entry_bytes = encountered.nbytes + 512
                    if entry_bytes <= cache_budget:
                        while cache_size + entry_bytes > cache_budget:
                            _key, evicted = timed_cache.popitem(last=False)
                            cache_size -= evicted[1].nbytes + 512
                        timed_cache[timed_key] = cached
                        cache_size += entry_bytes
                        cache_peak = max(cache_peak, cache_size)
                elif not persistent_timing:
                    child_stamp, child_length, paid, added, blocked = evaluate_edge(
                        stamp, length, parent_paid, times, lengths, offsets, spaces,
                        table.offsets, table.steps, table.row_ids, table.blocked, table.positive,
                        table.marks, table.generation, cfg.dt_s, cfg.time_buffer_s,
                        absolute_times,
                    )
            if blocked:
                continue
            if cache_budget or persistent_timing:
                table.generation += 1
                child_length, paid, added = apply_timed_edge(
                    length, parent_paid, lengths, encountered, table.positive,
                    table.marks, table.generation,
                )
            if child_length + remaining > graph.max_length + 1e-7:
                continue
            child_positive = positive + added
            floor = max(labels.floors[label], bound(context, child_length, remaining, child_positive))
            if floor >= limit:
                continue
            sequence_key = sequence, node, target, first, terminal
            child_sequence = sequence_cache.get(sequence_key)
            if child_sequence is None:
                child_sequence = sequence
                for part_length in lengths:
                    key = child_sequence, float(part_length)
                    identifier = sequence_ids.get(key)
                    if identifier is None:
                        identifier = len(sequence_ids) + 1
                        sequence_ids[key] = identifier
                    child_sequence = identifier
                # This memo is optional; avoid a second unbounded history structure.
                if len(sequence_cache) < 32768:
                    sequence_cache[sequence_key] = child_sequence
            enqueue(target, label, context, paid, child_sequence, terminal, False,
                    child_stamp, child_length, child_positive, floor,
                    canonical_length + canonical_increment,
                    labels.retired[label],
                    # A sink has no continuation, so every heading is future-equivalent.
                    0 if turns is None or terminal else turns.heading(cells[node], cells[target]))

    try:
        setup_started = time.perf_counter()
        for column in known_routes:
            if deadline is not None and time.perf_counter() >= deadline:
                return finish(False, "deadline", global_floor)
            accept(column, known=True)
        if incumbent is None:
            for column in graph.seed_routes(model, deadline):
                accept(column)
        incumbent_setup_s = time.perf_counter() - setup_started
        if early_candidate_ready:
            # Roots are not built yet; retain the same global floor used on interrupted setup.
            return finish(False, "improving_column", global_floor)
        if forward_dp:
            setup_started = time.perf_counter()
            try:
                topology = ForwardTopology.build(graph, deadline)
            except TimeoutError:
                topology_setup_s = time.perf_counter() - setup_started
                return finish(False, "deadline", global_floor)
            topology_setup_s = time.perf_counter() - setup_started
            assert topology.cells == cells
            frontier = ForwardFrontier(topology)
        if graph.options.lattice_distance_bound:
            from .lattice_bound import LatticeDistanceBound
            bound_started = time.perf_counter()
            lattice_bound = LatticeDistanceBound.build(graph, model, prices, history_guard,
                                                       deadline=deadline)
            bound_build_s += time.perf_counter() - bound_started
        if graph.options.completion_bounds:
            from .completion import CompletionBounds
            completion_bounds = CompletionBounds.build(graph, prices, model, deadline)
            bound_build_s += completion_bounds.build_s
            if not completion_bounds.components:
                completion_bounds = None
        for dep in range(fg.base_step, fg.latest_departure_step + 1):
            if deadline is not None and time.perf_counter() >= deadline:
                return finish(False, "deadline", global_floor)
            ground = model.ground_weight * ((dep - fg.base_step) * cfg.dt_s)
            # Even granting every negative row price cannot rescue this or a later departure.
            if price_floor(ground, 0.0, negative, price_error, tol) >= limit:
                break
            origin_rows = graph.rows(graph.endpoint(True, dep * cfg.dt_s))
            if not origin_rows.isdisjoint(forbidden_rows):
                continue
            paid = np.asarray(sorted(table.ids[row] for row in origin_rows if row in prices), dtype=np.int64)
            positive = math.fsum(prices[row] for row in origin_rows if prices.get(row, 0.0) > 0.0)
            for oi, start, lane_steps in graph.origins:
                for di, goal in graph.destinations:
                    if topology is not None and not topology.reaches(cell_ids[goal])[cell_ids[start]]:
                        continue
                    context = len(contexts)
                    contexts.append((dep, oi, di, goal, ground, graph.fold_length(oi, di)))
                    stamp = (dep + fg.takeoff_steps[0] + lane_steps) * cfg.dt_s
                    remaining = (math.dist(graph.point(start), graph.point(goal))
                                 if lattice_bound is None else lattice_bound.remaining(start, goal))
                    floor = bound(context, 0.0, remaining, positive)
                    if floor < limit:
                        enqueue(cell_ids[start], -1, context, paid, 0, False, True,
                                stamp, 0.0, positive, floor)
                        if graph.options.max_labels is not None and generated >= graph.options.max_labels:
                            return finish(False, "search_cap", global_floor)

        while heap or (frontier is not None and frontier.peek() is not None):
            while heap and history is not None and heap[0][3] in history.dead:
                _floor, _serial, _kind, dead_label, _index = heapq.heappop(heap)
                scans.pop(dead_label, None)
            if not heap and not forward_dp:
                break
            floor = frontier.floor if forward_dp else heap[0][0]
            if floor >= best - tol:
                return finish(True, "optimal", min(best, floor))
            if floor >= limit:
                # Every remaining prefix reaches the pricing target (target_cutoff).
                break
            # All departure roots and deferred continuations are represented in this bound.
            uncertainty = 0.0 if history is None else history.merge_error_bound
            certified_upper = pricing_margin - snapped_lower_bound(min(best, floor), snap)
            if uncertainty > 0.0:
                certified_upper = math.nextafter(certified_upper + uncertainty, math.inf)
            if early_return and certified_upper <= improvement_tolerance:
                return finish(True, "no_improvement", floor)
            if early_candidate_ready:
                return finish(False, "improving_column", floor)
            if deadline is not None and time.perf_counter() >= deadline:
                return finish(False, "deadline", floor)
            if ((graph.options.max_expansions is not None
                 and expanded >= graph.options.max_expansions)
                    or (graph.options.max_labels is not None
                        and generated >= graph.options.max_labels)):
                return finish(False, "search_cap", floor)
            if forward_dp:
                label_floor, label = frontier.pop()
                active_floor = label_floor
                if ((history is not None and label in history.dead) or label_floor >= limit):
                    continue
                kind = index = 0
            else:
                _floor, _serial, kind, label, index = heapq.heappop(heap)
            if kind:
                successor_pops += 1
                entry = (scans[label][0][scans[label][1][index]] if compiled_search
                         else scans[label][index][1])
                schedule(label, index + 1)
                expand(label, entry)
                continue
            expanded += 1
            context = labels.int_view[2, label]
            if labels.int_view[5, label]:
                dep, oi, di, _goal, _ground, _fold = contexts[context]
                path = labels.path(label, cells)
                column = Route(fg.request.flight_id, dep, 0, oi, di, path, 0.0)
                canonical = graph.canonical_route(column, model)
                if canonical is not None:
                    sink_rows = graph.rows(graph.endpoint(False, labels.float_view[0, label]))
                    paid = set(paid_arrays[labels.int_view[3, label]])
                    paid.update(table.ids[row] for row in sink_rows if row in prices)
                    canonical_paid = {table.ids[row] for row in canonical.claims if row in prices}
                    if history is None:
                        matches = paid == canonical_paid
                    else:
                        cutoff = labels.float_view[0, label] - max(0.0, cfg.time_buffer_s)
                        canonical_live = ({row for row in canonical_paid if row_ends[row] > cutoff}
                                          if graph.options.expire_history else canonical_paid)
                        matches = (paid == canonical_live and labels.retired[label]
                                   + history.charge(paid) == history.charge(canonical_paid))
                    if not matches:
                        raise AssertionError("compiled and canonical reservation prices disagree")
                    accept(canonical)
                    if forward_dp and early_candidate_ready:
                        return finish(False, "improving_column", frontier.floor)
                continue
            if lazy:
                schedule(label, 0)
            else:
                for entry in targets(label):
                    expand(label, entry)
        if best > limit:
            # Exhausted below the target: nothing improves, but the incumbent is not proven optimal.
            return finish(True, "no_improvement", limit)
        return finish(True, "optimal" if incumbent else "infeasible", best)
    except ForwardLimit as stopped:
        # The active parent covers every child omitted by an interrupted successor batch.
        return finish(False, str(stopped), min(active_floor, frontier.floor))
    finally:
        # Cython closure environments retain helpers that point back to the whole search.
        # Break those references on every exit so arenas are released without waiting for GC.
        bound = enqueue = intern_paid = prepared_edge = targets = heading_of = None
