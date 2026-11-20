"""Experimental exact any-angle pricing over a declared finite waypoint graph.

The master still receives ordinary coefficient-one hex/time and terminal claims.
Optional exact history pruning and speed-feasible arrival snapping retain certificates
over the configured finite route and timing choices.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import heapq
import itertools
import math
import operator
import time
from typing import AbstractSet

from ...config import SimConfig
from ...conflict import volumes_conflict
from ...geometry import BoxSpec
from ...types import OperationalIntent
from ...volumes import (
    Volume4D, column_dwell_s, corridor_segment_volume, enroute_reference_m,
    exit_radius, hover_reservation, segment_overlaps_column, terminal_radius,
)
from .. import hexgrid as hg
from ..route_domain import FlightGraph, RowIndex, RowKey
from ..route_cost import CostModel
from .model import SearchSettings
from ..route_geometry import _fold_leg_s, _terminal_fold_leg_s
from .model import Route
from .geometry import anyangle_to_intent, reservation_claims, volume_claims
from .history import PriceHistory
from .history_guard import HistoryGuard
from .price_floor import prepare_price_floor, price_floor
from .timing import edge_end_times

Cell = tuple[int, int]
Point = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ZetaOptions:
    """Finite connection choices and explicit, non-certifying compute caps.

    Attributes
    ------------
    - connection_mode (str): ``full`` radius graph, ``hex12`` nearest twelve links,
      or ``hex12_long`` with twelve additional links of length sqrt(7) pitches.
    - connection_radius_m (float): Euclidean limit applied to every connection mode.
    - seed_validation_radius_m (float | None): optional full-graph radius used only
      to validate explicitly supplied initial columns; does not widen search or its certificate.
    - merge_equivalent_histories (bool): merge guarded equal-price future states across
      segment-length orders, including their numerical uncertainty in pricing bounds.
    - forward_cruise (bool): require more than 1e-8 m progress along the fixed
      origin-to-destination XY direction on every cruise waypoint link.
    - lattice_distance_bound (bool): strengthen remaining distance on supported
      twelve-direction step sets, with conservative numerical and endpoint guards.
    - merge_shorter_histories (bool): retain guarded equal-price prefixes no longer
      in either accumulated length, with the same whole-route numerical allowance.
    - traversal (str): ``best_first`` or opt-in ``forward_dp`` waypoint sweep;
      the latter requires the compiled backend and forward-only twelve-link graph.
    - native_state_index (bool): intern exact continuation keys in native numeric tables;
      available to every connection mode and traversal using the compiled loop.
    """

    connection_radius_m: float = 480.0
    max_expansions: int | None = 50_000
    max_labels: int | None = 250_000
    merge_identical_states: bool = True
    backend: str = "reference"
    lazy_successors: bool = False
    expire_history: bool = False
    history_dominance: bool = False
    arrival_snap_s: float = 0.0
    speed_fraction: float = 0.05
    compiled_search: bool = True
    timed_cache_bytes: int = 16 * 1024 * 1024
    early_return: bool = False
    compiled_loop: bool = False
    completion_bounds: tuple[str, ...] = ()
    leg_cache: bool = False
    persistent_timing_cache: bool = False
    strong_history_dominance: bool = False
    compiled_history: bool = False
    history_group_skips: bool = False
    route_coverage_cache: bool = False
    compiled_route_coverage: bool = False
    connection_mode: str = "full"
    seed_validation_radius_m: float | None = None
    merge_equivalent_histories: bool = False
    forward_cruise: bool = False
    lattice_distance_bound: bool = False
    merge_shorter_histories: bool = False
    traversal: str = "best_first"
    native_state_index: bool = False

    def __post_init__(self) -> None:
        """Reject nonfinite geometry and invalid enabled compute limits."""
        if not math.isfinite(self.connection_radius_m) or self.connection_radius_m <= 0:
            raise ValueError("connection radius must be finite and positive")
        if self.seed_validation_radius_m is not None and (
            not math.isfinite(self.seed_validation_radius_m)
            or self.seed_validation_radius_m < self.connection_radius_m
        ):
            raise ValueError("seed_validation_radius_m must be finite and at least connection_radius_m")
        if self.connection_mode not in ("full", "hex12", "hex12_long"):
            raise ValueError("connection_mode must be 'full', 'hex12' or 'hex12_long'")
        for name in ("max_expansions", "max_labels"):
            value = getattr(self, name)
            if value is not None and operator.index(value) <= 0:
                raise ValueError(f"{name} must be a positive integer or None")
        if not math.isfinite(self.arrival_snap_s) or self.arrival_snap_s < 0:
            raise ValueError("arrival_snap_s must be finite and nonnegative")
        if not math.isfinite(self.speed_fraction) or not 0 <= self.speed_fraction < 1:
            raise ValueError("speed_fraction must be finite and in [0, 1)")
        if self.backend not in ("reference", "compiled"):
            raise ValueError("backend must be 'reference' or 'compiled'")
        if self.traversal not in ("best_first", "forward_dp"):
            raise ValueError("traversal must be 'best_first' or 'forward_dp'")
        if self.traversal == "forward_dp" and (
            self.backend != "compiled" or not self.forward_cruise
            or self.connection_mode != "hex12"
        ):
            raise ValueError("forward_dp requires compiled backend, forward_cruise and hex12")
        if self.compiled_loop and self.backend != "compiled":
            raise ValueError("compiled_loop requires backend='compiled'")
        if self.strong_history_dominance and not self.history_dominance:
            raise ValueError("strong_history_dominance requires history_dominance=True")
        if self.history_group_skips and not self.strong_history_dominance:
            raise ValueError("history_group_skips requires strong_history_dominance=True")
        if self.merge_equivalent_histories and not self.strong_history_dominance:
            raise ValueError("merge_equivalent_histories requires strong_history_dominance=True")
        if self.merge_shorter_histories and not self.strong_history_dominance:
            raise ValueError("merge_shorter_histories requires strong_history_dominance=True")
        if self.compiled_history and not self.compiled_loop:
            raise ValueError("compiled_history requires compiled_loop=True")
        if self.native_state_index and not self.compiled_loop:
            raise ValueError("native_state_index requires compiled_loop=True")
        if self.compiled_route_coverage and not (self.compiled_loop and self.route_coverage_cache):
            raise ValueError("compiled_route_coverage requires compiled_loop=True and route_coverage_cache=True")
        if operator.index(self.timed_cache_bytes) < 0:
            raise ValueError("timed_cache_bytes must be a nonnegative integer")
        choices = tuple(self.completion_bounds)
        if (len(set(choices)) != len(choices)
                or any(name not in {"arrival", "resource", "selected", "crossing"} for name in choices)):
            raise ValueError("completion_bounds must contain distinct supported bound names")
        object.__setattr__(self, "completion_bounds", choices)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A feasible route, its priced cost and a conservative minimization lower bound.

    With a target cost, ``complete`` can mean ``no_improvement`` at that target;
    without a target, only exhaustion or a valid optimality bound completes search.

    Attributes
    ------------
    - route (Route | None): canonical feasible route, if one was found.
    - priced_cost (float): physical cost plus signed prices over the union of claims.
    - lower_bound (float): conservative bound over all unprocessed states and seeds.
    - raw_lower_bound (float): pre-merge-error floor for numerical adapter diagnostics.
    - equivalent_histories_merged (int): guarded equal-price histories represented by another prefix.
    - merge_error_bound (float): whole-route numerical allowance included in lower_bound.
    - shorter_histories_merged (int): subset of guarded merges comparing both lengths.
    """

    route: Route | None
    priced_cost: float
    lower_bound: float
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
    raw_lower_bound: float = -math.inf


@dataclass(frozen=True, slots=True)
class _Part:
    """One builder-identical static sub-box and its exact duration increment."""

    shape: BoxSpec
    duration: float
    length: float
    terminal_id: object


@dataclass(frozen=True, slots=True)
class _Label:
    """A physical prefix, live priced reservations and an exact retired charge."""

    path: tuple[Cell, ...]
    departure: int
    origin_lane: int | None
    dest_lane: int | None
    goal: Cell
    stamp: float
    length: float
    paid: frozenset[RowKey]
    positive_paid: float
    ground_cost: float
    fold_m: float | None
    terminal: bool = False
    segment_lengths: tuple[float, ...] = ()
    canonical_length: float = 0.0
    retired_charge: int = 0
    history_id: int = -1
    priced_floor: float = -math.inf


@dataclass(frozen=True, slots=True)
class _Scan:
    """Unexamined successors whose smallest completion bound remains on the heap."""

    label: _Label
    targets: tuple[tuple[float, Cell], ...]
    index: int = 0


class ZetaGraph:
    """A flight's finite candidate waypoints, geometry caches and shared claim authority.

    Waypoints are the existing flight corridor's hex centres. Connections may join
    any two in the configured connection set and radius. New long connections cannot cut through
    an own-terminal exit disk; adjacent connections retain their existing choices.
    Walks are bounded by the existing lateral distance budget; forward-only cruise
    additionally excludes waypoint revisits and nonpositive projected progress.
    """

    def __init__(
        self, fg: FlightGraph, cfg: SimConfig, row_index: RowIndex,
        options: ZetaOptions = ZetaOptions(),
        *, endpoint_validator: Callable[[OperationalIntent], bool] | None = None,
    ) -> None:
        """Freeze the declared route universe and validate terminal protection.

        Parameters
        ------------
        - fg (FlightGraph): existing departure, terminal and corridor choices.
        - cfg (SimConfig): unchanged simulator geometry and physical speed.
        - row_index (RowIndex): registered terminal capacities used by the master.
        - options (ZetaOptions): maximum connection length and compute caps.
        - endpoint_validator (Callable | None): frozen endpoint feasibility check. It must
          depend only on departure/context and final arrival, or reject routes already
          ruled out by reservation rows; arbitrary path predicates invalidate dominance.

        Return
        --------
        - output (None): initializes lazy static geometry and seed caches.
        """
        if cfg != fg._cfg or len(fg.levels) != 1:
            raise ValueError("prototype requires the graph's one-level configuration")
        self.fg, self.cfg, self.row_index, self.options = fg, cfg, row_index, options
        self.endpoint_validator = endpoint_validator
        self.pitch = cfg.nominal_speed_mps * cfg.dt_s
        self.max_length = fg.max_air_hops * self.pitch
        self.cells = frozenset(fg.corridor_cells)
        self.protected = frozenset(w.terminal_id for w in fg.static_walls)
        for point, terminal in ((fg.request.origin, fg.origin_terminal),
                                (fg.request.dest, fg.dest_terminal)):
            if terminal is None:
                continue
            matching = [w for w in fg.static_walls if w.terminal_id == terminal.id]
            if not matching or not any(
                abs(w.shape.cx - float(point[0])) < 1e-9
                and abs(w.shape.cy - float(point[1])) < 1e-9
                and abs(w.shape.radius - terminal_radius(terminal, cfg)) < 1e-9
                for w in matching
            ):
                raise ValueError("each endpoint terminal needs its matching permanent wall")
            if row_index.cap(RowKey.term(terminal.id, 0)) != terminal.capacity:
                raise ValueError("terminal wall and master capacities disagree")
        self.origins = tuple((i, lane.cell, lane.steps) for i, lane in enumerate(fg.origin_lanes)) if fg.origin_terminal else ((None, fg.origin_cell, 0),)
        self.destinations = tuple((i, lane.cell) for i, lane in enumerate(fg.dest_lanes)) if fg.dest_terminal else ((None, fg.dest_cell),)
        self.reference = enroute_reference_m(fg.request.origin, fg.request.dest,
                                            fg.origin_terminal, fg.dest_terminal, cfg)
        self._points: dict[Cell, Point] = {}
        self._edges: dict[tuple[Cell, Cell, bool, bool], tuple[_Part, ...] | None] = {}
        self._neighbors: dict[Cell, tuple[Cell, ...]] = {}
        self._seeds: tuple[tuple[int | None, int | None, tuple[Cell, ...]], ...] | None = None
        self._canonical: dict[tuple[object, ...], Route | None] = {}
        from .route_cache import RouteCoverageCache
        self.route_coverage = RouteCoverageCache()
        radius = hg.circumradius(cfg)
        extent = math.ceil(2 * options.connection_radius_m / self.pitch) + 1
        allowed_norms = ({1, 3} if options.connection_mode == "hex12"
                         else {1, 3, 7} if options.connection_mode == "hex12_long" else None)
        dx = float(fg.request.dest[0] - fg.request.origin[0])
        dy = float(fg.request.dest[1] - fg.request.origin[1])
        magnitude = math.hypot(dx, dy)
        forward = (dx / magnitude, dy / magnitude) if magnitude else (0.0, 0.0)
        self._offsets = tuple((q, r) for q in range(-extent, extent + 1)
                              for r in range(-extent, extent + 1)
                              if (q or r) and (allowed_norms is None or q*q + q*r + r*r in allowed_norms)
                              and math.hypot(*hg.hex_center(q, r, radius)) <= options.connection_radius_m + 1e-8)
        if options.forward_cruise:
            self._offsets = tuple((q, r) for q, r in self._offsets
                                  if math.fsum(a*b for a, b in zip(
                                      hg.hex_center(q, r, radius), forward,
                                  )) > 1e-8)

    def point(self, cell: Cell) -> Point:
        """Return one stable physical waypoint without allocating NumPy arrays."""
        if cell not in self._points:
            x, y = hg.hex_center(*cell, hg.circumradius(self.cfg))
            self._points[cell] = (float(x), float(y), self.fg.levels[0])
        return self._points[cell]

    def neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        """Return every endpoint in the declared connection set, radius and corridor."""
        if cell not in self._neighbors:
            q, r = cell
            self._neighbors[cell] = tuple((q + dq, r + dr) for dq, dr in self._offsets
                                          if (q + dq, r + dr) in self.cells)
        return self._neighbors[cell]

    def _hits_static_wall(self, volume: Volume4D) -> bool:
        """Use the graph's conservative wall index before the exact conflict predicate."""
        # The index preserves touching AABBs and wall order; terminal exemptions remain
        # the simulator predicate's responsibility, including tagged corridor boxes.
        return any(
            volume.time_overlaps(wall) and volumes_conflict(volume, wall)
            for wall, _bound in self.fg._wall_index.candidates(volume.flat_aabb())
        )

    def _cuts_own_disk(self, a: Point, b: Point) -> bool:
        """Exclude only new long chords through an own terminal's exit disk."""
        if math.dist(a, b) <= self.pitch + 1e-8:
            return False
        dx, dy = b[0] - a[0], b[1] - a[1]
        for center, term in ((self.fg.request.origin, self.fg.origin_terminal),
                             (self.fg.request.dest, self.fg.dest_terminal)):
            if term is None:
                continue
            t = max(0.0, min(1.0, ((float(center[0]) - a[0]) * dx +
                                  (float(center[1]) - a[1]) * dy) / (dx * dx + dy * dy)))
            if math.hypot(a[0] + t * dx - float(center[0]),
                          a[1] + t * dy - float(center[1])) < exit_radius(term, self.cfg) - 1e-8:
                return True
        return False

    def parts(self, source: Cell, target: Cell, first: bool, last: bool) -> tuple[_Part, ...] | None:
        """Cache exact sub-box geometry, duration increments and static feasibility.

        Parameters
        ------------
        - source (Cell): segment source waypoint.
        - target (Cell): segment destination waypoint.
        - first (bool): whether this is the route's first segment.
        - last (bool): whether this is the route's final segment.

        Return
        --------
        - output (tuple[_Part, ...] | None): builder-equivalent subsegments, or forbidden edge.
        """
        key = source, target, first, last
        if key in self._edges:
            return self._edges[key]
        if source not in self.cells or target not in self.neighbors(source):
            return None
        a, b = self.point(source), self.point(target)
        if self._cuts_own_disk(a, b):
            self._edges[key] = None
            return None
        cfg, fg = self.cfg, self.fg
        dx, dy = b[0] - a[0], b[1] - a[1]
        total = math.sqrt(dx * dx + dy * dy)
        nsub = max(1, math.ceil(total / cfg.corridor_segment_len_m))
        shared = None
        if self.options.leg_cache:
            from .leg_cache import get_leg
            shared = get_leg(a, b, cfg)
        parts = []
        for k in range(1, nsub + 1):
            if shared is None:
                f0, f1 = (k - 1) / nsub, k / nsub
                sa = a[0] + f0 * dx, a[1] + f0 * dy, a[2]
                sb = a[0] + f1 * dx, a[1] + f1 * dy, a[2]
                hx, hy = sb[0] - sa[0], sb[1] - sa[1]
                length = math.sqrt(hx * hx + hy * hy)
                duration = max(length / cfg.nominal_speed_mps, 1e-3)
            else:
                part = shared[k - 1]
                sa, sb, length, duration = part.start, part.end, part.length, part.duration
            tag = None
            for center, term in ((fg.request.origin, fg.origin_terminal),
                                 (fg.request.dest, fg.dest_terminal)):
                if term is not None and segment_overlaps_column(sa, sb, center, terminal_radius(term, cfg), cfg):
                    tag = term.id
                    break
            if cfg.fixed_exit_lanes:
                if first and k == 1 and fg.origin_terminal is not None:
                    tag = fg.origin_terminal.id
                if last and k == nsub and fg.dest_terminal is not None and not (first and nsub == 1 and fg.origin_terminal is not None):
                    tag = fg.dest_terminal.id
            volume = (corridor_segment_volume(sa, 0.0, sb, duration, cfg, terminal_id=tag)
                      if shared is None else Volume4D(
                          shared[k - 1].shape, 0.0 - cfg.time_buffer_s,
                          duration + cfg.time_buffer_s, terminal_id=tag,
                      ))
            if self._hits_static_wall(volume):
                self._edges[key] = None
                return None
            parts.append(_Part(volume.shape, duration, length, tag))
        self._edges[key] = tuple(parts)
        return self._edges[key]

    def endpoint(self, origin: bool, stamp: float) -> Volume4D:
        """Build the unchanged customer or terminal hover reservation at its true stamp."""
        fg, cfg = self.fg, self.cfg
        point, term = (fg.request.origin, fg.origin_terminal) if origin else (fg.request.dest, fg.dest_terminal)
        return hover_reservation(point, stamp, cfg, terminal_id=term.id if term else None,
                                 radius=terminal_radius(term, cfg) if term else None,
                                 climb_time_s=column_dwell_s(point, term, cfg, fg.levels[0]))

    def rows(self, volume: Volume4D) -> frozenset[RowKey]:
        """Use the same full-volume claim conversion as canonical returned columns."""
        return volume_claims(volume, self.cfg, row_index=self.row_index,
                             protected_terminal_ids=self.protected)

    def intent(self, column: Route) -> OperationalIntent:
        """Rebuild an any-angle column through the simulator's reservation authority."""
        return anyangle_to_intent(
            column, self.fg.request, self.cfg, arrival_snap_s=self.options.arrival_snap_s,
            speed_fraction=self.options.speed_fraction,
        )

    def canonical_route(self, column: Route, model: CostModel) -> Route | None:
        """Validate a route against this universe and rebuild all cost/claim coefficients.

        Parameters
        ------------
        - column (Route): route whose supplied coefficients are not trusted.
        - model (CostModel): objective weights owned by the current solve.

        Return
        --------
        - output (Route | None): complete physical column, or an infeasible route.
        """
        key = (column.flight_id, column.departure_step, column.level,
               column.origin_lane_idx, column.dest_lane_idx, column.cell_path, model)
        if key in self._canonical:
            return self._canonical[key]
        path = column.cell_path
        length = 0.0
        for i, (a, b) in enumerate(zip(path, path[1:])):
            parts = self.parts(a, b, i == 0, i + 2 == len(path))
            if parts is None:
                self._canonical[key] = None
                return None
            length += sum(p.length for p in parts)
        if length > self.max_length + 1e-7:
            self._canonical[key] = None
            return None
        intent = self.intent(column)
        if (not intent.accepted or any(self._hits_static_wall(v) for v in intent.volumes)
                or (self.endpoint_validator is not None and not self.endpoint_validator(intent))):
            self._canonical[key] = None
            return None
        claims = reservation_claims(intent.volumes, self.cfg, row_index=self.row_index,
                                     protected_terminal_ids=self.protected)
        result = replace(column, physical_cost=model.intent_cost(intent, self.cfg), claims=claims)
        self._canonical[key] = result
        return result

    def _shortest_path(self, start: Cell, goal: Cell, deadline: float | None) -> tuple[Cell, ...] | None:
        """Find a static shortest seed; failure never certifies dynamic infeasibility."""
        serial = itertools.count()
        heap = [(math.dist(self.point(start), self.point(goal)), next(serial), 0.0, (start,))]
        best = {start: 0.0}
        while heap:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            _, _, length, path = heapq.heappop(heap)
            node = path[-1]
            if length > best.get(node, math.inf) + 1e-8:
                continue
            if node == goal and len(path) > 1:
                return path
            for target in self.neighbors(node):
                lower = length + math.dist(self.point(node), self.point(target))
                remaining = math.dist(self.point(target), self.point(goal))
                if lower + remaining > self.max_length + 1e-7 or lower >= best.get(target, math.inf) - 1e-8:
                    continue
                parts = self.parts(node, target, len(path) == 1, target == goal)
                if parts is None:
                    continue
                new_length = length + sum(p.length for p in parts)
                best[target] = new_length
                heapq.heappush(heap, (new_length + remaining, next(serial), new_length, path + (target,)))
        return None

    def seed_routes(self, model: CostModel, deadline: float | None = None) -> tuple[Route, ...]:
        """Construct base-departure spatial seeds for the permitted endpoint lane pairs.

        Parameters
        ------------
        - model (CostModel): physical route objective weights.
        - deadline (float | None): monotonic cutoff; incomplete seeds are not cached.

        Return
        --------
        - output (tuple[Route, ...]): valid base-time seeds sorted by cost and cell path.
        """
        if self._seeds is None:
            paths = []
            complete = True
            for oi, start, _steps in self.origins:
                for di, goal in self.destinations:
                    path = self._shortest_path(start, goal, deadline)
                    if path is not None:
                        paths.append((oi, di, path))
                    if deadline is not None and time.perf_counter() >= deadline:
                        complete = False
                        break
                if not complete:
                    break
            if complete:
                self._seeds = tuple(paths)
        else:
            paths = self._seeds
        result = []
        for oi, di, path in paths:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            col = self.canonical_route(Route(self.fg.request.flight_id, self.fg.base_step,
                                               0, oi, di, path, 0.0), model)
            if col is not None:
                result.append(col)
        return tuple(sorted(set(result), key=lambda c: (c.physical_cost, c.cell_path)))

    def fold_length(self, oi: int | None, di: int | None) -> float | None:
        """Return additive endpoint length only when exact endpoint folding retains both."""
        total = 0.0
        for point, terminal, lanes, index in (
            (self.fg.request.origin, self.fg.origin_terminal, self.fg.origin_lanes, oi),
            (self.fg.request.dest, self.fg.dest_terminal, self.fg.dest_lanes, di),
        ):
            if terminal is None:
                total += _fold_leg_s(point, None, None, self.cfg) * self.cfg.nominal_speed_mps
            else:
                seconds, retained = _terminal_fold_leg_s(point, terminal, lanes[index].cell, self.cfg)
                if not retained:
                    return None
                total += seconds * self.cfg.nominal_speed_mps
        return total


def search(
    graph: ZetaGraph, prices: Mapping[RowKey, float], settings: SearchSettings = SearchSettings(),
    *, known_routes: Sequence[Route] = (), forbidden_rows: AbstractSet[RowKey] = frozenset(),
    deadline: float | None = None,
) -> SearchResult:
    """Search the declared route universe with an optional priced-cost target.

    Parameters
    ------------
    - graph (ZetaGraph): physical candidate graph and canonical reservation authority.
    - prices (Mapping[RowKey, float]): frozen finite row prices; negative prices are retained.
    - settings (SearchSettings): objective, tolerance and optional target cost.
    - known_routes (Sequence[Route]): feasible incumbent routes, revalidated before pruning.
    - forbidden_rows (AbstractSet[RowKey]): saturated rows that every returned route must avoid.
    - deadline (float | None): monotonic cutoff; a cap returns a frontier bound, never convergence.

    Return
    --------
    - output (SearchResult): feasible route, priced cost, conservative lower bound and search status.
      A threshold discovery can return incomplete. ``no_improvement`` certifies the
      supplied target only; setting target_cost=None seeks the best route.
    """
    if not graph.options.leg_cache and not graph.options.persistent_timing_cache:
        return _search(graph, prices, settings, known_routes=known_routes,
                           forbidden_rows=forbidden_rows, deadline=deadline)
    from .leg_cache import stats
    started = time.perf_counter()
    before = stats()
    result = _search(graph, prices, settings, known_routes=known_routes,
                         forbidden_rows=forbidden_rows, deadline=deadline)
    after = stats()
    return replace(
        result, wall_s=time.perf_counter() - started,
        shared_leg_cache_hits=after["leg_hits"] - before["leg_hits"],
        shared_leg_cache_misses=after["leg_misses"] - before["leg_misses"],
        shared_footprint_cache_hits=after["footprint_hits"] - before["footprint_hits"],
        shared_footprint_cache_misses=after["footprint_misses"] - before["footprint_misses"],
        persistent_timing_cache_hits=after["timing_hits"] - before["timing_hits"],
        persistent_timing_cache_misses=after["timing_misses"] - before["timing_misses"],
    )


def _search(
    graph: ZetaGraph, duals: Mapping[RowKey, float], settings: SearchSettings = SearchSettings(),
    *, known_routes: Sequence[Route] = (), forbidden_rows: AbstractSet[RowKey] = frozenset(),
    deadline: float | None = None,
) -> SearchResult:
    """Run the selected pricing backend without cache-statistics bookkeeping.

    Parameters
    ------------
    - graph (ZetaGraph): fixed finite route and timing universe.
    - duals (Mapping[RowKey, float]): frozen signed reservation prices.
    - settings (SearchSettings): objective, tolerance and optional target cost.
    - known_routes (Sequence[Route]): previously supplied canonical routes.
    - forbidden_rows (AbstractSet[RowKey]): current unavailable reservations.
    - deadline (float | None): monotonic cutoff covering setup and search.

    Return
    --------
    - output (SearchResult): feasible route and a valid bound for unexamined work.
    """
    if graph.options.backend == "compiled":
        if getattr(graph.options, "compiled_loop", False):
            from .native import load_compiled_loop
            started = time.perf_counter()
            try:
                price_loop = load_compiled_loop(deadline)
            except TimeoutError:
                # Reuse the ordinary interrupted-setup bound; no search may run after this cutoff.
                from .compact import search_compact
                price_loop = search_compact
                deadline = time.perf_counter()
            result = price_loop(
                graph, duals, settings, known_routes=known_routes,
                forbidden_rows=forbidden_rows, deadline=deadline,
            )
            # First-use compilation and cached loading belong to the caller's elapsed time.
            return replace(result, wall_s=time.perf_counter() - started)
        from .compact import search_compact
        return search_compact(
            graph, duals, settings, known_routes=known_routes,
            forbidden_rows=forbidden_rows, deadline=deadline,
        )
    started = time.perf_counter()
    if (deadline is not None and not math.isfinite(deadline)):
        raise ValueError("deadline must be finite")
    prices = {RowKey(r): float(p) for r, p in duals.items() if p != 0.0}
    if any(not math.isfinite(p) for p in prices.values()):
        raise ValueError("row prices must be finite")
    negative, price_error = prepare_price_floor(prices.values())
    cfg, fg = graph.cfg, graph.fg
    model = settings.model
    best, incumbent = math.inf, None
    pricing_margin = (settings.target_cost or 0.0)
    improvement_tolerance = settings.epsilon
    early_return = graph.options.early_return and settings.target_cost is not None
    known_canonical: set[Route] = set()
    claim_prices: dict[frozenset[RowKey], float] = {}
    route_prices = graph.route_coverage.bind(prices) if graph.options.route_coverage_cache else None
    early_candidate_ready = False
    expanded = generated = 0
    edge_evaluations = successor_pops = 0
    serial = itertools.count()
    heap: list[tuple[float, int, _Label | _Scan]] = []
    seen_states: set[tuple[object, ...]] = set()
    history = (PriceHistory(prices, group_skips=graph.options.history_group_skips,
                            merge_equivalent=graph.options.merge_equivalent_histories,
                            merge_shorter=graph.options.merge_shorter_histories)
               if graph.options.expire_history or graph.options.history_dominance else None)
    history_guard = HistoryGuard(graph, model, prices) if graph.options.strong_history_dominance else None
    merge_shorter = (graph.options.merge_shorter_histories
                     and math.isfinite(model.air_weight) and model.air_weight >= 0.0)
    expired_rows = 0
    completion_bounds = None
    lattice_bound = None
    bound_build_s = 0.0
    bound_lookups = bound_pruned = 0
    tol = min(1e-7, settings.epsilon * 0.1)
    global_floor = price_floor(0.0, 0.0, negative, price_error, tol)

    def finish(complete: bool, reason: str, floor: float) -> SearchResult:
        """Report all pending costs, including the numerical history-merge allowance."""
        rc = pricing_margin - best
        upper = pricing_margin - min(best, floor)
        uncertainty = 0.0 if history is None else history.merge_error_bound
        if uncertainty > 0.0:
            upper = math.nextafter(max(rc, upper) + uncertainty, math.inf)
            if complete and math.isfinite(rc):
                complete = early_return and upper <= improvement_tolerance
                reason = "no_improvement" if complete else "merge_uncertainty"
        return SearchResult(incumbent, best,
                                 (math.nextafter(min(best, floor) - uncertainty, -math.inf)
                                  if uncertainty > 0 else min(best, floor)), complete, reason,
                                 expanded, generated, time.perf_counter() - started,
                                 edge_evaluations, successor_pops,
                                 0 if history is None else history.pruned, expired_rows,
                                 0 if history is None else history.comparisons,
                                 bound_build_s,
                                 0 if completion_bounds is None else completion_bounds.table_states,
                                 bound_lookups, bound_pruned,
                                 strong_history_pruned=0 if history is None else history.strong_pruned,
                                 history_group_reject_skips=0 if history is None else history.group_reject_skips,
                                 history_group_replace_skips=0 if history is None else history.group_replace_skips,
                                 route_coverage_hits=0 if route_prices is None else route_prices.hits,
                                 route_coverage_misses=0 if route_prices is None else route_prices.misses,
                                 equivalent_histories_merged=0 if history is None else history.merged,
                                 merge_error_bound=uncertainty,
                                 lattice_bound_calls=int(lattice_bound is not None),
                                 shorter_histories_merged=0 if history is None else history.shorter_merged, raw_lower_bound=min(best, floor))

    def accept(column: Route, *, known: bool = False) -> None:
        """Validate supplied or discovered incumbents before using their cutoff."""
        nonlocal best, incumbent, early_candidate_ready
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
        if value < best:
            best, incumbent = value, canonical
            # Both associations must pass epsilon, or the driver could reject an early exit.
            early_candidate_ready = (early_return
                                     and canonical not in known_canonical
                                     and (settings.target_cost or 0.0) - best > settings.epsilon
                                     and settings.accepts(canonical.physical_cost, dual_cost))

    def bound(label: _Label, node: Cell | None = None, length: float | None = None) -> float:
        """Lower-bound every completion, including all possible unpaid negative credit."""
        cost = label.ground_cost
        if label.fold_m is not None and graph.reference > 1e-9:
            source = label.path[-1] if node is None else node
            remaining = (math.dist(graph.point(source), graph.point(label.goal))
                         if lattice_bound is None else lattice_bound.remaining(source, label.goal))
            detour = max(0.0, (label.length if length is None else length) + remaining
                         + label.fold_m - graph.reference)
            cost += model.air_weight * detour / cfg.nominal_speed_mps
        return max(label.priced_floor,
                   price_floor(cost, label.positive_paid, negative, price_error, tol))

    def enqueue(label: _Label, floor: float) -> None:
        """Expire unreachable paid rows, then compare only future-equivalent prefixes."""
        nonlocal generated, expired_rows, bound_lookups, bound_pruned
        if graph.options.expire_history:
            cutoff = label.stamp - max(0.0, cfg.time_buffer_s)
            expired = frozenset(row for row in label.paid if (row.step + 1) * cfg.dt_s <= cutoff)
            expired_rows += len(expired)
            label = replace(label, paid=label.paid - expired,
                            retired_charge=label.retired_charge + history.charge(expired))
        if completion_bounds is not None:
            bound_lookups += 1
            floor = max(floor, completion_bounds.floor(
                label.path[-1], label.goal, label.stamp, label.length,
                label.positive_paid, label.ground_cost, negative, label.fold_m,
                label.paid, label.terminal,
            ))
            if floor >= best:
                bound_pruned += 1
                return
            # A child's completions are a subset of its parent's, including deferred scans.
            label = replace(label, priced_floor=floor)
        if label.fold_m is not None or graph.reference <= 1e-9:
            # Both reduction orders must match the canonical cost and length-budget checks.
            key = (label.path[-1], label.departure, label.origin_lane, label.dest_lane,
                   label.stamp, label.length, label.canonical_length, label.terminal,
                   len(label.path) == 1, label.segment_lengths)
            if graph.options.history_dominance:
                margin = None if history_guard is None else history_guard.margin(
                    label.origin_lane, label.dest_lane,
                )
                if margin is None:
                    accepted = history.accept(key, label.paid, label.retired_charge, generated)
                else:
                    accepted = history.accept(key, label.paid, label.retired_charge, generated,
                                              broad_key=key[:-1], margin=margin,
                                              length_key=key[:5] + key[7:9] if merge_shorter else None,
                                              lengths=(label.length, label.canonical_length) if merge_shorter else None)
                if not accepted:
                    return
            elif graph.options.merge_identical_states:
                exact = key + (label.paid, label.retired_charge)
                if exact in seen_states:
                    return
                seen_states.add(exact)
        label = replace(label, history_id=generated)
        heapq.heappush(heap, (floor, next(serial), label))
        generated += 1

    def extend(label: _Label, target: Cell) -> None:
        """Build one candidate connection, preserving both destination endpoint roles."""
        nonlocal edge_evaluations
        node = label.path[-1]
        remaining = (math.dist(graph.point(target), graph.point(label.goal))
                     if lattice_bound is None else lattice_bound.remaining(target, label.goal))
        lower_length = label.length + math.dist(graph.point(node), graph.point(target))
        if lower_length + remaining > graph.max_length + 1e-7:
            return
        trial = replace(label, path=label.path + (target,), length=lower_length)
        if bound(trial) >= best:
            return
        for terminal in ((True, False) if target == label.goal else (False,)):
            edge_evaluations += 1
            parts = graph.parts(node, target, len(label.path) == 1, terminal)
            if parts is None:
                continue
            stamp, length = label.stamp, label.length
            rows: set[RowKey] = set()
            end_times = edge_end_times(stamp, tuple(part.duration for part in parts),
                                       graph.options.arrival_snap_s, graph.options.speed_fraction)
            for part, end in zip(parts, end_times, strict=True):
                volume = Volume4D(part.shape, stamp - cfg.time_buffer_s,
                                  end + cfg.time_buffer_s, part.terminal_id)
                rows.update(graph.rows(volume))
                stamp, length = end, length + part.length
            if not rows.isdisjoint(forbidden_rows):
                continue
            if length + remaining > graph.max_length + 1e-7:
                continue
            new_paid = frozenset(r for r in rows if r in prices) - label.paid
            positive = label.positive_paid + math.fsum(prices[r] for r in new_paid if prices[r] > 0.0)
            child = replace(trial, stamp=stamp, length=length, paid=label.paid | new_paid,
                            positive_paid=positive, terminal=terminal,
                            segment_lengths=label.segment_lengths + tuple(p.length for p in parts),
                            canonical_length=label.canonical_length + sum(p.length for p in parts))
            floor = bound(child)
            if floor < best:
                enqueue(child, floor)

    for column in known_routes:
        if deadline is not None and time.perf_counter() >= deadline:
            return finish(False, "deadline", global_floor)
        accept(column, known=True)
    if incumbent is None:
        for column in graph.seed_routes(model, deadline):
            accept(column)

    if early_candidate_ready:
        # Roots are not built yet; retain the same global floor used on interrupted setup.
        return finish(False, "improving_column", global_floor)

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

    # All departures remain in the certificate, including roots not yet built at a cap.
    for dep in range(fg.base_step, fg.latest_departure_step + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            return finish(False, "deadline", global_floor)
        ground = model.ground_weight * ((dep - fg.base_step) * cfg.dt_s)
        # Even granting every negative row price cannot rescue this or a later departure.
        if price_floor(ground, 0.0, negative, price_error, tol) >= best:
            break
        endpoint = graph.endpoint(True, dep * cfg.dt_s)
        origin_rows = graph.rows(endpoint)
        if not origin_rows.isdisjoint(forbidden_rows):
            continue
        paid = frozenset(r for r in origin_rows if r in prices)
        positive = math.fsum(prices[r] for r in paid if prices[r] > 0.0)
        for oi, start, lane_steps in graph.origins:
            for di, goal in graph.destinations:
                label = _Label((start,), dep, oi, di, goal,
                               (dep + fg.takeoff_steps[0] + lane_steps) * cfg.dt_s,
                               0.0, paid, positive,
                               ground,
                               graph.fold_length(oi, di))
                floor = bound(label)
                if floor < best:
                    enqueue(label, floor)
                    if graph.options.max_labels is not None and generated >= graph.options.max_labels:
                        return finish(False, "search_cap", global_floor)

    while heap:
        while heap and history is not None:
            first = heap[0][2]
            identifier = first.label.history_id if isinstance(first, _Scan) else first.history_id
            if identifier not in history.dead:
                break
            heapq.heappop(heap)
        if not heap:
            break
        floor = heap[0][0]
        if floor >= best - tol:
            return finish(True, "optimal", min(best, floor))
        # All departure roots and deferred continuations are represented in this bound.
        uncertainty = 0.0 if history is None else history.merge_error_bound
        certified_upper = pricing_margin - min(best, floor)
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
        _, _, entry = heapq.heappop(heap)
        if isinstance(entry, _Scan):
            successor_pops += 1
            # Keep all unexamined targets represented before any deadline/cap return.
            next_index = entry.index + 1
            if next_index < len(entry.targets):
                next_floor = entry.targets[next_index][0]
                if next_floor < best:
                    heapq.heappush(heap, (next_floor, next(serial), replace(entry, index=next_index)))
            extend(entry.label, entry.targets[entry.index][1])
            continue
        label = entry
        expanded += 1
        node = label.path[-1]
        if label.terminal:
            column = Route(fg.request.flight_id, label.departure, 0,
                            label.origin_lane, label.dest_lane, label.path, 0.0)
            canonical = graph.canonical_route(column, model)
            if canonical is not None:
                sink_rows = graph.rows(graph.endpoint(False, label.stamp))
                paid = label.paid | frozenset(r for r in sink_rows if r in prices)
                canonical_paid = frozenset(r for r in canonical.claims if r in prices)
                if history is None:
                    matches = paid == canonical_paid
                else:
                    cutoff = label.stamp - max(0.0, cfg.time_buffer_s)
                    canonical_live = (frozenset(row for row in canonical_paid
                                               if (row.step + 1) * cfg.dt_s > cutoff)
                                      if graph.options.expire_history else canonical_paid)
                    matches = (paid == canonical_live and label.retired_charge
                               + history.charge(paid) == history.charge(canonical_paid))
                if not matches:
                    raise AssertionError("incremental and canonical reservation prices disagree")
                accept(canonical)
            continue
        if graph.options.lazy_successors:
            targets = []
            for target in graph.neighbors(node):
                length = label.length + math.dist(graph.point(node), graph.point(target))
                remaining = (math.dist(graph.point(target), graph.point(label.goal))
                             if lattice_bound is None else lattice_bound.remaining(target, label.goal))
                if length + remaining > graph.max_length + 1e-7:
                    continue
                child_floor = bound(label, target, length)
                if child_floor < best:
                    targets.append((child_floor, target))
            if targets:
                targets.sort(key=lambda candidate: candidate[0])
                heapq.heappush(heap, (targets[0][0], next(serial), _Scan(label, tuple(targets))))
            continue
        # Complete each expansion before honoring a cap: every remaining child stays in the heap.
        for target in graph.neighbors(node):
            extend(label, target)
    return finish(True, "optimal" if incumbent else "infeasible", best)
