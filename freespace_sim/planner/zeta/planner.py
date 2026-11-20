"""Fit one Zeta route around the reservations already filed with the simulator."""

from __future__ import annotations

import importlib.util
import math
import operator
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from ...config import SimConfig
from ...geometry import CylinderSpec
from ...ledger import ReservationLedger
from ...types import DenialReason, FlightRequest, IntentStatus, OperationalIntent
from ...volumes import ground_dwell_reservation
from ..itinerary import reject_itinerary
from ..route_cost import cost_model
from ..route_domain import (
    FlightGraphInfeasible,
    RowIndex,
    RowKey,
    StaticTerminalCatalog,
    build_flight_graph,
)
from .geometry import volume_claims
from .model import SearchSettings
from .search import SearchResult, ZetaGraph, ZetaOptions, search

# Absolute slack (m) on every footprint margin below; it only widens the exact ledger read.
_SCOPE_GUARD_M = 1.0


def tuned_options(*, native: bool = True) -> ZetaOptions:
    """Return the fastest measured search settings over the default route domain.

    Every option here only prunes or caches; none changes the route domain (radius,
    connection mode, forward restriction, arrival snapping). The history merges certify
    results within their reported guarded allowance rather than exactly.

    Parameters
    ------------
    - native (bool): use the Cython whole-loop build (``zeta-native`` extra plus a C
      compiler); False keeps the same pruning on the pure-Python reference backend.

    Return
    --------
    - output (ZetaOptions): tuned search options.
    """
    pruning = dict(
        lazy_successors=True, expire_history=True, history_dominance=True,
        strong_history_dominance=True, history_group_skips=True,
        merge_equivalent_histories=True, merge_shorter_histories=True,
        route_coverage_cache=True, leg_cache=True, persistent_timing_cache=True,
        lattice_distance_bound=True,
    )
    if not native:
        return ZetaOptions(**pruning)
    return ZetaOptions(**pruning, backend="compiled", compiled_loop=True, compiled_history=True,
                       compiled_route_coverage=True, native_state_index=True)


@dataclass(frozen=True, slots=True)
class ZetaPlannerParams:
    """Configure an individual Zeta search against the live ledger.

    Attributes
    ------------
    - options (ZetaOptions): waypoint connections and work caps. The default is
      ``tuned_options()``: full pruning on the Cython native loop.
    - objective (str): weighted total cost or equal-weight total delay.
    - max_air_overrun_hops (int): additional lattice hops in the route length budget.
    - time_limit_s (float | None): optional elapsed time budget per leg.
    """

    options: ZetaOptions = field(default_factory=tuned_options)
    objective: str = "total_cost"
    max_air_overrun_hops: int = 6
    time_limit_s: float | None = None

    def __post_init__(self) -> None:
        """Reject unknown objectives and invalid route or compute budgets."""
        if self.objective not in {"total_cost", "total_delay"}:
            raise ValueError("objective must be 'total_cost' or 'total_delay'")
        if operator.index(self.max_air_overrun_hops) < 0:
            raise ValueError("max_air_overrun_hops must be nonnegative")
        if self.time_limit_s is not None and (
            not math.isfinite(self.time_limit_s) or self.time_limit_s <= 0.0
        ):
            raise ValueError("time_limit_s must be finite and positive")


def validate_zeta_setup(cfg: SimConfig, *, terminal_endpoints: bool) -> None:
    """Refuse a configuration the zeta planner cannot represent, naming the setting.

    Parameters
    ------------
    - cfg (SimConfig): run configuration; zeta searches exactly one flight level.
    - terminal_endpoints (bool): whether a planned leg starts or ends at a hub terminal, which
      needs permanent terminal walls and fixed exit lanes.

    Return
    --------
    - output (None): raises NotImplementedError for an unsupported setting.
    """
    if cfg.n_levels != 1:
        raise NotImplementedError(
            f"the zeta planner searches a single flight level, but flight_levels_m has "
            f"{cfg.n_levels}; pin the scenario to one level, e.g. flight_levels_m=(100.0,)"
        )
    if terminal_endpoints:
        missing = [name for name in ("terminal_airspace_always_active", "fixed_exit_lanes")
                   if not getattr(cfg, name)]
        if missing:
            raise NotImplementedError(
                "the zeta planner needs " + " and ".join(f"{name}=True" for name in missing)
                + " for hub terminal endpoints"
            )


def _check_deadline(deadline: float | None) -> None:
    """Stop between preparation stages without using a partial reservation map."""
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("Zeta reservation preparation deadline")


@dataclass(frozen=True, slots=True)
class _ClaimScope:
    """A space-time region containing every reservation row one leg could ever claim.

    Attributes
    ------------
    - xy_boxes (tuple[tuple[float, float, float, float], ...]): ``(xmin, ymin, xmax, ymax)``
      boxes; a volume that shares a claimable row with the leg meets one of them.
    - t_start (float): no claimable row period starts more than one clock step earlier (s).
    - t_end (float): no claimable row period starts more than one clock step later (s).
    """

    xy_boxes: tuple[tuple[float, float, float, float], ...]
    t_start: float
    t_end: float


def _claim_scope(graph: ZetaGraph) -> _ClaimScope:
    """Bound every row the leg's candidate routes can claim, so the ledger read stays local.

    Both the leg and another volume claim a hex row only when each footprint comes within
    ``1e-8`` m of that hex, so their footprints are within one hex diameter of each other.
    Route footprints are corridor sub-boxes on waypoint links no longer than the connection
    radius (each point lies within half of it of a waypoint, plus half the box diagonal), and
    the two endpoint cylinders. Times run from the earliest departure, less the time buffer,
    to the latest arrival at the slowest permitted speed, plus the longer of the buffer and
    the landing dwell. Terminal pad rows are claimed only by cylinders filed at the hub
    centre, which the endpoint boxes contain.

    Parameters
    ------------
    - graph (ZetaGraph): the leg's frozen waypoint graph, timing bounds and endpoint builder.

    Return
    --------
    - output (_ClaimScope): conservative xy boxes and time window, each widened by a guard.
    """
    fg, cfg, options = graph.fg, graph.cfg, graph.options
    dt = cfg.dt_s
    hex_diameter = 2.0 * cfg.corridor_segment_len_m / math.sqrt(3.0)
    width = cfg.corridor_width_m
    reach = (options.connection_radius_m / 2.0 + math.hypot(width / 2.0, width / 2.0)
             + hex_diameter + _SCOPE_GUARD_M)
    waypoints = set(graph.cells)
    waypoints.update(cell for _index, cell, _steps in graph.origins)
    waypoints.update(cell for _index, cell in graph.destinations)
    boxes = []
    for cell in waypoints:
        x, y, _z = graph.point(cell)
        boxes.append((x - reach, y - reach, x + reach, y + reach))
    takeoff = graph.endpoint(True, 0.0)
    landing = graph.endpoint(False, 0.0)
    guard = hex_diameter + _SCOPE_GUARD_M
    for volume in (takeoff, landing):
        xmin, ymin, _zmin, xmax, ymax, _zmax = volume.flat_aabb()
        boxes.append((xmin - guard, ymin - guard, xmax + guard, ymax + guard))
    # Cruise sub-boxes number at most two per hop of route length; each nominal duration is
    # floored at 1e-3 s, and arrival snapping may slow a link by 1 / (1 - speed_fraction).
    max_parts = 2 * fg.max_air_hops + 2
    cruise_s = ((graph.max_length + 1e-7) / cfg.nominal_speed_mps + max_parts * 1e-3) / (
        1.0 - options.speed_fraction)
    latest_corridor = (fg.latest_departure_step + max(fg.takeoff_steps)
                       + max((steps for _index, _cell, steps in graph.origins), default=0)) * dt
    landing_dwell = landing.t_end - landing.t_start
    t_start = fg.base_step * dt - max(0.0, cfg.time_buffer_s) - dt
    t_end = max(fg.latest_departure_step * dt + takeoff.t_end - takeoff.t_start,
                latest_corridor + cruise_s + max(cfg.time_buffer_s, landing_dwell)) + dt
    return _ClaimScope(tuple(boxes), t_start, t_end)


def _saturated_rows(
    ledger: ReservationLedger,
    cfg: SimConfig,
    catalog: StaticTerminalCatalog,
    row_index: RowIndex,
    deadline: float | None = None,
    scope: _ClaimScope | None = None,
) -> frozenset[RowKey]:
    """Count live buffered reservations once per flight and block saturated rows.

    A scope restricts the read to volumes that can share a row with the leg; on every row
    the leg can claim the result equals the whole-ledger computation, and elsewhere it is a
    subset of it, because an excluded volume claims none of those rows.

    Parameters
    ------------
    - ledger (ReservationLedger): live dynamic reservations.
    - cfg (SimConfig): reservation geometry and row clock.
    - catalog (StaticTerminalCatalog): walls filed independently of dynamic reservations.
    - row_index (RowIndex): capacity of each cell or terminal row.
    - deadline (float | None): cooperative cutoff checked between volume conversions.
    - scope (_ClaimScope | None): the leg's claimable region; None reads the whole ledger.

    Return
    --------
    - output (frozenset[RowKey]): saturated rows, including pad rows with multiple occupants.
    """
    protected = frozenset(terminal.id for _, terminal in catalog.entries)
    reservations: dict[int, set[RowKey]] = defaultdict(set)
    committed = (ledger.iter_committed() if scope is None
                 else ledger.committed_near(scope.xy_boxes, scope.t_start, scope.t_end))
    for flight_id, volume in committed:
        _check_deadline(deadline)
        if (isinstance(volume.shape, CylinderSpec) and volume.terminal_id is not None
                and volume.terminal_id not in protected):
            # A cylinder without a matching wall must retain its spatial footprint.
            volume = replace(volume, terminal_id=None)
        reservations[flight_id].update(volume_claims(
            volume, cfg, row_index=row_index, protected_terminal_ids=protected,
        ))
    _check_deadline(deadline)
    loads = Counter(row for claims in reservations.values() for row in claims)
    return frozenset(row for row, load in loads.items() if load >= row_index.cap(row))


def _reservation_clear(
    intent: OperationalIntent, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig,
) -> bool:
    """Check exact filed geometry and any ground hold on an ordinary pad.

    Parameters
    ------------
    - intent (OperationalIntent): candidate with its final any-angle volume times.
    - req (FlightRequest): leg whose neighboring itinerary segment may require a pad hold.
    - ledger (ReservationLedger): live reservations and permanent terminal walls.
    - cfg (SimConfig): ground pad geometry.

    Return
    --------
    - output (bool): whether the candidate and its required pad hold fit.
    """
    volumes = intent.volumes
    if not volumes or ledger.any_conflict(volumes):
        return False
    holds = []
    if req.parked_since_s is not None and req.origin_terminal is None:
        takeoff = volumes[0].t_start
        if takeoff > req.parked_since_s:
            holds.append(ground_dwell_reservation(
                req.origin, req.parked_since_s, takeoff - req.parked_since_s, cfg,
            ))
    if req.pad_hold_after_s > 0.0 and req.dest_terminal is None:
        # Match itinerary release time even when a buffered corridor ends after the landing column.
        release = max(volume.t_end for volume in volumes)
        holds.append(ground_dwell_reservation(req.dest, release, req.pad_hold_after_s, cfg))
    return not holds or not ledger.conflicting_flights(holds)


class ZetaPlanner:
    """Plan one flight at a time using the simulator's FCFS filing order.

    Attributes
    ------------
    - params (ZetaPlannerParams): configured geometry, objective and search limits.
    - last_result (SearchResult | None): result of the latest leg search, including
      its completion status when a feasible route survived an interrupted search.
    """

    plans_terminal_airspace = True

    def __init__(self, params: ZetaPlannerParams | None = None) -> None:
        """Store standalone search settings and initialize per-leg diagnostics."""
        if params is not None and not isinstance(params, ZetaPlannerParams):
            raise TypeError("zeta planner params must be ZetaPlannerParams")
        self.params = params if params is not None else ZetaPlannerParams()
        if self.params.options.compiled_loop and importlib.util.find_spec("Cython") is None:
            # The native loop never falls back silently; say how to get it or opt out.
            raise RuntimeError(
                "the zeta planner's compiled loop needs the 'zeta-native' extra (Cython) and a "
                "C compiler: install it with `uv sync --extra zeta-native`, or pass "
                "ZetaPlannerParams(options=tuned_options(native=False))"
            )
        self.last_result: SearchResult | None = None

    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig,
    ) -> OperationalIntent:
        """Search for the least-cost admissible route for one leg.

        Parameters
        ------------
        - req (FlightRequest): leg to plan, including its requested departure and terminals.
        - ledger (ReservationLedger): authoritative committed reservations and walls.
        - cfg (SimConfig): physical, timing and cost configuration.

        Return
        --------
        - output (OperationalIntent): a conflict-checked plan ready for FCFS filing or a denial
          distinguishing an infeasible route from an interrupted search.
        """
        validate_zeta_setup(cfg, terminal_endpoints=(
            req.origin_terminal is not None or req.dest_terminal is not None))
        reject_itinerary(req, "zeta")
        if self.params.options.compiled_loop:
            from .native import prepare_compiled_loop

            # Build once outside the leg's budget: a cold build (~8 s) under a shorter budget is
            # killed on every leg and caches nothing, so every leg would be denied as a deadline.
            prepare_compiled_loop()
        started = time.perf_counter()
        budget = self.params.time_limit_s
        deadline = None if budget is None else started + budget
        self.last_result = None
        try:
            _check_deadline(deadline)
            catalog = StaticTerminalCatalog(ledger.static_terminals(), cfg)
            row_index = RowIndex()
            for _, terminal in catalog.entries:
                row_index.register_terminal(terminal)
            _check_deadline(deadline)
            fg = build_flight_graph(req, cfg, catalog, self.params)
            _check_deadline(deadline)
            graph = ZetaGraph(
                fg, cfg, row_index, options=self.params.options,
                endpoint_validator=lambda intent: _reservation_clear(intent, req, ledger, cfg),
            )
            forbidden = _saturated_rows(ledger, cfg, catalog, row_index, deadline,
                                        _claim_scope(graph))
            result = search(
                graph, {}, SearchSettings(model=cost_model(cfg, self.params)),
                forbidden_rows=forbidden, deadline=deadline,
            )
            self.last_result = result
            if result.route is not None:
                return replace(
                    graph.intent(result.route), request=req, planner="zeta",
                    solve_time_s=time.perf_counter() - started,
                )
            reason = DenialReason.BUDGET_EXCEEDED if result.complete else DenialReason.SEARCH_EXHAUSTED
        except FlightGraphInfeasible:
            reason = DenialReason.BUDGET_EXCEEDED
        except TimeoutError:
            self.last_result = SearchResult(None, math.inf, -math.inf, False, "deadline", 0, 0,
                                            time.perf_counter() - started)
            reason = DenialReason.SEARCH_EXHAUSTED
        return OperationalIntent(
            request=req, status=IntentStatus.REJECTED, volumes=[], centerline=[],
            denial_reason=reason, planner="zeta", solve_time_s=time.perf_counter() - started,
        )
