"""Fit one Zeta route around the reservations already filed with the simulator."""

from __future__ import annotations

import math
import operator
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

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


@dataclass(frozen=True, slots=True)
class ZetaPlannerParams:
    """Configure an individual Zeta search against the live ledger.

    Attributes
    ------------
    - options (ZetaOptions): waypoint connections and work caps. The default uses
      lazy successor scans with the reference backend.
    - objective (str): weighted total cost or equal-weight total delay.
    - max_air_overrun_hops (int): additional lattice hops in the route length budget.
    - time_limit_s (float | None): optional elapsed time budget per leg.
    """

    options: ZetaOptions = ZetaOptions(lazy_successors=True)
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


def _check_deadline(deadline: float | None) -> None:
    """Stop between preparation stages without using a partial reservation map."""
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError("Zeta reservation preparation deadline")


def _saturated_rows(
    ledger: ReservationLedger,
    cfg: SimConfig,
    catalog: StaticTerminalCatalog,
    row_index: RowIndex,
    deadline: float | None = None,
) -> frozenset[RowKey]:
    """Count live buffered reservations once per flight and block saturated rows.

    Parameters
    ------------
    - ledger (ReservationLedger): live dynamic reservations.
    - cfg (SimConfig): reservation geometry and row clock.
    - catalog (StaticTerminalCatalog): walls filed independently of dynamic reservations.
    - row_index (RowIndex): capacity of each cell or terminal row.
    - deadline (float | None): cooperative cutoff checked between volume conversions.

    Return
    --------
    - output (frozenset[RowKey]): saturated rows, including pad rows with multiple occupants.
    """
    protected = frozenset(terminal.id for _, terminal in catalog.entries)
    reservations: dict[int, set[RowKey]] = defaultdict(set)
    for flight_id, volume in ledger.iter_committed():
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
        reject_itinerary(req, "zeta")
        started = time.perf_counter()
        budget = self.params.time_limit_s
        deadline = None if budget is None else started + budget
        self.last_result = None
        if cfg.n_levels != 1:
            raise NotImplementedError("zeta requires a single configured flight level")
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
            forbidden = _saturated_rows(ledger, cfg, catalog, row_index, deadline)
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
