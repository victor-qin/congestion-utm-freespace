"""Whole-schedule column-generation orchestration.

This module deliberately contains no capacity-row geometry and no optimization
backend details.  It wires the three owners together instead: ``network``
certifies canonical claims, ``pricing`` searches one flight's finite DAG, and
``master`` solves the restricted master problem.  Keeping that boundary sharp
also makes the final repair pass use exactly the same column universe and row
semantics as the LP/IP loop.
"""

from __future__ import annotations

import logging

import collections
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ...config import SimConfig
from ...types import FlightRequest, as_terminal
from .objective import DELAY_MODEL, CostModel, cost_model
from .network import (
    FlightGraph,
    FlightGraphInfeasible,
    RowIndex,
    RowKey,
    StaticTerminalCatalog,
    build_flight_graph,
    column_claims,
)
from .master import BackendTimeout, RestrictedMaster
from .master import gurobi_lp_method as _gurobi_lp_method
from .master import gurobi_threads as _gurobi_threads
from .params import ColGenParams
from .pricing import (
    DualView,
    PricingTimeout,
    find_feasible_column,
    price_flight,
    seed_column,
)
from .pricing_pool import PricingPool, certified_sweep_columns, price_sweep
from .translate import Column, column_to_intent

log = logging.getLogger(__name__)

_FEASIBILITY_TOL = 1e-7
_REDUCED_COST_TOL = 1e-9
_OBJECTIVE_TOL = 1e-8


@dataclass(frozen=True, slots=True)
class ColGenResult:
    """Selected columns and diagnostics for one whole-schedule solve.

    A flight absent from :attr:`columns` is either an exhaustive optimizer
    denial or a compute-cap artifact.  ``budget_denied_flight_ids`` and
    ``search_exhausted_flight_ids`` partition those cases for ``batch.run_batch``,
    which turns the distinction into a :class:`DenialReason`.

    ``global_integer_gap`` (also exposed as ``ip_gap``) certifies the returned
    schedule against the global pricing bound. ``restricted_ip_gap`` and
    ``ip_optimal`` describe only the final IP over its generated column pool.
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


def _coverage_diagnostics(master, x, rc_by_flight, benefit) -> dict:
    """Cross-reference LP coverage against the flights whose reduced cost is ~M.

    A flight the LP leaves uncovered has a slack cover constraint, so complementary
    slackness sets its dual to zero and pricing reports ``rc = M - cost - dual_cost``,
    i.e. roughly the whole benefit.  Those terms then dominate the Lagrangian bound
    ``LP + sum(max(0, rc))``.  This measures the overlap directly rather than inferring
    it, and reports the largest column cost -- the quantity that actually bounds how
    small ``M`` is allowed to be.

    Parameters
    ------------
    - master (RestrictedMaster): the solved master, read for its ``columns``.
    - x (Sequence[float]): the LP primal solution, one weight per master column.
    - rc_by_flight (Mapping[int, float]): per-flight reduced cost from the last pricing sweep.
    - benefit (float): the per-flight benefit ``M`` the near-``M`` test compares against.

    Return
    --------
    - output (dict): ``max_column_cost`` plus the counts ``n_uncovered``, ``n_rc_near_M``, and
      their overlap ``n_overlap``.
    """

    coverage: dict[int, float] = collections.defaultdict(float)
    max_cost = 0.0
    for value, column in zip(np.asarray(x, dtype=float), master.columns, strict=False):
        coverage[column.flight_id] += float(value)
        if column.delay_s > max_cost:
            max_cost = float(column.delay_s)
    uncovered = {f for f, v in coverage.items() if v < 1.0 - 1e-6}
    contaminated = {f for f, rc in rc_by_flight.items() if rc > 0.5 * float(benefit)}
    return {
        "max_column_cost": max_cost,
        "n_uncovered": len(uncovered),
        "n_rc_near_M": len(contaminated),
        "n_overlap": len(uncovered & contaminated),
    }


def _canonical_column(
    column: Column, graph: FlightGraph, cfg: SimConfig, *, model: CostModel | None = None
) -> Column:
    """Certify claims, and reprice imported columns in the current solve's model.

    Internal pricing columns already carry certified costs. Imported seeds may come
    from another objective or config, so their stored cost is never authoritative.
    """

    intent = None if model is None else column_to_intent(column, graph.request, cfg)
    claims = column_claims(column, graph, cfg, _intent=intent)
    cost = column.delay_s if model is None else model.intent_cost(intent, cfg)
    if column.claims == claims and column.delay_s == cost:
        return column
    return replace(column, claims=claims, delay_s=cost)


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


def _shift_column(
    column: Column, departure_step: int, cfg: SimConfig, model: CostModel = DELAY_MODEL
) -> Column:
    """Return the same certified spatial route at a later integer departure.

    Parameters
    ------------
    - column (Column): the seed whose spatial route is preserved.
    - departure_step (int): the new departure step; must be at or after ``column``'s own.
    - cfg (SimConfig): supplies ``dt_s`` for the added ground-delay term.
    - model (CostModel): supplies ``ground_weight`` for the added delay; defaults to
      ``DELAY_MODEL``.

    Return
    --------
    - output (Column): the route at ``departure_step`` with ground delay added and claims
      shifted; raises ``ValueError`` if ``departure_step`` is earlier than the column's.
    """

    delta = departure_step - column.departure_step
    if delta < 0:
        raise ValueError("a seed column can only be shifted later")
    return replace(
        column,
        departure_step=departure_step,
        # A pure clock translation adds GROUND delay only; the route, and so the air
        # term, is invariant.  See colgen.objective.
        delay_s=column.delay_s + model.ground_weight * (delta * cfg.dt_s),
        claims=_shift_claims(column.claims, delta),
    )


def _add_departure_ladder(master, seed, graph, cfg, model, steps: int, stride: int = 1) -> int:
    """Offer the master `steps` pure clock translations of one flight's seed.

    One rung of the warm pool's init (see context/figures/cg_loop.png).
    Pricing otherwise spends its early iterations rediscovering exactly these (most of the
    columns a converged solve adds after the first are time shifts of a route already in
    the pool).  A shift is a `_shift_column` translation -- arithmetic, no DP -- so handing
    them over up front converts that search into addition.

    Depth has a knee; more is not better.  Past the largest delay any optimal schedule
    actually uses, extra rungs add departures no schedule wants and tied columns feed the
    master's degeneracy instead of resolving it.  So the useful depth is set by the
    solution's slip, not by `max_ground_delay_s` (which permits far more).  Denser traffic
    slips further and moves the knee, so the shipped depth is calibrated, not universal.

    Parameters
    ------------
    - master (RestrictedMaster): receives each rung via ``master.add_column``.
    - seed (Column): the flight's seed column whose clock is translated.
    - graph (FlightGraph): the flight's pricing graph, supplying the departure/horizon bounds.
    - cfg (SimConfig): supplies ``dt_s`` and lattice geometry.
    - model (CostModel): cost weights for each shifted column's ``delay_s``.
    - steps (int): maximum number of departure alternatives to offer.
    - stride (int): lattice steps between alternatives; trades resolution for span.

    Return
    --------
    - output (int): the number of ladder columns added (``0`` when ``steps <= 0`` or none fit
      before the flight's latest feasible departure).
    """

    if steps <= 0:
        return 0
    if stride < 1:
        raise ValueError("stride must be positive")
    # BOTH bounds, stated.  `latest_departure_step` alone is not the limit: this path must
    # also arrive by `max_step`, which is the bound `_initial_feasible_selection` computes
    # explicitly for the same reason.  Leaving it to be caught as a `ValueError` from
    # `column_claims` works -- the horizon is the only shift-dependent failure, since a
    # clock translation leaves the wall verdict and the detour budget invariant, and it is
    # monotone in `step` so `break` is right -- but `column_claims` raises `ValueError` for
    # eight other reasons too, and every one of them would be silently read as "the ladder
    # ended here" rather than as the regression it would be.
    origin_lane_steps = (
        0 if seed.origin_lane_idx is None else graph.origin_lanes[seed.origin_lane_idx].steps
    )
    path_latest_departure = (
        graph.max_step
        - graph.takeoff_steps[seed.level]
        - origin_lane_steps
        - (len(seed.cell_path) - 1)
    )
    latest = min(graph.latest_departure_step, path_latest_departure)
    added = 0
    # `steps` stays a COLUMN COUNT under any stride -- the span is `steps * stride` -- so
    # the two knobs are independent and the calibrated depth above keeps its meaning.
    # `latest` still truncates, so a strided ladder on a tight horizon simply yields fewer
    # rungs rather than overshooting it.
    for k in range(1, steps + 1):
        step = seed.departure_step + k * stride
        if step > latest:
            break
        shifted = replace(
            seed, departure_step=step, claims=frozenset(),
            delay_s=seed.delay_s + model.ground_weight * ((step - seed.departure_step) * cfg.dt_s),
        )
        master.add_column(_canonical_column(shifted, graph, cfg))
        added += 1
    return added


def _initial_feasible_selection(
    seeds: Mapping[int, Column],
    graphs: Mapping[int, FlightGraph],
    fixed_loads: Mapping[RowKey, int],
    row_index: RowIndex,
    cfg: SimConfig,
    *,
    deadline: float | None = None,
    model: CostModel = DELAY_MODEL,
) -> dict[int, Column]:
    """Greedily ground-shift seed paths into a globally feasible RMP incumbent.

    This is an acceleration, not a restriction: nominal seeds remain in the
    master, and pricing can add every route in the original universe.  Seeding
    the RMP with a real all-flight incumbent prevents a time-capped solve from
    discarding known coverage and then repeating the same search in repair.

    Parameters
    ------------
    - seeds (Mapping[int, Column]): flight id -> the flight's nominal seed column.
    - graphs (Mapping[int, FlightGraph]): flight id -> the flight's pricing graph.
    - fixed_loads (Mapping[RowKey, int]): pre-committed row occupancy every pick must respect.
    - row_index (RowIndex): supplies each capacity row's ``cap``.
    - cfg (SimConfig): supplies ``dt_s`` and geometry.
    - deadline (float | None): ``time.monotonic`` cutoff; ``None`` disables the timeout.
    - model (CostModel): cost weights for each shift's ``delay_s``; defaults to ``DELAY_MODEL``.

    Return
    --------
    - output (dict[int, Column]): flight id -> chosen shifted column, sorted by flight id;
      flights that never fit (or that the deadline cut off) are omitted.
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
            candidate = _shift_column(seed, departure_step, cfg, model)
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
    initial: Mapping[int, Column] | None = None,
) -> tuple[dict[int, Column], bool]:
    """Improve a complete seed incumbent with lazy best-first local searches.

    The shifted-seed schedule above is a cheap fallback, but it cannot route
    around a busy cell.  This post-first-LP pass removes one flight at a time
    from the incumbent, asks the pricing topology for a lower-delay replacement
    avoiding every other selected column, and restores the old column when no
    improvement is found.  A timeout therefore keeps a complete feasible
    schedule instead of discarding a partially built prefix.  It remains only
    an incumbent heuristic and never contributes to a global bound.

    Parameters
    ------------
    - graphs (Mapping[int, FlightGraph]): flight id -> the flight's pricing graph.
    - fixed_loads (Mapping[RowKey, int]): pre-committed row occupancy every pick must respect.
    - row_index (RowIndex): supplies each capacity row's ``cap``.
    - cfg (SimConfig): supplies geometry and, with ``params``, the cost model.
    - params (ColGenParams): supplies ``n_heuristic_tries`` (the candidate cap) and cost weights.
    - deadline (float): ``time.monotonic`` cutoff for the whole sweep.
    - initial (Mapping[int, Column] | None): the complete incumbent to improve; ``None`` starts
      from an empty selection.

    Return
    --------
    - output (tuple[dict[int, Column], bool]): the improved selection sorted by flight id, and
      a flag that is ``True`` only when every candidate flight was tried within the deadline.
    """

    model = cost_model(cfg, params)
    selected: dict[int, Column] = dict(initial or {})
    loads = _loads_for(selected, fixed_loads)
    saturated = {
        row for row, load in loads.items() if load >= row_index.cap(row)
    }

    def remove_claims(claims: frozenset[RowKey]) -> None:
        for row in claims:
            old_load = loads[row]
            new_load = old_load - 1
            if new_load:
                loads[row] = new_load
            else:
                del loads[row]
            if old_load >= row_index.cap(row) and new_load < row_index.cap(row):
                saturated.discard(row)

    def add_claims(claims: frozenset[RowKey]) -> None:
        for row in claims:
            old_load = loads[row]
            new_load = old_load + 1
            loads[row] = new_load
            if old_load < row_index.cap(row) <= new_load:
                saturated.add(row)

    order = sorted(
        (
            graph
            for graph in graphs.values()
            if graph.request.flight_id not in selected
            or selected[graph.request.flight_id].delay_s > _OBJECTIVE_TOL
        ),
        key=lambda graph: (
            -selected[graph.request.flight_id].delay_s
            if graph.request.flight_id in selected
            else -math.inf,
            graph.request.t_departure,
            graph.request.flight_id,
        ),
    )
    candidate_limit = max(64, params.n_heuristic_tries * 16)
    completed = len(order) <= candidate_limit
    order = order[:candidate_limit]
    for graph_index, graph in enumerate(order):
        if time.monotonic() >= deadline:
            return dict(sorted(selected.items())), False
        flight_id = graph.request.flight_id
        incumbent = selected.pop(flight_id, None)
        if incumbent is not None:
            remove_claims(incumbent.claims)

        now = time.monotonic()
        attempts_remaining = len(order) - graph_index
        # Reserve an equal share for every remaining flight.  Easy A* searches
        # return early and donate their unused share; a hard flight times out
        # locally instead of preventing the rest of the sweep from being tried.
        flight_deadline = min(deadline, now + (deadline - now) / attempts_remaining)
        try:
            candidate = find_feasible_column(
                graph,
                cfg,
                forbidden_rows=saturated,
                improve_below_delay_s=(
                    None if incumbent is None else incumbent.delay_s
                ),
                deadline=flight_deadline,
                model=model,
            )
        except PricingTimeout:
            completed = False
            if incumbent is not None:
                selected[flight_id] = incumbent
                add_claims(incumbent.claims)
            continue
        if candidate is None or (
            incumbent is not None
            and not _better_selection(
                {flight_id: candidate},
                {flight_id: incumbent},
                benefit=0.0,
            )
        ):
            if incumbent is not None:
                selected[flight_id] = incumbent
                add_claims(incumbent.claims)
            continue
        candidate = _canonical_column(candidate, graph, cfg)
        if any(loads[row] + 1 > row_index.cap(row) for row in candidate.claims):
            raise RuntimeError(
                f"greedy pricing returned an infeasible column for flight "
                f"{graph.request.flight_id}"
            )
        selected[flight_id] = candidate
        add_claims(candidate.claims)
    return dict(sorted(selected.items())), completed


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
    # ``delay_s`` already carries whatever the objective weights made it -- see
    # colgen.objective -- so this stays a plain difference.
    return math.fsum(benefit - column.delay_s for column in selection.values())


def _better_selection(
    candidate: Mapping[int, Column],
    incumbent: Mapping[int, Column],
    benefit: float,
) -> bool:
    """Compare maximize-sense incumbents, with a deterministic exact-tie rule.

    Parameters
    ------------
    - candidate (Mapping[int, Column]): the proposed selection.
    - incumbent (Mapping[int, Column]): the selection to beat.
    - benefit (float): the per-flight benefit ``M`` both selections are scored against.

    Return
    --------
    - output (bool): ``True`` when ``candidate`` scores higher (maximize sense), or on an exact
      tie has the lexicographically smaller selection key.
    """

    candidate_obj = _selection_objective(candidate, benefit)
    incumbent_obj = _selection_objective(incumbent, benefit)
    if candidate_obj > incumbent_obj + _OBJECTIVE_TOL:
        return True
    if incumbent_obj > candidate_obj + _OBJECTIVE_TOL:
        return False
    return _selection_key(candidate) < _selection_key(incumbent)


def _relative_revenue_gap(upper_bound: float, rmp_value: float) -> float:
    """The paper's gap, equations (10) and (11): ``(UB - RMP) / RMP``.

    Measured on the maximize objective, whose scale includes ``n * M``.  That
    normalization is the whole difference from :func:`_relative_cost_gap`: the same
    absolute slack reads as a tiny fraction of revenue here but a large fraction of cost
    there, for the identical solution.  The paper's thresholds (0.01% for the LP, 0.1% for
    the heuristic) are calibrated against this revenue scale.
    """

    if not math.isfinite(upper_bound) or not math.isfinite(rmp_value):
        return math.inf
    # Test the numerator before the scale.  A bound that coincides with the RMP has
    # closed, and that is true at any scale including zero -- which is reachable here,
    # because a one-flight master whose only column exactly consumes its benefit prices
    # out at objective 0.  Dividing first would report that tight bound as `inf`.
    absolute = max(0.0, upper_bound - rmp_value)
    if absolute <= 0.0:
        return 0.0
    denominator = abs(rmp_value)
    if denominator <= 0.0:
        return math.inf
    return absolute / denominator


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


def _integer_gap_stats(
    upper_bound: float, objective: float, total_benefit: float, params: ColGenParams
) -> dict[str, float | bool]:
    """Global certificates for the returned integer schedule, including repair.

    The restricted IP's own bound cannot certify columns outside its pool. Keep
    that diagnostic separate, and use the pricing bound on every result path.
    """

    revenue_gap = _relative_revenue_gap(upper_bound, objective)
    cost_gap = _relative_cost_gap(total_benefit - objective, total_benefit - upper_bound)
    gap = revenue_gap if params.gap_metric == "revenue" else cost_gap
    return {
        "global_integer_gap": gap,
        "global_integer_gap_cost": cost_gap,
        "global_integer_gap_revenue": revenue_gap,
        "ip_gap": gap,
        "ip_gap_met": gap <= params.ip_gap,
        "ip_gap_revenue": revenue_gap,
    }


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
    """Independently referee an integer selection against all canonical claims.

    Parameters
    ------------
    - selection (Mapping[int, Column]): the integer selection under audit.
    - fixed_loads (Mapping[RowKey, int]): pre-committed occupancy folded into the totals.
    - row_index (RowIndex): supplies each capacity row's ``cap``.

    Return
    --------
    - output (Counter[RowKey]): the aggregate per-row load (selection plus fixed); raises
      ``RuntimeError`` if any row's load exceeds its capacity.
    """

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


def _dual_regime_stats(backend_name: str) -> dict[str, int | None]:
    """WHICH DUAL VECTOR PRICED THIS RUN, keyed off the backend that ACTUALLY ran.

    Recorded because these are answer-affecting: the same pool priced under simplex and
    under barrier reaches materially different objectives, so an archived run without them
    cannot be compared to anything.  ``lp_method`` ``-1`` is Gurobi's automatic choice;
    ``2 / 0`` is barrier without crossover, the default.

    Keyed on the RESOLVED backend rather than on ``params.solver``, and the difference is
    the whole point: ``solver="auto"`` falls back to HiGHS whenever gurobipy is missing, so
    a `params`-keyed test reports ``lp_method=2`` for a run HiGHS actually priced -- exactly
    the mislabelling these fields exist to prevent.  ``None`` means "no Gurobi LP ran", which
    covers HiGHS, the auto fall-back, and the exits that never build a master at all.

    Returned as a dict spliced into each stats block with ``**``, because
    `test_every_exit_reports_the_same_stats_keys` pins the key SET across all three exits
    and hand-copying these keys into three dicts is what produced a duplicated pair before.
    """

    if backend_name != "gurobi":
        return {"lp_method": None, "lp_crossover": None, "gurobi_threads": None}
    method, crossover = _gurobi_lp_method()
    return {
        "lp_method": method,
        "lp_crossover": crossover,
        "gurobi_threads": _gurobi_threads(),
    }


def _pre_master_timeout_result(
    flight_ids: Sequence[int],
    started: float,
    params: ColGenParams,
    *,
    stage: str = "startup",
    graphs: Mapping[int, FlightGraph] | None = None,
    catalog: StaticTerminalCatalog | None = None,
    graph_build_elapsed_s: float = 0.0,
    seed_elapsed_s: float = 0.0,
    seeds_completed: int = 0,
    seed_flights_processed: int = 0,
    master: RestrictedMaster | None = None,
    time_to_master_s: float = 0.0,
    seedless_flight_ids: Sequence[int] = (),
    graph_infeasible_flight_ids: Sequence[int] = (),
) -> ColGenResult:
    """Return an explicit compute-cap verdict before a usable master exists.

    Parameters
    ------------
    - flight_ids (Sequence[int]): every flight in the batch; all are denied as
      search-exhausted and priced at ``params.M`` each.
    - started (float): the ``time.monotonic`` timestamp the solve began, for the elapsed and
      overrun stats.
    - params (ColGenParams): solver config; supplies ``M``, ``time_limit_s``, ``gap_metric``,
      ``objective``, ``ip_time_limit_s``, and ``warm_start_planner``.
    - stage (str): the pre-master stage that ran out of time, recorded as
      ``preprocessing_stage``.
    - graphs (Mapping[int, FlightGraph] | None): graphs built so far, read for arc-cache and
      corridor-materialization counts.
    - catalog (StaticTerminalCatalog | None): supplies the static-terminal and wall-index
      counts; ``None`` reports zeros.
    - graph_build_elapsed_s (float): seconds spent building graphs, echoed into stats.
    - seed_elapsed_s (float): seconds spent seeding, echoed into stats.
    - seeds_completed (int): number of seed columns completed before the cutoff.
    - seed_flights_processed (int): number of flights the seeding loop reached.
    - master (RestrictedMaster | None): partial master, read for its backend name and column
      count; ``None`` reports backend ``"none"``.
    - time_to_master_s (float): seconds until the master was (nearly) ready, echoed into stats.
    - seedless_flight_ids (Sequence[int]): flights with no seed column, recorded sorted.

    Return
    --------
    - output (ColGenResult): a result with empty ``columns`` and a stats dict flagging a
      ``time_limit`` termination with every flight search-exhausted.
    """

    elapsed_s = time.monotonic() - started
    denied = tuple(flight_ids)
    graph_infeasible = frozenset(graph_infeasible_flight_ids)
    heuristic_cost = len(denied) * params.M
    graph_values = tuple((graphs or {}).values())
    arc_stats: Counter[str] = Counter()
    for graph in graph_values:
        arc_stats.update(graph.arc_cache_stats)
    wall_stats = {} if catalog is None else catalog.wall_index.stats
    backend_name = "none" if master is None else _backend_name(master)
    return ColGenResult(
        columns={},
        stats={
            "backend": backend_name,
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
            "iteration_ip_calls": 0,
            "iteration_ip_wall_s": 0.0,
            "iteration_ip_records": (),
            "final_lp_objective": -math.inf,
            "upper_bound": math.inf,
            "cost_upper_bound": math.inf,
            "cost_lower_bound": -math.inf,
            "lp_gap": math.inf,
            "heuristic_objective": 0.0,
            "heuristic_cost": heuristic_cost,
            # The full path's gap and timing keys, at the values a pre-master timeout
            # actually implies.  `batch.run_batch` reads five of these by name; a missing
            # one logged as "unknown" on exactly the run whose cost you most want to see.
            "gap_metric": params.gap_metric,
            "lp_gap_revenue": math.inf,
            "lp_gap_cost": math.inf,
            "heuristic_gap_revenue": math.inf,
            "heuristic_gap_cost": math.inf,
            **_integer_gap_stats(math.inf, 0.0, heuristic_cost, params),
            "pricing_wall_s": 0.0,
            "pricing_task_total_s": 0.0,
            "pricing_pool_setup_s": 0.0,
            "n_pricing_workers": 0,
            "kernel_priced": 0,
            "kernel_fell_back": 0,
            "pricing_worker_lost": 0,
            "kernel_label_restarts": 0,
            "kernel_budget_declined": 0,
            "kernel_declined_by_reason": {},
            "seeded_columns": 0,
            "ladder_columns": 0,
            **_dual_regime_stats(backend_name),
            "ip_elapsed_s": 0.0,
            "ip_time_limit_s": params.ip_time_limit_s,
            "ip_reserve_s": params.effective_ip_reserve_s,
            "ip_eager_rows": 0,
            "ip_separation_rounds": 0,
            "ip_setup_s": 0.0,
            "ip_trajectory": (),
            "ip_objective": None,
            "ip_upper_bound": None,
            "ip_cost_lower_bound": None,
            "ip_cost_upper_bound": None,
            "restricted_ip_gap": None,
            "ip_status": "time_limit_skipped",
            "ip_optimal": None,
            "ip_skipped": True,
            "objective": 0.0,
            "objective_name": params.objective,
            "master_objective": 0.0,
            "selected_flights": 0,
            "denied_flight_ids": denied,
            "budget_denied_flight_ids": tuple(f for f in denied if f in graph_infeasible),
            "search_exhausted_flight_ids": tuple(f for f in denied if f not in graph_infeasible),
            "repair_added": 0,
            "seedless_flight_ids": tuple(sorted(set(seedless_flight_ids) | graph_infeasible)),
            "warm_start_planner": params.warm_start_planner,
            "initial_heuristic_strategy": "time_limit",
            "initial_heuristic_flights": 0,
            "initial_heuristic_delay_s": 0.0,
            "initial_greedy_completed": False,
            "initial_greedy_flights": 0,
            "initial_greedy_elapsed_s": 0.0,
            "initial_seed_columns": seeds_completed,
            "graphs_built": len(graph_values),
            "seeds_completed": seeds_completed,
            "seed_flights_processed": seed_flights_processed,
            "preprocessing_stage": stage,
            "graph_build_elapsed_s": graph_build_elapsed_s,
            "seed_elapsed_s": seed_elapsed_s,
            "time_to_master_s": time_to_master_s,
            "static_terminal_count": 0 if catalog is None else len(catalog.entries),
            "static_excluded_cells": 0 if catalog is None else catalog.excluded_cell_count,
            "corridor_domains_materialized": sum(
                bool(getattr(graph.corridor_cells, "is_materialized", True))
                for graph in graph_values
            ),
            "arc_expanded_nodes": arc_stats["expanded_nodes"],
            "arc_checks": arc_stats["arc_checks"],
            "arc_cache_hits": arc_stats["cache_hits"],
            "arc_allowed": arc_stats["allowed_arcs"],
            "arc_blocked": arc_stats["blocked_arcs"],
            "wall_index_queries": int(wall_stats.get("queries", 0)),
            "wall_index_candidates": int(wall_stats.get("candidates", 0)),
            "pricing_flights_completed": 0,
            "pricing_certified_columns": 0,
            "pricing_revalidated_columns": 0,
            "pricing_sweeps_completed": 0,
            "cheap_pricing_sweeps": 0,
            "exact_pricing_sweeps": 0,
            "cheap_pricing_wall_s": 0.0,
            "exact_pricing_wall_s": 0.0,
            "pricing_timeout_flight_id": None,
            "n_columns": 0 if master is None else len(master.columns),
            "n_materialized_rows": 0,
            "lazy_rows_added": 0,
            "lazy_row_rounds": 0,
            "time_limit_overrun_s": max(0.0, elapsed_s - params.time_limit_s),
            "elapsed_s": elapsed_s,
        },
    )


class ColGenSolver:
    """Whole-schedule column generation solver orchestrating restricted master LP and pricing DAGs."""

    def solve(
        self,
        requests: Sequence[FlightRequest],
        cfg: SimConfig,
        static_terms,
        params: ColGenParams,
        *,
        fixed_claims: Sequence[frozenset[RowKey]] = (),
        on_iteration=None,
        seed_columns: Mapping[int, Sequence[Column]] | None = None,
    ) -> ColGenResult:
        """Solve whole-schedule flight scheduling and deconfliction via column generation.

        The loop shape (warm pool → master LP → price against duals → add columns → repeat,
        then a final restricted-master IP) is in context/figures/cg_loop.png.

        Pricing is a sweep over the flights: each subproblem is independent given the
        iteration's duals, so the loop order affects only which columns a timed-out sweep
        managed to reach, never their value.  A sweep that FINISHES is therefore
        answer-identical whether run in-process or across worker processes; a sweep that
        hits ``pricing_deadline`` is not, because the deadline is a wall clock and a pool
        gets further through ``pricing_order`` before it, keeping a longer accepted prefix
        (more pricing inside the same budget, not the same answer -- see
        :mod:`.pricing_pool`).  Parallelism is configured only through
        ``params.n_pricing_workers``; there is no separate ``parallel=`` argument.

        Parameters
        ------------
        - requests (Sequence[FlightRequest]): the flights to schedule together; flight ids
          must be unique.
        - cfg (SimConfig): geometry, timestep, and cost configuration for the whole solve.
        - static_terms: static terminal/wall catalogue (often a generator owned by the
          simulation); snapshotted once so every flight graph sees the same walls.
        - params (ColGenParams): network and solver controls (backend, gaps, time limits,
          worker count, ladder/bootstrap depth).
        - fixed_claims (Sequence[frozenset[RowKey]]): capacity claims already committed by
          flights outside this batch (the rolling-horizon seam); their load is reserved
          before this batch is placed.
        - on_iteration: optional callback invoked once per iteration with a dict of that
          iteration's master state (LP objective, global upper bound, gaps, column and
          dual diagnostics).  Without it the per-iteration bound and gap surface only in
          the final stats, so a killed or watched run discards them.
        - seed_columns (Mapping[int, Sequence[Column]] | None): optional warm-start columns
          per flight.  ORDER is load-bearing: element 0 is that flight's entry in the
          candidate incumbent, elements 1.. are pool contents only.  Every column is
          re-canonicalised through the same claim gate, so a warm start cannot introduce a
          trajectory pricing could not have produced.

        Return
        --------
        - output (ColGenResult): the selected column per placed flight plus a stats dict
          (termination reason, bounds, gaps, IP diagnostics, and the denial partition).

        ``lp_columns`` and read-only ``lp_x`` are aligned snapshots; new columns
        added after that LP have zero values, with ``lp_column_count`` identifying the
        original LP prefix. ``master`` remains live and may grow after the callback.
        """
        started = time.monotonic()
        pricing_wall_s = 0.0
        pricing_task_total_s = 0.0
        pricing_pool_setup_s = 0.0
        # Every compiled-pricing counter each sweep reported, summed: `priced`, `fell_back`,
        # the label-pool `label_restarts` / `budget_declined` pair, and one
        # `declined_<reason>` key per `pricing.Declined` cause that fired.
        kernel_counters: Counter[str] = Counter()
        deadline = started + params.time_limit_s
        # Hold back the configured final-IP stage budget. An incomplete
        # pricing sweep cannot certify a global bound, but every completed
        # column is still valid and can improve that final incumbent.
        ip_reserve_s = params.effective_ip_reserve_s
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
                    "iteration_ip_calls": 0,
                    "iteration_ip_wall_s": 0.0,
                    "iteration_ip_records": (),
                    "final_lp_objective": 0.0,
                    "upper_bound": 0.0,
                    "cost_upper_bound": 0.0,
                    "cost_lower_bound": 0.0,
                    "lp_gap": 0.0,
                    "heuristic_objective": 0.0,
                    "heuristic_cost": 0.0,
                    "gap_metric": params.gap_metric,
                    "lp_gap_revenue": 0.0,
                    "lp_gap_cost": 0.0,
                    "heuristic_gap_revenue": 0.0,
                    "heuristic_gap_cost": 0.0,
                    **_integer_gap_stats(0.0, 0.0, 0.0, params),
                    "pricing_wall_s": 0.0,
                    "pricing_task_total_s": 0.0,
                    "pricing_pool_setup_s": 0.0,
                    "n_pricing_workers": 0,
                    "kernel_priced": 0,
                    "kernel_fell_back": 0,
                    "pricing_worker_lost": 0,
                    "kernel_label_restarts": 0,
                    "kernel_budget_declined": 0,
                    "kernel_declined_by_reason": {},
                    "seeded_columns": 0,
                    "ladder_columns": 0,
                    # "none": this exit precedes any master, so no LP ran to have duals.
                    **_dual_regime_stats("none"),
                    "ip_elapsed_s": 0.0,
                    "ip_time_limit_s": params.ip_time_limit_s,
                    "ip_reserve_s": params.effective_ip_reserve_s,
                    "ip_eager_rows": 0,
                    "ip_separation_rounds": 0,
                    "ip_setup_s": 0.0,
                    "ip_trajectory": (),
                    "ip_objective": None,
                    "ip_upper_bound": None,
                    "ip_cost_lower_bound": None,
                    "ip_cost_upper_bound": None,
                    "restricted_ip_gap": None,
                    "ip_status": "skipped",
                    "ip_optimal": None,
                    "ip_skipped": True,
                    "objective": 0.0,
                    "objective_name": params.objective,
                    "master_objective": 0.0,
                    "selected_flights": 0,
                    "denied_flight_ids": (),
                    "repair_added": 0,
                    "seedless_flight_ids": (),
                    "budget_denied_flight_ids": (),
                    "search_exhausted_flight_ids": (),
                    "warm_start_planner": params.warm_start_planner,
                    "initial_heuristic_strategy": "empty",
                    "initial_heuristic_flights": 0,
                    "initial_heuristic_delay_s": 0.0,
                    "initial_greedy_completed": True,
                    "initial_greedy_flights": 0,
                    "initial_greedy_elapsed_s": 0.0,
                    "initial_seed_columns": 0,
                    "graphs_built": 0,
                    "seeds_completed": 0,
                    "seed_flights_processed": 0,
                    "preprocessing_stage": "empty",
                    "graph_build_elapsed_s": 0.0,
                    "seed_elapsed_s": 0.0,
                    "time_to_master_s": 0.0,
                    "static_terminal_count": 0,
                    "static_excluded_cells": 0,
                    "corridor_domains_materialized": 0,
                    "arc_expanded_nodes": 0,
                    "arc_checks": 0,
                    "arc_cache_hits": 0,
                    "arc_allowed": 0,
                    "arc_blocked": 0,
                    "wall_index_queries": 0,
                    "wall_index_candidates": 0,
                    "pricing_flights_completed": 0,
                    "pricing_certified_columns": 0,
                    "pricing_revalidated_columns": 0,
                    "pricing_sweeps_completed": 0,
                    "cheap_pricing_sweeps": 0,
                    "exact_pricing_sweeps": 0,
                    "cheap_pricing_wall_s": 0.0,
                    "exact_pricing_wall_s": 0.0,
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
        if not params.seed_nominal_routes and not any((seed_columns or {}).values()):
            raise ValueError("seed_nominal_routes=False requires non-empty seed_columns")
        graph_build_started = time.monotonic()
        static_catalog = StaticTerminalCatalog(static_terms, cfg)
        static_term_snapshot = static_catalog.entries
        graphs: dict[int, FlightGraph] = {}
        graph_infeasible: set[int] = set()
        for request in ordered_requests:
            if time.monotonic() >= pricing_deadline:
                return _pre_master_timeout_result(
                    flight_ids,
                    started,
                    params,
                    stage="graph_build",
                    graphs=graphs,
                    catalog=static_catalog,
                    graph_build_elapsed_s=time.monotonic() - graph_build_started,
                    graph_infeasible_flight_ids=tuple(graph_infeasible),
                )
            try:
                graphs[request.flight_id] = build_flight_graph(
                    request, cfg, static_catalog, params,
                )
            except FlightGraphInfeasible as exc:
                graph_infeasible.add(request.flight_id)
                log.info("colgen flight %s has no usable graph: %s", request.flight_id, exc)
        graph_build_elapsed_s = time.monotonic() - graph_build_started
        if time.monotonic() >= pricing_deadline:
            return _pre_master_timeout_result(
                flight_ids,
                started,
                params,
                stage="graph_build",
                graphs=graphs,
                catalog=static_catalog,
                graph_build_elapsed_s=graph_build_elapsed_s,
                graph_infeasible_flight_ids=tuple(graph_infeasible),
            )

        # One objective for the whole solve, threaded into seeding, the greedy
        # heuristic and pricing.  See colgen.objective.
        model = cost_model(cfg, params)

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
        time_to_master_s = time.monotonic() - started
        seedless_flights: set[int] = set(graph_infeasible)
        seeds: dict[int, Column] = {}
        ladder_columns = 0
        seed_started = time.monotonic()
        for flight_id in (graphs if params.seed_nominal_routes else ()):
            try:
                seed = seed_column(
                    graphs[flight_id], cfg, deadline=pricing_deadline, model=model
                )
            except PricingTimeout:
                return _pre_master_timeout_result(
                    flight_ids,
                    started,
                    params,
                    stage="seeding",
                    graphs=graphs,
                    catalog=static_catalog,
                    graph_build_elapsed_s=graph_build_elapsed_s,
                    seed_elapsed_s=time.monotonic() - seed_started,
                    seeds_completed=len(seeds),
                    seed_flights_processed=len(seeds) + len(seedless_flights),
                    master=master,
                    time_to_master_s=time_to_master_s,
                    seedless_flight_ids=tuple(seedless_flights),
                    graph_infeasible_flight_ids=tuple(graph_infeasible),
                )
            except ValueError:
                # A disconnected/static-blocked graph is a legitimate
                # optimizer-verdict denial, not a batch-wide solver failure.
                # Its <=1 flight row can remain empty while other flights run.
                seedless_flights.add(flight_id)
                continue
            seed = _canonical_column(seed, graphs[flight_id], cfg)
            seeds[flight_id] = seed
            master.add_column(seed)
            ladder_columns += _add_departure_ladder(
                master, seed, graphs[flight_id], cfg, model,
                params.seed_ladder_steps, params.seed_ladder_stride,
            )
        seed_elapsed_s = time.monotonic() - seed_started

        shifted_seed_heuristic = _initial_feasible_selection(
            seeds,
            graphs,
            committed_loads,
            row_index,
            cfg,
            deadline=pricing_deadline,
            model=model,
        ) if params.seed_nominal_routes else {}
        # Keep initialization deliberately small: one certified shortest seed
        # per flight plus at most one time-shifted seed selected by the cheap
        # claim-feasible pass above.  Route alternatives belong to reduced-cost
        # pricing after the first LP, not to an eager all-flight prepass.
        greedy_heuristic: dict[int, Column] = {}
        greedy_completed = False
        greedy_elapsed_s = 0.0
        initial_heuristic = dict(shifted_seed_heuristic)
        initial_heuristic_strategy = (
            "shifted_seeds" if params.seed_nominal_routes else "provided_only"
        )
        best_heuristic = dict(initial_heuristic)
        for column in initial_heuristic.values():
            master.add_column(column)
        # Optional warm start.  The policy above is a deliberate bet -- that route
        # alternatives are cheaper to discover by reduced-cost pricing than to enumerate
        # up front -- and `seed_columns` is how that bet gets tested rather than assumed.
        # Every column still goes through the same canonical claim gate, so it cannot
        # introduce a trajectory pricing could not have produced.
        #
        # The ORDER of each flight's sequence is load-bearing: element 0 is the flight's
        # entry in the candidate INCUMBENT assembled below, and elements 1.. are pool
        # contents alone.  `warm_start.build` relies on that -- it returns the repaired,
        # mutually row-feasible column first and its departure-shifted alternatives after --
        # so a caller that re-orders a sequence silently changes which schedule is offered
        # as the incumbent, not just which columns exist.  A caller with no incumbent to
        # propose can pass any order it likes; the feasibility and improvement guards below
        # reject a candidate that is not jointly claim-feasible or not better than the
        # heuristic.
        seeded_columns = 0
        seeded_first: dict[int, Column] = {}
        for flight_id, extras in (seed_columns or {}).items():
            if flight_id not in flight_ids:
                raise KeyError(f"seed_columns names flight {flight_id}, which is not in this batch")
            if flight_id in graph_infeasible:
                continue
            for position, column in enumerate(extras):
                canonical = _canonical_column(column, graphs[flight_id], cfg, model=model)
                master.add_column(canonical)
                seeded_columns += 1
                if position == 0:
                    seeded_first[flight_id] = canonical
                    half_width = params.provided_seed_ladder_steps
                    if half_width:
                        graph = graphs[flight_id]
                        _add_departure_ladder(master, canonical, graph, cfg, model, half_width)
                        # Earlier variants cannot predate the requested lattice
                        # departure. Pure clock shifts preserve certified geometry.
                        before = min(half_width, canonical.departure_step - graph.base_step)
                        for steps in range(1, before + 1):
                            master.add_column(_canonical_column(replace(
                                canonical,
                                departure_step=canonical.departure_step - steps,
                                delay_s=canonical.delay_s - model.ground_weight * steps * cfg.dt_s,
                                claims=_shift_claims(canonical.claims, -steps),
                            ), graph, cfg))
        # The seed columns are also a candidate INCUMBENT, not only pool contents.  Adding
        # them to the pool alone leaves them reachable exclusively through the final IP --
        # and when that IP is truncated the run returns the shifted-seed heuristic instead,
        # so a caller who supplied a whole better schedule gets none of it.  Taking it as
        # the incumbent makes the truncated-IP fallback the BETTER of the two schedules.
        #
        # Guarded, not assumed: the seeds are only an incumbent if they are jointly claim
        # feasible (a caller's schedule need not be, and a warm start must never smuggle in
        # an infeasible selection) and only if they actually beat the heuristic.
        #
        # The flights the seeds do NOT name are re-picked around them rather than left on
        # their shifted-seed columns.  Overlaying the seeds on the heuristic is the obvious
        # reading and it does not work: the few routes the graph cannot express keep
        # heuristic columns that clash with the seeded ones, `is_claim_feasible` fails on
        # the overlay, and the whole warm start is thrown away.  `complete_selection` pins
        # the seeds and fills the rest greedily, so a leftover flight can only cost what
        # its own best compatible column costs.
        if seeded_first:
            seeds_feasible = master.is_claim_feasible(seeded_first)
            candidate = (
                (master.complete_selection(seeded_first) if params.seed_nominal_routes
                 else dict(seeded_first))
                if seeds_feasible else {}
            )
            # WHY a warm start was refused is not recoverable after the fact: a candidate
            # missing k flights loses on `k*M` while a clashing one never gets built at
            # all, and both surface only as `initial_heuristic_strategy == "shifted_seeds"`.
            # The two point at completely different fixes, so the run has to log which one
            # happened.
            log.info(
                "  warm start: %d seeds, pins %s, completed to %d/%d flights, "
                "objective %.1f vs heuristic %.1f",
                len(seeded_first),
                "feasible" if seeds_feasible else "INFEASIBLE",
                len(candidate),
                len(flight_ids),
                _selection_objective(candidate, params.M) if candidate else -math.inf,
                _selection_objective(best_heuristic, params.M),
            )
            if candidate and _selection_objective(candidate, params.M) > _selection_objective(
                best_heuristic, params.M
            ):
                # BOTH, and they have to move together.  `initial_heuristic` is what
                # `initial_heuristic_flights` / `initial_heuristic_delay_s` are computed
                # from, while `initial_heuristic_strategy` is set here -- so updating only
                # the strategy leaves a run reporting "seed_columns" beside the flight count
                # and delay of the shifted-seed selection it just replaced, which is a
                # sentence about two different schedules.
                initial_heuristic = candidate
                best_heuristic = candidate
                initial_heuristic_strategy = "seed_columns"
        if best_heuristic:
            master.set_heuristic(best_heuristic)

        rng = np.random.default_rng(cfg.seed)
        total_benefit = len(flight_ids) * params.M
        lp_objectives: list[float] = []
        upper_bounds: list[float] = []
        cost_upper_bounds: list[float] = []
        cost_lower_bounds: list[float] = []
        lp_gaps: list[float] = []
        # Previous iteration's capacity duals, kept only to measure dual movement.
        # Tailing-off with a frozen bound has two very different causes -- duals
        # converging slowly, or duals oscillating -- and they call for opposite fixes
        # (more iterations vs. stabilization).  ||pi_k - pi_{k-1}|| separates them.
        prev_capacity_duals: dict | None = None
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
        pricing_certified_columns = pricing_revalidated_columns = 0
        pricing_sweeps_completed = 0
        cheap_pricing_sweeps = exact_pricing_sweeps = 0
        cheap_pricing_wall_s = exact_pricing_wall_s = 0.0
        pricing_timeout_flight_id: int | None = None
        iteration_ip_records: list[dict[str, Any]] = []

        # ONE pool for the whole solve, not one per sweep.  Constructed here but not
        # STARTED: `spawn` happens on the first sweep, so no worker process exists during
        # the parent-only greedy stage and peak RSS is where it was.  See `PricingPool`
        # for what a solve-scoped pool buys and why each flight is pinned to a worker.
        pricing_requests = tuple(r for r in ordered_requests if r.flight_id in graphs)
        sweep_pool = (
            None
            if params.n_pricing_workers == 0 or not graphs
            else PricingPool(pricing_requests, cfg, params, static_catalog)
        )
        try:
            for iteration in range(params.max_iterations):
                # One line per iteration BEFORE any of its work, so a run that wedges says
                # which iteration it wedged in.  The stage lines below say where inside it.
                log.info(
                    "colgen iteration %d/%d: %d columns in the master",
                    iteration + 1, params.max_iterations, len(master.columns),
                )
                # Wall for THIS iteration, so the stage timers can be checked against the
                # clock rather than trusted.  The residual `iteration_wall_s - sweep_s -
                # sum(stage_s)` is the block nothing names, and on a 4,636-flight run that
                # residual was 146 s of a 192 s serial tail -- the largest single term in
                # the solve, invisible because only the parts someone thought to time were
                # reported.  A total cannot expose it; only a per-iteration wall can.
                iteration_started = time.monotonic()
                # Per-iteration stage timings.  The master block was one unattributed lump in
                # the serial tail, and "the LP is slow" is only one of four candidates in it:
                # the LP itself, the lazy-row re-solve loop around it, `_canonical_column`
                # (run once per priced column, in the parent), and `add_column`.
                stage_s: dict[str, float] = collections.defaultdict(float)
                stage_n: dict[str, int] = collections.defaultdict(int)

                def _timed(key, fn, *a, **kw):
                    t0 = time.perf_counter()
                    try:
                        return fn(*a, **kw)
                    finally:
                        stage_s[key] += time.perf_counter() - t0
                        stage_n[key] += 1

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
                        lp_objective, capacity_duals, x = _timed("solve_lp", master.solve_lp)
                    except BackendTimeout:
                        break
                    added_rows = _timed(
                        "add_violated_rows", master.add_violated_rows, x, _FEASIBILITY_TOL
                    )
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
                columns_at_lp = len(master.columns)

                # LNS needs something to start from; `best_heuristic` is the greedy seed on
                # iteration 1 and a real incumbent after, so it is never empty here -- but
                # guard anyway, because an empty incumbent would silently return {} and
                # look like a heuristic that found nothing.
                if params.iteration_ip_time_limit_s > 0.0:
                    # The IP below sees this sweep's new columns. Keep the previous
                    # feasible incumbent available to pricing until that call finishes.
                    heuristic = dict(best_heuristic)
                elif params.lns_destroy_flights and best_heuristic:
                    heuristic = _timed(
                        "lns_heuristic", master.lns_heuristic, last_x, rng,
                        best_heuristic, params.lns_destroy_flights, params.n_heuristic_tries,
                        deadline=pricing_deadline,
                        lp_objective=last_lp_objective,
                    )
                else:
                    heuristic = _timed(
                        "round_heuristic", master.round_heuristic, last_x, rng,
                        params.n_heuristic_tries,
                    )
                heuristic = _timed(
                    "canonical_heuristic",
                    lambda h: {
                        flight_id: _canonical_column(column, graphs[flight_id], cfg)
                        for flight_id, column in h.items()
                    },
                    heuristic,
                )
                _assert_claim_feasible(heuristic, committed_loads, row_index)
                if not best_heuristic or _better_selection(heuristic, best_heuristic, params.M):
                    best_heuristic = dict(heuristic)
                    master.set_heuristic(best_heuristic)

                if iteration == 0 and params.seed_nominal_routes:
                    # The first LP has now established the real column-generation
                    # cycle.  Build at most one route-aware incumbent column per
                    # flight using the same lazy topology, bounded independently of
                    # the pricing loop.
                    #
                    # The budget is per FLIGHT and clamped only by the absolute
                    # `pricing_deadline`.  The shipped rate is 0, which disables the stage
                    # outright (see `greedy_budget_s_per_flight` for why), so none of this
                    # arithmetic runs by default.  An ENABLED rate's cost scales with batch
                    # size, so a very large batch wants its rate chosen rather than copied.
                    greedy_started = time.monotonic()
                    # PER FLIGHT, because the stage divides its budget across candidates and a
                    # fixed total therefore starves as the batch grows -- each search then gets
                    # far less than one density search needs.  A fixed total also breaks
                    # reproducibility above a few hundred flights: when nearly every candidate
                    # is decided by a stopwatch, machine load picks the winners.
                    greedy_budget_s = params.greedy_budget_s_per_flight * len(flight_ids)
                    greedy_deadline = min(pricing_deadline, greedy_started + greedy_budget_s)
                    greedy_heuristic, greedy_completed = _greedy_feasible_selection(
                        graphs,
                        committed_loads,
                        row_index,
                        cfg,
                        params,
                        deadline=greedy_deadline,
                        initial=best_heuristic,
                    )
                    greedy_elapsed_s = time.monotonic() - greedy_started
                    for column in greedy_heuristic.values():
                        _timed("add_column", master.add_column, column)
                    if greedy_heuristic and _better_selection(
                        greedy_heuristic,
                        best_heuristic,
                        params.M,
                    ):
                        best_heuristic = dict(greedy_heuristic)
                        master.set_heuristic(best_heuristic)

                best_reduced_costs: list[float] = []
                rc_by_flight: dict[int, float] = {}
                priced_columns: list[Column] = []
                # The LP and the heuristic are done; pricing is the long block that follows.
                # Naming the handoff is what turns "it is still going" into "it is in the
                # part that takes minutes", which is the whole question while watching a run.
                log.info(
                    "  master LP + heuristic done in %.1fs (%d rows materialized); pricing next",
                    sum(stage_s.values()),
                    len(getattr(master, "materialized_rows", ()) or ()),
                )
                pricing_order = sorted(
                    graphs,
                    key=lambda flight_id: (
                        flight_id in best_heuristic,
                        -best_heuristic[flight_id].delay_s
                        if flight_id in best_heuristic
                        else -math.inf,
                        flight_id,
                    ),
                )
                pricing_complete = True
                dual_view = DualView(capacity_duals, cfg)
                flight_duals = master.flight_duals
                # Per-iteration sweep cost, which is the whole cost of a solve.  A total
                # cannot answer "is pricing getting cheaper as the cutoffs tighten, or is
                # iteration 80 as expensive as iteration 1" -- and that question decides
                # whether more iterations are affordable.
                pricing_exact = (
                    not params.cheap_pricing
                    or iterations % params.exact_pricing_interval == 0
                    or iterations == params.max_iterations
                )
                iteration_sweep_s = cheap_attempt_s = 0.0
                cheap_columns_added = 0
                iteration_task_total_s = 0.0
                iteration_flight_records = []
                while True:
                    sweep_started = time.perf_counter()
                    sweep = price_sweep(
                        pricing_order,
                        pricing_requests,
                        graphs,
                        cfg,
                        params,
                        static_catalog,
                        capacity_duals,
                        dual_view,
                        flight_duals,
                        best_heuristic,
                        deadline=pricing_deadline,
                        pool=sweep_pool,
                        heuristic_only=not pricing_exact,
                    )
                    # Consumed in index order, which is what keeps a parallel sweep's answer equal
                    # to the sequential one: `master.upper_bound` sums these with plain `sum`, and
                    # float addition is not associative.  `SweepResult` has already discarded
                    # everything at or past the first timeout, reproducing the loop's `break`.
                    certified_ids = {
                        id(column) for column in _timed(
                            "pricing_certificate_check", certified_sweep_columns,
                            sweep, graphs, cfg, params, static_catalog,
                        )
                    }
                    for flight_id, reduced_cost, column in zip(
                        sweep.flight_ids, sweep.reduced_costs, sweep.columns, strict=True
                    ):
                        pricing_flights_completed += 1
                        best_reduced_costs.append(reduced_cost)
                        rc_by_flight[flight_id] = reduced_cost
                        if column is not None and reduced_cost > _REDUCED_COST_TOL:
                            if id(column) in certified_ids:
                                pricing_certified_columns += 1
                                priced_columns.append(column)
                            else:
                                pricing_revalidated_columns += 1
                                priced_columns.append(_timed(
                                    "canonical_priced", _canonical_column,
                                    column, graphs[flight_id], cfg, model=model,
                                ))
                    # Extra candidates improve the pool without adding another bound term
                    # for the same flight. Pricing certifies each against the same duals.
                    for column in sweep.extra_columns:
                        if id(column) in certified_ids:
                            pricing_certified_columns += 1
                            priced_columns.append(column)
                        else:
                            pricing_revalidated_columns += 1
                            priced_columns.append(_timed(
                                "canonical_priced", _canonical_column,
                                column, graphs[column.flight_id], cfg, model=model,
                            ))
                    if not sweep.complete:
                        pricing_complete = False
                        pricing_timeout_flight_id = sweep.timeout_flight_id

                    attempt_s = time.perf_counter() - sweep_started
                    iteration_sweep_s += attempt_s
                    pricing_wall_s += attempt_s
                    if pricing_exact:
                        exact_pricing_wall_s += attempt_s
                    else:
                        cheap_pricing_wall_s += attempt_s
                        cheap_attempt_s += attempt_s
                    # Production telemetry, summed across the solve.  A compiled-path fallback is
                    # the one regression nothing else can see: same column, same objective, 3-4.5x
                    # the time.  `sweep_task_total_s` against `pricing_wall_s * n_workers` is the
                    # pool's occupancy, which is the difference between "parallelism is not paying"
                    # and "parallelism is paying and the machine is saturated".
                    iteration_task_total_s += sweep.task_total_s
                    iteration_flight_records.extend(
                        dict(record, pricing_exact=pricing_exact) for record in sweep.flight_records
                    )
                    pricing_task_total_s += sweep.task_total_s
                    pricing_pool_setup_s += sweep.pool_setup_s
                    kernel_counters.update(sweep.kernel_counters)

                    log.info(
                        "  pricing returned %d columns in %.1fs (%d with positive reduced "
                        "cost); back to the master LP + heuristic",
                        len(priced_columns), attempt_s,
                        sum(1 for rc in sweep.reduced_costs if rc > 0.0),
                    )
                    before_pricing = len(master.columns)
                    for column in sorted(
                        priced_columns, key=lambda item: (item.flight_id, _column_key(item))
                    ):
                        _timed("add_column", master.add_column, column)
                    if not pricing_complete:
                        # A LOST WORKER and an expired clock both end the sweep the same way,
                        # and reporting them the same way would send a reader to the wrong
                        # knob: one means "raise the time limit", the other means "a worker
                        # died, probably to the OOM killer -- lower `n_pricing_workers`".
                        # Same reason `ip_not_proven` exists rather than reusing `time_limit`.
                        termination_reason = (
                            "pricing_worker_lost"
                            if sweep.kernel_counters.get("pool_worker_lost")
                            else "time_limit"
                        )
                        break
                    pricing_sweeps_completed += 1

                    if pricing_exact:
                        exact_pricing_sweeps += 1
                    else:
                        cheap_pricing_sweeps += 1
                        cheap_columns_added = len(master.columns) - before_pricing
                        if cheap_columns_added == 0:
                            # A restricted search cannot certify stagnation. Retry all
                            # flights with the SAME LP duals through the exact oracle.
                            pricing_exact = True
                            best_reduced_costs.clear()
                            rc_by_flight.clear()
                            priced_columns.clear()
                            log.info("  cheap pricing stalled; running a full sweep")
                            continue
                    break
                if not pricing_complete:
                    break

                iteration_ip_info = {}
                if params.iteration_ip_time_limit_s > 0.0:
                    incumbent_cost = total_benefit - _selection_objective(best_heuristic, params.M)
                    master.backend.ip_gap = (
                        params.ip_gap * max(incumbent_cost, 1.0) / max(1.0, total_benefit)
                    )
                    rows_before_ip = len(master.materialized_rows)
                    ip_round_started = time.monotonic()
                    ip_candidate = _timed(
                        "iteration_ip", master.solve_ip, best_heuristic,
                        deadline=pricing_deadline,
                        budget_s=params.iteration_ip_time_limit_s,
                        eager=params.iteration_ip_eager,
                        max_eager_rows=params.max_eager_ip_rows,
                    )
                    ip_candidate = _timed(
                        "canonical_iteration_ip",
                        lambda: {
                            fid: _canonical_column(column, graphs[fid], cfg)
                            for fid, column in ip_candidate.items()
                        },
                    )
                    _assert_claim_feasible(ip_candidate, committed_loads, row_index)
                    if _better_selection(ip_candidate, best_heuristic, params.M):
                        best_heuristic = dict(ip_candidate)
                    master.set_heuristic(best_heuristic)
                    iteration_ip_info = {
                        "iteration": iterations,
                        "budget_s": params.iteration_ip_time_limit_s,
                        "wall_s": time.monotonic() - ip_round_started,
                        "status": master.last_ip_status,
                        "optimal": master.last_ip_optimal,
                        "columns": len(master.columns),
                        "rows_added": len(master.materialized_rows) - rows_before_ip,
                        "separation_rounds": master.last_ip_rounds,
                        "covered": len(best_heuristic),
                        "cost": total_benefit - _selection_objective(best_heuristic, params.M),
                    }
                    iteration_ip_records.append(iteration_ip_info)
                    log.info("  iteration IP: %s, %d/%d flights, cost %.3f in %.2fs",
                             iteration_ip_info["status"], len(best_heuristic), len(flight_ids),
                             iteration_ip_info["cost"], iteration_ip_info["wall_s"])

                # Restricted-search scores underestimate the best reduced costs.
                # Only an exact sweep supplies a new valid global bound. Between
                # exact sweeps retain the previous certificate (infinity initially).
                raw_upper_bound = (
                    master.upper_bound(last_lp_objective, best_reduced_costs)
                    if pricing_exact else last_upper_bound
                )
                # Each iteration's value is a valid global upper bound.  Their
                # running minimum remains valid and removes harmless solver jitter.
                last_upper_bound = min(last_upper_bound, max(last_lp_objective, raw_upper_bound))
                # Transform maximize revenue back to the user objective before
                # measuring gaps.  Revenue-scale gaps are diluted by n*M and can
                # incorrectly certify a trajectory that is tens of seconds worse.
                cost_upper_bound = total_benefit - last_lp_objective
                cost_lower_bound = total_benefit - last_upper_bound
                heuristic_objective = _selection_objective(best_heuristic, params.M)
                heuristic_cost = total_benefit - heuristic_objective
                # Both scales are computed every iteration so neither is lost; only which one
                # the thresholds are applied to depends on params.gap_metric.
                lp_gap_cost = _relative_cost_gap(cost_upper_bound, cost_lower_bound)
                heuristic_gap_cost = _relative_cost_gap(heuristic_cost, cost_lower_bound)
                lp_gap_revenue = _relative_revenue_gap(last_upper_bound, last_lp_objective)
                heuristic_gap_revenue = _relative_revenue_gap(
                    last_upper_bound, heuristic_objective
                )
                if params.gap_metric == "revenue":
                    lp_gap, heuristic_gap = lp_gap_revenue, heuristic_gap_revenue
                else:
                    lp_gap, heuristic_gap = lp_gap_cost, heuristic_gap_cost

                lp_objectives.append(last_lp_objective)
                upper_bounds.append(last_upper_bound)
                cost_upper_bounds.append(cost_upper_bound)
                cost_lower_bounds.append(cost_lower_bound)
                lp_gaps.append(lp_gap)
                heuristic_objectives.append(heuristic_objective)
                heuristic_costs.append(heuristic_cost)

                if on_iteration is not None:
                    # Reduced-cost spread.  The bound is LP + sum(max(0, rc)), so whether the
                    # gap is a few pathological flights or all of them uniformly decides
                    # whether this is targetable or structural -- and the values are already
                    # computed here and otherwise discarded.
                    positives = sorted(v for v in best_reduced_costs if v > 0.0)
                    def _pct(frac: float) -> float:
                        if not positives:
                            return 0.0
                        return positives[min(len(positives) - 1, int(frac * len(positives)))]

                    # Dual movement over the union of both iterations' support.
                    dual_l2 = dual_linf = float("nan")
                    if prev_capacity_duals is not None:
                        keys = set(capacity_duals) | set(prev_capacity_duals)
                        diffs = [
                            capacity_duals.get(k, 0.0) - prev_capacity_duals.get(k, 0.0)
                            for k in keys
                        ]
                        dual_l2 = math.sqrt(math.fsum(d * d for d in diffs))
                        dual_linf = max((abs(d) for d in diffs), default=0.0)

                    callback_columns = master.columns
                    callback_x = np.pad(last_x, (0, len(callback_columns) - len(last_x)))
                    callback_x.setflags(write=False)
                    on_iteration({
                        "iteration": iterations,
                        "lp_objective": last_lp_objective,
                        "upper_bound": last_upper_bound,
                        "raw_upper_bound": raw_upper_bound,
                        "cost_upper_bound": cost_upper_bound,
                        "cost_lower_bound": cost_lower_bound,
                        "lp_gap": lp_gap,
                        "lp_gap_revenue": lp_gap_revenue,
                        "lp_gap_cost": lp_gap_cost,
                        "heuristic_gap_revenue": heuristic_gap_revenue,
                        "heuristic_gap_cost": heuristic_gap_cost,
                        "heuristic_cost": heuristic_cost,
                        "heuristic_gap": heuristic_gap,
                        # The live master, for analysis that has to ask it something the
                        # scalars above cannot answer -- solving the IP at THIS iteration
                        # being the motivating case, since the difference between the
                        # rounding heuristic and the true restricted IP is exactly what
                        # `heuristic_cost` alone cannot show.  Callers that mutate it own the
                        # consequences; the solver reads back only `last_ip_*`, which its own
                        # final solve overwrites.
                        "master": master,
                        # Immutable callback snapshot. Priced variables appended
                        # since the LP have value zero; only the first lp_column_count
                        # variables participated in that solve. The live master may
                        # grow again, so retained callbacks must use lp_columns.
                        "lp_x": callback_x,
                        "lp_columns": callback_columns,
                        "lp_column_count": columns_at_lp,
                        # This iteration's capacity duals, by row.  `dual_nonzero` counts them
                        # but cannot say WHICH rows are expensive, and that is the question
                        # behind "why do the new columns conflict": pricing steers every
                        # flight away from the same few hot rows at once, so their proposals
                        # collide on the alternatives.
                        "capacity_duals": capacity_duals,
                        # This iteration's pricing cost -- the wall the solver spent in the
                        # sweep, which is very nearly the wall of the whole iteration.
                        "sweep_s": iteration_sweep_s,
                        "pricing_exact": pricing_exact,
                        "cheap_sweep_s": cheap_attempt_s,
                        "cheap_columns_added": cheap_columns_added,
                        # The work the sweep actually did, against the wall above.  With a
                        # pool, `sweep_task_total_s / (sweep_s * n_workers)` is the fraction of
                        # worker-seconds spent computing; the rest is workers idle behind a
                        # straggler.  That distinguishes scheduling loss from dispatch overhead,
                        # and only the latter is something `chunksize` can address.
                        "sweep_task_total_s": iteration_task_total_s,
                        # Per-flight rows behind that sum, in `pricing_order` index order.  The
                        # sum says how much work the sweep did; only these say WHERE it went,
                        # and a pool's wall clock is set by its slowest single task rather than
                        # by the total.  Available under a pool as well as sequentially, which
                        # `prof_colgen_cutoff.py` cannot be: its instrumentation rebinds a
                        # module global in this process and a spawned worker imports its own.
                        "sweep_flight_records": tuple(iteration_flight_records),
                        "columns": len(master.columns),
                        "columns_added": len(priced_columns),
                        "rc_sum": math.fsum(positives),
                        "rc_n_positive": len(positives),
                        "rc_max": positives[-1] if positives else 0.0,
                        "rc_p50": _pct(0.5),
                        "rc_p90": _pct(0.9),
                        "dual_l2": dual_l2,
                        "dual_linf": dual_linf,
                        "dual_nonzero": sum(1 for v in capacity_duals.values() if v != 0.0),
                        "elapsed_s": time.monotonic() - started,
                        # This iteration alone, against which `sweep_s` and `stage_s` are a
                        # partial accounting.  Their residual is the unattributed block.
                        "iteration_wall_s": time.monotonic() - iteration_started,
                        **_coverage_diagnostics(master, callback_x, rc_by_flight, params.M),
                        "stage_s": dict(stage_s),
                        "stage_n": dict(stage_n),
                        # Per-try outcomes of this iteration's rounding heuristic, and how
                        # many columns the LP left integral enough to commit outright.  The
                        # stage has never once beaten the greedy seed at scale, and without
                        # these a reader cannot tell a try that stranded one flight (costing
                        # a full M) from one that was merely slower everywhere.
                        "round_stats": dict(getattr(master, "last_round_stats", {}) or {}),
                        "iteration_ip": dict(iteration_ip_info),
                        "lazy_rows_added": lazy_rows_added,
                        "lazy_row_rounds": lazy_row_rounds,
                    })
                prev_capacity_duals = dict(capacity_duals)

                new_columns_since_lp = len(master.columns) > columns_at_lp
                # The pricing bound remains valid when this sweep banks columns, and the
                # final IP already sees them, so an in-threshold gap terminates immediately.
                if pricing_exact and lp_gap <= params.lp_gap:
                    termination_reason = "lp_gap"
                    break
                if pricing_exact and best_heuristic and heuristic_gap <= params.ip_gap:
                    termination_reason = "heuristic_gap"
                    break
                if pricing_exact and not priced_columns and not new_columns_since_lp:
                    termination_reason = "no_improving_columns"
                    break

                if pricing_exact and len(master.columns) == before_pricing and not new_columns_since_lp:
                    termination_reason = "no_new_columns"
                    break
            else:
                termination_reason = "iteration_limit"
        finally:
            # Before the final IP, deliberately: each worker holds ~3 GB and the MILP is
            # this solve's other memory peak, so releasing them here rather than at
            # function exit keeps the two peaks from overlapping.  A `finally` because it
            # has to cover all three exits -- running out of iterations, every `break`
            # (including the time limit), and an exception on any of them.
            if sweep_pool is not None:
                sweep_pool.close()

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
        restricted_ip_gap: float | None = None
        ip_elapsed_s = 0.0
        ip_status = "skipped"
        ip_optimal: bool | None = None
        if not ip_skipped and time.monotonic() >= deadline:
            ip_skipped = True
            ip_status = "time_limit_skipped"
            termination_reason = "time_limit"
        elif not ip_skipped:
            # The IP gets its OWN budget, not "whatever is left".  Both bounds are real:
            # `deadline` keeps the solve inside the time limit it was given, and
            # `ip_time_limit_s` keeps a loop that converged early from handing the MILP
            # hours (see ColGenParams.ip_time_limit_s).  Whichever binds first wins.
            # `ip_time_limit_s` is handed to `solve_ip` as a BUDGET rather than folded into
            # a deadline here, because eager row materialization happens inside `solve_ip`
            # and is setup, not search: pre-shrinking the deadline would charge that setup
            # to the solver and starve the MILP.  The whole-solve `deadline` still binds
            # and is still the hard wall.
            master.backend.time_limit_s = max(1e-6, params.ip_time_limit_s)
            master.set_heuristic(incumbent)
            # Re-scale the MIP tolerance now that the incumbent's COST is known.
            # `create_backend` can only assume the worst case (cost -> 0) and so hands the
            # backend `ip_gap / (n*M)` -- an ABSOLUTE tolerance of `ip_gap` revenue units,
            # orders of magnitude tighter than the user-facing `ip_gap` reads as and a
            # measured cause of the final MILP blowing up.  Here the incumbent gives a real
            # cost scale, so the same user-facing tolerance can be expressed against it.
            # `max(..., 1.0)` keeps the conservative value when cost is genuinely tiny,
            # which is the case the original conversion was protecting.
            incumbent_cost = total_benefit - _selection_objective(incumbent, params.M)
            revenue_scale = max(1.0, total_benefit)
            master.backend.ip_gap = params.ip_gap * max(incumbent_cost, 1.0) / revenue_scale
            # Timed because it was the one unattributed block left in the solve, and "the
            # IP is slow" is a hypothesis that needs a number before anyone acts on it: a
            # converged pool can spend nearly all its wall in pricing and almost none here.
            ip_started = time.monotonic()
            ip_selection = master.solve_ip(
                deadline=deadline,
                budget_s=params.ip_time_limit_s,
                max_eager_rows=params.max_eager_ip_rows,
            )
            ip_elapsed_s = time.monotonic() - ip_started
            ip_selection = {
                flight_id: _canonical_column(column, graphs[flight_id], cfg)
                for flight_id, column in ip_selection.items()
            }
            _assert_claim_feasible(ip_selection, committed_loads, row_index)
            ip_objective = _selection_objective(ip_selection, params.M)
            ip_upper_bound = master.last_ip_bound
            ip_cost_upper_bound = total_benefit - ip_objective
            # This bound certifies only the generated column pool. Global gap
            # fields are calculated below from the final schedule after repair.
            if ip_upper_bound is not None:
                ip_cost_lower_bound = total_benefit - ip_upper_bound
                restricted_ip_gap = (
                    _relative_revenue_gap(ip_upper_bound, ip_objective)
                    if params.gap_metric == "revenue"
                    else _relative_cost_gap(ip_cost_upper_bound, ip_cost_lower_bound)
                )
            ip_status = master.last_ip_status or "unknown"
            ip_optimal = master.last_ip_optimal
            if ip_optimal is False and termination_reason not in {
                "time_limit", "iteration_limit"
            }:
                # The restricted IP could not PROVE its selection optimal over the pool it
                # was given.  That is a different fact from the CG loop running out of
                # budget: the loop may have converged on `lp_gap` with only the final MILP
                # left uncertified, and only the second is fixed by raising the time limit.
                # Both mean an absent flight is unproven rather than physically impossible,
                # so the denial partition below treats them alike -- but the reason a run
                # reports is read by a person, and "time_limit" sent them to the wrong knob.
                #
                # The IP can also fail to prove because it ran out of WALL CLOCK -- it gets
                # only the small `ip_reserve_s` tail -- and that really is a budget
                # exhaustion, whatever the loop did before it.  Distinguish the two here
                # rather than calling both uncertified, or a sweep filtering runs on
                # `planner_termination == "time_limit"` silently drops them.
                termination_reason = (
                    "time_limit" if time.monotonic() >= deadline else "ip_not_proven"
                )
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
            (graph for flight_id, graph in graphs.items() if flight_id not in incumbent),
            key=lambda graph: (graph.request.t_departure, graph.request.flight_id),
        )
        saturated = {
            row for row, load in loads.items() if load >= row_index.cap(row)
        }
        for repair_index, graph in enumerate(repair_order):
            if time.monotonic() >= deadline:
                termination_reason = "time_limit"
                search_exhausted_flights.update(
                    pending.request.flight_id for pending in repair_order[repair_index:]
                )
                break
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
            for row in repaired.claims:
                old_load = loads[row]
                new_load = old_load + 1
                loads[row] = new_load
                if old_load < row_index.cap(row) <= new_load:
                    saturated.add(row)
            repair_added += 1

        incumbent = dict(sorted(incumbent.items()))
        _assert_claim_feasible(incumbent, committed_loads, row_index)
        denied = tuple(flight_id for flight_id in flight_ids if flight_id not in incumbent)
        # Every truncated exit, by whatever mechanism: a flight the solve did not place is
        # then a compute-cap artifact, not the optimizer's verdict that no plan exists.
        if termination_reason in {
            "time_limit", "iteration_limit", "ip_not_proven", "pricing_worker_lost",
        }:
            search_exhausted_flights.update(set(denied) - graph_infeasible)
        search_exhausted = tuple(
            flight_id for flight_id in denied if flight_id in search_exhausted_flights
        )
        budget_denied = tuple(
            flight_id for flight_id in denied if flight_id not in search_exhausted_flights
        )
        # NOT seconds under every objective: `Column.delay_s` carries whatever currency the
        # cost model priced in, so at objective="total_cost" this is weighted cost units
        # (1*ground + 3*air).  Reported beside `objective_name` for that reason -- the run
        # folder's own delay metrics, computed from the filed intents, are the seconds.
        objective_value = math.fsum(column.delay_s for column in incumbent.values())
        master_objective = _selection_objective(incumbent, params.M)
        elapsed_s = time.monotonic() - started
        arc_stats: Counter[str] = Counter()
        for graph in graphs.values():
            arc_stats.update(graph.arc_cache_stats)
        corridor_domains_materialized = sum(
            bool(getattr(graph.corridor_cells, "is_materialized", True))
            for graph in graphs.values()
        )

        materialized_rows = getattr(master, "materialized_rows", ())
        # A LOST WORKER OUTRANKS EVERY LATER REASON, and it has to be re-asserted here
        # because four phases downstream of the generation loop overwrite
        # `termination_reason` on their own terms -- skipping the final IP, failing to
        # prove it, and two repair paths -- all of which report `time_limit` or
        # `ip_not_proven`. Every one of those tells a reader to raise a budget, which is
        # the wrong action: a worker that vanished without an exception is the OOM killer,
        # and the fix is FEWER workers. The generation loop's fault is also the earlier
        # fact, so it is the one that explains the rest.
        #
        # Derived from the counter rather than tracked in a second flag: `kernel_counters`
        # already accumulates it across sweeps, so there is no way for the two to disagree.
        worker_losses = kernel_counters.get("pool_worker_lost", 0)
        if worker_losses:
            termination_reason = "pricing_worker_lost"
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
            "iteration_ip_calls": len(iteration_ip_records),
            "iteration_ip_wall_s": sum(record["wall_s"] for record in iteration_ip_records),
            "iteration_ip_records": tuple(iteration_ip_records),
            "final_lp_objective": last_lp_objective,
            "upper_bound": last_upper_bound,
            "cost_upper_bound": total_benefit - last_lp_objective,
            "cost_lower_bound": final_cost_lower_bound,
            "lp_gap": (
                (
                    _relative_revenue_gap(last_upper_bound, last_lp_objective)
                    if params.gap_metric == "revenue"
                    else _relative_cost_gap(
                        total_benefit - last_lp_objective, final_cost_lower_bound
                    )
                )
                if math.isfinite(last_lp_objective)
                else math.inf
            ),
            "lp_gap_revenue": (
                _relative_revenue_gap(last_upper_bound, last_lp_objective)
                if math.isfinite(last_lp_objective)
                else math.inf
            ),
            "lp_gap_cost": (
                _relative_cost_gap(
                    total_benefit - last_lp_objective, final_cost_lower_bound
                )
                if math.isfinite(last_lp_objective)
                else math.inf
            ),
            # The heuristic gap on both scales, for the same reason the LP's is reported on
            # both: `gap_metric` decides which one gates, and a run that stopped on
            # `heuristic_gap` cannot otherwise be asked how converged it really was -- these
            # were computed every iteration and then discarded with the loop frame.
            # `heuristic_gap_cost` is also the quantity the `ip_skipped` decision uses.
            "heuristic_gap_revenue": _relative_revenue_gap(
                last_upper_bound, heuristic_objective
            ),
            "heuristic_gap_cost": _relative_cost_gap(
                heuristic_cost, final_cost_lower_bound
            ),
            **_integer_gap_stats(last_upper_bound, master_objective, total_benefit, params),
            "gap_metric": params.gap_metric,
            "heuristic_objective": heuristic_objective,
            "heuristic_cost": heuristic_cost,
            # Native IP diagnostics describe the final restricted master.
            # ``cost_lower_bound`` above remains the global pricing bound.
            "ip_objective": ip_objective,
            "ip_upper_bound": ip_upper_bound,
            "ip_cost_lower_bound": ip_cost_lower_bound,
            "ip_cost_upper_bound": ip_cost_upper_bound,
            "restricted_ip_gap": restricted_ip_gap,
            "ip_status": ip_status,
            "ip_optimal": ip_optimal,
            "ip_skipped": ip_skipped,
            "ip_elapsed_s": ip_elapsed_s,
            # Reported alongside the elapsed time so a reader can tell an IP that finished
            # from one the cap cut off, without inferring it from `ip_optimal` alone.
            "ip_time_limit_s": params.ip_time_limit_s,
            "ip_reserve_s": params.effective_ip_reserve_s,
            # Rows the bindability filter pre-materialized, and how many separation rounds
            # were still needed afterwards.  A nonzero round count after an eager solve is
            # the signal that some violable row was not predicted -- worth a look, since the
            # filter is supposed to be exact rather than approximate.
            "ip_eager_rows": 0 if ip_skipped else master.last_ip_eager_rows,
            "ip_separation_rounds": 0 if ip_skipped else master.last_ip_rounds,
            "ip_setup_s": 0.0 if ip_skipped else master.last_ip_setup_s,
            # (elapsed_s, incumbent, bound) per MILP incumbent, Gurobi only -- scipy's
            # `milp` exposes no callback, so this is empty on HiGHS.  Recorded because the
            # final MILP is otherwise a black box: a 25-minute solve and a hang look
            # identical, and "would a looser ip_gap have stopped it, and when" is
            # unanswerable without the trajectory.  Both values are in MAXIMIZE revenue
            # sense, matching `ip_upper_bound`.
            "ip_trajectory": (() if ip_skipped else
                              tuple(getattr(master, "last_ip_trajectory", ()) or ())),
            # ``objective`` is the user-facing minimization objective.  The
            # maximize-sense master value is retained under an explicit name.
            "objective": objective_value,
            "objective_name": params.objective,
            "master_objective": master_objective,
            "selected_flights": len(incumbent),
            "denied_flight_ids": denied,
            "budget_denied_flight_ids": budget_denied,
            "search_exhausted_flight_ids": search_exhausted,
            "repair_added": repair_added,
            "seedless_flight_ids": tuple(sorted(seedless_flights)),
            "warm_start_planner": params.warm_start_planner,
            "initial_heuristic_strategy": initial_heuristic_strategy,
            "initial_heuristic_flights": len(initial_heuristic),
            "initial_heuristic_delay_s": math.fsum(
                column.delay_s for column in initial_heuristic.values()
            ),
            "initial_greedy_completed": greedy_completed,
            "initial_greedy_flights": len(greedy_heuristic),
            "initial_greedy_elapsed_s": greedy_elapsed_s,
            "initial_seed_columns": len(seeds),
            "graphs_built": len(graphs),
            "seeds_completed": len(seeds),
            "seed_flights_processed": len(flight_ids),
            "preprocessing_stage": "complete",
            "graph_build_elapsed_s": graph_build_elapsed_s,
            "seed_elapsed_s": seed_elapsed_s,
            "time_to_master_s": time_to_master_s,
            "corridor_domains_materialized": corridor_domains_materialized,
            "static_terminal_count": len(static_catalog.entries),
            "static_excluded_cells": static_catalog.excluded_cell_count,
            "arc_expanded_nodes": arc_stats["expanded_nodes"],
            "arc_checks": arc_stats["arc_checks"],
            "arc_cache_hits": arc_stats["cache_hits"],
            "arc_allowed": arc_stats["allowed_arcs"],
            "arc_blocked": arc_stats["blocked_arcs"],
            "wall_index_queries": static_catalog.wall_index.stats["queries"],
            "wall_index_candidates": static_catalog.wall_index.stats["candidates"],
            "pricing_flights_completed": pricing_flights_completed,
            "pricing_certified_columns": pricing_certified_columns,
            "pricing_revalidated_columns": pricing_revalidated_columns,
            "pricing_sweeps_completed": pricing_sweeps_completed,
            "cheap_pricing_sweeps": cheap_pricing_sweeps,
            "exact_pricing_sweeps": exact_pricing_sweeps,
            "cheap_pricing_wall_s": cheap_pricing_wall_s,
            "exact_pricing_wall_s": exact_pricing_wall_s,
            "pricing_timeout_flight_id": pricing_timeout_flight_id,
            # Total pricing wall across every sweep.  Pricing is where a solve spends
            # its time, so this against `elapsed_s` says how much of the run was the
            # subproblem and how much was the master.
            "pricing_wall_s": pricing_wall_s,
            # What the sweep's workers COMPUTED, against the wall above.  Divided by
            # `pricing_wall_s * n_pricing_workers` this is the pool's occupancy; equal to
            # `pricing_wall_s` when the sweep ran sequentially.  Without it a slow run
            # cannot be told apart from a badly-scheduled one after the fact.
            "pricing_task_total_s": pricing_task_total_s,
            # Seconds spent STARTING workers, across the solve.  Under a solve-scoped
            # pool this is paid on the first sweep and never again, so a value near
            # `pricing_wall_s` means the pool is being rebuilt and the workers are cold.
            "pricing_pool_setup_s": pricing_pool_setup_s,
            "n_pricing_workers": params.n_pricing_workers,
            # Exact-pricing calls and how many fell back to the pure-Python reference.
            # A fallback returns the SAME column 3-4.5x slower, so it moves no other number
            # in this dict; a nonzero count is the only way a run reports that its compiled
            # path was not actually serving it.
            "kernel_priced": kernel_counters.get("priced", 0),
            "kernel_fell_back": kernel_counters.get("fell_back", 0),
            # WHY it fell back, which the count alone cannot say: "numba is missing" and "a
            # partial expansion saturated on real data" call for opposite responses, and the
            # warning that distinguishes them goes to stderr and dies with the process.  The
            # pair below is the label pool's precursor and its failure -- restarts climbing
            # while declines stay 0 is a run paying for the pool it needs without having lost
            # the compiled path yet.
            "kernel_label_restarts": kernel_counters.get("label_restarts", 0),
            "kernel_budget_declined": kernel_counters.get("budget_declined", 0),
            # Its own key rather than a member of the histogram below, which filters to
            # `declined_*`: a worker that disappeared is not a pricing verdict, and the
            # evidence has to survive even when `termination_reason` does not -- a run that
            # lost a worker and then also ran out of clock should still show WHY.
            "pricing_worker_lost": worker_losses,
            "kernel_declined_by_reason": {
                key.removeprefix("declined_"): value
                for key, value in sorted(kernel_counters.items())
                if key.startswith("declined_")
            },
            "n_columns": len(master.columns),
            "seeded_columns": seeded_columns,
            "ladder_columns": ladder_columns,
            **_dual_regime_stats(_backend_name(master)),
            "n_materialized_rows": len(materialized_rows),
            "lazy_rows_added": lazy_rows_added,
            "lazy_row_rounds": lazy_row_rounds,
            "time_limit_overrun_s": max(0.0, elapsed_s - params.time_limit_s),
            "elapsed_s": elapsed_s,
        }
        return ColGenResult(columns=incumbent, stats=stats)


__all__ = ["ColGenResult", "ColGenSolver"]
