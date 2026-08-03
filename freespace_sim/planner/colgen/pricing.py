"""Reduced-cost pricing for the single-level column-generation network.

The master problem is written as a maximization problem.  A route therefore
has reduced cost ``M - delay_s - capacity_duals - flight_dual`` and is useful
only when that value is positive.  The dynamic program below uses the same
integer clock as :mod:`.network`; every returned path is subsequently passed
through :func:`.network.column_claims`, which is the authoritative geometry,
budget, and claim-membership gate.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterable, Mapping, Set as AbstractSet
from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np

from ...config import SimConfig
from ...types import IntentStatus, Terminal
from ...volumes import (
    column_dwell_s,
    enroute_detour_m,
    enroute_flown_m,
    enroute_reference_m,
    exit_radius,
)
from .. import hexgrid as hg
from .network import Cell, FlightGraph, RowKey, column_claims
from .translate import Column, column_to_intent
from .windows import (
    derive_cell_window,
    endpoint_claim_cells,
    endpoint_claim_steps,
    terminal_claim_steps,
    visit_rows,
)

_SCORE_EPS = 1e-12
_RECOMPUTE_EPS = 1e-8
_EMPTY_ROWS: frozenset[RowKey] = frozenset()


class PricingTimeout(TimeoutError):
    """Raised when an exact pricing call reaches its caller-owned deadline."""


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise PricingTimeout("column pricing reached its wall-clock deadline")


@dataclass(frozen=True, slots=True)
class _PrefixSeries:
    """Dense prefix sums for one sparse row-key time series."""

    first_step: int
    prefix: tuple[float, ...]

    def range_sum(self, start: int, stop: int) -> float:
        """Return the sum over the half-open integer interval ``[start, stop)``."""

        if stop <= start or len(self.prefix) <= 1:
            return 0.0
        series_stop = self.first_step + len(self.prefix) - 1
        lo = min(max(start, self.first_step), series_stop)
        hi = min(max(stop, self.first_step), series_stop)
        if hi <= lo:
            return 0.0
        return self.prefix[hi - self.first_step] - self.prefix[lo - self.first_step]


def _prefix_series(values: Mapping[int, float]) -> _PrefixSeries:
    first = min(values)
    last = max(values)
    prefix = [0.0]
    running = 0.0
    for step in range(first, last + 1):
        running += values.get(step, 0.0)
        prefix.append(running)
    return _PrefixSeries(first, tuple(prefix))


class DualView:
    """Indexed view of non-negative master-row dual prices.

    Cell and terminal row values are converted once to dense per-resource
    prefix sums.  A visit-window or terminal interval query is consequently
    O(1); a customer endpoint query is O(number of claimed cells and levels).
    ``claim_cost`` retains the normalized mapping for the final de-duplicated
    reduced-cost recomputation.
    """

    __slots__ = (
        "_cell",
        "_duals",
        "_has_active_duals",
        "_max_negative_credit",
        "_offsets",
        "_terminal",
    )

    def __init__(
        self,
        duals: Mapping[RowKey | tuple[Any, ...], float],
        cfg: SimConfig,
    ) -> None:
        normalized: dict[RowKey, float] = {}
        cell_values: dict[tuple[Cell, int], dict[int, float]] = {}
        terminal_values: dict[Hashable, dict[int, float]] = {}
        for raw_key, raw_value in duals.items():
            key = raw_key if isinstance(raw_key, RowKey) else RowKey(raw_key)
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"dual for row {key!r} must be finite, got {value!r}")
            # LP tolerances can expose a signed zero or a minute negative value.
            # Preserve it: backend sign normalization, not pricing, owns the
            # mathematical sign convention.
            normalized[key] = normalized.get(key, 0.0) + value
            if key.kind == "cell":
                bucket = cell_values.setdefault((key.cell_coord, key.level), {})
            else:
                bucket = terminal_values.setdefault(key.terminal_id, {})
            bucket[key.step] = bucket.get(key.step, 0.0) + value

        self._duals = normalized
        self._has_active_duals = any(value != 0.0 for value in normalized.values())
        self._max_negative_credit = -math.fsum(min(0.0, value) for value in normalized.values())
        self._offsets = derive_cell_window(cfg)
        self._cell = {resource: _prefix_series(values) for resource, values in cell_values.items()}
        self._terminal = {
            terminal_id: _prefix_series(values) for terminal_id, values in terminal_values.items()
        }

    @property
    def offsets(self) -> tuple[int, int]:
        """The geometry-derived inclusive visit-window offsets."""

        return self._offsets

    def row_cost(self, key: RowKey | tuple[Any, ...]) -> float:
        """Return one normalized row dual, or zero for an unmaterialized row."""

        normalized = key if isinstance(key, RowKey) else RowKey(key)
        return self._duals.get(normalized, 0.0)

    def claim_cost(self, claims: Iterable[RowKey]) -> float:
        """Sum a de-duplicated claim collection in the master's maximize sense."""

        return math.fsum(self._duals.get(key, 0.0) for key in claims)

    def active_claims(self, claims: Iterable[RowKey]) -> frozenset[RowKey]:
        """Return the claim keys whose dual can affect a later union cost."""

        return frozenset(key for key in claims if self._duals.get(key, 0.0) != 0.0)

    @property
    def has_active_duals(self) -> bool:
        """Whether any capacity row has a nonzero price."""

        return self._has_active_duals

    @property
    def max_negative_credit(self) -> float:
        """Largest possible RC gain from tiny negative backend-tolerance duals."""

        return self._max_negative_credit

    def visit_cost(self, cell: Cell, level: int, visit_step: int) -> float:
        """Return all cell-row duals charged by a centre visit in O(1)."""

        series = self._cell.get((cell, level))
        if series is None:
            return 0.0
        lo, hi = self._offsets
        return series.range_sum(visit_step + lo, visit_step + hi + 1)

    def endpoint_cost(
        self,
        cells: Iterable[Cell],
        levels: Iterable[int],
        steps: range,
    ) -> float:
        """Return customer-cylinder cell-row duals in O(cells * levels)."""

        start, stop = _range_bounds(steps)
        total = 0.0
        level_tuple = tuple(levels)
        for cell in cells:
            for level in level_tuple:
                series = self._cell.get((cell, level))
                if series is not None:
                    total += series.range_sum(start, stop)
        return total

    def dwell_cost(self, terminal_id: Hashable, steps: int | range) -> float:
        """Return one terminal-row dual or an interval sum in O(1)."""

        series = self._terminal.get(terminal_id)
        if series is None:
            return 0.0
        if isinstance(steps, range):
            start, stop = _range_bounds(steps)
            return series.range_sum(start, stop)
        step = int(steps)
        return series.range_sum(step, step + 1)


def _range_bounds(steps: range) -> tuple[int, int]:
    if steps.step != 1:
        raise ValueError("capacity-row ranges must have unit stride")
    return steps.start, steps.stop


@dataclass(frozen=True, slots=True)
class _Label:
    """One compressed DAG label; ``path`` also supplies deterministic ties."""

    score: float
    departure_step: int
    origin_lane_idx: int | None
    path: tuple[Cell, ...]
    origin_paid_rows: frozenset[RowKey]

    @property
    def hops(self) -> int:
        return len(self.path) - 1

    @property
    def tie_key(self) -> tuple[Any, ...]:
        return (
            self.hops,
            self.departure_step,
            -1 if self.origin_lane_idx is None else self.origin_lane_idx,
            self.path,
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A sink-reaching label ranked by its de-duplicated provisional RC."""

    reduced_cost: float
    delay_s: float
    label: _Label
    dest_lane_idx: int | None

    @property
    def tie_key(self) -> tuple[Any, ...]:
        return (
            self.label.hops,
            self.label.departure_step,
            -1 if self.label.origin_lane_idx is None else self.label.origin_lane_idx,
            -1 if self.dest_lane_idx is None else self.dest_lane_idx,
            self.label.path,
        )


def _prefer(new: _Label, old: _Label | None) -> bool:
    if old is None or new.score > old.score + _SCORE_EPS:
        return True
    return abs(new.score - old.score) <= _SCORE_EPS and new.tie_key < old.tie_key


def _visit_claims(
    cell: Cell,
    level: int,
    visit_step: int,
    offsets: tuple[int, int],
) -> frozenset[RowKey]:
    q, r = cell
    return frozenset(
        RowKey.cell(q, r, level, row_step) for row_step in visit_rows(visit_step, offsets)
    )


def _endpoint_claims(
    fg: FlightGraph,
    cfg: SimConfig,
    *,
    origin: bool,
    step: int,
    timing_steps: int,
) -> frozenset[RowKey]:
    point = fg.request.origin if origin else fg.request.dest
    terminal = fg.origin_terminal if origin else fg.dest_terminal
    z = fg.levels[0]
    t0 = step * cfg.dt_s
    t1 = t0 + cfg.hover_time_s + column_dwell_s(point, terminal, cfg, z)
    if terminal is not None:
        return frozenset(
            RowKey.term(terminal.id, row_step) for row_step in terminal_claim_steps(t0, t1, cfg)
        )
    cells = endpoint_claim_cells(point, cfg.effective_hover_radius_m, cfg)
    steps = endpoint_claim_steps(t0, t1, cfg, timing_steps=timing_steps)
    return frozenset(
        RowKey.cell(q, r, level, row_step)
        for q, r in cells
        for level in range(len(fg.levels))
        for row_step in steps
    )


def _origin_options(fg: FlightGraph) -> tuple[tuple[int | None, Cell, int], ...]:
    if fg.origin_terminal is None:
        return ((None, fg.origin_cell, 0),)
    return tuple(
        (index, lane.cell, lane.steps)
        for index, lane in enumerate(fg.origin_lanes)
        if lane.cell in fg.corridor_cells
    )


def _destination_options(fg: FlightGraph) -> dict[Cell, tuple[int | None, ...]]:
    if fg.dest_terminal is None:
        return {fg.dest_cell: (None,)}
    result: dict[Cell, list[int | None]] = {}
    for index, lane in enumerate(fg.dest_lanes):
        if lane.cell in fg.corridor_cells:
            result.setdefault(lane.cell, []).append(index)
    return {cell: tuple(indices) for cell, indices in result.items()}


def _distance_to_destinations(
    fg: FlightGraph,
    destination_cells: AbstractSet[Cell],
    *,
    deadline: float | None = None,
) -> dict[Cell, int]:
    """Reverse BFS distances respecting directed static-hop exclusions."""

    distance = {cell: 0 for cell in destination_cells if cell in fg.corridor_cells}
    queue = deque(sorted(distance))
    while queue:
        _check_deadline(deadline)
        target = queue.popleft()
        next_distance = distance[target] + 1
        tq, tr = target
        for dq, dr in hg.AXIAL_NEIGHBORS:
            predecessor = tq + dq, tr + dr
            if predecessor not in fg.corridor_cells or predecessor in distance:
                continue
            if (predecessor, target) in fg.forbidden_hops:
                continue
            distance[predecessor] = next_distance
            queue.append(predecessor)
    return distance


def _shortest_cell_path(
    fg: FlightGraph,
    start: Cell,
    destination: Cell,
    *,
    deadline: float | None = None,
) -> tuple[Cell, ...] | None:
    """Return one deterministic directed BFS path inside the frozen corridor."""

    if start == destination:
        return None  # A column must contain a real lateral hop.
    predecessor: dict[Cell, Cell | None] = {start: None}
    queue = deque((start,))
    while queue:
        _check_deadline(deadline)
        cell = queue.popleft()
        for neighbour in sorted(hg.hex_neighbors(*cell)):
            if neighbour not in fg.corridor_cells or neighbour in predecessor:
                continue
            if (cell, neighbour) in fg.forbidden_hops:
                continue
            predecessor[neighbour] = cell
            if neighbour == destination:
                reversed_path = [destination]
                cursor = cell
                while cursor is not None:
                    reversed_path.append(cursor)
                    cursor = predecessor[cursor]
                return tuple(reversed(reversed_path))
            queue.append(neighbour)
    return None


def _path_claims(
    fg: FlightGraph,
    cfg: SimConfig,
    label: _Label,
    dest_lane_idx: int | None,
) -> frozenset[RowKey]:
    """Build the intended row union cheaply before canonical certification."""

    del dest_lane_idx  # The selected destination lane changes geometry, not dwell row membership.
    origin_lane_steps = (
        0 if label.origin_lane_idx is None else fg.origin_lanes[label.origin_lane_idx].steps
    )
    corridor_start = label.departure_step + fg.takeoff_steps[0] + origin_lane_steps
    claims = set(
        _endpoint_claims(
            fg,
            cfg,
            origin=True,
            step=label.departure_step,
            timing_steps=0,
        )
    )
    offsets = derive_cell_window(cfg)
    for offset, cell in enumerate(label.path):
        claims.update(_visit_claims(cell, 0, corridor_start + offset, offsets))
    arrival_step = corridor_start + label.hops
    claims.update(
        _endpoint_claims(
            fg,
            cfg,
            origin=False,
            step=arrival_step,
            timing_steps=label.hops,
        )
    )
    return frozenset(claims)


def _path_delay_s(fg: FlightGraph, cfg: SimConfig, label: _Label) -> float:
    """Compute the exact v1 delay ruler without building reservation volumes."""

    radius = hg.circumradius(cfg)
    z = fg.levels[0]
    points = [np.array([*hg.hex_center(q, r, radius), z]) for q, r in label.path]
    reference = enroute_reference_m(
        fg.request.origin,
        fg.request.dest,
        fg.origin_terminal,
        fg.dest_terminal,
        cfg,
    )
    flown = enroute_flown_m(
        points,
        fg.request.origin,
        fg.request.dest,
        fg.origin_terminal,
        fg.dest_terminal,
        cfg,
    )
    detour = enroute_detour_m(flown, reference)
    return (label.departure_step - fg.base_step) * cfg.dt_s + detour / cfg.nominal_speed_mps


def _fold_leg_s(point, terminal: Terminal | None, lane_dist: float | None, cfg: SimConfig) -> float:
    """Return the endpoint leg charged by the arc-form delay objective."""

    if terminal is not None:
        assert lane_dist is not None
        return max(0.0, lane_dist - exit_radius(terminal, cfg)) / cfg.nominal_speed_mps
    assert lane_dist is None
    radius = hg.circumradius(cfg)
    cell = hg.enu_to_axial(float(point[0]), float(point[1]), radius)
    center = hg.hex_center(*cell, radius)
    return (
        math.hypot(float(point[0]) - float(center[0]), float(point[1]) - float(center[1]))
        / cfg.nominal_speed_mps
    )


def _terminal_fold_leg_s(
    point,
    terminal: Terminal,
    cell: Cell,
    cfg: SimConfig,
) -> tuple[float, bool]:
    """Return a terminal fold leg and whether folding retains its lane cell.

    ``fold_corners_to_columns`` drops endpoint corners that are even slightly
    inside the exit radius.  Boundary-lane construction permits a sub-nanometre
    inward round-off, so using ``Lane.dist`` (or a tolerance) to decide this can
    revive the terminal fold-replacement pruning bug.  Recompute the scalar
    distance with the same ``sqrt(dx*dx + dy*dy)`` predicate as folding and
    enable the arc lower bound only when the lane is retained exactly.
    """

    radius = hg.circumradius(cfg)
    x, y = hg.hex_center(*cell, radius)
    dx = float(x) - float(point[0])
    dy = float(y) - float(point[1])
    distance = math.sqrt(dx * dx + dy * dy)
    edge_radius = exit_radius(terminal, cfg)
    return max(0.0, distance - edge_radius) / cfg.nominal_speed_mps, distance >= edge_radius


def _arc_delay_lower_bound_s(
    *,
    ground_delay_s: float,
    origin_fold_s: float,
    hops: int,
    remaining_hops: int,
    destination_fold_s: float,
    reference_time_s: float,
    dt_s: float,
    folding_exact: bool,
) -> float:
    """Lower-bound canonical delay for every completion of one prefix.

    When both terminal folds retain their selected boundary cells, the raw
    centreline consists of the origin fold, one equal-length lattice arc per
    hop, and the destination fold.  Reverse-BFS distance lower-bounds the
    unflown arcs, while independently minimizing the destination fold can only
    weaken that bound.  Customer snap legs have the same additive form.

    A terminal lane inside its fold radius invalidates that decomposition: a
    later hop can be dropped by folding without increasing canonical flown
    distance.  The safe fallback is then the irrevocable ground delay.  The
    same fallback is required for a zero reference because
    ``enroute_detour_m`` deliberately defines its detour as zero.
    """

    if not folding_exact or reference_time_s <= 0.0:
        return ground_delay_s
    flown_time_lb = origin_fold_s + (hops + remaining_hops) * dt_s + destination_fold_s
    return ground_delay_s + max(0.0, flown_time_lb - reference_time_s)


def _pricing_tolerance(params: Any) -> float:
    for name in ("reduced_cost_tol", "pricing_tol"):
        if hasattr(params, name):
            value = float(getattr(params, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"params.{name} must be finite and non-negative")
            return value
    return 1e-9


def _shift_claims(claims: Iterable[RowKey], delta_steps: int) -> frozenset[RowKey]:
    """Translate a canonical claim set by an integer number of clock steps."""

    if delta_steps == 0:
        return frozenset(claims)
    shifted: set[RowKey] = set()
    for row in claims:
        if row.kind == "cell":
            q, r = row.cell_coord
            shifted.add(RowKey.cell(q, r, row.level, row.step + delta_steps))
        else:
            shifted.add(RowKey.term(row.terminal_id, row.step + delta_steps))
    return frozenset(shifted)


def _shifted_seed_incumbent(
    seed: Column,
    fg: FlightGraph,
    duals: DualView,
    pi_f: float,
    cfg: SimConfig,
    benefit: float,
    forbidden_rows: frozenset[RowKey],
    incumbent: tuple[float, Column] | None,
    *,
    deadline: float | None = None,
) -> tuple[float, Column] | None:
    """Strengthen an incumbent by scanning time-translations of its seed path.

    The spatial path, endpoint lanes, wall verdict, and detour are invariant
    under an integer clock translation.  Canonical capacity claims translate
    by the same integer, and ground delay grows by exactly ``delta * dt``.
    These columns are used only as certified lower bounds for the exact DAG;
    the prepass never proves optimality or replaces pricing.

    With non-negative duals, the first feasible translation that pays no dual
    dominates every later seed translation, so the scan may stop there.  A
    negative backend-tolerance dual disables that stopping rule.
    """

    origin_lane_steps = (
        0 if seed.origin_lane_idx is None else fg.origin_lanes[seed.origin_lane_idx].steps
    )
    path_latest_departure = (
        fg.max_step
        - fg.takeoff_steps[seed.level]
        - origin_lane_steps
        - (len(seed.cell_path) - 1)
    )
    latest_departure = min(fg.latest_departure_step, path_latest_departure)
    best = incumbent
    for departure_step in range(seed.departure_step + 1, latest_departure + 1):
        _check_deadline(deadline)
        delta_steps = departure_step - seed.departure_step
        claims = _shift_claims(seed.claims, delta_steps)
        if not claims.isdisjoint(forbidden_rows):
            continue
        delay_s = seed.delay_s + delta_steps * cfg.dt_s
        dual_cost = duals.claim_cost(claims)
        reduced_cost = benefit - delay_s - dual_cost - pi_f
        if best is None or reduced_cost > best[0] + _SCORE_EPS:
            best = (
                reduced_cost,
                Column(
                    flight_id=seed.flight_id,
                    departure_step=departure_step,
                    level=seed.level,
                    origin_lane_idx=seed.origin_lane_idx,
                    dest_lane_idx=seed.dest_lane_idx,
                    cell_path=seed.cell_path,
                    delay_s=delay_s,
                    claims=claims,
                ),
            )
        if dual_cost == 0.0 and duals.max_negative_credit == 0.0:
            break
    return best


def _benefit(params: Any) -> float:
    try:
        value = float(params.M)
    except AttributeError as exc:
        raise AttributeError("Phase-2 pricing requires params.M") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("params.M must be finite and positive")
    return value


def _canonical_candidate(
    candidate: _Candidate,
    fg: FlightGraph,
    duals: DualView,
    pi_f: float,
    cfg: SimConfig,
    benefit: float,
    forbidden_rows: frozenset[RowKey],
) -> tuple[float, Column] | None:
    label = candidate.label
    provisional = Column(
        flight_id=fg.request.flight_id,
        departure_step=label.departure_step,
        level=0,
        origin_lane_idx=label.origin_lane_idx,
        dest_lane_idx=candidate.dest_lane_idx,
        cell_path=label.path,
        delay_s=candidate.delay_s,
    )
    try:
        claims = column_claims(provisional, fg, cfg)
    except (ValueError, NotImplementedError):
        return None
    if not claims.isdisjoint(forbidden_rows):
        return None

    # ``column_claims`` already translated the path as its canonical budget
    # gate.  Translate once more to set the objective from precisely the same
    # metric fields exposed to callers; this is only done for top sink labels.
    intent = column_to_intent(provisional, fg.request, cfg)
    if intent.status is not IntentStatus.ACCEPTED:
        return None
    exact_delay = (
        intent.ground_delay_s + intent.air_hold_s + intent.air_detour_m / cfg.nominal_speed_mps
    )
    column = Column(
        flight_id=provisional.flight_id,
        departure_step=provisional.departure_step,
        level=provisional.level,
        origin_lane_idx=provisional.origin_lane_idx,
        dest_lane_idx=provisional.dest_lane_idx,
        cell_path=provisional.cell_path,
        delay_s=exact_delay,
        claims=claims,
    )
    reduced_cost = benefit - exact_delay - duals.claim_cost(claims) - pi_f
    return reduced_cost, column


def _shortest_seed_columns(
    fg: FlightGraph,
    cfg: SimConfig,
    *,
    deadline: float | None = None,
) -> tuple[Column, ...]:
    """Certify deterministic BFS columns for every endpoint-lane pairing."""

    view = DualView({}, cfg)
    candidates: list[_Candidate] = []
    for origin_lane_idx, start, lane_steps in _origin_options(fg):
        for destination, destination_lane_indices in sorted(_destination_options(fg).items()):
            _check_deadline(deadline)
            path = _shortest_cell_path(fg, start, destination, deadline=deadline)
            if path is None:
                continue
            arrival_step = fg.base_step + fg.takeoff_steps[0] + lane_steps + len(path) - 1
            if arrival_step > fg.max_step:
                continue
            label = _Label(0.0, fg.base_step, origin_lane_idx, path, frozenset())
            delay_s = _path_delay_s(fg, cfg, label)
            for dest_lane_idx in destination_lane_indices:
                candidates.append(_Candidate(-delay_s, delay_s, label, dest_lane_idx))

    candidates.sort(key=lambda candidate: (-candidate.reduced_cost, candidate.tie_key))
    columns: list[Column] = []
    for candidate in candidates:
        canonical = _canonical_candidate(
            candidate,
            fg,
            view,
            0.0,
            cfg,
            0.0,
            _EMPTY_ROWS,
        )
        if canonical is not None:
            columns.append(canonical[1])
    return tuple(columns)


def _shortest_seed(
    fg: FlightGraph,
    cfg: SimConfig,
    *,
    deadline: float | None = None,
) -> Column | None:
    """Certify the best deterministic BFS seed without expanding the time DAG."""

    columns = _shortest_seed_columns(fg, cfg, deadline=deadline)
    return None if not columns else columns[0]


def _best_column(
    fg: FlightGraph,
    dual_view: DualView,
    pi_f: float,
    cfg: SimConfig,
    benefit: float,
    forbidden_rows: frozenset[RowKey],
    *,
    seed: bool,
    incumbent: tuple[float, Column] | None = None,
    deadline: float | None = None,
) -> tuple[float, Column | None]:
    _check_deadline(deadline)
    if len(fg.levels) != 1:
        raise NotImplementedError(
            "colgen v1 pricing supports a single flight level; multi-level pricing is planned"
        )
    if not math.isfinite(pi_f):
        raise ValueError(f"flight-row dual must be finite, got {pi_f!r}")

    destination_options = _destination_options(fg)
    if not destination_options:
        return -math.inf, None
    distances = _distance_to_destinations(fg, set(destination_options), deadline=deadline)
    origin_options = _origin_options(fg)
    # ``detour_slack_hops`` sizes the spatial ellipse; it is not a route-length
    # budget.  Ordinary pricing may spend the clock slack on W-separated wide
    # loops (the network's en-route-waiting lever), subject to the canonical
    # detour gate at the sink.  A zero-dual seed has no reason to loop, so its
    # tighter hop limit avoids exploring value-tied cyclic walks.
    seed_hop_limit = fg.shortest_hops + fg.detour_slack_hops
    if seed_hop_limit < 1:
        return -math.inf, None

    offsets = dual_view.offsets
    revisit_depth = offsets[1] - offsets[0]
    # Revisit exclusion and dominance need different histories.  With W=2
    # (time_buffer_s=0), no predecessor is forbidden, but two equal-score
    # labels arriving from different predecessors can differ in which visit
    # rows the destination endpoint later de-duplicates.  Preserve at least
    # the predecessor in the state while consulting only ``revisit_depth``
    # cells for the ban.
    state_history_depth = max(2, revisit_depth)
    track_first_hop = bool(fg.static_walls and fg.origin_terminal is not None)
    # The state keeps enough tail to distinguish sink unions; the revisit
    # check below reads only the prefix whose windows could overlap.
    layers: dict[
        int,
        dict[
            tuple[
                Cell,
                tuple[Cell, ...],
                frozenset[RowKey],
                tuple[Cell, Cell] | None,
            ],
            _Label,
        ],
    ] = {}
    departure_steps: Iterable[int]
    if seed:
        departure_steps = (fg.base_step,)
    else:
        departure_steps = range(fg.base_step, fg.latest_departure_step + 1)

    origin_leg_by_lane: dict[int | None, float] = {}
    origin_fold_lb_by_lane: dict[int | None, tuple[float, bool]] = {}
    for lane_idx, cell, _lane_steps in origin_options:
        lane_dist = None if lane_idx is None else fg.origin_lanes[lane_idx].dist
        origin_leg_by_lane[lane_idx] = _fold_leg_s(
            fg.request.origin,
            fg.origin_terminal,
            lane_dist,
            cfg,
        )
        if fg.origin_terminal is None:
            origin_fold_lb_by_lane[lane_idx] = origin_leg_by_lane[lane_idx], True
        else:
            origin_fold_lb_by_lane[lane_idx] = _terminal_fold_leg_s(
                fg.request.origin,
                fg.origin_terminal,
                cell,
                cfg,
            )

    destination_fold_exact = True
    if fg.dest_terminal is None:
        destination_fold_lb = _fold_leg_s(fg.request.dest, None, None, cfg)
    else:
        destination_folds: list[float] = []
        for destination, lane_indices in destination_options.items():
            for lane_idx in lane_indices:
                assert lane_idx is not None
                fold_s, retained = _terminal_fold_leg_s(
                    fg.request.dest,
                    fg.dest_terminal,
                    destination,
                    cfg,
                )
                destination_folds.append(fold_s)
                destination_fold_exact &= retained
        destination_fold_lb = min(destination_folds)

    reference_m = enroute_reference_m(
        fg.request.origin,
        fg.request.dest,
        fg.origin_terminal,
        fg.dest_terminal,
        cfg,
    )
    reference_time_s = reference_m / cfg.nominal_speed_mps
    detour_defined = reference_m > 1e-9

    def delay_lower_bound(
        departure_step: int,
        lane_idx: int | None,
        hops: int,
        remaining_hops: int,
    ) -> float:
        origin_fold_s, origin_fold_exact = origin_fold_lb_by_lane[lane_idx]
        return _arc_delay_lower_bound_s(
            ground_delay_s=(departure_step - fg.base_step) * cfg.dt_s,
            origin_fold_s=origin_fold_s,
            hops=hops,
            remaining_hops=remaining_hops,
            destination_fold_s=destination_fold_lb,
            reference_time_s=reference_time_s,
            dt_s=cfg.dt_s,
            folding_exact=detour_defined and origin_fold_exact and destination_fold_exact,
        )

    # Every completion pays its destination endpoint union.  It may duplicate
    # an earlier visit row, so adding the two prices would be unsafe; the
    # positive price of their union is nevertheless at least the maximum of
    # (a) positive duals already paid and (b) positive destination-row duals.
    # Cache both parts over total hop count: arrival time, endpoint dwell rows,
    # and arc delay then depend only on that count.
    completion_envelopes: dict[
        tuple[int, int | None],
        tuple[tuple[float, ...], tuple[float, ...]],
    ] = {}
    destination_lane_tie = min(
        -1 if lane_idx is None else lane_idx
        for lane_indices in destination_options.values()
        for lane_idx in lane_indices
    )

    def completion_envelope(
        departure_step: int,
        lane_idx: int | None,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        key = departure_step, lane_idx
        cached = completion_envelopes.get(key)
        if cached is not None:
            return cached

        lane_steps = 0 if lane_idx is None else fg.origin_lanes[lane_idx].steps
        corridor_start = departure_step + fg.takeoff_steps[0] + lane_steps
        max_total_hops = fg.max_step - corridor_start
        delay_lbs = [math.inf]
        destination_positive_costs = [math.inf]
        for total_hops in range(1, max_total_hops + 1):
            _check_deadline(deadline)
            delay_lb = delay_lower_bound(departure_step, lane_idx, total_hops, 0)
            # Delay is monotone in total hops whenever the arc form is enabled,
            # and constant in the conservative ground-only fallback.  Once even
            # collecting every negative dual cannot match the incumbent, no
            # later hop count can matter to this exact search.
            if incumbent is not None and (
                benefit - pi_f - delay_lb + dual_view.max_negative_credit
                < incumbent[0] - _RECOMPUTE_EPS
            ):
                break

            arrival_step = corridor_start + total_hops
            endpoint_claims = _endpoint_claims(
                fg,
                cfg,
                origin=False,
                step=arrival_step,
                timing_steps=total_hops,
            )
            destination_cost = math.inf
            for destination in destination_options:
                final_visit_claims = _visit_claims(destination, 0, arrival_step, offsets)
                unavoidable_claims = endpoint_claims | final_visit_claims
                if not unavoidable_claims.isdisjoint(forbidden_rows):
                    continue
                destination_cost = min(
                    destination_cost,
                    math.fsum(max(0.0, dual_view.row_cost(row)) for row in unavoidable_claims),
                )
            if not math.isfinite(destination_cost):
                delay_lbs.append(math.inf)
                destination_positive_costs.append(math.inf)
                continue
            delay_lbs.append(delay_lb)
            destination_positive_costs.append(destination_cost)

        result = tuple(delay_lbs), tuple(destination_positive_costs)
        completion_envelopes[key] = result
        return result

    def completion_can_compete(
        departure_step: int,
        lane_idx: int | None,
        minimum_total_hops: int,
        paid_duals: float,
        *,
        paid_duals_exact: bool,
    ) -> bool:
        """Whether the relaxed completion can improve or win an exact RC tie."""

        if incumbent is None:
            return True
        delay_lbs, destination_positive_costs = completion_envelope(
            departure_step,
            lane_idx,
        )
        first_hops = max(1, minimum_total_hops)
        if first_hops >= len(delay_lbs):
            return False

        incumbent_column = incumbent[1]
        incumbent_prefix = (
            len(incumbent_column.cell_path) - 1,
            incumbent_column.departure_step,
            -1 if incumbent_column.origin_lane_idx is None else incumbent_column.origin_lane_idx,
            -1 if incumbent_column.dest_lane_idx is None else incumbent_column.dest_lane_idx,
        )
        origin_lane_tie = -1 if lane_idx is None else lane_idx
        paid_positive_lb = max(
            0.0,
            paid_duals - (0.0 if paid_duals_exact else _RECOMPUTE_EPS),
        )
        for total_hops in range(first_hops, len(delay_lbs)):
            union_positive_lb = max(
                paid_positive_lb,
                destination_positive_costs[total_hops],
            )
            hop_rc_bound = (
                benefit
                - pi_f
                - delay_lbs[total_hops]
                - union_positive_lb
                + dual_view.max_negative_credit
            )
            if hop_rc_bound > incumbent[0] + _SCORE_EPS:
                return True
            if not paid_duals_exact and (hop_rc_bound >= incumbent[0] - _RECOMPUTE_EPS):
                # Label scores reconstruct paid duals by cancellation.  Keep
                # the wider numerical band competitive; lexicographic equality
                # pruning below is reserved for direct claim sums.
                return True
            if abs(hop_rc_bound - incumbent[0]) <= _SCORE_EPS:
                # The path itself is unknown in this relaxation.  Equality in
                # the first four fields may still hide a lexicographically
                # better path, so retain it; a strictly worse prefix cannot win
                # the pricing tie and is safe to discard.
                possible_prefix = (
                    total_hops,
                    departure_step,
                    origin_lane_tie,
                    destination_lane_tie,
                )
                if possible_prefix <= incumbent_prefix:
                    return True
        return False

    for departure_step in departure_steps:
        _check_deadline(deadline)
        ground_score = -(departure_step - fg.base_step) * cfg.dt_s
        if incumbent is not None:
            start_upper_bound = benefit + ground_score - pi_f + dual_view.max_negative_credit
            if start_upper_bound < incumbent[0] - _RECOMPUTE_EPS:
                continue
        origin_claims = _endpoint_claims(
            fg,
            cfg,
            origin=True,
            step=departure_step,
            timing_steps=0,
        )
        if not origin_claims.isdisjoint(forbidden_rows):
            continue
        for lane_idx, cell, lane_steps in origin_options:
            remaining_distance = distances.get(cell)
            if remaining_distance is None:
                continue
            start_step = departure_step + fg.takeoff_steps[0] + lane_steps
            if start_step >= fg.max_step:
                continue
            if start_step + remaining_distance > fg.max_step:
                continue
            if seed and remaining_distance > seed_hop_limit:
                continue
            visit_claims = _visit_claims(cell, 0, start_step, offsets)
            start_claims = origin_claims | visit_claims
            if not start_claims.isdisjoint(forbidden_rows):
                continue
            start_dual_cost = dual_view.claim_cost(start_claims)
            origin_paid_rows = dual_view.active_claims(start_claims)
            if not completion_can_compete(
                departure_step,
                lane_idx,
                remaining_distance,
                start_dual_cost,
                paid_duals_exact=True,
            ):
                continue
            score = ground_score - origin_leg_by_lane[lane_idx] - start_dual_cost
            label = _Label(score, departure_step, lane_idx, (cell,), origin_paid_rows)
            recent = (cell,)
            key = (cell, recent, origin_paid_rows, None)
            layer = layers.setdefault(start_step, {})
            if _prefer(label, layer.get(key)):
                layer[key] = label

    candidates: list[_Candidate] = []
    for step in range(fg.min_step, fg.max_step + 1):
        _check_deadline(deadline)
        layer = layers.get(step)
        if not layer:
            continue
        for label_index, ((cell, recent, origin_paid_rows, first_hop), label) in enumerate(
            sorted(
                layer.items(),
                key=lambda item: (item[0][0], item[0][1], item[1].tie_key),
            )
        ):
            if label_index % 128 == 0:
                _check_deadline(deadline)
            hops = label.hops
            if hops >= 1 and cell in destination_options:
                for dest_lane_idx in destination_options[cell]:
                    destination_claims = _endpoint_claims(
                        fg,
                        cfg,
                        origin=False,
                        step=step,
                        timing_steps=hops,
                    )
                    if destination_claims.isdisjoint(forbidden_rows):
                        claims = _path_claims(fg, cfg, label, dest_lane_idx)
                        if claims.isdisjoint(forbidden_rows):
                            delay_s = _path_delay_s(fg, cfg, label)
                            reduced_cost = benefit - delay_s - dual_view.claim_cost(claims) - pi_f
                            candidate = _Candidate(
                                reduced_cost,
                                delay_s,
                                label,
                                dest_lane_idx,
                            )
                            candidates.append(candidate)
                            # A certified improving sink tightens the safe
                            # lower-bound pruning for every later time layer.
                            if incumbent is None or reduced_cost > incumbent[0] + _SCORE_EPS:
                                canonical = _canonical_candidate(
                                    candidate,
                                    fg,
                                    dual_view,
                                    pi_f,
                                    cfg,
                                    benefit,
                                    forbidden_rows,
                                )
                                if canonical is not None and (
                                    incumbent is None or canonical[0] > incumbent[0] + _SCORE_EPS
                                ):
                                    incumbent = canonical

            if (seed and hops >= seed_hop_limit) or step + 1 > fg.max_step:
                continue
            if incumbent is not None:
                ground_delay = (label.departure_step - fg.base_step) * cfg.dt_s
                origin_leg = origin_leg_by_lane[label.origin_lane_idx]
                # ``label.score`` is the negative sum of ground delay, flown
                # time so far, and the de-duplicated duals paid so far.
                paid_duals = -label.score - ground_delay - origin_leg - hops * cfg.dt_s
                remaining_distance = distances[cell]
                # The endpoint-aware envelope lower-bounds the positive price
                # of the eventual row union without double-counting overlaps.
                # It also handles exact RC ties in the same hops-first order as
                # candidates.
                if not completion_can_compete(
                    label.departure_step,
                    label.origin_lane_idx,
                    hops + remaining_distance,
                    paid_duals,
                    paid_duals_exact=False,
                ):
                    continue
            cq, cr = cell
            for dq, dr in hg.AXIAL_NEIGHBORS:
                neighbour = cq + dq, cr + dr
                if neighbour not in fg.corridor_cells or neighbour in recent[:revisit_depth]:
                    continue
                if (cell, neighbour) in fg.forbidden_hops:
                    continue
                remaining_distance = distances.get(neighbour)
                next_step = step + 1
                if remaining_distance is None or next_step + remaining_distance > fg.max_step:
                    continue
                if seed and hops + 1 + remaining_distance > seed_hop_limit:
                    continue
                claims = _visit_claims(neighbour, 0, next_step, offsets)
                if not claims.isdisjoint(forbidden_rows):
                    continue
                next_recent = (neighbour, *recent[: state_history_depth - 1])
                next_label = _Label(
                    label.score - cfg.dt_s - dual_view.claim_cost(claims - origin_paid_rows),
                    label.departure_step,
                    label.origin_lane_idx,
                    (*label.path, neighbour),
                    origin_paid_rows,
                )
                next_first_hop = (
                    ((cell, neighbour) if first_hop is None else first_hop)
                    if track_first_hop
                    else None
                )
                key = (
                    neighbour,
                    next_recent,
                    origin_paid_rows,
                    next_first_hop,
                )
                next_layer = layers.setdefault(next_step, {})
                if _prefer(next_label, next_layer.get(key)):
                    next_layer[key] = next_label

    if not candidates:
        return (-math.inf, None) if incumbent is None else incumbent
    candidates.sort(key=lambda candidate: (-candidate.reduced_cost, candidate.tie_key))

    best: tuple[float, Column] | None = incumbent
    for candidate_index, candidate in enumerate(candidates):
        if candidate_index % 128 == 0:
            _check_deadline(deadline)
        # Provisional and canonical RC differ only at floating resampling ulps.
        # Once below the best by a generous numerical envelope, no later label
        # can win; exact ties are still certified and broken deterministically.
        if best is not None and candidate.reduced_cost < best[0] - _RECOMPUTE_EPS:
            break
        canonical = _canonical_candidate(
            candidate,
            fg,
            dual_view,
            pi_f,
            cfg,
            benefit,
            forbidden_rows,
        )
        if canonical is None:
            continue
        if best is None:
            best = canonical
            continue
        rc, column = canonical
        best_rc, best_column = best
        column_key = (
            len(column.cell_path) - 1,
            column.departure_step,
            -1 if column.origin_lane_idx is None else column.origin_lane_idx,
            -1 if column.dest_lane_idx is None else column.dest_lane_idx,
            column.cell_path,
        )
        best_key = (
            len(best_column.cell_path) - 1,
            best_column.departure_step,
            -1 if best_column.origin_lane_idx is None else best_column.origin_lane_idx,
            -1 if best_column.dest_lane_idx is None else best_column.dest_lane_idx,
            best_column.cell_path,
        )
        if rc > best_rc + _SCORE_EPS or (abs(rc - best_rc) <= _SCORE_EPS and column_key < best_key):
            best = canonical

    return (-math.inf, None) if best is None else best


def price_flight(
    fg: FlightGraph,
    duals: Mapping[RowKey | tuple[Any, ...], float] | DualView,
    pi_f: float,
    cfg: SimConfig,
    params: Any,
    *,
    forbidden_rows: frozenset[RowKey] = _EMPTY_ROWS,
    require_improving: bool = True,
    deadline: float | None = None,
) -> tuple[float, Column | None]:
    """Return the best positive-reduced-cost column for one flight.

    ``forbidden_rows`` is the repair seam: touching one of those already
    saturated rows makes an origin, visit, arrival, or final canonical column
    infeasible rather than merely expensive.  Repair also sets
    ``require_improving=False`` because it needs the best feasible trajectory
    even when a user-supplied benefit ``M`` is smaller than that trajectory's
    delay.  The returned reduced cost is always recomputed from the canonical
    de-duplicated claim set.
    """

    if not isinstance(require_improving, bool):
        raise TypeError("require_improving must be a boolean")
    if deadline is not None:
        deadline = float(deadline)
        if not math.isfinite(deadline):
            raise ValueError("pricing deadline must be finite")
    _check_deadline(deadline)
    pi_value = float(pi_f)
    if not math.isfinite(pi_value):
        raise ValueError(f"flight-row dual must be finite, got {pi_f!r}")
    view = duals if isinstance(duals, DualView) else DualView(duals, cfg)
    forbidden = frozenset(key if isinstance(key, RowKey) else RowKey(key) for key in forbidden_rows)
    benefit = _benefit(params)
    incumbent: tuple[float, Column] | None = None
    shortest_columns: tuple[Column, ...] = ()
    try:
        shortest_columns = _shortest_seed_columns(fg, cfg, deadline=deadline)
        seed = (
            shortest_columns[0]
            if shortest_columns
            else seed_column(fg, cfg, deadline=deadline)
        )
    except ValueError:
        # A deterministic shortest-path seed is an acceleration, not a
        # feasibility precondition.  The full DAG may still find a usable
        # path when the first geodesic fails a path-dependent wall check; a
        # truly disconnected graph simply falls through to ``None`` below.
        seed = None
    if seed is not None:
        if seed.claims.isdisjoint(forbidden):
            seed_dual_cost = view.claim_cost(seed.claims)
            seed_rc = benefit - seed.delay_s - seed_dual_cost - pi_value
            incumbent = seed_rc, seed
            # The seed is the globally minimum-delay column.  If it remains
            # feasible and pays no row price, non-negative duals and additional
            # exclusions can only make every alternative weakly worse.  This
            # locality check matters at batch scale: a dual or saturated row on the
            # other side of the region must not trigger this flight's full DAG.
            # Tiny negative backend-tolerance duals deliberately disable the
            # shortcut because another route could collect their credit.
            if seed_dual_cost == 0.0 and view.max_negative_credit == 0.0:
                if require_improving and seed_rc <= _pricing_tolerance(params):
                    return seed_rc, None
                return seed_rc, seed
        incumbent = _shifted_seed_incumbent(
            seed,
            fg,
            view,
            pi_value,
            cfg,
            benefit,
            forbidden,
            incumbent,
            deadline=deadline,
        )
        for alternative in shortest_columns[1:]:
            dual_free = False
            if alternative.claims.isdisjoint(forbidden):
                alternative_dual_cost = view.claim_cost(alternative.claims)
                alternative_rc = benefit - alternative.delay_s - alternative_dual_cost - pi_value
                if incumbent is None or alternative_rc > incumbent[0] + _SCORE_EPS:
                    incumbent = alternative_rc, alternative
                dual_free = alternative_dual_cost == 0.0 and view.max_negative_credit == 0.0
            if not dual_free:
                incumbent = _shifted_seed_incumbent(
                    alternative,
                    fg,
                    view,
                    pi_value,
                    cfg,
                    benefit,
                    forbidden,
                    incumbent,
                    deadline=deadline,
                )
    reduced_cost, column = _best_column(
        fg,
        view,
        pi_value,
        cfg,
        benefit,
        forbidden,
        seed=False,
        incumbent=incumbent,
        deadline=deadline,
    )
    if column is None or (require_improving and reduced_cost <= _pricing_tolerance(params)):
        return reduced_cost, None
    return reduced_cost, column


def seed_column(
    fg: FlightGraph,
    cfg: SimConfig,
    *,
    deadline: float | None = None,
) -> Column:
    """Return a deterministic, dual-free shortest-delay feasible seed.

    Only the nominal departure is considered.  With zero row prices the DAG
    minimizes ground hold, lateral hops, and endpoint fold/snap legs; shortest
    paths never exercise the short-revisit restriction, so this is the plan's
    unconstrained shortest-path seed while retaining the canonical wall and
    detour gates.
    """

    if deadline is not None:
        deadline = float(deadline)
        if not math.isfinite(deadline):
            raise ValueError("seed deadline must be finite")
    _check_deadline(deadline)
    direct = _shortest_seed(fg, cfg, deadline=deadline)
    if direct is not None:
        return direct

    # Rare path-position-dependent terminal-wall tagging can invalidate the
    # one deterministic BFS geodesic while leaving another corridor walk
    # usable.  Fall back to the bounded zero-dual DAG before declaring the
    # graph unseedable.
    view = DualView({}, cfg)
    _score, column = _best_column(
        fg,
        view,
        0.0,
        cfg,
        benefit=0.0,
        forbidden_rows=_EMPTY_ROWS,
        seed=True,
        incumbent=None,
        deadline=deadline,
    )
    if column is None:
        raise ValueError(f"flight {fg.request.flight_id} has no feasible seed column")
    return column


__all__ = ["DualView", "PricingTimeout", "price_flight", "seed_column"]
