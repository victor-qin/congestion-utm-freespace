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

    __slots__ = ("_cell", "_duals", "_offsets", "_terminal")

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

        return any(value != 0.0 for value in self._duals.values())

    @property
    def max_negative_credit(self) -> float:
        """Largest possible RC gain from tiny negative backend-tolerance duals."""

        return -math.fsum(min(0.0, value) for value in self._duals.values())

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
) -> dict[Cell, int]:
    """Reverse BFS distances respecting directed static-hop exclusions."""

    distance = {cell: 0 for cell in destination_cells if cell in fg.corridor_cells}
    queue = deque(sorted(distance))
    while queue:
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


def _shortest_cell_path(fg: FlightGraph, start: Cell, destination: Cell) -> tuple[Cell, ...] | None:
    """Return one deterministic directed BFS path inside the frozen corridor."""

    if start == destination:
        return None  # A column must contain a real lateral hop.
    predecessor: dict[Cell, Cell | None] = {start: None}
    queue = deque((start,))
    while queue:
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
        # Import locally to keep the pricing hot-path module surface small.
        from ...volumes import exit_radius

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


def _pricing_tolerance(params: Any) -> float:
    for name in ("reduced_cost_tol", "pricing_tol"):
        if hasattr(params, name):
            value = float(getattr(params, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"params.{name} must be finite and non-negative")
            return value
    return 1e-9


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


def _shortest_seed(fg: FlightGraph, cfg: SimConfig) -> Column | None:
    """Certify the best deterministic BFS seed without expanding the time DAG."""

    view = DualView({}, cfg)
    candidates: list[_Candidate] = []
    for origin_lane_idx, start, lane_steps in _origin_options(fg):
        for destination, destination_lane_indices in sorted(_destination_options(fg).items()):
            path = _shortest_cell_path(fg, start, destination)
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
            return canonical[1]
    return None


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
) -> tuple[float, Column | None]:
    if len(fg.levels) != 1:
        raise NotImplementedError(
            "colgen v1 pricing supports a single flight level; multi-level pricing is planned"
        )
    if not math.isfinite(pi_f):
        raise ValueError(f"flight-row dual must be finite, got {pi_f!r}")

    destination_options = _destination_options(fg)
    if not destination_options:
        return -math.inf, None
    distances = _distance_to_destinations(fg, set(destination_options))
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
    for lane_idx, _cell, _lane_steps in _origin_options(fg):
        lane_dist = None if lane_idx is None else fg.origin_lanes[lane_idx].dist
        origin_leg_by_lane[lane_idx] = _fold_leg_s(
            fg.request.origin,
            fg.origin_terminal,
            lane_dist,
            cfg,
        )
    for departure_step in departure_steps:
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
        for lane_idx, cell, lane_steps in _origin_options(fg):
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
            origin_paid_rows = dual_view.active_claims(start_claims)
            score = ground_score - origin_leg_by_lane[lane_idx] - dual_view.claim_cost(start_claims)
            label = _Label(score, departure_step, lane_idx, (cell,), origin_paid_rows)
            recent = (cell,)
            key = (cell, recent, origin_paid_rows, None)
            layer = layers.setdefault(start_step, {})
            if _prefer(label, layer.get(key)):
                layer[key] = label

    candidates: list[_Candidate] = []
    for step in range(fg.min_step, fg.max_step + 1):
        layer = layers.get(step)
        if not layer:
            continue
        for (cell, recent, origin_paid_rows, first_hop), label in sorted(
            layer.items(),
            key=lambda item: (item[0][0], item[0][1], item[1].tie_key),
        ):
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
                # Only ground delay and already-unioned row prices are
                # irrevocable.  Lateral clock spent so far is not a delay
                # lower bound for terminal flights: another lattice hop can
                # replace destination folding distance and leave canonical
                # detour unchanged.  Future detour is therefore bounded by
                # zero here; ``column_claims`` computes it exactly at sinks.
                optimistic_rc = (
                    benefit - pi_f - ground_delay - paid_duals + dual_view.max_negative_credit
                )
                if optimistic_rc < incumbent[0] - _RECOMPUTE_EPS:
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
    for candidate in candidates:
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
    view = duals if isinstance(duals, DualView) else DualView(duals, cfg)
    forbidden = frozenset(key if isinstance(key, RowKey) else RowKey(key) for key in forbidden_rows)
    benefit = _benefit(params)
    incumbent: tuple[float, Column] | None = None
    try:
        seed = seed_column(fg, cfg)
    except ValueError:
        # A deterministic shortest-path seed is an acceleration, not a
        # feasibility precondition.  The full DAG may still find a usable
        # path when the first geodesic fails a path-dependent wall check; a
        # truly disconnected graph simply falls through to ``None`` below.
        seed = None
    if seed is not None and seed.claims.isdisjoint(forbidden):
        seed_rc = benefit - seed.delay_s - view.claim_cost(seed.claims) - float(pi_f)
        incumbent = seed_rc, seed
        # With no priced capacity or repair exclusions, every extra hold or
        # loop only adds delay.  Avoid expanding the long default ground-clock
        # horizon merely to rediscover the canonical seed.
        if not view.has_active_duals and not forbidden:
            if require_improving and seed_rc <= _pricing_tolerance(params):
                return seed_rc, None
            return seed_rc, seed
    reduced_cost, column = _best_column(
        fg,
        view,
        float(pi_f),
        cfg,
        benefit,
        forbidden,
        seed=False,
        incumbent=incumbent,
    )
    if column is None or (require_improving and reduced_cost <= _pricing_tolerance(params)):
        return reduced_cost, None
    return reduced_cost, column


def seed_column(fg: FlightGraph, cfg: SimConfig) -> Column:
    """Return a deterministic, dual-free shortest-delay feasible seed.

    Only the nominal departure is considered.  With zero row prices the DAG
    minimizes ground hold, lateral hops, and endpoint fold/snap legs; shortest
    paths never exercise the short-revisit restriction, so this is the plan's
    unconstrained shortest-path seed while retaining the canonical wall and
    detour gates.
    """

    direct = _shortest_seed(fg, cfg)
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
    )
    if column is None:
        raise ValueError(f"flight {fg.request.flight_id} has no feasible seed column")
    return column


__all__ = ["DualView", "price_flight", "seed_column"]
