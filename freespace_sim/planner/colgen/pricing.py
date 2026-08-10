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
import sys
import threading
import time
import heapq
import itertools
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
from .objective import DELAY_MODEL, CostModel, cost_model
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

# Optional compiled search.  The kernel imports Numba unconditionally (as
# ``astar_kernel`` does), so the availability guard lives here at the host import
# site, mirroring ``AStarPlanner.__init__``.
try:
    from . import dp_kernel as _dp_kernel
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _dp_kernel = None

_kernel_fallback_warned = False


# Outcome of the most recent `_best_column` call in THIS process, overwritten each
# time.  A module global rather than a return value because the compiled gate sits
# several frames below the pricing entry point and threading a stats object through
# would touch every signature on the way.  Read it immediately after `price_flight`
# returns, in the same process -- `pricing_pool` does exactly that, which is the only
# way to attribute a straggler in the parallel sweep (the workers are where the time
# goes, and parent-side patches cannot reach them).
_LAST_KERNEL_STATS: dict = {}


def _warn_kernel_fallback() -> None:
    """One stderr line, once per process, when the compiled pricing DP is absent.

    The fallback is the reference oracle, so nothing downstream notices a missing
    kernel except the clock -- which is exactly how a large slowdown can hide in a
    whole sweep (see the same guard in ``astar.py``).
    """

    global _kernel_fallback_warned
    if _kernel_fallback_warned:
        return
    _kernel_fallback_warned = True
    print(
        "WARNING: compiled colgen pricing kernel unavailable (numba import failed) -- "
        "using the pure-Python reference DP. Results are identical. Fix: run via plain "
        "`uv run` (numba is in tool.uv default-groups) or `uv sync`.",
        file=sys.stderr,
    )


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

    def shift_terms(
        self, claims: Iterable[RowKey]
    ) -> tuple[tuple[tuple[Any, ...], int], ...]:
        """Pre-resolve a claim set for repeated integer-clock translation.

        Returns ``(key_prefix, base_step)`` pairs, where ``key_prefix + (step,)`` is
        a lookup key for :attr:`_duals`.  Two facts make this exact rather than
        merely close:

        * ``RowKey`` is a plain ``tuple`` subclass built by
          ``tuple.__new__(cls, ("cell", q, r, level, step))``, so an ordinary tuple
          with those contents has the same hash and compares equal -- the dict
          lookup is identical, it just skips ``RowKey.__new__``'s validation and its
          four ``operator.index`` calls.
        * Rows whose *resource* carries no dual at any step are dropped.  They would
          contribute exactly ``0.0`` at every translation, and adding exact zeros
          cannot change an ``fsum``.

        Translating a claim set is injective (every row's step moves by the same
        delta), so no two rows can collide into one and summing the terms is exactly
        summing the translated set.
        """

        terms: list[tuple[tuple[Any, ...], int]] = []
        for row in claims:
            if row.kind == "cell":
                q, r = row.cell_coord
                if (row.cell_coord, row.level) not in self._cell:
                    continue
                terms.append((("cell", q, r, row.level), row.step))
            else:
                if row.terminal_id not in self._terminal:
                    continue
                terms.append((("term", row.terminal_id), row.step))
        return tuple(terms)

    def shifted_claim_cost(
        self, terms: tuple[tuple[tuple[Any, ...], int], ...], delta_steps: int
    ) -> float:
        """Cost of a pre-resolved claim set translated by ``delta_steps``.

        Bit-identical to ``claim_cost(_shift_claims(claims, delta_steps))``: same
        values, same exact ``fsum``.
        """

        duals = self._duals
        return math.fsum(
            duals.get(prefix + (step + delta_steps,), 0.0) for prefix, step in terms
        )

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


def _rows_hit_forbidden(
    claims: Iterable[RowKey],
    forbidden: AbstractSet[RowKey],
    *,
    delta_steps: int = 0,
) -> bool:
    """``not _shift_claims(claims, delta_steps).isdisjoint(forbidden)``, without the set.

    The translated claim set is built purely to be asked a membership question, and building
    it costs one ``RowKey.__new__`` per row -- four ``operator.index`` calls each -- plus a
    frozenset.  ``RowKey`` is ``tuple.__new__(cls, ("cell", q, r, level, step))``, so the plain
    tuple written here hashes and compares equal to the ``RowKey`` it mirrors and the lookup is
    identical; and the walk stops at the first hit instead of translating every remaining row.

    ``tuple.__getitem__`` rather than the ``.kind`` / ``.step`` / ``.level`` properties, because
    each of those re-reads ``.kind`` to choose its index -- several attribute lookups per row on
    the hottest path in the greedy heuristic.
    """
    if not forbidden:
        return False
    if delta_steps == 0:
        return any(row in forbidden for row in claims)
    for row in claims:
        if tuple.__getitem__(row, 0) == "cell":
            key: tuple[Any, ...] = (
                "cell",
                tuple.__getitem__(row, 1),
                tuple.__getitem__(row, 2),
                tuple.__getitem__(row, 3),
                tuple.__getitem__(row, 4) + delta_steps,
            )
        else:
            key = ("term", tuple.__getitem__(row, 1), tuple.__getitem__(row, 2) + delta_steps)
        if key in forbidden:
            return True
    return False


def _visit_hits_forbidden(
    cell: Cell,
    level: int,
    visit_step: int,
    offsets: tuple[int, int],
    forbidden: AbstractSet[RowKey],
) -> bool:
    """``not _visit_claims(...).isdisjoint(forbidden)`` without materializing the window.

    Same identity as :func:`_rows_hit_forbidden`, applied to the per-visit cell window: this
    is called once per relaxed arc, so the frozenset it replaces was the single largest source
    of ``RowKey`` construction in ``find_feasible_column``.
    """
    if not forbidden:
        return False
    q, r = cell
    return any(
        ("cell", q, r, level, row_step) in forbidden
        for row_step in visit_rows(visit_step, offsets)
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


def _distance_lower_bound(cell: Cell, destination_cells: AbstractSet[Cell]) -> int:
    """Admissible spatial distance without forcing a reverse graph traversal."""

    return min(hg.hex_distance(cell, destination) for destination in destination_cells)


def _shortest_cell_path(
    fg: FlightGraph,
    start: Cell,
    destination: Cell,
    *,
    deadline: float | None = None,
) -> tuple[Cell, ...] | None:
    """Return one deterministic shortest path using lazy, cached A* expansion."""

    if start == destination:
        return None  # A column must contain a real lateral hop.
    start_path = (start,)
    # (f=g+h, h, path, cell): preferring smaller h on equal f follows one
    # promising geodesic instead of breadth-expanding every equal-f cell.
    frontier: list[tuple[int, int, tuple[Cell, ...], Cell]] = [
        (hg.hex_distance(start, destination), hg.hex_distance(start, destination), start_path, start)
    ]
    best: dict[Cell, tuple[int, tuple[Cell, ...]]] = {start: (0, start_path)}
    while frontier:
        _check_deadline(deadline)
        _estimate, _remaining, path, cell = heapq.heappop(frontier)
        distance = len(path) - 1
        if best.get(cell) != (distance, path):
            continue
        if cell == destination:
            return path
        for neighbour in sorted(fg.outgoing_neighbors(cell)):
            next_distance = distance + 1
            next_path = (*path, neighbour)
            previous = best.get(neighbour)
            if previous is not None and (
                previous[0] < next_distance
                or (previous[0] == next_distance and previous[1] <= next_path)
            ):
                continue
            best[neighbour] = next_distance, next_path
            remaining = hg.hex_distance(neighbour, destination)
            heapq.heappush(
                frontier,
                (next_distance + remaining, remaining, next_path, neighbour),
            )
    return None


def _path_claims(
    fg: FlightGraph,
    cfg: SimConfig,
    label: _Label,
    dest_lane_idx: int | None,
    endpoint_cache: dict[tuple[bool, int, int], frozenset[RowKey]] | None = None,
    visit_cache: dict[tuple[Cell, int], frozenset[RowKey]] | None = None,
) -> frozenset[RowKey]:
    """Build the intended row union cheaply before canonical certification.

    Both caches memoize pure functions across a batch of candidates, and both exist
    because that batch overlaps far more than it looks:

    * ``endpoint_cache`` -- the two endpoint row sets depend only on
      ``(origin, step, timing_steps)``, of which a batch has very few distinct values.
    * ``visit_cache`` -- sink proposals share path prefixes and corridor start steps,
      so ``(cell, visit_step)`` repeats heavily.  Measured at 90% redundant (1106
      calls, 114 distinct) on one ranking pass.
    """

    del dest_lane_idx  # The selected destination lane changes geometry, not dwell row membership.
    origin_lane_steps = (
        0 if label.origin_lane_idx is None else fg.origin_lanes[label.origin_lane_idx].steps
    )
    corridor_start = label.departure_step + fg.takeoff_steps[0] + origin_lane_steps

    def endpoints(origin: bool, step: int, timing_steps: int) -> frozenset[RowKey]:
        if endpoint_cache is None:
            return _endpoint_claims(
                fg, cfg, origin=origin, step=step, timing_steps=timing_steps
            )
        key = (origin, step, timing_steps)
        cached = endpoint_cache.get(key)
        if cached is None:
            cached = _endpoint_claims(
                fg, cfg, origin=origin, step=step, timing_steps=timing_steps
            )
            endpoint_cache[key] = cached
        return cached

    claims = set(endpoints(True, label.departure_step, 0))
    offsets = derive_cell_window(cfg)
    for offset, cell in enumerate(label.path):
        visit_step = corridor_start + offset
        if visit_cache is None:
            claims.update(_visit_claims(cell, 0, visit_step, offsets))
        else:
            key = (cell, visit_step)
            cached = visit_cache.get(key)
            if cached is None:
                cached = _visit_claims(cell, 0, visit_step, offsets)
                visit_cache[key] = cached
            claims.update(cached)
    arrival_step = corridor_start + label.hops
    claims.update(endpoints(False, arrival_step, label.hops))
    return frozenset(claims)


def _path_delay_s(
    fg: FlightGraph, cfg: SimConfig, label: _Label, model: CostModel = DELAY_MODEL
) -> float:
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
    return model.evaluate(
        ground_s=(label.departure_step - fg.base_step) * cfg.dt_s,
        air_detour_s=detour / cfg.nominal_speed_mps,
    )


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
    model: CostModel = DELAY_MODEL,
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
        return model.evaluate(ground_s=ground_delay_s)
    flown_time_lb = origin_fold_s + (hops + remaining_hops) * dt_s + destination_fold_s
    return model.evaluate(
        ground_s=ground_delay_s, air_detour_s=max(0.0, flown_time_lb - reference_time_s)
    )


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
    forbidden_rows: AbstractSet[RowKey],
    incumbent: tuple[float, Column] | None,
    *,
    deadline: float | None = None,
    model: CostModel = DELAY_MODEL,
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
    # The scan asks each translation only for its dual cost and a disjointness verdict, then
    # throws the row set away; every translation that is not the winner was materialized for
    # nothing.  ``shift_terms`` resolves the seed's rows once so each step is a handful of
    # dict lookups instead of rebuilding a frozenset of RowKey objects.
    terms = duals.shift_terms(seed.claims)
    best_delta: int | None = None
    best_delay_s = 0.0
    for departure_step in range(seed.departure_step + 1, latest_departure + 1):
        _check_deadline(deadline)
        delta_steps = departure_step - seed.departure_step
        # Row exclusions used to force this scan back onto `_shift_claims` + `claim_cost`,
        # rebuilding a frozenset of RowKeys per translation purely to answer a disjointness
        # question.  That fallback was the dominant cost of the greedy heuristic, which always
        # supplies `forbidden_rows` (its saturated set).  Both halves are now set-free.
        if _rows_hit_forbidden(seed.claims, forbidden_rows, delta_steps=delta_steps):
            continue
        dual_cost = duals.shifted_claim_cost(terms, delta_steps)
        # A pure clock translation adds GROUND delay only -- the spatial path, and hence
        # the air term, is invariant -- so this is the one weight that applies.
        delay_s = seed.delay_s + model.ground_weight * (delta_steps * cfg.dt_s)
        reduced_cost = model.reduced_cost(
            benefit=benefit, cost=delay_s, dual_cost=dual_cost, pi_f=pi_f
        )
        if best is None or reduced_cost > best[0] + _SCORE_EPS:
            best = (reduced_cost, None)  # column built once, after the scan
            best_delta = delta_steps
            best_delay_s = delay_s
        if dual_cost == 0.0 and duals.max_negative_credit == 0.0:
            break
    if best_delta is not None:
        best = (
            best[0],
            Column(
                flight_id=seed.flight_id,
                departure_step=seed.departure_step + best_delta,
                level=seed.level,
                origin_lane_idx=seed.origin_lane_idx,
                dest_lane_idx=seed.dest_lane_idx,
                cell_path=seed.cell_path,
                delay_s=best_delay_s,
                claims=_shift_claims(seed.claims, best_delta),
            ),
        )
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
    forbidden_rows: AbstractSet[RowKey],
    model: CostModel = DELAY_MODEL,
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
    intent = column_to_intent(provisional, fg.request, cfg)
    if intent.status is not IntentStatus.ACCEPTED:
        return None
    try:
        claims = column_claims(provisional, fg, cfg, _intent=intent)
    except (ValueError, NotImplementedError):
        return None
    if not claims.isdisjoint(forbidden_rows):
        return None

    # ``column_claims`` already translated the path as its canonical budget
    # gate.  Translate once more to set the objective from precisely the same
    # metric fields exposed to callers; this is only done for top sink labels.
    exact_delay = model.evaluate(
        ground_s=intent.ground_delay_s,
        air_hold_s=intent.air_hold_s,
        air_detour_s=intent.air_detour_m / cfg.nominal_speed_mps,
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
    reduced_cost = model.reduced_cost(
        benefit=benefit, cost=exact_delay, dual_cost=duals.claim_cost(claims), pi_f=pi_f
    )
    return reduced_cost, column


def _shortest_seed_columns(
    fg: FlightGraph,
    cfg: SimConfig,
    *,
    deadline: float | None = None,
    model: CostModel = DELAY_MODEL,
) -> tuple[Column, ...]:
    """Certify and cache one best deterministic shortest-delay seed column.

    Endpoint lane pairs are ordered by an admissible delay lower bound.  Once a
    canonical seed beats the remaining bounds, no further spatial path is
    generated.  Exact ties are still explored for deterministic tie-breaking.

    The result is cached on the graph, keyed on ``model``.  It used to assume one model
    per graph -- true inside a solve, since the objective is fixed and graphs are built
    per solve -- but the assumption was enforced only by convention, and violating it
    returned the first model's seed with no error.  Anything comparing two objectives on
    one graph, which is the natural way to write such a comparison, silently got the same
    answer twice.
    """

    cache = fg._search_cache
    with cache.lock:
        if cache.seed_columns is not None and cache.seed_model == model:
            return cache.seed_columns

        view = DualView({}, cfg)
        reference_time_s = enroute_reference_m(
            fg.request.origin,
            fg.request.dest,
            fg.origin_terminal,
            fg.dest_terminal,
            cfg,
        ) / cfg.nominal_speed_mps
        specs: list[
            tuple[float, int, int, int, int | None, Cell, int, int | None]
        ] = []
        destination_options = _destination_options(fg)
        for origin_lane_idx, start, lane_steps in _origin_options(fg):
            origin_lane_dist = (
                None if origin_lane_idx is None else fg.origin_lanes[origin_lane_idx].dist
            )
            origin_fold = _fold_leg_s(
                fg.request.origin,
                fg.origin_terminal,
                origin_lane_dist,
                cfg,
            )
            if fg.origin_terminal is None:
                origin_fold_lb = origin_fold
                origin_exact = True
            else:
                origin_fold_lb, origin_exact = _terminal_fold_leg_s(
                    fg.request.origin,
                    fg.origin_terminal,
                    start,
                    cfg,
                )
            for destination, destination_lane_indices in sorted(destination_options.items()):
                hops_lb = hg.hex_distance(start, destination)
                for dest_lane_idx in destination_lane_indices:
                    dest_lane_dist = (
                        None if dest_lane_idx is None else fg.dest_lanes[dest_lane_idx].dist
                    )
                    destination_fold = _fold_leg_s(
                        fg.request.dest,
                        fg.dest_terminal,
                        dest_lane_dist,
                        cfg,
                    )
                    if fg.dest_terminal is None:
                        destination_fold_lb = destination_fold
                        destination_exact = True
                    else:
                        destination_fold_lb, destination_exact = _terminal_fold_leg_s(
                            fg.request.dest,
                            fg.dest_terminal,
                            destination,
                            cfg,
                        )
                    delay_lb = _arc_delay_lower_bound_s(
                        ground_delay_s=0.0,
                        origin_fold_s=origin_fold_lb,
                        hops=0,
                        remaining_hops=hops_lb,
                        destination_fold_s=destination_fold_lb,
                        reference_time_s=reference_time_s,
                        dt_s=cfg.dt_s,
                        folding_exact=(
                            reference_time_s > 0.0 and origin_exact and destination_exact
                        ),
                        model=model,
                    )
                    specs.append(
                        (
                            delay_lb,
                            hops_lb,
                            -1 if origin_lane_idx is None else origin_lane_idx,
                            -1 if dest_lane_idx is None else dest_lane_idx,
                            origin_lane_idx,
                            start,
                            lane_steps,
                            dest_lane_idx,
                        )
                    )

        specs.sort(key=lambda spec: spec[:4])
        best: Column | None = None
        best_key: tuple[Any, ...] | None = None
        # The path-independent arc oracle must admit an edge whenever any of
        # its endpoint-tag roles is safe.  Consequently a complete path can
        # still fail the canonical role-specific wall gate.  If that happens,
        # one deterministic shortest path does not prove that another path of
        # the same length is absent; retain the valid seed, but do not use it
        # later as a global minimum-delay certificate.
        unresolved_shortest_path = False
        for (
            delay_lb,
            _hops_lb,
            _origin_tie,
            _dest_tie,
            origin_lane_idx,
            start,
            lane_steps,
            dest_lane_idx,
        ) in specs:
            _check_deadline(deadline)
            if best is not None and delay_lb > best.delay_s + _RECOMPUTE_EPS:
                break
            destination = (
                fg.dest_cell
                if dest_lane_idx is None
                else fg.dest_lanes[dest_lane_idx].cell
            )
            path = _shortest_cell_path(fg, start, destination, deadline=deadline)
            if path is None:
                continue
            arrival_step = fg.base_step + fg.takeoff_steps[0] + lane_steps + len(path) - 1
            if arrival_step > fg.max_step:
                continue
            label = _Label(0.0, fg.base_step, origin_lane_idx, path, frozenset())
            delay_s = _path_delay_s(fg, cfg, label, model)
            candidate = _Candidate(-delay_s, delay_s, label, dest_lane_idx)
            canonical = _canonical_candidate(
                candidate,
                fg,
                view,
                0.0,
                cfg,
                0.0,
                _EMPTY_ROWS,
                model,
            )
            if canonical is None:
                unresolved_shortest_path = True
                continue
            column = canonical[1]
            key = (
                column.delay_s,
                len(column.cell_path) - 1,
                column.departure_step,
                -1 if column.origin_lane_idx is None else column.origin_lane_idx,
                -1 if column.dest_lane_idx is None else column.dest_lane_idx,
                column.cell_path,
            )
            if best_key is None or key < best_key:
                best = column
                best_key = key

        result = () if best is None else (best,)
        cache.seed_columns = result
        cache.seed_model = model
        cache.seed_delay_certified = best is not None and not unresolved_shortest_path
        return result


def _shortest_seed(
    fg: FlightGraph,
    cfg: SimConfig,
    *,
    deadline: float | None = None,
    model: CostModel = DELAY_MODEL,
) -> Column | None:
    """Certify the best deterministic BFS seed without expanding the time DAG."""

    columns = _shortest_seed_columns(fg, cfg, deadline=deadline, model=model)
    return None if not columns else columns[0]


def _topology_for(fg: FlightGraph, cfg: SimConfig):
    """Return the flight's flat-array topology, building it at most once.

    Built on first *compiled pricing* use, never at graph construction: the
    zero-dual seed shortcut in :func:`price_flight` lets most flights in a batch
    skip the DAG entirely, and they must not pay to drain the lazy arc oracle.
    """

    from . import dp_prepare

    cache = fg._search_cache
    with cache.lock:
        topology = cache.topology
        if topology is None:
            topology = dp_prepare.prepare_topology(fg, cfg)
            cache.topology = topology
        return topology



_MAX_KERNEL_ATTEMPTS = 3


def _certify_candidates(
    result,
    fg: FlightGraph,
    cfg: SimConfig,
    dual_view: DualView,
    pi_f: float,
    benefit: float,
    forbidden_rows: AbstractSet[RowKey],
    incumbent: tuple[float, Column] | None,
    *,
    deadline: float | None = None,
    want_residual: bool = False,
    model: CostModel = DELAY_MODEL,
):
    """Turn kernel proposals into a certified column, in the reference's order.

    Returns ``best`` normally, or ``(best, residual)`` when ``want_residual``.  The
    residual is the largest reduced-cost upper bound left unexamined -- the kernel's
    own, raised by anything the tier-1 break skipped -- and is what licenses the
    caller's optimality claim.
    """

    destination_options = _destination_options(fg)
    endpoint_cache: dict[tuple[bool, int, int], frozenset[RowKey]] = {}
    visit_cache: dict[tuple[Cell, int], frozenset[RowKey]] = {}
    provisional: list[_Candidate] = []
    residual = result.remaining_rc_upper_bound
    best_provisional: float | None = None
    # Tier 1 -- rank by a PROVISIONAL reduced cost built from real path geometry,
    # not by the kernel's admissible bound.  That bound is deliberately loose (it
    # prices the row union by a max, which can undershoot the true union), so
    # ranking by it would leave the tier-2 early exit unable to fire and turn
    # certification into the bottleneck.  This mirrors ``consider_sink``.
    #
    # Proposals arrive sorted by ``rc_upper_bound`` descending, and that bound
    # dominates the provisional cost below, so once it falls under the best
    # provisional score already seen the rest cannot win and need not be priced --
    # measured at 62% of them.  Whatever is skipped is folded into ``residual``.
    for proposal in result.candidates:
        if (
            best_provisional is not None
            and proposal.rc_upper_bound <= best_provisional - _RECOMPUTE_EPS
        ):
            residual = max(residual, proposal.rc_upper_bound)
            break
        for dest_lane_idx in destination_options.get(proposal.cell_path[-1], ()):
            label = _Label(
                0.0, proposal.departure_step, proposal.origin_lane_idx,
                proposal.cell_path, _EMPTY_ROWS,
            )
            claims = _path_claims(
                fg, cfg, label, dest_lane_idx, endpoint_cache, visit_cache
            )
            if not claims.isdisjoint(forbidden_rows):
                continue
            delay_s = _path_delay_s(fg, cfg, label, model)
            reduced_cost = model.reduced_cost(
                benefit=benefit,
                cost=delay_s,
                dual_cost=dual_view.claim_cost(claims),
                pi_f=pi_f,
            )
            if best_provisional is None or reduced_cost > best_provisional:
                best_provisional = reduced_cost
            provisional.append(_Candidate(reduced_cost, delay_s, label, dest_lane_idx))
    provisional.sort(key=lambda candidate: (-candidate.reduced_cost, candidate.tie_key))

    # Tier 2 -- exact certification, in the reference's order and with its break.
    best = incumbent
    for index, candidate in enumerate(provisional):
        if index % 128 == 0:
            _check_deadline(deadline)
        if best is not None and candidate.reduced_cost < best[0] - _RECOMPUTE_EPS:
            break
        canonical = _canonical_candidate(
            candidate, fg, dual_view, pi_f, cfg, benefit, forbidden_rows, model
        )
        if canonical is None:
            continue
        if best is None:
            best = canonical
            continue
        rc, column = canonical
        if rc > best[0] + _SCORE_EPS or (
            abs(rc - best[0]) <= _SCORE_EPS
            and _column_sort_key(column) < _column_sort_key(best[1])
        ):
            best = canonical
    return (best, residual) if want_residual else best


def _best_column_compiled(
    fg: FlightGraph,
    dual_view: DualView,
    pi_f: float,
    cfg: SimConfig,
    benefit: float,
    forbidden_rows: AbstractSet[RowKey],
    *,
    seed: bool,
    incumbent: tuple[float, Column] | None,
    deadline: float | None,
    model: CostModel = DELAY_MODEL,
) -> tuple[tuple[float, Column] | None, bool]:
    """Run the compiled DP and certify its proposals.

    Returns ``(best, proved)``.  ``proved`` means the kernel's residual bound rules
    out everything it did not return, so the answer is optimal with respect to the
    same dominance the reference applies.  When it is ``False`` the caller runs the
    reference search, warm-started with whatever was certified here.
    """

    from . import dp_prepare

    _LAST_KERNEL_STATS.clear()
    _LAST_KERNEL_STATS["hops"] = fg.shortest_hops
    _LAST_KERNEL_STATS["steps"] = fg.max_step - fg.min_step + 1
    topology = _topology_for(fg, cfg)
    if topology.unsupported_reason is not None:
        _LAST_KERNEL_STATS["unsupported"] = topology.unsupported_reason
        return incumbent, False
    _LAST_KERNEL_STATS["n_cells"] = topology.n_cells

    cutoff = incumbent[0] if incumbent is not None else None
    # The kernel filters only CELL rows.  Terminal and endpoint exclusions stay in
    # Python: `_canonical_candidate` re-tests every proposal against the full
    # `forbidden_rows` set below.  That is sound for the optimality proof because the
    # kernel then searches a SUPERSET of the feasible space, so its residual bound
    # dominates the true one -- it can propose an infeasible sink, never hide a
    # feasible one.
    forbidden_pack = dp_prepare.prepare_forbidden(forbidden_rows, topology)
    duals = dp_prepare.prepare_duals(dual_view, topology)
    variants = dp_prepare.prepare_variants(
        fg, cfg, dual_view, topology, seed=seed,
        benefit=benefit, pi_f=pi_f, cost_cutoff=cutoff, model=model,
    )
    if variants.n_variants == 0:
        # Every start option was ruled out by ground delay alone.  With a cutoff
        # that is a proof; without one it just means an empty problem.
        return incumbent, cutoff is not None

    cancel_flag = np.zeros(1, dtype=np.uint8)
    timer = None
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise PricingTimeout("column pricing reached its wall-clock deadline")
        timer = threading.Timer(remaining, lambda: cancel_flag.__setitem__(0, 1))
        timer.daemon = True
        timer.start()
    try:
        # The reference tightens its own cutoff mid-sweep (``consider_sink``
        # certifies improving sinks and feeds them straight to the bound), which is
        # what collapses its search on hard flights.  The kernel cannot do that
        # in-flight, so it is given the same advantage across runs: if a pass
        # exhausts its label pool, certify the sinks it *did* reach and re-run with
        # that stronger cutoff.  Each retry starts from a strictly better bound, so
        # it explores strictly less.  Without this a hard flight burned the whole
        # kernel budget, returned nothing, and paid for the reference DP as well.
        # Bootstrap the cutoff from ONE departure variant before searching them all.
        #
        # The sweep advances by step, so a sink only exists once some label has reached
        # `start_step + hops`.  Spread across many variants the pool fills long before
        # that, and the retry loop below is then useless -- it needs candidates to
        # tighten with and there are none.  Measured on a captured 500-flight subproblem
        # (fid=3176, 83 surviving variants): the full search burned 32,274,881 labels and
        # returned ZERO candidates, while the single best-scoring variant reached 1,352
        # sinks in 3,521,185 labels and lifted the cutoff from rc=112.0 to rc=140.8.
        # The full pass then needed 294,693 labels and 0.12s. End to end 33.1s -> 4.5s
        # for the identical rc=144.000 column, at a ninth of the peak labels.
        #
        # Answer-neutral: the bootstrap only ever produces a *certified* incumbent, and
        # pruning against an attainable score cannot discard anything strictly better.
        # If it finds nothing, the cutoff is unchanged and only its own cost is lost.
        if variants.n_variants > 1:
            order = np.argsort(-np.asarray(variants.score), kind="stable")
            boot = _dp_kernel.search_dag(
                topology, duals, dp_prepare.restrict_variants(variants, order[:1]),
                cfg=cfg, benefit=benefit, pi_f=pi_f, cost_cutoff=cutoff,
                seed=seed, forbidden=forbidden_pack, cancel_flag=cancel_flag,
                model=model,
            )
            if boot.status == _dp_kernel.FB_CANCELLED:
                raise PricingTimeout("column pricing reached its wall-clock deadline")
            if boot.candidates:
                improved = _certify_candidates(
                    boot, fg, cfg, dual_view, pi_f, benefit, forbidden_rows, incumbent,
                    model=model,
                )
                if improved is not None and (
                    cutoff is None or improved[0] > cutoff + _SCORE_EPS
                ):
                    incumbent = improved
                    cutoff = improved[0]
                    # Re-filter: a tighter cutoff prunes departure variants outright,
                    # which is most of the win (83 -> 4 on the captured flight).
                    variants = dp_prepare.prepare_variants(
                        fg, cfg, dual_view, topology, seed=seed,
                        benefit=benefit, pi_f=pi_f, cost_cutoff=cutoff, model=model,
                    )
                    if variants.n_variants == 0:
                        return incumbent, True

        for _attempt in range(_MAX_KERNEL_ATTEMPTS):
            result = _dp_kernel.search_dag(
                topology, duals, variants,
                cfg=cfg, benefit=benefit, pi_f=pi_f, cost_cutoff=cutoff,
                seed=seed, forbidden=forbidden_pack, cancel_flag=cancel_flag,
                model=model,
            )
            _LAST_KERNEL_STATS["attempts"] = _LAST_KERNEL_STATS.get("attempts", 0) + 1
            _LAST_KERNEL_STATS["status"] = result.status_name
            _LAST_KERNEL_STATS["regrow"] = (
                _LAST_KERNEL_STATS.get("regrow", 0) + result.regrow
            )
            _LAST_KERNEL_STATS["labels"] = max(
                _LAST_KERNEL_STATS.get("labels", 0), result.n_labels
            )
            _LAST_KERNEL_STATS["candidates"] = len(result.candidates)
            if result.status == _dp_kernel.FB_CANCELLED:
                raise PricingTimeout("column pricing reached its wall-clock deadline")
            if result.ok or not result.candidates:
                break
            improved = _certify_candidates(
                result, fg, cfg, dual_view, pi_f, benefit, forbidden_rows, incumbent,
                model=model,
            )
            if improved is None or (cutoff is not None and improved[0] <= cutoff + _SCORE_EPS):
                break  # no tighter bound to retry with; let the reference finish
            incumbent = improved
            cutoff = improved[0]
    finally:
        if timer is not None:
            timer.cancel()
    if not result.ok:
        return incumbent, False

    # Tier 1 -- rank by a PROVISIONAL reduced cost built from real path geometry,
    # not by the kernel's admissible bound.  The bound is deliberately loose (it
    # prices the row union by a max, which can undershoot the true union), so
    # ranking by it leaves the tier-2 early exit unable to fire and turns
    # certification into the bottleneck.  This mirrors ``consider_sink``.
    best, residual = _certify_candidates(
        result, fg, cfg, dual_view, pi_f, benefit, forbidden_rows, incumbent,
        deadline=deadline, want_residual=True, model=model,
    )
    proved = best is not None and residual <= best[0] + _RECOMPUTE_EPS
    return best, proved


def _column_sort_key(column: Column) -> tuple[Any, ...]:
    """The reference's canonical column ordering (pricing's final tie-break)."""

    return (
        len(column.cell_path) - 1,
        column.departure_step,
        -1 if column.origin_lane_idx is None else column.origin_lane_idx,
        -1 if column.dest_lane_idx is None else column.dest_lane_idx,
        column.cell_path,
    )


def _best_column(
    fg: FlightGraph,
    dual_view: DualView,
    pi_f: float,
    cfg: SimConfig,
    benefit: float,
    forbidden_rows: AbstractSet[RowKey],
    *,
    seed: bool,
    incumbent: tuple[float, Column] | None = None,
    deadline: float | None = None,
    model: CostModel = DELAY_MODEL,
) -> tuple[float, Column | None]:
    _check_deadline(deadline)
    # Row exclusions no longer force the reference: the kernel carries a packed set of
    # forbidden cell rows (dp_prepare.prepare_forbidden), so the repair path
    # (solver.py's re-pricing against saturated rows) is compiled too.
    if _dp_kernel is not None and len(fg.levels) == 1:
        certified, proved = _best_column_compiled(
            fg, dual_view, pi_f, cfg, benefit, forbidden_rows,
            seed=seed, incumbent=incumbent, deadline=deadline, model=model,
        )
        if proved:
            return certified if certified is not None else (-math.inf, None)
        _LAST_KERNEL_STATS["reference_fallback"] = True
        # Not proved: fall through to the reference search, warm-started with
        # whatever the kernel did certify so its pruning starts tighter.
        incumbent = certified if certified is not None else incumbent
    elif _dp_kernel is None:
        _warn_kernel_fallback()
    if len(fg.levels) != 1:
        raise NotImplementedError(
            "colgen v1 pricing supports a single flight level; multi-level pricing is planned"
        )
    if not math.isfinite(pi_f):
        raise ValueError(f"flight-row dual must be finite, got {pi_f!r}")

    destination_options = _destination_options(fg)
    if not destination_options:
        return -math.inf, None
    destination_cells = frozenset(destination_options)
    distance_cache: dict[Cell, int] = {}

    def remaining_distance(cell: Cell) -> int:
        cached = distance_cache.get(cell)
        if cached is None:
            cached = _distance_lower_bound(cell, destination_cells)
            distance_cache[cell] = cached
        return cached

    paid_cell_row_cache: dict[
        frozenset[RowKey], tuple[frozenset[Cell], dict[tuple[Cell, int], float]]
    ] = {}

    def _paid_cell_rows(
        paid: frozenset[RowKey],
    ) -> tuple[frozenset[Cell], dict[tuple[Cell, int], float]]:
        """Index one origin endpoint's already-paid rows for visit-window lookup.

        Only single-level cell rows can recur in a later visit window; terminal
        rows never can, so they are dropped rather than searched.  Keyed by the
        frozenset itself because it is constant along a label's whole trajectory
        and shared by every label from the same start option.
        """

        entry = paid_cell_row_cache.get(paid)
        if entry is None:
            lookup: dict[tuple[Cell, int], float] = {}
            for row in paid:
                if row.kind == "cell" and row.level == 0:
                    lookup[(row.cell_coord, row.step)] = dual_view.row_cost(row)
            entry = frozenset(cell for cell, _step in lookup), lookup
            paid_cell_row_cache[paid] = entry
        return entry

    origin_options = _origin_options(fg)
    # ``detour_slack_hops`` sizes the spatial ellipse; it is not a route-length
    # budget.  Ordinary pricing may spend the clock slack on W-separated wide
    # loops (the network's en-route-waiting lever), subject to the canonical
    # detour gate at the sink.  A zero-dual seed has no reason to loop, so its
    # tighter hop limit avoids exploring value-tied cyclic walks.
    #
    # ``pricing_slack_hops`` is the optional budget on exactly that looping, and it is
    # None by default -- in which case the limit is ``max_step - min_step + 1``, which
    # no label can reach, because every arc advances the clock exactly one step.  The
    # uncapped path is therefore a provable no-op, not a behaviour change.  Mirrors the
    # kernel's ``pricing_hop_limit``; the two must agree or parity breaks.
    if seed:
        hop_limit = fg.shortest_hops + fg.detour_slack_hops
    elif fg.pricing_slack_hops is None:
        hop_limit = fg.max_step - fg.min_step + 1
    else:
        hop_limit = fg.shortest_hops + fg.pricing_slack_hops
    if hop_limit < 1:
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
            model=model,
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
            distance_to_go = remaining_distance(cell)
            start_step = departure_step + fg.takeoff_steps[0] + lane_steps
            if start_step >= fg.max_step:
                continue
            if start_step + distance_to_go > fg.max_step:
                continue
            if distance_to_go > hop_limit:
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
                distance_to_go,
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

    def consider_sink(label: _Label, step: int, cell: Cell) -> None:
        """Register one role-certified final arc before label dominance."""

        nonlocal incumbent
        hops = label.hops
        for dest_lane_idx in destination_options[cell]:
            destination_claims = _endpoint_claims(
                fg,
                cfg,
                origin=False,
                step=step,
                timing_steps=hops,
            )
            if not destination_claims.isdisjoint(forbidden_rows):
                continue
            # Reference path: deliberately left uncached, so it stays the verbatim
            # oracle the compiled path is measured and certified against.
            claims = _path_claims(fg, cfg, label, dest_lane_idx)
            if not claims.isdisjoint(forbidden_rows):
                continue
            delay_s = _path_delay_s(fg, cfg, label, model)
            reduced_cost = model.reduced_cost(
                benefit=benefit,
                cost=delay_s,
                dual_cost=dual_view.claim_cost(claims),
                pi_f=pi_f,
            )
            candidate = _Candidate(
                reduced_cost,
                delay_s,
                label,
                dest_lane_idx,
            )
            candidates.append(candidate)
            # A certified improving sink tightens the safe lower-bound pruning
            # for every later time layer.
            if incumbent is None or reduced_cost > incumbent[0] + _SCORE_EPS:
                canonical = _canonical_candidate(
                    candidate,
                    fg,
                    dual_view,
                    pi_f,
                    cfg,
                    benefit,
                    forbidden_rows,
                    model,
                )
                if canonical is not None and (
                    incumbent is None or canonical[0] > incumbent[0] + _SCORE_EPS
                ):
                    incumbent = canonical

    for step in range(fg.min_step, fg.max_step + 1):
        _check_deadline(deadline)
        layer = layers.pop(step, None)
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
            if hops >= hop_limit or step + 1 > fg.max_step:
                continue
            paid_cells, paid_cell_rows = _paid_cell_rows(origin_paid_rows)
            if incumbent is not None:
                ground_delay = (label.departure_step - fg.base_step) * cfg.dt_s
                origin_leg = origin_leg_by_lane[label.origin_lane_idx]
                # ``label.score`` is the negative sum of ground delay, flown
                # time so far, and the de-duplicated duals paid so far.
                paid_duals = -label.score - ground_delay - origin_leg - hops * cfg.dt_s
                distance_to_go = remaining_distance(cell)
                # The endpoint-aware envelope lower-bounds the positive price
                # of the eventual row union without double-counting overlaps.
                # It also handles exact RC ties in the same hops-first order as
                # candidates.
                if not completion_can_compete(
                    label.departure_step,
                    label.origin_lane_idx,
                    hops + distance_to_go,
                    paid_duals,
                    paid_duals_exact=False,
                ):
                    continue
            for neighbour in fg.outgoing_neighbors(cell):
                if neighbour in recent[:revisit_depth]:
                    continue
                first_arc = hops == 0
                finish_allowed = (
                    neighbour in destination_options
                    and fg.hop_allowed_for_role(
                        cell,
                        neighbour,
                        first=first_arc,
                        last=True,
                    )
                )
                continuation_allowed = fg.hop_allowed_for_role(
                    cell,
                    neighbour,
                    first=first_arc,
                    last=False,
                )
                if not finish_allowed and not continuation_allowed:
                    continue
                distance_to_go = remaining_distance(neighbour)
                next_step = step + 1
                if next_step + distance_to_go > fg.max_step:
                    continue
                if hops + 1 + distance_to_go > hop_limit:
                    continue
                # Same per-arc guard as the feasible search: a set built only to be tested.
                if _visit_hits_forbidden(neighbour, 0, next_step, offsets, forbidden_rows):
                    continue
                # Price the visit window from ``DualView``'s prefix sums instead of
                # materializing its ``RowKey`` set.  Building that set to sum it was
                # measured at ~44% of this search (718k ``RowKey.__new__`` calls, each
                # running ``operator.index`` four times, for one number).  Rows the
                # origin endpoint already paid must not be charged twice; that overlap
                # is confined to the endpoint's own cells, so the guard below is a
                # miss on essentially every arc.
                visit_cost = dual_view.visit_cost(neighbour, 0, next_step)
                if neighbour in paid_cells:
                    visit_cost -= math.fsum(
                        price
                        for row_step in visit_rows(next_step, offsets)
                        if (price := paid_cell_rows.get((neighbour, row_step))) is not None
                    )
                next_recent = (neighbour, *recent[: state_history_depth - 1])
                next_label = _Label(
                    label.score - cfg.dt_s - visit_cost,
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
                if finish_allowed:
                    consider_sink(next_label, next_step, neighbour)
                if not continuation_allowed:
                    continue
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
            model,
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


def find_feasible_column(
    fg: FlightGraph,
    cfg: SimConfig,
    *,
    forbidden_rows: AbstractSet[RowKey] = _EMPTY_ROWS,
    improve_below_delay_s: float | None = None,
    deadline: float | None = None,
    model: CostModel = DELAY_MODEL,
) -> Column | None:
    """Best-first incumbent search over the lazy space-time topology.

    This is deliberately an incumbent heuristic, not the reduced-cost oracle:
    it runs only after the first LP and therefore cannot contribute to a global
    pricing bound.  Static adjacency and the certified seed are reused, while
    row exclusions and labels remain call-local.  A raw-hex delay lower bound
    orders the frontier; the exact canonical gate judges every sink.  When
    ``improve_below_delay_s`` is supplied, the first certified strict
    improvement may be returned.  That early exit is useful for incumbent
    construction but is intentionally unavailable to formal pricing.
    """

    _check_deadline(deadline)
    if cfg != fg._cfg:
        raise ValueError("feasible search requires the SimConfig used to build the flight graph")
    if improve_below_delay_s is not None:
        improve_below_delay_s = float(improve_below_delay_s)
        if not math.isfinite(improve_below_delay_s):
            raise ValueError("improvement delay threshold must be finite")
    forbidden = forbidden_rows
    destination_options = _destination_options(fg)
    if not destination_options:
        return None
    destination_cells = frozenset(destination_options)
    # ``_best_column`` has memoized this since it was written; this search never got the
    # same treatment and paid for it: measured at 3,940,131 calls driving 24,760,154
    # ``hex_distance`` calls (6.3 destination lanes each), ~9s of a 32s stage.  The value
    # depends only on the cell and the fixed destination set, and the corridor holds a few
    # thousand cells against millions of arc relaxations, so this is nearly all hits.
    distance_cache: dict[Cell, int] = {}

    def remaining_distance(cell: Cell) -> int:
        cached = distance_cache.get(cell)
        if cached is None:
            cached = _distance_lower_bound(cell, destination_cells)
            distance_cache[cell] = cached
        return cached

    origin_options = _origin_options(fg)
    offsets = derive_cell_window(cfg)
    revisit_depth = offsets[1] - offsets[0]
    state_history_depth = max(2, revisit_depth)
    track_first_hop = bool(fg.static_walls and fg.origin_terminal is not None)
    view = DualView({}, cfg)

    reference_m = enroute_reference_m(
        fg.request.origin,
        fg.request.dest,
        fg.origin_terminal,
        fg.dest_terminal,
        cfg,
    )
    reference_time_s = reference_m / cfg.nominal_speed_mps
    origin_fold_lb: dict[int | None, tuple[float, bool]] = {}
    for lane_idx, cell, _lane_steps in origin_options:
        if fg.origin_terminal is None:
            origin_fold_lb[lane_idx] = (
                _fold_leg_s(fg.request.origin, None, None, cfg),
                True,
            )
        else:
            origin_fold_lb[lane_idx] = _terminal_fold_leg_s(
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
        for destination in destination_options:
            fold_s, retained = _terminal_fold_leg_s(
                fg.request.dest,
                fg.dest_terminal,
                destination,
                cfg,
            )
            destination_folds.append(fold_s)
            destination_fold_exact &= retained
        destination_fold_lb = min(destination_folds)

    def delay_bound(
        departure_step: int,
        lane_idx: int | None,
        hops: int,
        remaining_hops: int,
    ) -> float:
        origin_fold_s, origin_exact = origin_fold_lb[lane_idx]
        return _arc_delay_lower_bound_s(
            ground_delay_s=(departure_step - fg.base_step) * cfg.dt_s,
            origin_fold_s=origin_fold_s,
            hops=hops,
            remaining_hops=remaining_hops,
            destination_fold_s=destination_fold_lb,
            reference_time_s=reference_time_s,
            dt_s=cfg.dt_s,
            model=model,
            folding_exact=(
                reference_m > 1e-9 and origin_exact and destination_fold_exact
            ),
        )

    best_column: Column | None = None
    try:
        seed = seed_column(fg, cfg, deadline=deadline, model=model)
    except ValueError:
        seed = None
    if seed is not None:
        if seed.claims.isdisjoint(forbidden):
            best_column = seed
            if (
                improve_below_delay_s is not None
                and seed.delay_s < improve_below_delay_s - _SCORE_EPS
            ):
                return seed
        shifted = _shifted_seed_incumbent(
            seed,
            fg,
            view,
            0.0,
            cfg,
            0.0,
            forbidden,
            None if best_column is None else (-best_column.delay_s, best_column),
            deadline=deadline,
            model=model,
        )
        if shifted is not None and (
            best_column is None or shifted[1].delay_s < best_column.delay_s - _SCORE_EPS
        ):
            best_column = shifted[1]
            if (
                improve_below_delay_s is not None
                and best_column.delay_s < improve_below_delay_s - _SCORE_EPS
            ):
                return best_column

    def column_key(column: Column) -> tuple[Any, ...]:
        return (
            column.delay_s,
            len(column.cell_path) - 1,
            column.departure_step,
            -1 if column.origin_lane_idx is None else column.origin_lane_idx,
            -1 if column.dest_lane_idx is None else column.dest_lane_idx,
            column.cell_path,
        )

    counter = itertools.count()
    frontier: list[
        tuple[
            float,
            int,
            int,
            int,
            tuple[Cell, ...],
            int,
            int,
            tuple[Cell, ...],
            tuple[Cell, Cell] | None,
            _Label,
        ]
    ] = []
    best_state_path: dict[tuple[Any, ...], tuple[Cell, ...]] = {}
    for departure_step in range(fg.base_step, fg.latest_departure_step + 1):
        _check_deadline(deadline)
        ground_delay_s = (departure_step - fg.base_step) * cfg.dt_s
        if best_column is not None and ground_delay_s > best_column.delay_s + _RECOMPUTE_EPS:
            break
        origin_claims = _endpoint_claims(
            fg,
            cfg,
            origin=True,
            step=departure_step,
            timing_steps=0,
        )
        if not origin_claims.isdisjoint(forbidden):
            continue
        for lane_idx, cell, lane_steps in origin_options:
            start_step = departure_step + fg.takeoff_steps[0] + lane_steps
            remaining = remaining_distance(cell)
            if start_step >= fg.max_step or start_step + remaining > fg.max_step:
                continue
            # `origin_claims` was proven disjoint from `forbidden` immediately above and is
            # invariant across this loop, so the old `(origin_claims | visit_claims)` union
            # re-tested it once per lane and allocated two sets to do it.  The union is
            # equivalent to testing the visit window alone.
            if _visit_hits_forbidden(cell, 0, start_step, offsets, forbidden):
                continue
            path = (cell,)
            recent = path
            label = _Label(0.0, departure_step, lane_idx, path, frozenset())
            bound = delay_bound(departure_step, lane_idx, 0, remaining)
            if best_column is not None and bound > best_column.delay_s + _RECOMPUTE_EPS:
                continue
            state_key = (start_step, cell, recent, departure_step, lane_idx, None)
            best_state_path[state_key] = path
            heapq.heappush(
                frontier,
                (
                    bound,
                    remaining,
                    departure_step,
                    -1 if lane_idx is None else lane_idx,
                    path,
                    next(counter),
                    start_step,
                    recent,
                    None,
                    label,
                ),
            )

    while frontier:
        _check_deadline(deadline)
        (
            bound,
            _estimated_hops,
            departure_step,
            _lane_tie,
            path,
            _serial,
            step,
            recent,
            first_hop,
            label,
        ) = heapq.heappop(frontier)
        if best_column is not None and bound > best_column.delay_s + _RECOMPUTE_EPS:
            break
        state_key = (
            step,
            path[-1],
            recent,
            departure_step,
            label.origin_lane_idx,
            first_hop,
        )
        if best_state_path.get(state_key) != path:
            continue
        cell = path[-1]
        hops = len(path) - 1
        if hops >= 1 and cell in destination_options:
            for dest_lane_idx in destination_options[cell]:
                destination_claims = _endpoint_claims(
                    fg,
                    cfg,
                    origin=False,
                    step=step,
                    timing_steps=hops,
                )
                if not destination_claims.isdisjoint(forbidden):
                    continue
                claims = _path_claims(fg, cfg, label, dest_lane_idx)
                if not claims.isdisjoint(forbidden):
                    continue
                delay_s = _path_delay_s(fg, cfg, label, model)
                canonical = _canonical_candidate(
                    _Candidate(-delay_s, delay_s, label, dest_lane_idx),
                    fg,
                    view,
                    0.0,
                    cfg,
                    0.0,
                    forbidden,
                    model,
                )
                if canonical is None:
                    continue
                candidate = canonical[1]
                if best_column is None or column_key(candidate) < column_key(best_column):
                    best_column = candidate
                    if (
                        improve_below_delay_s is not None
                        and candidate.delay_s < improve_below_delay_s - _SCORE_EPS
                    ):
                        return candidate

        if step + 1 > fg.max_step:
            continue
        for neighbour in fg.outgoing_neighbors(cell):
            if neighbour in recent[:revisit_depth]:
                continue
            next_step = step + 1
            remaining = remaining_distance(neighbour)
            if next_step + remaining > fg.max_step:
                continue
            # Once per relaxed arc, and the set was built only to be tested: measured at
            # 3,764,765 calls and 11.76s of this search's 37.72s.
            if _visit_hits_forbidden(neighbour, 0, next_step, offsets, forbidden):
                continue
            next_path = (*path, neighbour)
            next_recent = (neighbour, *recent[: state_history_depth - 1])
            next_first_hop = (
                ((cell, neighbour) if first_hop is None else first_hop)
                if track_first_hop
                else None
            )
            next_bound = delay_bound(
                departure_step,
                label.origin_lane_idx,
                hops + 1,
                remaining,
            )
            if best_column is not None and next_bound > best_column.delay_s + _RECOMPUTE_EPS:
                continue
            next_key = (
                next_step,
                neighbour,
                next_recent,
                departure_step,
                label.origin_lane_idx,
                next_first_hop,
            )
            previous_path = best_state_path.get(next_key)
            if previous_path is not None and previous_path <= next_path:
                continue
            best_state_path[next_key] = next_path
            next_label = _Label(
                0.0,
                departure_step,
                label.origin_lane_idx,
                next_path,
                frozenset(),
            )
            heapq.heappush(
                frontier,
                (
                    next_bound,
                    hops + 1 + remaining,
                    departure_step,
                    -1 if label.origin_lane_idx is None else label.origin_lane_idx,
                    next_path,
                    next(counter),
                    next_step,
                    next_recent,
                    next_first_hop,
                    next_label,
                ),
            )

    return best_column


def price_flight(
    fg: FlightGraph,
    duals: Mapping[RowKey | tuple[Any, ...], float] | DualView,
    pi_f: float,
    cfg: SimConfig,
    params: Any,
    *,
    forbidden_rows: AbstractSet[RowKey] = _EMPTY_ROWS,
    require_improving: bool = True,
    known_column: Column | None = None,
    deadline: float | None = None,
) -> tuple[float, Column | None]:
    """Return the best positive-reduced-cost column for one flight.

    ``known_column`` is a column the caller already holds for this flight -- the
    restricted master's current selection.  Its reduced cost under the current duals is
    a *proven achievable* score, so it is a valid pruning cutoff, and a far better one
    than the shortest-path seed this function otherwise builds for itself.  That matters
    enormously: on a captured 500-flight subproblem the seed gave a cutoff of rc=112
    against an optimum of rc=144, and closing that 32-unit gap collapsed the search from
    **32,274,881 labels to 73,541** -- 439x -- with the departure-variant prefilter going
    from 503 surviving variants to 1.  The label pool, its regrowth ladder and the
    reference-DP fallback are all downstream of this one number.

    Passing it never changes the answer, only the work: pruning against a score that is
    actually attainable cannot discard anything strictly better.  A column equal to the
    one supplied is reported as no column at all, so the caller is not handed back what
    it already has and cannot mistake it for pricing progress.

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
    if cfg != fg._cfg:
        raise ValueError("pricing requires the SimConfig used to build the flight graph")
    if deadline is not None:
        deadline = float(deadline)
        if not math.isfinite(deadline):
            raise ValueError("pricing deadline must be finite")
    _check_deadline(deadline)
    pi_value = float(pi_f)
    if not math.isfinite(pi_value):
        raise ValueError(f"flight-row dual must be finite, got {pi_f!r}")
    view = duals if isinstance(duals, DualView) else DualView(duals, cfg)
    forbidden = forbidden_rows
    benefit = _benefit(params)
    # Resolved once per call and threaded down; every objective expression below and in
    # dp_prepare/dp_kernel derives from it.  See colgen.objective.
    model = cost_model(cfg, params)
    incumbent: tuple[float, Column] | None = None
    try:
        seed = seed_column(fg, cfg, deadline=deadline, model=model)
    except ValueError:
        # A deterministic shortest-path seed is an acceleration, not a
        # feasibility precondition.  The full DAG may still find a usable
        # path when the first geodesic fails a path-dependent wall check; a
        # truly disconnected graph simply falls through to ``None`` below.
        seed = None
    if seed is not None:
        if seed.claims.isdisjoint(forbidden):
            seed_dual_cost = view.claim_cost(seed.claims)
            seed_rc = model.reduced_cost(
                benefit=benefit, cost=seed.delay_s, dual_cost=seed_dual_cost, pi_f=pi_value
            )
            incumbent = seed_rc, seed
            # The seed is the globally minimum-delay column.  If it remains
            # feasible and pays no row price, non-negative duals and additional
            # exclusions can only make every alternative weakly worse.  This
            # locality check matters at batch scale: a dual or saturated row on the
            # other side of the region must not trigger this flight's full DAG.
            # Tiny negative backend-tolerance duals deliberately disable the
            # shortcut because another route could collect their credit.
            with fg._search_cache.lock:
                seed_delay_certified = fg._search_cache.seed_delay_certified
            if (
                seed_delay_certified
                and seed_dual_cost == 0.0
                and view.max_negative_credit == 0.0
            ):
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
            model=model,
        )
    # Fold in the caller's existing column, after the seed work so it can only tighten.
    # Its claims are re-checked against the exclusion set because the repair path may have
    # saturated a row the column occupies since it was filed.
    if known_column is not None and known_column.claims.isdisjoint(forbidden):
        known_rc = model.reduced_cost(
            benefit=benefit,
            cost=known_column.delay_s,
            dual_cost=view.claim_cost(known_column.claims),
            pi_f=pi_value,
        )
        if incumbent is None or known_rc > incumbent[0] + _SCORE_EPS:
            incumbent = (known_rc, known_column)
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
        model=model,
    )
    if column is None or (require_improving and reduced_cost <= _pricing_tolerance(params)):
        return reduced_cost, None
    if known_column is not None and column == known_column:
        # The best column IS the one the caller already holds.  Reporting it would let a
        # column the master already owns read as pricing progress and keep column
        # generation iterating on nothing.
        return reduced_cost, None
    return reduced_cost, column


def seed_column(
    fg: FlightGraph,
    cfg: SimConfig,
    *,
    deadline: float | None = None,
    model: CostModel = DELAY_MODEL,
) -> Column:
    """Return a deterministic, dual-free shortest-delay feasible seed.

    Only the nominal departure is considered.  With zero row prices the DAG
    minimizes ground hold, lateral hops, and endpoint fold/snap legs; shortest
    paths never exercise the short-revisit restriction, so this is the plan's
    unconstrained shortest-path seed while retaining the canonical wall and
    detour gates.
    """

    if cfg != fg._cfg:
        raise ValueError("seeding requires the SimConfig used to build the flight graph")
    if deadline is not None:
        deadline = float(deadline)
        if not math.isfinite(deadline):
            raise ValueError("seed deadline must be finite")
    _check_deadline(deadline)
    with fg._search_cache.lock:
        # Keyed on the model for the same reason as `_shortest_seed_columns`: a seed's
        # cost is the objective's verdict, so a cache hit under a different weighting
        # would answer the wrong question without saying so.
        if (
            fg._search_cache.seed_search_complete
            and fg._search_cache.seed_model == model
        ):
            cached = fg._search_cache.seed_columns or ()
            if cached:
                return cached[0]
            raise ValueError(f"flight {fg.request.flight_id} has no feasible seed column")

    direct = _shortest_seed(fg, cfg, deadline=deadline, model=model)
    if direct is not None:
        with fg._search_cache.lock:
            fg._search_cache.seed_search_complete = True
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
        model=model,
    )
    if column is None:
        with fg._search_cache.lock:
            fg._search_cache.seed_columns = ()
            fg._search_cache.seed_model = model
            fg._search_cache.seed_delay_certified = False
            fg._search_cache.seed_search_complete = True
        raise ValueError(f"flight {fg.request.flight_id} has no feasible seed column")
    with fg._search_cache.lock:
        fg._search_cache.seed_columns = (column,)
        fg._search_cache.seed_model = model
        # The fallback is a bounded feasibility search.  It is a valid seed,
        # but path-dependent wall tagging may mean it is not a globally
        # minimum-delay column, so exact pricing must not take the zero-dual
        # locality shortcut from it.
        fg._search_cache.seed_delay_certified = False
        fg._search_cache.seed_search_complete = True
    return column


__all__ = [
    "DualView",
    "PricingTimeout",
    "find_feasible_column",
    "price_flight",
    "seed_column",
]
