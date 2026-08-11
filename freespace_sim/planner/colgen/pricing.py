"""Reduced-cost pricing for the single-level column-generation network.

The master problem is written as a maximization problem.  A route therefore
has reduced cost ``M - delay_s - capacity_duals - flight_dual`` and is useful
only when that value is positive.  The dynamic program below uses the same
integer clock as :mod:`.network`; every returned path is subsequently passed
through :func:`.network.column_claims`, which is the authoritative geometry,
budget, and claim-membership gate.

This is the *reference* implementation: one exact, dominance-pruned label search
in plain Python, which is what makes it readable and what makes it the oracle a
compiled or parallel pricing path would have to agree with.  It is also the whole
cost of a solve -- pricing dominates the LP by orders of magnitude -- so expect
minutes per iteration at scenario scale.  See :func:`price_flight` for the entry
point and the module-level constants for the pruning envelopes.
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
from . import dp_prepare
from .network import (
    _MAX_ENDPOINT_CLAIMS,
    Cell,
    FlightGraph,
    RowKey,
    column_claims,
)
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
# A column has to beat zero reduced cost by this much to count as improving.  Mirrors
# ``solver._REDUCED_COST_TOL``; the two are the same threshold read from opposite sides of
# the call.  This used to be probed off ``params.reduced_cost_tol`` / ``params.pricing_tol``,
# neither of which ``ColGenParams`` has -- so it was always this value behind a lookup that
# read like a knob.
_IMPROVING_RC_TOL = 1e-9
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
    """Dwell rows one endpoint occupies, memoized on the graph.

    A pure function of ``(fg, origin, step, timing_steps)``, which is what makes the cache
    answer-neutral rather than a heuristic: the body below reads only the request's two
    endpoints, the two terminals, ``fg.levels`` and ``cfg`` scalars, all fixed for the
    graph's life -- and every caller reaches this through :func:`price_flight` or
    :func:`find_feasible_column`, which refuse a ``cfg`` that is not ``fg._cfg``.

    Worth the machinery because the redundancy is extreme rather than marginal: one sink
    proposal per ``(arrival step, hop count)`` pair means the reachable key space is far
    smaller than the number of sinks reaching it.  Measured on ``colgen_test``'s first 12
    flights over three iterations -- 286,705 calls, **1,511 distinct**, 99.5% redundant,
    and the solve went 38.85 s to 12.28 s with a bit-identical objective and column set.
    """

    key = (origin, step, timing_steps)
    cache = fg._search_cache
    with cache.lock:
        hit = cache.endpoint_claims.get(key)
        if hit is not None:
            cache.endpoint_claims.move_to_end(key)
            return hit
    # Built outside the lock: it is a pure function, so two threads racing to fill the same
    # key compute equal sets and either may win.
    value = _endpoint_claims_uncached(
        fg, cfg, origin=origin, step=step, timing_steps=timing_steps
    )
    with cache.lock:
        cache.endpoint_claims[key] = value
        cache.endpoint_claims.move_to_end(key)
        while len(cache.endpoint_claims) > _MAX_ENDPOINT_CLAIMS:
            cache.endpoint_claims.popitem(last=False)
    return value


def _endpoint_claims_uncached(
    fg: FlightGraph,
    cfg: SimConfig,
    *,
    origin: bool,
    step: int,
    timing_steps: int,
) -> frozenset[RowKey]:
    """Compute one endpoint's dwell rows.  The oracle :func:`_endpoint_claims` memoizes.

    Kept separate rather than inlined so a test can assert the cache reproduces it exactly
    over the whole reachable key space, instead of trusting the purity argument above.
    """

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
    hop, and the destination fold.  ``remaining_hops`` lower-bounds the unflown
    arcs, while independently minimizing the destination fold can only weaken
    that bound.  Customer snap legs have the same additive form.

    That remaining-hop count is plain hex distance (:func:`_distance_lower_bound`),
    NOT a reverse traversal of the corridor: it ignores walls and corridor shape,
    so it is a relaxation of the true remaining distance and the bound stays
    admissible.  A real reverse BFS would be tighter wherever the corridor is
    non-convex, at the cost of one traversal per graph.

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
        raise AttributeError("colgen pricing requires params.M") from exc
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


def _sink_certifier(
    fg: FlightGraph,
    dual_view: DualView,
    pi_f: float,
    cfg: SimConfig,
    benefit: float,
    forbidden_rows: AbstractSet[RowKey],
    model: CostModel = DELAY_MODEL,
    *,
    deadline: float | None = None,
):
    """``consider_sink``'s certification half, for a caller that found the sink elsewhere.

    Lifted out of :func:`_best_column` rather than reimplemented, because it is the thing
    that makes the reference's pruning *safe*: ``consider_sink`` assigns to a ``nonlocal
    incumbent``, so the cutoff improves mid-sweep and every later time layer prunes against
    a score that is **certified achievable** rather than merely proposed.  A cutoff taken
    from a provisional reduced cost can sit above the true optimum and discard it.

    The compiled search cannot do any of this in flight -- ``_path_delay_s`` reaches
    ``fold_corners_to_columns`` and ``column_to_intent`` reaches the whole geometry stack --
    so it pauses and calls this instead.  Returning the reference's own verdict is the
    point: the two forbidden-row gates, the provisional improvement test and the canonical
    improvement test are all here, in the same order, so the incumbent trajectory is the
    reference's whatever found the sink.

    ``label.score`` and ``label.origin_paid_rows`` are not read by anything downstream, so
    the reconstructed :class:`_Label` carries placeholders rather than pretending to a
    provenance it does not have.

    Returns the new ``(reduced_cost, Column)`` incumbent, or ``None`` when this sink does
    not improve on the one passed in.
    """

    def certify(
        incumbent: tuple[float, Column] | None,
        departure_step: int,
        origin_lane_idx: int | None,
        dest_lane_idx: int | None,
        arrival_step: int,
        path: tuple[Cell, ...],
    ) -> tuple[float, Column] | None:
        _check_deadline(deadline)
        label = _Label(0.0, departure_step, origin_lane_idx, tuple(path), _EMPTY_ROWS)
        destination_claims = _endpoint_claims(
            fg, cfg, origin=False, step=arrival_step, timing_steps=label.hops
        )
        if not destination_claims.isdisjoint(forbidden_rows):
            return None
        claims = _path_claims(fg, cfg, label, dest_lane_idx)
        if not claims.isdisjoint(forbidden_rows):
            return None
        delay_s = _path_delay_s(fg, cfg, label, model)
        reduced_cost = model.reduced_cost(
            benefit=benefit,
            cost=delay_s,
            dual_cost=dual_view.claim_cost(claims),
            pi_f=pi_f,
        )
        if incumbent is not None and reduced_cost <= incumbent[0] + _SCORE_EPS:
            return None
        canonical = _canonical_candidate(
            _Candidate(reduced_cost, delay_s, label, dest_lane_idx),
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
            return canonical
        return None

    return certify


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


def _column_sort_key(column: Column) -> tuple[Any, ...]:
    """The reference's canonical column ordering (pricing's final tie-break)."""

    return (
        len(column.cell_path) - 1,
        column.departure_step,
        -1 if column.origin_lane_idx is None else column.origin_lane_idx,
        -1 if column.dest_lane_idx is None else column.dest_lane_idx,
        column.cell_path,
    )


def _certify_candidates(
    candidates: list[_Candidate],
    fg: FlightGraph,
    dual_view: DualView,
    pi_f: float,
    cfg: SimConfig,
    benefit: float,
    forbidden_rows: AbstractSet[RowKey],
    model: CostModel = DELAY_MODEL,
    *,
    incumbent: tuple[float, Column] | None = None,
    deadline: float | None = None,
) -> tuple[float, Column] | None:
    """Tier 2: rank sink proposals and return the best certified column, or ``None``.

    This is ``_best_column``'s own post-loop block, and it lives here rather than inside it
    because the compiled Tier 1 produces the same proposals and must be ranked by the same
    rule. Two copies of a tie-break is how the two searches quietly start preferring
    different -- equally optimal -- columns, which is precisely what the parity gate
    measures and precisely the failure that does not raise.

    ``incumbent`` is the score already certified before ranking started: ``_best_column``
    passes the one ``consider_sink`` improved during its sweep, and the compiled host
    passes the one its pause-and-resume protocol arrived at. It is what the early
    ``_RECOMPUTE_EPS`` break is measured against, so passing ``None`` is not a neutral
    default -- it certifies far more candidates than the reference would.
    """

    if not candidates:
        return incumbent
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
        column_key = _column_sort_key(column)
        best_key = _column_sort_key(best_column)
        if rc > best_rc + _SCORE_EPS or (abs(rc - best_rc) <= _SCORE_EPS and column_key < best_key):
            best = canonical

    return best


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
    """Return the most negative-reduced-cost column, or ``(-inf, None)`` when none exists.

    This is the reference dynamic program: an exact, dominance-pruned label search
    over the flight's space-time DAG.  It defines what a correct answer *is* -- every
    later acceleration is measured against it -- so it is written for legibility over
    speed, and every pruning rule below is an argument that the discarded label cannot
    lead to a strictly better column.  ``incumbent`` warm-starts that pruning with an
    already-certified ``(reduced_cost, column)`` pair.
    """

    _check_deadline(deadline)
    if len(fg.levels) != 1:
        raise NotImplementedError(
            "colgen v1 pricing supports a single flight level; multi-level pricing is planned"
        )
    if not math.isfinite(pi_f):
        raise ValueError(f"flight-row dual must be finite, got {pi_f!r}")

    # Hoisted: the objective's weights are constant for the whole search, and `air_dt_s` in
    # particular sits in the arc relaxation below -- the innermost loop of the innermost
    # loop, run tens of millions of times per sweep.  Computing the product once is the
    # same float, so this is bit-identical, not an approximation.
    ground_weight = model.ground_weight
    air_weight = model.air_weight
    air_dt_s = air_weight * cfg.dt_s

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
    # The air-time ceiling (`params.max_air_overrun_hops`, resolved at graph build), and the
    # ONLY route-length bound this search has.  A label at the cap cannot be extended, and one
    # that could not reach a destination within it is dead on creation.  It also implies the
    # corridor -- a route within `max_air_hops` cannot touch a cell outside the ellipse of
    # radius `max_air_hops - shortest_hops`, which is why the two are one knob and why the
    # per-arc form below is a stronger test than corridor membership, not a redundant one.
    # Costs optimality -- a route needing more hops is unreachable even if it is the true
    # optimum -- and buys the thing that matters: labels never created.
    air_hop_limit = fg.max_air_hops
    if air_hop_limit < 1:
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
        # The horizon term is ~920 hops for an early colgen_test departure against a ceiling of
        # ~20, and every entry past the ceiling describes a completion the search cannot make.
        #
        # THE COST OF THOSE ENTRIES IS NOT THE POINT, and measuring it is how you talk yourself
        # out of this line.  Building one costs an `_endpoint_claims` call, and the `break`
        # below already contains THAT to `max_air_hops + 1`, so construction waste is one call
        # per (departure, lane).  The length of the result is what matters: it bounds
        # `completion_can_compete`'s scan (`range(first_hops, len(delay_lbs))`), which keeps a
        # label alive as soon as SOME hop count in range could beat the incumbent.  Entries past
        # the ceiling let a label survive on the strength of a completion the ceiling forbids.
        # Measured on the 50-flight harness in `ColGenParams`: 36.3M labels -> 16.7M, 2.17x, at
        # a byte-identical schedule and objective.
        #
        # Exact, not a heuristic.  A completion above `max_air_hops` cannot occur, so a label
        # that only competes there could never have won; `completion_can_compete` reading a
        # short envelope as "cannot compete" is the right verdict.  And no caller is cut off
        # early: every label satisfies `hops + remaining_distance <= max_air_hops` by the guards
        # in the two loops below, so `first_hops` is always inside the capped range.
        #
        # The `break` also cannot be relied on to do this.  It needs `delay_lb` monotone in
        # hops, which holds in the arc form and NOT in the ground-only fallback, where
        # `delay_lb` is constant and the break fires at `total_hops == 1` or never -- "never"
        # meaning the full horizon range.
        #
        # Keep the `min()` rather than the ceiling alone: the two are provably equal today
        # (single level, so `takeoff_steps[0]` is the max), but the horizon is a real bound and
        # should stay visible if `_graph_max_step` changes.
        max_total_hops = min(fg.max_step - corridor_start, fg.max_air_hops)
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
        # The label score is the search's ranking currency, and `_prefer` prunes on it, so
        # it has to be denominated in the OBJECTIVE -- not in raw seconds.  Within one time
        # layer `ground + flown` is invariant, so at unit weights the split between them
        # cannot change a comparison and the two currencies coincide exactly.  Under
        # `total_cost` it can: trading one step of ground for one hop of air is free in
        # seconds and worth 2*dt in cost, so an unweighted score calls two labels tied
        # where the objective strictly prefers one, and dominance then keeps whichever the
        # tie-break happened to reach first.
        ground_score = -ground_weight * (departure_step - fg.base_step) * cfg.dt_s
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
            if distance_to_go > air_hop_limit:
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
            score = (
                ground_score
                - air_weight * origin_leg_by_lane[lane_idx]
                - start_dual_cost
            )
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
        # Hoisted out of the lane loop below: the arrival dwell is a property of when the
        # flight lands, not of which lane it lands on, and `destination_options` groups
        # every lane sharing a cell -- so inside the loop this recomputed one answer per
        # lane.  Its `forbidden_rows` verdict is likewise lane-independent, hence `return`.
        destination_claims = _endpoint_claims(
            fg,
            cfg,
            origin=False,
            step=step,
            timing_steps=hops,
        )
        if not destination_claims.isdisjoint(forbidden_rows):
            return
        for dest_lane_idx in destination_options[cell]:
            # `_path_claims` is deliberately left uncached: this is the verbatim oracle any
            # later acceleration is measured and certified against, and a cache here would
            # be one more thing to keep honest.  (Its own `endpoint_cache`/`visit_cache`
            # parameters exist for callers that have made that argument; this one has not.)
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
            if hops >= air_hop_limit:
                continue
            if step + 1 > fg.max_step:
                continue
            paid_cells, paid_cell_rows = _paid_cell_rows(origin_paid_rows)
            if incumbent is not None:
                ground_delay = (label.departure_step - fg.base_step) * cfg.dt_s
                origin_leg = origin_leg_by_lane[label.origin_lane_idx]
                # ``label.score`` is the negative sum of the WEIGHTED ground delay and
                # flown time so far, plus the de-duplicated duals paid so far.  Duals are
                # already in reduced-cost currency and are never weighted, so inverting
                # the same decomposition recovers them exactly -- which is why the two
                # weights below must track the ones the score was built with.
                # Term by term, NOT `air_weight * (origin_leg + hops * dt)`: the grouped
                # form changes the association and so is not bit-identical to the
                # unweighted expression it replaced (measured: 62,673 of 200,000 random
                # draws differ by ~1e-13).  This function is the oracle any compiled
                # pricing path gets certified against, so its arithmetic has to be
                # reproducible exactly, not just to within a tolerance.
                paid_duals = (
                    -label.score
                    - ground_weight * ground_delay
                    - air_weight * origin_leg
                    - air_weight * (hops * cfg.dt_s)
                )
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
                if hops + 1 + distance_to_go > air_hop_limit:
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
                    label.score - air_dt_s - visit_cost,
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

    best = _certify_candidates(
        candidates,
        fg,
        dual_view,
        pi_f,
        cfg,
        benefit,
        forbidden_rows,
        model,
        incumbent=incumbent,
        deadline=deadline,
    )
    return (-math.inf, None) if best is None else best


_kernel_fallback_warned = False
# Same warn-once-per-process discipline as `_kernel_fallback_warned`, and for the same
# reason: a budget the pool could not grow into is silent, and per-flight it would be a
# thousand identical lines rather than a signal.  Two flags rather than one because the two
# conditions want different responses -- see `_warn_budget_growth`.
_kernel_restart_warned = False
_kernel_budget_warned = False

# Per-process tally of exact-pricing calls and how many could not be proved in the kernel.
#
# A fallback is a 3-4.5x slowdown that produces the RIGHT answer, so nothing downstream can
# notice it: the objective, the columns and the tests are all identical, only the clock
# moves.  `[[run-astar-with-compiled-extra]]` records the same failure mode costing a whole
# issue on the A* side.  Counting it is the only way a production run can report "the
# compiled path served 100% of pricing" rather than assume it.
#
# Per PROCESS, deliberately: under a worker pool each worker keeps its own tally and
# `pricing_pool` returns the delta per task, because a parent-side counter would report zero
# forever while every fallback happened somewhere else.
#
# `fell_back` alone cannot be acted on, because it conflates causes that call for opposite
# responses: numba missing (install it), a graph the packer refuses (a modelling limit), a
# deadline (raise the budget) and a label pool that could not grow far enough (a SCALE
# problem, and the only one that gets worse as the instance does).  The last two counters
# split that one out, and they are a pair on purpose -- `label_restarts` is the precursor
# and `budget_declined` is the failure, so a run whose restarts climb while declines stay 0
# is one that is paying for the pool it needs without losing the compiled path yet.
_KERNEL_STATS = {"priced": 0, "fell_back": 0, "label_restarts": 0, "budget_declined": 0}


def kernel_stats() -> dict[str, int]:
    """Snapshot this process's compiled-pricing tally."""

    return dict(_KERNEL_STATS)


def _warn_budget_growth(kernel, fg: FlightGraph, result) -> None:
    """Record and announce what the compiled search's label budget cost this flight.

    Two conditions, warned once per process each, because they call for different responses
    and reporting them as one number would hide the cheaper one behind the louder one:

    * **Restarts.** ``result.attempts > 1`` means a budget filled and the search re-ran from
      its first layer, throwing away every sink certification the previous attempt had
      already paid for.  The answer is unchanged -- a budget bounds work, never the search --
      so this is invisible except as a slow flight, which is exactly the shape
      ``[[run-astar-with-compiled-extra]]`` records costing a whole issue on the A* side.
      ``DagResult.attempts`` has always carried this ("the number to read when a flight is
      unexpectedly expensive"); nothing read it.
    * **Declines.** The search stopped without finishing, so this flight fell back to the
      Python reference -- same column, 3-4.5x the time.  The advice is split by cause,
      because the two that land here are opposites: a budget status means the pool hit
      :data:`~.dp_kernel.MAX_LABEL_CAPACITY`, which is a knob and not a wall, while
      ``FSUM_OVERFLOW`` means a partial expansion saturated and a SCORE would have been
      wrong -- telling someone to raise a ceiling in that case would be worse than saying
      nothing.

    Per process, so under a worker pool each worker warns for itself.  The counters are
    per process too, which means a parallel sweep's totals live in the workers -- the
    aggregate a parent already sees is ``kernel_fell_back``, which every decline here also
    increments one level up in :func:`price_flight`.
    """

    global _kernel_restart_warned, _kernel_budget_warned

    if result.attempts > 1:
        _KERNEL_STATS["label_restarts"] += result.attempts - 1
        if not _kernel_restart_warned:
            _kernel_restart_warned = True
            print(
                f"WARNING: compiled colgen pricing restarted its label pool "
                f"{result.attempts - 1}x on flight {fg.request.flight_id} (now "
                f"{result.budget[0]:,} labels) -- the answer is unchanged, the search is "
                f"not; a graph-cached budget means later iterations should not repeat this",
                file=sys.stderr,
            )
    if result.status in (kernel.STATUS_OK, kernel.STATUS_CANCELLED):
        return
    _KERNEL_STATS["budget_declined"] += 1
    if _kernel_budget_warned:
        return
    _kernel_budget_warned = True
    name = kernel.STATUS_NAMES.get(result.status, str(result.status))
    if result.status == kernel.STATUS_FSUM_OVERFLOW:
        remedy = (
            "this is a CORRECTNESS stop, not a budget one -- an exact-sum expansion "
            "saturated, so the kernel refused to report a score it could not stand behind"
        )
    else:
        remedy = (
            "raise dp_kernel.MAX_LABEL_CAPACITY / MAX_LOG2CAP if this instance is simply "
            "larger than they assume"
        )
    print(
        f"WARNING: compiled colgen pricing gave up on flight {fg.request.flight_id} with "
        f"{name} after {result.attempts} attempts ({result.budget[0]:,} labels, 2^"
        f"{result.budget[1]} states) -- falling back to the pure-Python reference search, "
        f"3-4.5x slower for the same column. {remedy}",
        file=sys.stderr,
    )


def _dp_kernel():
    """The compiled kernel module, or ``None`` when numba is unavailable.

    Warns once per process rather than per flight. The warning exists because the failure
    is silent and expensive: a sweep that quietly ran the reference everywhere looks
    exactly like a slow sweep, and ``[[run-astar-with-compiled-extra]]`` records the same
    lesson from the A* side, where a 5-7x regression stayed invisible for a whole issue.
    """

    global _kernel_fallback_warned
    try:
        from . import dp_kernel
    except ImportError:
        if not _kernel_fallback_warned:
            _kernel_fallback_warned = True
            print(
                "WARNING: compiled colgen pricing kernel unavailable (numba import "
                "failed) -- using the pure-Python reference search",
                file=sys.stderr,
            )
        return None
    return dp_kernel


def _dag_candidates(
    result,
    topology,
    fg: FlightGraph,
    cfg: SimConfig,
    dual_view: DualView,
    pi_f: float,
    benefit: float,
    forbidden_rows: AbstractSet[RowKey],
    model: CostModel,
    *,
    deadline: float | None = None,
) -> list[_Candidate]:
    """Price the compiled search's sinks into the ``_Candidate`` list Tier 2 ranks.

    This is ``consider_sink``'s pricing half -- the part the kernel cannot do, because
    ``_path_delay_s`` reaches ``fold_corners_to_columns`` and a ``np.linalg.norm(...).sum()``
    whose pairwise summation numba does not reproduce.

    Two memos that the reference declines and this is entitled to, both answer-identical:

    * ``_path_claims``' own ``endpoint_cache``/``visit_cache`` parameters, which exist for
      exactly this caller. Sink proposals share path prefixes and corridor start steps;
      measured 90% redundant (1106 calls, 114 distinct) on one ranking pass.
    * the provisional reduced cost itself, per LABEL rather than per lane. Neither
      ``_path_claims`` (which opens with ``del dest_lane_idx``) nor ``_path_delay_s`` takes
      the destination lane, so the reference recomputes an identical number once per lane
      on every multi-lane arrival.

    The two ``forbidden_rows`` gates are applied here rather than in the kernel, which is
    why the kernel is allowed to register a sink the reference rejects: reproducing the two
    endpoint span rules in numba to save work Tier 2 redoes anyway would be a second place
    for them to drift.
    """

    cells = list(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    endpoint_cache: dict[tuple[bool, int, int], frozenset[RowKey]] = {}
    visit_cache: dict[tuple[Cell, int], frozenset[RowKey]] = {}
    priced: dict[int, tuple[frozenset[RowKey], float, float] | None] = {}
    labels: dict[int, _Label] = {}
    candidates: list[_Candidate] = []

    for index, entry in enumerate(result.candidates):
        if index % 128 == 0:
            _check_deadline(deadline)
        departure_step, origin_lane, dest_lane, arrival_step, label_index = entry
        label = labels.get(label_index)
        if label is None:
            label = _Label(
                0.0,
                departure_step,
                None if origin_lane < 0 else origin_lane,
                tuple(cells[cell] for cell in result.paths[label_index]),
                _EMPTY_ROWS,
            )
            labels[label_index] = label
        entry_priced = priced.get(label_index, ...)
        if entry_priced is ...:
            destination_claims = _endpoint_claims(
                fg, cfg, origin=False, step=arrival_step, timing_steps=label.hops
            )
            claims = _path_claims(fg, cfg, label, None, endpoint_cache, visit_cache)
            if not destination_claims.isdisjoint(forbidden_rows) or not claims.isdisjoint(
                forbidden_rows
            ):
                entry_priced = None
            else:
                delay_s = _path_delay_s(fg, cfg, label, model)
                entry_priced = (
                    claims,
                    delay_s,
                    model.reduced_cost(
                        benefit=benefit,
                        cost=delay_s,
                        dual_cost=dual_view.claim_cost(claims),
                        pi_f=pi_f,
                    ),
                )
            priced[label_index] = entry_priced
        if entry_priced is None:
            continue
        _claims, delay_s, reduced_cost = entry_priced
        candidates.append(
            _Candidate(
                reduced_cost, delay_s, label, None if dest_lane < 0 else dest_lane
            )
        )
    return candidates


def _best_column_compiled(
    fg: FlightGraph,
    dual_view: DualView,
    pi_f: float,
    cfg: SimConfig,
    benefit: float,
    forbidden_rows: AbstractSet[RowKey],
    *,
    incumbent: tuple[float, Column] | None = None,
    deadline: float | None = None,
    model: CostModel = DELAY_MODEL,
) -> tuple[tuple[float, Column | None], bool]:
    """``_best_column`` over the compiled search: ``((reduced_cost, column), proved)``.

    ``proved`` is the whole contract. ``True`` means the compiled search ran to completion,
    and because it reproduces the reference's explored set exactly -- same roots, same
    completion gate, same mid-sweep incumbent, same dominance ties -- its sink set is the
    reference's, so ranking it gives the reference's column. ``False`` means the caller
    must run ``_best_column``: numba missing, a graph the packer refuses (multi-level, no
    reachable destination), or a budget the kernel could not grow into.

    Note what ``proved`` is NOT: a residual-bound argument over a superset search. PR #76
    needed one because its kernel searched more than the reference and certified
    separately. This one does not, which is why there is no bootstrap round and no
    ``label_limit`` ladder here -- ``price_dag`` grows its own budgets and either finishes
    or says it did not.

    A deadline is enforced two ways, because neither alone is enough: ``_check_deadline``
    between the Python stages, and a watchdog that sets the kernel's ``cancel`` flag, which
    it polls per time layer. An ``@njit(nogil=True)`` function cannot read a clock, and
    with geometric budget growth one call can run for minutes.
    """

    kernel = _dp_kernel()
    if kernel is None or len(fg.levels) != 1:
        return (-math.inf, None), False

    _check_deadline(deadline)
    topology, rows = dp_prepare.prepared_for(fg, cfg)
    if not (topology.ok and rows.ok):
        return (-math.inf, None), False

    duals = dp_prepare.prepare_duals(dual_view, fg, topology, rows)
    envelopes = dp_prepare.CompletionEnvelopes(
        fg,
        cfg,
        dual_view,
        benefit=benefit,
        pi_f=pi_f,
        model=model,
        forbidden_rows=forbidden_rows,
        incumbent=incumbent,
        deadline=deadline,
    )
    variants = dp_prepare.prepare_variants(
        fg,
        cfg,
        dual_view,
        topology,
        rows,
        benefit=benefit,
        pi_f=pi_f,
        cost_cutoff=None if incumbent is None else incumbent[0],
        model=model,
        forbidden_rows=forbidden_rows,
        envelopes=envelopes,
    )
    if not variants.ok:
        return (-math.inf, None), False
    pack = dp_prepare.prepare_forbidden(forbidden_rows, fg, rows, topology)

    with fg._search_cache.lock:
        budget = fg._search_cache.dag_budget
    sizes = {}
    if budget is not None:
        # What the last completed search on this graph needed.  The duals move every
        # iteration but the geometry does not, so this is a good estimate and never more
        # than an estimate -- `price_dag` still grows from it if this iteration explores
        # further.  A budget bounds work, never the search, so it cannot move an answer.
        sizes = dict(
            label_capacity=budget[0], log2cap=budget[1], candidate_capacity=budget[2]
        )

    cancel = np.zeros(1, dtype=np.uint8)
    watchdog = None
    if deadline is not None:
        _check_deadline(deadline)
        watchdog = threading.Timer(
            max(0.0, deadline - time.monotonic()), _cancel_search, args=(cancel,)
        )
        watchdog.daemon = True
        watchdog.start()
    try:
        result = kernel.price_dag(
            topology,
            rows,
            duals,
            variants,
            pack,
            air_weight=model.air_weight,
            dt_s=cfg.dt_s,
            benefit=benefit,
            pi_f=pi_f,
            envelopes=envelopes,
            certify=_sink_certifier(
                fg, dual_view, pi_f, cfg, benefit, forbidden_rows, model, deadline=deadline
            ),
            cancel=cancel,
            **sizes,
        )
    finally:
        if watchdog is not None:
            watchdog.cancel()

    # Before the status checks, so a search that restarted and THEN ran out of clock is
    # still recorded as having restarted -- `_check_deadline` below raises out of here.
    _warn_budget_growth(kernel, fg, result)

    if result.status == kernel.STATUS_CANCELLED:
        # The watchdog fired, so the deadline has passed.  Falling back to the reference
        # here would spend the caller's whole remaining budget re-running what was just
        # abandoned; `solver.py` turns this into `termination_reason = "time_limit"`.
        _check_deadline(deadline)
        return (-math.inf, None), False
    if result.status != kernel.STATUS_OK:
        return (-math.inf, None), False

    with fg._search_cache.lock:
        fg._search_cache.dag_budget = result.budget

    candidates = _dag_candidates(
        result,
        topology,
        fg,
        cfg,
        dual_view,
        pi_f,
        benefit,
        forbidden_rows,
        model,
        deadline=deadline,
    )
    best = _certify_candidates(
        candidates,
        fg,
        dual_view,
        pi_f,
        cfg,
        benefit,
        forbidden_rows,
        model,
        incumbent=result.incumbent,
        deadline=deadline,
    )
    return ((-math.inf, None) if best is None else best), True


def _cancel_search(flag) -> None:
    """Watchdog body: ask the kernel to stop at its next time layer."""

    flag[0] = 1


#: "The compiled feasible search declined."  A distinct sentinel because ``None`` is a real
#: answer from ``find_feasible_column`` -- a flight with no feasible column -- and returning
#: it for "numba is missing" would report an infeasible flight instead of falling back.
_UNPROVED = object()


def _feasible_compiled(
    fg: FlightGraph,
    cfg: SimConfig,
    *,
    forbidden: AbstractSet[RowKey],
    best_column: Column | None,
    improve_below_delay_s: float | None,
    origin_options,
    offsets,
    origin_fold_lb,
    destination_fold_lb: float,
    destination_fold_exact: bool,
    reference_time_s: float,
    reference_m: float,
    remaining_distance,
    delay_bound,
    column_key,
    view: DualView,
    deadline: float | None,
    model: CostModel,
):
    """``find_feasible_column``'s search, compiled; ``_UNPROVED`` when it cannot run.

    The **start loop stays in Python** and the kernel gets its result. That is not laziness:
    the guards need ``_endpoint_claims`` sets and the reference's own ``break`` on the
    incumbent's delay, and the loop runs a few hundred times against the search's hundreds
    of thousands of arc relaxations. Measured on a density flight: 141,553 arcs against 491
    endpoint-claim calls.

    Every sink still goes back to Python, because ``_canonical_candidate`` is the exact gate
    that judges them and it reaches the whole geometry stack. 115 of those against the same
    141,553 arcs is what makes pausing per sink affordable.
    """

    kernel = _dp_kernel()
    if kernel is None or len(fg.levels) != 1:
        return _UNPROVED
    topology, rows = dp_prepare.prepared_for(fg, cfg)
    if not (topology.ok and rows.ok):
        return _UNPROVED
    if topology.dest_lane_idx.size == 0:
        return _UNPROVED
    cell_index = {
        (int(q), int(r)): i
        for i, (q, r) in enumerate(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    }

    # Fold legs by lane, indexed `lane + 1` so the laneless origin (-1) lands at 0.
    n_lanes = 1 + max((idx for idx, _c, _s in origin_options if idx is not None), default=-1)
    lane_fold_s = [0.0] * (n_lanes + 1)
    lane_fold_exact = [0] * (n_lanes + 1)
    for lane_idx, _cell, _steps in origin_options:
        fold_s, exact = origin_fold_lb[lane_idx]
        slot = 0 if lane_idx is None else lane_idx + 1
        lane_fold_s[slot] = fold_s
        lane_fold_exact[slot] = 1 if (reference_m > 1e-9 and exact and destination_fold_exact) else 0

    # --- the reference's start loop, verbatim, emitting roots instead of heap entries ----
    roots: list[tuple[int, int, int, int, float, int]] = []
    for departure_step in range(fg.base_step, fg.latest_departure_step + 1):
        _check_deadline(deadline)
        ground_delay_s = (departure_step - fg.base_step) * cfg.dt_s
        if best_column is not None and ground_delay_s > best_column.delay_s + _RECOMPUTE_EPS:
            break
        origin_claims = _endpoint_claims(
            fg, cfg, origin=True, step=departure_step, timing_steps=0
        )
        if not origin_claims.isdisjoint(forbidden):
            continue
        for lane_idx, cell, lane_steps in origin_options:
            index = cell_index.get(cell)
            if index is None:
                return _UNPROVED
            start_step = departure_step + fg.takeoff_steps[0] + lane_steps
            remaining = remaining_distance(cell)
            if start_step >= fg.max_step or start_step + remaining > fg.max_step:
                continue
            if remaining > fg.max_air_hops:
                continue
            if _visit_hits_forbidden(cell, 0, start_step, offsets, forbidden):
                continue
            bound = delay_bound(departure_step, lane_idx, 0, remaining)
            if best_column is not None and bound > best_column.delay_s + _RECOMPUTE_EPS:
                continue
            roots.append(
                (
                    index,
                    start_step,
                    departure_step,
                    -1 if lane_idx is None else lane_idx,
                    bound,
                    remaining,
                )
            )
    if not roots:
        return best_column

    pack = dp_prepare.prepare_forbidden(forbidden, fg, rows, topology)
    state: dict[str, Any] = {"best": best_column}

    def certify(departure_step, origin_lane, dest_lane, step, hops, path):
        """``find_feasible_column``'s per-sink block, arm for arm (pricing.py:2337-2369)."""

        _check_deadline(deadline)
        label = _Label(0.0, departure_step, origin_lane, path, _EMPTY_ROWS)
        destination_claims = _endpoint_claims(
            fg, cfg, origin=False, step=step, timing_steps=hops
        )
        if not destination_claims.isdisjoint(forbidden):
            return None, False
        claims = _path_claims(fg, cfg, label, dest_lane)
        if not claims.isdisjoint(forbidden):
            return None, False
        delay_s = _path_delay_s(fg, cfg, label, model)
        canonical = _canonical_candidate(
            _Candidate(-delay_s, delay_s, label, dest_lane),
            fg,
            view,
            0.0,
            cfg,
            0.0,
            forbidden,
            model,
        )
        if canonical is None:
            return None, False
        candidate = canonical[1]
        current = state["best"]
        if current is not None and not (column_key(candidate) < column_key(current)):
            return None, False
        state["best"] = candidate
        stop = (
            improve_below_delay_s is not None
            and candidate.delay_s < improve_below_delay_s - _SCORE_EPS
        )
        return candidate.delay_s, stop

    status, stopped_early = kernel.feasible_dag(
        topology,
        rows,
        pack,
        roots,
        lane_fold_s=lane_fold_s,
        lane_fold_exact=lane_fold_exact,
        destination_fold_lb=destination_fold_lb,
        reference_time_s=reference_time_s,
        dt_s=cfg.dt_s,
        ground_weight=model.ground_weight,
        air_weight=model.air_weight,
        base_step=fg.base_step,
        offsets=offsets,
        incumbent_delay=None if best_column is None else best_column.delay_s,
        certify=certify,
    )
    del stopped_early  # the early exit already put its column in `state`
    if status != kernel.STATUS_OK:
        return _UNPROVED
    return state["best"]


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

    # The compiled best-first search, when the graph has a packing and numba is present.
    # `_UNPROVED` -- not `None` -- because `None` is a legitimate answer here: a flight with
    # no feasible column at all. Conflating the two would silently turn "the kernel declined"
    # into "this flight cannot fly".
    compiled = _feasible_compiled(
        fg,
        cfg,
        forbidden=forbidden,
        best_column=best_column,
        improve_below_delay_s=improve_below_delay_s,
        origin_options=origin_options,
        offsets=offsets,
        origin_fold_lb=origin_fold_lb,
        destination_fold_lb=destination_fold_lb,
        destination_fold_exact=destination_fold_exact,
        reference_time_s=reference_time_s,
        reference_m=reference_m,
        remaining_distance=remaining_distance,
        delay_bound=delay_bound,
        column_key=column_key,
        view=view,
        deadline=deadline,
        model=model,
    )
    if compiled is not _UNPROVED:
        return compiled

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
            # Same air-time ceiling the reduced-cost search applies.  Without it this
            # incumbent heuristic could hand the master a column pricing is forbidden to
            # reproduce, so the two would disagree about what the flight's domain is.
            if remaining > fg.max_air_hops:
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

        # A label at the ceiling cannot be extended, so stop before enumerating arcs at all.
        # The per-arc test below already implies this one -- `remaining` is a hex distance and
        # so non-negative, making `hops + 1 + remaining > max_air_hops` true for every
        # neighbour once `hops == max_air_hops` -- but only after six `remaining_distance`
        # calls that cannot change the outcome.  `_best_column` guards its extension loop the
        # same way; these two searches share a domain and should agree on how they bound it.
        if hops >= fg.max_air_hops:
            continue
        if step + 1 > fg.max_step:
            continue
        for neighbour in fg.outgoing_neighbors(cell):
            if neighbour in recent[:revisit_depth]:
                continue
            next_step = step + 1
            remaining = remaining_distance(neighbour)
            if next_step + remaining > fg.max_step:
                continue
            if hops + 1 + remaining > fg.max_air_hops:
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
    from 503 surviving variants to 1.  Cutoff quality dominates every other lever in this
    search, so this argument is the one to preserve when changing anything below.

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
    # Resolved once per call and threaded down; every objective expression below
    # derives from it.  See colgen.objective.
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
                if require_improving and seed_rc <= _IMPROVING_RC_TOL:
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
    # The compiled search first, the reference when it cannot prove it ran to completion.
    # `forbidden_rows` deliberately does NOT force the fallback: repair is O(flights) inside
    # the greedy, so a Python round trip per repair would be a scaling cliff at thousands of
    # flights, and the exclusion set is a bitset over dense row ids inside the kernel.
    (reduced_cost, column), proved = _best_column_compiled(
        fg,
        view,
        pi_value,
        cfg,
        benefit,
        forbidden,
        incumbent=incumbent,
        deadline=deadline,
        model=model,
    )
    _KERNEL_STATS["priced"] += 1
    if not proved:
        _KERNEL_STATS["fell_back"] += 1
        # The ORIGINAL incumbent, deliberately, not whatever the abandoned compiled attempt
        # managed to certify first.  Warm-starting the fallback would be optimality-safe --
        # pruning against an achievable score never discards anything strictly better -- but
        # it is not parity-safe: a stronger cutoff explores less than the reference did and
        # can return a different, equally optimal column.  The fallback exists to reproduce
        # the oracle, and it is rare enough that its speed is not the thing to optimize.
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
    if column is None or (require_improving and reduced_cost <= _IMPROVING_RC_TOL):
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
