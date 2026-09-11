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

The cutoff a pricing call could assemble for free is not merely weak, it is
**structurally 0.0** -- free to the LP.  Every column reachable for free is already in
the master's pool, LP optimality forces those reduced costs ``<= 0``, and complementary
slackness pins the basic one at exactly ``0``; the only route to a positive cutoff is to
SEARCH for one (see :func:`_bootstrap_incumbent`).
"""

from __future__ import annotations

import enum
import math
import sys
import threading
import time
import weakref
import heapq
import itertools
from collections import Counter
from collections.abc import Iterable, Mapping, Set as AbstractSet
from dataclasses import dataclass, replace
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
    shifted_claims_are_exact,
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
# ``solver._REDUCED_COST_TOL`` -- the same threshold read from opposite sides of the call,
# so the two must stay equal.
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
        "_cell_steps",
        "_duals",
        "_max_negative_credit",
        "_offsets",
        "_terminal",
        "_terminal_steps",
    )

    def __init__(
        self,
        duals: Mapping[RowKey | tuple[Any, ...], float],
        cfg: SimConfig,
    ) -> None:
        """Normalize duals to per-resource prefix sums for O(1) window queries.

        Parameters
        ------------
        - duals (Mapping[RowKey | tuple, float]): raw master-row dual prices; non-``RowKey``
          keys are coerced and duplicate keys summed.
        - cfg (SimConfig): supplies the geometry-derived visit-window offsets.

        Return
        --------
        - output (None): builds the indexed view; raises ``ValueError`` on a non-finite dual.
        """
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
        # The same buckets, kept rather than discarded, so a per-flight consumer
        # (`dp_prepare.prepare_duals`) can enumerate the resources IT owns instead of scanning
        # every global row -- an O(flights x rows) cost that grows with the master while each
        # flight stays the same size.
        #
        # These are the accumulated values, not a recomputation: `prefix[k+1] - prefix[k]`
        # would recover a DIFFERENT float, which `prepare_duals`' docstring forbids.  Tuples
        # rather than dicts because they are smaller and immutable, and pricing reads one view
        # from several flights concurrently.
        self._cell_steps = {
            resource: tuple(sorted(values.items())) for resource, values in cell_values.items()
        }
        self._terminal_steps = {
            terminal_id: tuple(sorted(values.items()))
            for terminal_id, values in terminal_values.items()
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

        Returns ``(key_prefix, base_step)`` pairs, where ``key_prefix + (step,)`` is a lookup
        key for ``_duals``.  Two facts make this exact rather than merely close:

        * ``RowKey`` is a plain ``tuple`` subclass, so the bare tuple written here hashes and
          compares equal to it -- the dict lookup is identical, it just skips
          ``RowKey.__new__``'s validation and its four ``operator.index`` calls.
        * Rows whose resource carries no dual at any step are dropped: they would contribute
          exactly ``0.0`` at every translation, and adding exact zeros cannot change an
          ``fsum``.

        Translation is injective (every row's step moves by the same delta), so no two rows
        collide into one and summing the terms is exactly summing the translated set.

        Parameters
        ------------
        - claims (Iterable[RowKey]): the canonical claim set to pre-resolve.

        Return
        --------
        - output (tuple[tuple[tuple, int], ...]): ``(key_prefix, base_step)`` pairs consumed
          by :meth:`shifted_claim_cost`.
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
        """Sum every cell-row dual charged by a centre visit at ``visit_step``, in O(1).

        Parameters
        ------------
        - cell (Cell): the visited cell.
        - level (int): the visit's flight level.
        - visit_step (int): integer clock step of the centre visit.

        Return
        --------
        - output (float): summed dual over the visit window, or 0.0 if the cell carries none.
        """

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
    """Return the cell-capacity rows one visit to ``cell`` at ``visit_step`` claims.

    Parameters
    ------------
    - cell (Cell): the axial ``(q, r)`` cell being visited.
    - level (int): the flight level of the visit.
    - visit_step (int): the step at which the cell centre is reached.
    - offsets (tuple[int, int]): inclusive ``(lo, hi)`` window from :func:`derive_cell_window`.

    Return
    --------
    - output (frozenset[RowKey]): the cell rows ``visit_step+lo .. visit_step+hi`` at that
      cell and level.
    """
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

    Parameters
    ------------
    - claims (Iterable[RowKey]): the untranslated canonical claim set to test.
    - forbidden (AbstractSet[RowKey]): rows that must stay clear.
    - delta_steps (int): integer clock shift applied to each claim before the membership test.

    Return
    --------
    - output (bool): ``True`` if any shifted claim falls in ``forbidden``; ``False`` when
      ``forbidden`` is empty.
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

    Parameters
    ------------
    - cell (Cell): the axial ``(q, r)`` cell being visited.
    - level (int): the flight level of the visit.
    - visit_step (int): the step at which the cell centre is reached.
    - offsets (tuple[int, int]): inclusive ``(lo, hi)`` window from :func:`derive_cell_window`.
    - forbidden (AbstractSet[RowKey]): rows that must stay clear.

    Return
    --------
    - output (bool): ``True`` if any of the visit's cell rows is in ``forbidden``; ``False``
      when ``forbidden`` is empty.
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
    answer-neutral rather than a heuristic: the body reads only the request's two endpoints,
    the two terminals, ``fg.levels`` and ``cfg`` scalars, all fixed for the graph's life --
    and every caller reaches this through :func:`price_flight` or
    :func:`find_feasible_column`, which refuse a ``cfg`` that is not ``fg._cfg``.  The
    redundancy it exploits is extreme: one sink proposal per ``(arrival step, hop count)``
    pair, so the reachable key space is far smaller than the number of sinks reaching it.

    Parameters
    ------------
    - fg (FlightGraph): the flight graph, supplying the endpoints, terminals, levels, and cache.
    - cfg (SimConfig): must be ``fg._cfg``; supplies the clock and geometry.
    - origin (bool): ``True`` for the origin endpoint, ``False`` for the destination.
    - step (int): the step at which the endpoint is occupied.
    - timing_steps (int): rebuilt-step count sizing the float-drift pad.

    Return
    --------
    - output (frozenset[RowKey]): the dwell rows that endpoint occupies, memoized on the graph.
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

    Parameters
    ------------
    - fg (FlightGraph): the flight graph, supplying the endpoints, terminals, and levels.
    - cfg (SimConfig): must be ``fg._cfg``; supplies the clock and geometry.
    - origin (bool): ``True`` for the origin endpoint, ``False`` for the destination.
    - step (int): the step at which the endpoint is occupied.
    - timing_steps (int): rebuilt-step count sizing the float-drift pad.

    Return
    --------
    - output (frozenset[RowKey]): terminal rows when the endpoint is a terminal, else the cell
      rows the endpoint cylinder touches across its dwell.
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
    """Return one deterministic shortest path using lazy, cached A* expansion.

    Parameters
    ------------
    - fg (FlightGraph): supplies ``outgoing_neighbors`` for lattice expansion.
    - start (Cell): the start cell.
    - destination (Cell): the goal cell.
    - deadline (float | None): ``time.monotonic`` deadline; ``None`` disables the check.

    Return
    --------
    - output (tuple[Cell, ...] | None): the cell path from ``start`` to ``destination``
      inclusive, or ``None`` when ``start == destination`` or no path exists. Raises
      ``PricingTimeout`` if ``deadline`` passes.
    """

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
      so ``(cell, visit_step)`` repeats heavily.

    Parameters
    ------------
    - fg (FlightGraph): the flight graph, supplying lanes, takeoff steps, and endpoints.
    - cfg (SimConfig): supplies the clock and the derived cell window.
    - label (_Label): the search label carrying the path, departure step, and hop count.
    - dest_lane_idx (int | None): the selected destination lane; ignored here (it changes
      geometry, not dwell-row membership).
    - endpoint_cache (dict | None): optional memo for endpoint row sets across a batch.
    - visit_cache (dict | None): optional memo for per-visit cell row sets across a batch.

    Return
    --------
    - output (frozenset[RowKey]): the intended row union (both endpoints plus every per-visit
      cell window) before canonical certification.
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
    """Compute the exact v1 delay ruler without building reservation volumes.

    Parameters
    ------------
    - fg (FlightGraph): supplies the request endpoints, terminals, and levels.
    - cfg (SimConfig): supplies the lattice geometry, clock, and nominal speed.
    - label (_Label): the label carrying the cell path and departure step.
    - model (CostModel): cost weights applied to the ground and air terms; defaults to
      ``DELAY_MODEL``.

    Return
    --------
    - output (float): the evaluated delay in the cost model's currency (ground delay plus the
      en-route detour term).
    """

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
    """Return the endpoint leg charged by the arc-form delay objective.

    Parameters
    ------------
    - point (Vec): the endpoint position in local ENU metres; only x/y are used.
    - terminal (Terminal | None): the endpoint's terminal, or ``None`` for a customer endpoint.
    - lane_dist (float | None): the lane distance; required when ``terminal`` is set and must
      be ``None`` otherwise.
    - cfg (SimConfig): supplies the geometry, exit radius, and nominal speed.

    Return
    --------
    - output (float): the endpoint leg time in seconds charged by the arc-form objective.
    """

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

    Parameters
    ------------
    - point (Vec): the endpoint position in local ENU metres; only x/y are used.
    - terminal (Terminal): the endpoint's terminal.
    - cell (Cell): the lane cell whose centre the fold measures from.
    - cfg (SimConfig): supplies the geometry, exit radius, and nominal speed.

    Return
    --------
    - output (tuple[float, bool]): the fold leg time in seconds, and whether folding retains
      the lane cell exactly (distance >= the exit radius).
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

    Parameters
    ------------
    - ground_delay_s (float): the prefix's irrevocable ground delay in seconds.
    - origin_fold_s (float): the origin fold leg time in seconds.
    - hops (int): hops already flown in the prefix.
    - remaining_hops (int): admissible lower bound on the unflown hops (plain hex distance).
    - destination_fold_s (float): the destination fold leg time in seconds.
    - reference_time_s (float): the reference (geodesic) flight time in seconds.
    - dt_s (float): the clock step in seconds.
    - folding_exact (bool): whether both terminal folds retain their boundary cells.
    - model (CostModel): cost weights for the ground and air terms; defaults to ``DELAY_MODEL``.

    Return
    --------
    - output (float): a lower bound on canonical delay over every completion of the prefix;
      falls back to the ground-only cost when ``folding_exact`` is False or the reference is
      non-positive.
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


def _remember_certified_column(
    column: Column, fg: FlightGraph, model: CostModel, air_detour_s: float
) -> Column:
    """Recognize live certified objects without extending their lifetime."""
    cache = fg._search_cache.certified_columns
    lock = fg._search_cache.lock
    key = id(column)

    def discard(reference):
        # The registry may already contain a newer reference at the same object ID.
        # Capture only the registry, lock and integer key, never the column or graph.
        with lock:
            entry = cache.get(key)
            if entry is not None and entry[0] is reference:
                cache.pop(key)

    with lock:
        cache[key] = (weakref.ref(column, discard), model, air_detour_s)
    return column


def is_certified_column(column: Column, fg: FlightGraph, model: CostModel) -> bool:
    """Recognize an exact internally certified object, never an equal replacement.

    Parameters
    ------------
    - column: Immutable column object to look up.
    - fg: Graph owning the weak identity certification registry.
    - model: Objective weights required for the cached cost.

    Return
    --------
    - certified (bool): Whether this exact object was certified in that context.
    """
    with fg._search_cache.lock:
        entry = fg._search_cache.certified_columns.get(id(column))
    return entry is not None and entry[0]() is column and entry[1] == model


def certify_column(column: Column, fg: FlightGraph, cfg: SimConfig, model: CostModel) -> Column:
    """Validate foreign claims and cost once before accepting a pricing cutoff.

    Parameters
    ------------
    - column: Internal or imported immutable route to validate.
    - fg: Graph supplying the immutable request, domain and static walls.
    - cfg: Configuration used to build the graph; mismatches raise ValueError.
    - model: Current objective weights for canonical trajectory cost.

    Return
    --------
    - canonical (Column): Original object or corrected claims/cost copy, certified
      in this graph and objective. Invalid routes raise ValueError.
    """
    if cfg != fg._cfg:
        raise ValueError("certification requires the SimConfig used to build the flight graph")
    if is_certified_column(column, fg, model):
        return column
    intent = column_to_intent(column, fg.request, cfg)
    if intent.status is not IntentStatus.ACCEPTED:
        raise ValueError("column does not translate to an accepted intent")
    claims = column_claims(column, fg, cfg, _intent=intent)
    cost = model.intent_cost(intent, cfg)
    canonical = column if claims == column.claims and cost == column.delay_s else replace(
        column, claims=claims, delay_s=cost)
    return _remember_certified_column(canonical, fg, model, intent.air_detour_m / cfg.nominal_speed_mps)


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
    under an integer clock translation. Endpoint rounding at the actual clock
    must agree before translating claims; otherwise the candidate is canonically
    rebuilt before scoring. Cost uses the canonical air term and the new absolute
    lattice ground delay, preserving floating-point evaluation order.
    These columns are used only as certified lower bounds for the exact DAG;
    the prepass never proves optimality or replaces pricing.

    With non-negative duals, the first feasible translation that pays no dual
    dominates every later seed translation, so the scan may stop there.  A
    negative backend-tolerance dual disables that stopping rule.

    Parameters
    ------------
    - seed (Column): the seed column whose integer time-translations are scanned.
    - fg (FlightGraph): the flight graph, supplying the departure bounds and lanes.
    - duals (DualView): the dual view scoring each translation's rows.
    - pi_f (float): the flight's dual (``pi_f``).
    - cfg (SimConfig): supplies ``dt_s`` for the ground-delay term.
    - benefit (float): the per-flight benefit ``M``.
    - forbidden_rows (AbstractSet[RowKey]): rows that must stay clear.
    - incumbent (tuple[float, Column] | None): the current best ``(reduced_cost, column)`` to
      beat.
    - deadline (float | None): ``time.monotonic`` deadline; ``None`` disables the check.
    - model (CostModel): cost weights for the added ground delay; defaults to ``DELAY_MODEL``.

    Return
    --------
    - output (tuple[float, Column] | None): the strengthened ``(reduced_cost, column)`` when a
      better translation is found, else ``incumbent`` unchanged (possibly ``None``).
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
    seed = certify_column(seed, fg, cfg, model)
    terms = duals.shift_terms(seed.claims)
    with fg._search_cache.lock:
        air_detour_s = fg._search_cache.certified_columns[id(seed)][2]
    best_delta: int | None = None
    best_delay_s = 0.0
    for departure_step in range(seed.departure_step + 1, latest_departure + 1):
        _check_deadline(deadline)
        delta_steps = departure_step - seed.departure_step
        exact_shift = shifted_claims_are_exact(seed, fg, cfg, departure_step)
        shifted = None
        if exact_shift:
            if _rows_hit_forbidden(seed.claims, forbidden_rows, delta_steps=delta_steps):
                continue
            dual_cost = duals.shifted_claim_cost(terms, delta_steps)
            delay_s = model.evaluate(
                ground_s=(departure_step - fg.base_step) * cfg.dt_s,
                air_detour_s=air_detour_s)
        else:
            shifted = certify_column(replace(seed, departure_step=departure_step), fg, cfg, model)
            if not shifted.claims.isdisjoint(forbidden_rows):
                continue
            dual_cost = duals.claim_cost(shifted.claims)
            delay_s = shifted.delay_s
        reduced_cost = model.reduced_cost(
            benefit=benefit, cost=delay_s, dual_cost=dual_cost, pi_f=pi_f
        )
        if best is None or reduced_cost > best[0] + _SCORE_EPS:
            best = (reduced_cost, shifted)  # fast-path column built once after the scan
            best_delta = delta_steps
            best_delay_s = delay_s
        if dual_cost == 0.0 and duals.max_negative_credit == 0.0:
            break
    if best_delta is not None and best[1] is None:
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
    if best is not None and best_delta is not None:
        _remember_certified_column(best[1], fg, model, air_detour_s)
    return best


def _root_bounds(topology, variants, envelopes, benefit: float, pi_f: float, dual_view):
    """``completion_can_compete``'s ``hop_rc_bound`` per root, for ranking.

    The gate already evaluates this expression for every surviving root; this reads out the
    NUMBER where ``can_compete`` returns only the verdict.  Nothing here is a new bound and
    nothing prunes: the value orders the bootstrap's root allowlist and has no effect on what
    any search explores, so it cannot discard a column.

    ``-inf`` for a root the envelope's own LENGTH rejects, which sorts it last.

    Parameters
    ------------
    - topology (PreparedTopology): supplies ``hex_remaining`` per corridor cell.
    - variants (PreparedVariants): the candidate roots (departure step, lane, cell, dual cost).
    - envelopes (CompletionEnvelopes): supplies ``_delay_envelope`` and ``_destination_cost``.
    - benefit (float): the per-flight benefit ``M``.
    - pi_f (float): the flight's dual (``pi_f``).
    - dual_view (DualView): supplies ``max_negative_credit``.

    Return
    --------
    - output (np.ndarray): per-root ``hop_rc_bound`` (float64), ``-inf`` for a root the
      envelope length rejects (sorts last); used only for ranking, never to prune.
    """

    n = int(variants.departure_step.size)
    out = np.full(n, -math.inf, dtype=np.float64)
    for i in range(n):
        lane_raw = int(variants.lane_idx[i])
        first_hops = max(1, int(topology.hex_remaining[int(variants.cell[i])]))
        delay_lbs, corridor_start = envelopes._delay_envelope(
            int(variants.departure_step[i]), None if lane_raw < 0 else lane_raw
        )
        if first_hops >= len(delay_lbs):
            continue
        destination_positive = envelopes._destination_cost(
            corridor_start + first_hops, first_hops
        )
        if not math.isfinite(destination_positive):
            continue
        paid_positive = max(0.0, float(variants.start_dual_cost[i]))
        out[i] = (
            benefit - pi_f - delay_lbs[first_hops]
            - max(paid_positive, destination_positive)
            + dual_view.max_negative_credit
        )
    return out


def _bootstrap_incumbent(
    fg: FlightGraph,
    dual_view: DualView,
    pi_f: float,
    cfg: SimConfig,
    benefit: float,
    forbidden_rows: AbstractSet[RowKey],
    model: CostModel,
    *,
    incumbent: tuple[float, Column] | None,
    roots: int,
    search_single_root: bool = False,
    ranking: str = "score",
    method: str = "dp",
    max_labels: int = 20000,
    deadline: float | None = None,
    preparation: dp_prepare.PricingPreparation | None = None,
) -> tuple[float, Column] | None:
    """Find an achievable reduced-cost cutoff over a few ranked start options.

    Parameters
    ------------
    - fg, dual_view, pi_f, cfg, benefit, forbidden_rows, model: Pricing subproblem.
    - incumbent: Previously certified reduced cost and column, or None.
    - roots (int): Number of start options selected by the ranking.
    - method (str): Goal-directed warm start plus restricted DP (astar), or DP alone (dp).
    - max_labels (int): Expansion limit for the goal-directed heuristic.
    - deadline: Caller-owned monotonic deadline.

    Return
    --------
    - incumbent: Improved, canonically certified cutoff, or the original incumbent.

    The heuristic only supplies a feasible route. Restricted DP refines its cutoff:
    an exhausted heuristic or a weak first goal must not leave the full search with
    worse pruning than the DP-only bootstrap. The unrestricted DP retains every root
    and certifies the final optimum; heuristic labels never enter either exact search.
    """

    _LAST_SEARCH.update(n_labels=0, bootstrap_goal_labels=0, bootstrap_goal_s=0.0,
                        bootstrap_goal_native=False, bootstrap_goal_decline_reason=None,
                        bootstrap_goal_sinks_skipped=0, bootstrap_goal_sinks_asked=0)

    # Rank ONCE, here, and hand the result to whichever search runs.  `prepare_variants` is
    # pure Python -- `dp_prepare` has no numba anywhere -- so this works identically on a
    # numba-less install, which is what keeps `--reference-baseline` a real gate rather than
    # a second code path.  `prepared_for` is memoized per graph, so the packing is not
    # rebuilt for this.
    if preparation is not None:
        preparation.check(fg, cfg, dual_view, benefit, pi_f, model, forbidden_rows)
    topology, rows = (dp_prepare.prepared_for(fg, cfg) if preparation is None
                      else preparation.topology_rows)
    if not (topology.ok and rows.ok):
        # Both searches decline this graph too, so skipping here keeps them symmetric.
        return incumbent
    # WITH `envelopes`, which is not optional.  The ranking has to happen over the same gated
    # set the restricted search will see, or the top-scoring root can be one
    # `completion_can_compete` rejects -- the allowlist then names a root the search
    # immediately discards, `prepare_variants` returns EMPTY, and the bootstrap silently does
    # nothing while still costing its own setup.  The failure is invisible -- no crash, a
    # successful `(-inf, None)`, the incumbent handed straight back.
    envelopes = preparation.envelopes(incumbent, deadline) if preparation else dp_prepare.CompletionEnvelopes(
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
        preparation=preparation,
    )
    if variants.departure_step.size == 0 or (
        variants.departure_step.size == 1 and not search_single_root
    ):
        # Nothing to bootstrap FROM: with one root the bootstrap IS the search.
        return incumbent
    # Descending, STABLE, so ties resolve by `prepare_variants`' insertion order --
    # `(departure, lane)` ascending -- and the selection is a pure function of the graph,
    # the duals and the incumbent.  It has to be, or the two searches could be handed
    # different allowlists and the whole symmetry argument goes with it.
    #
    # `ranking` picks WHAT to sort on, and the two options are `g` versus `g + h`:
    #
    #   "score"  `variants.score` = -w_ground*ground_delay - w_air*origin_leg
    #                               - start_dual_cost.  Cost already incurred at the root, with
    #            no lookahead.
    #   "bound"  `_root_bounds` below: `completion_can_compete`'s own `hop_rc_bound` at the
    #            root's minimum feasible hop count -- an UPPER bound on what the root can
    #            ACHIEVE, already computed for every surviving root by the gate above.
    #
    # They rank identically whenever every root shares one lane geometry, because then the only
    # thing separating roots is departure time and both collapse to "depart earlier".  They
    # diverge on multi-lane flights: `score` charges `-w_air*origin_leg`, so it demotes long
    # lanes outright; `bound` charges `-delay_lbs[h0]` with `h0 = hex_remaining[root cell]`, so
    # it accounts for where the lane LEAVES you relative to the destination.
    if ranking == "bound":
        key = _root_bounds(topology, variants, envelopes, benefit, pi_f, dual_view)
    else:
        key = variants.score
    order = np.argsort(-key, kind="stable")[:roots]
    keep = frozenset(
        (int(variants.departure_step[i]), int(variants.lane_idx[i])) for i in order
    )
    goal_labels = 0
    goal_s = 0.0
    if method == "astar":
        goal_started = time.perf_counter()
        incumbent = _goal_directed_bootstrap(
            fg, dual_view, pi_f, cfg, benefit, forbidden_rows, model,
            variants, order, topology, envelopes, incumbent=incumbent,
            max_labels=max_labels, deadline=deadline,
            preparation=preparation,
        )
        goal_labels = int(_LAST_SEARCH.get("n_labels", 0))
        goal_s = time.perf_counter() - goal_started
    _LAST_SEARCH["n_labels"] = 0
    outcome = _best_column_compiled(
        fg,
        dual_view,
        pi_f,
        cfg,
        benefit,
        forbidden_rows,
        incumbent=incumbent,
        deadline=deadline,
        model=model,
        keep_roots=keep,
        record_budget=False,
        preparation=preparation,
    )
    if isinstance(outcome, Declined):
        # `roots` roots rather than the whole window, so the reference is affordable here in
        # a way it is not for the main search.
        outcome = _best_column(
            fg,
            dual_view,
            pi_f,
            cfg,
            benefit,
            forbidden_rows,
            seed=False,
            incumbent=incumbent,
            deadline=deadline,
            model=model,
            keep_roots=keep,
        )
    _LAST_SEARCH.update(bootstrap_goal_labels=goal_labels, bootstrap_goal_s=goal_s)
    score, column = outcome
    if column is None:
        return incumbent
    if incumbent is not None and score <= incumbent[0] + _SCORE_EPS:
        return incumbent
    return score, column


def _goal_directed_bootstrap(
    fg, duals, pi_f, cfg, benefit, forbidden_rows, model,
    variants, order, topology, envelopes, *, incumbent, max_labels, deadline,
    preparation=None,
):
    """Run the bounded native heuristic, retaining the Python oracle on explicit decline.

    Parameters
    ------------
    - fg, duals, pi_f, cfg, benefit, forbidden_rows, model: Fixed pricing subproblem.
    - variants, order, topology, envelopes: Ranked roots and this stage's pruning bounds.
    - incumbent, max_labels, deadline: Certified cutoff and heuristic work/time limits.
    - preparation: Optional invocation-local packed inputs shared with both DP stages.

    Return
    --------
    - incumbent: Canonically certified improvement or the original cutoff.
    """
    _check_deadline(deadline)
    native = None
    if _dp_kernel() is not None:
        try:
            from .bootstrap_kernel import goal_directed_bootstrap as native
        except ImportError:
            pass
    diagnostics = dict(native_used=False, decline_reason="numba unavailable")
    if native is not None:
        packed = {}
        if preparation is not None:
            preparation.check(fg, cfg, duals, benefit, pi_f, model, forbidden_rows)
            packed = dict(prepared_rows=preparation.topology_rows[1],
                          prepared_duals=preparation.duals,
                          prepared_forbidden=preparation.forbidden)
        result = native(
            fg, duals, pi_f, cfg, benefit, forbidden_rows, model,
            variants, order, topology, envelopes, incumbent=incumbent,
            max_labels=max_labels, deadline=deadline, diagnostics=diagnostics, **packed,
        )
        if result is not None:
            incumbent, expanded = result
            _LAST_SEARCH.update(n_labels=expanded, bootstrap_goal_native=True,
                                bootstrap_goal_decline_reason=None,
                                bootstrap_goal_sinks_skipped=int(diagnostics.get("sinks_skipped", 0)),
                                bootstrap_goal_sinks_asked=int(diagnostics.get("sinks_asked", 0)))
            return incumbent
    _LAST_SEARCH.update(bootstrap_goal_native=False,
                        bootstrap_goal_decline_reason=diagnostics["decline_reason"])
    return _goal_directed_bootstrap_python(
        fg, duals, pi_f, cfg, benefit, forbidden_rows, model,
        variants, order, topology, envelopes, incumbent=incumbent,
        max_labels=max_labels, deadline=deadline,
    )


def _goal_directed_bootstrap_python(
    fg, duals, pi_f, cfg, benefit, forbidden_rows, model,
    variants, order, topology, envelopes, *, incumbent, max_labels, deadline,
):
    """Find a certified cutoff with bounded best-first search over ranked roots.

    Parameters
    ------------
    - variants, order: Prepared start options and their selected ranking.
    - max_labels (int): Maximum expanded labels across the heuristic search.
    - incumbent: Achievable reduced cost and column, or None.
    - deadline: Caller-owned monotonic deadline.

    Return
    --------
    - incumbent: Best certified route found, retaining the input on failure.

    Only a canonically checked route leaves this heuristic. Its queue order, label
    limit and early exit cannot certify optimality; the unrestricted DP still runs.
    """

    destinations = _destination_options(fg)
    remaining_cache = {}
    offsets = duals.offsets
    recent_depth = max(2, offsets[1] - offsets[0])
    revisit_depth = offsets[1] - offsets[0]
    serial = itertools.count()
    expanded = 0
    certify = _sink_certifier(fg, duals, pi_f, cfg, benefit, forbidden_rows, model,
                             deadline=deadline)

    def remaining(cell):
        """Memoize the distance used only for queue order and the hop ceiling."""
        if cell not in remaining_cache:
            remaining_cache[cell] = _distance_lower_bound(cell, destinations)
        return remaining_cache[cell]

    for root in order:
        _check_deadline(deadline)
        departure = int(variants.departure_step[root])
        lane_raw = int(variants.lane_idx[root])
        lane = None if lane_raw < 0 else lane_raw
        cell_index = int(variants.cell[root])
        cell = (int(topology.cell_q[cell_index]), int(topology.cell_r[cell_index]))
        start = int(variants.start_step[root])
        origin_rows = _endpoint_claims(fg, cfg, origin=True, step=departure, timing_steps=0)
        paid_rows = origin_rows | _visit_claims(cell, 0, start, offsets)
        paid_lookup = {(r.cell_coord, r.step): duals.row_cost(r)
                       for r in paid_rows if r.kind == "cell" and r.level == 0}
        paid_cells = {c for c, _step in paid_lookup}
        delay_bounds, _corridor_start = envelopes._delay_envelope(departure, lane)

        def priority(hops, distance, paid):
            """Rank by a completion estimate; only certification supplies a cutoff."""
            total = max(1, hops + distance)
            if total >= len(delay_bounds):
                return math.inf
            destination_cost = envelopes._destination_cost(start + total, total)
            return float(delay_bounds[total]) + max(paid, destination_cost)

        initial_paid = duals.claim_cost(paid_rows)
        frontier = [(priority(0, remaining(cell), initial_paid), remaining(cell),
                     next(serial), start, (cell,), initial_paid, False)]
        best_paid = {}
        while frontier and expanded < max_labels:
            _check_deadline(deadline)
            _bound, _distance, _serial, step, path, paid, finished = heapq.heappop(frontier)
            if finished:
                for dest_lane in destinations[path[-1]]:
                    candidate = certify(incumbent, departure, lane, dest_lane, step, path)
                    if candidate is not None:
                        _LAST_SEARCH["n_labels"] = expanded
                        return candidate
                continue
            expanded += 1
            hops = len(path) - 1
            if hops >= fg.max_air_hops or step >= fg.max_step:
                continue
            recent = path[-recent_depth:]
            for neighbor in fg.outgoing_neighbors(path[-1]):
                if revisit_depth and neighbor in path[-revisit_depth:]:
                    continue
                distance = remaining(neighbor)
                next_step = step + 1
                if hops + 1 + distance > fg.max_air_hops or next_step + distance > fg.max_step:
                    continue
                finish = neighbor in destinations and fg.hop_allowed_for_role(
                    path[-1], neighbor, first=hops == 0, last=True)
                onward = fg.hop_allowed_for_role(path[-1], neighbor, first=hops == 0, last=False)
                if not (finish or onward) or _visit_hits_forbidden(neighbor, 0, next_step, offsets, forbidden_rows):
                    continue
                visit = duals.visit_cost(neighbor, 0, next_step)
                if neighbor in paid_cells:
                    visit -= math.fsum(paid_lookup.get((neighbor, t), 0.0)
                                       for t in visit_rows(next_step, offsets))
                new_paid = paid + visit
                new_path = (*path, neighbor)
                bound = priority(hops + 1, distance, new_paid)
                if finish:
                    heapq.heappush(frontier, (bound, 0, next(serial), next_step, new_path, new_paid, True))
                if onward:
                    first_hop = new_path[:2] if fg.static_walls and fg.origin_terminal is not None else ()
                    state = (neighbor, next_step, (*recent, neighbor)[-recent_depth:], first_hop)
                    if new_paid < best_paid.get(state, math.inf) - _SCORE_EPS:
                        best_paid[state] = new_paid
                        heapq.heappush(frontier, (bound, distance, next(serial), next_step, new_path, new_paid, False))
    _LAST_SEARCH["n_labels"] = expanded
    return incumbent


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
    """Certify one search candidate's route and return its exact reduced cost, or ``None``.

    Parameters
    ------------
    - candidate (_Candidate): the search candidate carrying the label, destination lane, and
      provisional delay.
    - fg (FlightGraph): the flight graph, supplying the request and geometry.
    - duals (DualView): the dual view scoring the certified claims.
    - pi_f (float): the flight's dual (``pi_f``).
    - cfg (SimConfig): supplies the geometry, clock, and nominal speed.
    - benefit (float): the per-flight benefit ``M``.
    - forbidden_rows (AbstractSet[RowKey]): rows that must stay clear.
    - model (CostModel): cost weights for the delay ruler; defaults to ``DELAY_MODEL``.

    Return
    --------
    - output (tuple[float, Column] | None): the certified ``(reduced_cost, column)``, or
      ``None`` when the route is not accepted, has illegal claims, or hits a forbidden row.
    """
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

    exact_delay = model.intent_cost(intent, cfg)
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
    return reduced_cost, _remember_certified_column(
        column, fg, model, intent.air_detour_m / cfg.nominal_speed_mps)


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

    Parameters
    ------------
    - fg (FlightGraph): the flight graph, supplying the request and geometry.
    - dual_view (DualView): the dual view scoring certified claims.
    - pi_f (float): the flight's dual (``pi_f``).
    - cfg (SimConfig): supplies the geometry, clock, and nominal speed.
    - benefit (float): the per-flight benefit ``M``.
    - forbidden_rows (AbstractSet[RowKey]): rows that must stay clear.
    - model (CostModel): cost weights for the delay ruler; defaults to ``DELAY_MODEL``.
    - deadline (float | None): ``time.monotonic`` deadline; ``None`` disables the check.

    Return
    --------
    - output (Callable): a ``certify(incumbent, departure_step, origin_lane_idx, dest_lane_idx,
      arrival_step, path)`` closure returning the improved ``(reduced_cost, Column)`` incumbent,
      or ``None`` when the sink does not improve on the one passed in.
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

    The result is cached on the graph, keyed on ``model`` -- not one seed per graph.  The key
    matters because comparing two objectives on one graph (the natural way to write such a
    comparison) would otherwise silently get the first model's seed twice, with no error.

    Parameters
    ------------
    - fg (FlightGraph): the flight graph, supplying endpoints, lanes, and the search cache.
    - cfg (SimConfig): supplies the geometry, clock, and nominal speed.
    - deadline (float | None): ``time.monotonic`` deadline; ``None`` disables the check.
    - model (CostModel): cost weights and the cache key; defaults to ``DELAY_MODEL``.

    Return
    --------
    - output (tuple[Column, ...]): a one-element tuple with the best certified shortest-delay
      seed, or empty when none certifies; cached on the graph keyed by ``model``.
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
    """Certify the best deterministic BFS seed without expanding the time DAG.

    Parameters
    ------------
    - fg (FlightGraph): the flight graph, supplying endpoints, lanes, and the search cache.
    - cfg (SimConfig): supplies the geometry, clock, and nominal speed.
    - deadline (float | None): ``time.monotonic`` deadline; ``None`` disables the check.
    - model (CostModel): cost weights and the cache key; defaults to ``DELAY_MODEL``.

    Return
    --------
    - output (Column | None): the best certified seed column, or ``None`` when none certifies.
    """

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
    additional_columns: list[Column] | None = None,
    column_limit: int = 1,
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

    Parameters
    ------------
    - candidates (list[_Candidate]): the sink proposals to rank and certify (sorted in place).
    - fg (FlightGraph): the flight graph, supplying the request and geometry.
    - dual_view (DualView): the dual view scoring certified claims.
    - pi_f (float): the flight's dual (``pi_f``).
    - cfg (SimConfig): supplies the geometry, clock, and nominal speed.
    - benefit (float): the per-flight benefit ``M``.
    - forbidden_rows (AbstractSet[RowKey]): rows that must stay clear.
    - model (CostModel): cost weights for the delay ruler; defaults to ``DELAY_MODEL``.
    - incumbent (tuple[float, Column] | None): the score already certified before ranking; the
      early break is measured against it, so ``None`` certifies far more candidates.
    - deadline (float | None): ``time.monotonic`` deadline; ``None`` disables the check.

    Return
    --------
    - output (tuple[float, Column] | None): the best certified ``(reduced_cost, column)``, or
      ``incumbent`` when nothing beats it (possibly ``None``).
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

    if additional_columns is not None and column_limit > 1:
        seen = set() if best is None else {_column_sort_key(best[1])}
        for candidate in candidates:
            _check_deadline(deadline)
            if candidate.reduced_cost <= _IMPROVING_RC_TOL:
                continue
            # Reuse the completed DP's surviving sinks. This is a bounded selection
            # of useful alternatives, not an exhaustive k-best pricing certificate.
            canonical = _canonical_candidate(candidate, fg, dual_view, pi_f, cfg,
                                             benefit, forbidden_rows, model)
            if canonical is None or canonical[0] <= _IMPROVING_RC_TOL:
                continue
            key = _column_sort_key(canonical[1])
            if key in seen:
                continue
            seen.add(key)
            additional_columns.append(canonical[1])
            if len(additional_columns) >= column_limit - 1:
                break

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
    keep_roots: frozenset[tuple[int, int]] | None = None,
    additional_columns: list[Column] | None = None,
    column_limit: int = 1,
) -> tuple[float, Column | None]:
    """Return the most negative-reduced-cost column, or ``(-inf, None)`` when none exists.

    This is the reference dynamic program: an exact, dominance-pruned label search
    over the flight's space-time DAG.  It defines what a correct answer *is* -- every
    later acceleration is measured against it -- so it is written for legibility over
    speed, and every pruning rule below is an argument that the discarded label cannot
    lead to a strictly better column.  ``incumbent`` warm-starts that pruning with an
    already-certified ``(reduced_cost, column)`` pair.

    ``keep_roots`` restricts the search to an explicit allowlist of
    ``(departure_step, lane_idx)`` pairs, ``-1`` for a bare origin.  This is the axis
    ``seed=True`` already restricts in its most degenerate form -- one departure, every lane
    -- so it generalizes a bound this search has always had rather than adding one.  The
    compiled search takes the same argument straight through to ``prepare_variants``, and
    the allowlist is built ONCE by :func:`_bootstrap_incumbent`, so neither search ranks
    roots for itself and the two cannot disagree about which ones they kept.

    Parameters
    ------------
    - fg (FlightGraph): the flight graph whose space-time DAG is searched.
    - dual_view (DualView): the dual view scoring each candidate's rows.
    - pi_f (float): the flight's dual (``pi_f``); must be finite.
    - cfg (SimConfig): supplies the geometry, clock, and nominal speed.
    - benefit (float): the per-flight benefit ``M``.
    - forbidden_rows (AbstractSet[RowKey]): rows that must stay clear.
    - seed (bool): restrict the search to the seed's single departure (every lane).
    - incumbent (tuple[float, Column] | None): an already-certified cutoff that warm-starts
      pruning.
    - deadline (float | None): ``time.monotonic`` deadline; ``None`` disables the check.
    - model (CostModel): cost weights for the delay ruler; defaults to ``DELAY_MODEL``.
    - keep_roots (frozenset[tuple[int, int]] | None): allowlist of ``(departure_step,
      lane_idx)`` roots (``-1`` for a bare origin); ``None`` searches every root.

    Return
    --------
    - output (tuple[float, Column | None]): the winning column and its reduced-cost score, or
      ``(-math.inf, None)`` when no feasible column exists.
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
    # max_air_hops is the route-length ceiling. The corridor lower bound uses actual
    # cruise endpoints; accumulated hops plus remaining distance is a stronger test.
    # See context/figures/colgen_lane_aware_corridor.png.
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
    # check below reads only the prefix whose windows could overlap. Consumed hops
    # are a separate resource: a better-scoring prefix with less remaining allowance
    # cannot dominate a prefix that can still complete a longer suffix.
    layers: dict[
        int,
        dict[
            tuple[
                Cell,
                tuple[Cell, ...],
                frozenset[RowKey],
                tuple[Cell, Cell] | None,
                int,  # consumed hops: labels with different remaining budgets cannot merge
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
        # Per-hop delay/dual bound vs the incumbent; a label lives while ANY hop in range could
        # still win (see context/figures/completion_envelope.png).
        # Cap the envelope at `max_air_hops`: the horizon term can run to hundreds of hops
        # against a ceiling of tens, and every entry past the ceiling describes a completion
        # the search cannot make.  The length of the result is what matters (not the cost of
        # building it, which the `break` below already bounds to `max_air_hops + 1`): it bounds
        # `completion_can_compete`'s scan (`range(first_hops, len(delay_lbs))`), which keeps a
        # label alive as soon as SOME hop count in range could beat the incumbent.  Entries
        # past the ceiling would let a label survive on a completion the ceiling forbids.
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
            if keep_roots is not None and (
                departure_step, -1 if lane_idx is None else int(lane_idx)
            ) not in keep_roots:
                continue
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
            key = (cell, recent, origin_paid_rows, None, 0)
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
        for label_index, ((cell, recent, origin_paid_rows, first_hop, _hops), label) in enumerate(
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
                # form changes the association and so is not bit-identical to the unweighted
                # expression it replaced.  This function is the oracle any compiled pricing
                # path gets certified against, so its arithmetic has to be reproducible
                # exactly, not just to within a tolerance.
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
                # materializing its ``RowKey`` set, which cost one ``RowKey.__new__`` (four
                # ``operator.index`` calls) per row purely to sum them.  Rows the origin
                # endpoint already paid must not be charged twice; that overlap is confined to
                # the endpoint's own cells, so the guard below is a miss on essentially every
                # arc.
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
                    next_label.hops,
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
        additional_columns=additional_columns, column_limit=column_limit,
    )
    return (-math.inf, None) if best is None else best


_kernel_fallback_warned = False
# Same warn-once-per-process discipline as `_kernel_fallback_warned`, and for the same
# reason: a budget the pool could not grow into is silent, and per-flight it would be a
# thousand identical lines rather than a signal.  Two flags rather than one because the two
# conditions want different responses -- see `_warn_budget_growth`.
_kernel_restart_warned = False
_kernel_budget_warned = False
# Flights already reported as near the label ceiling.  Per FLIGHT rather than once per
# process, because "which flights have no headroom left" is the diagnostic -- and per
# flight rather than per (flight, sweep), because a flight that is big in one sweep is big
# in all of them and repeating it every iteration would bury the next distinct one.
_kernel_high_water_warned: set = set()

# Per-process tally of exact-pricing calls and how many fell back out of the kernel.
#
# A fallback is slower but produces the RIGHT answer, so nothing downstream can notice it:
# the objective, the columns and the tests are all identical, only the clock moves.  Counting
# it is the only way a production run can report "the compiled path served 100% of pricing"
# rather than assume it.
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
#
# A `Counter` rather than a fixed dict because `price_flight` also tallies one key per
# `Declined` member, and the reachable set of those depends on the instance.
_KERNEL_STATS: Counter[str] = Counter(
    {"priced": 0, "fell_back": 0, "label_restarts": 0, "budget_declined": 0}
)


def kernel_stats() -> dict[str, int]:
    """Snapshot this process's compiled-pricing tally."""

    return dict(_KERNEL_STATS)


#: The LAST compiled search's per-flight facts, for the caller that wants them PER FLIGHT
#: rather than summed.  `_KERNEL_STATS` is a running total and cannot answer "which flight
#: was the straggler" -- the question that decides whether skip-filtering the cheap flights
#: would help at all, since a pool's makespan is set by its slowest single task.
#:
#: A module global rather than a return value because `_best_column_compiled` has nine
#: return sites and every one of them is a `Declined` member or a column; widening that
#: contract to carry diagnostics would put a profiling concern in the type every caller
#: matches on.  It is written unconditionally and read by whoever cares.
_LAST_SEARCH: dict[str, Any] = {}


def last_search_record() -> dict[str, Any]:
    """The last compiled search's facts, or ``{}`` if none ran in this process.

    Returns a COPY: the caller in `pricing_pool` ships this across a process boundary, and
    handing out the live dict would let the next flight's search mutate a record already
    queued for pickling.
    """

    return dict(_LAST_SEARCH)


def clear_search_record() -> None:
    """Forget the last search, so a caller can tell "did not run" from "ran previously".

    Without this a flight that declines BEFORE reaching the kernel -- no numba, multi-level
    graph, refused topology -- would report the previous flight's labels as its own, which
    is the most misleading possible answer for a straggler hunt.
    """

    _LAST_SEARCH.clear()


class Declined(enum.Enum):
    """Why the compiled search did not run. Members are the reasons the reference is used.

    There is NO runtime cross-check that the kernel explored the same set as ``_best_column``:
    a completing kernel returns ``(reduced_cost, column)`` and the inference from "ran to
    completion" to "this is the reference's column" is carried entirely by the test suite at
    build time, by design. So the return value must distinguish "ran" from "declined, run the
    reference", and say WHY -- the reasons warrant opposite responses ("numba isn't installed"
    is a shrug; a saturated partial expansion means the numeric machinery the parity argument
    rests on hit a wall). Only the budget members vary at runtime with the instance and that
    iteration's duals, which is why this cannot be a startup check and the reason has to ride
    on the return value.

    An ``Enum`` because it pickles by name. A plain ``object()`` sentinel pickles happily but
    arrives in a pool worker as a *different instance*, so ``result is sentinel`` is always
    False across a process boundary -- a bug the sequential path (nothing pickled) could never
    catch, so it would surface first on a production timeout under a pool.
    """

    NO_NUMBA = "numba_unavailable"
    MULTI_LEVEL = "multi_level_graph"
    TOPOLOGY = "topology_refused"
    ROWS = "rows_refused"
    NO_DESTINATION = "no_reachable_destination"
    MISSING_CELL = "origin_cell_not_packed"
    #: A forbidden row landed outside the packed clock, so the bitmap does not carry it.
    #: The kernel would then explore a state the reference forbids and return a column the
    #: reference cannot -- a WRONG answer, not a slow one, and the one failure mode the
    #: parity harness cannot catch, because nothing raises. `prepare_forbidden` already
    #: counted these and said "the caller must refuse the compiled path"; this is the
    #: caller doing so.
    FORBIDDEN_UNMAPPED = "forbidden_row_outside_clock"
    LABEL_BUDGET = "label_pool_exhausted"
    STATE_BUDGET = "dominance_table_exhausted"
    HEAP_BUDGET = "heap_exhausted"
    FSUM_OVERFLOW = "fsum_partials_overflowed"
    DEADLINE = "cancelled"
    #: A kernel status with no member of its own.  Unreachable today; it exists so that
    #: adding a status to `dp_kernel` degrades to a fallback rather than to a `KeyError`
    #: raised from the function whose entire job is to decline gracefully.
    KERNEL_STATUS = "unmapped_kernel_status"


def _status_reason(kernel, status) -> Declined:
    """Map a kernel stop status onto its reason. Built per call: `kernel` is imported lazily."""

    return {
        kernel.STATUS_LABEL_LIMIT: Declined.LABEL_BUDGET,
        kernel.STATUS_STATE_LIMIT: Declined.STATE_BUDGET,
        # `feasible_dag`'s heap, which `price_dag` never raises: the two searches share the
        # status code, and only the feasible one grows a heap.
        kernel.STATUS_CANDIDATE_LIMIT: Declined.HEAP_BUDGET,
        kernel.STATUS_FSUM_OVERFLOW: Declined.FSUM_OVERFLOW,
    }.get(status, Declined.KERNEL_STATUS)


def _warn_budget_growth(kernel, fg: FlightGraph, result) -> None:
    """Record and announce what the compiled search's label budget cost this flight.

    Two conditions, warned once per process each, because they call for different responses
    and reporting them as one number would hide the cheaper one behind the louder one:

    * **Restarts.** ``result.attempts > 1`` means a budget filled and the search re-ran from
      its first layer, throwing away every sink certification the previous attempt had
      already paid for.  The answer is unchanged -- a budget bounds work, never the search --
      so this is invisible except as a slow flight (``DagResult.attempts`` is the number to
      read when a flight is unexpectedly expensive).
    * **Declines.** The search stopped without finishing, so this flight fell back to the
      Python reference -- same column, slower.  The advice is split by cause, because the two
      that land here are opposites: a budget status means the pool hit
      :data:`~.dp_kernel.MAX_LABEL_CAPACITY`, which is a knob and not a wall, while
      ``FSUM_OVERFLOW`` means a partial expansion saturated and a SCORE would have been
      wrong -- telling someone to raise a ceiling in that case would be worse than saying
      nothing.

    Per process, so under a worker pool each worker warns for itself.  The counters are
    per process too, which means a parallel sweep's totals live in the workers -- the
    aggregate a parent already sees is ``kernel_fell_back``, which every decline here also
    increments one level up in :func:`price_flight`.

    Parameters
    ------------
    - kernel (module): the compiled ``dp_kernel`` module, supplying the ``STATUS_*`` codes and
      label-capacity constants.
    - fg (FlightGraph): the flight being priced, read for its ``flight_id`` in the warnings.
    - result (DagResult): the compiled search's result -- ``status``, ``n_labels``,
      ``attempts``, and ``budget``.

    Return
    --------
    - output (None): increments the per-process ``_KERNEL_STATS`` counters and prints
      once-per-process warnings to stderr; mutates the module-global warn-once flags.
    """

    global _kernel_restart_warned, _kernel_budget_warned

    # A search that FILLED past `LABEL_HIGH_WATER_WARN` is one that would have declined to the
    # pure-Python reference under the previous, lower ceiling.  It succeeded, so nothing here
    # is wrong -- but its headroom is gone, and the decline warning below only fires once it is
    # too late.  This is the one that fires while there is still time to act (a bootstrap
    # cutoff, a narrower objective, fewer flights per batch).
    #
    # Gated on STATUS_OK, and not merely to be tidy: a search that DECLINED already gets a
    # more specific warning below, and saying "no headroom left" about a search that
    # already ran out of it is noise on top of the real message.  The status test also
    # short-circuits before `n_labels`, which is what lets a caller hand this function a
    # decline that never filled a pool.
    if result.status == kernel.STATUS_OK and result.n_labels >= kernel.LABEL_HIGH_WATER_WARN:
        flight_id = fg.request.flight_id
        if flight_id not in _kernel_high_water_warned:
            _kernel_high_water_warned.add(flight_id)
            print(
                f"WARNING: compiled colgen pricing filled {result.n_labels:,} labels on "
                f"flight {flight_id} -- at or past {kernel.LABEL_HIGH_WATER_WARN:,}, which "
                f"was the ceiling before it was raised to "
                f"{kernel.MAX_LABEL_CAPACITY:,} ({result.n_labels * 40 / 1e9:.2f} GB of "
                f"label arena, PER pricing worker). The answer is correct; the headroom is "
                f"not. This flight declines to the reference search if its demand grows",
                file=sys.stderr,
            )

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

    Warns once per process rather than per flight. The warning exists because the failure is
    silent and expensive: a sweep that quietly ran the reference everywhere looks exactly like
    a slow sweep.
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
      exactly this caller. Sink proposals share path prefixes and corridor start steps, so
      ``(cell, visit_step)`` repeats heavily.
    * the provisional reduced cost itself, per LABEL rather than per lane. Neither
      ``_path_claims`` (which opens with ``del dest_lane_idx``) nor ``_path_delay_s`` takes
      the destination lane, so the reference recomputes an identical number once per lane
      on every multi-lane arrival.

    The two ``forbidden_rows`` gates are applied here rather than in the kernel, which is
    why the kernel is allowed to register a sink the reference rejects: reproducing the two
    endpoint span rules in numba to save work Tier 2 redoes anyway would be a second place
    for them to drift.

    Parameters
    ------------
    - result (DagResult): the compiled search result, supplying ``candidates`` and ``paths``.
    - topology (PreparedTopology): supplies ``cell_q``/``cell_r`` for cell reconstruction.
    - fg (FlightGraph): the flight graph, supplying the request and geometry.
    - cfg (SimConfig): supplies the geometry, clock, and nominal speed.
    - dual_view (DualView): the dual view scoring each sink's rows.
    - pi_f (float): the flight's dual (``pi_f``).
    - benefit (float): the per-flight benefit ``M``.
    - forbidden_rows (AbstractSet[RowKey]): rows that must stay clear.
    - model (CostModel): cost weights for the delay ruler.
    - deadline (float | None): ``time.monotonic`` deadline; ``None`` disables the check.

    Return
    --------
    - output (list[_Candidate]): one priced ``_Candidate`` per sink that clears both
      forbidden-row gates, for Tier 2 to rank.
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
    keep_roots: frozenset[tuple[int, int]] | None = None,
    record_budget: bool = True,
    additional_columns: list[Column] | None = None,
    column_limit: int = 1,
    preparation: dp_prepare.PricingPreparation | None = None,
) -> tuple[float, Column | None] | Declined:
    """``_best_column`` over the compiled search: ``(reduced_cost, column)``, or a reason.

    A 2-tuple means the compiled search ran to completion, and because it reproduces the
    reference's explored set exactly -- same roots, same completion gate, same mid-sweep
    incumbent, same dominance ties -- its sink set is the reference's, so ranking it gives
    the reference's column. A :class:`Declined` means the caller must run ``_best_column``,
    and says which of the eight reachable causes applied.

    Note what completing is NOT: a residual-bound argument over a superset search. This
    kernel reproduces the reference's explored set exactly rather than searching more and
    certifying separately, which is why there is no ``label_limit`` ladder here --
    ``price_dag`` grows its own budgets and either finishes or says it did not. (A
    *bootstrap* round is a different thing and does exist, in ``price_flight``, where both
    searches receive its result; see :func:`_bootstrap_incumbent`.)

    A deadline is enforced two ways, because neither alone is enough: ``_check_deadline``
    between the Python stages, and a watchdog that sets the kernel's ``cancel`` flag, which
    it polls per time layer. An ``@njit(nogil=True)`` function cannot read a clock, and
    with geometric budget growth one call can run for minutes.

    Parameters
    ------------
    - fg (FlightGraph): the flight graph whose space-time DAG is searched.
    - dual_view (DualView): the dual view scoring each candidate's rows.
    - pi_f (float): the flight's dual (``pi_f``).
    - cfg (SimConfig): supplies the geometry, clock, and nominal speed.
    - benefit (float): the per-flight benefit ``M``.
    - forbidden_rows (AbstractSet[RowKey]): rows that must stay clear.
    - incumbent (tuple[float, Column] | None): an already-certified cutoff that warm-starts
      pruning.
    - deadline (float | None): ``time.monotonic`` deadline; ``None`` disables the check.
    - model (CostModel): cost weights for the delay ruler; defaults to ``DELAY_MODEL``.
    - keep_roots (frozenset[tuple[int, int]] | None): allowlist of ``(departure_step,
      lane_idx)`` roots; ``None`` searches every root.
    - record_budget (bool): whether to record and warn about the label budget via
      :func:`_warn_budget_growth`; defaults to ``True``.

    Return
    --------
    - output (tuple[float, Column | None] | Declined): the winning ``(reduced_cost, column)``
      (or ``(-math.inf, None)``) when the compiled search completes, else a ``Declined`` reason
      telling the caller to run ``_best_column``.
    """

    kernel = _dp_kernel()
    if kernel is None:
        return Declined.NO_NUMBA
    if len(fg.levels) != 1:
        return Declined.MULTI_LEVEL

    _check_deadline(deadline)
    if preparation is not None:
        preparation.check(fg, cfg, dual_view, benefit, pi_f, model, forbidden_rows)
    topology, rows = (dp_prepare.prepared_for(fg, cfg) if preparation is None
                      else preparation.topology_rows)
    if not topology.ok:
        return Declined.TOPOLOGY
    if not rows.ok:
        return Declined.ROWS

    duals = (dp_prepare.prepare_duals(dual_view, fg, topology, rows) if preparation is None
             else preparation.duals)
    envelopes = preparation.envelopes(incumbent, deadline) if preparation else dp_prepare.CompletionEnvelopes(
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
        keep_roots=keep_roots,
        preparation=preparation,
    )
    if not variants.ok:
        # Unreachable today, and kept anyway.  `prepare_variants` has exactly two returns and
        # the failing one only ever propagates `topology.unsupported_reason or
        # rows.unsupported_reason`, both of which the guard above already caught -- so this
        # cannot fire without one of those two first.  A root set that the gate pruned to
        # nothing is a different thing entirely: it comes back `ok` with empty arrays and
        # correctly PROVES that no improving column exists.
        return Declined.TOPOLOGY if not topology.ok else Declined.ROWS
    pack = (dp_prepare.prepare_forbidden(forbidden_rows, fg, rows, topology)
            if preparation is None else preparation.forbidden)
    if pack.n_unmapped:
        # Every other Declined here costs time; this one would cost correctness. A dropped
        # forbidden row does not narrow the search, it WIDENS it past what the reference
        # allows, so the kernel can return a column carrying a claim the master forbade.
        return Declined.FORBIDDEN_UNMAPPED

    # `record_budget=False` skips BOTH halves of the graph's budget memo, and it has to be
    # both.  A restricted search shares this cache with the unrestricted one that follows it,
    # and would corrupt it in each direction: reading, a 4-departure bootstrap would allocate
    # the FULL search's pool (up to `MAX_LABEL_CAPACITY`) for a search that needs a sliver of
    # it; writing, the memo would end up holding the bootstrap's tiny budget, and
    # since the write below sits past the `status != OK` guard, a declining flight -- the one
    # this is all for -- would keep that number and make every later iteration re-climb the
    # ladder from it.  Answer-neutral either way; the cost is entirely in wasted work.
    budget = None
    if record_budget:
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
    # Recorded at the same site and for the same reason: every path out of here below this
    # line is a decline, and a decline is exactly the case a straggler hunt needs the
    # numbers for.  Plain ints and a str so this pickles back from a pool worker without
    # dragging `DagResult` or the kernel module across the boundary.
    _LAST_SEARCH.update(
        flight_id=int(fg.request.flight_id),
        n_labels=int(result.n_labels),
        attempts=int(result.attempts),
        label_budget=int(result.budget[0]),
        status=kernel.STATUS_NAMES.get(result.status, str(result.status)),
    )

    if result.status == kernel.STATUS_CANCELLED:
        # The watchdog fired, so the deadline has passed.  Falling back to the reference
        # here would spend the caller's whole remaining budget re-running what was just
        # abandoned; `solver.py` turns this into `termination_reason = "time_limit"`.
        _check_deadline(deadline)
        return Declined.DEADLINE
    if result.status != kernel.STATUS_OK:
        return _status_reason(kernel, result.status)

    if record_budget:
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
        additional_columns=additional_columns, column_limit=column_limit,
    )
    return (-math.inf, None) if best is None else best


def _cancel_search(flag) -> None:
    """Watchdog body: ask the kernel to stop at its next time layer."""

    flag[0] = 1


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
    """``find_feasible_column``'s search, compiled; a :class:`Declined` when it cannot run.

    The **start loop stays in Python** and the kernel gets its result. That is not laziness:
    the guards need ``_endpoint_claims`` sets and the reference's own ``break`` on the
    incumbent's delay, and the loop runs a few hundred times against the search's hundreds of
    thousands of arc relaxations.

    Every sink still goes back to Python, because ``_canonical_candidate`` is the exact gate
    that judges them and it reaches the whole geometry stack. There are a few hundred sinks
    against those same hundreds of thousands of arcs, which is what makes pausing per sink
    affordable.

    Parameters
    ------------
    - fg (FlightGraph): the flight's space-time graph; must have been built with ``cfg``.
    - cfg (SimConfig): the config used to build ``fg``; supplies geometry and the clock.
    - forbidden (AbstractSet[RowKey]): rows a feasible column may not claim.
    - best_column (Column | None): the incumbent to beat; bounds the start loop and ranking.
    - improve_below_delay_s (float | None): when set, stop at the first certified column whose
      delay is strictly below it.
    - origin_options (Iterable): the origin start options as ``(lane_idx, cell, steps)`` tuples.
    - offsets (tuple[int, int]): the inclusive derived cell window ``(lo, hi)``.
    - origin_fold_lb (Mapping): per-lane origin fold, lane index -> ``(fold_s, exact)``.
    - destination_fold_lb (float): the destination fold leg lower bound in seconds.
    - destination_fold_exact (bool): whether the destination fold retains its cell.
    - reference_time_s (float): the reference (geodesic) flight time in seconds.
    - reference_m (float): the reference distance in metres, gating exact folding.
    - remaining_distance (Callable): maps a cell to its admissible remaining hop count.
    - delay_bound (Callable): maps ``(departure_step, lane_idx, hops, remaining)`` to a delay
      lower bound.
    - column_key (Callable): maps a column to the canonical sort key used to break ties.
    - view (DualView): the (empty) dual view used when certifying candidates.
    - deadline (float | None): ``time.monotonic`` cutoff; reaching it raises ``PricingTimeout``.
    - model (CostModel): the objective the bounds and canonical gate use.

    Return
    --------
    - output (Column | None | Declined): the best (or first-improving) feasible column,
      ``best_column`` when no root survives, or a ``Declined`` reason when the compiled search
      cannot run.
    """

    kernel = _dp_kernel()
    if kernel is None:
        return Declined.NO_NUMBA
    if len(fg.levels) != 1:
        return Declined.MULTI_LEVEL
    topology, rows = dp_prepare.prepared_for(fg, cfg)
    if not topology.ok:
        return Declined.TOPOLOGY
    if not rows.ok:
        return Declined.ROWS
    if topology.dest_lane_idx.size == 0:
        return Declined.NO_DESTINATION
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
                return Declined.MISSING_CELL
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
    if pack.n_unmapped:
        # Same refusal as `_best_column_compiled`'s, and it matters here too: this arm feeds
        # the greedy's incumbent, and an incumbent carrying a forbidden claim is committed
        # to the schedule rather than merely priced.
        return Declined.FORBIDDEN_UNMAPPED
    state: dict[str, Any] = {"best": best_column}

    def certify(departure_step, origin_lane, dest_lane, step, hops, path):
        """``find_feasible_column``'s per-sink block, arm for arm."""

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
        return _status_reason(kernel, status)
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

    Deliberately an incumbent heuristic, not the reduced-cost oracle: it runs only after the
    first LP and therefore cannot contribute to a global pricing bound.  Static adjacency and
    the certified seed are reused, while row exclusions and labels remain call-local.  A
    raw-hex delay lower bound orders the frontier; the exact canonical gate judges every sink.
    The ``improve_below_delay_s`` early exit is useful for incumbent construction but is
    intentionally unavailable to formal pricing.

    Parameters
    ------------
    - fg (FlightGraph): the flight's space-time graph; must have been built with ``cfg``.
    - cfg (SimConfig): the config used to build ``fg``; a mismatch raises ``ValueError``.
    - forbidden_rows (AbstractSet[RowKey]): rows a feasible column may not claim (the repair
      seam's saturated set).
    - improve_below_delay_s (float | None): when set, return the first certified column whose
      delay is strictly below this; ``None`` searches for the best feasible column.
    - deadline (float | None): monotonic wall-clock cutoff; reaching it raises
      :class:`PricingTimeout`.
    - model (CostModel): objective the delay lower bound and canonical gate use.

    Return
    --------
    - output (Column | None): the best (or first-improving) feasible column, or ``None`` when
      the flight has none.
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
    # Memoized like ``_best_column``'s: the value depends only on the cell and the fixed
    # destination set, and the corridor holds a few thousand cells against millions of arc
    # relaxations, so this is nearly all hits.
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
        seed = certify_column(seed, fg, cfg, model)
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
    # A `Declined` -- not `None` -- because `None` is a legitimate answer here: a flight with
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
    if not isinstance(compiled, Declined):
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
            # Once per relaxed arc, and the set was built only to be tested.
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
    heuristic_only: bool = False,
    known_column: Column | None = None,
    deadline: float | None = None,
    additional_columns: list[Column] | None = None,
) -> tuple[float, Column | None]:
    """Return the best positive-reduced-cost column for one flight.

    ``heuristic_only`` returns after the restricted bootstrap. Its score is a
    feasible reduced cost, NOT the subproblem optimum: it must never certify a
    global bound or absence of improving columns. The normal exact path and
    repair calls retain their existing behavior.

    ``known_column`` is a column the caller already holds -- the restricted master's current
    selection.  Its reduced cost under the current duals is a *proven achievable* score, so it
    is a valid pruning cutoff and a far better one than the shortest-path seed this function
    otherwise builds for itself.  Cutoff quality dominates every other lever in this search, so
    this is the argument to preserve when changing anything below.  Passing it never changes
    the answer, only the work: pruning against an attainable score cannot discard anything
    strictly better.  A column equal to the one supplied is reported as no column at all, so
    the caller cannot mistake what it already holds for pricing progress.

    ``forbidden_rows`` is the repair seam: touching one of those already-saturated rows makes
    an origin, visit, arrival, or final canonical column infeasible rather than merely
    expensive.  Repair also sets ``require_improving=False`` because it needs the best feasible
    trajectory even when a user-supplied benefit ``M`` is smaller than that trajectory's delay.
    The returned reduced cost is always recomputed from the canonical de-duplicated claim set.

    Parameters
    ------------
    - fg (FlightGraph): the flight's space-time graph; must have been built with ``cfg``.
    - duals (Mapping[RowKey | tuple, float] | DualView): current master-row duals.
    - pi_f (float): the flight-row dual; must be finite.
    - cfg (SimConfig): the config used to build ``fg``; a mismatch raises ``ValueError``.
    - params (Any): colgen params; supplies ``M`` (benefit), the objective, and the bootstrap.
    - forbidden_rows (AbstractSet[RowKey]): rows a returned column may not claim.
    - require_improving (bool): when True, return no column unless its reduced cost clears
      ``_IMPROVING_RC_TOL``.
    - known_column (Column | None): the caller's current column, used as a cutoff.
    - deadline (float | None): monotonic wall-clock cutoff; reaching it raises
      :class:`PricingTimeout`.

    Return
    --------
    - output (tuple[float, Column | None]): the reduced cost and the best improving column, or
      ``(reduced_cost, None)`` when none improves or the best equals ``known_column``.
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
    if known_column is not None:
        known_column = certify_column(known_column, fg, cfg, model)
    if known_column is not None and known_column.claims.isdisjoint(forbidden):
        known_rc = model.reduced_cost(
            benefit=benefit,
            cost=known_column.delay_s,
            dual_cost=view.claim_cost(known_column.claims),
            pi_f=pi_value,
        )
        if incumbent is None or known_rc > incumbent[0] + _SCORE_EPS:
            incumbent = (known_rc, known_column)
    # Everything folded in above is a column the master already holds, so LP optimality caps
    # its reduced cost at 0 and the search would enter with no usable cutoff at all.  The
    # bootstrap is the only way to a positive one -- see `_bootstrap_incumbent`.
    #
    # HERE, in the shared caller, and deliberately not inside the two searches.  `incumbent`
    # is computed once and handed to `_best_column_compiled` and to the `_best_column`
    # fallback below as the same object, so a bootstrap at this seam reaches BOTH by
    # construction: whichever one runs explores the space the other would have.  Putting it
    # in two bodies is how they drift, and the `Declined` contract -- that the compiled
    # search explored exactly what the reference would -- is what would drift first.
    _bootstrap_s = 0.0
    _bootstrap_labels = 0
    _bootstrap_goal_labels = 0
    _bootstrap_goal_s = 0.0
    _bootstrap_dp_labels = 0
    preparation = dp_prepare.PricingPreparation(
        fg, cfg, view, benefit=benefit, pi_f=pi_value, model=model, forbidden_rows=forbidden
    )
    if params.bootstrap_roots or heuristic_only:
        _bootstrap_started = time.perf_counter()
        incumbent = _bootstrap_incumbent(
            fg,
            view,
            pi_value,
            cfg,
            benefit,
            forbidden,
            model,
            incumbent=incumbent,
            roots=max(1, params.bootstrap_roots) if heuristic_only else params.bootstrap_roots,
            search_single_root=heuristic_only,
            ranking=getattr(params, "bootstrap_ranking", "score"),
            method=params.bootstrap_method,
            max_labels=params.bootstrap_max_labels,
            deadline=deadline,
            preparation=preparation,
        )
        _bootstrap_s = time.perf_counter() - _bootstrap_started
        # Snapshot NOW: `_bootstrap_incumbent` runs its own restricted
        # `_best_column_compiled`, which writes `_LAST_SEARCH`, and the main search below
        # overwrites it. Read late and this reports the main search's labels twice.
        _bootstrap_goal_labels = int(_LAST_SEARCH.get("bootstrap_goal_labels", 0))
        _bootstrap_goal_s = float(_LAST_SEARCH.get("bootstrap_goal_s", 0.0))
        _bootstrap_dp_labels = int(_LAST_SEARCH.get("n_labels", 0))
        _bootstrap_labels = _bootstrap_goal_labels + _bootstrap_dp_labels
    _bootstrap_record = dict(
        bootstrap_s=_bootstrap_s, bootstrap_labels=_bootstrap_labels,
        bootstrap_goal_labels=_bootstrap_goal_labels, bootstrap_goal_s=_bootstrap_goal_s,
        bootstrap_dp_labels=_bootstrap_dp_labels,
    )
    if heuristic_only:
        _KERNEL_STATS["cheap_priced"] += 1
        reduced_cost, column = incumbent if incumbent is not None else (-math.inf, None)
        _LAST_SEARCH.update(heuristic_only=True, **_bootstrap_record, final_rc=float(reduced_cost))
        if (column is None or column == known_column
                or (require_improving and reduced_cost <= _IMPROVING_RC_TOL)):
            return reduced_cost, None
        return reduced_cost, column
    # The compiled search first, the reference when it cannot prove it ran to completion.
    # `forbidden_rows` deliberately does NOT force the fallback: repair is O(flights) inside
    # the greedy, so a Python round trip per repair would be a scaling cliff at thousands of
    # flights, and the exclusion set is a bitset over dense row ids inside the kernel.
    _compiled_started = time.perf_counter()
    outcome = _best_column_compiled(
        fg,
        view,
        pi_value,
        cfg,
        benefit,
        forbidden,
        incumbent=incumbent,
        deadline=deadline,
        model=model,
        preparation=preparation,
        **({"additional_columns": additional_columns, "column_limit": params.columns_per_flight}
            if additional_columns is not None else {}),
    )
    # Split here rather than timing the whole call, because on a flight that DECLINES the
    # two halves want opposite fixes and a combined number cannot tell them apart: most of a
    # declining straggler's wall is the pure-Python fallback, not the compiled ladder that
    # triggered it, and raising the label ceiling attacks only the smaller half. Both are
    # written unconditionally so a flight that never fell back still reports `fallback_s = 0.0`
    # rather than a missing key.
    _LAST_SEARCH["compiled_s"] = time.perf_counter() - _compiled_started
    _LAST_SEARCH["fallback_s"] = 0.0
    _LAST_SEARCH["declined"] = isinstance(outcome, Declined)
    # The bootstrap is a SEPARATE `_best_column_compiled` call at the seam above, so it is not
    # inside `compiled_s`; timed on its own so a straggler's bootstrap wall is not lost between
    # the two fields.
    _LAST_SEARCH.update(_bootstrap_record)
    # What the main search actually ENTERS with, which is the number that separates the two
    # explanations for an expensive flight: a cutoff at or near the final reduced cost means
    # the cutoff was fine and the labels went somewhere else. `-inf` is no incumbent at all,
    # the worst case for pruning, and LP optimality pins this at 0 without a bootstrap.
    _LAST_SEARCH["entry_rc"] = (
        float("-inf") if incumbent is None else float(incumbent[0])
    )
    _KERNEL_STATS["priced"] += 1
    if isinstance(outcome, Declined):
        _KERNEL_STATS["fell_back"] += 1
        _LAST_SEARCH["declined_reason"] = outcome.value
        # The reason, not just the count.  `fell_back` alone cannot be acted on -- "install
        # numba" and "a partial expansion saturated on real data" are opposite responses --
        # and this is the only place both are in scope.  `solver.py` sums these into
        # `planner_stats.json`, so an archived run can say WHICH.
        _KERNEL_STATS[f"declined_{outcome.value}"] += 1
        # The ORIGINAL incumbent, deliberately, not whatever the abandoned compiled attempt
        # managed to certify first.  Warm-starting the fallback would be optimality-safe --
        # pruning against an achievable score never discards anything strictly better -- but
        # it is not parity-safe: a stronger cutoff explores less than the reference did and
        # can return a different, equally optimal column.  The fallback exists to reproduce
        # the oracle, and it is rare enough that its speed is not the thing to optimize.
        _fallback_started = time.perf_counter()
        # `finally`, because `_best_column` RAISES `PricingTimeout` when it reaches the
        # deadline -- and `_price_one` deliberately still ships the record for a timed-out
        # flight, since that is the most interesting row in a straggler hunt. Recording only
        # on the success path leaves `fallback_s = 0.0` on exactly the flights where the
        # fallback consumed the rest of the task, which is the misattribution this split
        # exists to prevent.
        try:
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
                **({"additional_columns": additional_columns, "column_limit": params.columns_per_flight}
                    if additional_columns is not None else {}),
            )
        finally:
            _LAST_SEARCH["fallback_s"] = time.perf_counter() - _fallback_started
    else:
        reduced_cost, column = outcome
    # Paired with `entry_rc` above, and the pair is the point: `final_rc - entry_rc` is how
    # much of the answer the cutoff did NOT already know. Near zero means the bootstrap
    # handed the search a bound worth having and the labels went elsewhere; a large gap
    # means the search discovered the answer itself and the cutoff was doing nothing.
    _LAST_SEARCH["final_rc"] = float(reduced_cost)
    if additional_columns is not None:
        additional_columns[:] = [c for c in additional_columns
                                 if c != column and c != known_column]

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

    Only the nominal departure is considered.  With zero row prices the DAG minimizes ground
    hold, lateral hops, and endpoint fold/snap legs; shortest paths never exercise the
    short-revisit restriction, so this is the plan's unconstrained shortest-path seed while
    retaining the canonical wall and detour gates.

    Parameters
    ------------
    - fg (FlightGraph): the flight's space-time graph; must have been built with ``cfg``.
    - cfg (SimConfig): the config used to build ``fg``; a mismatch raises ``ValueError``.
    - deadline (float | None): monotonic wall-clock cutoff; reaching it raises
      :class:`PricingTimeout`.
    - model (CostModel): objective the seed's delay is scored under (the cache is keyed on it).

    Return
    --------
    - output (Column): the seed column; raises ``ValueError`` when the flight has no feasible
      seed.
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
