"""Flat-array packing of one flight's pricing subproblem, for the compiled DP.

Everything here is pure Python and pure NumPy; the module deliberately does not
import Numba so ``colgen`` still works without the compiled extra.  The split is:

* :class:`PreparedTopology` restates the *dual-independent* graph — cells, arcs,
  role masks, reverse distances — as dense arrays.  It is a mirror of what
  :meth:`FlightGraph.outgoing_neighbors` and :meth:`FlightGraph.hop_role_mask`
  already computed, so it is answer-neutral and cacheable for a graph's whole
  life.  Building it is the one place lazy arc expansion is deliberately drained.
* :class:`PreparedDuals` restates one iteration's row prices.  Duals change every
  column-generation iteration, so this is rebuilt per pricing call.

The reference search in :mod:`.pricing` remains the oracle.  Nothing here may
change an answer; the unit tests assert arc-for-arc and value-for-value identity
against the object API.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .network import RowKey

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config import SimConfig
    from .network import FlightGraph
    from .pricing import DualView

Cell = tuple[int, int]

# Sentinel for "no path to any destination".  Deliberately not INT32_MAX: the
# kernel evaluates ``step + 1 + rev_remaining[cell] > max_step``, and INT32_MAX
# would overflow that addition.  2**24 dwarfs any realistic step count while
# leaving ~127x headroom inside int32.
UNREACHABLE = 1 << 24


@dataclass(frozen=True, slots=True)
class PreparedTopology:
    """Dense, dual-independent mirror of one flight's spatial search domain."""

    # Cell interning.  Index order is the sorted axial order, so it is stable
    # across processes and independent of BFS discovery order.
    cell_q: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    cell_r: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))

    # CSR adjacency.  Arc order within a cell matches ``outgoing_neighbors``
    # (that is, ``hexgrid.AXIAL_NEIGHBORS`` order), which is the order the
    # reference DP iterates; reordering would silently change which label wins
    # an insertion-order tie.
    arc_start: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(1, np.int32))
    arc_target: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    arc_roles: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.uint8))

    # Admissible hop count from each cell to the nearest destination, over the
    # any-role arc superset.  A lower bound on every role-specific completion.
    rev_remaining: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))

    dest_mask: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.uint8))
    dest_lane_start: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(1, np.int32))
    dest_lane_idx: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))

    # Origin start options, parallel arrays over ``_origin_options`` order.
    origin_lane_idx: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    origin_cell: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    origin_lane_steps: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))

    # Clock and search-shape scalars lifted off the graph.
    base_step: int = 0
    latest_departure_step: int = 0
    min_step: int = 0
    max_step: int = 0
    takeoff_steps0: int = 0
    seed_hop_limit: int = 0
    revisit_depth: int = 0
    state_history_depth: int = 0
    track_first_hop: bool = False

    unsupported_reason: str | None = None

    @property
    def n_cells(self) -> int:
        return int(self.cell_q.shape[0])

    @property
    def n_arcs(self) -> int:
        return int(self.arc_target.shape[0])

    def index_of(self, cell: Cell) -> int:
        """Return the dense index of ``cell``, or ``-1``.

        Linear-search-free but Python-side only: the kernel never needs this,
        because preparation hands it indices everywhere.  Used by tests and by
        dual packing.
        """

        found = np.flatnonzero((self.cell_q == cell[0]) & (self.cell_r == cell[1]))
        return int(found[0]) if found.size else -1


@dataclass(frozen=True, slots=True)
class PreparedDuals:
    """One iteration's cell-row prices as flat prefix sums, indexed by cell id.

    A copy of :class:`~.pricing.DualView`'s per-resource ``_PrefixSeries``, keyed
    by dense cell index instead of ``(cell, level)``.  Rebuilt every pricing call
    because duals change every column-generation iteration; the topology it is
    indexed against does not.
    """

    dual_first: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    dual_start: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(1, np.int32))
    dual_prefix: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))
    # Prefix sums of max(0, dual), parallel to dual_prefix.  Positive-part sums are
    # not additive, so they cannot be recovered from dual_prefix; the endpoint
    # bound needs them and would otherwise have to rebuild RowKey sets per step.
    pos_prefix: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))
    window_lo: int = 0
    window_hi: int = 0
    max_negative_credit: float = 0.0
    has_active_duals: bool = False

    def visit_cost(self, cell_index: int, visit_step: int) -> float:
        """Price one cell visit's whole row window.

        Mirrors ``DualView.visit_cost`` -> ``_PrefixSeries.range_sum`` operation for
        operation, including the clamping, so the kernel and the object oracle agree
        on which rows a window touches.  Kept in Python purely as the executable
        specification the kernel is tested against.
        """

        lo_off = int(self.dual_start[cell_index])
        length = int(self.dual_start[cell_index + 1]) - lo_off
        if length <= 1:
            return 0.0
        first = int(self.dual_first[cell_index])
        series_stop = first + length - 1
        start = visit_step + self.window_lo
        stop = visit_step + self.window_hi + 1
        a = min(max(start, first), series_stop)
        b = min(max(stop, first), series_stop)
        if b <= a:
            return 0.0
        return float(self.dual_prefix[lo_off + b - first] - self.dual_prefix[lo_off + a - first])


def _positive_range(duals: PreparedDuals, cell_index: int, start: int, stop: int) -> float:
    """Sum max(0, dual) over a half-open step interval, clamped like range_sum."""

    lo_off = int(duals.dual_start[cell_index])
    length = int(duals.dual_start[cell_index + 1]) - lo_off
    if length <= 1:
        return 0.0
    first = int(duals.dual_first[cell_index])
    series_stop = first + length - 1
    a = min(max(start, first), series_stop)
    b = min(max(stop, first), series_stop)
    if b <= a:
        return 0.0
    return float(duals.pos_prefix[lo_off + b - first] - duals.pos_prefix[lo_off + a - first])


def prepare_duals(view: DualView, topology: PreparedTopology) -> PreparedDuals:
    """Re-index one ``DualView``'s cell prefix sums onto dense cell ids."""

    n = topology.n_cells
    dual_first = np.zeros(n, dtype=np.int32)
    dual_start = np.zeros(n + 1, dtype=np.int32)
    chunks: list[np.ndarray] = []
    pos_chunks: list[np.ndarray] = []
    index = {
        (int(q), int(r)): i
        for i, (q, r) in enumerate(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    }
    # ``_cell`` is keyed by (cell, level); colgen v1 pricing is single-level, and
    # ``prepare_topology`` already refused anything else.
    by_index: dict[int, Any] = {}
    for (cell, level), series in view._cell.items():
        if level != 0:
            continue
        cell_index = index.get(cell)
        if cell_index is not None:
            by_index[cell_index] = series

    total = 0
    for i in range(n):
        series = by_index.get(i)
        if series is not None:
            dual_first[i] = series.first_step
            prefix = np.asarray(series.prefix, dtype=np.float64)
            chunks.append(prefix)
            steps = np.diff(prefix)
            pos_chunks.append(
                np.concatenate((np.zeros(1), np.cumsum(np.maximum(steps, 0.0))))
            )
            total += len(series.prefix)
        dual_start[i + 1] = total
    dual_prefix = (
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    )
    pos_prefix = (
        np.concatenate(pos_chunks) if pos_chunks else np.empty(0, dtype=np.float64)
    )
    for array in (dual_first, dual_start, dual_prefix, pos_prefix):
        array.setflags(write=False)
    lo, hi = view.offsets
    return PreparedDuals(
        dual_first=dual_first,
        dual_start=dual_start,
        dual_prefix=dual_prefix,
        pos_prefix=pos_prefix,
        window_lo=lo,
        window_hi=hi,
        max_negative_credit=view.max_negative_credit,
        has_active_duals=view.has_active_duals,
    )


@dataclass(frozen=True, slots=True)
class PreparedVariants:
    """Start options crossed with departure steps, plus their endpoint pricing.

    Mirrors the initialization loop of ``pricing._best_column``.  One "variant" is
    one ``(departure_step, origin lane)`` pair: the reference creates exactly one
    root label per variant, so the kernel does too.

    ``paid_class`` deserves care.  The reference's dominance state key holds the
    *set* ``origin_paid_rows``, not the departure step -- so two roots from
    different departure steps that happen to have paid the same rows are allowed
    to merge downstream.  Keying on the variant id instead would keep them apart:
    still optimal (a finer state never loses a completion) but it would explore
    more labels and, worse, could break a tie the reference breaks the other way.
    So distinct paid-row sets are interned to a dense class id and the kernel keys
    on that, reproducing the reference's merge behaviour exactly.
    """

    departure_step: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    lane_idx: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    cell: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    start_step: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    score: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))
    origin_leg_s: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))
    ground_delay_s: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))
    origin_fold_s: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))
    origin_fold_exact: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.uint8))
    paid_class: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))

    # Paid-row corrections, CSR over paid CLASS (not variant).
    paid_start: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(1, np.int32))
    paid_cell: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    paid_step: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    paid_value: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))

    destination_fold_s: float = 0.0
    destination_fold_exact: bool = True
    reference_time_s: float = 0.0

    # Positive dual price of the rows every completion arriving at a destination
    # cell must pay, indexed [dest slot, arrival step - step_base].  This is the
    # reference's ``destination_positive_costs`` (pricing.py:994).  Dropping it and
    # bounding a sink by its along-path duals alone is still *admissible*, but it
    # leaves the candidate ranking so loose that the host's early break never fires
    # and certification -- not the search -- becomes the bottleneck.  Computed with
    # ``timing_steps=0``, whose narrower rounding pad yields a subset of the real
    # row set and therefore still a valid lower bound on the union's positive price.
    dest_slot_of_cell: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    dest_positive: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros((0, 0), np.float64)
    )
    dest_step_base: int = 0

    @property
    def n_variants(self) -> int:
        return int(self.departure_step.shape[0])


def prepare_variants(
    fg: FlightGraph,
    cfg: SimConfig,
    view: DualView,
    topology: PreparedTopology,
    *,
    seed: bool,
    benefit: float = 0.0,
    pi_f: float = 0.0,
    cost_cutoff: float | None = None,
) -> PreparedVariants:
    """Price every root option once, exactly as ``_best_column`` initialization does.

    ``cost_cutoff`` enables the same cheap ground-delay pre-filter the reference
    applies at the top of its initialization loop, *before* building any endpoint
    claim set.  It matters more here than there: a graph with
    ``max_ground_delay_s=3600`` has 901 departure steps, and constructing the
    origin endpoint rows for every one of them costs far more than the compiled
    search that consumes them.  A departure whose ground delay alone cannot beat
    the incumbent can never win, whatever route follows it.
    """

    from ...volumes import enroute_reference_m
    from .pricing import (
        _arc_delay_lower_bound_s,
        _destination_options,
        _endpoint_claims,
        _fold_leg_s,
        _origin_options,
        _terminal_fold_leg_s,
        _visit_claims,
    )

    offsets = view.offsets
    cell_index = {
        (int(q), int(r)): i
        for i, (q, r) in enumerate(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    }

    # Endpoint legs, computed once per lane exactly as the reference does.
    origin_leg_by_lane: dict[int | None, float] = {}
    origin_fold_by_lane: dict[int | None, tuple[float, bool]] = {}
    for lane_idx, cell, _steps in _origin_options(fg):
        lane_dist = None if lane_idx is None else fg.origin_lanes[lane_idx].dist
        origin_leg_by_lane[lane_idx] = _fold_leg_s(
            fg.request.origin, fg.origin_terminal, lane_dist, cfg
        )
        origin_fold_by_lane[lane_idx] = (
            (origin_leg_by_lane[lane_idx], True)
            if fg.origin_terminal is None
            else _terminal_fold_leg_s(fg.request.origin, fg.origin_terminal, cell, cfg)
        )

    destination_options = _destination_options(fg)
    destination_fold_exact = True
    if fg.dest_terminal is None:
        destination_fold_s = _fold_leg_s(fg.request.dest, None, None, cfg)
    else:
        folds: list[float] = []
        for destination, lane_indices in destination_options.items():
            for _lane in lane_indices:
                fold_s, retained = _terminal_fold_leg_s(
                    fg.request.dest, fg.dest_terminal, destination, cfg
                )
                folds.append(fold_s)
                destination_fold_exact &= retained
        destination_fold_s = min(folds)

    reference_m = enroute_reference_m(
        fg.request.origin, fg.request.dest, fg.origin_terminal, fg.dest_terminal, cfg
    )

    departure_steps = (
        (fg.base_step,) if seed else range(fg.base_step, fg.latest_departure_step + 1)
    )

    paid_classes: dict[frozenset[RowKey], int] = {}
    paid_rows_by_class: list[list[tuple[int, int, float]]] = []
    rows: list[dict[str, Any]] = []
    _RECOMPUTE_EPS = 1e-8
    for departure_step in departure_steps:
        ground_delay_s = (departure_step - fg.base_step) * cfg.dt_s
        if cost_cutoff is not None:
            # pricing.py's identical guard, and for the same reason: ground delay is
            # irrevocable, so it upper-bounds every completion from this departure.
            start_upper_bound = benefit - ground_delay_s - pi_f + view.max_negative_credit
            if start_upper_bound < cost_cutoff - _RECOMPUTE_EPS:
                continue
        origin_claims = _endpoint_claims(
            fg, cfg, origin=True, step=departure_step, timing_steps=0
        )
        for lane_idx, cell, lane_steps in _origin_options(fg):
            index = cell_index.get(cell)
            if index is None:
                continue
            start_step = departure_step + fg.takeoff_steps[0] + lane_steps
            if start_step >= fg.max_step:
                continue
            distance_to_go = int(topology.rev_remaining[index])
            if start_step + distance_to_go > fg.max_step:
                continue
            if seed and distance_to_go > topology.seed_hop_limit:
                continue
            if cost_cutoff is not None:
                # Per-LANE bound, strictly tighter than the departure-level one above.
                # It is the same admissible delay bound the kernel applies at every
                # label (``_arc_delay_lower_bound_s``), evaluated at hops=0 with the
                # reverse-BFS distance from this lane's cell -- so it charges the lane
                # for its own fold leg AND for how far it leaves the flight from the
                # destination.  An exit lane on the far side of the terminal therefore
                # prunes itself, without any heuristic "points the wrong way" test: a
                # far-side lane that can still win, because the near side is congested,
                # keeps a bound above the cutoff and survives.
                fold_s, fold_exact = origin_fold_by_lane[lane_idx]
                delay_lb = _arc_delay_lower_bound_s(
                    ground_delay_s=ground_delay_s,
                    origin_fold_s=fold_s,
                    hops=0,
                    remaining_hops=distance_to_go,
                    destination_fold_s=destination_fold_s,
                    reference_time_s=reference_m / cfg.nominal_speed_mps,
                    dt_s=cfg.dt_s,
                    folding_exact=(
                        reference_m > 1e-9 and fold_exact and destination_fold_exact
                    ),
                )
                if benefit - delay_lb - pi_f + view.max_negative_credit < (
                    cost_cutoff - _RECOMPUTE_EPS
                ):
                    continue
            start_claims = origin_claims | _visit_claims(cell, 0, start_step, offsets)
            start_dual_cost = view.claim_cost(start_claims)
            origin_paid_rows = view.active_claims(start_claims)

            paid_class = paid_classes.get(origin_paid_rows)
            if paid_class is None:
                paid_class = len(paid_rows_by_class)
                paid_classes[origin_paid_rows] = paid_class
                # Only single-level cell rows can recur inside a later visit
                # window; terminal rows never can, so they are dropped.
                entries: list[tuple[int, int, float]] = []
                for row in origin_paid_rows:
                    if row.kind != "cell" or row.level != 0:
                        continue
                    row_index = cell_index.get(row.cell_coord)
                    if row_index is not None:
                        entries.append((row_index, row.step, view.row_cost(row)))
                entries.sort()
                paid_rows_by_class.append(entries)

            origin_fold_s, origin_fold_exact = origin_fold_by_lane[lane_idx]
            rows.append(
                {
                    "departure_step": departure_step,
                    "lane_idx": -1 if lane_idx is None else lane_idx,
                    "cell": index,
                    "start_step": start_step,
                    "score": -ground_delay_s - origin_leg_by_lane[lane_idx] - start_dual_cost,
                    "origin_leg_s": origin_leg_by_lane[lane_idx],
                    "ground_delay_s": ground_delay_s,
                    "origin_fold_s": origin_fold_s,
                    "origin_fold_exact": 1 if origin_fold_exact else 0,
                    "paid_class": paid_class,
                }
            )

    paid_start = np.zeros(len(paid_rows_by_class) + 1, dtype=np.int32)
    flat: list[tuple[int, int, float]] = []
    for i, entries in enumerate(paid_rows_by_class):
        flat.extend(entries)
        paid_start[i + 1] = len(flat)

    # Destination endpoint prices.  The customer cylinder's CELLS are fixed; only
    # its step interval slides with arrival, so this is a prefix-sum sweep rather
    # than 925 RowKey set constructions.  Taking the endpoint rows alone (a subset
    # of the reference's endpoint-union-with-visit) keeps it a valid lower bound.
    from .windows import endpoint_claim_cells, endpoint_claim_steps
    from ...volumes import column_dwell_s

    duals_for_bound = prepare_duals(view, topology)
    dest_cells = [c for c in destination_options if c in cell_index]
    dest_slot_of_cell = np.full(topology.n_cells, -1, dtype=np.int32)
    step_base = fg.min_step
    n_steps = fg.max_step - step_base + 1
    dest_positive = np.zeros((max(1, len(dest_cells)), max(1, n_steps)), dtype=np.float64)
    if fg.dest_terminal is None:
        endpoint_cells = [
            cell_index[c]
            for c in endpoint_claim_cells(fg.request.dest, cfg.effective_hover_radius_m, cfg)
            if c in cell_index
        ]
        z = fg.levels[0]
        dwell = cfg.hover_time_s + column_dwell_s(fg.request.dest, fg.dest_terminal, cfg, z)
        for arrival in range(step_base, fg.max_step + 1):
            t0 = arrival * cfg.dt_s
            steps = endpoint_claim_steps(t0, t0 + dwell, cfg, timing_steps=0)
            total = 0.0
            for ci in endpoint_cells:
                total += _positive_range(duals_for_bound, ci, steps.start, steps.stop)
            for slot in range(len(dest_cells)):
                dest_positive[slot, arrival - step_base] = total
    for slot, destination in enumerate(dest_cells):
        dest_slot_of_cell[cell_index[destination]] = slot

    def column(name: str, dtype) -> np.ndarray:
        return np.asarray([row[name] for row in rows], dtype=dtype)

    return PreparedVariants(
        departure_step=column("departure_step", np.int32),
        lane_idx=column("lane_idx", np.int32),
        cell=column("cell", np.int32),
        start_step=column("start_step", np.int32),
        score=column("score", np.float64),
        origin_leg_s=column("origin_leg_s", np.float64),
        ground_delay_s=column("ground_delay_s", np.float64),
        origin_fold_s=column("origin_fold_s", np.float64),
        origin_fold_exact=column("origin_fold_exact", np.uint8),
        paid_class=column("paid_class", np.int32),
        paid_start=paid_start,
        paid_cell=np.asarray([c for c, _s, _v in flat], dtype=np.int32),
        paid_step=np.asarray([s for _c, s, _v in flat], dtype=np.int32),
        paid_value=np.asarray([v for _c, _s, v in flat], dtype=np.float64),
        dest_slot_of_cell=dest_slot_of_cell,
        dest_positive=dest_positive,
        dest_step_base=step_base,
        destination_fold_s=destination_fold_s,
        destination_fold_exact=destination_fold_exact,
        reference_time_s=reference_m / cfg.nominal_speed_mps,
    )


def _reachable_cells(fg: FlightGraph, seeds: list[Cell]) -> list[Cell]:
    """Forward-reachable cells, expanding arcs lazily and never materializing.

    Walks ``outgoing_neighbors``, which both expands-and-caches the arc oracle and
    already filters to arcs admissible in at least one role.  Iterating
    ``fg.corridor_cells`` instead would materialize the lazy ellipse, which is
    exactly what the lazy-expansion commit exists to avoid.
    """

    seen: set[Cell] = set()
    queue: deque[Cell] = deque()
    for seed in seeds:
        if seed not in seen and seed in fg.corridor_cells:
            seen.add(seed)
            queue.append(seed)
    while queue:
        for target in fg.outgoing_neighbors(queue.popleft()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    # Sorted, so the dense index is a function of the cell set alone.
    return sorted(seen)


def _reverse_remaining(
    n_cells: int,
    arc_start: np.ndarray,
    arc_target: np.ndarray,
    destinations: list[int],
) -> np.ndarray:
    """Multi-source reverse BFS giving admissible hops-to-destination per cell."""

    remaining = np.full(n_cells, UNREACHABLE, dtype=np.int32)

    # Reverse the CSR once; the DP only ever needs distances, not the reverse
    # adjacency itself, so it stays local.
    in_degree = np.zeros(n_cells + 1, dtype=np.int64)
    for target in arc_target:
        in_degree[int(target) + 1] += 1
    rev_start = np.cumsum(in_degree, dtype=np.int64)
    rev_source = np.empty(arc_target.shape[0], dtype=np.int32)
    cursor = rev_start.copy()
    for source in range(n_cells):
        for a in range(int(arc_start[source]), int(arc_start[source + 1])):
            target = int(arc_target[a])
            rev_source[cursor[target]] = source
            cursor[target] += 1

    queue: deque[int] = deque()
    for destination in destinations:
        if remaining[destination] != 0:
            remaining[destination] = 0
            queue.append(destination)
    while queue:
        cell = queue.popleft()
        next_hops = remaining[cell] + 1
        for a in range(int(rev_start[cell]), int(rev_start[cell + 1])):
            source = int(rev_source[a])
            if next_hops < remaining[source]:
                remaining[source] = next_hops
                queue.append(source)
    return remaining


def prepare_topology(fg: FlightGraph, cfg: SimConfig) -> PreparedTopology:
    """Drain the lazy arc oracle for one flight into dense arrays, once.

    Answer-neutral: every arc and role recorded here is exactly what
    ``fg.outgoing_neighbors`` / ``fg.hop_role_mask`` would return on demand.
    """

    # Imported here, not at module scope: pricing imports this module.
    from .pricing import _destination_options, _origin_options
    from .windows import derive_cell_window

    if len(fg.levels) != 1:
        return PreparedTopology(unsupported_reason="colgen v1 pricing is single-level")

    origin_options = _origin_options(fg)
    destination_options = _destination_options(fg)
    if not origin_options or not destination_options:
        return PreparedTopology(unsupported_reason="no origin or destination option")

    cells = _reachable_cells(fg, [cell for _lane, cell, _steps in origin_options])
    if not cells:
        return PreparedTopology(unsupported_reason="no reachable corridor cell")
    index = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)

    arc_start = np.zeros(n + 1, dtype=np.int32)
    arc_target_list: list[int] = []
    arc_roles_list: list[int] = []
    for i, cell in enumerate(cells):
        for target in fg.outgoing_neighbors(cell):
            target_index = index.get(target)
            if target_index is None:
                # Unreachable-from-origin targets cannot appear: BFS followed the
                # same arcs.  Guard anyway rather than emit a dangling index.
                continue
            arc_target_list.append(target_index)
            arc_roles_list.append(fg.hop_role_mask(cell, target))
        arc_start[i + 1] = len(arc_target_list)
    arc_target = np.asarray(arc_target_list, dtype=np.int32)
    arc_roles = np.asarray(arc_roles_list, dtype=np.uint8)

    dest_mask = np.zeros(n, dtype=np.uint8)
    dest_lane_start = np.zeros(n + 1, dtype=np.int32)
    dest_lane_list: list[int] = []
    for i, cell in enumerate(cells):
        lanes = destination_options.get(cell)
        if lanes is not None:
            dest_mask[i] = 1
            dest_lane_list.extend(-1 if lane is None else int(lane) for lane in lanes)
        dest_lane_start[i + 1] = len(dest_lane_list)

    destinations = [index[cell] for cell in destination_options if cell in index]
    rev_remaining = _reverse_remaining(n, arc_start, arc_target, destinations)

    offsets = derive_cell_window(cfg)
    revisit_depth = offsets[1] - offsets[0]

    origin_kept = [(lane, cell, steps) for lane, cell, steps in origin_options if cell in index]
    if not origin_kept:
        return PreparedTopology(unsupported_reason="no origin option inside the corridor")

    for array in (arc_start, arc_target, arc_roles, rev_remaining, dest_mask):
        array.setflags(write=False)

    return PreparedTopology(
        cell_q=np.asarray([q for q, _r in cells], dtype=np.int32),
        cell_r=np.asarray([r for _q, r in cells], dtype=np.int32),
        arc_start=arc_start,
        arc_target=arc_target,
        arc_roles=arc_roles,
        rev_remaining=rev_remaining,
        dest_mask=dest_mask,
        dest_lane_start=dest_lane_start,
        dest_lane_idx=np.asarray(dest_lane_list, dtype=np.int32),
        origin_lane_idx=np.asarray(
            [-1 if lane is None else lane for lane, _c, _s in origin_kept], dtype=np.int32
        ),
        origin_cell=np.asarray([index[cell] for _l, cell, _s in origin_kept], dtype=np.int32),
        origin_lane_steps=np.asarray([steps for _l, _c, steps in origin_kept], dtype=np.int32),
        base_step=int(fg.base_step),
        latest_departure_step=int(fg.latest_departure_step),
        min_step=int(fg.min_step),
        max_step=int(fg.max_step),
        takeoff_steps0=int(fg.takeoff_steps[0]),
        seed_hop_limit=int(fg.shortest_hops + fg.detour_slack_hops),
        revisit_depth=revisit_depth,
        # Two different widths, deliberately.  ``revisit_depth`` bans re-entering a
        # recently-held cell; the STATE key must keep at least the predecessor even
        # when that ban is narrower, because two equal-score labels reaching one
        # cell from different predecessors can de-duplicate the destination endpoint
        # union differently.  See pricing.py's identical pair of constants.
        state_history_depth=max(2, revisit_depth),
        track_first_hop=bool(fg.static_walls and fg.origin_terminal is not None),
    )
