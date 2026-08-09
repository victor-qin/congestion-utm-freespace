"""Flat-array packing of one flight's pricing subproblem, for the compiled DP.

Pure Python and NumPy; this module deliberately does **not** import Numba, so colgen
still runs without the compiled extra. The reference search in :mod:`.pricing` remains the
oracle — nothing here may change an answer, and the tests assert arc-for-arc and
value-for-value identity against the object API.

The split is by **what changes when**, and it is what makes both flight-parallel pricing
and thousands-of-flights scale possible rather than merely convenient:

===================== ========================= ===================== ==================
structure             rebuilt                   shared by             dense?
===================== ========================= ===================== ==================
`PreparedTopology`    once per **graph**        all threads, readonly CSR, ~100 KB/flight
`PreparedRows`        once per **graph**        all threads, readonly **no tables**
`PreparedForbidden`   once per **repair call**  that call             bitset over row ids
`PricingWorkspace`    once per **thread**       nobody -- caller-owned dense, largest flight
===================== ========================= ===================== ==================

Two consequences worth stating outright, because they are the reason for the shape:

* **Nothing per-graph is O(cells x steps).** A density flight reaches ~3k cells over ~1.2k
  steps, so a dense per-graph row table would be ~14 MB and 4,636 of them would be ~65 GB.
  :class:`PreparedRows` therefore *numbers* rows arithmetically and stores no table at all;
  the only dense array is the de-duplication stamp, which lives in the per-**thread**
  workspace and is sized to the largest single flight.
* **The kernel allocates nothing.** Every mutable buffer is owned by the caller and passed
  in, so threads reuse arenas instead of churning them -- the allocator residue that forced
  PR #76's process pool to recycle workers is a property of allocating per flight, not of
  the search.

Row numbering, which the kernel and :func:`prepare_forbidden` both depend on::

    cell row  (cell_index, step) -> cell_index * n_steps + (step - row_step0)
    term row  (term_index, step) -> n_cells * n_steps + term_index * n_steps + (step - row_step0)

Both are O(1) and require no lookup structure, which is the whole point.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ...volumes import column_dwell_s
from .network import RowKey
from .windows import (
    derive_cell_window,
    endpoint_claim_cells,
    endpoint_claim_steps,
    terminal_claim_steps,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...config import SimConfig
    from .network import FlightGraph

Cell = tuple[int, int]

# Sentinel for "no path to any destination".  Deliberately not INT32_MAX: the kernel
# evaluates ``step + 1 + rev_remaining[cell] > max_step`` and INT32_MAX would overflow that
# addition.  2**24 dwarfs any realistic step count while leaving ~127x headroom in int32.
UNREACHABLE = 1 << 24

# Arc role bits, mirroring network.py's private constants.  Restated rather than imported
# so a change there fails a test here instead of silently re-tagging every arc.
ARC_INTERNAL = 1 << 0     # not first, not last
ARC_FIRST = 1 << 1        # first, not last
ARC_LAST = 1 << 2         # not first, last
ARC_FIRST_LAST = 1 << 3   # first and last


@dataclass(frozen=True, slots=True)
class PreparedTopology:
    """Dense, dual-independent mirror of one flight's spatial search domain.

    ``unsupported_reason`` is set instead of raising when the flight has no compiled
    representation (multi-level, no reachable destination). The caller then uses the
    reference search, which is the fallback for every other failure too.
    """

    # Cell interning.  Index order is sorted axial order, so it is a function of the cell
    # SET alone -- stable across processes and independent of BFS discovery order.
    cell_q: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    cell_r: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))

    # CSR adjacency.  Arc order within a cell matches ``outgoing_neighbors`` (that is,
    # ``hexgrid.AXIAL_NEIGHBORS`` order), which is the order the reference DP iterates;
    # reordering would silently change which label wins an insertion-order tie.
    arc_start: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(1, np.int32))
    arc_target: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    arc_roles: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.uint8))

    # Admissible hop count from each cell to the nearest destination over the any-role arc
    # superset.  A lower bound on every role-specific completion.
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
    shortest_hops: int = 0
    # The ONLY route-length bound the search has, post-#78.  Read from ``fg.max_air_hops``
    # rather than rebuilt from ``shortest_hops + overrun``: the graph resolves the ceiling
    # at build time so both searches over the domain agree on it, and reconstructing it
    # here would reintroduce exactly the second, drifting copy #78 removed.
    air_hop_limit: int = 0
    revisit_depth: int = 0
    # Two different widths, deliberately.  ``revisit_depth`` bans re-entering a recently
    # held cell; the STATE key must keep at least the predecessor even when that ban is
    # narrower, because two equal-score labels reaching one cell from different
    # predecessors can de-duplicate the destination endpoint union differently.  See the
    # identical pair of constants in ``pricing._best_column``.
    state_history_depth: int = 0
    track_first_hop: bool = False

    unsupported_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.unsupported_reason is None

    @property
    def n_cells(self) -> int:
        return int(self.cell_q.shape[0])


def _reachable_cells(fg: FlightGraph, seeds: list[Cell]) -> list[Cell]:
    """Forward-reachable cells, expanding arcs lazily and never materializing.

    Walks ``outgoing_neighbors``, which both expands-and-caches the arc oracle and already
    filters to arcs admissible in at least one role.  Iterating ``fg.corridor_cells``
    instead would materialize the lazy ellipse, which is what lazy expansion exists to
    avoid.
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
    """Multi-source reverse BFS giving admissible hops-to-destination per cell.

    Strictly tighter than the reference's ``_distance_lower_bound`` (plain hex distance),
    because it follows real arcs rather than assuming the corridor is convex -- but only
    ever *smaller or equal*, so substituting it cannot make an inadmissible bound.  The
    kernel must therefore not use it where the reference's looser value decides a tie; see
    the parity tests.
    """

    remaining = np.full(n_cells, UNREACHABLE, dtype=np.int32)

    # Reverse the CSR once; the DP only ever needs distances, not the reverse adjacency
    # itself, so it stays local.
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


def _role_mask(fg: FlightGraph, source: Cell, target: Cell) -> int:
    """Pack one arc's four path-position verdicts into a bitmask.

    The lazy oracle already stores exactly this mask, so ask it for the packed value when
    it is present and fall back to four ``hop_allowed_for_role`` calls otherwise --
    ``forbidden_hops`` is a plain frozenset on a transported graph, where only the public
    path works. The two agree by construction: ``_LazyForbiddenHops.allows`` is itself a
    bit test against the same mask, which the parity test asserts arc for arc.
    """

    lazy = fg.forbidden_hops
    role_mask = getattr(lazy, "role_mask", None)
    if role_mask is not None:
        return int(role_mask(source, target))

    mask = 0
    if fg.hop_allowed_for_role(source, target, first=False, last=False):
        mask |= ARC_INTERNAL
    if fg.hop_allowed_for_role(source, target, first=True, last=False):
        mask |= ARC_FIRST
    if fg.hop_allowed_for_role(source, target, first=False, last=True):
        mask |= ARC_LAST
    if fg.hop_allowed_for_role(source, target, first=True, last=True):
        mask |= ARC_FIRST_LAST
    return mask


def prepare_topology(fg: FlightGraph, cfg: SimConfig) -> PreparedTopology:
    """Drain the lazy arc oracle for one flight into dense arrays, once.

    Answer-neutral: every arc and role recorded here is exactly what
    ``fg.outgoing_neighbors`` / ``fg.hop_allowed_for_role`` would return on demand.

    Draining is also what makes the graph **read-only for the rest of the solve**, which
    is the precondition for pricing flights on threads: after this call the search touches
    only arrays, so no lock is contended and no lazy cache is mutated concurrently.
    """

    # Imported here, not at module scope: pricing imports this module.
    from .pricing import _destination_options, _origin_options

    if len(fg.levels) != 1:
        return PreparedTopology(unsupported_reason="colgen v1 pricing is single-level")

    origin_options = _origin_options(fg)
    destination_options = _destination_options(fg)
    if not origin_options or not destination_options:
        return PreparedTopology(unsupported_reason="no origin or destination option")

    reachable = _reachable_cells(fg, [cell for _lane, cell, _steps in origin_options])
    if not reachable:
        return PreparedTopology(unsupported_reason="no reachable corridor cell")

    # An endpoint's hover cylinder CLAIMS cells that no route can ever VISIT -- the disc
    # spreads around the origin/destination point, and its rim regularly falls outside the
    # forward-reachable set.  Measured on `density_faa_wing_zipline`: 4 of the first 12
    # flights have exactly one such cell.  Those cells still need row ids, or the endpoint
    # dwell would go partly unpriced, so they are interned here as claim-only: no arcs, and
    # `rev_remaining` leaves them UNREACHABLE, so the search cannot enter them.
    claim_only: set[Cell] = set()
    for is_origin in (True, False):
        if (fg.origin_terminal if is_origin else fg.dest_terminal) is not None:
            continue  # a terminal endpoint claims `term` rows, not cell rows
        point = fg.request.origin if is_origin else fg.request.dest
        claim_only.update(endpoint_claim_cells(point, cfg.effective_hover_radius_m, cfg))
    cells = sorted(set(reachable) | claim_only)
    index = {cell: i for i, cell in enumerate(cells)}
    n = len(cells)

    reachable_set = set(reachable)
    arc_start = np.zeros(n + 1, dtype=np.int32)
    arc_target_list: list[int] = []
    arc_roles_list: list[int] = []
    for i, cell in enumerate(cells):
        # Claim-only cells get no arcs, so the CSR is exactly what interning the reachable
        # set alone would have produced.  Sound as well as tidy: a claim-only cell has no
        # incoming arc from a reachable one -- the BFS follows the same arcs, so it would
        # have been reached -- and therefore cannot shorten any reachable cell's
        # `rev_remaining` either.
        for target in (fg.outgoing_neighbors(cell) if cell in reachable_set else ()):
            target_index = index.get(target)
            if target_index is None:
                # Unreachable-from-origin targets cannot appear: the BFS above followed
                # the same arcs.  Guard anyway rather than emit a dangling index.
                continue
            arc_target_list.append(target_index)
            arc_roles_list.append(_role_mask(fg, cell, target))
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

    prepared = PreparedTopology(
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
        shortest_hops=int(fg.shortest_hops),
        air_hop_limit=int(fg.max_air_hops),
        revisit_depth=revisit_depth,
        state_history_depth=max(2, revisit_depth),
        track_first_hop=bool(fg.static_walls and fg.origin_terminal is not None),
    )
    # Read-only after construction: these are shared across threads without a lock, and
    # the guarantee that nobody mutates them is the entire basis for doing so.
    for array in (
        prepared.cell_q, prepared.cell_r, prepared.arc_start, prepared.arc_target,
        prepared.arc_roles, prepared.rev_remaining, prepared.dest_mask,
        prepared.dest_lane_start, prepared.dest_lane_idx, prepared.origin_lane_idx,
        prepared.origin_cell, prepared.origin_lane_steps,
    ):
        array.setflags(write=False)
    return prepared


@dataclass(frozen=True, slots=True)
class PreparedRows:
    """Arithmetic numbering of every capacity row this flight can claim.

    Deliberately holds **no table**. ``row_of_cell``/``row_of_term`` are closed-form, so
    the whole structure is a handful of scalars plus the endpoint discs -- which is what
    lets thousands of graphs stay resident while the only dense array (the kernel's
    de-duplication stamp, sized ``n_rows``) lives once per thread.

    ``step0``/``n_steps`` bound the clock generously rather than exactly: a row id must
    exist for every step any endpoint window can reach, including the padding
    ``endpoint_claim_steps`` adds outside ``[min_step, max_step]``.
    """

    n_cells: int = 0
    n_terminals: int = 0
    step0: int = 0
    n_steps: int = 0

    # Endpoint discs, resolved once per graph.  ``endpoint_claim_cells`` was measured at
    # 202,044 calls per 12-flight solve before this.
    origin_disc: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    dest_disc: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    origin_is_terminal: bool = False
    dest_is_terminal: bool = False
    # Terminal slots, -1 when that endpoint is not a terminal.
    origin_term_slot: int = -1
    dest_term_slot: int = -1

    # Dwell duration per endpoint, in seconds.  ``t1 - t0`` is constant for a given
    # endpoint, so the span is a pure translation of a fixed pattern by ``step``.
    origin_dwell_s: float = 0.0
    dest_dwell_s: float = 0.0

    unsupported_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.unsupported_reason is None

    @property
    def n_rows(self) -> int:
        return (self.n_cells + self.n_terminals) * self.n_steps

    def row_of_cell(self, cell_index: int, step: int) -> int:
        """Row id for a cell visit, or -1 when the step is outside the numbering."""

        offset = step - self.step0
        if offset < 0 or offset >= self.n_steps or not 0 <= cell_index < self.n_cells:
            return -1
        return cell_index * self.n_steps + offset

    def row_of_term(self, term_slot: int, step: int) -> int:
        """Row id for a terminal dwell period, or -1 when outside the numbering."""

        offset = step - self.step0
        if offset < 0 or offset >= self.n_steps or not 0 <= term_slot < self.n_terminals:
            return -1
        return (self.n_cells + term_slot) * self.n_steps + offset


def prepare_rows(fg: FlightGraph, cfg: SimConfig, topology: PreparedTopology) -> PreparedRows:
    """Number this flight's row universe and resolve its two endpoint discs.

    The two endpoint shapes are kept **separate on purpose**, and unifying them is the
    mistake this docstring exists to prevent. ``windows.py`` states the divergence is
    deliberate:

    * a **terminal** endpoint claims ``term`` rows over ``terminal_claim_steps``, which
      applies no floating padding and ignores ``timing_steps`` entirely;
    * a **bare point** claims ``cell`` rows over ``endpoint_claim_steps``, whose outward
      rounding carries a drift bound that scales *with* ``timing_steps``.

    Origin and destination choose independently, so all four combinations occur --
    and terminal endpoints are the density-scenario shape, not an edge case.
    """

    if not topology.ok:
        return PreparedRows(unsupported_reason=topology.unsupported_reason)
    if len(fg.levels) != 1:
        return PreparedRows(unsupported_reason="colgen v1 pricing is single-level")

    cell_index = {
        (int(q), int(r)): i
        for i, (q, r) in enumerate(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    }
    z = fg.levels[0]

    discs: dict[bool, np.ndarray] = {}
    dwell: dict[bool, float] = {}
    term_slot: dict[bool, int] = {}
    terminals: list[Any] = []
    for is_origin in (True, False):
        point = fg.request.origin if is_origin else fg.request.dest
        terminal = fg.origin_terminal if is_origin else fg.dest_terminal
        dwell[is_origin] = cfg.hover_time_s + column_dwell_s(point, terminal, cfg, z)
        if terminal is not None:
            if terminal.id not in [t.id for t in terminals]:
                terminals.append(terminal)
            term_slot[is_origin] = [t.id for t in terminals].index(terminal.id)
            discs[is_origin] = np.empty(0, np.int32)
            continue
        term_slot[is_origin] = -1
        # Every disc cell has an index: `prepare_topology` interns the union of the
        # reachable set and both discs precisely so this cannot miss one.  A missing cell
        # would silently under-price the endpoint dwell -- a cheaper wrong answer rather
        # than a slower right one -- so it stays a hard refusal rather than a drop.
        cells = endpoint_claim_cells(point, cfg.effective_hover_radius_m, cfg)
        missing = [c for c in cells if c not in cell_index]
        if missing:
            return PreparedRows(
                unsupported_reason=(
                    f"{len(missing)} endpoint disc cell(s) were not interned; "
                    "prepare_topology and prepare_rows disagree about the cell set"
                )
            )
        discs[is_origin] = np.asarray(
            sorted(cell_index[c] for c in cells), dtype=np.int32
        )

    # Bound the clock generously: endpoint windows pad outside [min_step, max_step], and a
    # row id that does not exist would silently drop a claim.  The pad is at most the dwell
    # plus the outward rounding, so one dwell's worth of steps on each side is ample.
    pad = int(max(dwell.values()) / cfg.dt_s) + 8
    step0 = topology.min_step - pad
    n_steps = (topology.max_step + pad) - step0 + 1

    return PreparedRows(
        n_cells=topology.n_cells,
        n_terminals=len(terminals),
        step0=step0,
        n_steps=n_steps,
        origin_disc=discs[True],
        dest_disc=discs[False],
        origin_is_terminal=fg.origin_terminal is not None,
        dest_is_terminal=fg.dest_terminal is not None,
        origin_term_slot=term_slot[True],
        dest_term_slot=term_slot[False],
        origin_dwell_s=dwell[True],
        dest_dwell_s=dwell[False],
    )


def endpoint_row_ids(
    rows: PreparedRows,
    cfg: SimConfig,
    *,
    origin: bool,
    step: int,
    timing_steps: int,
) -> list[int]:
    """Row ids one endpoint dwell claims — the row-id image of ``_endpoint_claims``.

    Kept in Python beside the packing rather than inlined into the kernel so a test can
    compare it against :func:`pricing._endpoint_claims_uncached` directly, key for key.
    The kernel reimplements this arithmetic; this is what pins it.
    """

    t0 = step * cfg.dt_s
    is_terminal = rows.origin_is_terminal if origin else rows.dest_is_terminal
    t1 = t0 + (rows.origin_dwell_s if origin else rows.dest_dwell_s)
    if is_terminal:
        slot = rows.origin_term_slot if origin else rows.dest_term_slot
        return [rows.row_of_term(slot, s) for s in terminal_claim_steps(t0, t1, cfg)]
    disc = rows.origin_disc if origin else rows.dest_disc
    steps = endpoint_claim_steps(t0, t1, cfg, timing_steps=timing_steps)
    return [
        rows.row_of_cell(int(cell_index), s) for cell_index in disc for s in steps
    ]


@dataclass(frozen=True, slots=True)
class PreparedForbidden:
    """Saturated rows, as a bitset over row ids.

    A bitset rather than PR #76's Fibonacci hash because rows are already interned to a
    dense range here: ``n_rows / 8`` bytes, O(1) membership, no collisions and no probe
    loop in the innermost test. Rows outside this flight's universe simply do not map,
    which is correct -- the flight cannot claim them.

    This is a first-class input rather than a fallback trigger: repair runs once per
    flight in the greedy, so routing it to the Python reference would be a scaling cliff
    exactly where thousands of flights are felt.
    """

    bits: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.uint64))
    n_set: int = 0
    # Forbidden rows this flight could claim but that could not be mapped.  Must be zero
    # for the compiled path to be trusted; non-zero means a claim would go unchecked.
    n_unmapped: int = 0

    @property
    def any(self) -> bool:
        return self.n_set > 0


def prepare_forbidden(
    forbidden_rows,
    fg: FlightGraph,
    rows: PreparedRows,
    topology: PreparedTopology,
) -> PreparedForbidden:
    """Map an exclusion set into this flight's row numbering."""

    n_words = (rows.n_rows + 63) // 64
    bits = np.zeros(max(1, n_words), dtype=np.uint64)
    if not forbidden_rows:
        return PreparedForbidden(bits=bits)

    cell_index = {
        (int(q), int(r)): i
        for i, (q, r) in enumerate(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    }
    term_slot: dict[Any, int] = {}
    if rows.origin_is_terminal:
        term_slot[fg.origin_terminal.id] = rows.origin_term_slot
    if rows.dest_is_terminal:
        term_slot[fg.dest_terminal.id] = rows.dest_term_slot

    n_set = 0
    n_unmapped = 0
    for row in forbidden_rows:
        key = row if isinstance(row, RowKey) else RowKey(row)
        if key.kind == "cell":
            # A different level cannot be claimed by a single-level flight, so it is out
            # of universe rather than unmapped.
            if key.level != 0:
                continue
            index = cell_index.get(key.cell_coord)
            if index is None:
                continue
            row_id = rows.row_of_cell(index, key.step)
        else:
            slot = term_slot.get(key.terminal_id)
            if slot is None:
                continue
            row_id = rows.row_of_term(slot, key.step)
        if row_id < 0:
            # In-universe by resource but outside the numbered clock: the flight COULD
            # touch this resource, so silently dropping it would let a forbidden claim
            # through.  Counted, and the caller must refuse the compiled path.
            n_unmapped += 1
            continue
        bits[row_id >> 6] |= np.uint64(1) << np.uint64(row_id & 63)
        n_set += 1
    bits.setflags(write=False)
    return PreparedForbidden(bits=bits, n_set=n_set, n_unmapped=n_unmapped)


@dataclass(frozen=True, slots=True)
class PreparedDuals:
    """One iteration's row prices, in this flight's row numbering.

    Mirrors :class:`pricing.DualView`'s **two** structures, and it has to be two rather
    than one for a reason that is easy to get wrong: a window query and an exact claim cost
    are not the same arithmetic.

    * ``series_*`` are the dense per-resource prefix sums a visit-window query subtracts,
      exactly as ``DualView.visit_cost`` does. O(1), and bit-identical because it is the
      same two floats subtracted.
    * ``row_id``/``row_value`` are the **exact per-row prices**, sorted for binary search,
      mirroring ``DualView._duals``. Deriving a single row's price from the prefix sums
      instead -- ``prefix[k+1] - prefix[k]`` -- would *not* be bit-identical, because
      ``(a + v) - a != v`` in floating point, and ``claim_cost`` sums precisely these
      values. A dense array is not an option: 5M rows would be 40 MB per flight.

    Rebuilt once per **sweep**, not per flight: duals are global to an iteration.
    """

    row_id: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int64))
    row_value: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))

    # Series table, shared by cells and terminals.  -1 means "no dual anywhere on this
    # resource", which the reference spells as a missing dict entry.
    cell_series: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    term_series: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    series_first: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    series_start: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(1, np.int64))
    series_prefix: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))

    offsets_lo: int = 0
    offsets_hi: int = 0
    max_negative_credit: float = 0.0
    # Duals on a resource this flight owns but at a step outside its numbering.  Expected
    # to be zero: every row this flight can claim lies within `[min_step - pad, max_step +
    # pad]` by construction, so an out-of-range dual is on a row it cannot claim and
    # dropping it is a no-op.  Counted anyway, because "expected zero" and "checked zero"
    # differ exactly when it matters.
    n_out_of_range: int = 0

    def range_sum(self, series: int, start: int, stop: int) -> float:
        """Sum over ``[start, stop)`` — the literal arithmetic of ``_PrefixSeries``."""

        if series < 0:
            return 0.0
        lo_index = int(self.series_start[series])
        hi_index = int(self.series_start[series + 1])
        length = hi_index - lo_index
        if stop <= start or length <= 1:
            return 0.0
        first = int(self.series_first[series])
        series_stop = first + length - 1
        lo = min(max(start, first), series_stop)
        hi = min(max(stop, first), series_stop)
        if hi <= lo:
            return 0.0
        return float(
            self.series_prefix[lo_index + hi - first] - self.series_prefix[lo_index + lo - first]
        )

    def visit_cost(self, cell_index: int, visit_step: int) -> float:
        """All cell-row duals charged by one centre visit, in O(1)."""

        if not 0 <= cell_index < self.cell_series.shape[0]:
            return 0.0
        return self.range_sum(
            int(self.cell_series[cell_index]),
            visit_step + self.offsets_lo,
            visit_step + self.offsets_hi + 1,
        )

    def row_cost(self, row: int) -> float:
        """Exact price of one row id, or zero when unpriced."""

        if row < 0 or self.row_id.shape[0] == 0:
            return 0.0
        position = int(np.searchsorted(self.row_id, row))
        if position < self.row_id.shape[0] and int(self.row_id[position]) == row:
            return float(self.row_value[position])
        return 0.0


def prepare_duals(
    view,
    fg: FlightGraph,
    topology: PreparedTopology,
    rows: PreparedRows,
) -> PreparedDuals:
    """Restate one iteration's duals in this flight's row numbering.

    Reads ``DualView``'s private structures deliberately: they are the values the reference
    search actually consults, so copying them is what makes the compiled path bit-identical
    rather than merely close. Recomputing prefix sums here from the raw dual mapping would
    reintroduce the possibility of a different summation order.
    """

    cell_index = {
        (int(q), int(r)): i
        for i, (q, r) in enumerate(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    }
    term_slot: dict[Any, int] = {}
    if rows.origin_is_terminal:
        term_slot[fg.origin_terminal.id] = rows.origin_term_slot
    if rows.dest_is_terminal:
        term_slot[fg.dest_terminal.id] = rows.dest_term_slot

    series_first: list[int] = []
    series_prefix: list[float] = []
    series_start: list[int] = [0]
    cell_series = np.full(topology.n_cells, -1, dtype=np.int32)
    term_series = np.full(max(rows.n_terminals, 0), -1, dtype=np.int32)

    def add_series(prefix_series) -> int:
        slot = len(series_first)
        series_first.append(int(prefix_series.first_step))
        series_prefix.extend(float(v) for v in prefix_series.prefix)
        series_start.append(len(series_prefix))
        return slot

    for (cell, level), prefix_series in view._cell.items():
        if level != 0:
            continue
        index = cell_index.get(cell)
        if index is not None:
            cell_series[index] = add_series(prefix_series)
    for terminal_id, prefix_series in view._terminal.items():
        slot = term_slot.get(terminal_id)
        if slot is not None:
            term_series[slot] = add_series(prefix_series)

    pairs: list[tuple[int, float]] = []
    n_out_of_range = 0
    for key, value in view._duals.items():
        if key.kind == "cell":
            if key.level != 0:
                continue
            index = cell_index.get(key.cell_coord)
            if index is None:
                continue
            row = rows.row_of_cell(index, key.step)
        else:
            slot = term_slot.get(key.terminal_id)
            if slot is None:
                continue
            row = rows.row_of_term(slot, key.step)
        if row < 0:
            n_out_of_range += 1
            continue
        pairs.append((row, float(value)))
    pairs.sort()

    prepared = PreparedDuals(
        row_id=np.asarray([r for r, _v in pairs], dtype=np.int64),
        row_value=np.asarray([v for _r, v in pairs], dtype=np.float64),
        cell_series=cell_series,
        term_series=term_series,
        series_first=np.asarray(series_first, dtype=np.int32),
        series_start=np.asarray(series_start, dtype=np.int64),
        series_prefix=np.asarray(series_prefix, dtype=np.float64),
        offsets_lo=int(view.offsets[0]),
        offsets_hi=int(view.offsets[1]),
        max_negative_credit=float(view.max_negative_credit),
        n_out_of_range=n_out_of_range,
    )
    for array in (
        prepared.row_id, prepared.row_value, prepared.cell_series, prepared.term_series,
        prepared.series_first, prepared.series_start, prepared.series_prefix,
    ):
        array.setflags(write=False)
    return prepared


def visit_row_ids(rows: PreparedRows, cell_index: int, visit_step: int, offsets) -> list[int]:
    """Row ids one centre visit claims — the row-id image of ``pricing._visit_claims``."""

    lo, hi = offsets
    return [rows.row_of_cell(cell_index, visit_step + o) for o in range(lo, hi + 1)]


@dataclass(frozen=True, slots=True)
class PreparedVariants:
    """Root labels: every ``(departure_step, origin lane)`` the reference would create.

    Mirrors the initialization loop of ``pricing._best_column``. One variant is one root
    label, so the kernel creates exactly the labels the reference does.

    ``paid_class`` is the subtle field. The reference's dominance key holds the *set*
    ``origin_paid_rows``, **not** the departure step, so two roots from different
    departures that happened to pay the same rows are allowed to merge downstream. Keying
    on variant id instead would keep them apart -- still optimal, since a finer state never
    loses a completion, but it explores more labels and, worse, can break a tie the
    reference breaks the other way. Distinct paid-row sets are therefore interned to a
    dense class id and the kernel keys on that.

    **Every time field here is already multiplied by the objective's weights**, which is
    what lets the kernel stay objective-agnostic: it charges ``air_weight * dt_s`` per arc
    and never sees a weight of its own. Dual prices are NOT weighted -- they already arrive
    in the master's currency. Getting this backwards is a silent wrong answer rather than a
    crash, and is the reason the label score must be denominated in the objective.
    """

    departure_step: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    lane_idx: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    cell: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    start_step: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    score: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))
    ground_delay_s: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))
    start_dual_cost: np.ndarray = field(
        repr=False, default_factory=lambda: np.empty(0, np.float64)
    )
    paid_class: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))

    # Rows a root already paid, CSR over paid CLASS (not variant).  The arc loop consults
    # these to avoid charging a cell row twice when a later visit window overlaps the
    # origin endpoint's own cells -- the reference's ``_paid_cell_rows``.
    paid_start: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(1, np.int32))
    paid_cell: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    paid_step: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.int32))
    paid_value: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0, np.float64))

    n_departures_considered: int = 0
    n_departures_prefiltered: int = 0
    unsupported_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.unsupported_reason is None

    @property
    def n_variants(self) -> int:
        return int(self.departure_step.shape[0])


def prepare_variants(
    fg: FlightGraph,
    cfg: SimConfig,
    view,
    topology: PreparedTopology,
    rows: PreparedRows,
    *,
    benefit: float = 0.0,
    pi_f: float = 0.0,
    cost_cutoff: float | None = None,
    model=None,
    forbidden_rows=frozenset(),
) -> PreparedVariants:
    """Price every root option once, exactly as ``_best_column``'s initialization does.

    ``cost_cutoff`` enables the same cheap ground-delay prefilter the reference applies at
    the top of its loop, *before* building any endpoint claim set. It matters more here
    than there: ``max_ground_delay_s=3600`` gives 901 departure steps, and building origin
    rows for every one of them costs more than the search that consumes them. A departure
    whose ground delay alone cannot beat the incumbent can never win, whatever route
    follows.

    Deliberately **omits** ``completion_can_compete``. That gate is a prune, not a
    correctness condition, and it needs the completion envelope that lives in the kernel;
    omitting a prune costs work and never an answer, whereas applying one the reference
    would not have is how a compiled search loses the optimum.
    """

    from .objective import DELAY_MODEL
    from .pricing import _RECOMPUTE_EPS, _fold_leg_s, _origin_options, _visit_claims

    if not (topology.ok and rows.ok):
        return PreparedVariants(
            unsupported_reason=topology.unsupported_reason or rows.unsupported_reason
        )
    if model is None:
        model = DELAY_MODEL
    w_ground, w_air = model.ground_weight, model.air_weight
    offsets = view.offsets

    cell_index = {
        (int(q), int(r)): i
        for i, (q, r) in enumerate(zip(topology.cell_q.tolist(), topology.cell_r.tolist()))
    }
    origin_options = _origin_options(fg)
    # Endpoint legs, computed once per lane exactly as the reference does.
    origin_leg_by_lane: dict[int | None, float] = {}
    for lane_idx, _cell, _steps in origin_options:
        lane_dist = None if lane_idx is None else fg.origin_lanes[lane_idx].dist
        origin_leg_by_lane[lane_idx] = _fold_leg_s(
            fg.request.origin, fg.origin_terminal, lane_dist, cfg
        )

    from .pricing import _destination_options, _distance_lower_bound
    destination_cells = frozenset(_destination_options(fg))
    distance_cache: dict[Cell, int] = {}

    def remaining_distance(cell: Cell) -> int:
        cached = distance_cache.get(cell)
        if cached is None:
            cached = _distance_lower_bound(cell, destination_cells)
            distance_cache[cell] = cached
        return cached

    paid_ids: dict[frozenset, int] = {}
    paid_rows_by_class: list[frozenset] = []

    dep: list[int] = []
    lanes: list[int] = []
    cells: list[int] = []
    starts: list[int] = []
    scores: list[float] = []
    grounds: list[float] = []
    dual_costs: list[float] = []
    classes: list[int] = []
    considered = prefiltered = 0

    for departure_step in range(fg.base_step, fg.latest_departure_step + 1):
        considered += 1
        ground_delay_s = (departure_step - fg.base_step) * cfg.dt_s
        ground_score = -w_ground * ground_delay_s
        if cost_cutoff is not None:
            start_upper_bound = benefit + ground_score - pi_f + view.max_negative_credit
            if start_upper_bound < cost_cutoff - _RECOMPUTE_EPS:
                prefiltered += 1
                continue
        origin_claims = pricing_endpoint_claims(fg, cfg, origin=True, step=departure_step)
        if not origin_claims.isdisjoint(forbidden_rows):
            continue
        for lane_idx, cell, lane_steps in origin_options:
            index = cell_index.get(cell)
            if index is None:
                continue
            distance_to_go = remaining_distance(cell)
            start_step = departure_step + fg.takeoff_steps[0] + lane_steps
            if start_step >= fg.max_step:
                continue
            if start_step + distance_to_go > fg.max_step:
                continue
            if distance_to_go > topology.air_hop_limit:
                continue
            start_claims = origin_claims | _visit_claims(cell, 0, start_step, offsets)
            if not start_claims.isdisjoint(forbidden_rows):
                continue
            start_dual_cost = view.claim_cost(start_claims)
            origin_paid_rows = view.active_claims(start_claims)
            paid_class = paid_ids.get(origin_paid_rows)
            if paid_class is None:
                paid_class = len(paid_rows_by_class)
                paid_ids[origin_paid_rows] = paid_class
                paid_rows_by_class.append(origin_paid_rows)

            dep.append(departure_step)
            lanes.append(-1 if lane_idx is None else int(lane_idx))
            cells.append(index)
            starts.append(start_step)
            scores.append(ground_score - w_air * origin_leg_by_lane[lane_idx] - start_dual_cost)
            grounds.append(w_ground * ground_delay_s)
            dual_costs.append(start_dual_cost)
            classes.append(paid_class)

    # Paid-row corrections, CSR over class.  Only single-level CELL rows can recur in a
    # later visit window; terminal rows never can, so they are dropped rather than searched
    # -- exactly what ``_paid_cell_rows`` does.
    paid_start = [0]
    paid_cell: list[int] = []
    paid_step: list[int] = []
    paid_value: list[float] = []
    for paid in paid_rows_by_class:
        for row in sorted(paid):
            if row.kind != "cell" or row.level != 0:
                continue
            index = cell_index.get(row.cell_coord)
            if index is None:
                continue
            paid_cell.append(index)
            paid_step.append(int(row.step))
            paid_value.append(float(view.row_cost(row)))
        paid_start.append(len(paid_cell))

    return PreparedVariants(
        departure_step=np.asarray(dep, dtype=np.int32),
        lane_idx=np.asarray(lanes, dtype=np.int32),
        cell=np.asarray(cells, dtype=np.int32),
        start_step=np.asarray(starts, dtype=np.int32),
        score=np.asarray(scores, dtype=np.float64),
        ground_delay_s=np.asarray(grounds, dtype=np.float64),
        start_dual_cost=np.asarray(dual_costs, dtype=np.float64),
        paid_class=np.asarray(classes, dtype=np.int32),
        paid_start=np.asarray(paid_start, dtype=np.int32),
        paid_cell=np.asarray(paid_cell, dtype=np.int32),
        paid_step=np.asarray(paid_step, dtype=np.int32),
        paid_value=np.asarray(paid_value, dtype=np.float64),
        n_departures_considered=considered,
        n_departures_prefiltered=prefiltered,
    )


def pricing_endpoint_claims(fg, cfg, *, origin: bool, step: int):
    """``pricing._endpoint_claims`` with ``timing_steps=0``, imported lazily.

    Wrapped rather than imported at module scope because ``pricing`` imports this module;
    named so the call site reads as the reference function it is.
    """

    from .pricing import _endpoint_claims

    return _endpoint_claims(fg, cfg, origin=origin, step=step, timing_steps=0)


class PricingWorkspace:
    """Caller-owned scratch for one compiled pricing call.

    **Owned by the caller, never allocated inside the kernel.** That is the single
    decision that makes flight-parallel pricing cheap: one workspace per thread, reused
    across every flight it prices, so the per-flight allocate-and-free churn that leaves
    a large allocator residue never happens. PR #76 measured 2.5 GB surviving
    ``gc.collect()`` with every graph unreachable, and answered it by recycling worker
    *processes*; not allocating is the cheaper answer.

    Growth is **geometric**, never exact-fit: a ladder of powers of two was measured to
    remove 82% of resize waste, and an exact-fit policy re-allocates on nearly every
    flight because label counts vary by orders of magnitude between them.
    """

    __slots__ = ("stamp", "stamp_gen", "claim_scratch", "_n_rows", "_n_claims")

    def __init__(self) -> None:
        self.stamp = np.zeros(0, dtype=np.int32)
        # Generation stamping instead of clearing: a claim set is a few hundred rows out
        # of millions, so zeroing the array per sink would dominate the sink itself.
        self.stamp_gen = 0
        self.claim_scratch = np.zeros(0, dtype=np.int32)
        self._n_rows = 0
        self._n_claims = 0

    @staticmethod
    def _grow(current: int, needed: int) -> int:
        size = max(current, 1024)
        while size < needed:
            size <<= 1
        return size

    def ensure(self, n_rows: int, n_claims: int) -> None:
        """Size the buffers for one flight, keeping whatever is already large enough."""

        if n_rows > self._n_rows:
            self._n_rows = self._grow(self._n_rows, n_rows)
            self.stamp = np.zeros(self._n_rows, dtype=np.int32)
            # A fresh array is all zeros, so restart stamping rather than carrying a
            # generation that would read as "already seen" on every row.
            self.stamp_gen = 0
        if n_claims > self._n_claims:
            self._n_claims = self._grow(self._n_claims, n_claims)
            self.claim_scratch = np.zeros(self._n_claims, dtype=np.int32)

    def next_generation(self) -> int:
        """Advance the stamp, clearing only when int32 would wrap."""

        self.stamp_gen += 1
        if self.stamp_gen >= (1 << 31) - 1:
            self.stamp.fill(0)
            self.stamp_gen = 1
        return self.stamp_gen

    @property
    def n_rows_capacity(self) -> int:
        return self._n_rows
