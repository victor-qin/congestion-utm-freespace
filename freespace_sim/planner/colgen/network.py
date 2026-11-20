"""Column claim construction over the shared physical route domain.

The aliases preserve archived graph and row pickles while the domain implementation
remains usable without importing the column-generation package.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

from ...config import SimConfig
from ...types import IntentStatus
from ...conflict import volumes_conflict
from ...volumes import column_dwell_s
from .. import hexgrid as hg
from ..turn_rule import path_turns_valid
from ..route_domain import (
    RouteBudget as RouteBudget,
    Cell as Cell,
    FlatAabb as FlatAabb,
    WallBound as WallBound,
    _ARC_INTERNAL as _ARC_INTERNAL,
    _ARC_FIRST as _ARC_FIRST,
    _ARC_LAST as _ARC_LAST,
    _ARC_FIRST_LAST as _ARC_FIRST_LAST,
    _ALL_ARC_ROLES as _ALL_ARC_ROLES,
    _MAX_CERTIFIED_COLUMNS as _MAX_CERTIFIED_COLUMNS,
    _MAX_ENDPOINT_CLAIMS as _MAX_ENDPOINT_CLAIMS,
    _aabbs_overlap as _aabbs_overlap,
    _WallSpatialIndex as _WallSpatialIndex,
    StaticTerminalCatalog as StaticTerminalCatalog,
    _ForeignTerminalCells as _ForeignTerminalCells,
    _CombinedCellSet as _CombinedCellSet,
    _ImmutableFlightRequest as _ImmutableFlightRequest,
    _snapshot_request as _snapshot_request,
    RowKey as RowKey,
    RowIndex as RowIndex,
    _LazyCorridorCells as _LazyCorridorCells,
    _FlightSearchCache as _FlightSearchCache,
    FlightGraph as FlightGraph,
    _ellipse_cells as _ellipse_cells,
    _graph_max_step as _graph_max_step,
    _segment_terminal_id as _segment_terminal_id,
    _forbidden_static_hops as _forbidden_static_hops,
    _static_hop_allowed_roles as _static_hop_allowed_roles,
    _arc_role as _arc_role,
    _LazyForbiddenHops as _LazyForbiddenHops,
    FlightGraphInfeasible as FlightGraphInfeasible,
    build_flight_graph as build_flight_graph,
)
from .windows import (
    derive_cell_window, endpoint_claim_cells, endpoint_claim_steps,
    terminal_claim_steps, visit_rows,
)

if TYPE_CHECKING:
    from .translate import Column


def _selected_lane(lanes: tuple[hg.Lane, ...], index: int | None, endpoint: str) -> hg.Lane:
    """Return ``lanes[index]``, raising ``ValueError`` if ``index`` is missing or out of range.

    Parameters
    ------------
    - lanes (tuple[hg.Lane, ...]): the terminal's exit lanes to index into.
    - index (int | None): the chosen lane index; ``None`` or a negative/out-of-range value raises.
    - endpoint (str): endpoint label ("origin"/"dest") used in error messages.

    Return
    --------
    - output (hg.Lane): ``lanes[index]``. Raises ``ValueError`` when ``index`` is ``None``,
      negative, or beyond the lane tuple.
    """
    if index is None:
        raise ValueError(f"{endpoint}_lane_idx is required for a terminal endpoint")
    try:
        lane = lanes[operator.index(index)]
    except (TypeError, IndexError) as exc:
        raise ValueError(f"invalid {endpoint} terminal lane index {index!r}") from exc
    if index < 0:
        raise ValueError(f"invalid {endpoint} terminal lane index {index!r}")
    return lane


def _column_endpoint_steps(column, fg, cfg, arrival_step, *, departure_step=None):
    """Compute endpoint windows at the actual clock, including float boundary rules."""

    departure = column.departure_step if departure_step is None else departure_step
    windows = []
    for point, terminal, step, hops in (
        (fg.request.origin, fg.origin_terminal, departure, 0),
        (fg.request.dest, fg.dest_terminal, arrival_step, len(column.cell_path) - 1),
    ):
        t0 = step * cfg.dt_s
        t1 = t0 + cfg.hover_time_s + column_dwell_s(point, terminal, cfg, fg.levels[column.level])
        windows.append(
            terminal_claim_steps(t0, t1, cfg) if terminal is not None else
            endpoint_claim_steps(t0, t1, cfg, timing_steps=hops)
        )
    return tuple(windows)


def shifted_claims_are_exact(column: Column, fg: FlightGraph, cfg: SimConfig, departure: int) -> bool:
    """Whether an already-certified row union translates at this endpoint clock.

    The caller must certify the source column first. Computing its endpoint windows
    directly avoids relying on the separate two-path claim cache surviving a DP call.
    """
    lane_steps = 0 if column.origin_lane_idx is None else fg.origin_lanes[column.origin_lane_idx].steps
    arrival = departure + fg.takeoff_steps[column.level] + lane_steps + len(column.cell_path) - 1
    if not fg.base_step <= departure <= fg.latest_departure_step or arrival > fg.max_step:
        return False
    windows = _column_endpoint_steps(column, fg, cfg, arrival, departure_step=departure)
    old_arrival = column.departure_step + fg.takeoff_steps[column.level] + lane_steps + len(column.cell_path) - 1
    old_windows = _column_endpoint_steps(column, fg, cfg, old_arrival)
    return tuple((w.start - departure, w.stop - departure) for w in windows) == tuple(
        (w.start - column.departure_step, w.stop - column.departure_step) for w in old_windows)


def _collect_column_claims(
    column: Column,
    fg: FlightGraph,
    cfg: SimConfig,
    offsets: tuple[int, int],
    corridor_start_step: int,
    windows: tuple[range, ...],
) -> frozenset[RowKey]:
    """Construct the complete signed-price-independent claim union without certifying it.

    Parameters
    ------------
    - column (Column): route whose cell visits and endpoint dwell rows are collected.
    - fg (FlightGraph): endpoint geometry and terminals.
    - cfg (SimConfig): resource windows and flight levels.
    - offsets (tuple[int, int]): exact derived cell window offsets.
    - corridor_start_step (int): first enroute visit on the original lattice clock.
    - windows (tuple[range, ...]): exact origin and destination occupancy windows.

    Return
    --------
    - output (frozenset[RowKey]): deduplicated union; no validation cache is read or written.
    """
    claims: set[RowKey] = set()
    for visit_step, (q, r) in enumerate(column.cell_path, start=corridor_start_step):
        claims.update(
            RowKey.cell(q, r, column.level, row_step)
            for row_step in visit_rows(visit_step, offsets)
        )

    endpoints = (
        (fg.request.origin, fg.origin_terminal),
        (fg.request.dest, fg.dest_terminal),
    )
    for (point, terminal), endpoint_steps in zip(endpoints, windows, strict=True):
        if terminal is not None:
            claims.update(
                RowKey.term(terminal.id, row_step)
                for row_step in endpoint_steps
            )
            continue

        # Customer cylinders are untagged ledger geometry.  They span the regulated tube, hence
        # claim every flight level in every nearby cell and time row that could meet a transit.
        endpoint_cells = endpoint_claim_cells(point, cfg.effective_hover_radius_m, cfg)
        claims.update(
            RowKey.cell(q, r, level, row_step)
            for q, r in endpoint_cells
            for level in range(len(fg.levels))
            for row_step in endpoint_steps
        )

    return frozenset(claims)


def column_claims(
    column: Column,
    fg: FlightGraph,
    cfg: SimConfig,
    W: int | tuple[int, int] | None = None,
    *,
    _intent=None,
    _validate: bool = True,
    _claims: frozenset[RowKey] | None = None,
) -> frozenset[RowKey]:
    """Return the de-duplicated capacity rows claimed by ``column``.

    Validates the column against the graph, certifies it against the exact permanent-wall
    geometry, then collects its cell-visit and terminal/endpoint dwell rows.  The result is
    memoized on the graph's search cache.

    ``W`` is retained as a compatibility/checking argument for the formulation's window-width
    notation.  The actual offsets always come from :func:`derive_cell_window`; a scalar width alone
    cannot describe the shifted ``(-1, 0)`` footprint used when ``time_buffer_s == 0``.

    Parameters
    ------------
    - column (Column): the candidate schedule (cell path, departure step, level, lane indices).
    - fg (FlightGraph): the graph the column was priced over; supplies geometry and caches.
    - cfg (SimConfig): must equal the config the graph was built with.
    - W (int | tuple[int, int] | None): optional cell-window width or ``(lo, hi)`` offsets, checked
      against :func:`derive_cell_window`; ``None`` skips the check.
    - _intent: optional pre-built intent to reuse instead of re-translating the column.
    - _validate (bool): check route feasibility and use certified-claim caching. False
      constructs diagnostic claims only and neither reads nor writes that cache.
    - _claims (frozenset[RowKey] | None): the row union the caller already built for this exact
      column with the same window helpers (pricing's pre-check); every validation still runs,
      only the collection is skipped.  ``None`` collects the rows here.

    Return
    --------
    - output (frozenset[RowKey]): the capacity rows the column claims; raises ``ValueError`` if the
      column is invalid, out of budget, or overlaps permanent static terminal airspace.
    """
    if not _validate:
        lane_steps = (0 if fg.origin_terminal is None else
                      fg.origin_lanes[column.origin_lane_idx].steps)
        start = column.departure_step + fg.takeoff_steps[column.level] + lane_steps
        windows = _column_endpoint_steps(column, fg, cfg, start + len(column.cell_path) - 1)
        return _collect_column_claims(column, fg, cfg, derive_cell_window(cfg), start, windows)
    if cfg != fg._cfg:
        raise ValueError("column claims require the SimConfig used to build the flight graph")
    if column.flight_id != fg.request.flight_id:
        raise ValueError(
            f"column flight_id {column.flight_id} does not match graph flight "
            f"{fg.request.flight_id}"
        )
    if not 0 <= column.level < len(fg.levels):
        raise ValueError(
            f"column level {column.level} is outside graph levels [0, {len(fg.levels)})"
        )
    if not fg.base_step <= column.departure_step <= fg.latest_departure_step:
        raise ValueError(
            f"column departure step {column.departure_step} is outside legal range "
            f"[{fg.base_step}, {fg.latest_departure_step}]"
        )

    offsets = derive_cell_window(cfg)
    width = offsets[1] - offsets[0] + 1
    if W is not None:
        if isinstance(W, tuple):
            if W != offsets:
                raise ValueError(f"supplied cell-window offsets {W} != derived offsets {offsets}")
        else:
            try:
                supplied_width = operator.index(W)
            except TypeError as exc:
                raise TypeError("W must be an integer width or an (lo, hi) offset tuple") from exc
            if supplied_width != width:
                raise ValueError(
                    f"supplied cell-window width {supplied_width} != derived width {width}"
                )

    path = tuple(tuple(cell) for cell in column.cell_path)
    certificate_key = (
        column.flight_id,
        column.level,
        column.origin_lane_idx,
        column.dest_lane_idx,
        path,
    )
    with fg._search_cache.lock:
        certificate = fg._search_cache.certified_claims.get(certificate_key)
        if certificate is not None:
            fg._search_cache.certified_claims.move_to_end(certificate_key)
    if certificate is not None:
        old_departure, cached_claims, old_windows = certificate
        lane_steps = (0 if column.origin_lane_idx is None else
                      fg.origin_lanes[column.origin_lane_idx].steps)
        arrival = column.departure_step + fg.takeoff_steps[column.level] + lane_steps + len(path) - 1
        if arrival > fg.max_step:
            raise ValueError(f"column arrives at step {arrival}, beyond graph maximum {fg.max_step}")
        delta = column.departure_step - old_departure
        if delta == 0:
            return cached_claims
        windows = _column_endpoint_steps(column, fg, cfg, arrival)
        relative_windows = tuple((w.start - column.departure_step, w.stop - column.departure_step)
                                 for w in windows)
        # Endpoint rounding depends on absolute floating-point timestamps. Only translate
        # the full union when its windows translate exactly; otherwise rebuild rows below.
        if relative_windows == old_windows:
            return frozenset(
                RowKey.cell(*row.cell_coord, row.level, row.step + delta)
                if row.kind == "cell" else RowKey.term(row.terminal_id, row.step + delta)
                for row in cached_claims
            )
    if len(path) < 2:
        raise ValueError("a column path must contain at least two cells and one lateral hop")
    if certificate is None and any(cell not in fg.corridor_cells for cell in path):
        invalid = next(cell for cell in path if cell not in fg.corridor_cells)
        raise ValueError(f"column cell {invalid} is outside the flight graph corridor")
    for a, b in (() if certificate is not None else zip(path, path[1:])):
        if hg.hex_distance(a, b) != 1:
            raise ValueError(f"column contains a non-neighbour lateral hop {a} -> {b}")
        if (a, b) in fg.forbidden_hops:
            raise ValueError(f"column hop {a} -> {b} overlaps permanent static terminal airspace")
    # Certificates are per graph, and the rule is part of graph identity, so a cached
    # path already passed this check on first certification.
    if certificate is None and fg.turn_below_90 and not path_turns_valid(path):
        raise ValueError("column turns by 90 degrees or more at an interior cruise cell")

    origin_lane_steps = 0
    if fg.origin_terminal is not None:
        origin_lane = _selected_lane(fg.origin_lanes, column.origin_lane_idx, "origin")
        if path[0] != origin_lane.cell:
            raise ValueError(
                f"column origin cell {path[0]} does not match selected lane {origin_lane.cell}"
            )
        origin_lane_steps = origin_lane.steps
    else:
        if column.origin_lane_idx is not None:
            raise ValueError("origin_lane_idx must be None for a customer origin")
        if path[0] != fg.origin_cell:
            raise ValueError(
                f"column origin cell {path[0]} does not match request cell {fg.origin_cell}"
            )

    if fg.dest_terminal is not None:
        dest_lane = _selected_lane(fg.dest_lanes, column.dest_lane_idx, "dest")
        if path[-1] != dest_lane.cell:
            raise ValueError(
                f"column destination cell {path[-1]} does not match selected lane {dest_lane.cell}"
            )
    else:
        if column.dest_lane_idx is not None:
            raise ValueError("dest_lane_idx must be None for a customer destination")
        if path[-1] != fg.dest_cell:
            raise ValueError(
                f"column destination cell {path[-1]} does not match request cell {fg.dest_cell}"
            )

    corridor_start_step = column.departure_step + fg.takeoff_steps[column.level] + origin_lane_steps
    arrival_step = corridor_start_step + len(path) - 1
    if arrival_step > fg.max_step:
        raise ValueError(
            f"column arrives at step {arrival_step}, beyond graph maximum {fg.max_step}"
        )

    # The graph's additive hop slack is an enumeration bound, not permission
    # to violate the simulation's independent budgets.  Translation is the
    # canonical gate: using its actual resampled centerline here prevents even
    # an ulp-level difference from making a claimed column fail at filing.
    from .translate import column_to_intent

    intent = (_intent if _intent is not None else column_to_intent(column, fg.request, cfg)) if certificate is None else None
    if intent is not None and intent.status is not IntentStatus.ACCEPTED:
        raise ValueError(
            f"column violates max_detour_factor or another translation budget "
            f"({intent.denial_reason})"
        )

    # Static walls are deliberately not master rows.  The discrete
    # ``terminal_cells`` pruning is useful for pricing, but it is not an exact
    # representation of the permanent cylinders (and a single box can touch
    # both endpoint hubs while carrying only one tag).  Certify every canonical
    # column against the exact ledger geometry before exposing its row claims.
    # Reusing ``volumes_conflict`` is essential: its same-terminal cylinder
    # exemption is precisely the one applied again when the intent is filed.
    if fg.static_walls and intent is not None:
        for volume in intent.volumes:
            for wall, _wall_bound in fg._wall_index.candidates(volume.flat_aabb()):
                if volumes_conflict(volume, wall):
                    raise ValueError(
                        f"column for flight {column.flight_id} overlaps permanent static "
                        f"terminal {wall.terminal_id!r}"
                    )

    windows = _column_endpoint_steps(column, fg, cfg, arrival_step)
    result = (
        _collect_column_claims(column, fg, cfg, offsets, corridor_start_step, windows)
        if _claims is None else _claims
    )
    _store_claim_certificate(
        fg, certificate_key, column.departure_step, result,
        tuple((w.start - column.departure_step, w.stop - column.departure_step) for w in windows),
    )
    return result


def _store_claim_certificate(
    fg: FlightGraph,
    key: tuple[Any, ...],
    departure_step: int,
    claims: frozenset[RowKey],
    relative_windows: tuple[tuple[int, int], ...],
) -> None:
    """Insert one claim certificate as most recent, evicting beyond ``_MAX_CERTIFIED_COLUMNS``."""
    cache = fg._search_cache
    with cache.lock:
        cache.certified_claims[key] = (departure_step, claims, relative_windows)
        cache.certified_claims.move_to_end(key)
        while len(cache.certified_claims) > _MAX_CERTIFIED_COLUMNS:
            cache.certified_claims.popitem(last=False)


def record_claim_certificate(
    column: Column, fg: FlightGraph, cfg: SimConfig, claims: frozenset[RowKey]
) -> None:
    """Leave the claim-certificate cache as ``column_claims`` would after certifying ``column``.

    For a caller that certified ``column`` by other means (pricing's volume-free path).  Later
    ``column_claims`` calls then find the same certificates, in the same LRU order, as if the
    route had gone through it: a same-departure or exactly translatable certificate is only
    refreshed, anything else is replaced.

    Parameters
    ------------
    - column (Column): the certified route; it must arrive by ``fg.max_step``.
    - fg (FlightGraph): the graph whose cache holds the certificates.
    - cfg (SimConfig): the graph's config, for the endpoint windows.
    - claims (frozenset[RowKey]): the route's exact claim rows.

    Return
    --------
    - output (None): updates ``fg._search_cache.certified_claims``.
    """
    path = tuple(tuple(cell) for cell in column.cell_path)
    key = (column.flight_id, column.level, column.origin_lane_idx, column.dest_lane_idx, path)
    cache = fg._search_cache
    with cache.lock:
        certificate = cache.certified_claims.get(key)
        if certificate is not None:
            cache.certified_claims.move_to_end(key)
    if certificate is not None and certificate[0] == column.departure_step:
        return
    lane_steps = (0 if column.origin_lane_idx is None else
                  fg.origin_lanes[column.origin_lane_idx].steps)
    arrival = column.departure_step + fg.takeoff_steps[column.level] + lane_steps + len(path) - 1
    windows = tuple(
        (w.start - column.departure_step, w.stop - column.departure_step)
        for w in _column_endpoint_steps(column, fg, cfg, arrival)
    )
    if certificate is not None and certificate[2] == windows:
        return
    _store_claim_certificate(fg, key, column.departure_step, claims, windows)


__all__ = [
    "Cell",
    "FlightGraph",
    "FlightGraphInfeasible",
    "RowIndex",
    "RowKey",
    "StaticTerminalCatalog",
    "build_flight_graph",
    "column_claims",
]
