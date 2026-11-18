"""Lightweight supergraph geometry used only by conservative completion bounds."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ...volumes import corridor_segment_volume
from .network import RowKey
from .zeta_bound import _Edge, _distance_error
from .zeta_geometry import volume_spatial_cells

if TYPE_CHECKING:
    from .zeta import Cell, ZetaGraph


_EMPTY: frozenset[tuple[int, int, int]] = frozenset()
_FOOTPRINT_CACHE_LIMIT = 2048


def _topology(
    graph: ZetaGraph, deadline: float | None,
) -> tuple[tuple[Cell, ...], tuple[tuple[_Edge, ...], ...]] | None:
    """Cache every neighbor connection with exact subpart lengths and no wall tests.

    Parameters
    ------------
    - graph (ZetaGraph): immutable waypoint, neighbor and subdivision authority.
    - deadline (float | None): monotonic preparation cutoff.

    Return
    --------
    - output (tuple | None): complete nodes and edges with empty spatial footprints,
      or None when preparation is interrupted.
    """
    cached = getattr(graph, "_completion_bound_topology", None)
    if cached is not None:
        return cached
    cells = tuple(sorted(graph.cells | {cell for _, cell, _ in graph.origins}
                         | {cell for _, cell in graph.destinations}))
    identifiers = {cell: index for index, cell in enumerate(cells)}
    cfg = graph.cfg
    outgoing = []
    for cell in cells:
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        a = graph.point(cell)
        edges = []
        for target in graph.neighbors(cell):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            b = graph.point(target)
            dx, dy = b[0] - a[0], b[1] - a[1]
            total = math.sqrt(dx * dx + dy * dy)
            nsub = max(1, math.ceil(total / cfg.corridor_segment_len_m))
            durations, lengths = [], []
            for k in range(1, nsub + 1):
                # Preserve graph.parts' interpolation and arithmetic order exactly.
                f0, f1 = (k - 1) / nsub, k / nsub
                sa = a[0] + f0 * dx, a[1] + f0 * dy, a[2]
                sb = a[0] + f1 * dx, a[1] + f1 * dy, a[2]
                hx, hy = sb[0] - sa[0], sb[1] - sa[1]
                length = math.sqrt(hx * hx + hy * hy)
                lengths.append(length)
                durations.append(max(length / cfg.nominal_speed_mps, 1e-3))
            edges.append(_Edge(identifiers[target], tuple(durations), tuple(lengths),
                               (_EMPTY,) * nsub))
        outgoing.append(tuple(edges))
    if deadline is not None and time.perf_counter() >= deadline:
        return None
    result = cells, tuple(outgoing)
    graph._completion_bound_topology = result
    return result


def geometry(
    graph: ZetaGraph, prices: Mapping[RowKey, float] | None, deadline: float | None = None,
) -> tuple[tuple[Cell, ...], tuple[tuple[_Edge, ...], ...]] | None:
    """Build a wall-relaxed graph retaining only nearby positively priced footprints.

    Parameters
    ------------
    - graph (ZetaGraph): immutable finite graph and simulator corridor geometry.
    - prices (Mapping[RowKey, float] | None): current signed prices; None requests
      lengths and durations only. Terminal rows do not price corridor footprints.
    - deadline (float | None): monotonic preparation cutoff; incomplete variants
      are not published.

    Return
    --------
    - output (tuple | None): ordered cells and every neighbor edge, with exact
      positive spatial membership on retained subparts, or an interrupted build.
      Omitting nonnegative charges and allowing forbidden edges only relaxes a bound.
    """
    if deadline is not None and time.perf_counter() >= deadline:
        return None
    static = _topology(graph, deadline)
    if static is None or prices is None:
        return static
    support = set()
    for index, (row, price) in enumerate(prices.items()):
        if index % 256 == 0 and deadline is not None and time.perf_counter() >= deadline:
            return None
        if price > 0.0 and row[0] == "cell":
            support.add(row[1:4])
    spaces = frozenset(support)
    if not spaces:
        return static
    cached = getattr(graph, "_completion_bound_priced_geometry", None)
    if cached is not None and cached[0] == spaces:
        return cached[1]

    cfg = graph.cfg
    pitch = float(cfg.corridor_segment_len_m)
    margin = cfg.corridor_width_m + pitch / math.sqrt(3.0) + 1e-7
    bucket_size = max(pitch, margin)
    row_height = pitch * math.sqrt(3.0) / 2.0
    buckets = {
        (math.floor(pitch * (q + r / 2.0) / bucket_size),
         math.floor(row_height * r / bucket_size))
        for q, r, _ in spaces
    }
    footprints = getattr(graph, "_completion_bound_footprints", None)
    if footprints is None:
        footprints = OrderedDict()
        graph._completion_bound_footprints = footprints
    cells, outgoing = static
    priced_outgoing = []
    for source, group in enumerate(outgoing):
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        a = graph.point(cells[source])
        priced_edges = []
        for edge in group:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            b = graph.point(cells[edge.target])
            dx, dy = b[0] - a[0], b[1] - a[1]
            nsub = len(edge.durations)
            covered = []
            for k, duration in enumerate(edge.durations, 1):
                f0, f1 = (k - 1) / nsub, k / nsub
                sa = a[0] + f0 * dx, a[1] + f0 * dy, a[2]
                sb = a[0] + f1 * dx, a[1] + f1 * dy, a[2]
                x0 = math.floor((min(sa[0], sb[0]) - margin) / bucket_size)
                x1 = math.floor((max(sa[0], sb[0]) + margin) / bucket_size)
                y0 = math.floor((min(sa[1], sb[1]) - margin) / bucket_size)
                y1 = math.floor((max(sa[1], sb[1]) + margin) / bucket_size)
                near = any((x, y) in buckets for x in range(x0, x1 + 1)
                           for y in range(y0, y1 + 1))
                if not near:
                    covered.append(_EMPTY)
                    continue
                if deadline is not None and time.perf_counter() >= deadline:
                    return None
                key = source, edge.target, k
                footprint = footprints.get(key)
                if footprint is None:
                    # Tags affect collision exemptions, never a BoxSpec's cell footprint.
                    shape = corridor_segment_volume(sa, 0.0, sb, duration, cfg).shape
                    # Bypass the global shape-retaining LRU; keep bounded cell sets only.
                    if graph.options.leg_cache:
                        from .zeta_leg_cache import get_footprint
                        footprint = get_footprint(shape, cfg)
                    else:
                        footprint = volume_spatial_cells.__wrapped__(shape, cfg)
                    footprints[key] = footprint
                    if len(footprints) > _FOOTPRINT_CACHE_LIMIT:
                        footprints.popitem(last=False)
                else:
                    footprints.move_to_end(key)
                covered.append(footprint & spaces)
            priced_edges.append(_Edge(edge.target, edge.durations, edge.lengths, tuple(covered))
                                if any(covered) else edge)
        priced_outgoing.append(tuple(priced_edges))
    if deadline is not None and time.perf_counter() >= deadline:
        return None
    result = cells, tuple(priced_outgoing)
    graph._completion_bound_priced_geometry = spaces, result
    return result


def distance_error(
    graph: ZetaGraph, cells: tuple[Cell, ...], outgoing: tuple[tuple[_Edge, ...], ...],
) -> tuple[float, int, float] | None:
    """Reuse the existing numerical allowance once for this lightweight topology.

    Parameters
    ------------
    - graph (ZetaGraph): immutable route length budget and endpoint coordinates.
    - cells (tuple[Cell, ...]): cells returned by geometry for this graph.
    - outgoing (tuple[tuple[_Edge, ...], ...]): complete lightweight neighbor topology;
      priced variants differ only in footprints and therefore share this allowance.

    Return
    --------
    - output (tuple | None): distance error, maximum edge count and canonical length
      ceiling, or None when the existing arithmetic guards cannot establish them.
    """
    if not hasattr(graph, "_completion_bound_distance_error"):
        graph._completion_bound_distance_error = (
            _distance_error(graph, cells, outgoing) if any(outgoing) else None
        )
    return graph._completion_bound_distance_error
