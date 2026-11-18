"""Conservative sampled-price continuations on relaxed waypoint/time intervals."""

from __future__ import annotations

import math
import time
import heapq
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from .network import RowKey
from .zeta_geometry import volume_spatial_cells

if TYPE_CHECKING:
    from .zeta import Cell, ZetaGraph


@dataclass(frozen=True, slots=True)
class _Edge:
    """A role-relaxed connection with simulator-owned subpart geometry and timing."""

    target: int
    durations: tuple[float, ...]
    lengths: tuple[float, ...]
    covered: tuple[frozenset[tuple[int, int, int]], ...]


def _bin_index(stamp: float, delta: float) -> int:
    """Use multiplied boundaries so nonbinary periods cannot misclassify exact contact."""
    index = math.floor(stamp / delta)
    while (index + 1) * delta <= stamp:
        index += 1
    while index * delta > stamp:
        index -= 1
    return index


def _geometry(
    graph: ZetaGraph, deadline: float | None,
) -> tuple[tuple[Cell, ...], tuple[tuple[_Edge, ...], ...]] | None:
    """Cache every connection admitted by any endpoint role, with its unchanged shapes.

    Parameters
    ------------
    - graph (ZetaGraph): exact candidate-waypoint and simulator-geometry authority.
    - deadline (float | None): preparation cutoff; incomplete caches are never published.

    Return
    --------
    - output (tuple | None): ordered nodes and outgoing relaxed edges, or an interrupted build.
    """
    cached = getattr(graph, "_future_price_geometry", None)
    if cached is not None:
        return cached
    cells = tuple(sorted(graph.cells | {cell for _, cell, _ in graph.origins}
                         | {cell for _, cell in graph.destinations}))
    identifiers = {cell: index for index, cell in enumerate(cells)}
    outgoing = []
    for cell in cells:
        edges = []
        for target in graph.neighbors(cell):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            parts = None
            for first, last in ((False, False), (True, False), (False, True), (True, True)):
                parts = graph.parts(cell, target, first, last)
                if parts is not None:
                    break
            if parts:
                edges.append(_Edge(
                    identifiers[target], tuple(part.duration for part in parts),
                    tuple(part.length for part in parts),
                    tuple(volume_spatial_cells(part.shape, graph.cfg) for part in parts),
                ))
        outgoing.append(tuple(edges))
    result = cells, tuple(outgoing)
    graph._future_price_geometry = result
    return result


def _completion_delta(graph: ZetaGraph, outgoing: tuple[tuple[_Edge, ...], ...]) -> float:
    """Use larger relaxed intervals while keeping every edge strictly time-advancing.

    Parameters
    ------------
    - graph (ZetaGraph): unchanged period length and permitted speed variation.
    - outgoing (tuple): all relaxed edges with exact nominal subpart durations.

    Return
    --------
    - output (float): candidate interval width; timing enclosures still validate advancement.
    """
    minimum = min(math.fsum(edge.durations) for group in outgoing for edge in group)
    fraction = graph.options.speed_fraction if graph.options.arrival_snap_s > 0.0 else 0.0
    return min(graph.cfg.dt_s, .9 * minimum / (1.0 + fraction))


def _price_units(
    prices: Mapping[RowKey, float], extra_budget: float = 0.0,
) -> tuple[dict[RowKey, int], float] | None:
    """Round positive prices downward to shared power-of-two units below an int64 budget.

    Parameters
    ------------
    - prices (Mapping[RowKey, float]): finite frozen row prices; nonpositive rows are ignored.
    - extra_budget (float): conservative distance-cost budget sharing the same integer scale.

    Return
    --------
    - output (tuple[dict[RowKey, int], float] | None): integer units and their quantum, or None
      when no usable positive prices exist. Total units remain strictly below 2**50.
    """
    positive = {RowKey(row): float(price) for row, price in prices.items() if price > 0.0}
    if not positive or any(not math.isfinite(price) for price in positive.values()):
        return None
    values = [*positive.values(), extra_budget] if extra_budget > 0.0 else list(positive.values())
    if any(not math.isfinite(value) for value in values):
        return None
    exponent = max(-30, max(math.frexp(price)[1] for price in values)
                   + (len(values) - 1).bit_length() - 50)
    if exponent > 1023:
        return None
    quantum = math.ldexp(1.0, exponent)
    weights = {}
    for row, price in positive.items():
        numerator, denominator = price.as_integer_ratio()
        units = (numerator // (denominator << exponent) if exponent >= 0
                 else (numerator << -exponent) // denominator)
        if units:
            weights[row] = units
    if (not weights or sum(weights.values()) >= 2**50
            or not math.isfinite(sum(weights.values()) * quantum)):
        return None
    return weights, quantum


def _distance_error(
    graph: ZetaGraph, cells: tuple[Cell, ...], outgoing: tuple[tuple[_Edge, ...], ...],
) -> tuple[float, int, float] | None:
    """Bound whole-route differences between graph lengths and canonical customer costs.

    Parameters
    ------------
    - graph (ZetaGraph): accepted routes' exact length budget and endpoint geometry.
    - cells (tuple[Cell, ...]): candidate waypoint coordinates.
    - outgoing (tuple[tuple[_Edge, ...], ...]): all permitted role-relaxed connections.

    Return
    --------
    - output (tuple | None): distance error, maximum edge count and canonical length ceiling,
      or None when finite arithmetic and summand-count guards cannot be established.
    """
    edges = [edge for group in outgoing for edge in group]
    budget = float(graph.max_length + 1e-7)
    unit = math.ulp(budget)
    minimum = min(sum(edge.lengths) for edge in edges)
    if not math.isfinite(budget) or minimum <= unit:
        return None
    quotient = Fraction(budget) / (Fraction(minimum) - Fraction(unit))
    edge_limit = (quotient.numerator + quotient.denominator - 1) // quotient.denominator
    count = edge_limit * max(len(edge.lengths) for edge in edges)
    if (count + 2) * 2.0**-53 >= 0.5:
        return None
    points = [graph.point(cell) for cell in cells]
    coordinate = max(abs(float(x)) for point in (*points, graph.fg.request.origin,
                                                graph.fg.request.dest) for x in point)
    discrepancy = Fraction(0)
    for source, group in enumerate(outgoing):
        a = points[source]
        for edge in group:
            b = points[edge.target]
            dx, dy = b[0] - a[0], b[1] - a[1]
            for k in range(1, len(edge.lengths) + 1):
                fraction = k / len(edge.lengths)
                x, y = a[0] + fraction * dx, a[1] + fraction * dy
                coordinate = max(coordinate, abs(x), abs(y))
            discrepancy = max(discrepancy, abs(Fraction(x) - Fraction(b[0]))
                              + abs(Fraction(y) - Fraction(b[1])))
    def up(value: float) -> float:
        """Round an already rounded scalar outward by one representable value."""
        return math.nextafter(value, math.inf)

    diameter = up(2.0 * coordinate)
    square = up(diameter * diameter)
    squared_norm = up(2.0 * square)
    norm = up(math.sqrt(squared_norm))
    if not math.isfinite(norm):
        return None
    norm_error = up(math.fsum((
        up(2.0 * math.ulp(diameter)),
        up(math.sqrt(up(math.fsum((up(2.0 * math.ulp(square)), math.ulp(squared_norm)))))),
        math.ulp(norm),
    )))
    sum_scale = up(2 * (count + 2) * norm)
    error = up(math.fsum((
        up(count * unit), up(2 * (count + 2) * norm_error),
        up((count + 1) * up(float(discrepancy))),
        up((count + 1) * math.ulp(sum_scale)),
    )))
    return (error, edge_limit, sum_scale) if math.isfinite(error) and error < budget else None


def _fraction_up(value: Fraction) -> float:
    """Convert an exact rational upward, retaining exact representable values."""
    result = float(value)
    return math.nextafter(result, math.inf) if Fraction(result) < value else result


def _fraction_down(value: Fraction) -> float:
    """Convert an exact rational downward, retaining exact representable values."""
    result = float(value)
    return math.nextafter(result, -math.inf) if Fraction(result) > value else result


def _customer_attachments(graph: ZetaGraph) -> float | None:
    """Lower-bound customer-to-waypoint attachment distances using exact squared norms.

    Parameters
    ------------
    - graph (ZetaGraph): original customer coordinates and permitted endpoint waypoints.

    Return
    --------
    - output (float | None): downward sum of the minimum origin and destination attachment,
      zero for terminal endpoints, or None when a finite lower bound cannot be constructed.
    """
    if graph.fg.origin_terminal is not None or graph.fg.dest_terminal is not None:
        return 0.0
    attachments = []
    for point, candidates in (
        (graph.fg.request.origin, (cell for _, cell, _ in graph.origins)),
        (graph.fg.request.dest, (cell for _, cell in graph.destinations)),
    ):
        distances = []
        for cell in candidates:
            waypoint = graph.point(cell)
            dx = Fraction(float(point[0])) - Fraction(waypoint[0])
            dy = Fraction(float(point[1])) - Fraction(waypoint[1])
            squared = dx * dx + dy * dy
            distance = math.hypot(float(point[0]) - waypoint[0], float(point[1]) - waypoint[1])
            if not math.isfinite(distance):
                return None
            while Fraction(distance) ** 2 > squared:
                distance = math.nextafter(distance, -math.inf)
            distances.append(distance)
        if not distances:
            return None
        attachments.append(min(distances))
    try:
        return _fraction_down(sum(map(Fraction, attachments), Fraction(0)))
    except OverflowError:
        return None


def _distance_boundary(
    outgoing: tuple[tuple[_Edge, ...], ...], costs: list[int], goals: Iterable[int],
) -> np.ndarray:
    """Compute exact integer remaining distance on the same relaxed static graph.

    Parameters
    ------------
    - outgoing (tuple[tuple[_Edge, ...], ...]): role-relaxed connections in cost order.
    - costs (list[int]): downward weighted distance of each connection.
    - goals (Iterable[int]): allowed stopping nodes.

    Return
    --------
    - output (np.ndarray): nonnegative lower bounds, with zero for unreachable nodes.
    """
    reverse: list[list[tuple[int, int]]] = [[] for _ in outgoing]
    index = 0
    for source, group in enumerate(outgoing):
        for edge in group:
            reverse[edge.target].append((source, costs[index]))
            index += 1
    distances: list[int | None] = [None] * len(outgoing)
    queue = []
    for goal in goals:
        distances[goal] = 0
        heapq.heappush(queue, (0, goal))
    while queue:
        value, target = heapq.heappop(queue)
        if distances[target] != value:
            continue
        for source, cost in reverse[target]:
            candidate = value + cost
            if distances[source] is None or candidate < distances[source]:
                distances[source] = candidate
                heapq.heappush(queue, (candidate, source))
    return np.asarray([0 if value is None else value for value in distances], dtype=np.int64)


@njit(cache=True, nogil=True)
def future_paid_units(
    paid: np.ndarray, row_units: np.ndarray, row_samples: np.ndarray, stamp: float,
) -> int:
    """Discount only paid rows whose ownership sample can occur in the continuation.

    Parameters
    ------------
    - paid (np.ndarray): unique live paid row IDs.
    - row_units (np.ndarray): aligned nonnegative sampled-row units; zero for unsampled rows.
    - row_samples (np.ndarray): aligned ownership times; minus infinity for unsampled rows.
    - stamp (float): exact current arrival timestamp.

    Return
    --------
    - output (int): exact quantized discount without allocating temporary arrays.
    """
    total = 0
    for row in paid:
        if row_samples[row] > stamp:
            total += row_units[row]
    return total


@njit(cache=True, nogil=True)
def _time_enclosures(
    first_bin: int, bins: int, delta: float, snap: float, fraction: float,
    part_offsets: np.ndarray, durations: np.ndarray, totals: np.ndarray, fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Enclose adapter clocks by monotone evaluation of their actual floating operations.

    Parameters
    ------------
    - first_bin (int): absolute starting interval index.
    - bins (int): number of source intervals.
    - delta (float): interval width.
    - snap (float): configured absolute arrival-grid spacing, or zero.
    - fraction (float): configured fractional speed allowance.
    - part_offsets (np.ndarray): per-edge ranges in flattened subpart arrays.
    - durations (np.ndarray): exact nominal subpart increments.
    - totals (np.ndarray): math.fsum of each edge's nominal increments.
    - fractions (np.ndarray): exact adapter prefix/total quotients for interior subparts.

    Return
    --------
    - output (tuple): first/last target bins, earliest starts, latest ends, latest starts,
      earliest ends and success flag.
      An unsupported grid or a nonadvancing relaxed edge declines the optional bound.
    """
    first_targets = np.empty((bins, len(totals)), dtype=np.int64)
    last_targets = np.empty_like(first_targets)
    starts = np.empty((bins, len(durations)), dtype=np.float64)
    ends = np.empty_like(starts)
    latest_starts = np.empty_like(starts)
    earliest_ends = np.empty_like(starts)
    for b in range(bins):
        lower = (first_bin + b) * delta
        upper = (first_bin + b + 1) * delta
        for edge in range(len(totals)):
            snap_low, snap_high = math.inf, -math.inf
            if snap > 0.0:
                low_position, high_position = (lower + totals[edge]) / snap, (upper + totals[edge]) / snap
                if (not math.isfinite(low_position) or not math.isfinite(high_position)
                        or max(abs(low_position), abs(high_position)) >= 2**48):
                    return first_targets, last_targets, starts, ends, latest_starts, earliest_ends, False
                first_tick, last_tick = math.floor(low_position), math.floor(high_position) + 1
                if last_tick - first_tick > 32:
                    return first_targets, last_targets, starts, ends, latest_starts, earliest_ends, False
                minimum = totals[edge] / (1.0 + fraction)
                maximum = totals[edge] / (1.0 - fraction)
                magnitude = max(abs(lower), abs(upper))
                # nextafter(x,+inf)-x equals ulp(x) for nonnegative finite x.
                start_ulp = np.nextafter(magnitude, math.inf) - magnitude
                total_ulp = np.nextafter(totals[edge], math.inf) - totals[edge]
                for tick in range(first_tick, last_tick + 1):
                    arrival = tick * snap
                    magnitude = abs(arrival)
                    arrival_ulp = np.nextafter(magnitude, math.inf) - magnitude
                    tolerance = start_ulp + arrival_ulp + total_ulp
                    if (arrival - lower > 0.0 and arrival - lower >= minimum - tolerance
                            and arrival - upper <= maximum + tolerance):
                        snap_low, snap_high = min(snap_low, arrival), max(snap_high, arrival)
            nominal_low, nominal_high = lower, upper
            previous_low = lower
            previous_high = upper
            for part in range(part_offsets[edge], part_offsets[edge + 1]):
                nominal_low += durations[part]
                nominal_high += durations[part]
                end_low, end_high = nominal_low, nominal_high
                if snap_low != math.inf:
                    if part + 1 == part_offsets[edge + 1]:
                        end_low, end_high = min(end_low, snap_low), max(end_high, snap_high)
                    else:
                        end_low = min(end_low, lower + (snap_low - upper) * fractions[part])
                        end_high = max(end_high, upper + (snap_high - lower) * fractions[part])
                starts[b, part], ends[b, part] = previous_low, end_high
                latest_starts[b, part], earliest_ends[b, part] = previous_high, end_low
                previous_low = end_low
                previous_high = end_high
            first = math.floor(end_low / delta)
            while (first + 1) * delta <= end_low:
                first += 1
            while first * delta > end_low:
                first -= 1
            last = math.floor(end_high / delta)
            while (last + 1) * delta <= end_high:
                last += 1
            while last * delta > end_high:
                last -= 1
            first_targets[b, edge], last_targets[b, edge] = first - first_bin, last - first_bin
            if first - first_bin <= b:
                return first_targets, last_targets, starts, ends, latest_starts, earliest_ends, False
    return first_targets, last_targets, starts, ends, latest_starts, earliest_ends, True


@njit(cache=True, nogil=True)
def _coverage_unions(membership: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Price every subpart union exactly, without repaying rows shared by several parts.

    Parameters
    ------------
    - membership (np.ndarray): integer prices grouped by exact subpart-membership mask/period.
    - offsets (np.ndarray): per-edge blocks of all possible subpart masks.

    Return
    --------
    - output (np.ndarray): union prices for each edge/mask/period. The input is scratch storage.
    """
    result = np.empty_like(membership)
    for edge in range(len(offsets) - 1):
        start, stop = offsets[edge], offsets[edge + 1]
        size = stop - start
        bit = 1
        while bit < size:
            for mask in range(size):
                if mask & bit:
                    for sample in range(membership.shape[1]):
                        membership[start + mask, sample] += membership[start + (mask ^ bit), sample]
            bit *= 2
        for mask in range(size):
            for sample in range(membership.shape[1]):
                result[start + mask, sample] = (membership[stop - 1, sample]
                                               - membership[start + (size - 1 - mask), sample])
    return result


@njit(cache=True, nogil=True)
def _backward_chunk(
    table: np.ndarray, first_bin: int, delta: float,
    edge_offsets: np.ndarray, targets: np.ndarray, part_offsets: np.ndarray,
    first_offsets: np.ndarray, last_offsets: np.ndarray,
    earliest_starts: np.ndarray, latest_ends: np.ndarray,
    sample_times: np.ndarray, part_costs: np.ndarray, goals: np.ndarray,
    distance_costs: np.ndarray, boundary: np.ndarray,
    latest_starts: np.ndarray, earliest_ends: np.ndarray,
    period_starts: np.ndarray, period_ends: np.ndarray, buffer: float,
    mask_offsets: np.ndarray, union_costs: np.ndarray,
    total_units: int, first: int, stop: int,
) -> None:
    """Fill a descending block of the acyclic, integer-valued continuation relaxation.

    Parameters
    ------------
    - table (np.ndarray): mutable [time-bin, node] lower-bound storage.
    - first_bin (int): absolute index of the table's first time interval.
    - delta (float): interval width.
    - edge_offsets (np.ndarray): outgoing-edge ranges for each node.
    - targets (np.ndarray): target node of each relaxed connection.
    - part_offsets (np.ndarray): ranges of subparts for each connection.
    - first_offsets (np.ndarray): smallest target bin per source-bin/connection.
    - last_offsets (np.ndarray): largest target bin per source-bin/connection.
    - earliest_starts (np.ndarray): lower absolute subpart starts by source bin.
    - latest_ends (np.ndarray): upper absolute subpart ends by source bin.
    - sample_times (np.ndarray): ordered interior samples of priced reservation periods.
    - part_costs (np.ndarray): positive integer row charges by subpart and sample.
    - goals (np.ndarray): relaxed destinations, at which stopping is free.
    - distance_costs (np.ndarray): exact downward-rounded weighted connection distances.
    - boundary (np.ndarray): remaining distance costs after the last priced sample.
    - latest_starts (np.ndarray): upper absolute subpart starts by source bin.
    - earliest_ends (np.ndarray): lower absolute subpart ends by source bin.
    - period_starts (np.ndarray): exact multiplied starts of sampled reservation periods.
    - period_ends (np.ndarray): exact multiplied ends of sampled reservation periods.
    - buffer (float): unchanged nonnegative reservation time buffer.
    - mask_offsets (np.ndarray): per-edge union-price block, empty for point-only fallback.
    - union_costs (np.ndarray): exact integer prices of each subpart union and period.
    - total_units (int): upper cap on all distinct positive row units.
    - first (int): latest table row to fill, inclusive.
    - stop (int): earliest table row to fill, exclusive.

    Return
    --------
    - output (None): fills the selected rows from already completed later rows.
    """
    sampled_prefix = np.empty(len(sample_times) + 1, dtype=np.int64)
    for b in range(first, stop, -1):
        upper = (first_bin + b + 1) * delta
        sample_first = np.searchsorted(sample_times, upper, side="right")
        for node in range(table.shape[1]):
            if goals[node] or edge_offsets[node] == edge_offsets[node + 1]:
                table[b, node] = 0
                continue
            best = total_units
            for edge in range(edge_offsets[node], edge_offsets[node + 1]):
                latest_lower = (first_bin + last_offsets[b, edge]) * delta
                last_sample = np.searchsorted(sample_times, latest_lower, side="left")
                sampled_prefix[sample_first] = 0
                for sample in range(sample_first, last_sample):
                    tau = sample_times[sample]
                    guaranteed = 0
                    if len(mask_offsets):
                        for part in range(part_offsets[edge], part_offsets[edge + 1]):
                            if (latest_starts[b, part] - buffer < period_ends[sample]
                                    and earliest_ends[b, part] + buffer > period_starts[sample]):
                                guaranteed |= 1 << (part - part_offsets[edge])
                    smallest = total_units
                    possible = False
                    for part in range(part_offsets[edge], part_offsets[edge + 1]):
                        if earliest_starts[b, part] <= tau <= latest_ends[b, part]:
                            possible = True
                            if len(mask_offsets):
                                mask = guaranteed | (1 << (part - part_offsets[edge]))
                                value = union_costs[mask_offsets[edge] + mask, sample]
                            else:
                                value = part_costs[part, sample]
                            smallest = min(smallest, value)
                            if smallest == 0:
                                break
                    sampled_prefix[sample + 1] = sampled_prefix[sample] + (smallest if possible else 0)
                for end_bin in range(first_offsets[b, edge], last_offsets[b, edge] + 1):
                    end_lower = (first_bin + end_bin) * delta
                    sample_stop = np.searchsorted(sample_times, end_lower, side="left")
                    charge = distance_costs[edge]
                    if sample_stop > sample_first:
                        charge += sampled_prefix[sample_stop]
                    if end_bin < table.shape[0]:
                        charge += table[end_bin, targets[edge]]
                    else:
                        charge += boundary[targets[edge]]
                    best = min(best, charge)
                    if best == 0:
                        break
                if best == 0:
                    break
            table[b, node] = best


@dataclass(slots=True)
class FuturePriceBound:
    """A sampled-price lower bound that discounts live reservations before use.

    Attributes
    ------------
    - row_units (dict[RowKey, int]): downward-rounded positive prices of sampled cell rows.
    - row_samples (dict[RowKey, float]): ownership sample of each row eligible for a discount.
    - quantum (float): power-of-two value of one integer unit.
    - build_s (float): wall time of geometry, price preparation and table construction.
    - table_states (int): number of waypoint/time states held by the table.
    """

    cell_ids: dict[Cell, int]
    first_bin: int
    delta: float
    table: np.ndarray
    row_units: dict[RowKey, int]
    row_samples: dict[RowKey, float]
    quantum: float
    build_s: float
    table_states: int
    edge_count: int
    distance_weight: float
    distance_error_m: float
    attachment_m: float
    boundary: np.ndarray
    _reference: float
    _coefficient_lo: float
    _coefficient_hi: float
    _price_errors: tuple[float, float, float]
    _air_error: float
    _air_ceiling: float
    _twice_abs_prices: float
    _resource_penalty: float | None = None

    @classmethod
    def build(
        cls, graph: ZetaGraph, prices: Mapping[RowKey, float], deadline: float | None = None,
        *, air_weight: float | None = None, buffered_periods: bool = True,
        distance_penalty: float | None = None,
    ) -> FuturePriceBound | None:
        """Prepare a complete bound or decline safely within finite preparation limits.

        Parameters
        ------------
        - graph (ZetaGraph): finite candidate connections, clocks and destination choices.
        - prices (Mapping[RowKey, float]): frozen prices; only positive cell rows are sampled.
        - deadline (float | None): absolute monotonic cutoff, including compilation time.
        - air_weight (float | None): canonical air-cost weight; its exact ratio to nominal speed
          is used by floor(). None retains the price-only table.
        - buffered_periods (bool): include guaranteed buffered subpart unions for owned periods;
          False retains point-only prices. Excessive union-table sizes also use point prices.
        - distance_penalty (float | None): nonnegative price per metre for a resource-relaxed
          table, including terminal endpoints. Such tables require a budget correction and
          must not be passed to floor() directly.

        Return
        --------
        - output (FuturePriceBound | None): a completed reusable lookup, or None for unsupported
          buffers, insufficient time, empty prices or a table exceeding preparation limits.
        """
        started = time.perf_counter()
        cfg = graph.cfg
        weight = 0.0 if air_weight is None else air_weight
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("air_weight must be finite and nonnegative")
        coefficient = Fraction(weight) / Fraction(cfg.nominal_speed_mps)
        if distance_penalty is not None:
            if not math.isfinite(distance_penalty) or distance_penalty < 0.0:
                raise ValueError("distance_penalty must be finite and nonnegative")
            coefficient = Fraction(distance_penalty)
        if cfg.time_buffer_s < 0.0 or not math.isfinite(cfg.dt_s) or cfg.dt_s <= 0.0:
            return None
        delta = min(1.0, cfg.dt_s / 4.0)
        if delta <= 0.0:
            return None
        quantized = _price_units(prices)
        if quantized is None:
            return None
        row_units, quantum = quantized
        by_space: dict[tuple[int, int, int], dict[int, int]] = defaultdict(dict)
        sample_stamps = {}
        for row, units in row_units.items():
            if row.kind != "cell":
                continue
            tau = (row.step + 0.5) * cfg.dt_s
            if not row.step * cfg.dt_s < tau < (row.step + 1) * cfg.dt_s:
                continue
            sample_stamps[row.step] = tau
            by_space[row[1:4]][row.step] = units
        if not sample_stamps:
            return None
        if distance_penalty is None:
            static = _geometry(graph, deadline)
        else:
            from .zeta_completion_geometry import geometry
            static = geometry(graph, prices, deadline)
        if static is None:
            return None
        cells, outgoing = static
        edges = [edge for group in outgoing for edge in group]
        if not edges:
            return None
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        if distance_penalty is None:
            geometry_error = _distance_error(graph, cells, outgoing)
        else:
            from .zeta_completion_geometry import distance_error
            geometry_error = distance_error(graph, cells, outgoing)
        if geometry_error is None:
            return None
        distance_error, edge_limit, length_ceiling = geometry_error
        attachments = _customer_attachments(graph)
        if attachments is None:
            return None
        distance_weight = float(coefficient) if distance_penalty is not None or (
            graph.fg.origin_terminal is None and graph.fg.dest_terminal is None
            and graph.reference > 1e-9
        ) else 0.0
        if distance_weight:
            exact_costs = [coefficient * sum(map(Fraction, edge.lengths)) for edge in edges]
            try:
                extra_budget = _fraction_up(max(exact_costs) * (len(cells) + 1))
            except OverflowError:
                return None
            quantized = _price_units(prices, extra_budget)
            if quantized is None:
                return None
            row_units, quantum = quantized
            # The time/spatial layout is unchanged when the common unit scale grows.
            by_space = defaultdict(dict)
            for row, units in row_units.items():
                if row.kind == "cell" and row.step in sample_stamps:
                    by_space[row[1:4]][row.step] = units
        if distance_weight:
            exact_quantum = Fraction(quantum)
            distance_costs = [int(value // exact_quantum) for value in exact_costs]
        else:
            distance_costs = [0] * len(edges)
        total_units = sum(row_units.values()) + (len(cells) + 1) * max(distance_costs)
        if total_units >= 2**50 or not math.isfinite(total_units * quantum):
            return None
        positive_sum = sum((Fraction(float(value)) for value in prices.values() if value > 0.0), Fraction(0))
        negative_sum = sum((-Fraction(float(value)) for value in prices.values() if value < 0.0), Fraction(0))
        if (edge_limit + 2) * 2.0**-52 >= 0.5:
            return None
        try:
            twice_positive = math.nextafter(2 * _fraction_up(positive_sum), math.inf)
            twice_negative = math.nextafter(2 * _fraction_up(negative_sum), math.inf)
            twice_absolute = math.nextafter(2 * _fraction_up(positive_sum + negative_sum), math.inf)
            positive_error = math.nextafter((2 * edge_limit + 2) * math.ulp(twice_positive), math.inf)
            price_errors = positive_error, math.ulp(twice_negative), math.ulp(twice_absolute)
            detour_ceiling = math.nextafter(length_ceiling + abs(graph.reference), math.inf)
            seconds_ceiling = math.nextafter(detour_ceiling / cfg.nominal_speed_mps, math.inf)
            air_ceiling = math.nextafter(weight * seconds_ceiling, math.inf)
            air_error = _fraction_up(coefficient * Fraction(math.ulp(detour_ceiling))
                                     + Fraction(weight) * Fraction(math.ulp(seconds_ceiling))
                                     + Fraction(math.ulp(air_ceiling)))
        except (OverflowError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (*price_errors, air_error, air_ceiling,
                                                      twice_absolute)):
            return None
        fraction = graph.options.speed_fraction if graph.options.arrival_snap_s > 0.0 else 0.0
        nominal = [math.fsum(edge.durations) for edge in edges]
        earliest = (
            graph.fg.base_step + graph.fg.takeoff_steps[0]
            + min(steps for _, _, steps in graph.origins)
        ) * cfg.dt_s
        horizon = max(sample_stamps.values())
        if max(abs(earliest / delta), abs(horizon / delta)) >= 2**48:
            return None
        first_bin = _bin_index(earliest, delta)
        last_bin = _bin_index(horizon, delta) + 1
        bins = last_bin - first_bin + 1
        parts_count = sum(len(edge.durations) for edge in edges)
        part_limit = 16_000_000 if distance_penalty is None else 24_000_000
        if (bins <= 0 or bins * len(cells) > 2_000_000
                or parts_count * max(len(sample_stamps), bins) > part_limit):
            return None
        steps = sorted(sample_stamps)
        sample_ids = {step: index for index, step in enumerate(steps)}
        sample_times = np.asarray([sample_stamps[step] for step in steps])
        period_starts = np.asarray([step * cfg.dt_s for step in steps])
        period_ends = np.asarray([(step + 1) * cfg.dt_s for step in steps])
        part_costs = np.zeros((parts_count, len(steps)), dtype=np.int64)
        mask_offsets = [0]
        if buffered_periods and max(len(edge.durations) for edge in edges) <= 8:
            for edge in edges:
                mask_offsets.append(mask_offsets[-1] + (1 << len(edge.durations)))
        if len(mask_offsets) == 1 or mask_offsets[-1] * len(steps) > 8_000_000:
            mask_offsets = []
        membership = np.zeros((mask_offsets[-1] if mask_offsets else 0, len(steps)), dtype=np.int64)
        part_offsets, durations, prefix_fractions = [0], [], []
        part_index = 0
        for edge_index, edge in enumerate(edges):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            for index, covered in enumerate(edge.covered):
                durations.append(edge.durations[index])
                prefix_fractions.append(math.fsum(edge.durations[:index + 1]) / math.fsum(edge.durations))
                for space in covered:
                    for step, units in by_space.get(space, {}).items():
                        part_costs[part_index, sample_ids[step]] += units
                part_index += 1
            part_offsets.append(part_index)
            if mask_offsets:
                spaces: dict[tuple[int, int, int], int] = {}
                for index, covered in enumerate(edge.covered):
                    for space in covered:
                        if space in by_space:
                            spaces[space] = spaces.get(space, 0) | (1 << index)
                for space, mask in spaces.items():
                    for step, units in by_space[space].items():
                        membership[mask_offsets[edge_index] + mask, sample_ids[step]] += units
        part_offsets = np.asarray(part_offsets, dtype=np.int64)
        mask_offsets = np.asarray(mask_offsets, dtype=np.int64)
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        union_costs = _coverage_unions(membership, mask_offsets) if len(mask_offsets) else membership
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        first_offsets, last_offsets, earliest_starts, latest_ends, latest_starts, earliest_ends, valid = _time_enclosures(
            first_bin, bins, delta, graph.options.arrival_snap_s, fraction, part_offsets,
            np.asarray(durations), np.asarray(nominal), np.asarray(prefix_fractions),
        )
        if not valid or int(np.sum(last_offsets - first_offsets + 1)) > 200_000_000:
            return None
        edge_offsets = np.asarray([0, *np.cumsum([len(group) for group in outgoing])], dtype=np.int64)
        target_ids = np.asarray([edge.target for edge in edges], dtype=np.int64)
        cell_ids = {cell: index for index, cell in enumerate(cells)}
        goals = np.zeros(len(cells), dtype=np.bool_)
        for _, goal in graph.destinations:
            goals[cell_ids[goal]] = True
        boundary = _distance_boundary(outgoing, distance_costs, np.flatnonzero(goals))
        table = np.zeros((bins, len(cells)), dtype=np.int64)
        arrays = (
            edge_offsets, target_ids, part_offsets, first_offsets, last_offsets,
            earliest_starts, latest_ends, sample_times, part_costs, goals,
            np.asarray(distance_costs, dtype=np.int64), boundary,
            latest_starts, earliest_ends, period_starts, period_ends, cfg.time_buffer_s,
            mask_offsets, union_costs,
        )
        for first in range(bins - 1, -1, -32):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            _backward_chunk(table, first_bin, delta, *arrays, total_units,
                            first, max(-1, first - 32))
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        row_samples = {row: sample_stamps[row.step] for row in row_units
                       if row.kind == "cell" and row.step in sample_stamps}
        discount_units = {row: row_units[row] for row in row_samples}
        return cls(cell_ids, first_bin, delta, table, discount_units, row_samples, quantum,
                   time.perf_counter() - started, int(table.size), len(edges),
                   distance_weight, distance_error, attachments, boundary, graph.reference,
                   _fraction_down(coefficient), _fraction_up(coefficient), price_errors,
                   air_error, air_ceiling, twice_absolute, distance_penalty)

    def units(self, node: Cell, stamp: float) -> int:
        """Return raw continuation units, retaining static distance after the price horizon."""
        index = _bin_index(stamp, self.delta) - self.first_bin
        identifier = self.cell_ids.get(node)
        if identifier is None or index < 0:
            return 0
        if index >= self.table.shape[0]:
            return int(self.boundary[identifier])
        return int(self.table[index, identifier])

    def value(self, node: Cell, stamp: float, paid: Iterable[RowKey]) -> float:
        """Discount live paid rows from the sampled continuation bound and round downward.

        Parameters
        ------------
        - node (Cell): exact current candidate waypoint.
        - stamp (float): exact current physical arrival timestamp.
        - paid (Iterable[RowKey]): unique live previously priced rows.

        Return
        --------
        - output (float): lower bound on newly paid positive prices, plus weighted graph-part
          length when enabled. Use floor() to apply canonical-distance and cost-rounding guards.
        """
        identifier = self.cell_ids.get(node)
        distance_units = 0 if identifier is None else int(self.boundary[identifier])
        units = max(distance_units, self.units(node, stamp) - sum(
            self.row_units[row] for row in paid
            if self.row_samples.get(row, -math.inf) > stamp
        ))
        return max(0.0, math.nextafter(units * self.quantum, -math.inf)) if units else 0.0

    def floor(
        self, cell: Cell, stamp: float, length: float, live_paid_units: int,
        positive_paid: float, ground: float, negative: float,
    ) -> float:
        """Lower-bound canonical completion cost with outward arithmetic and roundoff guards.

        Parameters
        ------------
        - cell (Cell): current exact waypoint.
        - stamp (float): exact prefix arrival time.
        - length (float): sequentially accumulated graph-part length.
        - live_paid_units (int): quantized charges of unique live sampled rows whose stored
          row_samples value is strictly greater than stamp; all other paid rows are excluded.
        - positive_paid (float): prefix's accumulated positive-price bound.
        - ground (float): canonical ground cost for the chosen departure.
        - negative (float): unchanged all-negative-row-price relaxation.

        Return
        --------
        - output (float): conservative minimization floor, including canonical cost and
          price-summation rounding. Terminal endpoints use the price-only fallback.
        """
        if self._resource_penalty is not None:
            raise ValueError("resource tables require the remaining-distance correction")

        def down(value: float) -> float:
            """Round one arithmetic operation downward."""
            return math.nextafter(value, -math.inf)

        identifier = self.cell_ids.get(cell)
        distance_units = 0 if identifier is None else int(self.boundary[identifier])
        units = max(distance_units, self.units(cell, stamp) - live_paid_units)
        sampled = units * self.quantum
        if not all(math.isfinite(value) for value in (sampled, length, positive_paid, ground, negative)):
            return -math.inf
        positive_error, negative_error, union_error = self._price_errors
        paid = down(down(down(positive_paid - positive_error) + negative) - negative_error)
        if not self.distance_weight:
            adjustment = down(down(paid + max(0.0, sampled)) - union_error)
            result = down(ground + adjustment)
            return result if math.isfinite(result) else -math.inf
        distance = down(down(down(length + self.attachment_m) - self._reference)
                        - self.distance_error_m)
        coefficient = self._coefficient_lo if distance >= 0.0 else self._coefficient_hi
        joint = max(0.0, down(down(down(coefficient * distance) + sampled) - self._air_error))
        cost_ceiling = math.nextafter(ground + self._air_ceiling, math.inf)
        final_scale = math.nextafter(cost_ceiling + self._twice_abs_prices, math.inf)
        if not all(math.isfinite(value) for value in (paid, distance, joint, cost_ceiling, final_scale)):
            return -math.inf
        result = down(down(ground + paid) + joint)
        result = down(result - math.ulp(cost_ceiling))
        result = down(result - union_error)
        result = down(result - math.ulp(final_scale))
        return result if math.isfinite(result) else -math.inf
