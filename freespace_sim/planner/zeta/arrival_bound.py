"""Admissible completion costs from unavoidable destination reservations."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from ..route_domain import RowKey
from ..reservation_windows import terminal_claim_steps
from .completion_geometry import distance_error as _distance_error, geometry as _geometry
from .bound import (
    _bin_index, _fraction_down, _fraction_up, _price_units,
)

if TYPE_CHECKING:
    from ..route_cost import CostModel
    from .search import Cell, ZetaGraph


def _down(value: float) -> float:
    """Round a single floating operation outward toward a lower bound."""
    return math.nextafter(value, -math.inf)


def _up(value: float) -> float:
    """Round a single floating operation outward toward an upper bound."""
    return math.nextafter(value, math.inf)


@njit(cache=True, nogil=True)
def _bin_index_kernel(stamp: float, delta: float) -> int:
    """Mirror bound._bin_index: multiplied boundaries classify exact contact identically."""
    index = int(np.floor(stamp / delta))
    while (index + 1) * delta <= stamp:
        index += 1
    while index * delta > stamp:
        index -= 1
    return index


@njit(cache=True, nogil=True)
def _ulp(value: float) -> float:
    """math.ulp for finite doubles, including subnormals and the largest finite value."""
    magnitude = abs(value)
    if magnitude == 0.0:
        return 5e-324
    _mantissa, exponent = math.frexp(magnitude)
    return math.ldexp(1.0, max(exponent - 53, -1074))


@njit(cache=True, nogil=True)
def _arrival_floor_kernel(
    scalars: np.ndarray, totals: np.ndarray, starts: np.ndarray, row_begin: np.ndarray,
    row_end: np.ndarray, row_weight: np.ndarray, paid: np.ndarray, stamp: float, length: float,
    positive_paid: float, ground: float, negative: float, remaining: float, couple: bool,
) -> float:
    """Non-terminal ArrivalBound.floor over packed table-id arrays.

    Every operation and its directed rounding match ArrivalBound.floor exactly; the parity
    test compares both bit for bit. No fast-math: each step rounds once, then moves outward.

    Parameters
    ------------
    - scalars (np.ndarray): packed static and price constants, see ArrivalBound.bind.
    - totals, starts (np.ndarray): per-bin destination units and bin start clocks.
    - row_begin, row_end, row_weight (np.ndarray): each table row's bin slice and units;
      weight 0 marks rows with no destination charge.
    - paid (np.ndarray): unique live paid table-row ids.
    - stamp, length, positive_paid, ground, negative (float): the label's state.
    - remaining (float): static remaining distance for this waypoint and goal.
    - couple (bool): whether destination distance coupling applies.

    Return
    --------
    - output (float): the same floor as the interpreted method, or -inf.
    """
    (_earliest, latest_static, minimum_ratio, maximum_ratio, time_error, length_error,
     delta, first_bin_f, max_length, attachment, reference, distance_error, coefficient,
     air_error, quantum, positive_error, negative_error, union_error, air_ceiling,
     twice_abs) = (scalars[0], scalars[1], scalars[2], scalars[3], scalars[4], scalars[5],
                   scalars[6], scalars[7], scalars[8], scalars[9], scalars[10], scalars[11],
                   scalars[12], scalars[13], scalars[14], scalars[15], scalars[16],
                   scalars[17], scalars[18], scalars[19])
    first_bin = int(first_bin_f)
    # Landing window, per-bin charges and distance coupling: context/figures/zeta_arrival_bound.png
    available = max(0.0, np.nextafter(np.nextafter(max_length + 1e-7 - length, np.inf)
                                      + length_error, np.inf))
    earliest = max(stamp, np.nextafter(np.nextafter(stamp + np.nextafter(
        remaining * minimum_ratio, -np.inf), -np.inf) - time_error, -np.inf))
    latest = min(latest_static, np.nextafter(np.nextafter(stamp + np.nextafter(
        available * maximum_ratio, np.inf), np.inf) + time_error, np.inf))
    if earliest > latest:
        return -np.inf
    first = max(0, _bin_index_kernel(earliest, delta) - first_bin)
    stop = min(totals.shape[0], _bin_index_kernel(latest, delta) - first_bin + 1)
    if first >= stop:
        return -np.inf
    units = totals[first:stop].copy()
    for index in range(paid.shape[0]):
        row = paid[index]
        weight = row_weight[row]
        if weight != 0:
            begin = max(first, row_begin[row])
            end = min(stop, row_end[row])
            for position in range(begin, end):
                units[position - first] -= weight
    joint = np.inf
    for position in range(stop - first):
        charge = float(units[position]) * quantum
        if couple:
            arrival = max(starts[first + position], earliest)
            elapsed = np.nextafter(arrival - stamp, -np.inf)
            elapsed = np.nextafter(elapsed - time_error, -np.inf)
            future = max(remaining, np.nextafter(elapsed / maximum_ratio, -np.inf))
            distance = np.nextafter(length + future, -np.inf)
            distance = np.nextafter(distance + attachment, -np.inf)
            distance = np.nextafter(distance - reference, -np.inf)
            distance = np.nextafter(distance - distance_error, -np.inf)
            air = np.nextafter(coefficient * max(0.0, distance), -np.inf)
            air = max(0.0, np.nextafter(air - air_error, -np.inf))
            value = np.nextafter(charge + air, -np.inf)
        else:
            value = charge
        joint = min(joint, value)
    prefix = np.nextafter(np.nextafter(np.nextafter(positive_paid - positive_error, -np.inf)
                                       + negative, -np.inf) - negative_error, -np.inf)
    cost_ceiling = np.nextafter(ground + air_ceiling, np.inf)
    final_scale = np.nextafter(cost_ceiling + twice_abs, np.inf)
    if not (np.isfinite(joint) and np.isfinite(prefix) and np.isfinite(cost_ceiling)
            and np.isfinite(final_scale)):
        return -np.inf
    result = np.nextafter(np.nextafter(ground + prefix, -np.inf) + joint, -np.inf)
    result = np.nextafter(np.nextafter(result - _ulp(cost_ceiling), -np.inf) - union_error,
                          -np.inf)
    result = np.nextafter(result - _ulp(final_scale), -np.inf)
    return result if np.isfinite(result) else -np.inf


@dataclass(slots=True)
class _ArrivalGeometry:
    """Price-independent destination windows and whole-route numerical allowances."""

    first_bin: int
    delta: float
    starts: np.ndarray
    firsts: np.ndarray
    stops: np.ndarray
    footprints: frozenset[tuple[object, ...]]
    remaining: dict[tuple[Cell, Cell], float]
    minimum_ratio: float
    maximum_ratio: float
    time_error: float
    length_error: float
    distance_error: float
    length_ceiling: float
    attachment: float
    edge_limit: int
    earliest: float
    latest: float


def _attachments(graph: ZetaGraph) -> float:
    """Lower-bound only customer attachments; terminal fold additions stay free.

    Parameters
    ------------
    - graph (ZetaGraph): original endpoints and their permitted waypoint choices.

    Return
    --------
    - output (float): downward sum of customer-to-waypoint distances.
    """
    total = Fraction(0)
    for point, terminal, cells in (
        (graph.fg.request.origin, graph.fg.origin_terminal,
         (cell for _, cell, _ in graph.origins)),
        (graph.fg.request.dest, graph.fg.dest_terminal,
         (cell for _, cell in graph.destinations)),
    ):
        if terminal is not None:
            continue
        distances = []
        for cell in cells:
            target = graph.point(cell)
            dx = Fraction(float(point[0])) - Fraction(target[0])
            dy = Fraction(float(point[1])) - Fraction(target[1])
            squared = dx * dx + dy * dy
            distance = math.hypot(float(point[0]) - target[0], float(point[1]) - target[1])
            while Fraction(distance) ** 2 > squared:
                distance = math.nextafter(distance, -math.inf)
            distances.append(distance)
        total += Fraction(min(distances))
    return _fraction_down(total)


def _arrival_geometry(graph: ZetaGraph, deadline: float | None) -> _ArrivalGeometry | None:
    """Cache complete arrival intervals containing every accepted continuation.

    Parameters
    ------------
    - graph (ZetaGraph): immutable finite graph, endpoint geometry and timing settings.
    - deadline (float | None): monotonic cutoff; interrupted builds are not cached.

    Return
    --------
    - output (_ArrivalGeometry | None): static intervals and arithmetic bounds, or None
      when finite conservative bounds or the preparation budget cannot be established.
    """
    cached = getattr(graph, "_arrival_bound_geometry", None)
    if cached is not None:
        return cached
    cfg, fg = graph.cfg, graph.fg
    if (not math.isfinite(cfg.dt_s) or cfg.dt_s <= 0.0
            or not math.isfinite(cfg.time_buffer_s) or cfg.time_buffer_s < 0.0):
        return None
    geometry = _geometry(graph, None, deadline)
    if geometry is None:
        return None
    cells, outgoing = geometry
    edges = [edge for group in outgoing for edge in group]
    if not edges:
        return None
    errors = _distance_error(graph, cells, outgoing)
    if errors is None:
        return None
    distance_error, edge_limit, length_ceiling = errors
    part_limit = edge_limit * max(len(edge.lengths) for edge in edges)
    budget = graph.max_length + 1e-7
    length_error = _up(part_limit * math.ulp(budget))
    ratios = [sum(map(Fraction, edge.durations)) / sum(map(Fraction, edge.lengths))
              for edge in edges]
    fraction = graph.options.speed_fraction if graph.options.arrival_snap_s > 0.0 else 0.0
    slow = 1.0 - fraction
    minimum_ratio = _fraction_down(min(ratios) / Fraction(1.0 + fraction))
    maximum_ratio = _fraction_up(max(ratios) / Fraction(slow))
    roots = [(dep + fg.takeoff_steps[0] + steps) * cfg.dt_s
             for dep in (fg.base_step, fg.latest_departure_step)
             for _, _, steps in graph.origins]
    earliest = min(roots)
    travel = _up(_up(budget + length_error) * maximum_ratio)
    scale = _up(2.0 * max(1.0, max(map(abs, roots)) + travel))
    if not math.isfinite(scale):
        return None
    # Covers fsum/division, snap acceptance tolerances, timestamp subtraction and
    # nominal fallback additions. The scale check closes the clock-magnitude bound.
    time_error = _up(_up(64.0 * (part_limit + edge_limit) * math.ulp(scale)) / slow)
    if not math.isfinite(time_error) or time_error >= scale / 4.0:
        return None
    latest = _up(_up(max(roots) + travel) + time_error)
    delta = min(1.0, cfg.dt_s / 4.0)
    if delta <= 0.0 or max(abs(earliest / delta), abs(latest / delta)) >= 2**48:
        return None
    first_bin = _bin_index(earliest, delta)
    bins = _bin_index(latest, delta) - first_bin + 1
    if bins <= 0 or bins > 100_000:
        return None
    starts = np.asarray([(first_bin + index) * delta for index in range(bins)])
    firsts, stops = np.empty(bins, dtype=np.int64), np.empty(bins, dtype=np.int64)
    for index, lower in enumerate(starts):
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        upper = (first_bin + index + 1) * delta
        low_volume, high_volume = graph.endpoint(False, float(lower)), graph.endpoint(False, upper)
        low_rows = terminal_claim_steps(low_volume.t_start, low_volume.t_end, cfg)
        high_rows = terminal_claim_steps(high_volume.t_start, high_volume.t_end, cfg)
        firsts[index], stops[index] = max(low_rows.start, high_rows.start), min(
            low_rows.stop, high_rows.stop,
        )
    footprints = frozenset(row[:-1] for row in graph.rows(graph.endpoint(False, earliest)))
    if not footprints:
        return None
    remaining = {
        (cell, goal): max(0.0, _down(math.dist(graph.point(cell), graph.point(goal))
                                   - distance_error))
        for cell in cells for _, goal in graph.destinations
    }
    result = _ArrivalGeometry(
        first_bin, delta, starts, firsts, stops, footprints, remaining,
        minimum_ratio, maximum_ratio, time_error, length_error, distance_error,
        length_ceiling, _attachments(graph), edge_limit, earliest, latest,
    )
    if deadline is not None and time.perf_counter() >= deadline:
        return None
    graph._arrival_bound_geometry = result
    return result


@dataclass(slots=True)
class ArrivalBound:
    """Joint lower bound on destination prices and the distance needed to arrive.

    Attributes
    ------------
    - build_s (float): geometry and frozen-price preparation wall time.
    - table_states (int): number of conservative arrival intervals.
    - row_units (dict[RowKey, int]): downward quantized positive destination prices.
    """

    build_s: float
    table_states: int
    row_units: dict[RowKey, int]
    _geometry: _ArrivalGeometry
    _graph: ZetaGraph
    _totals: np.ndarray
    _slices: dict[RowKey, tuple[int, int, int]]
    _quantum: float
    _coefficient: float
    _price_errors: tuple[float, float, float]
    _air_error: float
    _air_ceiling: float
    _twice_abs_prices: float
    _packed: tuple | None = None

    @classmethod
    def build(
        cls, graph: ZetaGraph, prices: Mapping[RowKey, float], model: CostModel,
        deadline: float | None = None,
    ) -> ArrivalBound | None:
        """Refresh destination prices over cached static arrival geometry.

        Parameters
        ------------
        - graph (ZetaGraph): exact candidate geometry, clocks and endpoint authority.
        - prices (Mapping[RowKey, float]): frozen finite signed row prices.
        - model (CostModel): canonical objective weights for this solve.
        - deadline (float | None): monotonic cutoff covering all preparation work.

        Return
        --------
        - output (ArrivalBound | None): completed bound, or None for unsupported arithmetic,
          absent destination prices, an expired deadline or excessive static table size.
        """
        started = time.perf_counter()
        if (deadline is not None and started >= deadline
                or not math.isfinite(model.air_weight) or model.air_weight < 0.0
                or any(not math.isfinite(price) for price in prices.values())):
            return None
        endpoint = graph.endpoint(False, graph.fg.base_step * graph.cfg.dt_s)
        footprints = frozenset(row[:-1] for row in graph.rows(endpoint))
        endpoint_prices = {row: price for row, price in prices.items()
                           if row[:-1] in footprints and price > 0.0}
        quantized = _price_units(endpoint_prices)
        if quantized is None:
            return None
        static = _arrival_geometry(graph, deadline)
        if static is None:
            return None
        weights, quantum = quantized
        weights = {row: units for row, units in weights.items()
                   if row[:-1] in static.footprints}
        if not weights:
            return None
        totals = np.zeros(len(static.starts), dtype=np.int64)
        slices = {}
        for row, units in weights.items():
            first = int(np.searchsorted(static.stops, row.step, side="right"))
            stop = int(np.searchsorted(static.firsts, row.step, side="right"))
            if first < stop:
                totals[first:stop] += units
                slices[row] = first, stop, units
        coefficient = Fraction(model.air_weight) / Fraction(graph.cfg.nominal_speed_mps)
        positive = sum((Fraction(float(p)) for p in prices.values() if p > 0.0), Fraction(0))
        negative = sum((-Fraction(float(p)) for p in prices.values() if p < 0.0), Fraction(0))
        if (static.edge_limit + 2) * 2.0**-52 >= 0.5:
            return None
        try:
            twice_positive = _up(2.0 * _fraction_up(positive))
            twice_negative = _up(2.0 * _fraction_up(negative))
            twice_absolute = _up(2.0 * _fraction_up(positive + negative))
            price_errors = (
                _up((2 * static.edge_limit + 2) * math.ulp(twice_positive)),
                math.ulp(twice_negative), math.ulp(twice_absolute),
            )
            detour_ceiling = _up(static.length_ceiling + abs(graph.reference))
            seconds_ceiling = _up(detour_ceiling / graph.cfg.nominal_speed_mps)
            air_ceiling = _up(model.air_weight * seconds_ceiling)
            air_error = _fraction_up(
                coefficient * Fraction(math.ulp(detour_ceiling))
                + Fraction(model.air_weight) * Fraction(math.ulp(seconds_ceiling))
                + Fraction(math.ulp(air_ceiling)),
            )
        except (OverflowError, ValueError):
            return None
        if (not all(math.isfinite(value) for value in
                    (*price_errors, air_error, air_ceiling, twice_absolute))
                or deadline is not None and time.perf_counter() >= deadline):
            return None
        return cls(time.perf_counter() - started, len(totals), weights, static, graph, totals, slices,
                   quantum, _fraction_down(coefficient), price_errors, air_error, air_ceiling,
                   twice_absolute)

    def bind(self, rows: tuple[RowKey, ...]) -> None:
        """Pack static constants and each table row's charge slice for floor_ids.

        Parameters
        ------------
        - rows (tuple[RowKey, ...]): the compact search's row table; ids index this tuple.

        Return
        --------
        - output (None): stores arrays used only with this exact row tuple.
        """
        static = self._geometry
        begin = np.zeros(len(rows), dtype=np.int64)
        end = np.zeros(len(rows), dtype=np.int64)
        weight = np.zeros(len(rows), dtype=np.int64)
        for index, row in enumerate(rows):
            interval = self._slices.get(row)
            if interval is not None:
                begin[index], end[index], weight[index] = interval
        positive_error, negative_error, union_error = self._price_errors
        scalars = np.asarray([
            static.earliest, static.latest, static.minimum_ratio, static.maximum_ratio,
            static.time_error, static.length_error, static.delta, float(static.first_bin),
            self._graph.max_length, static.attachment, self._graph.reference,
            static.distance_error, self._coefficient, self._air_error, self._quantum,
            positive_error, negative_error, union_error, self._air_ceiling,
            self._twice_abs_prices,
        ], dtype=np.float64)
        couple_terminal = self._graph.fg.dest_terminal is None and self._graph.reference > 1e-9
        self._packed = rows, scalars, begin, end, weight, couple_terminal

    def floor_ids(
        self, cell: Cell, goal: Cell, stamp: float, length: float,
        positive_paid: float, ground: float, negative: float, fold_m: float | None,
        paid: np.ndarray, terminal: bool = False,
    ) -> float:
        """Same value as floor() for unique table-row ids, compiled for non-terminal labels.

        Parameters
        ------------
        - cell, goal, stamp, length, positive_paid, ground, negative, fold_m: as floor().
        - paid (np.ndarray): unique live paid ids into the tuple given to bind().
        - terminal (bool): terminal labels keep the interpreted endpoint-geometry path.

        Return
        --------
        - output (float): identical to floor() on the same state.
        """
        rows, scalars, begin, end, weight, couple_terminal = self._packed
        static = self._geometry
        if terminal:
            return self.floor(cell, goal, stamp, length, positive_paid, ground, negative,
                              fold_m, (rows[int(row)] for row in paid), terminal)
        if (not all(math.isfinite(value) for value in
                    (stamp, length, positive_paid, ground, negative))
                or stamp < static.earliest or stamp > static.latest):
            return -math.inf
        remaining = static.remaining.get((cell, goal))
        if remaining is None:
            return -math.inf
        return float(_arrival_floor_kernel(
            scalars, self._totals, static.starts, begin, end, weight, paid, stamp, length,
            positive_paid, ground, negative, remaining, fold_m is not None and couple_terminal))

    def floor(
        self, cell: Cell, goal: Cell, stamp: float, length: float,
        positive_paid: float, ground: float, negative: float, fold_m: float | None,
        paid: Iterable[RowKey], terminal: bool = False,
    ) -> float:
        """Lower-bound every canonical completion, discounting rows already paid once.

        Parameters
        ------------
        - cell (Cell): current exact candidate waypoint.
        - goal (Cell): this label's selected destination waypoint.
        - stamp (float): exact prefix arrival timestamp, including nominal fallback.
        - length (float): sequentially accumulated graph-part distance.
        - positive_paid (float): accumulated positive prefix charge, including retired rows.
        - ground (float): canonical departure cost.
        - negative (float): sum of every negative frozen price, granting all possible credit.
        - fold_m (float | None): endpoint-retention proof from graph.fold_length().
        - paid (Iterable[RowKey]): unique live paid rows after optional history expiry.
        - terminal (bool): whether this label must end at its exact current timestamp.

        Return
        --------
        - output (float): full minimization floor, or negative infinity if the state or
          numerical guards are unsupported. Destination-terminal distance coupling is relaxed.
        """
        static, graph = self._geometry, self._graph
        if (not all(math.isfinite(value) for value in
                    (stamp, length, positive_paid, ground, negative))
                or stamp < static.earliest or stamp > static.latest
                or (cell, goal) not in static.remaining):
            return -math.inf
        live = frozenset(paid)
        remaining = 0.0 if terminal else static.remaining[cell, goal]
        if terminal:
            rows = graph.rows(graph.endpoint(False, stamp))
            units = sum(self.row_units.get(row, 0) for row in rows - live)
            charges = np.asarray([units * self._quantum])
            future = np.asarray([0.0])
        else:
            # Landing window, per-bin charges and distance coupling: context/figures/zeta_arrival_bound.png
            available = max(0.0, _up(_up(graph.max_length + 1e-7 - length)
                                      + static.length_error))
            earliest = max(stamp, _down(_down(stamp + _down(
                remaining * static.minimum_ratio)) - static.time_error))
            latest = min(static.latest, _up(_up(stamp + _up(
                available * static.maximum_ratio)) + static.time_error))
            if earliest > latest:
                return -math.inf
            first = max(0, _bin_index(earliest, static.delta) - static.first_bin)
            stop = min(len(self._totals), _bin_index(latest, static.delta) - static.first_bin + 1)
            if first >= stop:
                return -math.inf
            units = self._totals[first:stop].copy()
            for row in live:
                interval = self._slices.get(row)
                if interval is not None:
                    begin, end, weight = interval
                    begin, end = max(first, begin), min(stop, end)
                    if begin < end:
                        units[begin - first:end - first] -= weight
            charges = units.astype(np.float64) * self._quantum
            arrivals = np.maximum(static.starts[first:stop], earliest)
            elapsed = np.nextafter(arrivals - stamp, -np.inf)
            elapsed = np.nextafter(elapsed - static.time_error, -np.inf)
            future = np.maximum(remaining, np.nextafter(elapsed / static.maximum_ratio, -np.inf))
        # A retained origin is the canonical centreline's exact first point. A terminal
        # destination can lose its reconstructed final point to inward rounding; omit
        # distance coupling there rather than assume that fold_m proves its retention.
        if fold_m is not None and graph.fg.dest_terminal is None and graph.reference > 1e-9:
            distance = np.nextafter(length + future, -np.inf)
            distance = np.nextafter(distance + static.attachment, -np.inf)
            distance = np.nextafter(distance - graph.reference, -np.inf)
            distance = np.nextafter(distance - static.distance_error, -np.inf)
            air = np.nextafter(self._coefficient * np.maximum(0.0, distance), -np.inf)
            air = np.maximum(0.0, np.nextafter(air - self._air_error, -np.inf))
            joint = float(np.min(np.nextafter(charges + air, -np.inf)))
        else:
            joint = float(np.min(charges))
        positive_error, negative_error, union_error = self._price_errors
        prefix = _down(_down(_down(positive_paid - positive_error) + negative) - negative_error)
        cost_ceiling = _up(ground + self._air_ceiling)
        final_scale = _up(cost_ceiling + self._twice_abs_prices)
        if not all(math.isfinite(value) for value in (joint, prefix, cost_ceiling, final_scale)):
            return -math.inf
        result = _down(_down(ground + prefix) + joint)
        result = _down(_down(result - math.ulp(cost_ceiling)) - union_error)
        result = _down(result - math.ulp(final_scale))
        return result if math.isfinite(result) else -math.inf
