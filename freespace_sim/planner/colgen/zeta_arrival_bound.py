"""Admissible completion costs from unavoidable destination reservations."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from .network import RowKey
from .windows import terminal_claim_steps
from .zeta_completion_geometry import distance_error as _distance_error, geometry as _geometry
from .zeta_bound import (
    _bin_index, _fraction_down, _fraction_up, _price_units,
)

if TYPE_CHECKING:
    from .objective import CostModel
    from .zeta import Cell, ZetaGraph


def _down(value: float) -> float:
    """Round a single floating operation outward toward a lower bound."""
    return math.nextafter(value, -math.inf)


def _up(value: float) -> float:
    """Round a single floating operation outward toward an upper bound."""
    return math.nextafter(value, math.inf)


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
