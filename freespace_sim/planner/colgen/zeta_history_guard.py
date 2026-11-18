"""Conservative floating-point guard for comparing different route histories."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from fractions import Fraction
import math
from typing import TYPE_CHECKING

from ...volumes import exit_radius
from .objective import CostModel

if TYPE_CHECKING:
    from .zeta import ZetaGraph


_EPS = Fraction(1, 1 << 52)
_SCALE_LIMIT = float(1 << 100)


def _up(value: Fraction) -> float:
    """Round a nonnegative rational upper bound toward positive infinity."""
    try:
        result = float(value)
    except OverflowError:
        return math.inf
    if math.isfinite(result) and Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


class HistoryGuard:
    """Bound canonical-cost conversion error for guarded route-prefix comparisons.

    Comparisons still require identical waypoint, exact timestamp, departure and
    terminal context and first/last roles. Equal-state comparisons retain both
    accumulated lengths; shorter-prefix comparisons require both to be no greater.
    Strict comparisons pay the margin with an exact signed-price advantage. Guarded
    equal-charge merges instead widen the pricing certificate by this whole-route
    allowance. With nonnegative air weight it also covers componentwise-shorter
    accumulators: their exact-arithmetic completion cost cannot increase, and the
    two endpoints of any replacement chain still satisfy this conversion envelope.
    Unsafe arithmetic scales and path-dependent terminal folds retain the older
    comparison that also requires the complete segment-length sequence.

    Attributes
    ------------
    - graph (ZetaGraph): fixed physical route graph and simulator configuration.
    - model (CostModel): common physical route cost weights.
    - max_parts (int): conservative number of physical subsegments per route.
    - length_error (float): outward-rounded canonical length conversion allowance.
    """

    def __init__(
        self, graph: ZetaGraph, model: CostModel, prices: Mapping[Hashable, float],
    ) -> None:
        """Prepare a finite arithmetic envelope for one frozen pricing call.

        Parameters
        ------------
        - graph (ZetaGraph): finite waypoint graph and unchanged simulator configuration.
        - model (CostModel): physical delay weights used by canonical columns.
        - prices (Mapping[Hashable, float]): the complete frozen signed row-price vector.

        Return
        --------
        - output (None): caches geometry bounds and an initially empty context cache.
        """
        self.graph, self.model = graph, model
        self._cache: dict[tuple[int | None, int | None], float | None] = {}
        self._enabled = False
        self.max_parts = 0
        self.length_error = math.inf
        self._origins = {index: cell for index, cell, _steps in graph.origins}
        self._destinations = {index: cell for index, cell in graph.destinations}
        cfg, fg = graph.cfg, graph.fg
        cells = graph.cells | set(self._origins.values()) | set(self._destinations.values())
        # Exact integer/half-integer axial arithmetic is part of the spacing bound.
        if any(abs(coordinate) > (1 << 26) for cell in cells for coordinate in cell):
            return
        values = [1.0, graph.max_length, graph.pitch, cfg.corridor_segment_len_m,
                  graph.reference, *fg.request.origin[:2], *fg.request.dest[:2]]
        values.extend(coordinate for cell in cells for coordinate in graph.point(cell)[:2])
        for terminal in (fg.origin_terminal, fg.dest_terminal):
            if terminal is not None:
                values.append(exit_radius(terminal, cfg))
        values = [float(value) for value in values]
        weights = (float(model.air_weight), float(model.ground_weight))
        speed = float(cfg.nominal_speed_mps)
        if (not all(math.isfinite(value) and abs(value) <= _SCALE_LIMIT
                    for value in (*values, *weights, speed, cfg.dt_s))
                or speed < 1.0 / _SCALE_LIMIT or cfg.dt_s <= 0.0
                or graph.max_length < 0.0 or graph.reference < 0.0):
            return
        self._scale = Fraction.from_float(max(abs(value) for value in values))
        spacing = min(float(graph.pitch), float(cfg.corridor_segment_len_m))
        if spacing <= 0.0 or 256 * _EPS * self._scale >= Fraction.from_float(spacing):
            return
        self._budget = Fraction.from_float(graph.max_length + 1e-7)
        count = 8 * (Fraction.from_float(graph.max_length) + 1) / Fraction.from_float(spacing)
        self.max_parts = (count.numerator + count.denominator - 1) // count.denominator + 4
        if self.max_parts * _EPS >= Fraction(1, 1024):
            return
        # The terms cover two reductions, local norm/interpolation errors, and folds.
        n = self.max_parts
        self._length_error = _EPS * self._scale * (32 * n * n + 128 * n + 1024)
        self.length_error = _up(self._length_error)
        try:
            absolute_prices = math.fsum(abs(float(price)) for price in prices.values())
        except (OverflowError, ValueError):
            return
        if not math.isfinite(absolute_prices):
            return
        # fsum can have a second rounding on some platforms; two outward ulps cover it.
        absolute_prices = math.nextafter(math.nextafter(absolute_prices, math.inf), math.inf)
        if not math.isfinite(absolute_prices):
            return
        self._price_scale = Fraction.from_float(absolute_prices)
        self._speed = Fraction.from_float(speed)
        self._air_weight = Fraction.from_float(abs(weights[0]))
        departures = max(0, fg.latest_departure_step - fg.base_step)
        self._ground_scale = (Fraction.from_float(abs(weights[1])) * departures
                              * Fraction.from_float(float(cfg.dt_s)))
        self._enabled = True

    def margin(self, oi: int | None, di: int | None) -> float | None:
        """Return a sufficient strict price advantage for one endpoint context.

        Parameters
        ------------
        - oi (int | None): the selected origin lane index, or the customer endpoint.
        - di (int | None): the selected destination lane index, or the customer endpoint.

        Return
        --------
        - output (float | None): conservative objective margin, or no stronger comparison.
        """
        key = oi, di
        if key not in self._cache:
            self._cache[key] = self._margin(oi, di)
        return self._cache[key]

    def _margin(self, oi: int | None, di: int | None) -> float | None:
        """Check fold stability and the final budget before bounding canonical cost.

        Parameters
        ------------
        - oi (int | None): selected origin lane index.
        - di (int | None): selected destination lane index.

        Return
        --------
        - output (float | None): guarded cost difference or a conservative fallback.
        """
        if not self._enabled or oi not in self._origins or di not in self._destinations:
            return None
        graph, cfg, fg = self.graph, self.graph.cfg, self.graph.fg
        reference = Fraction.from_float(graph.reference)
        if graph.reference > 1e-9:
            for index, cells, point, terminal, destination in (
                (oi, self._origins, fg.request.origin, fg.origin_terminal, False),
                (di, self._destinations, fg.request.dest, fg.dest_terminal, True),
            ):
                if terminal is None:
                    continue
                x, y = graph.point(cells[index])[:2]
                dx, dy = x - float(point[0]), y - float(point[1])
                distance = math.sqrt(dx * dx + dy * dy)
                radius = float(exit_radius(terminal, cfg))
                if distance < max(radius, 1e-9):
                    return None
                # The builder's final a + (b - a) need not equal the stored waypoint.
                if (destination and Fraction.from_float(distance) - Fraction.from_float(max(radius, 1e-9))
                        <= 64 * _EPS * self._scale):
                    return None
            fold = graph.fold_length(oi, di)
            if fold is None or not math.isfinite(fold) or fold < 0.0:
                return None
            flown = self._budget + Fraction.from_float(fold) + self._length_error
            factor = float(cfg.max_detour_factor)
            if math.isnan(factor) or factor <= 0.0:
                return None
            # The upward quotient also bounds the simulator's rounded division.
            if _up(flown / reference) >= factor:
                return None
            physical = 2 * self._length_error * self._air_weight / self._speed
            air_scale = self._air_weight * (flown + reference) / self._speed
        else:
            # The simulator defines detour as exactly zero for these references.
            physical = air_scale = Fraction(0)
        arithmetic = 64 * _EPS * (self._ground_scale + air_scale + self._price_scale + 1)
        result = _up(physical + arithmetic)
        return result if math.isfinite(result) else None
