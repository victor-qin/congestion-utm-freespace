"""Guarded remaining-distance bounds for the twelve-direction hex lattice."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import math
import time
from typing import TYPE_CHECKING

from .network import RowKey
from .objective import CostModel
from .zeta_history_guard import HistoryGuard, _EPS, _up

if TYPE_CHECKING:
    from .zeta import Cell, ZetaGraph


def _diagonal_extra() -> float:
    """Return a binary rational proved no larger than sqrt(3)-1."""
    value = math.sqrt(3.0) - 1.0
    while (Fraction.from_float(value) + 1) ** 2 > 3:
        value = math.nextafter(value, -math.inf)
    return value


_DIAGONAL_EXTRA = _diagonal_extra()


def lattice_distance_lower(q: int, r: int, pitch: float) -> float:
    """Round the ideal twelve-direction grid metric downward at every operation.

    Parameters
    ------------
    - q (int): axial displacement, with absolute value at most 2**27.
    - r (int): axial displacement, with absolute value at most 2**27.
    - pitch (float): positive physical center pitch represented as a binary float.

    Return
    --------
    - output (float): nonnegative lower bound on the ideal, unfiltered grid distance.
    """
    if (abs(q) > 1 << 27 or abs(r) > 1 << 27
            or not math.isfinite(pitch) or pitch <= 0.0):
        raise ValueError("lattice metric requires bounded axial integers and finite positive pitch")
    # The bound keeps every integer and half-integer conversion exact. Runtime
    # eligibility is stricter: HistoryGuard bounds each endpoint by 2**26.
    a, b, _c = sorted((abs(q), abs(r), abs(q + r)))
    if b == 0:
        return 0.0
    diagonal = math.nextafter(_DIAGONAL_EXTRA * a, -math.inf) if a else 0.0
    total = math.nextafter(float(b) + diagonal, -math.inf)
    return max(0.0, math.nextafter(pitch * total, -math.inf))


class LatticeDistanceBound:
    """Cache ideal geometric distances and apply one whole-route arithmetic guard.

    Attributes
    ------------
    - graph (ZetaGraph): immutable finite physical graph defining all continuations.
    - correction_m (float): outward allowance for full-route geometry and cost arithmetic.
    - cache (dict): price-independent Euclidean and ideal-lattice distances per cell pair.
    """

    def __init__(self, graph: ZetaGraph, correction_m: float) -> None:
        """Bind a scalar call-specific guard to reusable price-independent distances."""
        self.graph = graph
        self.correction_m = correction_m
        cache = getattr(graph, "_lattice_distance_cache", None)
        if cache is None:
            cache = {}
            graph._lattice_distance_cache = cache
        self.cache: dict[tuple[Cell, Cell], tuple[float, float]] = cache

    @classmethod
    def build(
        cls, graph: ZetaGraph, model: CostModel, prices: Mapping[RowKey, float],
        history_guard: HistoryGuard | None = None,
        *, deadline: float | None = None,
    ) -> LatticeDistanceBound | None:
        """Enable only proved step sets and stable endpoint/arithmetic contexts.

        Parameters
        ------------
        - graph (ZetaGraph): candidate graph; wall, corridor and forward filters may delete edges.
        - model (CostModel): actual positive physical objective weights.
        - prices (Mapping[RowKey, float]): signed current prices, used only for a scalar error guard.
        - history_guard (HistoryGuard | None): compatible guard already built for this pricing call.
        - deadline (float | None): absolute monotonic deadline; expiry declines this optional bound.

        Return
        --------
        - output (LatticeDistanceBound | None): guarded lookup, or unchanged Euclidean fallback.
        """
        options = graph.options
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        supported = options.connection_mode == "hex12" or (
            options.connection_mode == "full"
            and options.connection_radius_m <= 2.0 * graph.pitch + 1e-8
        )
        if (not supported or not graph._offsets
                or any(q*q + q*r + r*r not in {1, 3, 4} for q, r in graph._offsets)):
            return None
        guard = history_guard or HistoryGuard(graph, model, prices)
        if not math.isfinite(guard.length_error) or model.air_weight <= 0.0:
            return None
        # This stronger spacing guard also bounds ideal-grid conversion error;
        # HistoryGuard alone covers differences between physical reductions.
        if 4096 * _EPS * guard._scale >= Fraction.from_float(
                min(graph.pitch, float(graph.cfg.corridor_segment_len_m))):
            return None
        margins = []
        for oi, _start, _steps in graph.origins:
            for di, _goal in graph.destinations:
                if deadline is not None and time.perf_counter() >= deadline:
                    return None
                margins.append(guard.margin(oi, di))
        if not margins or any(margin is None for margin in margins):
            return None
        # Reserve the whole-route envelope before using the metric for either
        # physical budget pruning or canonical cost. Do not charge it per edge.
        correction = _up(
            4 * Fraction.from_float(guard.length_error)
            + 4 * Fraction.from_float(max(margins))
            * Fraction.from_float(float(graph.cfg.nominal_speed_mps))
            / Fraction.from_float(float(model.air_weight))
        )
        if not math.isfinite(correction):
            return None
        return cls(graph, correction)

    def remaining(self, source: Cell, goal: Cell) -> float:
        """Strengthen the old Euclidean bound without charging row prices again.

        Parameters
        ------------
        - source (Cell): current or candidate next waypoint.
        - goal (Cell): destination waypoint for the fixed endpoint-lane context.

        Return
        --------
        - output (float): existing Euclidean bound or a stronger guarded lattice bound.
        """
        key = source, goal
        values = self.cache.get(key)
        if values is None:
            euclidean = math.dist(self.graph.point(source), self.graph.point(goal))
            ideal = lattice_distance_lower(goal[0] - source[0], goal[1] - source[1],
                                           self.graph.pitch)
            values = euclidean, ideal
            self.cache[key] = values
        euclidean, ideal = values
        guarded = math.nextafter(ideal - self.correction_m, -math.inf)
        return max(euclidean, guarded)
