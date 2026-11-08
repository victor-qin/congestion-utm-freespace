"""Conservative canonical geometry and mandatory-origin bounds for GPU pricing."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .. import hexgrid as hg
from ...config import SimConfig
from ...volumes import enroute_reference_m, exit_radius
from . import pricing
from .gpu_prices import summary_for
from .dp_prepare import PreparedTopology, PreparedVariants, PricingPreparation
from .network import RowKey
from .objective import CostModel, cost_model
from .pricing_platform import PricingProblem


def _down(value: float) -> float:
    """Round one floating operation outward toward a smaller bound."""
    return math.nextafter(value, -math.inf)


def _up(value: float) -> float:
    """Round one floating operation outward toward a larger bound."""
    return math.nextafter(value, math.inf)


def _distance_interval(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Enclose exact Euclidean distance between two represented coordinate pairs.

    Parameters
    ------------
    - a (tuple[float, float]): first finite coordinate pair.
    - b (tuple[float, float]): second finite coordinate pair.

    Return
    --------
    - output (tuple[float, float]): outward lower and upper distance bounds.
    """
    lower = upper = 0.0
    for first, second in zip(a, b, strict=True):
        lo, hi = _down(second - first), _up(second - first)
        magnitude_lo = 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))
        magnitude_hi = max(abs(lo), abs(hi))
        lower = max(0.0, _down(lower + max(0.0, _down(magnitude_lo * magnitude_lo))))
        upper = _up(upper + _up(magnitude_hi * magnitude_hi))
    return max(0.0, _down(math.sqrt(lower))), _up(math.sqrt(upper))


@dataclass(frozen=True, slots=True)
class _GeometryBounds:
    """One graph's immutable geometry bounds, independent of prices and root selection."""

    cfg: SimConfig
    topology: PreparedTopology
    model: CostModel
    length_offset: np.ndarray
    air_offset: np.ndarray
    edge_min: float = 0.0
    reference: float = 0.0
    air_per_hop: float = 0.0
    air_scale: float = math.inf


class CompletionBounds:
    """Bound canonical route cost without running the CPU pricing search.

    The geometry calculation assumes ordinary IEEE binary64 arithmetic without
    overflow. Unsupported magnitudes or uncertain terminal folds retain the
    ground-only bound. ``air_offset`` and ``air_per_hop`` are GPU proposal
    coefficients; ``cost_lower`` applies the canonical operation order when
    verifying a deletion on CPU.
    """

    def __init__(
        self,
        problem: PricingProblem,
        topology: PreparedTopology,
        variants: PreparedVariants,
        view: pricing.DualView,
        *,
        preparation: PricingPreparation | None = None,
    ) -> None:
        """Prepare geometry bounds and lazy mandatory-origin claim caches.

        Parameters
        ------------
        - problem (PricingProblem): flight, geometry and objective configuration.
        - topology (PreparedTopology): all arcs a GPU completion may traverse.
        - variants (PreparedVariants): departures and lanes, including weighted ground cost.
        - view (pricing.DualView): original binary64 capacity prices.
        - preparation (PricingPreparation | None): this call's existing root unions;
          omitted for callers that have not prepared them already.

        Return
        --------
        - output (None): initializes conservative affine proposals and CPU verification data.
        """
        self.problem, self.topology, self.variants, self.view = problem, topology, variants, view
        self.model = cost_model(problem.cfg, problem.params)
        self.air_per_hop = 0.0
        self.air_offset = np.full(variants.n_variants, -math.inf, np.float64)
        self._length_offset = np.full(variants.n_variants, -math.inf, np.float64)
        self._edge_min = 0.0
        self._reference = 0.0
        self._cost_scale = math.inf
        self._absolute_prices = summary_for(view).absolute
        self._claims: dict[int, frozenset[RowKey]] = {}
        if preparation is not None:
            preparation.check(
                problem.graph, problem.cfg, view, pricing._benefit(problem.params),
                problem.pi_f, self.model, problem.forbidden_rows,
            )
            # Keep only immutable unions, releasing the preparation's other price caches.
            for variant in range(variants.n_variants):
                departure = int(variants.departure_step[variant])
                lane = int(variants.lane_idx[variant])
                cell = int(variants.cell[variant])
                self._claims[variant] = preparation.root_claims(
                    departure, None if lane < 0 else lane,
                    preparation.origin_claims(departure),
                    (int(topology.cell_q[cell]), int(topology.cell_r[cell])),
                    int(variants.start_step[variant]),
                )
        self._positive: dict[int, float] = {}
        self._prepare_geometry()

    def _prepare_geometry(self) -> None:
        """Reuse fixed geometry and gather bounds for this call's surviving roots.

        Parameters
        ------------
        - self (CompletionBounds): current graph, objective and root arrays.

        Return
        --------
        - output (None): refreshes root offsets and their ground-dependent roundoff scale.
        """
        p, t, v = self.problem, self.topology, self.variants
        if not t.ok or not v.ok or self.model.air_weight == 0:
            return
        cache = p.graph._search_cache
        with cache.lock:
            geometry = cache.gpu_geometry
        if (
            geometry is None
            or geometry.topology is not t
            or geometry.cfg != p.cfg
            or geometry.model != self.model
        ):
            geometry = self._build_geometry()
            # Pure builds may race; each caller keeps its own correctly keyed result.
            with cache.lock:
                cache.gpu_geometry = geometry
        self._edge_min, self._reference = geometry.edge_min, geometry.reference
        self.air_per_hop = geometry.air_per_hop
        self._length_offset = geometry.length_offset[v.cell]
        self.air_offset = geometry.air_offset[v.cell]
        ground_upper = max(0.0, float(v.ground_delay_s.max(initial=0.0)))
        self._cost_scale = _up(ground_upper + geometry.air_scale)

    def _build_geometry(self) -> _GeometryBounds:
        """Bound interpolation, endpoint folding, norm and sum roundoff explicitly.

        Parameters
        ------------
        - self (CompletionBounds): graph and objective data to prepare.

        Return
        --------
        - output (_GeometryBounds): cell-indexed air bounds, or ground-only fallback data.

        With unit roundoff u and coordinate/radius magnitude C>=1, each
        resampled coordinate ``a + (k/n)*(b-a)`` differs from its ideal position
        by less than 10*u*C (subtraction, division, multiplication and addition).
        Triangle inequalities therefore lose at most 32*u*C per original hop.
        The 64*u*C*(N+2) allowance also covers both rounded endpoint projections.
        Norm evaluation and summation use gamma_(K+8) times a total-length upper
        bound, where K counts all resampled segments plus two endpoint legs.
        N is the maximum allowed hop count, so the allowance covers every
        completion without depending on the path or reduction order.
        """
        p, t = self.problem, self.topology
        cfg, graph = p.cfg, p.graph
        empty = np.full(t.n_cells, -math.inf, np.float64)
        empty.flags.writeable = False
        fallback = _GeometryBounds(cfg, t, self.model, empty, empty)
        radius = hg.circumradius(cfg)
        cells = [(int(q), int(r)) for q, r in zip(t.cell_q, t.cell_r, strict=True)]
        points = [tuple(map(float, hg.hex_center(*cell, radius))) for cell in cells]
        origin = tuple(map(float, graph.request.origin[:2]))
        destination = tuple(map(float, graph.request.dest[:2]))
        radii = [
            exit_radius(term, cfg)
            for term in (graph.origin_terminal, graph.dest_terminal)
            if term is not None
        ]
        scale = max(
            1.0, *(abs(x) for point in (*points, origin, destination) for x in point), *radii
        )
        reference = enroute_reference_m(
            graph.request.origin,
            graph.request.dest,
            graph.origin_terminal,
            graph.dest_terminal,
            cfg,
        )
        if (
            not math.isfinite(scale)
            or scale > 1e100
            or not math.isfinite(reference)
            or reference <= 1e-9
            or not math.isfinite(cfg.corridor_segment_len_m)
            or cfg.corridor_segment_len_m <= 0
            or not math.isfinite(cfg.nominal_speed_mps)
            or cfg.nominal_speed_mps <= 0
        ):
            return fallback
        edge_min, edge_max, max_sub = math.inf, 0.0, 1
        for source, point in enumerate(points):
            for target in t.arc_target[int(t.arc_start[source]) : int(t.arc_start[source + 1])]:
                other = points[int(target)]
                lo, hi = _distance_interval(point, other)
                edge_min, edge_max = min(edge_min, lo), max(edge_max, hi)
                dx, dy = other[0] - point[0], other[1] - point[1]
                length = math.sqrt(dx * dx + dy * dy)
                subdivision_ratio = length / cfg.corridor_segment_len_m
                if not math.isfinite(subdivision_ratio) or subdivision_ratio > 2**30:
                    return fallback
                max_sub = max(max_sub, math.ceil(subdivision_ratio))
        if not math.isfinite(edge_min) or edge_min <= 0.0:
            return fallback
        n = max(1, int(t.air_hop_limit))
        # The canonical builder subdivides raw lattice arcs; enroute_flown_m
        # adds the two fold/snap legs afterward without subdividing them.
        count = n * max_sub + 2
        unit = 2.0**-53
        scaled_unit = _up(unit * scale)
        gamma_n = _up((count + 8) * unit)
        if gamma_n >= 0.01:
            return fallback
        gamma = _up(gamma_n / _down(1.0 - gamma_n))
        length_upper = _up(_up(n * edge_max) + _up(8.0 * scale))
        length_upper = _up(length_upper + _up(_up(32.0 * scaled_unit) * count))
        error = _up(_up(_up(64.0 * scaled_unit) * (n + 2)) + _up(gamma * length_upper))
        margin = _up(64.0 * scaled_unit)
        if not math.isfinite(error):
            return fallback
        flown_upper = _up(length_upper + _up(gamma * length_upper))
        air_scale = _up(
            _up(_up(flown_upper + _up(reference)) / cfg.nominal_speed_mps) * self.model.air_weight
        )
        fallback = replace(fallback, air_scale=air_scale)

        def fold_lower(
            point: tuple[float, float], endpoint: tuple[float, float], terminal: object | None
        ) -> float | None:
            """Require enough radial clearance to retain rounded boundary points."""
            lo, _hi = _distance_interval(point, endpoint)
            if terminal is None:
                return lo
            exit_r = exit_radius(terminal, cfg)
            if lo <= max(_up(exit_r + margin), _up(1e-9)):
                return None
            return max(0.0, _down(lo - exit_r))

        # Disconnected terminal lanes are absent from the prepared search domain.
        destination_legs = [
            fold_lower(points[int(cell)], destination, graph.dest_terminal)
            for cell in np.flatnonzero(t.dest_mask)
        ]
        if not destination_legs or any(leg is None for leg in destination_legs):
            return fallback
        dest_lower = min(destination_legs)
        reference_upper = _up(reference)
        air_per_hop = max(
            0.0, _down(_down(edge_min / cfg.nominal_speed_mps) * self.model.air_weight)
        )
        length_offset, air_offset = empty.copy(), empty.copy()
        # Every departure at the same origin cell has the same fold geometry.
        # Include every topology origin, even when this call filtered its roots.
        for cell in np.unique(t.origin_cell):
            origin_lower = fold_lower(points[int(cell)], origin, graph.origin_terminal)
            if origin_lower is None:
                continue
            offset = _down(_down(origin_lower + dest_lower) - error)
            length_offset[cell] = offset
            air_offset[cell] = _down(
                _down(_down(offset - reference_upper) / cfg.nominal_speed_mps)
                * self.model.air_weight
            )
        length_offset.flags.writeable = air_offset.flags.writeable = False
        return replace(
            fallback,
            length_offset=length_offset,
            air_offset=air_offset,
            edge_min=edge_min,
            reference=reference_upper,
            air_per_hop=air_per_hop,
        )

    def cost_lower(self, variant: int, total_hops: int) -> float:
        """Lower-bound weighted canonical ground and air cost for this hop count.

        Parameters
        ------------
        - variant (int): departure/lane root index.
        - total_hops (int): lower bound on the completion's total number of hops.

        Return
        --------
        - output (float): downward binary64 cost bound, including geometry roundoff.
        """
        ground = max(0.0, _down(float(self.variants.ground_delay_s[variant])))
        offset = float(self._length_offset[variant])
        if not math.isfinite(offset):
            return ground
        length = _down(_down(total_hops * self._edge_min) + offset)
        detour = max(0.0, _down(length - self._reference))
        air = max(
            0.0, _down(_down(detour / self.problem.cfg.nominal_speed_mps) * self.model.air_weight)
        )
        return max(0.0, _down(ground + air))

    def origin_claims(self, variant: int) -> frozenset[RowKey]:
        """Return mandatory origin endpoint and first lattice-visit claims.

        Parameters
        ------------
        - variant (int): departure/lane root index.

        Return
        --------
        - output (frozenset[RowKey]): exact union, including terminal rows omitted from paid arrays.
        """
        claims = self._claims.get(variant)
        if claims is None:
            t, v, p = self.topology, self.variants, self.problem
            cell = int(v.cell[variant])
            claims = pricing._endpoint_claims(
                p.graph,
                p.cfg,
                origin=True,
                step=int(v.departure_step[variant]),
                timing_steps=0,
            ) | pricing._visit_claims(
                (int(t.cell_q[cell]), int(t.cell_r[cell])),
                0,
                int(v.start_step[variant]),
                self.view.offsets,
            )
            self._claims[variant] = claims
        return claims

    def _affine_upper(
        self,
        variant: int,
        hops: int,
        positive_prefix: float,
        suffix: float,
        benefit: float,
        pi: float,
        negative_lower: float,
    ) -> float:
        """Bound reduced cost using affine air cost and disjoint price/air lower terms.

        Parameters
        ------------
        - variant (int): departure/lane root index.
        - hops (int): already completed moves, excluding the root visit.
        - positive_prefix (float): downward sum of already claimed positive prices.
        - suffix (float): downward residual air/price bound, with prices disjoint from prefix.
        - benefit (float): original per-flight master benefit.
        - pi (float): original assignment dual.
        - negative_lower (float): downward sum of all negative input row prices.

        Return
        --------
        - output (float): certified upper bound, or infinity when the affine bound is unavailable.

        In real arithmetic, ``ground + b + hops*a + prefix + suffix + negative``
        is a cost-plus-price lower bound. Clamping canonical air detour at zero
        can only raise it. Coefficients and all bound operations are rounded
        downward, independently of suffix construction. Canonical cost still
        rounds subtraction, division, multiplication and addition; canonical
        dual summation and the three left-associated RC subtractions also round.
        Their combined absolute error is bounded by gamma_32 times
        ``|benefit| + |pi| + cost_scale + sum(abs(all prices))``. Eight operations
        suffice for cost, two roundoffs cover fsum, and three cover reduced cost;
        32 also covers propagation between these stages. Geometry reconstruction
        error is already covered by the affine coefficients. ``cost_scale``
        bounds ground plus weighted (maximum flown length + reference)/speed.
        """
        offset = float(self.air_offset[variant])
        operands = (
            offset,
            positive_prefix,
            suffix,
            benefit,
            pi,
            negative_lower,
            self.air_per_hop,
            self._cost_scale,
            self._absolute_prices,
        )
        if (
            not all(math.isfinite(value) for value in operands)
            or hops < 0
            or hops > self.topology.air_hop_limit
        ):
            return math.inf
        lower = _down(float(self.variants.ground_delay_s[variant]) + offset)
        lower = _down(lower + _down(hops * self.air_per_hop))
        lower = _down(lower + positive_prefix)
        lower = _down(lower + suffix)
        lower = _down(lower + negative_lower)
        return self._roundoff_upper(lower, benefit, pi)

    def _roundoff_upper(self, lower: float, benefit: float, pi: float) -> float:
        """Retain canonical cost, dual-sum and reduced-cost rounding allowances.

        Parameters
        ------------
        - lower (float): outward cost-plus-price lower bound.
        - benefit (float): original benefit.
        - pi (float): assignment dual.

        Return
        --------
        - output (float): upper bound including the canonical gamma_32 error allowance.
        """
        scale = _up(_up(abs(benefit) + abs(pi)) + self._cost_scale)
        scale = _up(scale + self._absolute_prices)
        unit = 2.0**-53
        gamma = _up((32.0 * unit) / _down(1.0 - 32.0 * unit))
        allowance = _up(gamma * scale)
        upper = _up(_up(benefit - lower) - pi)
        return _up(upper + allowance)

    def future_upper(
        self, variant: int, hops: int, positive_prefix: float, suffix: float,
        benefit: float, pi: float, negative_lower: float,
    ) -> float:
        """Bound a relaxed future table using visit-only prefix prices.

        Parameters
        ------------
        - variant (int): current root index.
        - hops (int): completed moves.
        - positive_prefix (float): downward positive VISIT sum, excluding separate origin rows.
        - suffix (float): relaxed future air/visit costs and endpoint rows disjoint from visits.
        - benefit (float): original benefit.
        - pi (float): assignment dual.
        - negative_lower (float): global negative-price credit.

        Return
        --------
        - output (float): canonical upper bound. Visit windows are disjoint under the revisit
          rule; origin claims are omitted because the relaxed suffix may overlap them.
        """
        return self._affine_upper(
            variant, hops, positive_prefix, suffix, benefit, pi, negative_lower,
        )

    def endpoint_upper(
        self, variant: int, hops: int, remaining: int, positive_union: float,
        residual: tuple[float, float], benefit: float, pi: float, negative_lower: float,
    ) -> float:
        """Combine independent clamped and affine arrival-only bounds without double charging.

        Parameters
        ------------
        - variant (int): current root index.
        - hops (int): completed moves.
        - remaining (int): relaxed remaining hops.
        - positive_union (float): verified positive sum of origin UNION all prefix visits.
        - residual (tuple[float, float]): Q minimum and affine-delay residual for unpaid rows.
        - benefit (float): original benefit.
        - pi (float): assignment dual.
        - negative_lower (float): global credit for every negative price, included once.

        Return
        --------
        - output (float): minimum of independent upper bounds, never a sum with future bounds.
          Future visit charges are omitted; they may pay the reserved endpoint rows later.
        """
        q, d = residual
        lower = _down(self.cost_lower(variant, hops + remaining) + positive_union)
        lower = _down(_down(lower + q) + negative_lower)
        clamped = self._roundoff_upper(lower, benefit, pi)
        affine = self._affine_upper(
            variant, hops, positive_union, d, benefit, pi, negative_lower,
        )
        return min(clamped, affine)

    def root_positive(self, variant: int) -> float:
        """Downward-sum only positive prices in the mandatory origin union.

        Parameters
        ------------
        - variant (int): departure/lane root index.

        Return
        --------
        - output (float): nonnegative lower bound without double-counted origin claims.
        """
        value = self._positive.get(variant)
        if value is None:
            value = 0.0
            for row in self.origin_claims(variant):
                price = self.view.row_cost(row)
                if price > 0.0:
                    value = _down(value + price)
            value = max(0.0, value)
            self._positive[variant] = value
        return value
