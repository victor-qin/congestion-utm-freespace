"""Operation-level integer enclosures of the GPU search's reference numerical bounds.

This module does not establish membership or delete search states. Callers must
provide the original FP64 preparation and independently enclosed prefix unions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np

from .gpu_integer import IntegerInterval, IntegerScale
from .gpu_bounds import _down as down, _up as up

if TYPE_CHECKING:
    from .gpu_bounds import CompletionBounds


@dataclass(frozen=True, slots=True)
class BoundOperands:
    """Carry one state's original-reference operands on a single immutable grid.

    Attributes
    ------------
    - benefit (IntegerInterval): original flight benefit.
    - assignment (IntegerInterval): original assignment dual.
    - negative (IntegerInterval): original global downward negative credit.
    - allowance (IntegerInterval): original gamma-32 canonical arithmetic allowance.
    - cost (IntegerInterval): original clamped canonical geometry/ground cost floor.
    - affine (IntegerInterval): original downward ground + offset + hop*slope prefix.
    - positive (IntegerInterval): full positive origin/prefix union enclosure.
    - visits (IntegerInterval): visit-only positive prefix enclosure.
    - future (IntegerInterval): original relaxed suffix cost; invalid means unavailable.
    - arrival (IntegerInterval): unpaid mandatory arrival price floor, or unavailable.
    - arrival_affine (IntegerInterval): arrival price/air residual, or unavailable.
    """

    benefit: IntegerInterval
    assignment: IntegerInterval
    negative: IntegerInterval
    allowance: IntegerInterval
    cost: IntegerInterval
    affine: IntegerInterval
    positive: IntegerInterval
    visits: IntegerInterval
    future: IntegerInterval
    arrival: IntegerInterval
    arrival_affine: IntegerInterval

    def packed(self) -> np.ndarray:
        """Pack the explicit shader order without changing scale or operand precision."""
        return np.stack([getattr(self, name).packed() for name in self.__dataclass_fields__])


def prepare_reference_operands(
    scale: IntegerScale, bounds: CompletionBounds, variant: int, hops: int, remaining: int,
    positive: IntegerInterval, visits: IntegerInterval, future: float,
    residual: tuple[float, float] | None, benefit: float, negative: float,
) -> BoundOperands:
    """Encode the reference geometry and canonical rounding allowance for a prototype state.

    Parameters
    ------------
    - scale (IntegerScale): immutable grid shared with both prefix intervals.
    - bounds (CompletionBounds): original prepared FP64 geometry, prices and objective.
    - variant (int): current departure/lane root index.
    - hops (int): completed moves from the root.
    - remaining (int): relaxed remaining hops from the current cell.
    - positive (IntegerInterval): independently enclosed positive origin/prefix union.
    - visits (IntegerInterval): independently enclosed visit-only prefix.
    - future (float): original relaxed suffix cost, or infinity when unavailable.
    - residual (tuple[float, float] | None): original unpaid arrival minima, or unavailable.
    - benefit (float): original master benefit.
    - negative (float): original global downward negative credit.

    Return
    --------
    - output (BoundOperands): encoded reference operands. This diagnostic helper does not
      establish membership or remove CPU per-state preparation from a production search.
    """
    pi = bounds.problem.pi_f
    affine = down(float(bounds.variants.ground_delay_s[variant]) + float(bounds.air_offset[variant]))
    affine = down(affine + down(hops * bounds.air_per_hop))
    if not 0 <= hops <= bounds.topology.air_hop_limit:
        affine = math.inf
    magnitude = up(up(abs(benefit) + abs(pi)) + bounds._cost_scale)
    magnitude = up(magnitude + bounds._absolute_prices)
    unit = 2.0**-53
    gamma = up((32.0 * unit) / down(1.0 - 32.0 * unit))
    allowance = up(gamma * magnitude)
    q, d = residual if residual is not None else (math.inf, math.inf)
    return BoundOperands(
        scale.encode(benefit), scale.encode(pi), scale.encode(negative), scale.encode(allowance),
        scale.encode(bounds.cost_lower(variant, hops + remaining)), scale.encode(affine),
        positive, visits, scale.encode(future), scale.encode(q), scale.encode(d),
    )


def reference_upper(scale: IntegerScale, operands: BoundOperands) -> IntegerInterval:
    """Enclose the reference's independent upper bounds with ordered interval operations.

    Parameters
    ------------
    - scale (IntegerScale): common immutable grid of every operand.
    - operands (BoundOperands): original FP64 prepared values and enclosed exact-membership sums.

    Return
    --------
    - output (IntegerInterval): upper endpoint at least the available reference minimum;
      unavailable/overflowed bounds are omitted, and no finite bound returns invalid.
    """
    o = operands

    def finish(lower: IntegerInterval) -> IntegerInterval:
        """Preserve both upward subtractions and the separately prepared canonical allowance."""
        upper = scale.subtract(o.benefit, lower, direction=1)
        upper = scale.subtract(upper, o.assignment, direction=1)
        return scale.add(upper, o.allowance, direction=1)

    def affine(prefix: IntegerInterval, suffix: IntegerInterval) -> IntegerInterval:
        """Combine only disjoint charges with the original prepared affine cost prefix."""
        lower = scale.add(o.affine, prefix, direction=-1)
        lower = scale.add(lower, suffix, direction=-1)
        lower = scale.add(lower, o.negative, direction=-1)
        return finish(lower)

    positive = scale.add(IntegerInterval.checked(0, 0), o.positive, direction=-1)
    dual = scale.add(positive, o.negative, direction=-1)
    basic = scale.subtract(o.benefit, scale.rounded(o.cost, -1), direction=1)
    basic = scale.subtract(basic, dual, direction=1)
    basic = scale.subtract(basic, o.assignment, direction=1)
    clamped = scale.add(o.cost, o.positive, direction=-1)
    clamped = scale.add(clamped, o.arrival, direction=-1)
    clamped = scale.add(clamped, o.negative, direction=-1)
    candidates = (basic, affine(o.visits, o.future), finish(clamped),
                  affine(o.positive, o.arrival_affine))
    valid = [bound for bound in candidates if bound.valid]
    return min(valid, key=lambda bound: bound.upper) if valid else IntegerInterval()
