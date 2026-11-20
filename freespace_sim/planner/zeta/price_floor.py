"""Conservative reservation-price arithmetic shared by every Zeta traversal."""

from __future__ import annotations

from collections.abc import Iterable
import math


def prepare_price_floor(prices: Iterable[float]) -> tuple[float, float]:
    """Bound subtotal and canonical-union rounding for one frozen price vector.

    Parameters
    ------------
    - prices (Iterable[float]): finite signed reservation prices, one per unique row.

    Return
    --------
    - output (tuple[float, float]): rounded all-negative credit and an outward error
      allowance. Infinite error disables ordinary price pruning conservatively.
    """
    values = tuple(prices)
    try:
        negative = math.fsum(value for value in values if value < 0.0)
    except OverflowError:
        return -math.inf, math.inf
    try:
        absolute = math.fsum(abs(value) for value in values)
    except OverflowError:
        return negative, math.inf
    if absolute == 0.0:
        return negative, 0.0
    count = sum(value > 0.0 for value in values)
    # Each positive row is charged once, even after history expiry. There are at
    # most count nonempty batches, each with one fsum rounding and one prefix add.
    # Two outward steps cover fsum's possible double rounding. The count guard
    # keeps every accumulated subtotal below twice the absolute exact row sum.
    absolute = math.nextafter(math.nextafter(absolute, math.inf), math.inf)
    scale = math.nextafter(2.0 * absolute, math.inf)
    if not math.isfinite(scale) or count >= 2**50:
        return negative, math.inf
    # Two additional ulps cover negative-only fsum and canonical signed fsum;
    # two spare ulps also cover subnormal/double-rounding boundary cases.
    error = math.nextafter((2 * count + 4) * math.ulp(scale), math.inf)
    return negative, error


def price_floor(
    cost: float, positive: float, negative: float, error: float, tolerance: float,
) -> float:
    """Combine a physical cost floor with signed credit using downward arithmetic.

    Parameters
    ------------
    - cost (float): existing physical completion-cost relaxation.
    - positive (float): sequentially accumulated positive prefix charge.
    - negative (float): all-negative credit from prepare_price_floor.
    - error (float): allowance for prefix, negative and canonical-union rounding.
    - tolerance (float): existing physical-cost comparison allowance.

    Return
    --------
    - output (float): conservative priced-cost floor, or negative infinity when
      arithmetic is unavailable. A native successor scan uses this same function.
    """
    if (not math.isfinite(cost) or not math.isfinite(positive)
            or not math.isfinite(negative) or not math.isfinite(error)):
        return -math.inf
    # Cancel prices before adding physical cost; otherwise cost itself can be
    # rounded away by an unrelated large positive subtotal.
    credit = math.nextafter(positive + negative, -math.inf)
    credit = math.nextafter(credit - error, -math.inf)
    value = math.nextafter(cost + credit, -math.inf)
    return math.nextafter(value - tolerance, -math.inf)
