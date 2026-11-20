"""Conservative reservation-price arithmetic shared by every Zeta traversal."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

from ..route_domain import RowKey


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


@dataclass(frozen=True, slots=True)
class PriceSnap:
    """Rows withheld from one pricing search and the accounting their removal needs.

    Withholding a positive price only lowers route costs, so search bounds stay valid
    for the full vector. Withholding a negative price raises costs; the total withheld
    credit is subtracted once from the final bound because a route claims each row once.

    Attributes
    ------------
    - rows (int): withheld rows, both positive and negative.
    - positive_mass (float): fsum of withheld positive prices below the tolerance.
    - negative_mass (float): fsum of every withheld negative price (nonpositive).
    - scale (float): fsum of all absolute full prices, sizing the rounding allowance.
    """

    rows: int = 0
    positive_mass: float = 0.0
    negative_mass: float = 0.0
    scale: float = 0.0


def snap_prices(
    prices: Mapping[RowKey, float], tolerance: float,
) -> tuple[dict[RowKey, float], PriceSnap]:
    """Withhold positive prices below a tolerance and every negative price.

    Parameters
    ------------
    - prices (Mapping[RowKey, float]): finite nonzero full row prices.
    - tolerance (float): smallest positive price a search still charges.

    Return
    --------
    - output (tuple[dict[RowKey, float], PriceSnap]): the search vector and its accounting.
    """
    kept: dict[RowKey, float] = {}
    positive: list[float] = []
    negative: list[float] = []
    for row, value in prices.items():
        if value >= tolerance:
            kept[row] = value
        elif value > 0.0:
            positive.append(value)
        elif value < 0.0:
            negative.append(value)
    try:
        scale = math.fsum(abs(value) for value in prices.values())
    except OverflowError:
        scale = math.inf
    return kept, PriceSnap(len(positive) + len(negative), math.fsum(positive),
                           math.fsum(negative), scale)


def snapped_lower_bound(lower: float, snap: PriceSnap | None) -> float:
    """Convert a withheld-price search bound into a bound under the full prices.

    Parameters
    ------------
    - lower (float): conservative lower bound on canonical withheld-price route costs.
    - snap (PriceSnap | None): withheld rows; None or positive-only leaves the bound exact,
      because correctly rounded sums are monotone in the added nonnegative prices.

    Return
    --------
    - output (float): conservative lower bound on canonical full-price route costs.
    """
    if snap is None or snap.negative_mass == 0.0 or not math.isfinite(lower):
        return lower
    value = math.nextafter(lower + snap.negative_mass, -math.inf)
    # Canonical costs round the route's union fsum and the physical addition once each;
    # eight ulps of the largest magnitude involved cover both sides of the comparison.
    error = 8.0 * math.ulp(abs(lower) + snap.scale + abs(snap.negative_mass))
    return math.nextafter(value - error, -math.inf)


def snap_certificate(
    complete: bool, reason: str, lower: float, best: float, upper: float,
    target: float | None, epsilon: float, tolerance: float, improving: bool,
) -> tuple[bool, str]:
    """Keep a completed withheld-price search complete only when full prices decide it.

    Parameters
    ------------
    - complete (bool): completion status in the withheld-price search.
    - reason (str): that search's stopping reason.
    - lower (float): full-price lower bound from snapped_lower_bound.
    - best (float): exact full-price cost of the reported route, or infinity.
    - upper (float): certified full-price reduced-cost upper bound.
    - target (float | None): reduced-cost target, or None when seeking the best route.
    - epsilon (float): master improvement tolerance.
    - tolerance (float): optimality comparison allowance.
    - improving (bool): whether the reported route improves under the full prices.

    Return
    --------
    - output (tuple[bool, str]): unchanged when the reported route is optimal within
      tolerance or the target is decided either way; otherwise ``snap_uncertainty``.
    """
    if not complete:
        return complete, reason
    if lower >= best - tolerance:
        return complete, reason
    if target is not None and (upper <= epsilon or improving):
        return complete, reason
    return False, "snap_uncertainty"
