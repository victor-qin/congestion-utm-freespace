"""Checked fixed-grid intervals for the GPU pruning prototype, not a search certificate."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


LIMIT = (1 << 63) - 1
WORD_MASK = (1 << 32) - 1


@dataclass(frozen=True, slots=True)
class IntegerInterval:
    """Enclose a value in grid units, or explicitly mark it unresolved.

    Attributes
    ------------
    - lower (int): inclusive signed lower endpoint in units of the owning scale.
    - upper (int): inclusive signed upper endpoint in units of the owning scale.
    - valid (bool): false means unbounded information; endpoints cannot authorize pruning.
    """

    lower: int = 0
    upper: int = 0
    valid: bool = False

    def __post_init__(self) -> None:
        """Prevent direct construction from bypassing the checked representation range."""
        if self.valid and not -LIMIT <= self.lower <= self.upper <= LIMIT:
            raise ValueError("valid integer interval endpoints must be ordered signed 63-bit values")

    @classmethod
    def checked(cls, lower: int, upper: int) -> IntegerInterval:
        """Reject out-of-range or reversed endpoints instead of saturating them."""
        return (cls(lower, upper, True) if -LIMIT <= lower <= upper <= LIMIT else cls())

    def packed(self) -> np.ndarray:
        """Return two low/high word pairs and an explicit validity word."""
        lo, hi = (self.lower, self.upper) if self.valid else (0, 0)
        return np.asarray((lo & WORD_MASK, (lo >> 32) & WORD_MASK,
                           hi & WORD_MASK, (hi >> 32) & WORD_MASK, int(self.valid)), np.uint32)

    @classmethod
    def unpacked(cls, words: np.ndarray) -> IntegerInterval:
        """Validate a device interval without interpreting invalid endpoints as finite."""
        if len(words) != 5 or int(words[4]) != 1:
            return cls()
        endpoints = []
        for index in (0, 2):
            value = int(words[index]) | (int(words[index + 1]) << 32)
            endpoints.append(value - (1 << 64) if value & (1 << 63) else value)
        return cls.checked(*endpoints)


@dataclass(frozen=True, slots=True)
class IntegerScale:
    """Define immutable power-of-two grid units for one numerical payload.

    Attributes
    ------------
    - exponent (int): grid spacing is exactly 2**exponent, never a rounded reciprocal.
    """

    exponent: int

    def __post_init__(self) -> None:
        """Restrict grids to binary64's subnormal spacing through its largest exponent."""
        if not -1074 <= self.exponent <= 1023:
            raise ValueError("integer grid exponent must be in [-1074, 1023]")

    @property
    def quantum(self) -> Fraction:
        """Return exact units for conversion and independent host diagnostics."""
        return (Fraction(1 << self.exponent) if self.exponent >= 0 else
                Fraction(1, 1 << -self.exponent))

    @classmethod
    def for_values(cls, values: Iterable[float], *, max_terms: int = 1) -> IntegerScale:
        """Choose a grid with headroom for a bounded sum of original finite inputs.

        Parameters
        ------------
        - values (Iterable[float]): immutable snapshot operands, including signed inputs.
        - max_terms (int): maximum number of operands of the largest magnitude to accumulate.

        Return
        --------
        - output (IntegerScale): finest supported grid whose magnitude estimate fits 60 bits.
          Every operation still checks overflow; the estimate is not a certificate.
        """
        if max_terms < 1:
            raise ValueError("max_terms must be positive")
        largest = Fraction(0)
        for value in values:
            if not math.isfinite(value):
                raise ValueError("integer scale requires finite original operands")
            largest = max(largest, abs(Fraction(value)))
        magnitude = largest * max_terms
        exponent = -1074
        if magnitude:
            exponent = max(exponent, magnitude.numerator.bit_length()
                           - magnitude.denominator.bit_length() - 59)
        while exponent <= 1023 and magnitude > ((1 << 60) - 1) * cls(exponent).quantum:
            exponent += 1
        if exponent > 1023:
            raise ValueError("bounded accumulation has no supported integer scale")
        return cls(exponent)

    def encode(self, value: float) -> IntegerInterval:
        """Enclose a represented binary64 value using exact floor and ceiling.

        Parameters
        ------------
        - value (float): original FP64 operand; nonfinite inputs become unresolved.

        Return
        --------
        - output (IntegerInterval): outward endpoints, or invalid if either cannot fit.
        """
        if not math.isfinite(value):
            return IntegerInterval()
        scaled = Fraction(value) / self.quantum
        numerator, denominator = scaled.numerator, scaled.denominator
        return IntegerInterval.checked(numerator // denominator, -(-numerator // denominator))

    def encode_array(self, values: np.ndarray) -> np.ndarray:
        """Pack original FP64 operands using exact power-of-two scaling and outward endpoints.

        Parameters
        ------------
        - values (np.ndarray): arbitrary shape of original binary64 operands.

        Return
        --------
        - output (np.ndarray): uint32 array with five trailing interval words; nonfinite and
          out-of-range values are invalid, and sub-grid signed values retain their outer unit.
        """
        values = np.asarray(values, np.float64)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled = np.ldexp(values, -self.exponent)
        valid = np.isfinite(scaled) & np.isfinite(values) & (scaled > -2.0**63) & (scaled < 2.0**63)
        scaled = np.where(valid, scaled, 0.0)
        lower, upper = np.floor(scaled), np.ceil(scaled)
        # Normal ldexp is exact. Underflow can only affect magnitudes below one
        # grid unit, whose outward integers are determined by the original sign.
        lower = np.where(valid & (values < 0.0) & (scaled == 0.0), -1.0, lower)
        upper = np.where(valid & (values > 0.0) & (scaled == 0.0), 1.0, upper)
        endpoints = np.stack((lower.astype(np.int64), upper.astype(np.int64)), axis=-1)
        result = np.empty((*values.shape, 5), np.uint32)
        result[..., :4] = endpoints.astype("<i8", copy=False).view("<u4").reshape(*values.shape, 4)
        result[..., 4] = valid
        return result

    def rounded(self, value: IntegerInterval, direction: int) -> IntegerInterval:
        """Enclose one binary64 rounding and optional directed nextafter operation.

        Parameters
        ------------
        - value (IntegerInterval): enclosure of the exact real operation in grid units.
        - direction (int): -1 for nextafter toward -inf, 0 for nearest, +1 toward +inf.

        Return
        --------
        - output (IntegerInterval): padded interval, or unresolved on finite-range/padding failure.
        """
        if direction not in (-1, 0, 1):
            raise ValueError("rounding direction must be -1, 0 or 1")
        if not value.valid:
            return IntegerInterval()
        bits = max(abs(value.lower), abs(value.upper)).bit_length()
        if self.exponent + bits > 1023:
            return IntegerInterval()
        padding = 1 << max(1, bits - 51)
        return IntegerInterval.checked(value.lower - (padding if direction <= 0 else 0),
                                       value.upper + (padding if direction >= 0 else 0))

    def add(
        self, first: IntegerInterval, second: IntegerInterval, *, direction: int | None = None,
    ) -> IntegerInterval:
        """Enclose exact addition, optionally including the reference FP64 rounding step.

        Parameters
        ------------
        - first (IntegerInterval): first operand on this grid.
        - second (IntegerInterval): second operand on this grid.
        - direction (int | None): optional nearest/directed FP64 enclosure.

        Return
        --------
        - output (IntegerInterval): checked sum; invalid inputs/overflow remain unresolved.
        """
        if not first.valid or not second.valid:
            return IntegerInterval()
        result = IntegerInterval.checked(first.lower + second.lower, first.upper + second.upper)
        return result if direction is None else self.rounded(result, direction)

    def subtract(
        self, first: IntegerInterval, second: IntegerInterval, *, direction: int | None = None,
    ) -> IntegerInterval:
        """Enclose subtraction by reversing the second interval's signed endpoints."""
        negative = IntegerInterval.checked(-second.upper, -second.lower) if second.valid else second
        return self.add(first, negative, direction=direction)

    def multiply(
        self, value: IntegerInterval, factor: int, *, direction: int | None = None,
    ) -> IntegerInterval:
        """Enclose multiplication by an exactly represented nonnegative uint32.

        Parameters
        ------------
        - value (IntegerInterval): signed interval on this grid.
        - factor (int): exact integer multiplier in [0, 2**32-1].
        - direction (int | None): optional nearest/directed FP64 enclosure.

        Return
        --------
        - output (IntegerInterval): checked scaled interval, invalid on overflow or invalid input.
        """
        if not 0 <= factor <= WORD_MASK:
            raise ValueError("factor must fit uint32")
        if not value.valid:
            return IntegerInterval()
        result = IntegerInterval.checked(value.lower * factor, value.upper * factor)
        return result if direction is None else self.rounded(result, direction)

    def upper_float(self, value: IntegerInterval) -> float:
        """Convert a certificate upper outward; unresolved intervals become positive infinity."""
        if not value.valid:
            return math.inf
        exact = value.upper * self.quantum
        try:
            result = float(exact)
        except OverflowError:
            return math.inf
        if not math.isfinite(result):
            return math.inf
        return math.nextafter(result, math.inf) if Fraction(result) < exact else result

    def closes(self, upper: IntegerInterval, cutoff: float) -> bool:
        """Permit closure only with a finite upper no greater than the downward FP64 cutoff."""
        encoded = self.encode(cutoff)
        return upper.valid and encoded.valid and upper.upper <= encoded.lower

    def positive_sum(self, values: Iterable[float], *, extra_rounds: int = 0) -> IntegerInterval:
        """Enclose every ordering of the reference's downward positive-price accumulation.

        Parameters
        ------------
        - values (Iterable[float]): distinct finite nonnegative row prices, with exact membership
          already established by the caller. Zero rows do not cause a reference addition.
        - extra_rounds (int): additional downward additions when joining disjoint prefix groups.

        Return
        --------
        - output (IntegerInterval): order-independent enclosure including the final zero clamp;
          excessive magnitudes/counts or any invalid input remain unresolved.
        """
        if not 0 <= extra_rounds <= WORD_MASK:
            raise ValueError("extra_rounds must fit uint32")
        total = IntegerInterval.checked(0, 0)
        count = 0
        for value in values:
            if not math.isfinite(value) or value < 0:
                return IntegerInterval()
            if value > 0:
                total = self.add(total, self.encode(value))
                count += 1
        count += extra_rounds
        if not total.valid or count > WORD_MASK:
            return IntegerInterval()
        padded = self.rounded(total, -1)
        if not padded.valid:
            return IntegerInterval()
        error = self.multiply(IntegerInterval.checked(total.lower - padded.lower,
                                                      total.lower - padded.lower), count)
        lower = self.subtract(total, error)
        if not lower.valid:
            return IntegerInterval()
        return IntegerInterval.checked(max(0, lower.lower), max(0, total.upper))
