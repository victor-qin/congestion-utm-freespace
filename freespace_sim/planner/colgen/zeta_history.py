"""Exact signed-price comparisons for future-equivalent physical route prefixes."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class _History:
    """Live reservations and irrevocably paid charges of one retained prefix."""

    identifier: int
    paid: frozenset[Hashable]
    positive: frozenset[Hashable]
    negative: frozenset[Hashable]
    retired: int


@dataclass(slots=True)
class _PriceBucket:
    """One live reservation set, indexed by its exact retired charges."""

    representative: _History
    charges: dict[int, list[int]]
    maximum: int


class PriceHistory:
    """Compare signed reservation unions using exact binary-rational price sums.

    Attributes
    ------------
    - weights (dict[Hashable, int]): prices at one shared power-of-two scale.
    - dead (set[int]): replaced labels whose pending search work may be skipped.
    - pruned (int): rejected candidates plus replaced retained labels.
    - comparisons (int): signed-price dominance comparisons performed, excluding length pairs.
    - merged (int): guarded equal-price prefixes represented by another prefix.
    - shorter_merged (int): merged prefixes compared by both accumulated lengths.
    - merge_error_bound (float): maximum whole-route guard required by a performed merge.
    """

    def __init__(
        self, prices: Mapping[Hashable, float], *, group_skips: bool = False,
        merge_equivalent: bool = False, merge_shorter: bool = False,
    ) -> None:
        """Pack finite prices once per frozen dual vector, with no rounding.

        Parameters
        ------------
        - prices (Mapping[Hashable, float]): row keys or integer row IDs and signed prices.
        - group_skips (bool): skip broad-group scans whose retired-charge range cannot
          meet the required strict price advantage.
        - merge_equivalent (bool): keep one guarded broad-state representative for
          identical live reservations and exact retired charge; expose its uncertainty.
        - merge_shorter (bool): also compare both unchanged length accumulators under
          an explicitly supplied continuation key and the same whole-route guard.

        Return
        --------
        - output (None): initializes exact weights and empty future-state groups.
        """
        ratios = {row: float(price).as_integer_ratio() for row, price in prices.items()}
        shift = max((denominator.bit_length() - 1 for _, denominator in ratios.values()), default=0)
        self.weights = {
            row: numerator << (shift - (denominator.bit_length() - 1))
            for row, (numerator, denominator) in ratios.items()
        }
        self._scale = 1 << shift
        self._minimums: dict[float, int] = {}
        self.positive_rows = frozenset(row for row, weight in self.weights.items() if weight > 0)
        self.negative_rows = frozenset(row for row, weight in self.weights.items() if weight < 0)
        self._paid_sets: dict[int, tuple[frozenset, frozenset, frozenset]] = {}
        self.groups: dict[Hashable, list[_History]] = {}
        self.broad_groups: dict[Hashable, dict[frozenset[Hashable], _PriceBucket]] = {}
        self._group_skips = group_skips
        self._merge_equivalent = merge_equivalent
        self._merge_shorter = merge_shorter
        self._length_groups: dict[tuple[object, ...], list[tuple[float, float, int]]] = {}
        self._broad_extrema: dict[tuple[object, ...], tuple[int, int]] = {}
        self.dead: set[int] = set()
        self.pruned = self.comparisons = 0
        self.strong_pruned = 0
        self.merged = 0
        self.shorter_merged = 0
        self.merge_error_bound = 0.0
        self.group_reject_skips = self.group_replace_skips = 0

    def charge(self, rows: Iterable[Hashable]) -> int:
        """Return an exact signed price sum at this call's common scale."""
        return sum(self.weights[row] for row in rows)

    def _dominates(
        self, first: _History, second: _History, minimum: int = 0,
        second_retired: int | None = None,
    ) -> bool:
        """Bound the worst price difference over every shared future reservation set."""
        self.comparisons += 1
        budget = (second.retired if second_retired is None else second_retired) - first.retired - minimum
        if budget < 0:
            return False
        extra_positive = first.positive - second.paid
        extra_negative = second.negative - first.paid
        if not extra_positive and not extra_negative:
            return True
        if budget == 0:
            return False
        return self.charge(extra_positive) - self.charge(extra_negative) <= budget

    def accept(
        self, key: Hashable, paid: Iterable[Hashable], retired: int, identifier: int,
        *, broad_key: Hashable | None = None, margin: float = 0.0,
        length_key: Hashable | None = None,
        lengths: tuple[float, float] | None = None,
        paid_id: int | None = None,
    ) -> bool:
        """Keep a prefix only if an existing future-equivalent prefix cannot replace it.

        Parameters
        ------------
        - key (Hashable): exact tuple or interned ID for identical future actions,
          physical costs and feasibility; IDs must not be reused within this call.
        - paid (Iterable[Hashable]): live priced reservations, never including expired rows.
        - retired (int): exact signed sum already charged for expired reservations.
        - identifier (int): unique accepted-label ID; rejected IDs need not be consumed.
        - broad_key (Hashable | None): same continuation state and both length accumulators,
          without length-sequence identity; None disables the stronger comparison.
        - margin (float): certified upper bound on final cost rounding differences for
          this broad group. Ordinary dominance needs a strictly larger price advantage;
          enabled equivalent-price merges instead widen the returned certificate.
        - length_key (Hashable | None): identical continuation state without either length
          or segment sequence. The caller must establish nonnegative air cost and
          the whole-route feasibility guard; None disables this comparison.
        - lengths (tuple[float, float] | None): sequential and per-edge accumulated lengths.
          A replacement must be no greater in both; neither accumulator is rounded.
        - paid_id (int | None): stable ID of an immutable sorted numeric paid array. The
          compact caller interns it by full byte equality and never reuses the ID. Cache
          its signed partitions only within this history object's frozen price vector.

        Return
        --------
        - output (bool): whether to enqueue the candidate before any interruption check.
          Replaced labels are marked dead; their previously generated children remain valid.
        """
        prepared = None if paid_id is None else self._paid_sets.get(paid_id)
        if prepared is None:
            tail = frozenset(paid if paid_id is None else paid.tolist())
            positive, negative = tail & self.positive_rows, tail & self.negative_rows
            if paid_id is not None:
                self._paid_sets[paid_id] = tail, positive, negative
        else:
            tail, positive, negative = prepared
        candidate = _History(identifier, tail, positive, negative, retired)
        broad = None
        extrema = None
        minimum = 0
        if broad_key is not None:
            minimum = self._minimums.get(margin)
            if minimum is None:
                if margin < 0.0:
                    raise ValueError("history rounding margin must be nonnegative")
                numerator, denominator = float(margin).as_integer_ratio()
                minimum = numerator * self._scale // denominator + 1
                self._minimums[margin] = minimum
            broad = self.broad_groups.get(broad_key, {})
            if self._group_skips:
                extrema = self._broad_extrema.get(broad_key)
            # The live-set penalty in either comparison is nonnegative. Retired
            # charges alone can therefore rule out every witness in the group.
            if extrema is not None and retired - extrema[0] < minimum:
                self.group_reject_skips += 1
            elif any(self._dominates(bucket.representative, candidate, minimum)
                     for bucket in broad.values()):
                self.pruned += 1
                self.strong_pruned += 1
                return False
        group = self.groups.get(key, [])
        # Reject first so mutually dominating ties always leave a surviving representative.
        if any(self._dominates(existing, candidate) for existing in group):
            self.pruned += 1
            return False
        if self._merge_equivalent and broad is not None:
            bucket = broad.get(tail)
            if bucket is not None and any(
                existing not in self.dead for existing in bucket.charges.get(retired, ())
            ):
                # The caller must widen its certificate by this whole-route guard.
                self.merged += 1
                self.pruned += 1
                self.merge_error_bound = max(self.merge_error_bound, margin)
                return False
        length_group_key = None
        length_group = []
        if (self._merge_shorter and broad is not None and length_key is not None
                and lengths is not None
                and all(math.isfinite(value) and value >= 0.0 for value in lengths)):
            length_group_key = length_key, tail, retired
            length_group = [entry for entry in self._length_groups.get(length_group_key, ())
                            if entry[2] not in self.dead]
            self._length_groups[length_group_key] = length_group
            if any(a <= lengths[0] and b <= lengths[1] for a, b, _identifier in length_group):
                self.pruned += 1
                self.merged += 1
                self.shorter_merged += 1
                self.merge_error_bound = max(self.merge_error_bound, margin)
                return False
        kept = []
        for existing in group:
            if self._dominates(candidate, existing):
                self.pruned += existing.identifier not in self.dead
                self.dead.add(existing.identifier)
            else:
                kept.append(existing)
        kept.append(candidate)
        self.groups[key] = kept
        if broad is not None:
            maximum_removed = False
            if extrema is not None and extrema[1] - retired < minimum:
                self.group_replace_skips += 1
                replacements = ()
            else:
                replacements = tuple(broad.items())
            for signature, bucket in replacements:
                existing = bucket.representative
                if not self._dominates(candidate, existing, minimum, bucket.maximum):
                    continue
                for charge, identifiers in tuple(bucket.charges.items()):
                    if self._dominates(candidate, existing, minimum, charge):
                        newly_dead = set(identifiers) - self.dead
                        self.dead.update(newly_dead)
                        self.pruned += len(newly_dead)
                        self.strong_pruned += len(newly_dead)
                        if extrema is not None and charge == extrema[1]:
                            maximum_removed = True
                        del bucket.charges[charge]
                if bucket.charges:
                    bucket.maximum = max(bucket.charges)
                else:
                    del broad[signature]
            if tail in broad:
                bucket = broad[tail]
                bucket.charges.setdefault(retired, []).append(identifier)
                bucket.maximum = max(bucket.maximum, retired)
                if retired < bucket.representative.retired:
                    bucket.representative = candidate
            else:
                broad[tail] = _PriceBucket(candidate, {retired: [identifier]}, retired)
            # For one live set, the cheapest retired charge is the strongest rejection
            # witness. Other charges/IDs stay queued and are removed only past the margin.
            self.broad_groups[broad_key] = broad
            if self._group_skips:
                if extrema is None:
                    self._broad_extrema[broad_key] = retired, retired
                else:
                    # Removing the previous minimum requires a strictly cheaper
                    # candidate, so insertion already supplies the new minimum.
                    low = min(extrema[0], retired)
                    high = (max(bucket.maximum for bucket in broad.values())
                            if maximum_removed else max(extrema[1], retired))
                    self._broad_extrema[broad_key] = low, high
        if length_group_key is not None:
            retained = []
            for a, b, previous in length_group:
                if previous in self.dead:
                    continue
                if lengths[0] <= a and lengths[1] <= b:
                    # Keep the new label's own path and floor; its children cover the
                    # removed work before the caller performs its next cutoff check.
                    self.dead.add(previous)
                    self.pruned += 1
                    self.merged += 1
                    self.shorter_merged += 1
                    self.merge_error_bound = max(self.merge_error_bound, margin)
                else:
                    retained.append((a, b, previous))
            retained.append((lengths[0], lengths[1], identifier))
            self._length_groups[length_group_key] = retained
        return True
