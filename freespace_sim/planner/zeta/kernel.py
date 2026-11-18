"""Compiled sparse reservation pricing for simulator-owned any-angle geometry."""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numba.extending import register_jitable

from ..route_numeric import FSUM_MAX_PARTIALS, _fsum_add, _fsum_finalize
from .price_floor import price_floor

_price_floor = register_jitable(price_floor)


@njit(cache=True, nogil=True)
def write_label(
    ints: np.ndarray, floats: np.ndarray, index: int, node: int, parent: int,
    context: int, paid_id: int, sequence: int, terminal: bool, first: bool,
    stamp: float, length: float, positive: float, canonical_length: float,
) -> None:
    """Store one label without allocating temporary arrays for column assignment.

    Parameters
    ------------
    - ints (np.ndarray): existing seven-row integer label arena.
    - floats (np.ndarray): existing four-row floating-point label arena.
    - index (int): valid destination column in both arenas.
    - node (int): current candidate waypoint index.
    - parent (int): previous label index, or minus one at an origin.
    - context (int): departure and endpoint-choice context index.
    - paid_id (int): interned paid-reservation set index.
    - sequence (int): exact subpart-length sequence index.
    - terminal (bool): whether this label has the destination endpoint role.
    - first (bool): whether this label precedes the first horizontal segment.
    - stamp (float): exact arrival timestamp.
    - length (float): sequentially accumulated physical length.
    - positive (float): accumulated nonnegative reservation price contribution.
    - canonical_length (float): separately accumulated canonical physical length.

    Return
    --------
    - output (None): the specified arena column is updated in place.
    """
    ints[0, index] = node
    ints[1, index] = parent
    ints[2, index] = context
    ints[3, index] = paid_id
    ints[4, index] = sequence
    ints[5, index] = terminal
    ints[6, index] = first
    floats[0, index] = stamp
    floats[1, index] = length
    floats[2, index] = positive
    floats[3, index] = canonical_length


@njit(cache=True, nogil=True)
def partition_expired(
    paid: np.ndarray, row_ends: np.ndarray, cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition paid rows without changing their order or the expiration boundary.

    Parameters
    ------------
    - paid (np.ndarray): immutable sorted unique paid-reservation ids.
    - row_ends (np.ndarray): exact reservation-period end timestamp by row id.
    - cutoff (float): earliest possible future buffered claim start.

    Return
    --------
    - output (tuple): immutable live and expired row arrays. Rows ending exactly at
      cutoff expire; their signed accumulated charge remains the caller's responsibility.
    """
    count = 0
    for row in paid:
        if row_ends[row] <= cutoff:
            count += 1
    if count == 0:
        return paid, np.empty(0, dtype=np.int64)
    if count == len(paid):
        return np.empty(0, dtype=np.int64), paid
    live = np.empty(len(paid) - count, dtype=np.int64)
    expired = np.empty(count, dtype=np.int64)
    i = j = 0
    for row in paid:
        if row_ends[row] <= cutoff:
            expired[j] = row
            j += 1
        else:
            live[i] = row
            i += 1
    return live, expired


@njit(cache=True, nogil=True)
def prepare_timed_edge(
    stamp: float, durations: np.ndarray, part_offsets: np.ndarray, spaces: np.ndarray,
    row_offsets: np.ndarray, steps: np.ndarray, row_ids: np.ndarray, blocked: np.ndarray,
    marks: np.ndarray, generation: int, dt: float, buffer: float,
    absolute_times: bool = False,
) -> tuple[float, np.ndarray, bool]:
    """Build an immutable timed footprint independent of the prefix's paid history.

    Parameters
    ------------
    - stamp (float): exact parent arrival timestamp, without cache-key rounding.
    - durations (np.ndarray): nominal subpart durations or exact absolute end times.
    - part_offsets (np.ndarray): offsets of each subpart's relevant spatial groups.
    - spaces (np.ndarray): spatial-group ids in simulator geometry traversal order.
    - row_offsets (np.ndarray): offsets of each spatial group's sorted reservation rows.
    - steps (np.ndarray): exact reservation periods, sorted within each spatial group.
    - row_ids (np.ndarray): current pricing call's integer reservation-row ids.
    - blocked (np.ndarray): forbidden-row flags, including zero-price exclusions.
    - marks (np.ndarray): reusable integer scratch for unique row membership.
    - generation (int): fresh scratch generation, distinct from all earlier operations.
    - dt (float): unchanged reservation period length.
    - buffer (float): unchanged symmetric reservation time buffer.
    - absolute_times (bool): treat durations as precomputed absolute end times.

    Return
    --------
    - output (tuple): exact end timestamp, unique row ids in first-visit order, and
      forbidden flag. Row ids are valid only for this pricing call's row table.
    """
    encountered = [np.int64(0)]
    encountered.pop()
    for part in range(len(durations)):
        end = durations[part] if absolute_times else stamp + durations[part]
        t0, t1 = stamp - buffer, end + buffer
        if t0 == t1:
            stamp = end
            continue
        # Preserve the builder's multiplied-grid boundary repair exactly.
        first = math.floor(t0 / dt)
        while (first + 1) * dt <= t0:
            first += 1
        while first * dt > t0:
            first -= 1
        stop = math.ceil(t1 / dt)
        while (stop - 1) * dt >= t1:
            stop -= 1
        while stop * dt < t1:
            stop += 1
        for offset in range(part_offsets[part], part_offsets[part + 1]):
            space = spaces[offset]
            begin, limit = row_offsets[space], row_offsets[space + 1]
            lo, hi = begin, limit
            while lo < hi:
                mid = (lo + hi) // 2
                if steps[mid] < first:
                    lo = mid + 1
                else:
                    hi = mid
            while lo < limit and steps[lo] < stop:
                row = row_ids[lo]
                if blocked[row]:
                    return stamp, np.empty(0, dtype=np.int64), True
                if marks[row] != generation:
                    marks[row] = generation
                    encountered.append(row)
                lo += 1
        stamp = end
    return stamp, np.asarray(encountered, dtype=np.int64), False


@njit(cache=True, nogil=True)
def prepare_windowed_edge(
    stamp: float, ends: np.ndarray, firsts: np.ndarray, stops: np.ndarray,
    part_offsets: np.ndarray, spaces: np.ndarray, row_offsets: np.ndarray,
    steps: np.ndarray, row_ids: np.ndarray, blocked: np.ndarray, marks: np.ndarray,
    generation: int,
) -> tuple[float, np.ndarray, bool]:
    """Resolve reusable exact time windows against this call's current reservation rows.

    Parameters
    ------------
    - stamp (float): exact initial leg timestamp.
    - ends (np.ndarray): immutable exact subpart end times.
    - firsts (np.ndarray): immutable first included reservation period for each subpart.
    - stops (np.ndarray): immutable excluded upper period for each subpart.
    - part_offsets (np.ndarray): subpart offsets into current spatial-group ids.
    - spaces (np.ndarray): relevant spatial groups in the original traversal order.
    - row_offsets (np.ndarray): offsets of each current group's reservation rows.
    - steps (np.ndarray): current period indices sorted within each spatial group.
    - row_ids (np.ndarray): reservation-row ids belonging only to this pricing call.
    - blocked (np.ndarray): fresh forbidden flags, including zero-price reservations.
    - marks (np.ndarray): scratch array for duplicate-row suppression.
    - generation (int): fresh scratch generation.

    Return
    --------
    - output (tuple): end timestamp, current unique row ids in first-visit order and
      current blocked flag. No row ids or blocked decisions enter the persistent cache.
    """
    encountered = [np.int64(0)]
    encountered.pop()
    for part in range(len(ends)):
        first, stop = firsts[part], stops[part]
        for offset in range(part_offsets[part], part_offsets[part + 1]):
            space = spaces[offset]
            begin, limit = row_offsets[space], row_offsets[space + 1]
            lo, hi = begin, limit
            while lo < hi:
                mid = (lo + hi) // 2
                if steps[mid] < first:
                    lo = mid + 1
                else:
                    hi = mid
            while lo < limit and steps[lo] < stop:
                row = row_ids[lo]
                if blocked[row]:
                    return stamp, np.empty(0, dtype=np.int64), True
                if marks[row] != generation:
                    marks[row] = generation
                    encountered.append(row)
                lo += 1
        stamp = ends[part]
    return stamp, np.asarray(encountered, dtype=np.int64), False


@njit(cache=True, nogil=True)
def apply_timed_edge(
    length: float, paid: np.ndarray, lengths: np.ndarray, encountered: np.ndarray,
    positive_prices: np.ndarray, marks: np.ndarray, generation: int,
) -> tuple[float, np.ndarray, float]:
    """Advance prefix length and price only newly paid rows of a cached timed footprint.

    Parameters
    ------------
    - length (float): sequentially accumulated prefix length.
    - paid (np.ndarray): immutable sorted unique ids already paid by this prefix.
    - lengths (np.ndarray): simulator-identical subpart lengths in original order.
    - encountered (np.ndarray): unique cached footprint ids in first-visit order.
    - positive_prices (np.ndarray): nonnegative parts of current reservation prices.
    - marks (np.ndarray): reusable integer scratch for membership checks.
    - generation (int): fresh scratch generation, distinct from footprint preparation.

    Return
    --------
    - output (tuple): sequential child length, immutable sorted union ids, and the
      exact sum of newly paid positive prices. Negative-price ids remain in the union.
    """
    for part_length in lengths:
        length = length + part_length
    for row in paid:
        marks[row] = generation
    added = np.empty(len(encountered), dtype=np.int64)
    count = 0
    partials = np.empty(FSUM_MAX_PARTIALS, dtype=np.float64)
    npartials = 0
    for row in encountered:
        if marks[row] != generation:
            added[count] = row
            count += 1
            if positive_prices[row] > 0.0:
                npartials = _fsum_add(partials, npartials, positive_prices[row])
                if npartials < 0:
                    raise ArithmeticError("Zeta positive-price sum overflowed its expansion")
    if count == 0:
        return length, paid, 0.0
    new = added[:count]
    new.sort()
    union = np.empty(len(paid) + count, dtype=np.int64)
    i = j = k = 0
    while i < len(paid) and j < count:
        if paid[i] < new[j]:
            union[k] = paid[i]
            i += 1
        else:
            union[k] = new[j]
            j += 1
        k += 1
    while i < len(paid):
        union[k] = paid[i]
        i += 1
        k += 1
    while j < count:
        union[k] = new[j]
        j += 1
        k += 1
    return length, union, _fsum_finalize(partials, npartials)


@njit(cache=True, nogil=True)
def build_successor_scan(
    distances: np.ndarray, remainings: np.ndarray, length: float, positive: float,
    inherited_floor: float, ground: float, fold: float, reference: float,
    air_weight: float, nominal_speed: float, negative: float, tolerance: float,
    max_length: float, best: float, price_error: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter and stably order deferred successors using unchanged floating-point bounds.

    Parameters
    ------------
    - distances (np.ndarray): next-segment geometric lengths in original neighbor order.
    - remainings (np.ndarray): matching distances from the targets to the destination.
    - length (float): accumulated prefix length.
    - positive (float): accumulated nonnegative price contribution.
    - inherited_floor (float): proved bound inherited from the parent prefix.
    - ground (float): departure ground-delay cost for this route context.
    - fold (float): exact endpoint fold length, or NaN when unavailable.
    - reference (float): unchanged reference route length.
    - air_weight (float): unchanged airborne delay weight.
    - nominal_speed (float): unchanged nominal speed used in cost conversion.
    - negative (float): all-negative-price credit used by the reference relaxation.
    - tolerance (float): unchanged conservative bound subtraction.
    - max_length (float): maximum physical route length.
    - best (float): current feasible priced cost cutoff.
    - price_error (float): shared whole-union rounding allowance for this price vector.

    Return
    --------
    - output (tuple): surviving neighbor indices and corresponding lower bounds, sorted
      by the actual bound with the original neighbor order retained for exact ties.
    """
    indices = np.empty(len(distances), dtype=np.int64)
    floors = np.empty(len(distances), dtype=np.float64)
    count = 0
    for index in range(len(distances)):
        lower = length + distances[index]
        remaining = remainings[index]
        if lower + remaining <= max_length + 1e-7:
            cost = ground
            if not math.isnan(fold) and reference > 1e-9:
                detour = max(0.0, lower + remaining + fold - reference)
                cost += air_weight * detour / nominal_speed
            floor = max(inherited_floor,
                        _price_floor(cost, positive, negative, price_error, tolerance))
            if floor < best:
                indices[count] = index
                floors[count] = floor
                count += 1
    order = np.argsort(floors[:count], kind="mergesort")
    return indices[order], floors[order]


@njit(cache=True, nogil=True)
def evaluate_edge(
    stamp: float, length: float, paid: np.ndarray,
    durations: np.ndarray, lengths: np.ndarray, part_offsets: np.ndarray,
    spaces: np.ndarray, row_offsets: np.ndarray, steps: np.ndarray, row_ids: np.ndarray,
    blocked: np.ndarray, positive_prices: np.ndarray, marks: np.ndarray,
    generation: int, dt: float, buffer: float, absolute_times: bool = False,
) -> tuple[float, float, np.ndarray, float, bool]:
    """Advance exact builder clocks and union only priced rows, checking all blocked rows.

    Parameters
    ------------
    - stamp (float): exact arrival timestamp of the parent prefix.
    - length (float): sequentially accumulated physical prefix length.
    - paid (np.ndarray): sorted unique integer ids of previously paid nonzero-price rows.
    - durations (np.ndarray): nominal durations, or absolute ends when absolute_times is true.
    - lengths (np.ndarray): builder-identical length of each sub-box.
    - part_offsets (np.ndarray): offsets into each sub-box's relevant spatial cells.
    - spaces (np.ndarray): spatial-cell group ids for the sub-boxes.
    - row_offsets (np.ndarray): offsets into each spatial group's sorted time rows.
    - steps (np.ndarray): exact integer reservation periods, sorted within spatial groups.
    - row_ids (np.ndarray): price/forbidden row ids corresponding to periods.
    - blocked (np.ndarray): forbidden-row flags, including rows with zero current price.
    - positive_prices (np.ndarray): nonnegative part of each row's actual price.
    - marks (np.ndarray): reusable integer scratch for exact union membership.
    - generation (int): fresh scratch generation for this edge evaluation.
    - dt (float): unchanged reservation period length.
    - buffer (float): unchanged symmetric time buffer.
    - absolute_times (bool): use precomputed bounded-speed sub-box ends without rounding again.

    Return
    --------
    - output (tuple): exact timestamp, length, union ids, newly paid positive price and
      forbidden flag. Every successful union is immutable to subsequent calls.
    """
    for row in paid:
        marks[row] = generation
    added = [np.int64(0)]
    added.pop()
    partials = np.empty(FSUM_MAX_PARTIALS, dtype=np.float64)
    npartials = 0
    for part in range(len(durations)):
        end = durations[part] if absolute_times else stamp + durations[part]
        t0, t1 = stamp - buffer, end + buffer
        if t0 == t1:
            stamp = end
            length = length + lengths[part]
            continue
        # Match windows._periods_overlapping, including multiplied-grid boundary repair.
        first = math.floor(t0 / dt)
        while (first + 1) * dt <= t0:
            first += 1
        while first * dt > t0:
            first -= 1
        stop = math.ceil(t1 / dt)
        while (stop - 1) * dt >= t1:
            stop -= 1
        while stop * dt < t1:
            stop += 1
        for offset in range(part_offsets[part], part_offsets[part + 1]):
            space = spaces[offset]
            begin, limit = row_offsets[space], row_offsets[space + 1]
            lo, hi = begin, limit
            while lo < hi:
                mid = (lo + hi) // 2
                if steps[mid] < first:
                    lo = mid + 1
                else:
                    hi = mid
            while lo < limit and steps[lo] < stop:
                row = row_ids[lo]
                if blocked[row]:
                    return stamp, length, paid, 0.0, True
                if marks[row] != generation:
                    marks[row] = generation
                    added.append(row)
                    if positive_prices[row] > 0.0:
                        npartials = _fsum_add(partials, npartials, positive_prices[row])
                        if npartials < 0:
                            raise ArithmeticError("Zeta positive-price sum overflowed its expansion")
                lo += 1
        stamp = end
        length = length + lengths[part]
    if not added:
        return stamp, length, paid, 0.0, False
    new = np.asarray(added, dtype=np.int64)
    new.sort()
    union = np.empty(len(paid) + len(new), dtype=np.int64)
    i = j = k = 0
    while i < len(paid) and j < len(new):
        if paid[i] < new[j]:
            union[k] = paid[i]
            i += 1
        else:
            union[k] = new[j]
            j += 1
        k += 1
    while i < len(paid):
        union[k] = paid[i]
        i += 1
        k += 1
    while j < len(new):
        union[k] = new[j]
        j += 1
        k += 1
    return stamp, length, union, _fsum_finalize(partials, npartials), False
