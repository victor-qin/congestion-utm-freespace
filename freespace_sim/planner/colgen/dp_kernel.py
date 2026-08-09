"""Compiled (numba) pricing DP for the column-generation planner.

:func:`pricing._best_column` is the oracle: an exact, dominance-pruned label search over one
flight's space-time DAG, and 95.8% of a colgen solve's wall clock.  This module is that
search over the flat arrays :mod:`.dp_prepare` packs, with the Python reference left
untouched beside it as both the definition of a correct answer and the fallback.

**This file holds the primitives the search is built from.**  They are separated out and
tested against Python oracles first because every one of them is a place where being
*approximately* right produces a plausible wrong answer rather than a crash: a different
column with the same reduced cost, a label pruned that should have survived, a tie broken
the other way.  None of those raise.

What has to be reproduced exactly, and why each is here:

``_fsum_*``
    ``math.fsum`` is exactly rounded and order-independent; a running ``+=`` is neither.
    The reference sums duals with it in the arc loop (pricing.py:1606), so an ulp of drift
    moves a label score across ``_SCORE_EPS`` and hands dominance to the other label.
``_range_sum`` / ``_visit_cost`` / ``_row_cost``
    The three dual queries, mirroring :class:`~.dp_prepare.PreparedDuals` term for term.
    ``visit_cost`` is a *subtraction of two prefix sums* and ``row_cost`` is a *stored
    value*: deriving the second from the first is the one shortcut that is not available,
    since ``(a + v) - a != v``.
``_path_cmp`` / ``_tie_lt`` / ``_prefer``
    The reference's dominance rule compares scores within ``_SCORE_EPS`` and breaks ties on
    ``(hops, departure_step, lane, path)`` -- the whole path, lexicographically from its
    root.  That epsilon band makes ``_prefer`` **non-transitive**, so insertion order is
    itself part of the answer and the comparison cannot be approximated.

Numba is optional: importing this module raises :class:`ImportError` without it, and the
host is expected to warn once and use the reference (see ``astar_kernel`` for the same
contract).
"""
from __future__ import annotations

import numpy as np
from numba import njit

# Must match `pricing._SCORE_EPS` / `pricing._RECOMPUTE_EPS` exactly.  These are not
# tolerances chosen here: they are the bands the reference's dominance and certification
# already use, and a kernel that widened either would prune what the oracle keeps.
SCORE_EPS = 1e-12
RECOMPUTE_EPS = 1e-8

_MAGIC = np.uint64(0x9E3779B97F4A7C15)  # Fibonacci hashing multiplier, as in `astar_kernel`

# Arc role bits.  Restated from `dp_prepare` rather than imported so numba can treat them as
# compile-time constants; `test_kernel_arc_roles_match_dp_prepare` fails if the two drift.
_ARC_INTERNAL = 1 << 0
_ARC_FIRST = 1 << 1
_ARC_LAST = 1 << 2
_ARC_FIRST_LAST = 1 << 3

# Search outcomes.  Only `STATUS_OK` licenses an optimality claim; everything else means the
# host must widen a budget and retry, or fall back to the reference.
STATUS_OK = 0
STATUS_LABEL_LIMIT = 1      # label pool full -- host doubles it and re-runs
STATUS_STATE_LIMIT = 2      # dominance table saturated -- host doubles it and re-runs
STATUS_CANDIDATE_LIMIT = 3  # candidate buffer full; the search itself completed
STATUS_CANCELLED = 4        # deadline fired mid-search

# The longest exactly-representable partial expansion `_fsum_add` will hold.  Shewchuk's
# algorithm needs one partial per distinct exponent range in play; 64 covers a full
# float64 exponent sweep with room to spare, and the arc-loop sums this backs are a
# handful of terms from one visit window.
FSUM_MAX_PARTIALS = 64


# --------------------------------------------------------------------------- exact sums


@njit(cache=True, nogil=True)
def _fsum_add(partials, n, x):
    """Accumulate ``x`` into a Shewchuk partial expansion; return the new partial count.

    This is CPython's ``math.fsum`` inner loop (``Modules/mathmodule.c``): each new term is
    added to every surviving partial with a ``two_sum``, keeping the exact low word of each
    addition and carrying the high word forward.  The expansion is therefore an exact
    representation of the running total as a sum of non-overlapping floats, and
    :func:`_fsum_finalize` rounds it once.

    Overflow of ``partials`` cannot silently truncate the answer, so it is reported rather
    than absorbed: a full expansion returns ``-1`` and the caller must treat the sum as
    unavailable.  In this kernel's use -- one visit window's worth of rows -- reaching 64
    partials is not possible, which is exactly why a hard failure is the right response if
    it ever happens.
    """

    i = 0
    for j in range(n):
        y = partials[j]
        if abs(x) < abs(y):
            x, y = y, x
        hi = x + y
        lo = y - (hi - x)
        if lo != 0.0:
            if i >= FSUM_MAX_PARTIALS:
                return -1
            partials[i] = lo
            i += 1
        x = hi
    if i >= FSUM_MAX_PARTIALS:
        return -1
    partials[i] = x
    return i + 1


@njit(cache=True, nogil=True)
def _fsum_finalize(partials, n):
    """Round a partial expansion to the nearest float64 -- CPython's ``math_fsum`` tail.

    The trailing half-way correction is not decoration.  Summing the partials naively from
    the top rounds twice in the rare case where the residual is exactly half an ulp, and
    ``math.fsum`` is specified to round once; without this the kernel and the reference
    disagree on precisely the inputs a tie-break is most likely to care about.
    """

    if n <= 0:
        return 0.0
    n -= 1
    hi = partials[n]
    lo = 0.0
    while n > 0:
        x = hi
        n -= 1
        y = partials[n]
        hi = x + y
        yr = hi - x
        lo = y - yr
        if lo != 0.0:
            break
    # Half-way case: the residual `lo` and the next partial agree in sign, so the true sum
    # is further from `hi` than a naive stop would suggest.  Double `lo` and re-add; the
    # result only sticks when it is exactly representable, which is what "round once" means.
    if n > 0 and ((lo < 0.0 and partials[n - 1] < 0.0) or (lo > 0.0 and partials[n - 1] > 0.0)):
        y = lo * 2.0
        x = hi + y
        yr = x - hi
        if y == yr:
            hi = x
    return hi


# ------------------------------------------------------------------------- dual queries


@njit(cache=True, nogil=True)
def _range_sum(series_first, series_start, series_prefix, series, start, stop):
    """Sum one resource's duals over ``[start, stop)``.

    Term for term :meth:`~.dp_prepare.PreparedDuals.range_sum`, which is itself term for
    term ``pricing._PrefixSeries.range_sum``.  Bit-identical because it is the same two
    stored floats subtracted in the same order -- not because the values are close.
    """

    if series < 0:
        return 0.0
    lo_index = series_start[series]
    hi_index = series_start[series + 1]
    length = hi_index - lo_index
    if stop <= start or length <= 1:
        return 0.0
    first = series_first[series]
    series_stop = first + length - 1
    lo = min(max(start, first), series_stop)
    hi = min(max(stop, first), series_stop)
    if hi <= lo:
        return 0.0
    return series_prefix[lo_index + hi - first] - series_prefix[lo_index + lo - first]


@njit(cache=True, nogil=True)
def _visit_cost(
    cell_series, series_first, series_start, series_prefix, cell_index, visit_step, lo, hi
):
    """Every cell-row dual a centre visit charges, in O(1)."""

    if cell_index < 0 or cell_index >= cell_series.shape[0]:
        return 0.0
    return _range_sum(
        series_first,
        series_start,
        series_prefix,
        cell_series[cell_index],
        visit_step + lo,
        visit_step + hi + 1,
    )


@njit(cache=True, nogil=True)
def _row_cost(row_id, row_value, row):
    """Exact stored price of one row id, or zero when the row carries no dual.

    Binary search over the sorted ids rather than a dense lookup: the row space is
    ``n_cells * n_steps`` -- ~5M on a density flight, 40 MB densely, per flight -- while the
    priced rows are a few thousand.  ``[[colgen-density-memory-ceiling]]``.
    """

    if row < 0:
        return 0.0
    lo = 0
    hi = row_id.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if row_id[mid] < row:
            lo = mid + 1
        else:
            hi = mid
    if lo < row_id.shape[0] and row_id[lo] == row:
        return row_value[lo]
    return 0.0


# ------------------------------------------------------------------- forbidden-row test


@njit(cache=True, nogil=True)
def _row_forbidden(bits, row):
    """Whether one row id is in the repair call's forbidden set.

    A bitset over dense row ids, so this is O(1) with no collisions -- the reason
    :mod:`.dp_prepare` interns rows to dense ints at all.  ``forbidden_rows`` is handled
    here rather than by returning to Python because repair is O(flights) inside the greedy,
    and a Python round trip per repair is a scaling cliff at thousands of flights.

    A negative row is *not* forbidden: it is a row outside this flight's numbering, which
    the flight cannot claim, so the question does not arise.
    """

    if row < 0 or bits.shape[0] == 0:
        return False
    word = row >> 6
    if word >= bits.shape[0]:
        return False
    return (bits[word] >> np.uint64(row & 63)) & np.uint64(1) != np.uint64(0)


# ------------------------------------------------------------------------ label compare


@njit(cache=True, nogil=True)
def _fill_path(label, label_parent, label_cell, out):
    """Write a label's cell path root-first into ``out``; return its length.

    Labels are stored as a tree of parent pointers -- the path is never materialized, which
    is what keeps a label to a few words instead of a tuple that grows with its length.
    Lexicographic comparison needs it root-first, so the chain is walked to the root and
    reversed.

    **``out`` must hold ``air_hop_limit + 1`` entries.**  Numba emits no bounds check, so a
    short buffer corrupts neighbouring memory silently rather than raising; the ceiling is
    the only bound on path length, so it is the only correct size.
    """

    n = 0
    node = label
    while node >= 0:
        out[n] = label_cell[node]
        n += 1
        node = label_parent[node]
    for i in range(n >> 1):
        tmp = out[i]
        out[i] = out[n - 1 - i]
        out[n - 1 - i] = tmp
    return n


@njit(cache=True, nogil=True)
def _path_cmp(a, b, label_parent, label_cell, scratch_a, scratch_b):
    """Compare two labels' cell paths lexicographically: -1, 0 or 1.

    Mirrors Python's tuple ordering, including its treatment of a common prefix -- a
    shorter path that is a prefix of a longer one sorts first.  Cells are compared by
    interned index, which :func:`~.dp_prepare.prepare_topology` assigns in **sorted axial
    order**, so index order is the ``Cell`` tuple order the reference compares.

    Only reached on an exact tie of ``(cell, recent, hops, departure_step, lane)``, so its
    O(hops) walk is off the hot path; correctness here decides which of two equally scored
    columns is returned.
    """

    n_a = _fill_path(a, label_parent, label_cell, scratch_a)
    n_b = _fill_path(b, label_parent, label_cell, scratch_b)
    n = min(n_a, n_b)
    for i in range(n):
        if scratch_a[i] < scratch_b[i]:
            return -1
        if scratch_a[i] > scratch_b[i]:
            return 1
    if n_a < n_b:
        return -1
    if n_a > n_b:
        return 1
    return 0


@njit(cache=True, nogil=True)
def _tie_lt(
    a, b, label_hops, label_departure, label_lane, label_parent, label_cell, scratch_a, scratch_b
):
    """``_Label.tie_key(a) < _Label.tie_key(b)`` -- ``(hops, departure_step, lane, path)``.

    ``lane`` is already stored as the reference's ``-1 if origin_lane_idx is None`` form, so
    a laneless start orders below lane 0 exactly as the tuple does.
    """

    if label_hops[a] != label_hops[b]:
        return label_hops[a] < label_hops[b]
    if label_departure[a] != label_departure[b]:
        return label_departure[a] < label_departure[b]
    if label_lane[a] != label_lane[b]:
        return label_lane[a] < label_lane[b]
    return _path_cmp(a, b, label_parent, label_cell, scratch_a, scratch_b) < 0


@njit(cache=True, nogil=True)
def _prefer(
    new,
    old,
    label_score,
    label_hops,
    label_departure,
    label_lane,
    label_parent,
    label_cell,
    scratch_a,
    scratch_b,
):
    """Whether ``new`` displaces ``old`` at the same dominance key -- ``pricing._prefer``.

    ``old < 0`` means the slot is empty.

    Note what this is *not*: a total order.  With scores ``0``, ``0.6e-12`` and ``1.4e-12``
    the first two tie, the last two tie, and the third strictly beats the first, so which
    label survives depends on the order they arrive in.  That is why the kernel reproduces
    the reference's arc order (``AXIAL_NEIGHBORS``) and its roots-before-arcs insertion
    order rather than treating dominance as a set operation.
    """

    if old < 0:
        return True
    if label_score[new] > label_score[old] + SCORE_EPS:
        return True
    if abs(label_score[new] - label_score[old]) > SCORE_EPS:
        return False
    return _tie_lt(
        new,
        old,
        label_hops,
        label_departure,
        label_lane,
        label_parent,
        label_cell,
        scratch_a,
        scratch_b,
    )


# ------------------------------------------------------------------------- state hashing


@njit(cache=True, nogil=True)
def _mix(value, log2cap):
    """Fibonacci-hash a packed key to a starting slot."""

    h = np.uint64(value) * _MAGIC
    return np.int64(h >> np.uint64(64 - log2cap))


@njit(cache=True, nogil=True)
def _state_hash(cell, recent, n_recent, paid_class, first_a, first_b):
    """Hash the reference's dominance key: ``(cell, recent, origin_paid_rows, first_hop)``.

    ``step`` is deliberately absent: it is the *layer*, and the table is layer-local.
    Folding it into the key instead would let two labels at different steps share a slot
    and merge -- the reference keys ``layers[step][key]``, so labels at different steps are
    never compared, and a flat table quietly loses that.

    ``recent`` carries its own length: a label ``k`` hops out has only ``min(k + 1, depth)``
    cells of history, and the reference's tuples of different lengths never compare equal.
    """

    h = np.uint64(cell) * np.uint64(0x100000001B3)
    for i in range(n_recent):
        h = (h ^ np.uint64(recent[i] + 1)) * np.uint64(0x100000001B3)
    h = (h ^ np.uint64(n_recent + 1)) * np.uint64(0x100000001B3)
    h = (h ^ np.uint64(paid_class + 1)) * np.uint64(0x100000001B3)
    h = (h ^ np.uint64(first_a + 1)) * np.uint64(0x100000001B3)
    h = (h ^ np.uint64(first_b + 1)) * np.uint64(0x100000001B3)
    return h


# ------------------------------------------------------------------- label state helpers


@njit(cache=True, nogil=True)
def _fill_recent(label, depth, label_parent, label_cell, out):
    """Write the reference's ``recent`` tuple for one label; return its length.

    ``recent`` is the last ``min(hops + 1, depth)`` cells of the path, most-recent first --
    the reference builds it as ``(neighbour, *recent[:depth - 1])``.  Its LENGTH is part of
    the identity: a label two hops out has a two-cell history, and Python tuples of
    different lengths never compare equal, so a fixed-width buffer would merge states the
    reference keeps apart if the length were dropped.
    """

    n = 0
    node = label
    while node >= 0 and n < depth:
        out[n] = label_cell[node]
        n += 1
        node = label_parent[node]
    return n


@njit(cache=True, nogil=True)
def _recent_cmp(a, n_a, b, n_b):
    """Compare two ``recent`` buffers as Python would compare the tuples: -1, 0 or 1."""

    n = min(n_a, n_b)
    for i in range(n):
        if a[i] < b[i]:
            return -1
        if a[i] > b[i]:
            return 1
    if n_a < n_b:
        return -1
    if n_a > n_b:
        return 1
    return 0


@njit(cache=True, nogil=True)
def _role_allows(roles, first, last):
    """Whether an arc may be traversed in the requested role -- ``hop_allowed_for_role``."""

    if first:
        bit = _ARC_FIRST_LAST if last else _ARC_FIRST
    else:
        bit = _ARC_LAST if last else _ARC_INTERNAL
    return (roles & bit) != 0


@njit(cache=True, nogil=True)
def _layer_lt(
    a, b, depth,
    label_score, label_cell, label_parent, label_hops, label_departure, label_lane,
    recent_a, recent_b, scratch_a, scratch_b,
):
    """The reference's layer iteration order: ``(cell, recent, tie_key)``.

    ``sorted(layer.items(), key=lambda item: (item[0][0], item[0][1], item[1].tie_key))``
    at pricing.py:1522.  This is not cosmetic: relaxation order decides which label lands
    first in the next layer, and ``_prefer`` is non-transitive inside its epsilon band, so
    the surviving label depends on arrival order.
    """

    if label_cell[a] != label_cell[b]:
        return label_cell[a] < label_cell[b]
    n_a = _fill_recent(a, depth, label_parent, label_cell, recent_a)
    n_b = _fill_recent(b, depth, label_parent, label_cell, recent_b)
    order = _recent_cmp(recent_a, n_a, recent_b, n_b)
    if order != 0:
        return order < 0
    return _tie_lt(
        a, b, label_hops, label_departure, label_lane, label_parent, label_cell,
        scratch_a, scratch_b,
    )


@njit(cache=True, nogil=True)
def _sort_layer(
    items, buffer, n, depth,
    label_score, label_cell, label_parent, label_hops, label_departure, label_lane,
    recent_a, recent_b, scratch_a, scratch_b,
):
    """Bottom-up merge sort of a layer's label ids under :func:`_layer_lt`.

    Merge sort rather than the obvious insertion sort because a congested density layer
    holds thousands of labels and O(n^2) there is not survivable; and rather than
    ``np.argsort`` because the ordering key ends in a whole path, which no numeric key can
    encode.  Stability is irrelevant to the result -- ``_layer_lt`` is a strict total order
    on distinct labels, since two labels in one layer cannot share the full key.
    """

    width = 1
    while width < n:
        i = 0
        while i < n:
            mid = min(i + width, n)
            end = min(i + 2 * width, n)
            left = i
            right = mid
            out = i
            while left < mid and right < end:
                if _layer_lt(
                    items[right], items[left], depth,
                    label_score, label_cell, label_parent, label_hops, label_departure,
                    label_lane, recent_a, recent_b, scratch_a, scratch_b,
                ):
                    buffer[out] = items[right]
                    right += 1
                else:
                    buffer[out] = items[left]
                    left += 1
                out += 1
            while left < mid:
                buffer[out] = items[left]
                left += 1
                out += 1
            while right < end:
                buffer[out] = items[right]
                right += 1
                out += 1
            i += 2 * width
        for k in range(n):
            items[k] = buffer[k]
        width *= 2
    return n


@njit(cache=True, nogil=True)
def _state_find(
    slot_label, slot_hash, log2cap, key_hash, depth,
    cell, recent, n_recent, paid_class, first_a, first_b,
    label_cell, label_parent, label_variant, var_paid_class, probe_recent,
):
    """Locate the slot for one dominance key: its occupant, or the first free slot.

    Returns ``(slot, found)``.  ``found`` means the slot holds a label with this exact key;
    otherwise ``slot`` is where an insertion belongs.  ``-1`` means the table is full.

    The slot stores only a label id and a hash -- the key is re-derived from the label on
    each probe rather than stored alongside.  That is what keeps the table to two words per
    slot instead of ``depth + 4``, which matters because the table is per-thread and sized
    to the largest layer of the largest flight.  Re-deriving costs a walk of ``depth``
    parent pointers, and ``depth`` is 2-4.
    """

    cap = 1 << log2cap
    slot = _mix(key_hash, log2cap)
    for _probe in range(cap):
        occupant = slot_label[slot]
        if occupant < 0:
            return slot, False
        if slot_hash[slot] == key_hash and label_cell[occupant] == cell:
            if var_paid_class[label_variant[occupant]] == paid_class:
                n_occ = _fill_recent(occupant, depth, label_parent, label_cell, probe_recent)
                if n_occ == n_recent and _recent_cmp(probe_recent, n_occ, recent, n_recent) == 0:
                    return slot, True
        slot += 1
        if slot >= cap:
            slot = 0
    return -1, False


@njit(cache=True, nogil=True)
def _paid_visit_correction(
    paid_start, paid_cell, paid_step, paid_value, paid_class, cell, visit_step, lo, hi,
    partials,
):
    """Duals this label's origin endpoint already paid inside a later visit window.

    The reference subtracts exactly these (pricing.py:1606) so a row the start option
    already bought is not charged twice.  The CSR slice is sorted by ``(cell, step)`` --
    ``sorted(paid)`` over ``RowKey`` tuples, and cell interning is sorted axial order -- so
    ascending ``step`` here is the same order ``visit_rows`` yields, and the ``fsum`` sees
    the same terms in the same sequence as ``math.fsum`` does.
    """

    start = paid_start[paid_class]
    stop = paid_start[paid_class + 1]
    n = 0
    found = False
    for i in range(start, stop):
        if paid_cell[i] != cell:
            continue
        step = paid_step[i]
        if step < visit_step + lo or step > visit_step + hi:
            continue
        found = True
        n = _fsum_add(partials, n, paid_value[i])
        if n < 0:
            return 0.0, False
    if not found:
        return 0.0, True
    return _fsum_finalize(partials, n), True


@njit(cache=True, nogil=True)
def _visit_hits_forbidden(bits, rows_n_steps, rows_step0, cell, visit_step, lo, hi):
    """``_visit_hits_forbidden``: does any row of this visit window carry an exclusion?

    Row ids are arithmetic (``cell * n_steps + (step - step0)``), so the whole window is a
    contiguous run and no ``RowKey`` is built -- which is the allocation the reference
    removed from its own hot path for the same reason.
    """

    if bits.shape[0] == 0:
        return False
    base = cell * rows_n_steps - rows_step0
    for step in range(visit_step + lo, visit_step + hi + 1):
        offset = step - rows_step0
        if offset < 0 or offset >= rows_n_steps:
            continue
        if _row_forbidden(bits, base + step):
            return True
    return False


def warm_kernel() -> bool:
    """Force compilation of every primitive, so a later timing run measures the search.

    Returns ``True`` once all of them are resident.  Called by the host before a timed
    sweep and by the test suite, since numba's per-process compile would otherwise be
    attributed to the first flight priced.
    """

    partials = np.zeros(FSUM_MAX_PARTIALS, np.float64)
    n = _fsum_add(partials, 0, 1.0)
    _fsum_finalize(partials, n)
    _range_sum(
        np.zeros(1, np.int32), np.zeros(2, np.int64), np.zeros(1, np.float64), 0, 0, 1
    )
    _visit_cost(
        np.zeros(1, np.int32),
        np.zeros(1, np.int32),
        np.zeros(2, np.int64),
        np.zeros(1, np.float64),
        0,
        0,
        0,
        0,
    )
    _row_cost(np.zeros(1, np.int64), np.zeros(1, np.float64), 0)
    _row_forbidden(np.zeros(1, np.uint64), 0)
    scratch_a = np.zeros(4, np.int32)
    scratch_b = np.zeros(4, np.int32)
    parent = np.full(1, -1, np.int32)
    cell = np.zeros(1, np.int32)
    _path_cmp(0, 0, parent, cell, scratch_a, scratch_b)
    zeros_i = np.zeros(1, np.int32)
    _prefer(
        0,
        -1,
        np.zeros(1, np.float64),
        zeros_i,
        zeros_i,
        zeros_i,
        parent,
        cell,
        scratch_a,
        scratch_b,
    )
    recent_a = np.zeros(2, np.int32)
    recent_b = np.zeros(2, np.int32)
    key_hash = np.uint64(_state_hash(0, recent_a, 1, 0, -1, -1))
    _mix(key_hash, 8)
    _fill_recent(0, 2, parent, cell, recent_a)
    _recent_cmp(recent_a, 1, recent_b, 1)
    _role_allows(_ARC_INTERNAL, False, False)
    items = np.zeros(1, np.int32)
    _sort_layer(
        items, np.zeros(1, np.int32), 1, 2,
        np.zeros(1, np.float64), cell, parent, zeros_i, zeros_i, zeros_i,
        recent_a, recent_b, scratch_a, scratch_b,
    )
    _state_find(
        np.full(2, -1, np.int32), np.zeros(2, np.uint64), 1, key_hash, 2,
        0, recent_a, 1, 0, -1, -1, cell, parent, zeros_i, zeros_i, recent_b,
    )
    _paid_visit_correction(
        np.zeros(2, np.int32), zeros_i, zeros_i, np.zeros(1, np.float64),
        0, 0, 0, 0, 0, partials,
    )
    _visit_hits_forbidden(np.zeros(1, np.uint64), 1, 0, 0, 0, 0, 0)
    return True
