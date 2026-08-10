"""Compiled (numba) pricing DP for the column-generation planner.

:func:`pricing._best_column` is the oracle: an exact, dominance-pruned label search over one
flight's space-time DAG, and 95.8% of a colgen solve's wall clock.  This module is that
search over the flat arrays :mod:`.dp_prepare` packs, with the Python reference left
untouched beside it as both the definition of a correct answer and the fallback.

**The primitives come first, and are tested against Python oracles individually,** because
every one of them is a place where being *approximately* right produces a plausible wrong
answer rather than a crash: a different column with the same reduced cost, a label pruned
that should have survived, a tie broken the other way.  None of those raise.

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
``_can_compete`` / ``_sink_may_improve``
    The reference's completion bound, which is a **parity requirement rather than an
    accelerator**: pruned labels have descendants, descendants win dominance slots, and a
    search that prunes less returns a different column.
    ``[[pruning-not-neutral-under-dominance]]``.

**The search pauses, and that is the design.**  ``consider_sink`` improves the reference's
cutoff mid-sweep by canonicalizing a sink -- geometry an ``@njit`` function cannot reach --
and a kernel holding one cutoff per round prunes strictly less, which under dominance
changes the answer rather than merely the work.  So :func:`_price_dag` returns
``STATUS_IMPROVING_SINK`` when a sink's admissible bound could beat the cutoff, the host
certifies it exactly as ``consider_sink`` would, and the search resumes from its saved
position.  Measured on one real search: 27,410 sinks, 6 certifications.

Numba is optional: importing this module raises :class:`ImportError` without it, and the
host is expected to warn once and use the reference (see ``astar_kernel`` for the same
contract).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .dp_prepare import EnvelopeArena

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
STATUS_CANDIDATE_LIMIT = 3  # unused by `_price_dag`; kept so status codes stay stable
STATUS_CANCELLED = 4        # deadline fired mid-search
STATUS_FSUM_OVERFLOW = 5    # a partial expansion saturated -- scores would be wrong
# The PAUSES.  None is a failure: the search saved its position and is waiting for the host
# to do something an `@njit` function cannot -- certify a sink through `column_to_intent`,
# build a completion envelope out of endpoint claim SETS, or take delivery of a full output
# buffer.  Resuming continues from exactly where it stopped; see `_price_dag`'s resume
# record.
STATUS_IMPROVING_SINK = 6   # a sink may beat the cutoff -- host certifies, then resumes
STATUS_NEED_ENVELOPE = 7    # the gate needs a variant's envelope -- host builds, resumes
# The candidate buffer is an OUTPUT, not a search structure, so a full one is drained and
# refilled rather than grown.  Growing it meant re-running the whole search: a real flight
# registers ~27,000 sinks against a 4,096 default, so that was three restarts, each
# throwing away every certification the previous one had paid for.
STATUS_CANDIDATE_FULL = 8

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
    label_cell, label_parent, label_variant, var_paid_class,
    label_first_a, label_first_b, probe_recent,
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
        # Every field of the key is verified, `first_hop` included.  Hashing a field but not
        # checking it makes the table correct only until two keys collide, which is a bug
        # that shows up rarely, on one graph shape, as a merged state and a missing column.
        if (
            slot_hash[slot] == key_hash
            and label_cell[occupant] == cell
            and label_first_a[occupant] == first_a
            and label_first_b[occupant] == first_b
            and var_paid_class[label_variant[occupant]] == paid_class
        ):
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


# ------------------------------------------------------------------- the completion gate


@njit(cache=True, nogil=True)
def _prefix_le(a0, a1, a2, a3, b0, b1, b2, b3):
    """``(a0, a1, a2, a3) <= (b0, b1, b2, b3)`` -- Python's tuple order over four ints."""

    if a0 != b0:
        return a0 < b0
    if a1 != b1:
        return a1 < b1
    if a2 != b2:
        return a2 < b2
    return a3 <= b3


@njit(cache=True, nogil=True)
def _paid_duals(
    label, hops, variant, label_score, var_ground_w, var_origin_leg_w, air_weight, dt_s
):
    """Recover the duals a label has paid by inverting its score (pricing.py:1548).

    The score is the negative sum of the WEIGHTED ground delay and flown time so far, plus
    the de-duplicated duals paid so far.  Duals are already in the master's currency and
    are never weighted, so subtracting the two weighted time terms back out recovers them.

    Term by term, and in this order, NOT ``air_weight * (origin_leg + hops * dt)``: the
    grouped form changes the association and stops being bit-identical.  The reference
    measured 62,673 of 200,000 random draws differing by ~1e-13 between the two, which is
    three orders of magnitude wider than the ``_SCORE_EPS`` band the result is compared in.
    """

    return (
        -label_score[label]
        - var_ground_w[variant]
        - var_origin_leg_w[variant]
        - air_weight * (hops * dt_s)
    )


@njit(cache=True, nogil=True)
def _can_compete(
    variant, min_total_hops, paid_duals, paid_exact,
    env_start, env_len, env_delay, env_dest,
    var_departure, var_lane,
    benefit, pi_f, max_negative_credit, destination_lane_tie,
    cutoff, inc_state,
):
    """``pricing.completion_can_compete``: 1 keep, 0 prune, -1 envelope not built yet.

    **This gate is a parity requirement, not an accelerator.**  Omitting a prune costs
    work and never an answer under pure enumeration, and that reasoning fails here: a
    pruned label still has DESCENDANTS, and those descendants compete for dominance slots.
    Measured on a terminal graph, three labels shared one dominance key at step 21 and the
    slot went to a label the reference never built -- its ancestor pruned by this gate --
    scoring -2271.395 against -2309.408.  It evicted the reference's survivor, whose sinks
    were then never generated.  The answer stayed optimal; the column did not stay the
    same.  ``[[pruning-not-neutral-under-dominance]]``.

    ``-1`` rather than a default is deliberate.  An envelope is frozen at the length the
    live incumbent gave it, and that length is itself a prune (``first_hops >= len``
    returns False), so "not built yet" cannot be answered by guessing either way -- the
    host has to build it against the cutoff that is current *now*.
    """

    if inc_state[0] == 0:
        return 1
    base = env_start[variant]
    if base < 0:
        return -1
    n = env_len[variant]
    first_hops = min_total_hops if min_total_hops > 1 else 1
    if first_hops >= n:
        return 0

    incumbent = cutoff[0]
    origin_lane_tie = var_lane[variant]
    departure_step = var_departure[variant]
    trial = paid_duals if paid_exact else paid_duals - RECOMPUTE_EPS
    paid_positive_lb = trial if trial > 0.0 else 0.0
    for total_hops in range(first_hops, n):
        destination_positive = env_dest[base + total_hops]
        if paid_positive_lb > destination_positive:
            union_positive_lb = paid_positive_lb
        else:
            union_positive_lb = destination_positive
        hop_rc_bound = (
            benefit
            - pi_f
            - env_delay[base + total_hops]
            - union_positive_lb
            + max_negative_credit
        )
        if hop_rc_bound > incumbent + SCORE_EPS:
            return 1
        if (not paid_exact) and hop_rc_bound >= incumbent - RECOMPUTE_EPS:
            # Label scores reconstruct paid duals by cancellation.  Keep the wider
            # numerical band competitive; lexicographic equality pruning is reserved for
            # direct claim sums.
            return 1
        if abs(hop_rc_bound - incumbent) <= SCORE_EPS:
            # The path itself is unknown in this relaxation, so equality in the first four
            # fields may still hide a lexicographically better path.
            if _prefix_le(
                total_hops, departure_step, origin_lane_tie, destination_lane_tie,
                inc_state[1], inc_state[2], inc_state[3], inc_state[4],
            ):
                return 1
    return 0


@njit(cache=True, nogil=True)
def _sink_may_improve(
    variant, hops, paid_duals,
    env_start, env_len, env_delay, env_dest,
    benefit, pi_f, max_negative_credit, cutoff, inc_state,
):
    """Whether the host must be asked to certify this sink: 1 ask, 0 skip, -1 no envelope.

    ``consider_sink`` decides this on the sink's *provisional* reduced cost, which needs
    ``_path_delay_s`` -- and that reaches ``fold_corners_to_columns`` and a
    ``np.linalg.norm(...).sum()`` whose pairwise summation numba does not reproduce.  So
    the kernel cannot decide it; it can only decide whether it is worth **asking**.

    The screen is the same completion bound the label gate uses, evaluated at the sink's
    exact hop count, which upper-bounds its reduced cost.  Being loose only costs a round
    trip -- the host then applies the reference's own test verbatim, so the answer is the
    reference's either way.  Being tight in the wrong direction would silently drop the
    mid-sweep incumbent update, so the comparison uses ``_RECOMPUTE_EPS``, the band the
    reference itself uses when a bound is being compared against a directly summed score.

    Measured on one real search: 27,410 sinks registered, ``_canonical_candidate`` called
    6 times -- 0.02%.  Asking per sink would be the whole of the cost this kernel exists
    to remove; asking per *improvement* is 6 round trips.
    """

    if inc_state[0] == 0:
        # No incumbent yet, so `consider_sink` certifies every sink it sees.
        return 1
    base = env_start[variant]
    if base < 0:
        return -1
    n = env_len[variant]
    if hops < 1 or hops >= n:
        return 0
    trial = paid_duals - RECOMPUTE_EPS
    paid_positive_lb = trial if trial > 0.0 else 0.0
    destination_positive = env_dest[base + hops]
    if paid_positive_lb > destination_positive:
        union_positive_lb = paid_positive_lb
    else:
        union_positive_lb = destination_positive
    bound = (
        benefit - pi_f - env_delay[base + hops] - union_positive_lb + max_negative_credit
    )
    if bound >= cutoff[0] - RECOMPUTE_EPS:
        return 1
    return 0


@njit(cache=True, nogil=True)
def _register_sinks(
    nxt, arrival_step, d_start, skip_first_append,
    dest_lane_start, dest_lane_idx,
    label_cell, label_hops, label_variant, label_score,
    var_ground_w, var_origin_leg_w, air_weight, dt_s,
    env_start, env_len, env_delay, env_dest,
    benefit, pi_f, max_negative_credit, cutoff, inc_state, sink_probe,
    cand_label, cand_lane, cand_step, n_cand,
):
    """Register one sink label's destination lanes; return ``(n_cand, code, lane_slot)``.

    Codes: 0 the whole lane range is done, 1 pause to certify at ``lane_slot``, 2 pause to
    build ``lane_slot``'s envelope, 3 the output buffer is full at ``lane_slot``.

    Factored out of the arc loop because it is entered from **two** places -- fresh, and
    again on resume after a pause -- and duplicating a loop that can itself pause is how
    a lane gets registered twice or skipped.  ``skip_first_append`` is what distinguishes
    the two resume flavours: after a certify or buffer round trip the append for
    ``d_start`` has not happened yet; after an envelope round trip it already did.

    Every sink is appended, exactly as the reference appends every one to ``candidates``.
    Tier 2 is the only thing that ranks, so a full buffer is something to empty and never a
    decision about which proposal deserves to survive.

    ``sink_probe`` off means no certifier is attached, so there is nobody to answer an
    improving sink and the screen is skipped entirely.
    """

    neighbour = label_cell[nxt]
    hops = label_hops[nxt]
    variant = label_variant[nxt]
    paid = _paid_duals(
        nxt, hops, variant, label_score, var_ground_w, var_origin_leg_w, air_weight, dt_s
    )
    cand_cap = cand_label.shape[0]
    skip = skip_first_append
    for d in range(d_start, dest_lane_start[neighbour + 1]):
        if not skip:
            if n_cand >= cand_cap:
                return n_cand, 3, d
            cand_label[n_cand] = nxt
            cand_lane[n_cand] = dest_lane_idx[d]
            cand_step[n_cand] = arrival_step
            n_cand += 1
        skip = False
        if sink_probe == 0:
            continue
        verdict = _sink_may_improve(
            variant, hops, paid, env_start, env_len, env_delay, env_dest,
            benefit, pi_f, max_negative_credit, cutoff, inc_state,
        )
        if verdict < 0:
            return n_cand, 2, d
        if verdict > 0:
            return n_cand, 1, d
    return n_cand, 0, d_start


@njit(cache=True, nogil=True)
def _seed_layer(
    seed_step, min_step, root_order, root_bucket_start,
    var_cell, var_score, var_paid_class, var_departure, var_lane,
    n_labels, label_score, label_cell, label_parent, label_hops, label_variant,
    label_departure, label_lane, label_first_a, label_first_b,
    tbl_label, tbl_hash, log2cap, depth, recent_a, probe_recent, scratch_a, scratch_b,
):
    """Insert every root whose start step is ``seed_step``; return the new label count.

    Negative returns are failures: ``-1`` label pool full, ``-2`` state table saturated.

    Roots go in through the same ``_prefer`` path as arcs because the reference inserts
    them into the very same per-layer dict, so two roots colliding on one dominance key
    resolve against each other exactly as two arcs would.
    """

    n_buckets = root_bucket_start.shape[0] - 1
    bucket = seed_step - min_step
    if bucket < 0 or bucket >= n_buckets:
        return n_labels
    pool_cap = label_score.shape[0]
    for k in range(root_bucket_start[bucket], root_bucket_start[bucket + 1]):
        variant = root_order[k]
        if n_labels >= pool_cap:
            return -1
        label = n_labels
        n_labels += 1
        cell = var_cell[variant]
        paid_class = var_paid_class[variant]
        label_score[label] = var_score[variant]
        label_cell[label] = cell
        label_parent[label] = -1
        label_hops[label] = 0
        label_variant[label] = variant
        label_departure[label] = var_departure[variant]
        label_lane[label] = var_lane[variant]
        label_first_a[label] = -1
        label_first_b[label] = -1
        recent_a[0] = cell
        key_hash = _state_hash(cell, recent_a, 1, paid_class, -1, -1)
        slot, found = _state_find(
            tbl_label, tbl_hash, log2cap, key_hash, depth,
            cell, recent_a, 1, paid_class, -1, -1,
            label_cell, label_parent, label_variant, var_paid_class,
            label_first_a, label_first_b, probe_recent,
        )
        if slot < 0:
            return -2
        if not found:
            tbl_label[slot] = label
            tbl_hash[slot] = key_hash
        elif _prefer(
            label, tbl_label[slot], label_score, label_hops, label_departure,
            label_lane, label_parent, label_cell, scratch_a, scratch_b,
        ):
            tbl_label[slot] = label
    return n_labels


@njit(cache=True, nogil=True)
def _price_dag(
    # --- topology
    arc_start, arc_target, arc_roles, hex_remaining,
    dest_mask, dest_lane_start, dest_lane_idx,
    air_hop_limit, revisit_depth, state_history_depth, track_first_hop,
    min_step, max_step,
    # --- roots, pre-bucketed by start step
    root_order, root_bucket_start,
    var_cell, var_score, var_paid_class, var_departure, var_lane,
    var_ground_w, var_origin_leg_w,
    paid_start, paid_cell, paid_step, paid_value,
    # --- duals
    cell_series, series_first, series_start, series_prefix, offsets_lo, offsets_hi,
    # --- rows and exclusions
    forbidden_bits, rows_n_steps, rows_step0,
    # --- objective
    air_dt_s, air_weight, dt_s, benefit, pi_f, max_negative_credit,
    # --- the completion gate
    env_start, env_len, env_delay, env_dest, destination_lane_tie, cutoff, inc_state,
    sink_probe,
    # --- workspace: label pool
    label_score, label_cell, label_parent, label_hops, label_variant,
    label_departure, label_lane, label_first_a, label_first_b,
    # --- workspace: two layer-local state tables
    tbl_label_a, tbl_hash_a, tbl_label_b, tbl_hash_b, log2cap,
    # --- workspace: scratch
    layer_items, layer_buffer, recent_a, recent_b, probe_recent,
    scratch_a, scratch_b, partials,
    # --- workspace: candidates
    cand_label, cand_lane, cand_step,
    # --- control
    cancel, out_counts, resume, out_sink,
):
    """The reference's Tier 1 over flat arrays -- ``pricing._best_column``'s label search.

    Returns a status code and writes ``(n_labels, n_candidates)`` into ``out_counts``.  Only
    ``STATUS_OK`` means the search ran to completion; ``STATUS_IMPROVING_SINK`` and
    ``STATUS_NEED_ENVELOPE`` are pauses to be resumed, and every other code is a budget the
    host must widen, or a fallback to the reference.

    **Layer discipline.**  Two tables are swapped rather than one keyed by step, because the
    reference keys ``layers[step][key]`` and therefore never compares labels at different
    steps; a flat table would merge them.  Roots for layer ``s + 1`` are seeded at the START
    of step ``s``, before any arc can write there -- the reference inserts every root before
    any arc runs, and ``_prefer`` is non-transitive inside its epsilon band, so
    roots-before-arcs is part of the answer rather than an implementation detail.

    **The mid-sweep incumbent, and why this function pauses.**  ``consider_sink`` assigns to
    a ``nonlocal incumbent``, so the reference's cutoff IMPROVES DURING its sweep and every
    later layer prunes against the better one.  A kernel holding one cutoff per round prunes
    strictly less, and under dominance that is not merely slower: extra labels win dominance
    slots and evict the reference's survivors, whose sinks are then never generated.  So the
    cutoff has to track the reference's exactly.

    Certifying a sink needs ``column_to_intent`` and ``_path_delay_s``, which an ``@njit``
    function cannot call -- but it can **return**.  Measured on one real search: 27,410 sinks
    registered, ``_canonical_candidate`` called 6 times.  Those differ by a factor of 4,500,
    which is what makes "pause per improvement" a different proposition from "pause per
    sink".

    **The resume record** (caller-owned, so the host only has to hand it back)::

        resume[0]  mode: 0 fresh, 1 resume the lane loop appending, 2 resume it skipping
                   the first append, 3 re-run the label preamble
        resume[1]  step        resume[4]  lane slot within `dest_lane_start`
        resume[2]  layer index resume[5]  the sink label id
        resume[3]  arc index   resume[6]  the sorted layer's size

    Which table is current is NOT saved: the two swap once per step, so it is the parity of
    ``step - min_step``.  Deriving it removes a way for a resumed call to disagree with the
    call it continues.

    **Order within an arc is deliberately not the reference's.**  The dominance insert runs
    BEFORE the sinks, where ``_best_column`` runs it after.  Neither reads what the other
    writes -- ``consider_sink`` touches only ``candidates`` and ``incumbent``, the insert
    only the next layer -- and ``_Candidate.tie_key`` is a strict total order, so the
    certification order cannot reach the answer either.  What it buys is that a pause can
    only ever happen with the arc's mutations already complete, so resuming never has to
    decide whether a label was allocated twice.
    """

    cap = 1 << log2cap
    pool_cap = label_score.shape[0]
    depth = state_history_depth

    mode = resume[0]
    if mode == 0:
        n_labels = 0
        n_cand = 0
        m = 0
        step_from = min_step
        idx_from = 0
        arc_from = 0
        d_from = 0
        nxt_from = -1
        for slot in range(cap):
            tbl_label_a[slot] = -1
            tbl_label_b[slot] = -1
        out_counts[0] = 0
        out_counts[1] = 0
        # The first layer is seeded before the loop; every later one is seeded by the
        # iteration BEFORE it, so that roots are always in place before any arc can write
        # there.  Folding both into the loop body is what dropped the roots at
        # `min_step + 1` in an earlier draft -- an off-by-one that only bites when a start
        # step happens to land there.
        n_labels = _seed_layer(
            min_step, min_step, root_order, root_bucket_start,
            var_cell, var_score, var_paid_class, var_departure, var_lane,
            n_labels, label_score, label_cell, label_parent, label_hops, label_variant,
            label_departure, label_lane, label_first_a, label_first_b,
            tbl_label_a, tbl_hash_a, log2cap, depth, recent_a, probe_recent,
            scratch_a, scratch_b,
        )
        if n_labels == -1:
            out_counts[2] = min_step - 1  # no layer was relaxed; the host cannot extrapolate
            return STATUS_LABEL_LIMIT
        if n_labels == -2:
            return STATUS_STATE_LIMIT
    else:
        n_labels = out_counts[0]
        n_cand = out_counts[1]
        step_from = resume[1]
        idx_from = resume[2]
        arc_from = resume[3]
        d_from = resume[4]
        nxt_from = resume[5]
        m = resume[6]
    pending = mode
    resume[0] = 0

    for step in range(step_from, max_step + 1):
        if cancel[0] != 0:
            out_counts[0] = n_labels
            out_counts[1] = n_cand
            return STATUS_CANCELLED

        # Table identity by parity rather than by a saved flag: they swap exactly once per
        # step, so `cur` is A on even offsets from `min_step` and B on odd ones.  Deriving
        # it means a resumed call cannot disagree with the call it continues.
        #
        # Only the CURRENT layer's labels are read (to gather and sort it) and only the NEXT
        # layer's hashes are written, so there is no `cur_hash`.  The stale hashes the
        # previous swap left behind are unreachable: `_state_find` tests `slot_label` before
        # it trusts `slot_hash`, and clearing the labels is what retires the slot.
        if ((step - min_step) & 1) == 0:
            cur_label = tbl_label_a
            nxt_label = tbl_label_b
            nxt_hash = tbl_hash_b
        else:
            cur_label = tbl_label_b
            nxt_label = tbl_label_a
            nxt_hash = tbl_hash_a

        if pending == 0:
            n_labels = _seed_layer(
                step + 1, min_step, root_order, root_bucket_start,
                var_cell, var_score, var_paid_class, var_departure, var_lane,
                n_labels, label_score, label_cell, label_parent, label_hops, label_variant,
                label_departure, label_lane, label_first_a, label_first_b,
                nxt_label, nxt_hash, log2cap, depth, recent_a, probe_recent,
                scratch_a, scratch_b,
            )
            if n_labels == -1:
                out_counts[1] = n_cand
                out_counts[2] = step
                return STATUS_LABEL_LIMIT
            if n_labels == -2:
                out_counts[1] = n_cand
                return STATUS_STATE_LIMIT

            # --- gather and order the current layer -------------------------------------
            m = 0
            for slot in range(cap):
                if cur_label[slot] >= 0:
                    layer_items[m] = cur_label[slot]
                    m += 1
            if m > 1:
                _sort_layer(
                    layer_items, layer_buffer, m, depth,
                    label_score, label_cell, label_parent, label_hops, label_departure,
                    label_lane, recent_a, recent_b, scratch_a, scratch_b,
                )
            idx_start = 0
        else:
            # Resuming into this step: it is already seeded and `layer_items` already
            # sorted.  Re-sorting would be harmless but re-seeding would not.
            idx_start = idx_from

        for idx in range(idx_start, m):
            label = layer_items[idx]
            hops = label_hops[label]
            variant = label_variant[label]
            cell = label_cell[label]
            a_start = arc_start[cell]
            run_preamble = True

            if pending == 1 or pending == 2:
                # Finish the lane loop the previous call stopped inside.  The arc's label
                # and dominance insert are already done, so only the sinks remain.
                nxt = nxt_from
                n_cand, code, lane_slot = _register_sinks(
                    nxt, step + 1, d_from, pending == 2,
                    dest_lane_start, dest_lane_idx,
                    label_cell, label_hops, label_variant, label_score,
                    var_ground_w, var_origin_leg_w, air_weight, dt_s,
                    env_start, env_len, env_delay, env_dest,
                    benefit, pi_f, max_negative_credit, cutoff, inc_state, sink_probe,
                    cand_label, cand_lane, cand_step, n_cand,
                )
                if code != 0:
                    out_counts[0] = n_labels
                    out_counts[1] = n_cand
                    resume[1] = step
                    resume[2] = idx
                    resume[3] = arc_from
                    resume[5] = nxt
                    resume[6] = m
                    resume[4] = lane_slot
                    out_sink[0] = label_variant[nxt]
                    if code == 2:
                        resume[0] = 2
                        return STATUS_NEED_ENVELOPE
                    resume[0] = 1
                    if code == 3:
                        return STATUS_CANDIDATE_FULL
                    resume[4] = lane_slot + 1
                    out_sink[1] = nxt
                    out_sink[2] = dest_lane_idx[lane_slot]
                    out_sink[3] = step + 1
                    out_sink[4] = label_hops[nxt]
                    return STATUS_IMPROVING_SINK
                a_start = arc_from + 1
                run_preamble = False
                pending = 0
            elif pending == 3:
                # The preamble stopped for an envelope and mutated nothing, so it simply
                # runs again -- this time with the envelope in place.
                pending = 0

            if hops >= air_hop_limit:
                continue
            if step + 1 > max_step:
                continue
            if run_preamble and inc_state[0] != 0:
                # The endpoint-aware envelope lower-bounds the positive price of the
                # eventual row union without double-counting overlaps, and handles exact RC
                # ties in the same hops-first order as candidates.  Run once per label, and
                # NOT re-run after a resume: the reference evaluates it once at the top and
                # keeps relaxing the label's arcs even if `consider_sink` improves the
                # incumbent underneath it.
                paid = _paid_duals(
                    label, hops, variant, label_score, var_ground_w, var_origin_leg_w,
                    air_weight, dt_s,
                )
                verdict = _can_compete(
                    variant, hops + hex_remaining[cell], paid, False,
                    env_start, env_len, env_delay, env_dest,
                    var_departure, var_lane,
                    benefit, pi_f, max_negative_credit, destination_lane_tie,
                    cutoff, inc_state,
                )
                if verdict < 0:
                    out_counts[0] = n_labels
                    out_counts[1] = n_cand
                    resume[0] = 3
                    resume[1] = step
                    resume[2] = idx
                    resume[6] = m
                    out_sink[0] = variant
                    return STATUS_NEED_ENVELOPE
                if verdict == 0:
                    continue

            paid_class = var_paid_class[variant]
            n_recent = _fill_recent(label, depth, label_parent, label_cell, recent_a)
            first_arc = hops == 0
            ban = min(n_recent, revisit_depth)

            for a in range(a_start, arc_start[cell + 1]):
                neighbour = arc_target[a]
                banned = False
                for j in range(ban):
                    if recent_a[j] == neighbour:
                        banned = True
                        break
                if banned:
                    continue
                roles = arc_roles[a]
                finish_allowed = dest_mask[neighbour] != 0 and _role_allows(
                    roles, first_arc, True
                )
                continuation_allowed = _role_allows(roles, first_arc, False)
                if not finish_allowed and not continuation_allowed:
                    continue
                distance_to_go = hex_remaining[neighbour]
                next_step = step + 1
                if next_step + distance_to_go > max_step:
                    continue
                if hops + 1 + distance_to_go > air_hop_limit:
                    continue
                if _visit_hits_forbidden(
                    forbidden_bits, rows_n_steps, rows_step0, neighbour, next_step,
                    offsets_lo, offsets_hi,
                ):
                    continue
                visit_cost = _visit_cost(
                    cell_series, series_first, series_start, series_prefix,
                    neighbour, next_step, offsets_lo, offsets_hi,
                )
                correction, ok = _paid_visit_correction(
                    paid_start, paid_cell, paid_step, paid_value, paid_class,
                    neighbour, next_step, offsets_lo, offsets_hi, partials,
                )
                if not ok:
                    out_counts[0] = n_labels
                    out_counts[1] = n_cand
                    return STATUS_FSUM_OVERFLOW
                visit_cost -= correction

                if n_labels >= pool_cap:
                    out_counts[0] = n_labels
                    out_counts[1] = n_cand
                    # How far the search got before the pool filled.  The host needs this to
                    # size the retry: doubling from a default that is 200x too small costs
                    # eight restarts, and every one of them throws away the certifications
                    # the previous attempt already paid for.
                    out_counts[2] = step
                    return STATUS_LABEL_LIMIT
                nxt = n_labels
                n_labels += 1
                label_score[nxt] = label_score[label] - air_dt_s - visit_cost
                label_cell[nxt] = neighbour
                label_parent[nxt] = label
                label_hops[nxt] = hops + 1
                label_variant[nxt] = variant
                label_departure[nxt] = label_departure[label]
                label_lane[nxt] = label_lane[label]
                if track_first_hop and label_first_a[label] < 0:
                    label_first_a[nxt] = cell
                    label_first_b[nxt] = neighbour
                else:
                    label_first_a[nxt] = label_first_a[label]
                    label_first_b[nxt] = label_first_b[label]

                if continuation_allowed:
                    # `next_recent = (neighbour, *recent[:depth - 1])`
                    recent_b[0] = neighbour
                    n_next = 1
                    while n_next < depth and n_next - 1 < n_recent:
                        recent_b[n_next] = recent_a[n_next - 1]
                        n_next += 1
                    key_hash = _state_hash(
                        neighbour, recent_b, n_next, paid_class,
                        label_first_a[nxt], label_first_b[nxt],
                    )
                    slot, found = _state_find(
                        nxt_label, nxt_hash, log2cap, key_hash, depth,
                        neighbour, recent_b, n_next, paid_class,
                        label_first_a[nxt], label_first_b[nxt],
                        label_cell, label_parent, label_variant, var_paid_class,
                        label_first_a, label_first_b, probe_recent,
                    )
                    if slot < 0:
                        out_counts[0] = n_labels
                        out_counts[1] = n_cand
                        return STATUS_STATE_LIMIT
                    if not found:
                        nxt_label[slot] = nxt
                        nxt_hash[slot] = key_hash
                    elif _prefer(
                        nxt, nxt_label[slot], label_score, label_hops, label_departure,
                        label_lane, label_parent, label_cell, scratch_a, scratch_b,
                    ):
                        nxt_label[slot] = nxt

                if finish_allowed:
                    n_cand, code, lane_slot = _register_sinks(
                        nxt, next_step, dest_lane_start[neighbour], False,
                        dest_lane_start, dest_lane_idx,
                        label_cell, label_hops, label_variant, label_score,
                        var_ground_w, var_origin_leg_w, air_weight, dt_s,
                        env_start, env_len, env_delay, env_dest,
                        benefit, pi_f, max_negative_credit, cutoff, inc_state, sink_probe,
                        cand_label, cand_lane, cand_step, n_cand,
                    )
                    if code != 0:
                        out_counts[0] = n_labels
                        out_counts[1] = n_cand
                        resume[1] = step
                        resume[2] = idx
                        resume[3] = a
                        resume[5] = nxt
                        resume[6] = m
                        resume[4] = lane_slot
                        out_sink[0] = variant
                        if code == 2:
                            resume[0] = 2
                            return STATUS_NEED_ENVELOPE
                        resume[0] = 1
                        if code == 3:
                            return STATUS_CANDIDATE_FULL
                        resume[4] = lane_slot + 1
                        out_sink[1] = nxt
                        out_sink[2] = dest_lane_idx[lane_slot]
                        out_sink[3] = next_step
                        out_sink[4] = hops + 1
                        return STATUS_IMPROVING_SINK

        # The consumed layer becomes the layer AFTER next's write target, so it is cleared
        # here -- which is also what keeps the very first iteration from relaxing into a
        # table it never emptied.
        for slot in range(cap):
            cur_label[slot] = -1

    out_counts[0] = n_labels
    out_counts[1] = n_cand
    return STATUS_OK


# ------------------------------------------------------- the feasible (min-delay) search

# `find_feasible_column`'s outcomes.  It shares the budget codes above so a host can treat
# "widen and retry" identically for both searches.
STATUS_SINK = 9        # a sink reached a destination -- host certifies, then resumes


@njit(cache=True, nogil=True)
def _delay_lower_bound(
    ground_delay_s, origin_fold_s, hops, remaining_hops, destination_fold_s,
    reference_time_s, dt_s, folding_exact, ground_weight, air_weight,
):
    """``pricing._arc_delay_lower_bound_s`` composed with ``CostModel.evaluate``.

    Written as the three-term sum ``w_g*ground + w_a*hold + w_a*detour`` with ``hold`` zero,
    rather than the two terms that survive, because ``evaluate`` sums in that order and this
    value is compared against an incumbent delay inside a ``_RECOMPUTE_EPS`` band.  The
    grouped form is a different association and so is not bit-identical.

    A terminal lane inside its fold radius invalidates the arc decomposition -- folding can
    then drop a hop without increasing canonical flown distance -- and the safe fallback is
    the irrevocable ground delay.  The same fallback covers a zero reference, because
    ``enroute_detour_m`` deliberately defines its detour as zero there.
    """

    if (not folding_exact) or reference_time_s <= 0.0:
        return ground_weight * ground_delay_s + air_weight * 0.0 + air_weight * 0.0
    flown_time_lb = origin_fold_s + (hops + remaining_hops) * dt_s + destination_fold_s
    detour = flown_time_lb - reference_time_s
    if not (detour > 0.0):
        detour = 0.0
    return ground_weight * ground_delay_s + air_weight * 0.0 + air_weight * detour


@njit(cache=True, nogil=True)
def _frontier_lt(
    a, b, lab_bound, lab_estimate, lab_departure, lab_lane, lab_serial,
    label_parent, label_cell, scratch_a, scratch_b,
):
    """The reference's frontier order: ``(bound, hops+remaining, departure, lane, path, serial)``.

    Every field matters and the last two are not decoration.  ``path`` is compared
    lexicographically because two labels can tie on all four numeric fields, and ``serial``
    -- the push counter -- is what makes the order total, so a heap cannot reorder equal
    keys and change which path is expanded first.
    """

    if lab_bound[a] != lab_bound[b]:
        return lab_bound[a] < lab_bound[b]
    if lab_estimate[a] != lab_estimate[b]:
        return lab_estimate[a] < lab_estimate[b]
    if lab_departure[a] != lab_departure[b]:
        return lab_departure[a] < lab_departure[b]
    if lab_lane[a] != lab_lane[b]:
        return lab_lane[a] < lab_lane[b]
    order = _path_cmp(a, b, label_parent, label_cell, scratch_a, scratch_b)
    if order != 0:
        return order < 0
    return lab_serial[a] < lab_serial[b]


@njit(cache=True, nogil=True)
def _heap_push(
    heap, n, item, lab_bound, lab_estimate, lab_departure, lab_lane, lab_serial,
    label_parent, label_cell, scratch_a, scratch_b,
):
    """Sift up.  Returns the new size, or -1 when the frontier array is full."""

    if n >= heap.shape[0]:
        return -1
    heap[n] = item
    child = n
    while child > 0:
        parent = (child - 1) >> 1
        if _frontier_lt(
            heap[child], heap[parent], lab_bound, lab_estimate, lab_departure, lab_lane,
            lab_serial, label_parent, label_cell, scratch_a, scratch_b,
        ):
            tmp = heap[parent]
            heap[parent] = heap[child]
            heap[child] = tmp
            child = parent
        else:
            break
    return n + 1


@njit(cache=True, nogil=True)
def _heap_pop(
    heap, n, lab_bound, lab_estimate, lab_departure, lab_lane, lab_serial,
    label_parent, label_cell, scratch_a, scratch_b,
):
    """Sift down.  Returns ``(item, new_size)``."""

    top = heap[0]
    n -= 1
    heap[0] = heap[n]
    parent = 0
    while True:
        left = 2 * parent + 1
        if left >= n:
            break
        smallest = left
        right = left + 1
        if right < n and _frontier_lt(
            heap[right], heap[left], lab_bound, lab_estimate, lab_departure, lab_lane,
            lab_serial, label_parent, label_cell, scratch_a, scratch_b,
        ):
            smallest = right
        if _frontier_lt(
            heap[smallest], heap[parent], lab_bound, lab_estimate, lab_departure, lab_lane,
            lab_serial, label_parent, label_cell, scratch_a, scratch_b,
        ):
            tmp = heap[parent]
            heap[parent] = heap[smallest]
            heap[smallest] = tmp
            parent = smallest
        else:
            break
    return top, n


@njit(cache=True, nogil=True)
def _feasible_state_hash(step, cell, recent, n_recent, departure_step, lane, first_a, first_b):
    """Hash ``(step, cell, recent, departure_step, lane, first_hop)``.

    A DIFFERENT key from the priced search's, and deliberately so.  That one is layer-local
    and carries ``origin_paid_rows``; this one is global -- best-first jumps between steps,
    so ``step`` has to be inside the key rather than implied by the table.
    """

    h = np.uint64(step + 1) * np.uint64(0x100000001B3)
    h = (h ^ np.uint64(cell + 1)) * np.uint64(0x100000001B3)
    for i in range(n_recent):
        h = (h ^ np.uint64(recent[i] + 1)) * np.uint64(0x100000001B3)
    h = (h ^ np.uint64(n_recent + 1)) * np.uint64(0x100000001B3)
    h = (h ^ np.uint64(departure_step + 1)) * np.uint64(0x100000001B3)
    h = (h ^ np.uint64(lane + 2)) * np.uint64(0x100000001B3)
    h = (h ^ np.uint64(first_a + 1)) * np.uint64(0x100000001B3)
    h = (h ^ np.uint64(first_b + 1)) * np.uint64(0x100000001B3)
    return h


@njit(cache=True, nogil=True)
def _feasible_state_find(
    slot_label, slot_hash, log2cap, key_hash, depth,
    step, cell, recent, n_recent, departure_step, lane, first_a, first_b,
    lab_step, label_cell, label_parent, label_departure, label_lane,
    label_first_a, label_first_b, probe_recent,
):
    """Locate the slot for one feasible-search state: ``(slot, found)``, ``-1`` when full.

    Every field of the key is verified on probe, not merely hashed.  Hashing a field and
    then trusting the hash makes the table correct only until two keys collide, which shows
    up rarely, on one graph shape, as a path silently dropped.
    """

    cap = 1 << log2cap
    slot = _mix(key_hash, log2cap)
    for _probe in range(cap):
        occupant = slot_label[slot]
        if occupant < 0:
            return slot, False
        if (
            slot_hash[slot] == key_hash
            and lab_step[occupant] == step
            and label_cell[occupant] == cell
            and label_departure[occupant] == departure_step
            and label_lane[occupant] == lane
            and label_first_a[occupant] == first_a
            and label_first_b[occupant] == first_b
        ):
            n_occ = _fill_recent(occupant, depth, label_parent, label_cell, probe_recent)
            if n_occ == n_recent and _recent_cmp(probe_recent, n_occ, recent, n_recent) == 0:
                return slot, True
        slot += 1
        if slot >= cap:
            slot = 0
    return -1, False


@njit(cache=True, nogil=True)
def _feasible_dag(
    # --- topology
    arc_start, arc_target, hex_remaining,
    dest_mask, dest_lane_start,
    air_hop_limit, revisit_depth, state_history_depth, track_first_hop,
    max_step,
    # --- roots, already filtered by the host in the reference's own order
    root_cell, root_step, root_departure, root_lane, root_bound, root_remaining,
    # --- delay bound
    lane_fold_s, lane_fold_exact, destination_fold_lb, reference_time_s, dt_s,
    ground_weight, air_weight, base_step,
    # --- exclusions
    forbidden_bits, rows_n_steps, rows_step0, offsets_lo, offsets_hi,
    # --- incumbent (value, valid-flag); the early exit is the host's call
    incumbent,
    # --- workspace: label pool
    label_cell, label_parent, label_hops, label_departure, label_lane,
    label_first_a, label_first_b,
    lab_step, lab_bound, lab_estimate, lab_serial,
    # --- workspace: frontier and state table
    heap, tbl_label, tbl_hash, log2cap,
    recent_a, recent_b, probe_recent, scratch_a, scratch_b,
    # --- control
    cancel, out_counts, resume, out_sink,
):
    """``pricing.find_feasible_column``'s search over flat arrays.

    A **best-first** search, not the layered DP of :func:`_price_dag`, and the difference is
    structural rather than cosmetic: the frontier is a priority queue ordered by an
    admissible delay bound, so it jumps between time layers and its dominance table has to
    be global with ``step`` inside the key.  Sharing ``_price_dag``'s outer loop was
    considered and is not possible; the arc guards are what the two have in common.

    **What it keeps per state is a PATH, not a score.**  ``best_state_path`` stores the
    lexicographically smallest path seen for each state and refuses anything not strictly
    smaller.  That is a different rule from ``_prefer``, and mixing the two up produces a
    search that is still optimal and still returns a different column.

    **Why it pauses.**  Every sink is judged by ``_canonical_candidate``, which reaches
    ``column_to_intent`` and the whole geometry stack -- so the kernel returns
    ``STATUS_SINK``, the host certifies exactly as the reference does, updates ``incumbent``
    and resumes.  Measured on a density flight: 141,553 arcs relaxed against 115
    certifications, which is the ratio that makes this worth doing at all.

    ``improve_below`` is the greedy's early exit: the FIRST certified strict improvement may
    be returned, which is what makes this an incumbent heuristic rather than an oracle. The
    host signals it with ``STATUS_IMPROVED``; the kernel never decides it, because deciding
    it needs the certified delay.

    The resume record is ``resume[0]`` mode (0 fresh, 1 continue the lane loop), ``[1]`` the
    popped label, ``[2]`` the lane slot, ``[3]`` the frontier size, ``[4]`` the label count,
    ``[5]`` the serial counter.
    """

    depth = state_history_depth
    pool_cap = label_cell.shape[0]
    cap = 1 << log2cap

    if resume[0] == 0:
        n_labels = 0
        n_heap = 0
        serial = 0
        for slot in range(cap):
            tbl_label[slot] = -1
        # Seed the frontier.  The host already applied every start guard the reference
        # applies, in its order, so this loop only has to preserve that order.
        for r in range(root_cell.shape[0]):
            if n_labels >= pool_cap:
                return STATUS_LABEL_LIMIT
            label = n_labels
            n_labels += 1
            cell = root_cell[r]
            label_cell[label] = cell
            label_parent[label] = -1
            label_hops[label] = 0
            label_departure[label] = root_departure[r]
            label_lane[label] = root_lane[r]
            label_first_a[label] = -1
            label_first_b[label] = -1
            lab_step[label] = root_step[r]
            lab_bound[label] = root_bound[r]
            lab_estimate[label] = root_remaining[r]
            lab_serial[label] = serial
            serial += 1
            recent_a[0] = cell
            key_hash = _feasible_state_hash(
                root_step[r], cell, recent_a, 1, root_departure[r], root_lane[r], -1, -1
            )
            slot, found = _feasible_state_find(
                tbl_label, tbl_hash, log2cap, key_hash, depth,
                root_step[r], cell, recent_a, 1, root_departure[r], root_lane[r], -1, -1,
                lab_step, label_cell, label_parent, label_departure, label_lane,
                label_first_a, label_first_b, probe_recent,
            )
            if slot < 0:
                return STATUS_STATE_LIMIT
            tbl_label[slot] = label
            tbl_hash[slot] = key_hash
            n_heap = _heap_push(
                heap, n_heap, label, lab_bound, lab_estimate, label_departure, label_lane,
                lab_serial, label_parent, label_cell, scratch_a, scratch_b,
            )
            if n_heap < 0:
                return STATUS_CANDIDATE_LIMIT
        popped = -1
        lane_from = 0
    else:
        n_heap = resume[3]
        n_labels = resume[4]
        serial = resume[5]
        popped = resume[1]
        lane_from = resume[2]
    pending = resume[0]
    resume[0] = 0

    while True:
        if pending == 0:
            if cancel[0] != 0:
                out_counts[0] = n_labels
                return STATUS_CANCELLED
            if n_heap == 0:
                break
            popped, n_heap = _heap_pop(
                heap, n_heap, lab_bound, lab_estimate, label_departure, label_lane,
                lab_serial, label_parent, label_cell, scratch_a, scratch_b,
            )
            if incumbent[1] != 0.0 and lab_bound[popped] > incumbent[0] + RECOMPUTE_EPS:
                break
            step = lab_step[popped]
            cell = label_cell[popped]
            n_recent = _fill_recent(popped, depth, label_parent, label_cell, recent_a)
            key_hash = _feasible_state_hash(
                step, cell, recent_a, n_recent, label_departure[popped],
                label_lane[popped], label_first_a[popped], label_first_b[popped],
            )
            slot, found = _feasible_state_find(
                tbl_label, tbl_hash, log2cap, key_hash, depth,
                step, cell, recent_a, n_recent, label_departure[popped],
                label_lane[popped], label_first_a[popped], label_first_b[popped],
                lab_step, label_cell, label_parent, label_departure, label_lane,
                label_first_a, label_first_b, probe_recent,
            )
            if slot < 0:
                out_counts[0] = n_labels
                return STATUS_STATE_LIMIT
            # `best_state_path.get(state_key) != path`.  Two labels sharing this key AND a
            # path are the same label, since the key already carries departure and lane --
            # so comparing ids is comparing paths.
            if (not found) or tbl_label[slot] != popped:
                continue
            lane_from = dest_lane_start[cell]
        else:
            step = lab_step[popped]
            cell = label_cell[popped]
            n_recent = _fill_recent(popped, depth, label_parent, label_cell, recent_a)
            pending = 0

        hops = label_hops[popped]

        # --- sinks: every destination lane is judged by the host ------------------------
        if hops >= 1 and dest_mask[cell] != 0:
            if lane_from < dest_lane_start[cell + 1]:
                out_counts[0] = n_labels
                resume[0] = 1
                resume[1] = popped
                resume[2] = lane_from + 1
                resume[3] = n_heap
                resume[4] = n_labels
                resume[5] = serial
                out_sink[0] = popped
                out_sink[1] = lane_from
                out_sink[2] = step
                out_sink[3] = hops
                return STATUS_SINK

        if hops >= air_hop_limit:
            continue
        if step + 1 > max_step:
            continue

        # --- relax --------------------------------------------------------------------
        departure_step = label_departure[popped]
        lane = label_lane[popped]
        ground_delay_s = (departure_step - base_step) * dt_s
        fold_s = lane_fold_s[lane + 1]
        exact = lane_fold_exact[lane + 1] != 0
        ban = min(n_recent, revisit_depth)
        next_step = step + 1
        for a in range(arc_start[cell], arc_start[cell + 1]):
            neighbour = arc_target[a]
            banned = False
            for j in range(ban):
                if recent_a[j] == neighbour:
                    banned = True
                    break
            if banned:
                continue
            remaining = hex_remaining[neighbour]
            if next_step + remaining > max_step:
                continue
            if hops + 1 + remaining > air_hop_limit:
                continue
            if _visit_hits_forbidden(
                forbidden_bits, rows_n_steps, rows_step0, neighbour, next_step,
                offsets_lo, offsets_hi,
            ):
                continue
            next_bound = _delay_lower_bound(
                ground_delay_s, fold_s, hops + 1, remaining, destination_fold_lb,
                reference_time_s, dt_s, exact, ground_weight, air_weight,
            )
            if incumbent[1] != 0.0 and next_bound > incumbent[0] + RECOMPUTE_EPS:
                continue

            if n_labels >= pool_cap:
                out_counts[0] = n_labels
                return STATUS_LABEL_LIMIT
            nxt = n_labels
            label_cell[nxt] = neighbour
            label_parent[nxt] = popped
            label_hops[nxt] = hops + 1
            label_departure[nxt] = departure_step
            label_lane[nxt] = lane
            if track_first_hop and label_first_a[popped] < 0:
                label_first_a[nxt] = cell
                label_first_b[nxt] = neighbour
            else:
                label_first_a[nxt] = label_first_a[popped]
                label_first_b[nxt] = label_first_b[popped]
            lab_step[nxt] = next_step

            recent_b[0] = neighbour
            n_next = 1
            while n_next < depth and n_next - 1 < n_recent:
                recent_b[n_next] = recent_a[n_next - 1]
                n_next += 1
            key_hash = _feasible_state_hash(
                next_step, neighbour, recent_b, n_next, departure_step, lane,
                label_first_a[nxt], label_first_b[nxt],
            )
            slot, found = _feasible_state_find(
                tbl_label, tbl_hash, log2cap, key_hash, depth,
                next_step, neighbour, recent_b, n_next, departure_step, lane,
                label_first_a[nxt], label_first_b[nxt],
                lab_step, label_cell, label_parent, label_departure, label_lane,
                label_first_a, label_first_b, probe_recent,
            )
            if slot < 0:
                out_counts[0] = n_labels
                return STATUS_STATE_LIMIT
            if found:
                # `previous_path is not None and previous_path <= next_path: continue`
                if _path_cmp(
                    tbl_label[slot], nxt, label_parent, label_cell, scratch_a, scratch_b
                ) <= 0:
                    continue
            n_labels += 1
            lab_bound[nxt] = next_bound
            lab_estimate[nxt] = hops + 1 + remaining
            lab_serial[nxt] = serial
            serial += 1
            tbl_label[slot] = nxt
            tbl_hash[slot] = key_hash
            n_heap = _heap_push(
                heap, n_heap, nxt, lab_bound, lab_estimate, label_departure, label_lane,
                lab_serial, label_parent, label_cell, scratch_a, scratch_b,
            )
            if n_heap < 0:
                out_counts[0] = n_labels
                return STATUS_CANDIDATE_LIMIT

    out_counts[0] = n_labels
    return STATUS_OK


class DagResult:
    """One compiled search's proposals, in the host's terms."""

    __slots__ = (
        "status", "n_labels", "candidates", "paths", "incumbent", "attempts", "budget",
    )

    def __init__(
        self, status, n_labels, candidates, paths, incumbent=None, attempts=1, budget=None
    ):
        self.status = status
        self.n_labels = n_labels
        # `(label_capacity, log2cap, candidate_capacity)` this run settled on.  Handing it
        # back to the next call for the same graph is what stops every colgen iteration
        # from re-discovering the same 13.3M-label pool by restarting into it.
        self.budget = budget
        # How many times the search ran from its first layer.  More than one means a budget
        # was exhausted and everything before it was thrown away -- including the
        # certifications that had already been paid for -- so this is the number to read
        # when a flight is unexpectedly expensive.
        self.attempts = attempts
        # ``(departure_step, origin_lane, dest_lane, arrival_step, label_index)`` per sink.
        self.candidates = candidates
        # ``label_index -> tuple of cell indices``, root-first.
        self.paths = paths
        # The incumbent the mid-sweep certifications left behind, in the reference's
        # ``(reduced_cost, Column)`` shape -- ``_best_column``'s ``incumbent`` at the point
        # its arc loop ends, which Tier 2 then starts its ranking from.
        self.incumbent = incumbent

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


def _root_buckets(variants, topology, rows_unused=None):
    """Order root variants by start step, preserving the reference's insertion order.

    Within one start step the order must stay ``(departure_step, lane)`` ascending, which is
    the order ``prepare_variants`` emits and the order ``_best_column``'s two nested loops
    insert -- hence a STABLE sort. ``_prefer`` is non-transitive, so two roots colliding on
    one dominance key resolve differently if they arrive in the other order.
    """

    # `prepare_variants` already resolved this, applying the reference's start guards in the
    # process, so recomputing it here would be a second chance to disagree.
    start_steps = variants.start_step.astype(np.int64)
    span = topology.max_step - topology.min_step + 2
    keep = (start_steps >= topology.min_step) & (start_steps < topology.min_step + span)
    order = np.argsort(np.where(keep, start_steps, np.iinfo(np.int64).max), kind="stable")
    order = order[keep[order]].astype(np.int32)
    bucket_start = np.zeros(span + 1, dtype=np.int32)
    counts = np.bincount(
        (start_steps[order] - topology.min_step).astype(np.int64), minlength=span
    )
    bucket_start[1:] = np.cumsum(counts[:span])
    return order, bucket_start


def _next_label_capacity(capacity, step_reached, min_step, max_step):
    """Size the retry pool from how far the filled one got, not from doubling alone.

    Doubling is the safe policy and a slow one: a density flight builds 13.3M labels
    against a 65,536 default, which is eight restarts, and every restart discards the sink
    certifications the previous attempt already paid for.  The kernel reports the step it
    reached, so the host can extrapolate instead.

    Deliberately crude, and hedged in both directions.  Labels per step are far from
    uniform -- the frontier widens for a while and then plateaus -- so an estimate taken
    early reads low; the 1.25 margin and the doubling FLOOR cover that, and a second
    attempt extrapolates from a later step and lands closer.  The 8x ceiling is what stops
    a pathological early fill from asking for gigabytes: at 44 bytes a label, 8x of a
    13.3M pool is already 4.7 GB.
    """

    span = max_step - min_step + 1
    progress = step_reached - min_step + 1
    doubled = capacity * 2
    if progress <= 0 or span <= 0:
        return doubled
    estimate = int(capacity * 1.25 * span / progress)
    return max(doubled, min(estimate, capacity * 8))


def price_dag(
    topology,
    rows,
    duals,
    variants,
    forbidden,
    *,
    air_weight: float,
    dt_s: float,
    benefit: float = 0.0,
    pi_f: float = 0.0,
    envelopes=None,
    certify=None,
    label_capacity: int = 1 << 16,
    log2cap: int = 14,
    candidate_capacity: int = 1 << 12,
    cancel=None,
    max_attempts: int = 12,
):
    """Run :func:`_price_dag` to completion, servicing its pauses and its budgets.

    Two loops, and they are not interchangeable.  The **inner** one services pauses: the
    search saved its position and wants a sink certified or an envelope built, both of
    which are Python, after which it resumes exactly where it stopped.  The **outer** one
    services budgets: a full label pool or a saturated dominance table means the search
    never finished, so it is re-run from the first layer with more room.

    A budget restart therefore has to restore the *starting* conditions, not merely the
    starting arrays.  The cutoff goes back to the incumbent this call was given and the
    envelope memo is rewound, because an envelope frozen against a mid-sweep incumbent
    would make the second attempt prune more than the first -- and the reference's column
    is defined by a search that never restarted.

    Budgets are grown rather than guessed because label counts vary by orders of magnitude
    between flights; an exact fit would re-allocate on nearly every one, which is the waste
    ``[[colgen-parallel-pricing-pool]]`` measured at 82%.
    """

    if certify is not None and envelopes is None:
        raise ValueError(
            "a sink certifier updates the cutoff mid-sweep, and the completion gate cannot "
            "be evaluated against a cutoff it has no envelopes for; pass `envelopes` too"
        )

    order, bucket_start = _root_buckets(variants, topology)
    if cancel is None:
        cancel = np.zeros(1, dtype=np.uint8)
    out_counts = np.zeros(3, dtype=np.int64)
    resume = np.zeros(8, dtype=np.int64)
    out_sink = np.zeros(8, dtype=np.int64)
    cutoff = np.zeros(1, dtype=np.float64)
    inc_state = np.zeros(5, dtype=np.int64)
    depth = max(1, topology.state_history_depth)
    hop_scratch = max(2, topology.air_hop_limit + 2)
    air_dt_s = air_weight * dt_s
    n_variants = variants.n_variants
    sink_probe = 1 if certify is not None else 0
    destination_lane_tie = (
        int(topology.dest_lane_idx.min()) if topology.dest_lane_idx.size else -1
    )
    # `-1` is how `PreparedVariants` spells "no origin lane"; `CompletionEnvelopes` keys on
    # the reference's `None`, so the two are translated once here rather than at every use.
    lane_of = [None if value < 0 else int(value) for value in variants.lane_idx.tolist()]
    departures = [int(value) for value in variants.departure_step.tolist()]
    cell_q = topology.cell_q.tolist()
    cell_r = topology.cell_r.tolist()

    initial_incumbent = None if envelopes is None else envelopes.incumbent
    initial_keys = () if envelopes is None else envelopes.built_keys()

    def _load_arena():
        """Mirror whatever the root gate already froze into arrays the kernel indexes."""

        arena = EnvelopeArena(n_variants)
        # Every variant that SURVIVED `prepare_variants` had `can_compete` answer True
        # against this incumbent, and the reference builds the full envelope in that call --
        # so building them here is reproducing the reference, not anticipating the kernel.
        # With no incumbent the reference builds nothing at root time, and neither does this.
        if envelopes is not None and envelopes.incumbent is not None:
            for variant in range(n_variants):
                key = departures[variant], lane_of[variant]
                arena.add(variant, *envelopes.envelope(*key))
        return arena

    def _publish(incumbent):
        if incumbent is None:
            inc_state[0] = 0
            cutoff[0] = 0.0
            return
        column = incumbent[1]
        cutoff[0] = incumbent[0]
        inc_state[0] = 1
        inc_state[1] = len(column.cell_path) - 1
        inc_state[2] = column.departure_step
        inc_state[3] = -1 if column.origin_lane_idx is None else column.origin_lane_idx
        inc_state[4] = -1 if column.dest_lane_idx is None else column.dest_lane_idx

    incumbent = initial_incumbent
    arena = _load_arena()
    status = STATUS_OK
    paths: dict[int, tuple[int, ...]] = {}
    candidates: list[tuple[int, int, int, int, int]] = []

    for attempt in range(max_attempts):
        cap = 1 << log2cap
        label_score = np.zeros(label_capacity, np.float64)
        label_cell = np.zeros(label_capacity, np.int32)
        label_parent = np.full(label_capacity, -1, np.int32)
        label_hops = np.zeros(label_capacity, np.int32)
        label_variant = np.zeros(label_capacity, np.int32)
        label_departure = np.zeros(label_capacity, np.int32)
        label_lane = np.zeros(label_capacity, np.int32)
        label_first_a = np.full(label_capacity, -1, np.int32)
        label_first_b = np.full(label_capacity, -1, np.int32)
        cand_label = np.zeros(candidate_capacity, np.int32)
        cand_lane = np.zeros(candidate_capacity, np.int32)
        cand_step = np.zeros(candidate_capacity, np.int32)
        tbl_label_a = np.full(cap, -1, np.int32)
        tbl_hash_a = np.zeros(cap, np.uint64)
        tbl_label_b = np.full(cap, -1, np.int32)
        tbl_hash_b = np.zeros(cap, np.uint64)
        layer_items = np.zeros(max(cap, 1), np.int32)
        layer_buffer = np.zeros(max(cap, 1), np.int32)
        recent_a = np.zeros(depth, np.int32)
        recent_b = np.zeros(depth, np.int32)
        probe_recent = np.zeros(depth, np.int32)
        scratch_a = np.zeros(hop_scratch, np.int32)
        scratch_b = np.zeros(hop_scratch, np.int32)
        partials = np.zeros(FSUM_MAX_PARTIALS, np.float64)

        def _drain():
            """Copy the filled part of the output buffer out, so it can be refilled."""

            for i in range(int(out_counts[1])):
                label = int(cand_label[i])
                if label not in paths:
                    chain = []
                    node = label
                    while node >= 0:
                        chain.append(int(label_cell[node]))
                        node = int(label_parent[node])
                    chain.reverse()
                    paths[label] = tuple(chain)
                candidates.append(
                    (
                        int(label_departure[label]),
                        int(label_lane[label]),
                        int(cand_lane[i]),
                        int(cand_step[i]),
                        label,
                    )
                )
            out_counts[1] = 0

        resume[0] = 0
        paths.clear()
        candidates.clear()
        if attempt:
            # Restart from the conditions the first attempt began with, not from the ones
            # it reached.  See the docstring: a mid-sweep envelope is a stronger prune.
            incumbent = initial_incumbent
            if envelopes is not None:
                envelopes.rewind(initial_keys)
            arena = _load_arena()
        _publish(incumbent)

        while True:
            status = _price_dag(
                topology.arc_start, topology.arc_target, topology.arc_roles,
                topology.hex_remaining, topology.dest_mask, topology.dest_lane_start,
                topology.dest_lane_idx,
                topology.air_hop_limit, topology.revisit_depth, depth,
                bool(topology.track_first_hop), topology.min_step, topology.max_step,
                order, bucket_start,
                variants.cell, variants.score, variants.paid_class,
                variants.departure_step, variants.lane_idx,
                variants.ground_delay_s, variants.origin_leg_w_s,
                variants.paid_start, variants.paid_cell, variants.paid_step,
                variants.paid_value,
                duals.cell_series, duals.series_first, duals.series_start,
                duals.series_prefix, duals.offsets_lo, duals.offsets_hi,
                forbidden.bits, rows.n_steps, rows.step0,
                air_dt_s, float(air_weight), float(dt_s), float(benefit), float(pi_f),
                duals.max_negative_credit,
                arena.start, arena.length, arena.delay, arena.dest,
                destination_lane_tie, cutoff, inc_state, sink_probe,
                label_score, label_cell, label_parent, label_hops, label_variant,
                label_departure, label_lane, label_first_a, label_first_b,
                tbl_label_a, tbl_hash_a, tbl_label_b, tbl_hash_b, log2cap,
                layer_items, layer_buffer, recent_a, recent_b, probe_recent,
                scratch_a, scratch_b, partials,
                cand_label, cand_lane, cand_step,
                cancel, out_counts, resume, out_sink,
            )
            if status == STATUS_IMPROVING_SINK:
                label = int(out_sink[1])
                path = []
                node = label
                while node >= 0:
                    path.append(int(label_cell[node]))
                    node = int(label_parent[node])
                path.reverse()
                lane = int(out_sink[2])
                certified = certify(
                    incumbent,
                    int(label_departure[label]),
                    None if label_lane[label] < 0 else int(label_lane[label]),
                    None if lane < 0 else lane,
                    int(out_sink[3]),
                    tuple((cell_q[c], cell_r[c]) for c in path),
                )
                if certified is not None:
                    incumbent = certified
                    envelopes.set_incumbent(certified)
                    _publish(incumbent)
                continue
            if status == STATUS_NEED_ENVELOPE:
                variant = int(out_sink[0])
                key = departures[variant], lane_of[variant]
                arena.add(variant, *envelopes.envelope(*key))
                continue
            if status == STATUS_CANDIDATE_FULL:
                _drain()
                continue
            break

        if status == STATUS_LABEL_LIMIT:
            label_capacity = _next_label_capacity(
                label_capacity, int(out_counts[2]), topology.min_step, topology.max_step
            )
            continue
        if status == STATUS_STATE_LIMIT:
            log2cap += 1
            continue
        break

    _drain()
    return DagResult(
        status, int(out_counts[0]), candidates, paths, incumbent, attempt + 1,
        (label_capacity, log2cap, candidate_capacity),
    )


def feasible_dag(
    topology,
    rows,
    forbidden,
    roots,
    *,
    lane_fold_s,
    lane_fold_exact,
    destination_fold_lb: float,
    reference_time_s: float,
    dt_s: float,
    ground_weight: float,
    air_weight: float,
    base_step: int,
    offsets,
    incumbent_delay: float | None = None,
    certify=None,
    label_capacity: int = 1 << 16,
    log2cap: int = 15,
    heap_capacity: int = 1 << 15,
    cancel=None,
    max_attempts: int = 12,
):
    """Run :func:`_feasible_dag`, servicing its sink pauses and growing its budgets.

    ``roots`` is the reference's start loop already evaluated, in its order: the kernel
    reproduces the search, not the guards, because those need endpoint claim SETS and the
    reference's own early ``break`` on the incumbent's delay.

    ``certify(departure_step, origin_lane, dest_lane, step, hops, path)`` is
    ``find_feasible_column``'s per-sink block -- canonicalize, compare on ``column_key``,
    adopt -- and returns ``(new_incumbent_delay_or_None, stop)``. All of that stays in
    :mod:`.pricing` because it is the reference's semantics; the kernel only decides *when*
    to ask.

    Returns ``(status, stopped_early)``. ``status == STATUS_OK`` means the frontier drained
    or the bound cut it off, which is what licenses using the result.
    """

    root_cell = np.asarray([r[0] for r in roots], np.int32)
    root_step = np.asarray([r[1] for r in roots], np.int32)
    root_departure = np.asarray([r[2] for r in roots], np.int32)
    root_lane = np.asarray([r[3] for r in roots], np.int32)
    root_bound = np.asarray([r[4] for r in roots], np.float64)
    root_remaining = np.asarray([r[5] for r in roots], np.int32)

    if cancel is None:
        cancel = np.zeros(1, dtype=np.uint8)
    out_counts = np.zeros(3, dtype=np.int64)
    resume = np.zeros(8, dtype=np.int64)
    out_sink = np.zeros(8, dtype=np.int64)
    incumbent = np.zeros(2, dtype=np.float64)
    depth = max(1, topology.state_history_depth)
    hop_scratch = max(2, topology.air_hop_limit + 2)
    cell_q = topology.cell_q.tolist()
    cell_r = topology.cell_r.tolist()
    lane_fold_s = np.asarray(lane_fold_s, np.float64)
    lane_fold_exact = np.asarray(lane_fold_exact, np.uint8)
    initial_delay = incumbent_delay
    stopped_early = False
    status = STATUS_OK

    for _attempt in range(max_attempts):
        label_cell = np.zeros(label_capacity, np.int32)
        label_parent = np.full(label_capacity, -1, np.int32)
        label_hops = np.zeros(label_capacity, np.int32)
        label_departure = np.zeros(label_capacity, np.int32)
        label_lane = np.zeros(label_capacity, np.int32)
        label_first_a = np.full(label_capacity, -1, np.int32)
        label_first_b = np.full(label_capacity, -1, np.int32)
        lab_step = np.zeros(label_capacity, np.int32)
        lab_bound = np.zeros(label_capacity, np.float64)
        lab_estimate = np.zeros(label_capacity, np.int32)
        lab_serial = np.zeros(label_capacity, np.int64)
        cap = 1 << log2cap
        heap = np.zeros(heap_capacity, np.int32)
        tbl_label = np.full(cap, -1, np.int32)
        tbl_hash = np.zeros(cap, np.uint64)

        resume[0] = 0
        stopped_early = False
        # A restart re-runs from the first root, so it must re-run against the incumbent the
        # first attempt began with.  A delay the abandoned attempt certified is a tighter
        # bound, and the reference's answer is defined by a search that never restarted.
        if initial_delay is None:
            incumbent[0] = 0.0
            incumbent[1] = 0.0
        else:
            incumbent[0] = initial_delay
            incumbent[1] = 1.0

        while True:
            status = _feasible_dag(
                topology.arc_start, topology.arc_target, topology.hex_remaining,
                topology.dest_mask, topology.dest_lane_start,
                topology.air_hop_limit, topology.revisit_depth, depth,
                bool(topology.track_first_hop), topology.max_step,
                root_cell, root_step, root_departure, root_lane, root_bound, root_remaining,
                lane_fold_s, lane_fold_exact, float(destination_fold_lb),
                float(reference_time_s), float(dt_s), float(ground_weight),
                float(air_weight), int(base_step),
                forbidden.bits, rows.n_steps, rows.step0,
                int(offsets[0]), int(offsets[1]),
                incumbent,
                label_cell, label_parent, label_hops, label_departure, label_lane,
                label_first_a, label_first_b,
                lab_step, lab_bound, lab_estimate, lab_serial,
                heap, tbl_label, tbl_hash, log2cap,
                np.zeros(depth, np.int32), np.zeros(depth, np.int32),
                np.zeros(depth, np.int32),
                np.zeros(hop_scratch, np.int32), np.zeros(hop_scratch, np.int32),
                cancel, out_counts, resume, out_sink,
            )
            if status != STATUS_SINK:
                break
            label = int(out_sink[0])
            path = []
            node = label
            while node >= 0:
                path.append(int(label_cell[node]))
                node = int(label_parent[node])
            path.reverse()
            dest_lane = int(topology.dest_lane_idx[int(out_sink[1])])
            new_delay, stop = certify(
                int(label_departure[label]),
                None if label_lane[label] < 0 else int(label_lane[label]),
                # `-1` is how the packing spells "no destination lane"; the geometry refuses
                # anything but `None` for a non-terminal endpoint.
                None if dest_lane < 0 else dest_lane,
                int(out_sink[2]),
                int(out_sink[3]),
                tuple((cell_q[c], cell_r[c]) for c in path),
            )
            if new_delay is not None:
                incumbent[0] = new_delay
                incumbent[1] = 1.0
            if stop:
                stopped_early = True
                status = STATUS_OK
                break

        if status == STATUS_LABEL_LIMIT:
            label_capacity *= 2
            continue
        if status == STATUS_STATE_LIMIT:
            log2cap += 1
            continue
        if status == STATUS_CANDIDATE_LIMIT:
            heap_capacity *= 2
            continue
        break

    return status, stopped_early


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
        0, recent_a, 1, 0, -1, -1, cell, parent, zeros_i, zeros_i,
        parent, parent, recent_b,
    )
    _paid_visit_correction(
        np.zeros(2, np.int32), zeros_i, zeros_i, np.zeros(1, np.float64),
        0, 0, 0, 0, 0, partials,
    )
    _visit_hits_forbidden(np.zeros(1, np.uint64), 1, 0, 0, 0, 0, 0)
    _prefix_le(0, 0, 0, 0, 0, 0, 0, 0)  # noqa: FURB120 - forcing a compile, not computing
    ones_f = np.ones(1, np.float64)
    _paid_duals(0, 0, 0, ones_f, ones_f, ones_f, 1.0, 1.0)
    env_start = np.zeros(1, np.int32)
    env_len = np.ones(1, np.int32)
    cutoff = np.zeros(1, np.float64)
    inc_state = np.zeros(5, np.int64)
    _can_compete(
        0, 1, 0.0, False, env_start, env_len, ones_f, ones_f, zeros_i, zeros_i,
        0.0, 0.0, 0.0, -1, cutoff, inc_state,
    )
    _sink_may_improve(
        0, 1, 0.0, env_start, env_len, ones_f, ones_f, 0.0, 0.0, 0.0, cutoff, inc_state,
    )
    _register_sinks(
        0, 0, 0, False, np.zeros(2, np.int32), zeros_i,
        cell, zeros_i, zeros_i, np.zeros(1, np.float64),
        ones_f, ones_f, 1.0, 1.0,
        env_start, env_len, ones_f, ones_f,
        0.0, 0.0, 0.0, cutoff, inc_state, 1,
        zeros_i, zeros_i, zeros_i, 0,
    )
    return True
