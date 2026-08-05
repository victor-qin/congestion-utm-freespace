"""Compiled label-setting search for the column-generation pricing subproblem.

``pricing._best_column`` stays the reference oracle and the fallback.  This kernel
replaces only its *label machinery* -- the part measurement showed the time
actually lives in once dual pricing was moved onto prefix sums:

* ``layers: dict[int, dict[key, _Label]]``, whose key is a 4-tuple holding a
  ``frozenset`` and two tuples and is rehashed on every probe, becomes one
  open-addressed table over flat arrays;
* ``_Label``, one frozen-dataclass allocation per surviving arc, becomes a row in
  parallel arrays;
* ``(*label.path, neighbour)``, an O(hops) tuple copy on *every* arc relaxation,
  becomes a single ``label_parent`` back-pointer.

What the kernel does NOT own: continuous wall geometry, filing budgets,
de-duplicated claims, and exact delay.  It returns *candidate* sinks ranked by an
admissible upper bound on reduced cost; ``pricing`` rebuilds every one through
``_canonical_candidate``, which remains the sole authority, and the reported
reduced cost is always the exact Python one.

Numba is imported unconditionally, matching ``astar_kernel``; the availability
guard lives at the host import site in ``pricing`` (see ``astar.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numba import njit

# Status codes.  ``FB_`` values mean the host must finish the job itself; there is
# deliberately no "numba unavailable" code, because the host guard short-circuits
# before any kernel call and it would be unreachable.
OK = 0
NO_PATH = 1
FB_LABEL_OVERFLOW = 2
FB_HASH_FULL = 3
FB_CANCELLED = 4

STATUS_NAMES = MappingProxyType(
    {
        OK: "ok",
        NO_PATH: "no_path",
        FB_LABEL_OVERFLOW: "fallback_label_overflow",
        FB_HASH_FULL: "fallback_hash_full",
        FB_CANCELLED: "cancelled",
    }
)

# Must match pricing._SCORE_EPS / _RECOMPUTE_EPS exactly: the tie band decides
# which of two equal-score labels survives, and the reference's choice is pinned
# by committed tests.
SCORE_EPS = 1e-12
RECOMPUTE_EPS = 1e-8

_MAGIC = np.uint64(0x9E3779B97F4A7C15)  # Fibonacci hashing multiplier, as astar_kernel

# Headroom over ``n_cells * n_steps`` when resizing the label pool after an overflow.
# The state key carries more than (cell, step) -- the revisit history, the origin's
# paid-row class and the first arc -- so several labels can share a cell-step and the
# product is a floor, not a bound.
#
# The multiplier grows with scenario size: 1.03-1.58x at 100 flights, 2.1-7.5x at 500,
# because more flights means more priced rows -- more origin paid-row classes and weaker
# duals -- and both widen the dominance key's effective span.  A captured 500-flight
# failure (fid=3176) needed 32,274,881 labels against cells*steps = 8,156,869: 3.96x.
#
# 2.0 nonetheless stays, because the CEILING was the actual bug and raising it alone
# suffices.  Measured on that flight at label_limit_max = 1<<25:
#     safety 2.0 -> 32.5s (2 regrowths)   safety 4.0 -> 23.0s (1 regrowth)
# Matching the observed ratio saves 9.5s on straggler flights but costs +59% peak RSS on
# EVERY flight (100-flight run: 6,656 -> 10,566 MB tree peak), because it over-allocates
# wherever the ratio is genuinely ~1.3.  Peak memory is what bounds worker count, so the
# extra regrowth is the cheaper side of that trade.
_LABEL_GEOMETRY_SAFETY = 2.0

# Arc role bits, mirroring network.py.
ARC_INTERNAL = np.uint8(1 << 0)
ARC_FIRST = np.uint8(1 << 1)
ARC_LAST = np.uint8(1 << 2)
ARC_FIRST_LAST = np.uint8(1 << 3)


@njit(cache=True, nogil=True)
def _mix(value, log2cap):
    """Fibonacci-mix a packed key into a slot index."""

    return np.int64((np.uint64(value) * _MAGIC) >> np.uint64(64 - log2cap))


# Must match dp_prepare._FORBIDDEN_STEP_OFFSET / _SPAN exactly -- the host packs the
# table, the kernel probes it, and a mismatch would silently miss every exclusion.
_FORBIDDEN_STEP_OFFSET = 1 << 20
_FORBIDDEN_STEP_SPAN = 1 << 21


@njit(cache=True, nogil=True)
def _visit_forbidden(
    cell, visit_step, window_lo, window_hi, forbidden_slots, forbidden_log2cap, forbidden_n
):
    """True if any row in this cell visit's window is excluded.

    Mirrors ``pricing._visit_hits_forbidden``: same window as ``visit_rows`` -- the
    inclusive ``[visit_step + lo, visit_step + hi]`` range -- and the same first-hit exit.
    """

    if forbidden_n == 0:
        return False
    mask = np.int64(forbidden_slots.shape[0] - 1)
    for offset in range(window_lo, window_hi + 1):
        key = (
            np.int64(cell) * np.int64(_FORBIDDEN_STEP_SPAN)
            + np.int64(visit_step + offset)
            + np.int64(_FORBIDDEN_STEP_OFFSET)
        )
        slot = _mix(key, forbidden_log2cap)
        while True:
            held = forbidden_slots[slot]
            if held == -1:
                break
            if held == key:
                return True
            slot = (slot + np.int64(1)) & mask
    return False


@njit(cache=True, nogil=True)
def _window_cost(
    cell,
    visit_step,
    dual_first,
    dual_start,
    dual_prefix,
    window_lo,
    window_hi,
    paid_class,
    paid_start,
    paid_cell,
    paid_step,
    paid_value,
):
    """Price one cell visit's row window, net of rows the origin endpoint paid.

    The prefix arithmetic reproduces ``_PrefixSeries.range_sum`` clamp for clamp;
    ``tests/test_colgen_solver.py`` pins that identity against the explicit
    ``RowKey`` sum in pure Python.
    """

    lo_off = dual_start[cell]
    length = dual_start[cell + 1] - lo_off
    total = 0.0
    if length > 1:
        first = dual_first[cell]
        series_stop = first + length - 1
        start = visit_step + window_lo
        stop = visit_step + window_hi + 1
        a = start
        if a < first:
            a = first
        if a > series_stop:
            a = series_stop
        b = stop
        if b < first:
            b = first
        if b > series_stop:
            b = series_stop
        if b > a:
            total = dual_prefix[lo_off + b - first] - dual_prefix[lo_off + a - first]

    # Subtract any window row already charged at the origin endpoint.  Entries are
    # sorted by (cell, step); the sets are tiny, so a bounded scan beats a search.
    ps = paid_start[paid_class]
    pe = paid_start[paid_class + 1]
    if pe > ps:
        lo_step = visit_step + window_lo
        hi_step = visit_step + window_hi
        for p in range(ps, pe):
            if paid_cell[p] == cell:
                s = paid_step[p]
                if lo_step <= s <= hi_step:
                    total -= paid_value[p]
            elif paid_cell[p] > cell:
                break
    return total


@njit(cache=True, nogil=True)
def _delay_lower_bound(
    ground_delay_s,
    origin_fold_s,
    total_hops,
    destination_fold_s,
    reference_time_s,
    dt_s,
    folding_exact,
):
    """Admissible delay for every completion, mirroring _arc_delay_lower_bound_s."""

    if (not folding_exact) or reference_time_s <= 0.0:
        return ground_delay_s
    flown = origin_fold_s + total_hops * dt_s + destination_fold_s
    excess = flown - reference_time_s
    if excess < 0.0:
        excess = 0.0
    return ground_delay_s + excess


@njit(cache=True, nogil=True)
def _path_cmp(a, b, label_parent, label_cell):
    """Lexicographic compare of two equal-length label paths, in one backward pass.

    Lexicographic order is decided by the FRONT-most difference, but back-pointers
    only walk backwards.  Overwriting ``diff`` at every difference leaves the last
    write holding the front-most one, so a single O(hops) pass with O(1) memory
    settles it.

    Cell ids are assigned in sorted axial order, so comparing indices is exactly
    comparing ``(q, r)`` tuples -- which is what the reference's ``tie_key`` does.
    """

    diff = 0
    while a >= 0 and b >= 0:
        ca = label_cell[a]
        cb = label_cell[b]
        if ca != cb:
            diff = -1 if ca < cb else 1
        a = label_parent[a]
        b = label_parent[b]
    return diff


@njit(cache=True, nogil=True)
def _state_slot(
    step,
    cell,
    recent,
    depth,
    paid_class,
    first_arc,
    state_key_step,
    state_key_cell,
    state_key_paid,
    state_key_first,
    state_recent,
    state_label,
    log2cap,
):
    """Find or claim the hash slot for one dominance state; -1 when the table is full.

    ``step`` is part of the key.  The reference keeps labels in ``layers[step][key]``,
    so the time layer is an *implicit* component of its dominance key; a single flat
    table without it would merge two labels that the reference keeps in separate
    layers -- most visibly two roots with the same origin cell and paid-row class but
    different departure steps, which is a real case whenever ground delay is allowed.
    """

    packed = np.uint64(step + 1) * np.uint64(1000003)
    packed = (packed ^ np.uint64(cell)) * np.uint64(1000003)
    for d in range(depth):
        packed = (packed ^ np.uint64(recent[d] + 1)) * np.uint64(1000003)
    packed = (packed ^ np.uint64(paid_class + 1)) * np.uint64(1000003)
    packed = (packed ^ np.uint64(first_arc + 2)) * np.uint64(1000003)

    cap = state_label.shape[0]
    slot = _mix(packed, log2cap)
    for _probe in range(cap):
        if state_label[slot] == -1:
            return slot
        if (
            state_key_step[slot] == step
            and state_key_cell[slot] == cell
            and state_key_paid[slot] == paid_class
            and state_key_first[slot] == first_arc
        ):
            same = True
            for d in range(depth):
                if state_recent[slot, d] != recent[d]:
                    same = False
                    break
            if same:
                return slot
        slot += 1
        if slot >= cap:
            slot = 0
    return -1


@njit(cache=True, nogil=True)
def _search_dag(
    # topology
    arc_start,
    arc_target,
    arc_roles,
    rev_remaining,
    dest_mask,
    min_step,
    max_step,
    revisit_depth,
    state_history_depth,
    seed_hop_limit,
    track_first_hop,
    is_seed,
    # duals
    dual_first,
    dual_start,
    dual_prefix,
    window_lo,
    window_hi,
    max_negative_credit,
    # variants
    v_cell,
    v_start_step,
    v_score,
    v_paid_class,
    v_departure,
    v_lane,
    v_ground_delay,
    v_origin_leg,
    v_origin_fold,
    v_origin_fold_exact,
    paid_start,
    paid_cell,
    paid_step,
    paid_value,
    destination_fold_s,
    destination_fold_exact,
    reference_time_s,
    dest_slot_of_cell,
    dest_positive,
    dest_step_base,
    # scalars
    dt_s,
    benefit,
    pi_f,
    cost_cutoff,
    have_cutoff,
    # workspace (label pool)
    label_cell,
    label_parent,
    label_hops,
    label_variant,
    label_first_arc,
    label_score,
    label_next,
    label_recent,
    layer_head,
    # workspace (state table)
    state_key_step,
    state_key_cell,
    state_key_paid,
    state_key_first,
    state_recent,
    state_label,
    log2cap,
    # outputs
    cand_parent,
    cand_cell,
    cand_step,
    cand_hops,
    cand_variant,
    cand_rc_ub,
    # forbidden rows (empty table => forbidden_n == 0 and every probe short-circuits)
    forbidden_slots,
    forbidden_log2cap,
    forbidden_n,
    cancel_flag,
):
    """Forward layer sweep over the time-expanded DAG.  See module docstring."""

    n_labels = 0
    n_cand = 0
    cap_labels = label_cell.shape[0]
    cap_cand = cand_parent.shape[0]
    remaining_rc_ub = -np.inf
    worst = 0
    worst_val = np.inf
    n_steps = max_step - min_step + 2
    for i in range(n_steps):
        layer_head[i] = -1

    scratch = np.empty(state_history_depth, dtype=np.int32)

    # ---- roots -------------------------------------------------------------
    for v in range(v_cell.shape[0]):
        cell = v_cell[v]
        start = v_start_step[v]
        if start < min_step or start > max_step:
            continue
        if n_labels >= cap_labels:
            return FB_LABEL_OVERFLOW, n_labels, n_cand, remaining_rc_ub
        for d in range(state_history_depth):
            scratch[d] = -1
        scratch[0] = cell
        slot = _state_slot(
            start, cell, scratch, state_history_depth, v_paid_class[v], -1,
            state_key_step, state_key_cell, state_key_paid, state_key_first,
            state_recent, state_label, log2cap,
        )
        if slot < 0:
            return FB_HASH_FULL, n_labels, n_cand, remaining_rc_ub
        keep = False
        old = state_label[slot]
        if old == -1:
            keep = True
        elif v_score[v] > label_score[old] + SCORE_EPS:
            keep = True
        elif v_score[v] >= label_score[old] - SCORE_EPS:
            # Equal score at hops=0: tie on (hops, departure, lane); paths are
            # single cells and identical here because the state pins the cell.
            if v_departure[v] < v_departure[label_variant[old]]:
                keep = True
            elif v_departure[v] == v_departure[label_variant[old]] and (
                v_lane[v] < v_lane[label_variant[old]]
            ):
                keep = True
        if not keep:
            continue
        if old != -1:
            # Overwrite the dominated label IN PLACE.  Appending a new row instead
            # would leave the old one linked into its layer bucket, so the search
            # would go on to expand a label the reference had already discarded --
            # the dict-overwrite semantics of ``layer[key] = label`` are what keep
            # the label set bounded by the number of distinct states.  Safe because
            # a label's children are only created when its own layer is swept, and
            # this state's layer has not been swept yet.
            i = old
        else:
            if n_labels >= cap_labels:
                return FB_LABEL_OVERFLOW, n_labels, n_cand, remaining_rc_ub
            i = n_labels
            n_labels += 1
            bucket = start - min_step
            label_next[i] = layer_head[bucket]
            layer_head[bucket] = i
            state_key_step[slot] = start
            state_key_cell[slot] = cell
            state_key_paid[slot] = v_paid_class[v]
            state_key_first[slot] = -1
            for d in range(state_history_depth):
                state_recent[slot, d] = scratch[d]
            state_label[slot] = i
        label_cell[i] = cell
        label_parent[i] = -1
        label_hops[i] = 0
        label_variant[i] = v
        label_first_arc[i] = -1
        label_score[i] = v_score[v]
        for d in range(state_history_depth):
            label_recent[i, d] = scratch[d]

    # ---- layer sweep -------------------------------------------------------
    for step in range(min_step, max_step + 1):
        if cancel_flag[0] != 0:
            return FB_CANCELLED, n_labels, n_cand, remaining_rc_ub
        li = layer_head[step - min_step]
        while li >= 0:
            hops = label_hops[li]
            cell = label_cell[li]
            v = label_variant[li]
            paid_class = v_paid_class[v]
            if (is_seed and hops >= seed_hop_limit) or step + 1 > max_step:
                li = label_next[li]
                continue

            folding_exact = (
                reference_time_s > 0.0
                and v_origin_fold_exact[v] != 0
                and destination_fold_exact
            )
            # Bound pruning.  ``_delay_lower_bound`` is non-decreasing in total
            # hops, so the loop the reference runs over hop counts is maximized at
            # its first iteration -- one expression, no table.  Dropping the
            # reference's destination-positive refinement only widens the bound,
            # so this never prunes a label the reference would keep.
            if have_cutoff:
                paid = -label_score[li] - v_ground_delay[v] - v_origin_leg[v] - hops * dt_s
                paid_positive = paid - RECOMPUTE_EPS
                if paid_positive < 0.0:
                    paid_positive = 0.0
                remain = rev_remaining[cell]
                total_hops = hops + remain
                if total_hops < 1:
                    total_hops = 1
                bound = (
                    benefit
                    - pi_f
                    - _delay_lower_bound(
                        v_ground_delay[v], v_origin_fold[v], total_hops,
                        destination_fold_s, reference_time_s, dt_s, folding_exact,
                    )
                    - paid_positive
                    + max_negative_credit
                )
                if bound <= cost_cutoff - RECOMPUTE_EPS:
                    li = label_next[li]
                    continue

            first_arc_flag = hops == 0
            for a in range(arc_start[cell], arc_start[cell + 1]):
                nb = arc_target[a]
                roles = arc_roles[a]

                banned = False
                for d in range(revisit_depth):
                    if label_recent[li, d] == nb:
                        banned = True
                        break
                if banned:
                    continue

                if first_arc_flag:
                    finish = (dest_mask[nb] != 0) and (roles & ARC_FIRST_LAST) != 0
                    cont = (roles & ARC_FIRST) != 0
                else:
                    finish = (dest_mask[nb] != 0) and (roles & ARC_LAST) != 0
                    cont = (roles & ARC_INTERNAL) != 0
                if (not finish) and (not cont):
                    continue

                next_step = step + 1
                remain_nb = rev_remaining[nb]
                if next_step + remain_nb > max_step:
                    continue
                if is_seed and hops + 1 + remain_nb > seed_hop_limit:
                    continue

                # Row exclusions.  Placed exactly where the reference tests them --
                # after the reachability guards, before the score is formed -- so an
                # excluded arc is dropped at the same point in both.
                if _visit_forbidden(
                    nb, next_step, window_lo, window_hi,
                    forbidden_slots, forbidden_log2cap, forbidden_n,
                ):
                    continue

                score = (
                    label_score[li]
                    - dt_s
                    - _window_cost(
                        nb, next_step, dual_first, dual_start, dual_prefix,
                        window_lo, window_hi, paid_class,
                        paid_start, paid_cell, paid_step, paid_value,
                    )
                )

                if finish:
                    fe = (
                        reference_time_s > 0.0
                        and v_origin_fold_exact[v] != 0
                        and destination_fold_exact
                    )
                    paid = -score - v_ground_delay[v] - v_origin_leg[v] - (hops + 1) * dt_s
                    if paid < 0.0:
                        paid = 0.0
                    # The row union's positive price is at least the max of what the
                    # path already paid and what arriving here unavoidably costs.
                    # max, never sum: the two sets overlap, so adding would
                    # double-count and make rc_ub stop being an upper bound.
                    dslot = dest_slot_of_cell[nb]
                    if dslot >= 0:
                        dp = dest_positive[dslot, next_step - dest_step_base]
                        if dp > paid:
                            paid = dp
                    rc_ub = (
                        benefit
                        - pi_f
                        - _delay_lower_bound(
                            v_ground_delay[v], v_origin_fold[v], hops + 1,
                            destination_fold_s, reference_time_s, dt_s, fe,
                        )
                        - paid
                        + max_negative_credit
                    )
                    if n_cand < cap_cand:
                        cand_parent[n_cand] = li
                        cand_cell[n_cand] = nb
                        cand_step[n_cand] = next_step
                        cand_hops[n_cand] = hops + 1
                        cand_variant[n_cand] = v
                        cand_rc_ub[n_cand] = rc_ub
                        n_cand += 1
                        if n_cand == cap_cand:
                            worst = 0
                            for c in range(1, cap_cand):
                                if cand_rc_ub[c] < cand_rc_ub[worst]:
                                    worst = c
                            worst_val = cand_rc_ub[worst]
                    elif rc_ub > worst_val:
                        # Buffer full and this beats the weakest: evict it, and book
                        # what leaves into the residual bound so the host's optimality
                        # proof still accounts for it.
                        if worst_val > remaining_rc_ub:
                            remaining_rc_ub = worst_val
                        cand_parent[worst] = li
                        cand_cell[worst] = nb
                        cand_step[worst] = next_step
                        cand_hops[worst] = hops + 1
                        cand_variant[worst] = v
                        cand_rc_ub[worst] = rc_ub
                        worst = 0
                        for c in range(1, cap_cand):
                            if cand_rc_ub[c] < cand_rc_ub[worst]:
                                worst = c
                        worst_val = cand_rc_ub[worst]
                    elif rc_ub > remaining_rc_ub:
                        # Weaker than everything held: only the residual moves.  The
                        # cached ``worst_val`` makes this O(1), which is what lets the
                        # buffer be sized generously -- rescanning per sink made a
                        # larger buffer quadratic in the number of sinks.
                        remaining_rc_ub = rc_ub

                if not cont:
                    continue

                scratch[0] = nb
                for d in range(1, state_history_depth):
                    scratch[d] = label_recent[li, d - 1]
                next_first = label_first_arc[li]
                if track_first_hop and next_first == -1:
                    next_first = a
                if not track_first_hop:
                    next_first = -1

                slot = _state_slot(
                    next_step, nb, scratch, state_history_depth, paid_class, next_first,
                    state_key_step, state_key_cell, state_key_paid, state_key_first,
                    state_recent, state_label, log2cap,
                )
                if slot < 0:
                    return FB_HASH_FULL, n_labels, n_cand, remaining_rc_ub

                old = state_label[slot]
                keep = False
                if old == -1:
                    keep = True
                elif score > label_score[old] + SCORE_EPS:
                    keep = True
                elif score >= label_score[old] - SCORE_EPS:
                    # Exact-score tie: reproduce _prefer's tie_key ordering
                    # (hops, departure_step, origin_lane_idx, path).
                    oh = label_hops[old]
                    if hops + 1 != oh:
                        keep = hops + 1 < oh
                    else:
                        ov = label_variant[old]
                        if v_departure[v] != v_departure[ov]:
                            keep = v_departure[v] < v_departure[ov]
                        elif v_lane[v] != v_lane[ov]:
                            keep = v_lane[v] < v_lane[ov]
                        else:
                            keep = _path_cmp(li, label_parent[old], label_parent, label_cell) < 0
                if not keep:
                    continue

                if old != -1:
                    j = old          # in-place overwrite; see the root branch above
                else:
                    if n_labels >= cap_labels:
                        return FB_LABEL_OVERFLOW, n_labels, n_cand, remaining_rc_ub
                    j = n_labels
                    n_labels += 1
                    bucket = next_step - min_step
                    label_next[j] = layer_head[bucket]
                    layer_head[bucket] = j
                    state_key_step[slot] = next_step
                    state_key_cell[slot] = nb
                    state_key_paid[slot] = paid_class
                    state_key_first[slot] = next_first
                    for d in range(state_history_depth):
                        state_recent[slot, d] = scratch[d]
                    state_label[slot] = j
                label_cell[j] = nb
                label_parent[j] = li
                label_hops[j] = hops + 1
                label_variant[j] = v
                label_first_arc[j] = next_first
                label_score[j] = score
                for d in range(state_history_depth):
                    label_recent[j, d] = scratch[d]

            li = label_next[li]

    status = OK if n_cand > 0 else NO_PATH
    return status, n_labels, n_cand, remaining_rc_ub


@dataclass(frozen=True, slots=True)
class DagCandidate:
    """One sink the kernel proposes, to be certified by the Python host."""

    cell_path: tuple[tuple[int, int], ...]
    departure_step: int
    origin_lane_idx: int | None
    hops: int
    rc_upper_bound: float


@dataclass(frozen=True, slots=True)
class DagSearchResult:
    """Kernel outcome plus the residual bound that licenses an optimality proof."""

    candidates: tuple[DagCandidate, ...] = ()
    remaining_rc_upper_bound: float = float("inf")
    status: int = OK
    n_labels: int = 0
    regrow: int = 0
    used_kernel: bool = True
    _cand_arrays: object = field(repr=False, default=None)

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, "unknown")

    @property
    def ok(self) -> bool:
        return self.status in (OK, NO_PATH)


def _next_pow2(value: int) -> int:
    cap = 8
    while cap < value:
        cap <<= 1
    return cap


def search_dag(
    topology,
    duals,
    variants,
    *,
    cfg,
    benefit: float,
    pi_f: float,
    cost_cutoff: float | None,
    seed: bool = False,
    # Sinks the buffer cannot hold are booked into ``remaining_rc_upper_bound``, which
    # is what the host's optimality proof must clear -- so an undersized buffer does
    # not lose columns, it loses *proofs*, and the flight then pays for the reference
    # DP as well.  Measured on colgen_test: at 2048 the buffer saturated on 36 of 97
    # flights and one could not certify; at 131072 none saturate, all 97 certify, and
    # the whole solve is faster despite ranking more candidates.  ~32 B/slot, so this
    # is ~4 MB against the label pool's ~1 GB.
    max_candidates: int = 1 << 17,
    label_limit: int | None = None,
    # ~33.6M labels.  Sizing note, because this constant is load-bearing and easy to
    # pick blindly: a label costs ~44 B (6 int32 + float64 + recent[depth] int32) and
    # its state slot ~32 B, and the table is the next power of two above 2x the pool,
    # so the ceiling is worth roughly 3.5 GB of transient workspace.  Flights that
    # exceed it fall back to the reference DP -- correct, just slow.  Measured on
    # colgen_test: the label counts form a continuum (2.6M / 2.9M / 3.1M / 3.3M / 3.4M
    # all certify), so this bound is a memory budget, not a semantic limit.
    #
    # Raised 1<<23 -> 1<<24 against a REAL scenario, because colgen_test never exercised
    # it: its flights are ~7 hops, density_faa's are a median 105 and a max 154, and the
    # label pool grows with hops.  On density_faa_wing_zipline (first 100 flights) two
    # flights exhausted 1<<23 and paid for the kernel AND the full Python reference DP --
    # 2 flights out of 100 owning 66.5% of the pricing wall (237.4s of 356.8s).  At 1<<24
    # both certify in-kernel (11,046,039 and 8,811,264 labels), the solve goes
    # 402.36s -> 245.05s (1.64x) with a bit-identical objective, and peak RSS rises only
    # 3145.6 -> 3384.9 MB (+7.6%) because the pool is transient, not retained.  This is
    # therefore a STRAGGLER knob at least as much as a memory knob: it also lifts the
    # makespan floor for flight-parallel pricing from 125s to 34s.
    label_limit_max: int = 1 << 25,
    forbidden=None,
    cancel_flag: np.ndarray | None = None,
) -> DagSearchResult:
    """Run the compiled DP, growing the workspace on overflow rather than failing.

    Overflow is recoverable and exact: a bigger pool re-runs the identical search.
    Only at the ceiling does the caller fall back to the Python reference.
    """

    n_cells = topology.n_cells
    n_variants = variants.n_variants
    if n_cells == 0 or n_variants == 0:
        return DagSearchResult(status=NO_PATH, remaining_rc_upper_bound=-float("inf"))

    if forbidden is None:
        from .dp_prepare import PreparedForbidden

        forbidden = PreparedForbidden()
    if cancel_flag is None:
        cancel_flag = np.zeros(1, dtype=np.uint8)
    limit = label_limit if label_limit is not None else max(4096, 64 * n_cells)
    n_steps = topology.max_step - topology.min_step + 1
    depth = topology.state_history_depth
    regrow = 0

    while True:
        state_cap = _next_pow2(2 * limit)
        log2cap = int(state_cap).bit_length() - 1
        label_cell = np.empty(limit, dtype=np.int32)
        label_parent = np.empty(limit, dtype=np.int32)
        label_hops = np.empty(limit, dtype=np.int32)
        label_variant = np.empty(limit, dtype=np.int32)
        label_first_arc = np.empty(limit, dtype=np.int32)
        label_score = np.empty(limit, dtype=np.float64)
        label_next = np.empty(limit, dtype=np.int32)
        label_recent = np.empty((limit, depth), dtype=np.int32)
        layer_head = np.empty(topology.max_step - topology.min_step + 2, dtype=np.int32)
        state_key_step = np.full(state_cap, np.iinfo(np.int32).min, dtype=np.int32)
        state_key_cell = np.full(state_cap, -1, dtype=np.int32)
        state_key_paid = np.full(state_cap, -1, dtype=np.int32)
        state_key_first = np.full(state_cap, -1, dtype=np.int32)
        state_recent = np.full((state_cap, depth), -1, dtype=np.int32)
        state_label = np.full(state_cap, -1, dtype=np.int32)
        cand_parent = np.full(max_candidates, -1, dtype=np.int32)
        cand_cell = np.empty(max_candidates, dtype=np.int32)
        cand_step = np.empty(max_candidates, dtype=np.int32)
        cand_hops = np.empty(max_candidates, dtype=np.int32)
        cand_variant = np.empty(max_candidates, dtype=np.int32)
        cand_rc_ub = np.full(max_candidates, -np.inf, dtype=np.float64)

        status, n_labels, n_cand, remaining = _search_dag(
            topology.arc_start, topology.arc_target, topology.arc_roles,
            topology.rev_remaining, topology.dest_mask,
            topology.min_step, topology.max_step,
            topology.revisit_depth, depth, topology.seed_hop_limit,
            topology.track_first_hop, seed,
            duals.dual_first, duals.dual_start, duals.dual_prefix,
            duals.window_lo, duals.window_hi, duals.max_negative_credit,
            variants.cell, variants.start_step, variants.score, variants.paid_class,
            variants.departure_step, variants.lane_idx,
            variants.ground_delay_s, variants.origin_leg_s,
            variants.origin_fold_s, variants.origin_fold_exact,
            variants.paid_start, variants.paid_cell, variants.paid_step, variants.paid_value,
            variants.destination_fold_s, variants.destination_fold_exact,
            variants.reference_time_s,
            variants.dest_slot_of_cell, variants.dest_positive, variants.dest_step_base,
            float(cfg.dt_s), float(benefit), float(pi_f),
            float(cost_cutoff) if cost_cutoff is not None else 0.0,
            cost_cutoff is not None,
            label_cell, label_parent, label_hops, label_variant, label_first_arc,
            label_score, label_next, label_recent, layer_head,
            state_key_step, state_key_cell, state_key_paid, state_key_first,
            state_recent, state_label, log2cap,
            cand_parent, cand_cell, cand_step, cand_hops, cand_variant, cand_rc_ub,
            forbidden.slots, forbidden.log2cap, forbidden.n_rows,
            cancel_flag,
        )

        if status in (FB_LABEL_OVERFLOW, FB_HASH_FULL) and limit < label_limit_max:
            # One overflow is enough to know the opening guess was wrong, and it is
            # wrong *structurally*: ``64 * n_cells`` has no step term, while the label
            # count tracks ``n_cells * n_steps`` closely -- measured ratio 1.03-1.58
            # across every flight that overflowed on density_faa.  So jump to the
            # geometry-derived size rather than climbing x4 and re-running the whole
            # search at each rung.
            #
            # Sizing only kicks in AFTER an overflow on purpose.  58 of 100 flights
            # never overflow, and opening at the geometric size would hand all of them
            # a pool orders of magnitude larger than they use -- the dual cutoff keeps
            # most searches tiny.  This way the common case is untouched and only the
            # flights that proved they need room get it.
            # Jumping on the FIRST overflow also uses less peak memory, which is the
            # opposite of what it looks like.  The product over-shoots badly for
            # well-pruned flights (measured ratios down to 0.07, a 16.7M pool holding
            # 800k labels), so an intermediate x4 rung was tried to avoid that -- and
            # measured worse on both axes (peak RSS 6809 MB vs 6656, wall 70.77s vs
            # 70.07s).  The reason: every rung is itself allocated and its state table
            # is ``np.full``-initialised, so an extra rung ADDS a transient peak rather
            # than avoiding one, and peak is a max over the run, not a sum.
            geometric = int(_LABEL_GEOMETRY_SAFETY * n_cells * n_steps)
            limit = min(max(limit * 4, geometric), label_limit_max)
            regrow += 1
            continue
        break

    # Candidates are surfaced even on a failure status.  Sinks are emitted *before*
    # dominance, so whatever the buffer holds when the pool overflows is a set of
    # genuinely reachable columns -- discarding them wasted the entire run and left
    # the caller with nothing to tighten its cutoff with.  The residual bound is
    # still infinite (an aborted search cannot bound what it never explored), so
    # these can seed an incumbent but can never prove optimality on their own.
    failed = status not in (OK, NO_PATH)
    residual = float("inf") if failed else float(remaining)

    order = np.argsort(-cand_rc_ub[:n_cand], kind="stable")
    cells_q = topology.cell_q
    cells_r = topology.cell_r
    candidates = []
    for c in order:
        parent = int(cand_parent[c])
        chain = [int(cand_cell[c])]
        node = parent
        while node >= 0:
            chain.append(int(label_cell[node]))
            node = int(label_parent[node])
        chain.reverse()
        v = int(cand_variant[c])
        lane = int(variants.lane_idx[v])
        candidates.append(
            DagCandidate(
                cell_path=tuple((int(cells_q[i]), int(cells_r[i])) for i in chain),
                departure_step=int(variants.departure_step[v]),
                origin_lane_idx=None if lane < 0 else lane,
                hops=int(cand_hops[c]),
                rc_upper_bound=float(cand_rc_ub[c]),
            )
        )
    return DagSearchResult(
        candidates=tuple(candidates),
        remaining_rc_upper_bound=residual,
        status=status,
        n_labels=int(n_labels),
        regrow=regrow,
    )


def warm_kernel() -> bool:
    """Compile the dispatcher signature off the hot path; assert a known answer."""

    from ...config import SimConfig
    from .. import hexgrid as hg
    from ...types import FlightRequest, vec
    from .dp_prepare import prepare_duals, prepare_topology, prepare_variants
    from .network import build_flight_graph
    from .params import ColGenParams
    from .pricing import DualView

    cfg = SimConfig(
        planner="colgen", flight_levels_m=(100.0,), airspace_ceiling_m=125.0,
        region_size_m=(20_000.0, 20_000.0), terminal_airspace_always_active=True,
        max_ground_delay_s=0.0, max_detour_factor=10.0,
    )
    radius = hg.circumradius(cfg)

    def point(cell):
        x, y = hg.hex_center(*cell, radius)
        return vec(x, y, cfg.ground_level_m)

    params = ColGenParams(solver="highs", detour_slack_hops=1)
    fg = build_flight_graph(FlightRequest(1, point((0, 0)), point((2, 0)), 0.0, 0.0), cfg, (), params)
    topology = prepare_topology(fg, cfg)
    if topology.unsupported_reason is not None:
        return False
    view = DualView({}, cfg)
    result = search_dag(
        topology, prepare_duals(view, topology),
        prepare_variants(fg, cfg, view, topology, seed=False),
        cfg=cfg, benefit=params.M, pi_f=0.0, cost_cutoff=None, max_candidates=8,
    )
    return result.ok and len(result.candidates) > 0
