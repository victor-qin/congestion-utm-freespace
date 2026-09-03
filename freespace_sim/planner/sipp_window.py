"""Per-plan safe-interval chains, built from the A* claim arena over the window a plan reads.

This is ``astar/window.py``'s move applied to SIPP. The reasoning is identical and the measurement
is in ``context/sipp_runtime_plan.md``: a global structure that derives free intervals from every
commit is maintained in full so each plan can read **1.37% of its cells** (p50 3,910 of 286,026,
over 28.5% of the steps). Worse, free intervals are the one representation that cannot be un-built —
a blocked span can be subtracted from a free run, but removing one flight's span cannot be undone
without re-applying every other flight's, which is why ``CompiledOccupancy.on_release`` costs 5.995
ms/flight against the arena's 0.034 and **grows with congestion** (amplification 1.88x at 150 warm
flights, 7.03x at 2,400).

So: keep the claims, derive the intervals. One pass per in-window cell reads that cell's claim slabs
out of the arena and emits the complement of them over ``[ws0, ws1]``, into the exact chain layout
``sipp_kernel._search`` already walks:

    slot s:  iv_lo[s], iv_hi[s], iv_nxt[s]        head of window-cell ``w`` IS slot ``w``
    overflow intervals live at [n_wcells, tail) and are linked from the head via ``iv_nxt``
    a degenerate head (``lo > hi``) is a cell with no free interval at all

**The chain must ascend, and unlike A*'s bitmap paint that is not free.** ``_search`` has two
``break`` statements that abandon a chain walk on the reasoning "the chain ascends, so every later
interval starts later still" — the lateral block's ``if a - 1 > hi_c`` and the rung block's
``if ap > hi_c - rsteps``. Feed it an unordered chain and it silently drops legal successors:
suboptimal plans, no crash, no failing test. ``build_window_claims`` explicitly does NOT need this
("the paint is an OR, so slab order is free, which is what lets a release swap-remove"), and the
arena's slabs are indeed unordered because removal is a swap-remove. Hence the sort: the complement
of a sorted, merged span list is ascending by construction rather than by assertion.

The own-column fold is A*'s, term for term — ``ov_own_gen[cell] == gen`` skips the column slab and
the always-active wall. That single boolean **is** the overlay ``SIPPPlanner._sbuild_overlay`` used
to build out of ``SafeIntervalIndex``, which is why this module replaces two structures rather than
one.
"""
from __future__ import annotations

import numpy as np

from .astar.window import W_Q0, W_Q1, W_R0, W_R1, W_RSPAN, W_S0, W_S1, W_STEPS

try:
    from numba import njit
except ImportError:                     # numba absent — same guard as `astar/window`: this module
    def njit(*_args, **_kwargs):        # is imported at module level by `sipp`, whose own numba
        def deco(fn):                   # is an ImportError guard around `.sipp_kernel`.
            def _needs_numba(*_a, **_kw):
                raise RuntimeError(
                    f"{fn.__name__} is a numba kernel and numba is not installed; the per-plan "
                    f"safe-interval window is only reachable from the compiled SIPP path")
            return _needs_numba
        return deco

# Above this many claims in one cell, insertion sort's quadratic term would beat the library sort's
# per-call setup. Measured claim counts per window cell at density_faa are single digits (p50 window:
# 10,488 cells, ~24k overflow intervals), so this is a bound on the tail rather than a tuned knob.
_INSERTION_MAX = 32


def window_bounds(cocc, wbox, *, q_cells, r_cells, base, max_step, lateral_margin) -> int:
    """Size the window around the cells a plan is anchored to; return its cell count, or 0 if it
    degenerated. Mirrors ``astar/window.window_bounds`` with two deliberate differences.

    **The step span is ``[base, max_step]`` exactly — no heuristic tail.** A* clips steps to
    ``base + n_gsteps + tail_steps`` because a step outside its span costs one probe, which the
    window-miss flag catches. A SIPP interval's ``hi`` is not a probe: it answers *"how long may I
    wait in this cell"*, so a short span does not miss anything detectably — it silently shortens a
    wait and returns a plan that is feasible and worse, with no flag raised and no test failing.
    ``max_step`` already bounds the search (``_search`` skips ``arr > max_step``), so clipping there
    is exact, removes a widen axis, and costs nothing: this structure stores interval ENDPOINTS,
    where A*'s bitmap pays row bytes per step.

    **No slot budget here.** A* prices its buffer with arithmetic (cells x row_bytes); an interval
    count is data-dependent. Summing ``slab_len`` over the window would be 20k-80k Python iterations
    per plan, which would exhaust the whole build budget before one interval is emitted, so the
    capacity check lives inside :func:`build_window_intervals` — which visits exactly those cells
    anyway, in compiled code.
    """
    q0 = int(min(q_cells)) - lateral_margin
    q1 = int(max(q_cells)) + lateral_margin
    r0 = int(min(r_cells)) - lateral_margin
    r1 = int(max(r_cells)) + lateral_margin
    # Clip to the global box rather than bailing, exactly as A* does: a plan near the region edge
    # keeps a window over the part that exists. A stray past the GLOBAL box is still FB_OOB.
    q0 = max(q0, cocc.qmin); q1 = min(q1, cocc.qmin + cocc.qspan - 1)
    r0 = max(r0, cocc.rmin); r1 = min(r1, cocc.rmin + cocc.rspan - 1)
    s0 = max(0, int(base))
    s1 = min(int(max_step), int(cocc.MAXS))
    if q1 < q0 or r1 < r0 or s1 < s0:
        return 0
    wbox[W_Q0] = q0; wbox[W_Q1] = q1
    wbox[W_R0] = r0; wbox[W_R1] = r1
    wbox[W_S0] = s0; wbox[W_S1] = s1
    wbox[W_RSPAN] = r1 - r0 + 1
    wbox[W_STEPS] = s1 - s0 + 1
    return (q1 - q0 + 1) * (r1 - r0 + 1) * cocc.n_levels


@njit(cache=True, nogil=True)
def build_window_intervals(arena, slab_start, slab_len, static_col, ov_own_gen, gen,
                           qmin, rmin, rspan, n_levels, wbox,
                           iv_lo, iv_hi, iv_nxt, scratch,
                           s0_shift, span_bits, field_mask):
    """Fill ``iv_*`` with each in-window cell's free-interval chain. Returns the slots used,
    or ``-needed`` (negative) if ``iv_*`` is too small — in which case **nothing has been written**.

    ``scratch`` is per-cell working space for the claim sort and must be at least as long as
    ``iv_lo``. That is not an arbitrary demand: the capacity pass below bounds the TOTAL claim count
    over the window by ``needed``, and a single cell's count cannot exceed the total, so one sizing
    rule covers both buffers and there is only ever one shortfall to report.

    Window-cell ``w`` and global cell id are the two encodings ``build_window_claims`` uses, and
    the iteration order (q-major, then r, then level) is identical so the two builds can be compared
    cell for cell:

        w    = ((q - wq0)  * wrspan * n_levels) + ...   sequential counter
        cell = ((q - qmin) * rspan  + (r - rmin)) * n_levels + L
    """
    wq0 = wbox[W_Q0]; wq1 = wbox[W_Q1]; wr0 = wbox[W_R0]
    ws0 = wbox[W_S0]; ws1 = wbox[W_S1]
    wrspan = wbox[W_RSPAN]
    n_wcells = (wq1 - wq0 + 1) * wrspan * n_levels
    cap_slots = iv_lo.shape[0]

    # ---- capacity pass. A cell whose K blocked spans merge has at most K+1 free intervals, one
    # of which is the head, so the overflow it can need is bounded by its claim count. Summing that
    # over the window bounds the whole build EXACTLY, and doing it first is what makes a shortfall
    # recoverable rather than half-applied — the same contract as `claim_arena.add_many`.
    total = 0
    for iq in range(wq1 - wq0 + 1):
        gq = (wq0 + iq) - qmin
        for ir in range(wrspan):
            gr = (wr0 + ir) - rmin
            for L in range(n_levels):
                cell = (gq * rspan + gr) * n_levels + L
                total += slab_len[cell << 1]
                if ov_own_gen[cell] != gen:
                    total += slab_len[(cell << 1) | 1]
    needed = n_wcells + total
    if needed > cap_slots or needed > scratch.shape[0]:
        return -needed

    # ---- build pass
    tail = n_wcells
    wcell = -1
    for iq in range(wq1 - wq0 + 1):
        gq = (wq0 + iq) - qmin
        for ir in range(wrspan):
            gr = (wr0 + ir) - rmin
            for L in range(n_levels):
                wcell += 1
                cell = (gq * rspan + gr) * n_levels + L
                own = ov_own_gen[cell] == gen
                if not own and static_col[cell]:
                    # Foreign always-active wall: blocked at EVERY step, so it subsumes this cell's
                    # claims entirely and the free set is empty. `lo > hi` is the encoding every
                    # reader already skips (`_search` guards each interval with `lo <= hi`).
                    iv_lo[wcell] = 1; iv_hi[wcell] = 0; iv_nxt[wcell] = -1
                    continue
                # Collect this cell's blocked spans, clipped to the window's step range, packed one
                # per int64 so a single sort orders by s0 (and ties by s1, which is harmless).
                n = 0
                key = cell << 1                          # corridor claims: always block
                b0 = slab_start[key]
                for m in range(slab_len[key]):
                    packed = arena[b0 + m]
                    a = packed >> s0_shift
                    b = (packed >> span_bits) & field_mask
                    if a < ws0:
                        a = ws0
                    if b > ws1:
                        b = ws1
                    if a <= b:
                        scratch[n] = (a << 32) | b
                        n += 1
                if not own:                              # own column is transparent — this single
                    key = (cell << 1) | 1                # branch IS the deleted `_sbuild_overlay`
                    b0 = slab_start[key]
                    for m in range(slab_len[key]):
                        packed = arena[b0 + m]
                        a = packed >> s0_shift
                        b = (packed >> span_bits) & field_mask
                        if a < ws0:
                            a = ws0
                        if b > ws1:
                            b = ws1
                        if a <= b:
                            scratch[n] = (a << 32) | b
                            n += 1
                if n == 0:                               # the common case: nothing ever claimed it
                    iv_lo[wcell] = ws0; iv_hi[wcell] = ws1; iv_nxt[wcell] = -1
                    continue
                # Insertion sort, not `scratch[:n].sort()`, and this is worth 8.9x of the whole
                # build — measured 3.041 ms/window against 0.341 (p90 6.639 against 0.867), where
                # collecting the claims and doing NOTHING else is 0.159. `n` is a handful per cell
                # and this runs once per in-window cell, so a general-purpose quicksort pays its
                # setup thousands of times per window to order about four elements. (`np.sort(x)`
                # is worse still: in nopython mode it returns a NEW array, so a heap allocation per
                # congested cell.) Above `_INSERTION_MAX` the quadratic term would win instead, so
                # hand those back to the library sort — a bound, not an expected case.
                if n <= _INSERTION_MAX:
                    for i in range(1, n):
                        v = scratch[i]
                        j = i - 1
                        while j >= 0 and scratch[j] > v:
                            scratch[j + 1] = scratch[j]
                            j -= 1
                        scratch[j + 1] = v
                else:
                    scratch[:n].sort()
                # Sweep the sorted spans, emitting the gaps. `lo` is the next possibly-free step;
                # taking max(lo, b+1) merges overlaps and containment without a separate merge pass.
                lo = ws0
                prev = -1
                for i in range(n):
                    a = scratch[i] >> 32
                    b = scratch[i] & 0xFFFFFFFF
                    if a > lo:
                        if prev < 0:
                            iv_lo[wcell] = lo; iv_hi[wcell] = a - 1
                            prev = wcell
                        else:
                            iv_lo[tail] = lo; iv_hi[tail] = a - 1
                            iv_nxt[prev] = tail
                            prev = tail
                            tail += 1
                    if b + 1 > lo:
                        lo = b + 1
                    if lo > ws1:
                        break
                if lo <= ws1:
                    if prev < 0:
                        iv_lo[wcell] = lo; iv_hi[wcell] = ws1
                        prev = wcell
                    else:
                        iv_lo[tail] = lo; iv_hi[tail] = ws1
                        iv_nxt[prev] = tail
                        prev = tail
                        tail += 1
                if prev < 0:                             # fully covered — no free interval anywhere
                    iv_lo[wcell] = 1; iv_hi[wcell] = 0; iv_nxt[wcell] = -1
                else:
                    iv_nxt[prev] = -1
    return tail


def empty_wbox() -> np.ndarray:
    """An off window. ``W_STEPS == 0`` is the disabled test, shared with ``astar/window``."""
    return np.zeros(9, np.int64)
