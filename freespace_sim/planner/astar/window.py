"""Per-plan DENSE occupancy window — MAPF-LNS's ``PathTable``, applied locally.

``kernel._blocked`` is the hottest read in the whole search (6 reroutes + a hover + the vertical
rungs, once per expansion). It answers by walking two flat free-interval pools
(:class:`compiled_hex_occupancy._Pool`) indexed by cell id: one list traversal per pool per probe,
plus a read of ``static_col`` and one of ``ov_own_gen``. At ``density_faa`` scale those pools hold
1.4M + 371k live interval slots over 323k cells — a mean chain of 4.4 and far longer on the hot
cells — and ``analysis/prof_ledger_scaling.py`` names that chain growth as the run's later-in-run
slowdown, since the (watermark-only) compiled eviction never reclaims historical fragments.

**What the win is NOT.** The pools total ~52 MB, four times past the 12 MB this machine's P-cores
share as L2, so the obvious argument is the ~137 ns DRAM latency measured on this box against ~2 ns
for an L1-resident set. That argument is wrong, and measuring it is what showed why: one plan
touches only ~5,550 cells (``.context/perf/probe_read_window.py``), i.e. ~180 KB of head slots,
which is L2-resident after first touch. The saving is the list WALK and the two per-cell reads, not
the latency of reaching the pool — and the measured speedup is correspondingly modest (~1.08-1.14x,
flat in worker count) rather than the order of magnitude the latency framing suggests.

MAPF-LNS's C++ ``PathTable`` has neither problem: its map is a ``vector<vector<int>>`` over
``[location][timestep]`` and ``constrained()`` is 3–5 direct array reads. On their 32×32 Room map
that table is ~0.8 MB and lives in L2. It does not generalise here — 144k hexes × 3 levels × ~1,800
steps is ~777M space-time cells, so a dense global table is impossible.

What IS possible is the same table over the region ONE plan reads. A repair replans a single flight
between two hubs; its probes fall inside a small axial box over a short step window (the kernel's
own ``read_bbox`` telemetry measures it — see ``.context/perf/probe_read_window.py``). So before the
search, materialise a **bitmap**

    bit k of win[wcell * row_bytes + (k >> 3)]      where k = s - ws0,
    wcell = (iq * wrspan + ir) * n_levels + L

and ``_blocked`` inside the window becomes one byte read, a shift and a mask, out of a buffer that
fits in cache — with each fragmented interval list walked ONCE at build time instead of once per
probe. One bit is enough because everything ``_blocked`` returns 1 for is an OR:

    blocked(cell, s) = corridor pool blocks it
                    OR a FOREIGN column walls it — transient (column pool) or always-active
                       (``static_col``) — with the flight not owning the cell (``ov_own_gen != gen``)

Folding ``static_col`` in is exact because that wall is step-independent, and folding ``ov_own_gen``
is exact because the overlay is stamped per plan with the same ``gen`` the window is built under
(``_plan_compiled`` rebuilds BOTH inside the FB_MASK/FB_HASH re-run loop, so a re-run can never read
a window built under a stale generation).

**The build is O(claims), not O(window).** Restating the definition above as its complement,
``free = corridor-free AND (own OR column-free)``, makes the whole row a set operation: a cell
nothing was ever committed to is one seed interval ``[0, MAXS]`` in both pools and writes nothing at
all, and a claimed cell is filled blocked and then cleared over the INTERSECTION of the two free
lists — a two-pointer merge, so the cost is the number of intervals plus the bits cleared, never the
window's area. That intersection walk is the one thing here that relies on the pools' ascending-sort
invariant, which ``_Pool.block_range``'s own early-exit already depends on.

Rows are padded to whole bytes so those clears are byte and word operations; that wastes under 8
bits per cell, against an 8x larger working set for a byte-per-entry table.

**Why this cannot change an answer.** The window is a pure cache of the pools it was built from,
which do not change during a search (commits happen between plans). Every probe outside it falls
through to the original list walk. So an undersized window costs speed, never correctness — which
is why the bounds below are a heuristic with telemetry (``win_stats``) rather than a proof.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:                     # numba absent — this module must still IMPORT.
    # `planner` imports it at module level (for `empty_wbox`/`window_bounds`/`disable`, which are
    # plain Python), while its numba fallback is an ImportError guard around `.kernel` inside
    # `AStarPlanner.__init__`. A hard import here would turn "degrade to the reference search" into
    # "the package will not import" — which is exactly what it did before this guard existed, and
    # what `tests/test_astar_window.py::test_window_module_imports_without_numba` now pins.
    #
    # The stand-in binds a body that RAISES rather than a pure-Python one that works. Nothing can
    # reach it — without a kernel every plan goes to `_plan_reference`, which never builds a window —
    # so a silently-interpreted `build_window` would be an unbounded slowdown nobody asked for.
    def njit(*_args, **_kwargs):
        def deco(fn):
            def _needs_numba(*_a, **_kw):
                raise RuntimeError(
                    f"{fn.__name__} is a numba kernel and numba is not installed; the dense "
                    f"occupancy window is only reachable from the compiled A* path")
            return _needs_numba
        return deco

# --- wbox: the window's geometry, as one int64[9] so `_blocked`'s signature stays readable.
# `W_STEPS == 0` is the OFF switch, and `_blocked` and `disable` are the only two places that know it.
W_Q0, W_Q1, W_R0, W_R1, W_S0, W_S1, W_RSPAN, W_STEPS = range(8)
W_ROWB = 8                  # row_bytes = ceil(steps / 8), precomputed so the kernel does no division
WBOX_N = 9

# --- win_stats: int64[3], probe accounting. HIT/MISS are cumulative counters for tuning the bounds;
# MISSED is a per-plan STICKY flag the host resets before each search and tests after it.
#
# One flag checked once per plan, rather than a distinct return value checked at each probe: only one
# of `_search`'s five `_blocked` call sites inspects the returned value (the neighbour test looks for
# the out-of-box -1); the other four compare against zero, so a new negative code would be read as
# "blocked" at four of them and silently prune a legal edge. A sticky flag cannot be misread that way,
# and it costs one branch per plan instead of five per probe.
WS_HIT, WS_MISS, WS_MISSED = 0, 1, 2
WSTATS_N = 3


@njit(cache=True, nogil=True)
def _fill_row(win, row, wsteps):
    """Mark every in-window step of ``row`` blocked. Padding bits past ``wsteps`` stay 0 — the kernel
    never reads them, and leaving them clear keeps the row's meaning unambiguous."""
    full = wsteps >> 3
    for i in range(full):
        win[row + i] = np.uint8(0xFF)
    rem = wsteps & 7
    if rem != 0:
        win[row + full] = np.uint8((1 << rem) - 1)


@njit(cache=True, nogil=True)
def _set_range(win, row, k0, k1):
    """Set bits ``k0..k1`` inclusive — the blocked side of a claim paint, and the exact mirror of
    :func:`_clear_range`."""
    b0 = k0 >> 3
    b1 = k1 >> 3
    if b0 == b1:
        win[row + b0] |= np.uint8(((((1 << (k1 - k0 + 1)) - 1) << (k0 & 7))) & 0xFF)
        return
    win[row + b0] |= np.uint8((0xFF << (k0 & 7)) & 0xFF)
    for i in range(b0 + 1, b1):
        win[row + i] = np.uint8(0xFF)
    win[row + b1] |= np.uint8(((1 << ((k1 & 7) + 1)) - 1) & 0xFF)


@njit(cache=True, nogil=True)
def build_window_claims(arena, slab_start, slab_len, static_col, ov_own_gen, gen,
                        qmin, rmin, rspan, n_levels, wbox, win,
                        s0_shift, span_bits, field_mask):
    """Fill the ``win`` bitmap from the flat claim arena (:mod:`claim_arena`).

    Simpler than the pool build it replaces, and measured 2.81x faster, for one reason: a claim IS a
    blocked span, so this is ``win |= span``. The pools store the COMPLEMENT, so building from them
    means filling each claimed row and then clearing back over a two-pointer merge of two free-interval
    lists — and that merge is the only thing in either build that depends on the pools' ascending-sort
    invariant. Nothing here does; the paint is an OR, so slab order is free (which is what lets a
    release swap-remove).

    The fold matches ``kernel._blocked`` term for term: a corridor claim always blocks; a column claim
    or an always-active wall blocks only when the flight does not own the cell. A static wall covers
    every step, so it subsumes that cell's column claims and they are skipped."""
    wq0 = wbox[W_Q0]; wq1 = wbox[W_Q1]; wr0 = wbox[W_R0]
    ws0 = wbox[W_S0]; ws1 = wbox[W_S1]
    wrspan = wbox[W_RSPAN]; wsteps = wbox[W_STEPS]; row_bytes = wbox[W_ROWB]
    win[:(wq1 - wq0 + 1) * wrspan * n_levels * row_bytes] = 0
    n_painted = 0
    wcell = -1
    for iq in range(wq1 - wq0 + 1):
        gq = (wq0 + iq) - qmin
        for ir in range(wrspan):
            gr = (wr0 + ir) - rmin
            for L in range(n_levels):
                wcell += 1
                cell = (gq * rspan + gr) * n_levels + L
                own = ov_own_gen[cell] == gen
                row = wcell * row_bytes
                touched = False
                key = cell << 1                       # corridor claims: always block
                base = slab_start[key]
                for m in range(slab_len[key]):
                    packed = arena[base + m]
                    a = packed >> s0_shift
                    b = (packed >> span_bits) & field_mask
                    if a < ws0:
                        a = ws0
                    if b > ws1:
                        b = ws1
                    if a <= b:
                        _set_range(win, row, a - ws0, b - ws0)
                        touched = True
                if not own:
                    if static_col[cell]:              # permanent wall: covers every step, so it
                        _fill_row(win, row, wsteps)   # subsumes this cell's column claims
                        touched = True
                    else:
                        key = (cell << 1) | 1
                        base = slab_start[key]
                        for m in range(slab_len[key]):
                            packed = arena[base + m]
                            a = packed >> s0_shift
                            b = (packed >> span_bits) & field_mask
                            if a < ws0:
                                a = ws0
                            if b > ws1:
                                b = ws1
                            if a <= b:
                                _set_range(win, row, a - ws0, b - ws0)
                                touched = True
                if touched:
                    n_painted += 1
    return n_painted


def window_bounds(cocc, wbox, *, q_cells, r_cells, base, max_step, n_gsteps,
                  lateral_margin, tail_steps, max_bytes):
    """Size the window around the cells a plan is anchored to; return its byte count, or 0 for off.

    ``q_cells``/``r_cells`` are the origin hex, its exit lanes and the destination's landing lanes.
    A* explores an ellipse between them, so their bbox plus ``lateral_margin`` hexes covers the
    reroute fan; the margin is set from the measured read bboxes in
    ``.context/perf/probe_read_window.py``, not from a bound.

    Steps run from ``base`` — nothing is probed earlier, the ground state starts there — to
    ``base + n_gsteps + tail_steps``, clipped to ``max_step``. ``n_gsteps`` is the ground-delay
    allowance the two-phase mask already bounds; ``tail_steps`` covers the takeoff climb, the lane
    traverse and the flight itself.

    Two failure returns, and they are NOT interchangeable — the caller can recover from one:

    * ``0``  the box degenerated (clipped to nothing). Nothing to build at any size.
    * ``-n`` the box would need ``n`` bytes, more than ``max_bytes``. The caller may grow its buffer
      and ask again; ``_build_window`` does exactly that.

    The distinction exists because a window failure is no longer cheap. This function once documented
    returning 0 as "window off, every probe takes the pool walk … byte-identical to one with it on",
    and that was true while the interval pools existed. They were deleted with the pool-less
    occupancy: no window now means no answers, so the whole plan falls to the pure-Python reference.
    A 9% overshoot was measured costing 19.3 s of an 88 s LNS loop, which is why the size is reported
    back rather than collapsed into a single "off"."""
    q0 = int(min(q_cells)) - lateral_margin
    q1 = int(max(q_cells)) + lateral_margin
    r0 = int(min(r_cells)) - lateral_margin
    r1 = int(max(r_cells)) + lateral_margin
    # Clip to the global box rather than bailing: a plan near the region edge keeps a window over
    # the part that exists, and strays outside fall through to the pools like any other miss.
    q0 = max(q0, cocc.qmin); q1 = min(q1, cocc.qmin + cocc.qspan - 1)
    r0 = max(r0, cocc.rmin); r1 = min(r1, cocc.rmin + cocc.rspan - 1)
    s0 = max(0, int(base))
    s1 = min(int(max_step), int(cocc.MAXS), s0 + int(n_gsteps) + int(tail_steps))
    if q1 < q0 or r1 < r0 or s1 < s0:
        return 0
    row_bytes = (s1 - s0 + 1 + 7) // 8
    nbytes = (q1 - q0 + 1) * (r1 - r0 + 1) * cocc.n_levels * row_bytes
    if nbytes > max_bytes:
        return -nbytes              # recoverable: this is how big a buffer would have to be
    wbox[W_Q0] = q0; wbox[W_Q1] = q1
    wbox[W_R0] = r0; wbox[W_R1] = r1
    wbox[W_S0] = s0; wbox[W_S1] = s1
    wbox[W_RSPAN] = r1 - r0 + 1
    wbox[W_STEPS] = s1 - s0 + 1
    wbox[W_ROWB] = row_bytes
    return nbytes


def disable(wbox) -> None:
    """Mark ``wbox`` off. ``W_STEPS == 0`` is the kernel's enable test, so this is the one place that
    has to agree with ``_blocked``."""
    wbox[W_STEPS] = 0


def empty_wbox() -> np.ndarray:
    """An off window — the argument every caller that does not want one passes."""
    return np.zeros(WBOX_N, np.int64)
