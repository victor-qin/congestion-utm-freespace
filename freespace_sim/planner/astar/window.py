"""Per-plan dense occupancy window — MAPF-LNS's ``PathTable``, applied locally.

A global ``[cell, step]`` table is prohibitively large, but one flight reads a small axial box over
a bounded step range. Before each compiled search, materialise that region as a bit-packed bitmap:

    bit k of win[wcell * row_bytes + (k >> 3)]      where k = s - ws0,
    wcell = (iq * wrspan + ir) * n_levels + L

``_blocked`` then becomes one byte read, shift, and mask. One bit is enough because blocked state is
the OR of corridor claims and foreign terminal-column occupancy:

    blocked(cell, s) = corridor claim covers it
                    OR a FOREIGN column walls it — transient claim or always-active
                       (``static_col``) — with the flight not owning the cell (``ov_own_gen != gen``)

Folding ``static_col`` in is exact because that wall is step-independent, and folding ``ov_own_gen``
is exact because the overlay is stamped per plan with the same ``gen`` the window is built under
(``_plan_compiled`` rebuilds BOTH inside the FB_MASK/FB_HASH re-run loop, so a re-run can never read
a window built under a stale generation).

The build scans each in-box cell's contiguous claim slabs and ORs overlapping spans; claim order is
irrelevant. Rows are byte-padded, wasting fewer than eight bits per cell. Any probe outside the
heuristic bounds marks the run invalid: the host widens and rebuilds, then uses the pure-Python
reference if misses remain at the ceiling. No partial-window result is consumed.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:                     # numba absent — this module must still IMPORT.
    # `planner` imports it at module level (for `empty_wbox`/`window_bounds`/`disable`, which are
    # plain Python), while its numba fallback is an ImportError guard around `.kernel` inside
    # `AStarPlanner.__init__`. A hard import here would turn "degrade to the reference search" into
    # "the package will not import"; that it still imports without numba is pinned by
    # `tests/test_astar_window.py::test_window_module_imports_without_numba`.
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
    never reads them, and leaving them clear keeps the row's meaning unambiguous.

    Parameters
    ------------
    - win (np.ndarray): bit-packed occupancy bitmap (uint8), mutated in place.
    - row (int): byte offset of this row's first byte within ``win``.
    - wsteps (int): number of active steps in the window; bits ``0..wsteps-1`` are set.

    Return
    --------
    - output (None): no return value; sets this row's step bits in ``win`` in place.
    """
    full = wsteps >> 3
    for i in range(full):
        win[row + i] = np.uint8(0xFF)
    rem = wsteps & 7
    if rem != 0:
        win[row + full] = np.uint8((1 << rem) - 1)


@njit(cache=True, nogil=True)
def _set_range(win, row, k0, k1):
    """Set bits ``k0..k1`` inclusive in ``row`` — paints one claim's steps blocked.

    Parameters
    ------------
    - win (np.ndarray): bit-packed occupancy bitmap (uint8), mutated in place.
    - row (int): byte offset of this row's first byte within ``win``.
    - k0 (int): first bit index within the row to set (inclusive).
    - k1 (int): last bit index within the row to set (inclusive).

    Return
    --------
    - output (None): no return value; sets bits ``k0..k1`` of the row in ``win`` in place.
    """
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

    A claim IS a blocked span, so the build is ``win |= span`` per cell; claim and slab order are
    therefore irrelevant, which is what lets a release swap-remove from the arena. The fold matches
    ``kernel._blocked`` term for term: a corridor claim always blocks; a column claim or an
    always-active wall blocks only when the flight does not own the cell; a static wall covers every
    step, so it subsumes that cell's column claims and they are skipped.

    Parameters
    ------------
    - arena (np.ndarray): flat claim store; each entry packs a start/end step span.
    - slab_start (np.ndarray): per-key start offset into ``arena``.
    - slab_len (np.ndarray): per-key slab length (claims for that key).
    - static_col (np.ndarray): per-cell always-active wall flag.
    - ov_own_gen (np.ndarray): per-cell owning generation of the plan overlay.
    - gen (int): this plan's generation; a cell is owned when ``ov_own_gen[cell] == gen``.
    - qmin (int): global minimum q, mapping a window q index to an arena cell.
    - rmin (int): global minimum r, mapping a window r index to an arena cell.
    - rspan (int): global r-span, the cell-index stride for q.
    - n_levels (int): altitude levels, the cell-index stride for a ``(q, r)`` pair.
    - wbox (np.ndarray): window geometry (the ``W_*`` fields).
    - win (np.ndarray): output bitmap; zeroed over the active region then OR-painted.
    - s0_shift (int): right shift yielding a claim's start step.
    - span_bits (int): right shift for a claim's end-step field.
    - field_mask (int): mask isolating the end step after the ``span_bits`` shift.

    Return
    --------
    - output (int): number of window cells that received at least one blocked span.
    """
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
    """Size the window around the cells a plan is anchored to; return bytes, or 0 if degenerate.

    The window is the anchor-cell bbox padded by ``lateral_margin`` hexes, sized to contain A*'s
    reroute ellipse (see context/figures/search_window.png); the margin is set from the measured
    read bboxes in ``.context/perf/probe_read_window.py``, not from a bound. Steps run from
    ``base`` (nothing is probed earlier — the ground state starts there) to
    ``base + n_gsteps + tail_steps``, clipped to ``max_step``.

    The two failure returns are NOT interchangeable: a window failure is not cheap — no window
    means no answers, so the whole plan falls to the pure-Python reference. That is why an
    over-budget box reports the size it needs rather than collapsing to a single "off".

    Parameters
    ------------
    - cocc (CompiledHexOccupancy): supplies the global cell bounds (qmin/rmin, spans, levels).
    - wbox (np.ndarray): window-geometry array, filled in place with the sized box on success only.
    - q_cells (Sequence[int]): anchor q indices — origin hex, its exit lanes, dest landing lanes.
    - r_cells (Sequence[int]): anchor r indices, paired with ``q_cells``.
    - base (int): first step the window covers; nothing is probed before it.
    - max_step (int): global last step, clipping the window's step span.
    - n_gsteps (int): ground-delay allowance (the two-phase mask already bounds it).
    - lateral_margin (int): hexes added around the anchor bbox to cover the reroute fan.
    - tail_steps (int): steps for the takeoff climb, lane traverse and flight itself.
    - max_bytes (int): buffer ceiling; a box needing more is reported as ``-n`` for a retry.

    Return
    --------
    - output (int): window size in bytes with ``wbox`` filled; ``0`` if the box degenerated
      (clipped to nothing); ``-n`` if it needs ``n`` > ``max_bytes`` bytes (``wbox`` untouched).
    """
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
