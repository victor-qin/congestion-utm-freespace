"""Cost-aware Safe Interval Path Planning (SIPP) on the hex lattice.

A drop-in for the space-time A* planner (:mod:`astar`): same cost model, terminal gating, and output
contract — it returns the **identical optimal weighted cost** — but collapses the per-timestep ``step``
axis into per-cell **safe intervals**, so the air search expands O(cell × interval) nodes instead of
O(cell × step). (Phillips & Likhachev, "SIPP: Safe Interval Path Planning for Dynamic Environments,"
ICRA 2011.)

Because our objective is **weighted cost** (``c_hold ≠ c_gd`` ⇒ earliest-arrival is not cheapest), the
classic single-best-per-state SIPP is unsound here; we keep a **Pareto frontier** of
``(arrival_time, cost)`` per ``(cell, interval)`` with dominance pruning, which recovers exact A*
optimality.

Design: subclass :class:`AStarPlanner` to inherit ``_occupancy`` (occupancy + ``TerminalCapacity``
sync) and ``_build`` (corner→volumes). States are keyed on the A*-shaped tuple
``("g"/"a", q, r, step)`` so ``_committed_arrival``/``_build``/reconstruction run verbatim. The air
reroute is the only lever collapsed into intervals; the ground-wait ray and a goal-cell hover stay
per-step because the terminal capacity gates are per-step (not interval-captured). See the plan file.
"""
from __future__ import annotations

import heapq
import itertools
import math
from array import array

import numpy as np

from ..cost import endpoint_altitude_change_m, trajectory_cost
from ..types import (
    DenialReason,
    IntentStatus,
    OperationalIntent,
    TimedPoint,
    as_terminal,
)
from ..geometry import CylinderSpec
from ..volumes import (
    enroute_detour_m,
    enroute_flown_m,
    enroute_reference_m,
    exit_radius,
    terminal_radius,
)
from . import hexgrid as hg
from . import sipp_window as SW
from .astar import AStarPlanner
from .astar._packed import aligned_2d
from .astar.compiled_hex_occupancy import (
    _FIELD_MASK,
    _S0_SHIFT,
    _SPAN_BITS,
    ground_delay_steps,
    search_horizon,
)
from .astar.planner import _BBOX_HUGE, _BindBatch, _absorb, _committed_arrival

_EPS = 1e-6

# ---- per-plan interval window (`sipp_window`) ----
# The same 24 hexes A* calibrated its dense window on. That number sizes for plan-level COVERAGE —
# the share of plans with ZERO misses, 100.0% there — not for the typical box, because one miss
# forces a widen-and-rerun. SIPP's measured read set is TIGHTER than A*'s (dirty rate 61.5% against
# 78.7% at 4 DROP workers; the interval collapse probes fewer cells), so this errs wide.
_SWINDOW_MARGIN_HEX = 24
_SWINDOW_WIDEN_MAX = 3      # each level doubles the lateral margin; past this, the reference
_SWINDOW_GROW_MAX = 4       # buffer regrowths per plan before giving up on the window


def _deny(req, reason):
    return OperationalIntent(
        request=req, status=IntentStatus.REJECTED, denial_reason=reason, planner="sipp"
    )


class SafeIntervalIndex:
    """Cell-keyed inverse of the committed occupancy — the v2 engine behind SIPP's speedup.

    ``HexOccupancyService`` maps ``step -> {cells}``; to build a cell's safe intervals SIPP needs the
    OPPOSITE (``cell -> occupied steps``). v1 recovered it by scanning ``is_blocked`` over the full
    ``[base, max_step]`` horizon PER CELL (dominated by the empty ground-delay tail) — which made SIPP
    slower than A*. This index instead records, per hex cell, the corridor-blocked steps and the
    per-step column hub-coverage, fed incrementally by the ledger commit hook (the same dual-sweep
    rasterization ``HexOccupancyService`` uses). A cell's safe intervals are then built in
    O(#occupied steps of that cell) — O(1) for the common never-touched cell — and :meth:`cell_blocked`
    exactly replicates ``HexOccupancyService.is_blocked`` (pinned by a test).

    NOTE: storage is not reclaimed on eviction yet — the search only ever reads steps >= the request
    clock (so this is correct), but memory reclaim for very long runs is a follow-up."""

    def __init__(self, cfg, track_removal: bool = False):
        self.cfg = cfg
        self.R = hg.circumradius(cfg)
        self.infl_blocked = cfg.corridor_width_m / 2.0 + self.R
        self.infl_pad = cfg.effective_hover_radius_m + self.R
        # Removal mode (LNS destroy): `corr`/`cols` hold per-step REFCOUNTS instead of sets — two
        # flights' inflated rasters can legitimately cover the same (cell, step), so removing one must
        # not free it — and each committed flight's applied rows are journaled so `on_release` can
        # reverse them exactly, keeping `n_added` in lockstep with the ledger (the shrink tripwire in
        # `_sipp_index` stays silent). The A* structural twin `HexOccupancyService` made the identical
        # choice for the identical reason; this is a port of it, not a parallel invention.
        #
        # Reader-transparent by construction: every consumer of `corr`/`cols` uses `in`, iteration or
        # truthiness (`cell_blocked`, `free_intervals`), all of which behave identically on a `set` and
        # on a `dict`. Flag off => the original set-based structures, byte-for-byte.
        #
        # Journal layout (mode on): `_rows[fid]` is a FLAT int64 array of 4-slot rows
        # `(cell_id, s_lo, s_hi, code)`, not a list of tuples — the tuple form was measured at ~185 B
        # a row against 32 B here, and this structure is per-(cell, span) so it is linear in schedule
        # size. `cell_id` indexes `_cells`, which interns each `(q, r, L)` ONCE and hands the SAME
        # tuple object back to be used as the `corr`/`cols` key: that sharing is what keeps the cost
        # 80 bytes per distinct cell rather than per row. `code`: -1 = corridor, >= 0 = terminal
        # column whose hub id is `_tids[code]`.
        self.track_removal = track_removal
        self._rows: dict[int, array] = {}                          # fid -> flat 4-slot rows
        self._cells: list[tuple[int, int, int]] = []               # cell_id -> (q, r, L)
        self._cell_ids: dict[tuple[int, int, int], int] = {}
        self._tids: list = []                                      # code -> terminal id
        self._tid_ids: dict = {}
        self.corr: dict[tuple[int, int], set[int]] = {}          # cell -> corridor-blocked steps
        self.cols: dict[tuple[int, int], dict[int, set]] = {}    # cell -> step -> {hub_id}
        self.static_cols: dict[tuple[int, int], set] = {}        # always-active: cell -> {hub_id} (step-indep)
        self.n_added = 0
        self.evicted_before: int | None = None

    def _intern(self, cell):
        """``(canonical_cell, cell_id)``, registering the cell if new — port of
        ``HexOccupancyService._intern``. Returning the canonical TUPLE (not just its id) is the point:
        the journal, the `corr` key and the `cols` key then share ONE tuple per distinct cell instead
        of allocating a fresh one per row."""
        cid = self._cell_ids.get(cell)
        if cid is None:
            cid = self._cell_ids[cell] = len(self._cells)
            self._cells.append(cell)
            return cell, cid
        return self._cells[cid], cid

    def _tid_code(self, tid) -> int:
        """Small-int code for a hub id, so a journal row packs into int64."""
        code = self._tid_ids.get(tid)
        if code is None:
            code = self._tid_ids[tid] = len(self._tids)
            self._tids.append(tid)
        return code

    def on_commit(self, flight_id, volumes) -> None:
        hg.prepare_range_cache_for_commit(volumes)
        own_cols = tuple((v.shape.cx, v.shape.cy, v.shape.radius) for v in volumes
                         if v.terminal_id is not None and isinstance(v.shape, CylinderSpec))
        rows = [] if self.track_removal else None
        for v in volumes:
            self._add(v, own_cols, rows)
        self.n_added += len(volumes)
        if self.track_removal:
            entry = self._rows.get(flight_id)
            if entry is None:
                self._rows[flight_id] = array("q", rows)   # 'q' = int64: cell ids, steps and codes
                #                                            cannot overflow it, so packing stays total
            else:
                # NOT optional: `_absorb` groups by `itertools.groupby` over `iter_committed`, which
                # yields a SECOND group for any fid whose volumes are non-contiguous in the ledger.
                entry.extend(rows)

    def _inside_a_column(self, q, r, cols) -> bool:
        c = hg.hex_center(q, r, self.R)
        return any((c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= rad * rad for cx, cy, rad in cols)

    def _add(self, vol, own_cols, _rows=None) -> None:
        tid = vol.terminal_id
        is_column = tid is not None and isinstance(vol.shape, CylinderSpec)
        track = self.track_removal
        # Same shared range producer as `CompiledHexOccupancy._add` and `HexOccupancyService._add`
        # (issue #114) — identical `R`/`infl_*`, so all three hit ONE memoized geometry sweep per
        # commit instead of three. This index is step-keyed, so like the hex service it expands the
        # span back out; the saving here is the sweep and the per-row dispatch, not the storage shape.
        for q, r, L, s_lo, s_hi, in_blk in hg.rasterize_ranges(
            vol, self.cfg, self.R, self.infl_blocked, self.infl_pad
        ):
            if not in_blk:
                continue                                        # is_blocked only consults in_blk cells
            cell = (q, r, L)
            if is_column:
                if track:
                    cell, cid = self._intern(cell)              # canonical tuple becomes the KEY too
                    _rows += (cid, s_lo, s_hi, self._tid_code(tid))
                    d = self.cols.setdefault(cell, {})
                    for s in range(s_lo, s_hi + 1):
                        e = d.setdefault(s, {})
                        e[tid] = e.get(tid, 0) + 1
                else:
                    d = self.cols.setdefault(cell, {})
                    for s in range(s_lo, s_hi + 1):
                        d.setdefault(s, set()).add(tid)
            elif not (own_cols and self._inside_a_column(q, r, own_cols)):
                if track:                                       # (skip own terminal interior)
                    cell, cid = self._intern(cell)
                    _rows += (cid, s_lo, s_hi, -1)
                    d = self.corr.setdefault(cell, {})
                    for s in range(s_lo, s_hi + 1):
                        d[s] = d.get(s, 0) + 1
                else:
                    self.corr.setdefault(cell, set()).update(range(s_lo, s_hi + 1))

    def on_release(self, flight_id, volumes) -> None:
        """Ledger release subscriber (removal mode): reverse the flight's journaled rows so `corr`/`cols`
        stay exact without a rebuild, and keep ``n_added`` in lockstep with the ledger so the shrink
        tripwire stays silent.

        DELIBERATELY NO ``evicted_before`` CLAMP, unlike ``HexOccupancyService.on_release``. That clamp
        is sound there only because its ``evict_before`` physically DELETES the sub-floor buckets and
        ``add_volume`` applies the identical clamp on insert — a matched pair. This structure's
        ``evict_before`` deletes nothing (see its comment), so `_add` recorded the full span and a
        clamped release would leave every step in ``[s_lo, evicted_before)`` permanently
        un-decremented: phantom blocked steps outliving the flight. Record the full span, reverse the
        full span. If reclaim ever lands here, clamp BOTH sides together or neither."""
        rows = self._rows.pop(flight_id)
        corr, cols, cells, tids = self.corr, self.cols, self._cells, self._tids
        for i in range(0, len(rows), 4):                       # flat 4-slot rows; see `__init__`
            cell = cells[rows[i]]
            s_lo, s_hi, code = rows[i + 1], rows[i + 2], rows[i + 3]
            if code < 0:                                        # corridor: cell -> {step: count}
                d = corr[cell]
                for s in range(s_lo, s_hi + 1):
                    c = d[s] - 1                                # KeyError here IS the drift signal
                    if c:
                        d[s] = c
                    else:
                        del d[s]
                if not d:
                    del corr[cell]                              # keep `if not corr` exact
            else:                                               # column: cell -> {step: {tid: count}}
                tid = tids[code]
                d = cols[cell]
                for s in range(s_lo, s_hi + 1):
                    e = d[s]
                    c = e[tid] - 1
                    if c:
                        e[tid] = c
                    else:
                        del e[tid]
                        if not e:
                            del d[s]
                if not d:
                    del cols[cell]
        self.n_added -= len(volumes)

    def evict_before(self, step) -> None:
        if self.evicted_before is None or step > self.evicted_before:
            self.evicted_before = step   # queries read steps >= request clock; storage reclaim is TODO

    def reset(self) -> None:
        self.corr.clear(); self.cols.clear(); self.n_added = 0; self.evicted_before = None
        # The journal describes the structures just cleared, so it MUST go with them: a surviving row
        # would decrement a count the fresh `_absorb` is about to rebuild. `_cells`/`_tids` stay —
        # pure interning pools, value-identical across a rebuild and never read except through a live
        # row (the A* twin keeps its own for the same reason).
        self._rows.clear()
        # static_cols intentionally preserved: always-active walls are infrastructure, not commit-derived

    def register_static_terminal(self, center, term) -> None:
        """Permanently wall a hub's terminal airspace (column + exit lanes) off from FOREIGN traffic
        (``cfg.terminal_airspace_always_active``) — the SafeIntervalIndex twin of
        ``HexOccupancyService.register_static_terminal``. Step-independent; idempotent per hub."""
        tid = as_terminal(term).id
        for cell in hg.terminal_cells(center, term, self.cfg):
            self.static_cols.setdefault(cell, set()).add(tid)

    _on_static = register_static_terminal   # ledger.subscribe_static hook name (main's A* contract)

    def cell_blocked(self, q, r, L, s, own, fixed_lanes) -> bool:
        """Exact replica of ``HexOccupancyService.is_blocked(q, r, L, s, own)`` — per-level ``cols``/``corr``
        plus the always-active ``static_cols`` walls, which are level-INDEPENDENT (a foreign hub column
        walls (q, r) at every flight level). Foreign in EITHER the per-step column OR the static set ⇒
        blocked."""
        cc = self.cols.get((q, r, L))
        hubs = cc.get(s) if cc else None
        stat = self.static_cols.get((q, r)) if self.static_cols else None   # level-independent
        if hubs is not None or stat is not None:
            if (hubs is not None and any(t not in own for t in hubs)) or \
                    (stat is not None and any(t not in own for t in stat)):
                return True                                     # foreign column (transient or static) → wall
            return fixed_lanes and s in self.corr.get((q, r, L), ())   # own-only column + sibling corridor
        return s in self.corr.get((q, r, L), ())

    def free_intervals(self, q, r, L, own, base, max_step, fixed_lanes):
        """Maximal free ``[lo,hi]`` step-runs in ``[base,max_step]`` for cell ``(q, r, L)`` — complement of
        its blocked steps. O(#occupied steps of the cell); O(1) for a never-occupied cell (the common case)."""
        stat = self.static_cols.get((q, r)) if self.static_cols else None
        if stat is not None and any(t not in own for t in stat):
            return []                            # always-active FOREIGN wall ⇒ blocked at EVERY step/level
        corr = self.corr.get((q, r, L))
        cols = self.cols.get((q, r, L))
        if not corr and not cols:
            return [(base, max_step)]
        cand = set()
        if corr:
            cand.update(s for s in corr if base <= s <= max_step)
        if cols:
            cand.update(s for s in cols if base <= s <= max_step)
        blk = sorted(s for s in cand if self.cell_blocked(q, r, L, s, own, fixed_lanes))
        out, lo = [], base
        for s in blk:
            if s > lo:
                out.append((lo, s - 1))
            lo = s + 1
        if lo <= max_step:
            out.append((lo, max_step))
        return out


class _SafeIntervals:
    """Per-plan memoised view over :class:`SafeIntervalIndex`: a cell's free intervals for THIS flight
    (its ``own`` terminals, step domain, and the ``fixed_lanes`` flag)."""

    def __init__(self, sidx, own, base, max_step, fixed_lanes):
        self.sidx = sidx
        self.own = own
        self.base = base
        self.max_step = max_step
        self.fixed_lanes = fixed_lanes
        self._cache: dict[tuple[int, int, int], list[tuple[int, int]]] = {}

    def intervals(self, q, r, L):
        iv = self._cache.get((q, r, L))
        if iv is None:
            iv = self.sidx.free_intervals(q, r, L, self.own, self.base, self.max_step, self.fixed_lanes)
            self._cache[(q, r, L)] = iv
        return iv

    def index_of(self, q, r, L, step):
        for i, (lo, hi) in enumerate(self.intervals(q, r, L)):
            if lo <= step <= hi:
                return i
        return -1   # step is blocked (no interval) — the search never targets such a step


def _nondominated(frontier, key, t, g, w):
    """Weighted-SIPP Pareto insert at ``key=(q, r, interval)`` on ``(arrival_time, cost)``.

    The *only* in-air wait is hover at rate ``w = c_air_hold_per_s``, so an EARLIER, cheaper label can
    reproduce a LATER one by hovering forward — but it pays for it. Hence the dominance is **not** plain
    ``(t2<=t and g2<=g)`` (which wrongly prunes a later arrival that was reached via cheap upfront ground
    delay, forcing expensive goal-hover instead — observed as ``(c_hold-c_gd)·dt`` cost gaps vs A*).
    Stored ``(t2,g2)`` dominates new ``(t,g)`` iff it is no later AND can hover to ``t`` for ``<= g``:
    ``t2 <= t and g2 + (t - t2)*w <= g``. Symmetric for eviction. Returns False ⇒ caller skips."""
    F = frontier.get(key)
    if F is None:
        frontier[key] = [(t, g)]
        return True
    evict = False
    for (t2, g2) in F:
        if t2 <= t and g2 + (t - t2) * w <= g + 1e-9:
            return False                                   # dominated by a stored label → skip
        if t <= t2 and g + (t2 - t) * w <= g2 + 1e-9:
            evict = True                                   # new label dominates this stored one
    if evict:                                              # rare: rebuild dropping now-dominated labels
        frontier[key] = [(t2, g2) for (t2, g2) in F if not (t <= t2 and g + (t2 - t) * w <= g2 + 1e-9)]
        frontier[key].append((t, g))
    else:
        F.append((t, g))                                   # common case: append in place (no realloc)
    return True


class SIPPPlanner(AStarPlanner):
    """Safe-interval cost-aware planner; inherits occupancy/terminal sync and corridor build from A*."""

    @property
    def needs_blocked_map(self) -> bool:
        """Keyed on ``sipp_compiled``, NOT on the inherited ``compiled``.

        SIPP does call ``HexOccupancyService.is_blocked`` — but only from ``_succ``, the pure-Python
        REFERENCE successor generator, never from ``_splan_compiled``, whose kernel answers the
        obstacle test out of the interval pool. So on the compiled path the map is written on every
        commit and read never: measured **1.807 ms/flight** at density_faa scale (5.562 against
        3.755), at a fallback rate of zero. ``_splan_reference`` arms it with ``enable_blocked``
        instead — A*'s existing lazy, sticky re-arm.

        A property rather than a class attribute because ``AStarPlanner._occupancy`` derives
        ``maintain_blocked`` from ``self.compiled``, which is **A*'s fallback-kernel flag**; SIPP
        dispatches on ``self.sipp_compiled``, and ``_swarm_jit``'s failure handler clears only the
        latter. A plain ``False`` would therefore make a JIT-degraded SIPP — every plan on the
        reference — turn the map off AND pay an O(schedule) ``enable_blocked`` replay on its first
        plan, then maintain it anyway: strictly more work than before this was touched, for no
        saving. Reading ``sipp_compiled`` gives the map back to exactly the planner that needs it.
        """
        return not self.sipp_compiled

    def __init__(self, max_expansions: int = 1 << 21, compiled: bool = True,
                 kernel_log2_min: int | None = None, incremental_release: bool = False,
                 **astar_kw):
        # Default budget is aligned with the compiled kernel's label cap (``_k_max = 1<<21``): the
        # pure-Python reference is the kernel's correctness ORACLE, so it must be able to reach at least
        # as far — a long multi-altitude flight can need ~700k expansions (3× the 2D count), which the old
        # 600k default truncated while the kernel (bigger cap) found the identical optimum. Only affects
        # ``_plan_reference`` / the A* ``_fallback``; the compiled path caps on ``_k_max`` directly.
        # A* owns ``self.compiled`` for the kernel used by our safety fallback. SIPP's kernel needs an
        # independent flag: an A* warm-up failure must remain recorded so ``_fallback`` dispatches to
        # A*'s reference path rather than calling a missing ``self._kernel``.
        # `kernel_log2_min` and `incremental_release` are A*'s, and BOTH matter to SIPP: a parallel
        # LNS worker passes the first (dropping it would silently run the A* fallback at the wrong
        # array floor), and the second release-hooks the two structures SIPP inherits — `_svc` and
        # `_tcap`, via `_occupancy`, which SIPP calls on its own hot path. SIPP's OWN two structures
        # read `self.incremental_release` in `_sipp_index`/`_scompiled_occ` below.
        # `**astar_kw` forwards A*-owned knobs SIPP does not interpret itself — today `window_bytes`
        # (#124's dense-window budget), which the A* fallback inside SIPP still honours.
        super().__init__(max_expansions, compiled=compiled, kernel_log2_min=kernel_log2_min,
                         incremental_release=incremental_release, **astar_kw)
        self._sidx: SafeIntervalIndex | None = None    # cell-keyed inverse index (per ledger)
        self._sidx_ledger = None
        self._sidx_epoch = 0                           # ledger.epoch at bind (detach tripwire, #109)
        # --- compiled (numba) air-cruise kernel; falls back to the pure-Python reference ---
        self.sipp_compiled = compiled
        self._skernel = None
        if compiled:
            try:
                from .sipp_kernel import _search
                self._skernel = _search
            except ImportError:
                self.sipp_compiled = False              # SIPP kernel absent → pure-Python SIPP
        self._k_iv_lo = self._k_iv_hi = self._k_iv_nxt = self._k_scratch = None
        self._k_wbox = SW.empty_wbox()  # per-plan window geometry; W_STEPS == 0 is OFF
        self._swin_widen = 0            # plans that grew their window and re-ran (diagnostics)
        self._swin_grown = 0            # plans that grew the interval buffers
        self._k_cap = -1                               # frontier size the kernel arrays are sized to
        self._k_lab_cell = None                        # kernel work arrays (allocated lazily)
        self._k_out_q = None
        self._sfb = 0                                   # kernel→A* fallbacks (diagnostics/tests)
        self._sfb_cap = 0                               # of which: label/heap overflow (hard/infeasible flight)
        self._sfb_oob = 0                               # of which: window miss at the widen ceiling
        self._sfb_overlap = 0                           # of which: own-foreign column (issue #3)
        self._sfb_hash = 0                              # of which: (cell, step) best-g table saturated
        self._n_expansions = 0                         # kernel expansions on the last compiled plan
        self._air = []                                  # last successful compiled path (diagnostics/tests)
        if self.sipp_compiled:
            self._swarm_jit()

    def _swarm_jit(self):
        """Compile the SIPP kernel once at construction with a tiny synthetic input, off the hot path —
        the twin of ``AStarPlanner._warm_jit``, and NOT optional under LNS.

        ``super().__init__`` warms the A* kernel (our safety fallback), but nothing warmed this one.
        That matters most exactly where it is least visible: ``LNSWorkerPool.start`` pre-imports
        ``astar.kernel`` on the argument that "the schedule being improved came out of an in-process A*
        run, so the on-disk cache=True artifact exists and the workers load it instead of racing to
        compile it". The LNS baseline is A*, so that argument does NOT extend to SIPP — with
        ``repair_planner='sipp'`` every spawned worker would meet an uncompiled kernel on its FIRST
        repair, all at once, which is the compile stampede that comment exists to prevent.

        Shapes mirror the production call so the compiled signature is the one actually used; a
        failure degrades to the pure-Python reference rather than propagating, as A*'s does."""
        if self._skernel is None:
            return
        try:
            cap, nlev, qspan, rspan = 9, 1, 3, 3
            maxs, ncap = 5, 64
            # int32 deliberately: `_skernel_state` allocates the production window pool as int32,
            # and
            # dtype is part of a numba specialization, so warming int64 here would compile a
            # signature
            # no real plan calls.
            iv_lo = np.zeros(cap, np.int32)
            iv_hi = np.full(cap, maxs, np.int32)
            iv_nxt = np.full(cap, -1, np.int32)
            gp = aligned_2d(ncap, 4)                      # same packed layout as production
            gp[:, 1] = 0
            self._skernel(
                iv_lo, iv_hi, iv_nxt,
                0, 0, rspan, qspan, 0, maxs, nlev,
                np.array([0], np.int64), np.array([0.0]), np.array([0], np.int64), 1,
                np.ones(nlev, np.bool_), 1, 1.0,
                np.array([0], np.int64), np.array([0.0]),
                np.array([1], np.int64), np.array([0.0]),
                np.zeros(cap, np.int64), np.zeros(cap, np.float64),
                np.array([0], np.int64), np.array([maxs], np.int64), np.array([0, 1], np.int64),
                3.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                1, np.full(cap, -1, np.int64), np.full(cap, -1, np.int64),
                np.zeros(cap, np.int64),
                np.zeros(ncap, np.int64), np.zeros(ncap, np.int64), np.zeros(ncap, np.int64),
                np.zeros(ncap, np.float64), np.full(ncap, -1, np.int64), np.full(ncap, -1, np.int64),
                np.full(ncap, -1, np.int64), np.full(ncap, -1, np.int64), ncap,
                np.zeros(ncap, np.float64), np.zeros(ncap, np.int64), np.zeros(ncap, np.int64), ncap,
                gp, gp.view(np.float64), ncap, 6, maxs + 2,
                np.zeros(32, np.int64), np.zeros(32, np.int64),
                np.zeros(32, np.int64), np.zeros(32, np.int64),
                np.zeros(8, np.int64),
            )
            # `build_window_intervals` is decorated separately from `_search` and owns its own numba
            # cache, so it needs its own warm call under this same guard — measured ~0.92 s cold and
            # ~132 ms even off a warm on-disk cache, against ~5 us hot. Without it every spawned DROP
            # worker pays that on its FIRST repair, all at once, which is precisely the compile
            # stampede this method exists to prevent. Mirrors `AStarPlanner._warm_jit`'s
            # `build_window_claims` call.
            warm_wbox = SW.empty_wbox()
            warm_wbox[SW.W_Q1] = warm_wbox[SW.W_R1] = 0
            warm_wbox[SW.W_S1] = maxs
            warm_wbox[SW.W_RSPAN] = 1
            warm_wbox[SW.W_STEPS] = maxs + 1
            SW.build_window_intervals(
                np.zeros(1, np.int64), np.zeros(2 * cap, np.int64), np.zeros(2 * cap, np.int64),
                np.zeros(cap, np.bool_), np.zeros(cap, np.int32), 1,
                0, 0, 3, 1, warm_wbox,
                np.zeros(cap, np.int32), np.zeros(cap, np.int32), np.full(cap, -1, np.int32),
                np.zeros(cap, np.int64), _S0_SHIFT, _SPAN_BITS, _FIELD_MASK,
            )
        except TypeError:
            # Arity/dtype drift between this call and `_search`'s signature is a BUG, not a
            # numba-availability problem. Swallowing it into `sipp_compiled = False` would route
            # every
            # plan to the pure-Python reference: the run stays exact, passes every parity gate, and
            # merely reports a massive slowdown — the least detectable failure this file can have.
            raise
        except Exception as e:                            # compile failure → degrade to pure Python
            import warnings as _w

            _w.warn(
                f"sipp numba kernel failed to warm/compile ({e!r}); falling back to the pure-Python "
                f"SIPP reference for ALL plans. Install a compatible numba, or clear stale .nbi/.nbc "
                f"caches after a kernel-signature change.",
                RuntimeWarning, stacklevel=2,
            )
            self.sipp_compiled = False
            self._skernel = None

    def _sipp_index(self, req, ledger, cfg) -> "SafeIntervalIndex":
        """Maintain the SafeIntervalIndex in lockstep with the ledger (mirrors ``_occupancy``): first use
        subscribes the commit hook + absorbs existing volumes; a ledger shrink rebuilds; then evict to
        the request clock."""
        sidx = self._sidx
        if sidx is not None and self._sidx_ledger is ledger and self._sidx_epoch != ledger.epoch:
            sidx = None     # detached mid-life: re-subscribe and re-absorb (the shrink tripwire
            #                 below cannot see it — a release + re-commit nets to the same n_volumes)
        if sidx is None or self._sidx_ledger is not ledger:
            sidx = self._sidx = SafeIntervalIndex(cfg, track_removal=self.incremental_release)
            self._sidx_ledger = ledger
            self._sidx_epoch = ledger.epoch
            ledger.subscribe(sidx.on_commit)
            if self.incremental_release:               # removal hook: release_many un-absorbs
                ledger.subscribe_release(sidx.on_release)
            _absorb(sidx, ledger)
            ledger.subscribe_static(sidx._on_static)   # replays every hub registered before we bound
        elif ledger.n_volumes < sidx.n_added:
            sidx.reset()                      # preserves static_cols (walls are infrastructure)
            _absorb(sidx, ledger)
        # `evict_floor` (Track A / LNS): out-of-order repair must evict to the COORDINATOR's clock,
        # never this flight's — the floor only ever evicts LESS. Mirrors `AStarPlanner._occupancy`.
        wm = req.t_request if self.evict_floor is None else min(req.t_request, self.evict_floor)
        sidx.evict_before(int(wm // cfg.dt_s))
        return sidx

    def plan(self, req, ledger, cfg):
        """Dispatch to the compiled safe-interval kernel, with the pure-Python reference as the fallback
        when numba is absent, legacy terminal folding is requested, a flight strays out of the kernel
        box, or a capacity valve trips."""
        # Per-plan diagnostics must never leak a previous compiled flight through an early host denial,
        # a reference dispatch, or a kernel fallback. Cumulative fallback counters remain cumulative.
        self.last_expansions = 0
        self._n_expansions = 0
        self._air = []
        # Never leak a previous flight's read set to a consumer. NOT redundant with A*'s own reset:
        # `_fallback` runs `AStarPlanner.plan`, which SETS `last_envelope`, and a subsequent NATIVE
        # SIPP plan sets nothing — so without this line `try_repair` files the fallback flight's read
        # set under the next flight's name, and a DROP coordinator merges a genuinely stale repair
        # (a worse cost, not a conflict, so `verify` cannot catch it).
        self.last_envelope = None
        if not self.sipp_compiled:
            return self._splan_reference(req, ledger, cfg)
        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        if (o_term is not None or d_term is not None) and not cfg.fixed_exit_lanes:
            return self._splan_reference(req, ledger, cfg)   # legacy-terminal landing needs _committed_arrival
        return self._splan_compiled(req, ledger, cfg)

    def _splan_reference(self, req, ledger, cfg):
        # ---- setup: identical to AStarPlanner.plan (so cost/terminals/output match exactly) ----
        dt = cfg.dt_s
        pitch = cfg.nominal_speed_mps * dt
        R = hg.circumradius(cfg)
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        t_depart = req.t_departure
        base = int(math.ceil(t_depart / dt))
        levels = cfg.flight_levels_m
        ground_z = cfg.ground_level_m
        c_alt = cfg.cost_altitude_change_per_m
        # per-level takeoff climb-steps/cost + per-rung (L↔L+1) mid-route change — mirror AStarPlanner.plan
        takeoff_steps = tuple(cfg.climb_steps_to(z) for z in levels)
        takeoff_cost = tuple(c_alt * (z - ground_z) for z in levels)
        rung_steps = tuple(max(1, int(math.ceil((levels[L + 1] - levels[L]) / (cfg.climb_rate_mps * dt))))
                           for L in range(len(levels) - 1))
        rung_cost = tuple(c_alt * (levels[L + 1] - levels[L]) for L in range(len(levels) - 1))

        oq, orr = hg.enu_to_axial(origin[0], origin[1], R)
        gq, grr = hg.enu_to_axial(dest[0], dest[1], R)
        gx, gy = R * hg.SQRT3 * (gq + grr / 2.0), R * 1.5 * grr
        straight = float(np.linalg.norm(dest[:2] - origin[:2]))
        # lane->lane reference: the SAME ruler main's A* gates and reports against (issue #50),
        # so SIPP's air_detour_m/cost cannot drift from A*'s.
        straight_ref = enroute_reference_m(origin, dest, req.origin_terminal, req.dest_terminal, cfg)

        svc = self._occupancy(req, ledger, cfg)
        # THIS is the search that reads `blocked` (via `_succ`), and on a compiled planner
        # the service was built not maintaining it. Arm it here — before any `is_blocked` —
        # rather than lazily inside the query: the service holds no ledger, so a query that
        # could not rebuild would have to choose between a wrong answer and a raise deep
        # inside the oracle. Sticky, so a run that falls back repeatedly pays the
        # O(schedule) rebuild once. Mirrors `AStarPlanner._plan_reference`.
        svc.enable_blocked(ledger)
        sidx = self._sipp_index(req, ledger, cfg)
        tcap = self._tcap
        dwell_steps = tuple(max(1, int(math.ceil((cfg.hover_time_s + cfg.climb_time_to(z)) / dt)))
                            for z in levels)          # per-level dwell (hover + actual climb to that level)

        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        own = frozenset(t.id for t in (o_term, d_term) if t is not None)
        o_cap = o_term.capacity if o_term else 1
        d_cap = d_term.capacity if d_term else 1

        fixed_lanes = cfg.fixed_exit_lanes
        o_lanes = hg.terminal_lanes(origin, o_term, cfg) if fixed_lanes and o_term is not None else []
        d_lanes = hg.terminal_lanes(dest, d_term, cfg) if fixed_lanes and d_term is not None else []
        d_lane_by_cell = {L.cell: L for L in d_lanes}
        h_off = max((L.dist for L in d_lanes), default=0.0)
        o_r = terminal_radius(o_term, cfg) if o_term is not None else 0.0

        sqrt3, c_lat = hg.SQRT3, cfg.cost_air_lateral_per_m
        d_r = exit_radius(d_term, cfg) if d_lanes else 0.0
        goal_cost_by_cell = {lane.cell: c_lat * (lane.dist - d_r) for lane in d_lanes}
        goal_cost_lb = min(goal_cost_by_cell.values(), default=0.0)

        def h_air(q, r, L):
            dx, dy = R * sqrt3 * (q + r / 2.0) - gx, R * 1.5 * r - gy
            return (c_lat * max(0.0, math.sqrt(dx * dx + dy * dy) - h_off)
                    + takeoff_cost[L] + goal_cost_lb)

        dx0, dy0 = R * sqrt3 * (oq + orr / 2.0) - gx, R * 1.5 * orr - gy
        h_off_o = terminal_radius(o_term, cfg) if o_lanes else 0.0   # takeoff charges only (dist - o_r)
        h_ground = (c_lat * max(0.0, math.sqrt(dx0 * dx0 + dy0 * dy0) - h_off - h_off_o)
                    + 2.0 * takeoff_cost[0] + goal_cost_lb)

        n_hops = int(math.ceil(max(straight, pitch) / pitch))
        climb_span = (int(math.ceil((levels[-1] - levels[0]) / (cfg.climb_rate_mps * dt)))
                      if cfg.n_levels > 1 else 0)      # extra step budget for mid-route rungs
        max_step = search_horizon(base, max(takeoff_steps) + max((ln.steps for ln in o_lanes), default=0),
                                  n_hops, climb_span, cfg)
        ground_max_step = base + ground_delay_steps(cfg)

        SI = _SafeIntervals(sidx, own, base, max_step, fixed_lanes)
        came: dict = {}

        def is_goal_cell(q, r):
            if d_term is not None and fixed_lanes:
                return (q, r) in d_lane_by_cell
            return q == gq and r == grr

        def goal_ok(st):
            q, r, L, s = st[1], st[2], st[3], st[4]     # air state now carries the flight level L
            if d_term is not None and fixed_lanes:
                return (q, r) in d_lane_by_cell and tcap.dwell_ok(d_term, dest, s * dt, d_cap, z=levels[L])
            if not (q == gq and r == grr):
                return False
            if d_term is not None:
                arr = _committed_arrival(st, came, R, dt, cfg, origin, dest, o_term, d_term)
                return tcap.dwell_ok(d_term, dest, arr, d_cap, origin, levels[L])
            return svc.pad_clear(gq, grr, s, dwell_steps[L])

        # ---- cost-aware safe-interval search; AS = ("g"/"a", q, r, step), A*-shaped ----
        start = ("g", oq, orr, base)
        g = {start: 0.0}
        wait_steps: dict = {}
        frontier: dict = {}
        counter = itertools.count()
        c_hold = cfg.cost_air_hold_per_s
        pq = [(h_ground, next(counter), start, 0.0, -1)]   # heap: (f, tie, AS, g, interval-index)
        goal_state = None
        goal_score = math.inf
        expansions = 0
        truncated = False

        while pq:
            fst, _, st, gst, iv = heapq.heappop(pq)
            if goal_state is not None and fst >= goal_score:
                break                                      # every remaining label has an objective LB >= incumbent
            if gst > g.get(st, math.inf):
                continue                                   # stale (a cheaper label for this AS won)
            if st[0] == "a" and is_goal_cell(st[1], st[2]) and goal_ok(st):
                # The lane-cell→terminal-edge segment is not a lattice edge, but it is part of the
                # reported en-route distance. Score it here and keep searching until the open-set lower
                # bound proves the best feasible lane exact; equal-hop lanes can have different radii.
                score = gst + takeoff_cost[st[3]] + goal_cost_by_cell.get((st[1], st[2]), 0.0)
                if score < goal_score:
                    goal_state, goal_score = st, score
                if fst >= goal_score:
                    break
            expansions += 1
            if expansions > self.max_expansions:
                truncated = True
                break
            for nst, cost, w, niv in self._succ(
                st, iv, SI, cfg, pitch, levels, takeoff_steps, takeoff_cost, rung_steps, rung_cost,
                dwell_steps, own, o_cap, o_term, origin, tcap, dest, o_lanes, o_r, fixed_lanes,
                ground_max_step, max_step, is_goal_cell,
            ):
                ng = gst + cost
                if ng >= g.get(nst, math.inf):
                    continue                               # a same cell-step label is already ≤ this cost
                # Pareto frontier applies only to ordinary AIR cruise cells; the ground ray (niv=-1) and
                # goal cells (per-step landing gate the frontier can't see) are exempt. `niv` comes from
                # _succ (no index_of). The g early-out above means this fires only for cost-improving
                # successors — the dominant cost saver (the frontier check is otherwise ~half the run).
                # The frontier key includes the flight level L (nst[3]): an interval index is per-(cell, L),
                # so two levels' interval 0 are distinct staircases; the step is now nst[4].
                if niv >= 0 and not is_goal_cell(nst[1], nst[2]) and \
                        not _nondominated(frontier, (nst[1], nst[2], nst[3], niv), nst[4] * dt, ng, c_hold):
                    continue                               # dominated at its (cell, level, interval) → prune
                g[nst] = ng
                came[nst] = st
                wait_steps[nst] = w
                hh = h_air(nst[1], nst[2], nst[3]) if nst[0] == "a" else h_ground
                heapq.heappush(pq, (ng + hh, next(counter), nst, ng, niv))

        self.last_expansions = expansions          # search-effort telemetry (mirrors A*; benchmarks/tests)
        if truncated:
            return _deny(req, DenialReason.SEARCH_EXHAUSTED)
        if goal_state is None:
            return _deny(req, DenialReason.BUDGET_EXCEEDED)

        # ---- reconstruct, re-expanding folded reroute waits so cruise_wps matches A*'s per-step list ----
        path = [goal_state]
        while path[-1] != start:
            path.append(came[path[-1]])
        path.reverse()
        expanded = []
        for i, cur in enumerate(path):
            expanded.append(cur)
            if i + 1 < len(path):
                w = wait_steps.get(path[i + 1], 0)         # hover steps spent IN cur before the move
                for k in range(1, w + 1):                  # w>0 only for an air reroute ⇒ cur is a 5-tuple
                    expanded.append((cur[0], cur[1], cur[2], cur[3], cur[4] + k))
        air = [s for s in expanded if s[0] == "a"]
        lane_steps = {ln.cell: ln.steps for ln in o_lanes}
        ground_steps = (air[0][4] - takeoff_steps[air[0][3]] - base
                        - lane_steps.get((air[0][1], air[0][2]), 0))   # issue #52
        delay = ground_steps * dt

        cruise_wps: list[TimedPoint] = [
            (np.array([*hg.hex_center(q, r, R), levels[L]]), s * dt)
            for (_, q, r, L, s) in air
        ]
        volumes, centerline, _cum_horiz, n_hover = self._build(
            cruise_wps, origin, dest, base, ground_steps, cfg,
            origin_term=req.origin_terminal, dest_term=req.dest_terminal,
        )
        flown = enroute_flown_m([p for p, _ in centerline], origin, dest,
                                req.origin_terminal, req.dest_terminal, cfg)
        if straight_ref > _EPS and flown / straight_ref > cfg.max_detour_factor:
            return self._file_deny(req, DenialReason.BUDGET_EXCEEDED, volumes, ledger)
        if ledger.any_conflict(volumes):
            return self._file_deny(req, DenialReason.CONFLICT_FILED, volumes, ledger)
        detour = enroute_detour_m(flown, straight_ref)

        # true vertical travel: takeoff climb + every cruise layer change + landing descent (mirror A*)
        z_takeoff, z_land = levels[air[0][3]], levels[air[-1][3]]
        cruise_dz = sum(abs(levels[air[i + 1][3]] - levels[air[i][3]]) for i in range(len(air) - 1))
        intent = OperationalIntent(
            request=req,
            status=IntentStatus.ACCEPTED,
            volumes=volumes,
            centerline=centerline,
            ground_delay_s=delay,
            air_hold_s=n_hover * dt,
            air_detour_m=detour,
            lattice_overhead_m=hg.lattice_overhead_m([(t[1], t[2]) for t in air], pitch, detour),
            altitude_change_m=endpoint_altitude_change_m(z_takeoff, z_land, cruise_dz, cfg),
            planner="sipp",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent

    # ================= compiled (numba) air-cruise path (Phase 1: non-terminal) =================
    def share_occupancy_from(self, master) -> None:
        """Plan against MASTER's committed occupancy (``cocc``/``svc``/``tcap``/``sidx``) without
        subscribing the ledger hook or re-absorbing — for optimistic-batch worker threads (#8 Track A).
        The caller must keep the ledger FROZEN (no commits) while workers plan in parallel; each worker
        keeps its OWN kernel state (``_k_*``), so the shared mutations are the benign ``evict_before``
        watermark and — since a worker's reference dispatch arms the map — ``enable_blocked`` on the
        master's ``_svc``. That second one is a real (if one-shot, sticky) cost leaked worker-to-master;
        it is not a race today because both runners use ``spawn`` and ``PARALLEL_PLANNERS`` excludes
        every ``sipp*`` name, so no two of these share a service in-process."""
        self._svc = master._svc
        self._svc_ledger = master._svc_ledger
        self._tcap = master._tcap
        self._svc_epoch = master._svc_epoch
        self._cocc = master._cocc                  # A*'s claim arena, shared with its own fallback
        self._cocc_ledger = master._cocc_ledger
        self._cocc_epoch = master._cocc_epoch
        self._sidx = master._sidx                  # only bound if the master ever took the reference
        self._sidx_ledger = master._sidx_ledger
        self._sidx_epoch = master._sidx_epoch

    def _skernel_state(self, cocc, n_slots: int):
        """Work arrays for one plan, sized to the WINDOW rather than to the whole box.

        Returns A*'s ``(ks, kc)`` pair, EXTENDED with SIPP's own arrays rather than replacing it.
        That is not tidiness: ``AStarPlanner._build_overlay`` — which this planner now reuses verbatim
        for own-column transparency — reads ``self._ks["ov_own_gen"]`` directly, and the A* fallback
        reaches ``_kernel_state``/``_build_window``, which read ``ks["NC"]``, ``ks["out_q"]``,
        ``ks["win"]``, ``ks["wbox"]`` and ``ks["win_stats"]``. Assigning a dict with only SIPP's keys
        would ``KeyError`` on the first terminal plan and again on the first ``FB_CAP``.

        ``n_slots`` is what the window build needs (head slot per window cell + overflow intervals).
        The frontier is per-SLOT and the goal arrays per-CELL, but both are grown against the same
        number: over-allocating the goal arrays by the overflow count is a rounding error next to the
        ``cocc.cap`` (654,266 slots) they used to be sized from, and one growth trigger cannot drift.
        """
        ks, kc = super()._kernel_state(cocc, self._log2_cap_max)
        if self._k_cap < n_slots:
            self._k_cap = max(n_slots, 2 * self._k_cap)   # amortise: a widen doubles the window
            self._k_front_head = np.full(self._k_cap, -1, np.int64)
            self._k_front_tail = np.full(self._k_cap, -1, np.int64)   # sorted-by-arr staircase per slot
            self._k_front_gen = np.zeros(self._k_cap, np.int64)
            self._k_goal_gen = np.zeros(self._k_cap, np.int64)        # per-cell goal flag (stamped)
            self._k_goal_cost = np.zeros(self._k_cap, np.float64)     # lane-cell → terminal-edge cost
            self._k_iv_lo = np.zeros(self._k_cap, np.int32)           # the per-plan window pool
            self._k_iv_hi = np.zeros(self._k_cap, np.int32)
            self._k_iv_nxt = np.full(self._k_cap, -1, np.int32)
            # `build_window_intervals` requires `scratch` at least as long as `iv_lo`: its capacity
            # pass bounds the window's TOTAL claim count, and one cell cannot exceed the total, so a
            # single sizing rule covers both and there is only ever one shortfall to report.
            self._k_scratch = np.zeros(self._k_cap, np.int64)
        self._k_read_bbox = ks["read_bbox"]           # shared with A*: `_mk_envelope` is inherited
        if self._k_lab_cell is None:                     # labels + heap: allocate once
            ml = 1 << 21
            self._k_max = ml
            self._k_lab_cell = np.empty(ml, np.int64)
            self._k_lab_slot = np.empty(ml, np.int64)
            self._k_lab_arr = np.empty(ml, np.int64)
            self._k_lab_g = np.empty(ml, np.float64)
            self._k_lab_par = np.empty(ml, np.int64)
            self._k_lab_next = np.empty(ml, np.int64)
            self._k_lab_prev = np.empty(ml, np.int64)      # doubly-linked sorted frontier
            self._k_lab_dead = np.full(ml, -1, np.int64)   # version-stamped: == gen ⇒ evicted, skip at pop
            self._k_heap_f = np.empty(ml, np.float64)
            self._k_heap_c = np.empty(ml, np.int64)
            self._k_heap_n = np.empty(ml, np.int64)
            # per-(cell, step) best-g dedup table: the reference's `g` dict, as a version-stamped
            # open-addressing hash of packed 32 B records (same layout as astar_kernel; see _packed).
            # Sized to the label cap so the load factor stays low — distinct (cell, step) states are
            # far fewer than labels. Saturation reports FB_HASH and falls back exactly like FB_CAP.
            self._k_hash_log2 = 21
            self._k_hash_cap = 1 << self._k_hash_log2
            gp = aligned_2d(self._k_hash_cap, 4)
            gp[:, 1] = 0                               # gen 0 = empty; the stamp starts above it
            self._k_gpack = gp
            self._k_gpackf = gp.view(np.float64)
        if self._k_out_q is None or self._k_out_q.shape[0] < cocc.MAXS + 8:
            self._k_out_q = np.empty(cocc.MAXS + 8, np.int64)
            self._k_out_r = np.empty(cocc.MAXS + 8, np.int64)
            self._k_out_s = np.empty(cocc.MAXS + 8, np.int64)
            self._k_out_L = np.empty(cocc.MAXS + 8, np.int64)      # flight level per output waypoint
        return ks, kc

    def _fallback(self, req, ledger, cfg):
        """Fallback when the compiled kernel bails (``FB_OOB``/``FB_CAP``): run **A\\*** — the superclass
        search — rather than the pure-Python SIPP reference.

        The flights that overflow the kernel are the hard / near-infeasible ones (e.g. always-active
        walled-in hubs), i.e. SIPP's *worst* regime: no early goal to terminate on, so the cost-aware
        Pareto search fans out (the ~``max_ground_delay/dt``-deep ground-delay fan × fragmented intervals)
        until the label cap. The pure-Python SIPP reference re-does that same explosion in interpreted
        Python (~38 s measured); A\\* reaches the identical accept/deny verdict ~9× faster (~4 s) because
        its per-node is C-level and it has no ground-delay Pareto fan. A\\* shares this planner's
        ``self._svc``/``self._tcap`` (inherited ``_occupancy``), so there is no occupancy re-sync."""
        intent = AStarPlanner.plan(self, req, ledger, cfg)
        if intent is not None:
            intent.planner = "sipp"                    # attribute to the selected planner (A* is internal)
        return intent

    def _file_deny(self, req, reason, volumes, ledger):
        """Preserve A*'s filed-corridor telemetry while attributing the native SIPP denial to SIPP."""
        intent = super()._file_deny(req, reason, volumes, ledger)
        intent.planner = "sipp"
        return intent

    def _splan_compiled(self, req, ledger, cfg):
        from .sipp_kernel import FB_CAP, FB_HASH, FB_OOB, NO_PATH
        dt = cfg.dt_s
        pitch = cfg.nominal_speed_mps * dt
        R = hg.circumradius(cfg)
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        base = int(math.ceil(req.t_departure / dt))
        levels = cfg.flight_levels_m
        nlev = cfg.n_levels
        ground_z = cfg.ground_level_m
        c_alt = cfg.cost_altitude_change_per_m
        takeoff_steps = tuple(cfg.climb_steps_to(z) for z in levels)          # per-level climb-steps/cost
        takeoff_cost = tuple(c_alt * (z - ground_z) for z in levels)
        rung_steps = tuple(max(1, int(math.ceil((levels[L + 1] - levels[L]) / (cfg.climb_rate_mps * dt))))
                           for L in range(nlev - 1))
        rung_cost = tuple(c_alt * (levels[L + 1] - levels[L]) for L in range(nlev - 1))
        oq, orr = hg.enu_to_axial(origin[0], origin[1], R)
        gq, grr = hg.enu_to_axial(dest[0], dest[1], R)
        gx, gy = R * hg.SQRT3 * (gq + grr / 2.0), R * 1.5 * grr
        straight = float(np.linalg.norm(dest[:2] - origin[:2]))
        # lane->lane reference: the SAME ruler main's A* gates and reports against (issue #50),
        # so SIPP's air_detour_m/cost cannot drift from A*'s.
        straight_ref = enroute_reference_m(origin, dest, req.origin_terminal, req.dest_terminal, cfg)

        # ONE bind transaction for both occupancy images, as `AStarPlanner._plan_compiled` does: a
        # failure in either constructor tears down both rather than leaving a half-subscribed
        # ledger.
        batch = _BindBatch()
        try:
            svc = self._occupancy(req, ledger, cfg, _batch=batch)
            cocc = self._compiled_occ(req, ledger, cfg, _batch=batch)
            batch.run(ledger)
        except BaseException:
            self._unbind_compiled_occupancy(ledger)
            self._unbind_reference_occupancy(ledger)
            raise
        dwell_steps = tuple(max(1, int(math.ceil((cfg.hover_time_s + cfg.climb_time_to(z)) / dt)))
                            for z in levels)              # per-level dwell (hover + actual climb to that level)
        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        own = frozenset(t.id for t in (o_term, d_term) if t is not None)
        self._own = own            # last plan's own terminal-id set (diagnostics + occupancy tests)
        o_cap = o_term.capacity if o_term is not None else 1
        d_cap = d_term.capacity if d_term is not None else 1
        fixed = cfg.fixed_exit_lanes
        o_lanes = hg.terminal_lanes(origin, o_term, cfg) if fixed and o_term is not None else []
        d_lanes = hg.terminal_lanes(dest, d_term, cfg) if fixed and d_term is not None else []
        h_off = max((L.dist for L in d_lanes), default=0.0)
        o_r = terminal_radius(o_term, cfg) if o_term is not None else 0.0
        tcap = self._tcap
        c_gd, c_hold, c_lat = (cfg.cost_ground_delay_per_s, cfg.cost_air_hold_per_s,
                               cfg.cost_air_lateral_per_m)
        d_r = exit_radius(d_term, cfg) if d_lanes else 0.0
        n_hops = int(math.ceil(max(straight, pitch) / pitch))
        climb_span = (int(math.ceil((levels[-1] - levels[0]) / (cfg.climb_rate_mps * dt)))
                      if nlev > 1 else 0)
        max_step = search_horizon(base, max(takeoff_steps) + max((ln.steps for ln in o_lanes), default=0),
                                  n_hops, climb_span, cfg)

        if cocc.cell_id(oq, orr, 0) < 0 or cocc.cell_id(gq, grr, 0) < 0 or max_step > cocc.MAXS:
            # A demand endpoint outside the cfg-only box, or a late/oversized-terminal flight beyond
            # its cfg-only time bound: take the unbounded reference rather than let a finite window
            # make every cell look blocked. Mirrors `AStarPlanner._plan_compiled`'s box guard.
            self._ref_dispatch["box-guard"] += 1
            return self._splan_reference(req, ledger, cfg)

        # ---- HOISTED out of the widen loop: neither depends on the window or on `gen`. `to_ok`
        # spans
        # [base, base + ground_delay_steps] and the landing runs span [base, max_step]; rebuilding
        # the
        # latter per iteration is ~12k Python `tcap.dwell_ok` calls for a value that cannot
        # change. ----
        smax = base + ground_delay_steps(cfg)
        n_to = smax - base + 1                                 # ground-delay steps; to_ok is per (step, level)
        to_ok = []                                             # flat mask, indexed [si*nlev + L]
        if fixed and o_term is not None:
            for s_ in range(base, smax + 1):                   # per-(step, level) dwell, lane-independent
                lv = tcap.dwell_ok_levels(o_term, origin, s_ * dt, o_cap, levels)
                to_ok.extend(bool(lv[L]) for L in range(nlev))
        else:                                                  # non-terminal: origin cell, per-level pad
            for s_ in range(base, smax + 1):
                to_ok.extend(svc.pad_clear(oq, orr, s_, dwell_steps[L]) for L in range(nlev))
        if not any(to_ok):
            return _deny(req, DenialReason.BUDGET_EXCEEDED)

        if fixed and d_term is not None:                       # column-capacity landing
            def _land(s_, L):
                return tcap.dwell_ok(d_term, dest, s_ * dt, d_cap, z=levels[L])
        else:                                                  # single dest hex; pad-clear landing
            def _land(s_, L):
                return svc.pad_clear(gq, grr, s_, dwell_steps[L])
        lf_lo, lf_hi, lf_off = [], [], [0]                     # per-level landing runs, concatenated
        for L in range(nlev):
            lo = -1
            for s_ in range(base, max_step + 1):
                if _land(s_, L):
                    if lo < 0:
                        lo = s_
                elif lo >= 0:
                    lf_lo.append(lo); lf_hi.append(s_ - 1); lo = -1
            if lo >= 0:
                lf_lo.append(lo); lf_hi.append(max_step)
            lf_off.append(len(lf_lo))
        if not lf_lo:
            return _deny(req, DenialReason.BUDGET_EXCEEDED)

        # Per-plan read-set reset, OUTSIDE the loop: a widen re-run ACCUMULATES onto the same read
        # set
        # (A* does the same). min > max means "never probed", which `_mk_envelope` reads as
        # `cell_bbox=None`. Slots 6-7 are the STEP window, filled ONCE here rather than per probe: a
        # chain walk reads a cell across the whole window, not at a point (see `_note_cell`).
        # `read_bbox` lives in A*'s `_ks` (shared, because `_mk_envelope` is inherited), so bind
        # that
        # dict before the loop — the SIPP arrays inside it are sized per widen, this one is not.
        rb = super()._kernel_state(cocc, self._log2_cap_max)[0]["read_bbox"]
        rb[0] = rb[2] = rb[4] = _BBOX_HUGE
        rb[1] = rb[3] = rb[5] = -_BBOX_HUGE
        rb[6] = base
        rb[7] = max_step

        lane_cells = ([ln.cell for ln in o_lanes] if fixed and o_term is not None else [(oq, orr)])
        lane_lat = ([c_lat * (ln.dist - o_r) for ln in o_lanes]
                    if fixed and o_term is not None else [0.0])
        lane_st = ([ln.steps for ln in o_lanes] if fixed and o_term is not None else [0])
        goal_cells = ([ln.cell for ln in d_lanes] if fixed and d_term is not None else [(gq, grr)])
        goal_lat = ([c_lat * (ln.dist - d_r) for ln in d_lanes]
                    if fixed and d_term is not None else [0.0])
        anchor_q = [oq, gq] + [c[0] for c in lane_cells] + [c[0] for c in goal_cells]
        anchor_r = [orr, grr] + [c[1] for c in lane_cells] + [c[1] for c in goal_cells]

        widen = 0
        while True:
            # `gen` INSIDE the loop, and this line is load-bearing: it stamps `ov_own_gen`, the
            # frontier, the goal flags AND the kernel's g-hash, so hoisting it makes a widen re-run
            # reuse the previous pass's closed set and return a spurious NO_PATH. Same discipline as
            # `AStarPlanner._plan_compiled`'s FB_MASK re-run.
            gen = self._bump_gen()
            if own and self._build_overlay(cocc, o_term, d_term, origin, dest, gen):
                # A cell under BOTH our column and a foreign hub's. One boolean per cell cannot say
                # that, so take the exact reference — A*'s issue-#3 exit, reused verbatim. Measured
                # unreachable on demand-generated layouts (see `sipp_window`'s header).
                self._sfb += 1
                self._sfb_overlap += 1
                return self._splan_reference(req, ledger, cfg)

            n_wcells = SW.window_bounds(
                cocc, self._k_wbox, q_cells=anchor_q, r_cells=anchor_r, base=base,
                max_step=max_step, lateral_margin=_SWINDOW_MARGIN_HEX * (1 << widen))
            if n_wcells <= 0:                       # box clipped to nothing at this widen level
                return self._splan_reference(req, ledger, cfg)

            # WINDOW-LOCAL indices, recomputed per widen because the window moved. The kernel
            # reconstructs world (q, r) as `iq + qmin`, so every index handed to it must be in the
            # same frame as the bounds it is given.
            wq0, wr0 = int(self._k_wbox[SW.W_Q0]), int(self._k_wbox[SW.W_R0])
            wrspan = int(self._k_wbox[SW.W_RSPAN])
            wqspan = int(self._k_wbox[SW.W_Q1]) - wq0 + 1

            def _wqr(qc, rc, wq0=wq0, wr0=wr0, wrspan=wrspan, wqspan=wqspan):
                iq, ir = qc - wq0, rc - wr0
                return iq * wrspan + ir if 0 <= iq < wqspan and 0 <= ir < wrspan else -1

            lane_qr = [(w, lat, st) for (qc, rc), lat, st in zip(lane_cells, lane_lat, lane_st)
                       if (w := _wqr(qc, rc)) >= 0]
            goal_pairs = [(w, lat) for (qc, rc), lat in zip(goal_cells, goal_lat)
                          if (w := _wqr(qc, rc)) >= 0]
            if not lane_qr or not goal_pairs:
                if widen < _SWINDOW_WIDEN_MAX:      # an endpoint fell outside: widen before giving up
                    self._swin_widen += 1
                    widen += 1
                    continue
                return self._splan_reference(req, ledger, cfg)

            ks, _kc = self._skernel_state(cocc, n_wcells)
            for _ in range(_SWINDOW_GROW_MAX):
                tail = SW.build_window_intervals(
                    cocc._arena.arena, cocc._arena.start, cocc._arena.length, cocc.static_col,
                    ks["ov_own_gen"], gen, cocc.qmin, cocc.rmin, cocc.rspan, cocc.n_levels,
                    self._k_wbox, self._k_iv_lo, self._k_iv_hi, self._k_iv_nxt, self._k_scratch,
                    _S0_SHIFT, _SPAN_BITS, _FIELD_MASK)
                if tail >= 0:
                    break
                self._swin_grown += 1
                self._skernel_state(cocc, -tail)   # grow to exactly what the builder asked for
            else:
                return self._splan_reference(req, ledger, cfg)   # would not fit even after growing

            goal_cost_lb = min((cost for _, cost in goal_pairs), default=0.0)
            for w, goal_cost in goal_pairs:        # land from ANY level ⇒ mark at all levels
                for L in range(nlev):
                    cell = w * nlev + L
                    self._k_goal_gen[cell] = gen
                    self._k_goal_cost[cell] = goal_cost

            n, _cost, _n_exp, flag = self._skernel(
                self._k_iv_lo, self._k_iv_hi, self._k_iv_nxt,
                wq0, wr0, wrspan, wqspan, base, max_step, nlev,
                np.asarray([w for w, _, _ in lane_qr], np.int64),
                np.asarray([lat for _, lat, _ in lane_qr], np.float64),
                np.asarray([st for _, _, st in lane_qr], np.int64), len(lane_qr),
                np.asarray(to_ok, np.bool_), n_to, c_gd,
                np.asarray(takeoff_steps, np.int64), np.asarray(takeoff_cost, np.float64),
                np.asarray(rung_steps, np.int64), np.asarray(rung_cost, np.float64),
                self._k_goal_gen, self._k_goal_cost, np.asarray(lf_lo, np.int64),
                np.asarray(lf_hi, np.int64), np.asarray(lf_off, np.int64),
                c_hold, c_lat, pitch, dt, gx, gy, R, h_off, goal_cost_lb,
                gen, self._k_front_head, self._k_front_tail, self._k_front_gen,
                self._k_lab_cell, self._k_lab_slot, self._k_lab_arr, self._k_lab_g, self._k_lab_par,
                self._k_lab_next, self._k_lab_prev, self._k_lab_dead, self._k_max,
                self._k_heap_f, self._k_heap_c, self._k_heap_n, self._k_max,
                self._k_gpack, self._k_gpackf, self._k_hash_cap, self._k_hash_log2, max_step + 2,
                self._k_out_q, self._k_out_r, self._k_out_s, self._k_out_L,
                rb,
            )
            # FB_OOB is now a WINDOW miss, not a global-box stray: `window_bounds` already clipped
            # to
            # the global box, so a cell outside the window is recoverable by widening. The kernel
            # returns it on the FIRST touch, before reading any chain, so no partial result exists
            # to
            # discard. At the ceiling this falls through to the unbounded reference like any other.
            if flag == FB_OOB and widen < _SWINDOW_WIDEN_MAX:
                self._swin_widen += 1
                widen += 1
                continue
            break

        self._n_expansions = int(_n_exp)
        self.last_expansions = self._n_expansions          # inherited public telemetry contract
        if self.record_envelope and flag not in (FB_OOB, FB_CAP, FB_HASH):
            # One build covering both remaining exits (NO_PATH deny + accept). A fallback is skipped
            # deliberately: `_fallback` runs `AStarPlanner.plan`, which builds the AUTHORITATIVE
            # envelope for the search that actually produced the answer — ours would describe an
            # abandoned partial search. `unbounded=False` because every native exit here is a
            # COMPLETE search: the kernel truncates only via FB_CAP/FB_HASH, which are fallbacks, so
            # NO_PATH means the heap drained rather than the budget ran out.
            self._mk_envelope(req, cfg, o_term, d_term, origin, dest, max_step, rb, unbounded=False)
        if flag == FB_OOB or flag == FB_CAP or flag == FB_HASH:
            self._sfb += 1
            if flag == FB_CAP:
                self._sfb_cap += 1                        # search too big (hard/near-infeasible flight)
            elif flag == FB_HASH:
                self._sfb_hash += 1                       # best-g table saturated (raise _k_hash_log2)
            else:
                self._sfb_oob += 1                        # reroute strayed outside the kernel box
            return self._fallback(req, ledger, cfg)
        if flag == NO_PATH:
            return _deny(req, DenialReason.BUDGET_EXCEEDED)

        # ---- reconstruct: out_* is goal→start; reverse + re-expand folded hover (a rung's climb is a gap) ----
        labels = [(int(self._k_out_q[i]), int(self._k_out_r[i]), int(self._k_out_L[i]), int(self._k_out_s[i]))
                  for i in range(n - 1, -1, -1)]
        air = []
        for i, (q, r, L, a) in enumerate(labels):
            air.append((q, r, L, a))
            if i + 1 < len(labels):
                nq, nr, nL, na = labels[i + 1]
                # fill folded pre-move hover at (q,r,L); a vertical rung's climb steps stay a GAP (the
                # transition segment), exactly as the reference/A* leave them, so cruise_wps matches
                stop = (na - rung_steps[min(L, nL)] + 1) if (nq == q and nr == r and nL != L) else na
                for k in range(a + 1, stop):
                    air.append((q, r, L, k))
        self._air = air            # last compiled per-step search path [(q,r,L,step)] (diagnostics + tests)
        lane_steps = {ln.cell: ln.steps for ln in o_lanes}
        ground_steps = (air[0][3] - takeoff_steps[air[0][2]] - base
                        - lane_steps.get((air[0][0], air[0][1]), 0))   # issue #52
        delay = ground_steps * dt
        cruise_wps: list[TimedPoint] = [
            (np.array([*hg.hex_center(q, r, R), levels[L]]), a * dt) for (q, r, L, a) in air]
        volumes, centerline, _cum_horiz, n_hover = self._build(
            cruise_wps, origin, dest, base, ground_steps, cfg,
            origin_term=req.origin_terminal, dest_term=req.dest_terminal,
        )
        flown = enroute_flown_m([p for p, _ in centerline], origin, dest,
                                req.origin_terminal, req.dest_terminal, cfg)
        if straight_ref > _EPS and flown / straight_ref > cfg.max_detour_factor:
            return self._file_deny(req, DenialReason.BUDGET_EXCEEDED, volumes, ledger)
        if ledger.any_conflict(volumes):
            return self._file_deny(req, DenialReason.CONFLICT_FILED, volumes, ledger)
        detour = enroute_detour_m(flown, straight_ref)
        z_takeoff, z_land = levels[air[0][2]], levels[air[-1][2]]     # true vertical travel (mirror A*)
        cruise_dz = sum(abs(levels[air[i + 1][2]] - levels[air[i][2]]) for i in range(len(air) - 1))
        intent = OperationalIntent(
            request=req, status=IntentStatus.ACCEPTED, volumes=volumes, centerline=centerline,
            ground_delay_s=delay, air_hold_s=n_hover * dt, air_detour_m=detour,
            lattice_overhead_m=hg.lattice_overhead_m([(t[0], t[1]) for t in air], pitch, detour),
            altitude_change_m=endpoint_altitude_change_m(z_takeoff, z_land, cruise_dz, cfg), planner="sipp",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent

    def _succ(self, st, iv, SI, cfg, pitch, levels, takeoff_steps, takeoff_cost, rung_steps, rung_cost,
              dwell_steps, own, o_cap, o_term, origin, tcap, dest, o_lanes, o_r, fixed_lanes,
              ground_max_step, max_step, is_goal_cell):
        """Successors as ``(AS, edge_cost, wait_steps, interval_index)`` — the multi-altitude safe-interval
        collapse. ``iv`` is the popped state's interval index (carried in the heap → no ``index_of`` scan in
        the hot loop); each air successor carries its OWN interval index (``-1`` for ground). Ground →
        ground-wait ray + a per-level takeoff at the current step (per-step pad/dwell gates match A*). Air →
        same-level reroute (one successor per reachable neighbour interval, folding pre-move hover) + vertical
        rungs to L±1 (folding pre-rung hover; both levels clear across the climb window — the interval-collapse
        image of the njit kernel's rung block) + a goal-cell hover to retry the landing gate. Mirrors
        :meth:`AStarPlanner._edges`."""
        dt = cfg.dt_s
        c_gd, c_hold, c_lat = (cfg.cost_ground_delay_per_s, cfg.cost_air_hold_per_s,
                               cfg.cost_air_lateral_per_m)
        svc = self._svc
        out = []
        if st[0] == "g":
            _, q, r, s = st
            if s + 1 <= ground_max_step:
                out.append((("g", q, r, s + 1), c_gd * dt, 0, -1))      # ground-wait ray (== A* g→g)
            if fixed_lanes and o_term is not None:                       # one takeoff edge per (lane, level)
                level_ok = tcap.dwell_ok_levels(o_term, origin, s * dt, o_cap, levels)
                for lane in o_lanes:
                    lq, lr = lane.cell
                    lane_st = lane.steps                     # issue #52: climb, THEN translate out
                    for L in range(len(levels)):
                        ts = s + takeoff_steps[L] + lane_st
                        if level_ok[L] and ts <= max_step and not svc.is_blocked(lq, lr, L, ts, own):
                            out.append((("a", lq, lr, L, ts),
                                        takeoff_cost[L] + c_lat * (lane.dist - o_r), 0,
                                        SI.index_of(lq, lr, L, ts)))
                return out
            hub_ok = (tcap.dwell_ok_levels(o_term, origin, s * dt, o_cap, levels, toward=dest)
                      if o_term is not None else None)
            for L in range(len(levels)):                                 # legacy / non-terminal: per level
                ts = s + takeoff_steps[L]
                pad_ok = hub_ok[L] if o_term is not None else svc.pad_clear(q, r, s, dwell_steps[L])
                if ts <= max_step and not svc.is_blocked(q, r, L, ts, own) and pad_ok:
                    out.append((("a", q, r, L, ts), takeoff_cost[L], 0, SI.index_of(q, r, L, ts)))
            return out

        _, q, r, L, s = st
        hi_c = SI.intervals(q, r, L)[iv][1] if iv >= 0 else s            # last step this cell stays free
        for dq, dr in hg.AXIAL_NEIGHBORS:                                # reroute (same level, collapsed)
            nq, nr = q + dq, r + dr
            for j, (lo, hi) in enumerate(SI.intervals(nq, nr, L)):
                arr = max(s + 1, lo)
                if arr > hi or arr > max_step:
                    continue
                if arr - 1 > hi_c:                                       # can't wait here that long
                    break                                               # later intervals need even more
                wait = arr - (s + 1)                                     # folded pre-move hover
                out.append((("a", nq, nr, L, arr), c_hold * dt * wait + c_lat * pitch, wait, j))
        for dL in ((-1, 1) if self.vertical_edges else ()):             # vertical rungs to an adjacent level
            L2 = L + dL
            if not (0 <= L2 < len(levels)):
                continue
            rung = L if dL == 1 else L2                                  # rung index = min(L, L2)
            rsteps = rung_steps[rung]
            if s + rsteps > hi_c:                                        # current level not free through climb
                continue
            for j, (lo, hi) in enumerate(SI.intervals(q, r, L2)):
                ap = max(s, lo - 1)                                      # rung-start step (fold pre-rung hover)
                if ap > hi_c - rsteps:                                   # current level can't hold climb window
                    break                                               # later target intervals need even more
                a = ap + rsteps                                          # arrival on the target level
                if a > hi or a > max_step:
                    continue                                            # target interval too short for transit
                wait = ap - s
                out.append((("a", q, r, L2, a), c_hold * dt * wait + rung_cost[rung], wait, j))
        if is_goal_cell(q, r) and s + 1 <= hi_c and s + 1 <= max_step:
            out.append((("a", q, r, L, s + 1), c_hold * dt, 0, iv))     # hover to retry the landing gate
        return out
