"""Dense per-(cell, level) free-interval pools for the compiled A* kernel (issue #8 Track B, A* port).

The numba kernel (:mod:`astar_kernel`) needs O(1) array reads to answer ``is (q, r, L) blocked at step
s?`` — A*'s per-node obstacle test — reproducing :meth:`HexOccupancyService.is_blocked` exactly. The
"cell" is a **(q, r, L)** triple (a hex at a flight level). Two flat interval pools, both maintained
incrementally from the ledger commit hook via :func:`hexgrid.rasterize_volume_dual`:

  * **corridor pool** (``corr``) — ordinary corridor cells (``in_blk`` from non-column volumes, minus the
    committing flight's own-column interior). Equals ``HexOccupancyService.blocked`` cell-for-cell.
  * **column pool** (``col``) — every terminal column's inner footprint. Equals ``term_cells`` (which
    hubs, dropped — only presence matters; own/foreign is resolved per-flight).
  * **static column** (``static_col``) — a per-cell bool for always-active terminals
    (``cfg.terminal_airspace_always_active``, #24): a permanent, step- AND level-independent foreign wall.
    Equals ``HexOccupancyService.static_term_cells``. NOT ledger-derived (survives ``reset()``); empty and
    zero-overhead when the flag is off.

``is_blocked(q,r,L,s,own)`` then folds to (kernel ``_blocked``):

    colb = column-blocked(cell,s) OR static_col(cell);  corb = corridor-blocked(cell,s)
    if colb and cell not in the flight's OWN-column footprint:  return True   # foreign column → wall
    return corb                                                  # corridor / own-col fixed-lane sibling

The flight's **own-column footprint** is a cheap per-cell mark (``ov_own_gen[cell] == gen``) the host sets
per plan by rasterizing the flight's 1–2 own hub columns — O(footprint), no per-step scan. This is exact
when own and foreign columns don't share a cell (hub spacing ≫ column radius); the node-count parity test
guards the assumption.

Cells live in a box from the region corners + a reroute ``margin``. A committed corridor cell outside the
box is skipped (counted in ``oob_corridor_cells``) — safe, because any *query* to that cell gets
``cell_id < 0`` and the kernel falls back via ``FB_OOB``; it never crashes on commit. ``MAXS`` covers the
worst-case per-flight ``max_step`` (a region-diagonal, latest-departing flight — see ``_box``), so every
reachable query step lies inside the seed interval. Committed steps *beyond* ``MAXS`` (a landing column's
hover tail) are dropped by ``_Pool.block``, which is harmless: every kernel query is ``≤ max_step ≤ MAXS``
(guarded in ``_plan_compiled``), so those far-future steps are never read.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from ..geometry import CylinderSpec
from ..types import as_terminal
from . import hexgrid as hg
from ._packed import P_HI, P_LO, P_NXT, aligned_2d


def search_horizon(base: int, takeoff_steps_max: int, n_hops: int, climb_span: int, cfg) -> int:
    """The largest ``step`` an A* plan can reach: takeoff + a 3× lateral detour budget + a full ground-
    delay allowance + the mid-route climb span. ONE definition (issue #5) — ``_plan_reference``,
    ``_plan_compiled``, and ``CompiledHexOccupancy._box`` (with worst-case args) all call it, so the
    kernel's search bound, the box guard, and ``MAXS`` cannot drift apart. Monotone in ``base``/``n_hops``,
    so ``_box``'s worst-case value bounds every per-flight one."""
    return (base + takeoff_steps_max + int(math.ceil(cfg.max_ground_delay_s / cfg.dt_s))
            + 3 * n_hops + 2 * climb_span + 6)


def hover_tail_steps(cfg) -> int:
    """Extra steps a committed landing column occupies PAST the arrival step — hover dwell + climb to the
    top level + the ASTM time buffer, in dt units (mirrors ``volumes.hover_reservation`` /
    ``hexgrid._step_range``). ``MAXS`` adds this so ``_Pool.block`` never silently drops a committed step;
    query correctness never needs it (every query is ``≤ max_step ≤ MAXS``), but it removes the old
    hand-tuned ``+16`` slack that only happened to cover the tail on default numbers (issue #1)."""
    max_climb = max(cfg.climb_time_to(z) for z in cfg.flight_levels_m)
    return int(math.ceil((cfg.hover_time_s + max_climb + cfg.time_buffer_s) / cfg.dt_s)) + 2


def schedulable_horizon_steps(cfg) -> int:
    """``MAXS`` — the occupancy box's step depth: the worst-case per-flight ``search_horizon`` (latest
    departure ``≤ horizon_s`` + a region-DIAGONAL flight; monotone, so it bounds every IN-BOX flight's
    ``max_step``) plus the landing hover tail. A flight whose ``base`` exceeds this (e.g. a late-departing
    return, ``t_departure > horizon_s``) box-guards to the pure-Python reference, so the kernel never queries a
    step past ``MAXS``. Hence this need NOT cover late departures — unlike the permanent terminal WALL, which
    is time-invariant and uses its own sentinel ``t_end`` (see ``volumes.permanent_terminal_reservation``).

    Nor need it cover the origin lane traverse the per-flight horizon now adds (issue #52): lanes come
    from DEMAND terminals whose radii can exceed ``cfg.terminal_radius_m``, so no cfg-only bound exists —
    a flight whose lane steps push ``max_step`` past ``MAXS`` box-guards to the reference the same way."""
    w, h = cfg.region_size_m
    dt = cfg.dt_s
    pitch = cfg.nominal_speed_mps * dt
    levels = cfg.flight_levels_m
    base_max = int(math.ceil(cfg.horizon_s / dt))
    takeoff_max = max(cfg.climb_steps_to(z) for z in levels)
    n_hops_max = int(math.ceil(math.hypot(w, h) / max(pitch, 1e-9)))
    climb_span = (int(math.ceil((levels[-1] - levels[0]) / (cfg.climb_rate_mps * dt)))
                  if cfg.n_levels > 1 else 0)
    return search_horizon(base_max, takeoff_max, n_hops_max, climb_span, cfg) + hover_tail_steps(cfg)


class _Pool:
    """Flat linked-list free-interval pool: cell ``c``'s intervals are walked from slot ``c`` along
    ``nxt``; a blocked step splits the containing interval in place. Slot 0..NC-1 pre-seeded ``[0, MAXS]``.

    Rows are packed ``(lo, hi, nxt, pad)`` int32 in one ``iv`` block — 16 B, 8 rows per cache line — so
    a list walk touches ONE line per node instead of one in each of three separate multi-MB arrays.
    This is the hottest layout in the whole search: the kernel's ``_blocked`` walks two of these lists
    for every neighbour of every expansion. See ``_packed`` for the measurement behind it."""

    def __init__(self, NC: int, MAXS: int):
        self.NC = NC
        self.MAXS = MAXS
        self.cap = max(2 * NC, 1 << 18)
        self.iv = aligned_2d(self.cap, 4, np.int32)
        self.reset()

    def reset(self):
        self.iv[: self.NC, P_LO] = 0
        self.iv[: self.NC, P_HI] = self.MAXS
        self.iv[: self.NC, P_NXT] = -1
        self.nslots = self.NC

    def _grow(self):
        cap = self.cap * 2
        iv = aligned_2d(cap, 4, np.int32)
        iv[: self.cap] = self.iv
        self.iv = iv
        self.cap = cap

    def _alloc(self, lo, hi, nxt) -> int:
        if self.nslots >= self.cap:
            self._grow()
        s = self.nslots
        self.iv[s, P_LO] = lo; self.iv[s, P_HI] = hi; self.iv[s, P_NXT] = nxt
        self.nslots += 1
        return s

    def block(self, c: int, s: int) -> None:
        """Split cell ``c``'s free interval containing ``s`` (in place). Equivalent to
        ``block_range(c, s, s)``; kept for callers/tests that block a single step."""
        self.block_range(c, s, s)

    def block_range(self, c: int, s0: int, s1: int) -> None:
        """Remove the whole contiguous span ``[s0, s1]`` from cell ``c``'s free intervals in one pass.

        A committed volume occupies each cell over a contiguous step range, so this replaces ``S``
        single-step splits with one walk (issue #8 Phase E). The free-STEP set is identical to
        blocking ``s0, s0+1, …, s1`` individually, so the compiled kernel is byte-unaffected.

        The interval list (head = slot ``c``) is kept sorted ascending by ``block``/this. The span may
        straddle several free intervals when earlier commits already punched holes: the first keeps a
        left remainder ``[a, s0-1]``, the last a right remainder ``[s1+1, b]``, and any interval fully
        inside ``[s0, s1]`` is marked empty (``lo>hi``, never matches a query) rather than unlinked —
        the flat pool has no cheap way to drop its fixed head/middle slots, and a dead slot is
        harmless (the kernel's interval walk simply skips it). Every access goes through ``self.iv``:
        ``_alloc`` can ``_grow``, which REPLACES the array, so a hoisted reference would write into the
        dead buffer."""
        if s0 < 0:
            s0 = 0
        if s1 > self.MAXS:
            s1 = self.MAXS
        if s0 > s1:
            return
        slot = c
        while slot != -1:
            a, b = int(self.iv[slot, P_LO]), int(self.iv[slot, P_HI])
            nxt = int(self.iv[slot, P_NXT])
            if b < s0:                                  # wholly left of the span → keep walking
                slot = nxt
                continue
            if a > s1:                                  # wholly right (list sorted) → done
                return
            keep_left = a <= s0 - 1
            keep_right = s1 + 1 <= b
            if keep_left and keep_right:                # span sits inside one interval → split once
                self.iv[slot, P_HI] = s0 - 1
                ns = self._alloc(s1 + 1, b, nxt)
                self.iv[slot, P_NXT] = ns
                return
            if keep_right:                              # right remainder is > s1 → nothing past it
                self.iv[slot, P_LO] = s1 + 1
                return
            if keep_left:                               # left remainder kept; span may reach further
                self.iv[slot, P_HI] = s0 - 1
            else:                                       # interval fully covered → mark empty (lo>hi)
                self.iv[slot, P_LO] = 1
                self.iv[slot, P_HI] = 0
            slot = nxt

    def blocked_at(self, c: int, s: int) -> bool:
        """True iff step ``s`` is in NO free interval of cell ``c``."""
        iv = self.iv                                   # no allocation here, so hoisting is safe
        slot = c
        while slot != -1:
            if int(iv[slot, P_LO]) <= s <= int(iv[slot, P_HI]):
                return False
            slot = int(iv[slot, P_NXT])
        return True


class CompiledHexOccupancy:
    """Two incremental flat pools (corridor + column) feeding the numba A* kernel. Commit-hook driven."""

    def __init__(self, cfg, margin: int = 64):
        self.cfg = cfg
        self.R = hg.circumradius(cfg)
        self.infl_blocked = cfg.corridor_width_m / 2.0 + self.R
        self.infl_pad = cfg.effective_hover_radius_m + self.R
        self.n_levels = cfg.n_levels
        self.n_added = 0
        self.evicted_before: int | None = None

        qmin, rmin, qspan, rspan, maxs = self._box(cfg, margin)
        self.qmin, self.rmin, self.qspan, self.rspan = qmin, rmin, qspan, rspan
        self.NC = qspan * rspan * self.n_levels
        self.MAXS = maxs
        self.corr = _Pool(self.NC, self.MAXS)
        self.col = _Pool(self.NC, self.MAXS)
        # cell → {terminal ids whose column EVER covers it, across all steps}. Lets the host detect an
        # own∩foreign shared cell (issue #3) and fall back to the reference, instead of the overlay boolean
        # silently treating a foreign column as transparent. Deliberately TIME-COLLAPSED and NOT pruned by
        # evict_before: it's a conservative SUPERSET of live columns, so the overlap check may fall back for
        # a temporally-past foreign column — safe (the reference is exact), and bounded by the hub layout
        # (distinct column cells × owning hubs, not per-flight), so it does not grow unboundedly.
        self.col_owners: dict[int, set] = {}
        # Always-active terminals (cfg.terminal_airspace_always_active, #24): permanent FOREIGN column walls,
        # step- AND level-independent (the [ground, ceiling] tube). A per-cell bool over the SAME (q,r,L) index
        # as the pools — a static cell reads as column-blocked at EVERY step (the kernel folds it into `colb`).
        # NOT ledger-derived, so reset() re-applies it from `_static_terms` (the hub set doesn't change); the
        # array itself is never cleared. Empty unless `_on_static` fires (ledger subscribe_static hook) ⇒ off = free.
        self.static_col = np.zeros(self.NC, np.bool_)
        self._static_terms: list = []                   # (center, term) per walled hub, for reset() re-apply
        # committed corridor cells that fell outside the box: skipped (never a crash); any query to such a
        # cell gets cell_id < 0 and the kernel falls back via FB_OOB. Non-zero ⇒ consider widening `margin`.
        self.oob_corridor_cells = 0
        self._warned_oob = False                        # warn ONCE per instance (persists across reset())

    def _box(self, cfg, margin):
        w, h = cfg.region_size_m
        R = self.R
        qs, rs = [], []
        for x, y in ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h)):
            q, r = hg.enu_to_axial(x, y, R)
            qs.append(q); rs.append(r)
        qmin, qmax = min(qs) - margin, max(qs) + margin
        rmin, rmax = min(rs) - margin, max(rs) + margin
        maxs = schedulable_horizon_steps(cfg)   # worst-case search_horizon + hover tail (see the shared fn)
        return qmin, rmin, qmax - qmin + 1, rmax - rmin + 1, maxs

    def cell_id(self, q: int, r: int, L: int) -> int:
        iq, ir = q - self.qmin, r - self.rmin
        if iq < 0 or iq >= self.qspan or ir < 0 or ir >= self.rspan or L < 0 or L >= self.n_levels:
            return -1
        return (iq * self.rspan + ir) * self.n_levels + L

    # ---------- commit hook ----------
    def on_commit(self, _flight_id, volumes) -> None:
        own_cols = tuple((v.shape.cx, v.shape.cy, v.shape.radius) for v in volumes
                         if v.terminal_id is not None and isinstance(v.shape, CylinderSpec))
        for v in volumes:
            self._add(v, own_cols)
        self.n_added += len(volumes)

    def _inside_a_column(self, q, r, cols) -> bool:
        c = hg.hex_center(q, r, self.R)
        return any((c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= rad * rad for cx, cy, rad in cols)

    def _add(self, vol, own_cols) -> None:
        tid = vol.terminal_id
        is_column = tid is not None and isinstance(vol.shape, CylinderSpec)
        for q, r, L, s_lo, s_hi, in_blk in hg.rasterize_ranges(
            vol, self.cfg, self.R, self.infl_blocked, self.infl_pad
        ):
            if not in_blk:
                continue
            if self.evicted_before is not None and s_lo < self.evicted_before:
                s_lo = self.evicted_before             # clip the span, never resurrect an evicted step
            if s_lo > s_hi:
                continue
            c = self.cell_id(q, r, L)
            if is_column:                               # → column pool (all columns; own/foreign per plan)
                if c >= 0:
                    self.col.block_range(c, int(s_lo), int(s_hi))
                    self.col_owners.setdefault(c, set()).add(tid)
            else:                                       # → corridor pool (minus committing own interior)
                if own_cols and self._inside_a_column(q, r, own_cols):
                    continue
                if c < 0:                               # outside the box → skip (never crash on commit);
                    if not self._warned_oob:            # a query to this cell gets cell_id<0 → kernel FB_OOB.
                        warnings.warn(                  # warn once/instance (a nonzero count ⇒ margin small)
                            "CompiledHexOccupancy: a committed corridor cell fell outside the kernel box — "
                            "skipped (its flights fall back via FB_OOB). Consider widening `margin`.",
                            RuntimeWarning, stacklevel=2)
                        self._warned_oob = True
                    self.oob_corridor_cells += 1
                    continue
                self.corr.block_range(c, int(s_lo), int(s_hi))

    def _on_static(self, center, term) -> None:
        """Derive the compiled routing wall from a ledger static-terminal registration — the
        ``ReservationLedger.subscribe_static`` hook target (bound in ``AStarPlanner._compiled_occ``, and named
        ``_on_static`` to match ``HexOccupancyService._on_static`` / the ``on_commit`` observer convention). Marks
        ``static_col`` at every flight level for each terminal hex (``hg.terminal_cells`` — the SAME cell set
        as ``HexOccupancyService.static_term_cells``, so the compiled wall is byte-identical to the
        reference) and records the owning ``tid`` in ``col_owners`` so the own∩foreign overlap check (issue
        #3) still fires. Appends to ``_static_terms`` so ``reset()`` re-applies it (col_owners is cleared on
        reset — unlike the reference's `static_term_cells` which reset() never touches). The hub's own
        flights pass through (the host overlay marks these cells own — see ``_build_overlay``). Idempotent
        per hub. The authoritative wall is the ledger's permanent volume; this is the derived routing view."""
        self._static_terms.append((center, term))
        self._mark_static(center, term)

    def _mark_static(self, center, term) -> None:
        """Set ``static_col`` + ``col_owners`` for one hub's terminal cells (all levels). Split from
        ``_on_static`` so ``reset()`` can re-apply without re-appending to ``_static_terms``."""
        tid = as_terminal(term).id
        for q, r in hg.terminal_cells(center, term, self.cfg):
            for L in range(self.n_levels):
                c = self.cell_id(q, r, L)
                if c >= 0:                              # OOB static cell ⇒ any query gets cell_id<0 → FB_OOB
                    self.static_col[c] = True
                    self.col_owners.setdefault(c, set()).add(tid)

    def evict_before(self, step) -> None:
        if self.evicted_before is None or step > self.evicted_before:
            self.evicted_before = step

    def reset(self) -> None:
        self.n_added = 0
        self.evicted_before = None
        self.col_owners.clear()
        self.oob_corridor_cells = 0
        self.corr.reset()
        self.col.reset()
        # Static terminals are NOT ledger-derived (a shrink rebuild must keep them) — re-mark them into the
        # freshly-cleared col_owners (static_col was never cleared, so this is idempotent). Mirrors
        # HexOccupancyService.reset() leaving static_term_cells intact.
        for center, term in self._static_terms:
            self._mark_static(center, term)

    # ---------- pure-Python oracle (kernel parity + tests) ----------
    def blocked_py(self, q: int, r: int, L: int, s: int, own_cells=None) -> bool:
        """Point query reproducing the kernel ``_blocked`` (and thus ``HexOccupancyService.is_blocked``).

        ``own_cells``: a set of ``cell_id``s that are the planning flight's OWN column footprint (empty /
        ``None`` for ``own=∅`` — the occupancy-parity contract vs ``is_blocked(..., own=())``). Out-of-box ⇒
        ``True`` (the kernel would FALLBACK)."""
        c = self.cell_id(q, r, L)
        if c < 0:
            return True
        colb = self.col.blocked_at(c, s) or bool(self.static_col[c])   # transient OR always-active column
        if colb and (own_cells is None or c not in own_cells):
            return True                                 # foreign column → wall
        return self.corr.blocked_at(c, s)               # corridor / own-column fixed-lane sibling
