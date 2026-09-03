"""Dense per-cell free-interval pool for the compiled SIPP kernel (issue #8, Track B).

The numba kernel (:mod:`sipp_kernel`) needs O(1) array reads of each hex cell's safe intervals. At
density a cell fragments into *many* disjoint free runs (many flights cross it at different times), so a
fixed-width ``[cell, K]`` table overflows — instead each interval is a **slot in a flat linked-list
pool**, maintained incrementally from the ledger commit hook:

    slot s:  iv_lo[s], iv_hi[s]  (free step-run, clipped per-flight to [base, max_step]),  iv_nxt[s]
    cell c's first interval is slot ``c`` (slots [0, NC) are pre-seeded, one per cell = [0, MAXS]);
    further intervals of c live in slots [NC, nslots) and are linked via iv_nxt from slot c.

So a cell's intervals are walked from slot ``c`` along ``iv_nxt``; the **slot index is the kernel's
frontier node id** (unique per (cell, interval), no wasted width, no overflow). A blocked range removes
its overlap in one pass (+ at most one new slot) — O(intervals in the cell) per (cell, span).
Degenerate slots (``lo > hi``, from fully covered intervals) remain in the chain and are skipped.

The global pool encodes the **non-terminal corridor** occupancy (flight-independent). Terminal cells
(shared columns) are own-dependent — recorded in ``self.cols`` for the Phase-2 per-flight patch, unused
by the Phase-1 non-terminal kernel.

Cells are encoded over a bounding box from the region corners + a reroute ``margin``. A committed corridor
cell outside the box is skipped and counted: a later kernel query to that cell is itself bounds-checked and
falls back to the pure-Python reference, so a fallback flight can never crash the ledger's commit hook.
The margin therefore only needs to cover realistic edge-skirting detours.
"""
from __future__ import annotations

import warnings
from array import array

import numpy as np

from ..geometry import CylinderSpec
from . import hexgrid as hg
# `_FIELD_MASK`/`_SPAN_BITS`/`_SPAN_LIMIT` are SHARED with the A* twin's removal journal rather than
# redefined here: two journals with independently-declared packing widths is a silent-corruption bug
# waiting for someone to widen one of them. That module is already a dependency of this one and does
# not import back, so this adds no cycle.
from .astar.compiled_hex_occupancy import (
    _FIELD_MASK,
    _SPAN_BITS,
    _SPAN_LIMIT,
    schedulable_horizon_steps,
)


class CompiledOccupancy:
    """Incremental flat-pool free-interval store feeding the numba SIPP kernel. Commit-hook driven."""

    def __init__(self, cfg, margin: int = 48, track_removal: bool = False):
        self.cfg = cfg
        self.R = hg.circumradius(cfg)
        self.infl_blocked = cfg.corridor_width_m / 2.0 + self.R
        self.infl_pad = cfg.effective_hover_radius_m + self.R
        self.n_added = 0
        self.evicted_before: int | None = None
        self._static_hubs: list = []   # always-active hubs, replayed after reset (statics survive a rebuild)
        # Removal mode (LNS destroy): a flat interval pool cannot be un-split in place — two flights
        # blocking the same step produce ONE split and the second `block_range` is a no-op — so the
        # only exact reversal is reset-the-cell and re-apply the survivors. That needs a per-cell claim
        # multiset plus a per-owner journal saying how many of each to drop. This is a port of
        # `CompiledHexOccupancy`'s, minus its pool index (it has two pools; we have one, so the key is
        # just the cell id). Flag off => zero bookkeeping.
        #     _claims[c]   = [packed, ...]           packed = s0 << _SPAN_BITS | s1
        #     _rows[fid]   = flat (c, packed) pairs  — 16 B/row against ~120 for the tuple form
        # Equal spans are a fungible MULTISET: the owner journal says how many identical spans to
        # remove, and which instance is removed cannot affect the rebuilt pool.
        self.track_removal = track_removal
        self._claims: dict[int, list[int]] = {}
        self._rows: dict[int, array] = {}
        # Cells carrying an always-active wall. This structure is the ONLY one in the repo that stores
        # such walls in the same array as commit-derived blocks (A* keeps a `static_col` bool array,
        # `SafeIntervalIndex` a `static_cols` dict) — and it has no choice: `sipp_kernel._search` takes
        # the interval pool and the overlay and NO static array, so the pool is the only channel
        # through which the kernel can learn about a wall. See `on_release` for what that costs.
        self._static_cells: set[int] = set()

        self.nlevels = cfg.n_levels                  # flight-level axis (multi-altitude): cell = (q, r, L)
        qmin, rmin, qspan, rspan, maxs = self._box(cfg, margin)
        self.qmin, self.rmin, self.qspan, self.rspan = qmin, rmin, qspan, rspan
        self.NC = qspan * rspan * self.nlevels        # one pre-seeded slot per (q, r, L) cell
        self.MAXS = maxs
        if track_removal and maxs >= _SPAN_LIMIT:    # see `_claims`: s0/s1 get _SPAN_BITS each.
            # The SINGLE bound on the packing, because `_record` clamps to MAXS before packing (the
            # A* twin records raw spans and therefore needs a second, per-row check).
            raise ValueError(
                f"CompiledOccupancy: horizon of {maxs} steps exceeds the removal journal's "
                f"{_SPAN_LIMIT}-step packing limit")
        self._init_pool()
        # Keep the same observable safety diagnostic as compiled A*: skipped cells are safe because every
        # query is bounds-checked, but a non-zero count signals that ``margin`` may be too narrow.
        self.oob_corridor_cells = 0
        self._warned_oob = False                       # warn once per instance (persists across reset)

    def _box(self, cfg, margin):
        w, h = cfg.region_size_m
        R = self.R
        qs, rs = [], []
        for x, y in ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h)):
            q, r = hg.enu_to_axial(x, y, R)
            qs.append(q); rs.append(r)
        qmin, qmax = min(qs) - margin, max(qs) + margin
        rmin, rmax = min(rs) - margin, max(rs) + margin
        # One owner for the absolute step horizon shared with compiled A*: latest in-envelope departure
        # + worst in-region route/detour + takeoff/climb + landing tail. This pool stores interval
        # ENDPOINTS rather than a dense time axis, so the correct bound does not multiply its memory.
        maxs = schedulable_horizon_steps(cfg)
        return qmin, rmin, qmax - qmin + 1, rmax - rmin + 1, maxs

    def _init_pool(self):
        # Freed overflow slots, refilled by `reset_cell`. `reset()` calls this, so `_free` needs no
        # separate clear there.
        self._free: list[int] = []
        cap = max(2 * self.NC, 1 << 18)
        self.cap = cap
        self.iv_lo = np.empty(cap, np.int32)
        self.iv_hi = np.empty(cap, np.int32)
        self.iv_nxt = np.empty(cap, np.int32)
        self.iv_lo[: self.NC] = 0                    # slot c = cell c's first interval [0, MAXS]
        self.iv_hi[: self.NC] = self.MAXS
        self.iv_nxt[: self.NC] = -1
        self.nslots = self.NC

    def _grow(self):
        cap = self.cap * 2
        for name in ("iv_lo", "iv_hi", "iv_nxt"):
            a = np.empty(cap, np.int32)
            a[: self.cap] = getattr(self, name)
            setattr(self, name, a)
        self.cap = cap

    def _alloc(self, lo, hi, nxt) -> int:
        if self._free:                  # reuse a slot freed by reset_cell before bumping
            s = self._free.pop()
        else:
            if self.nslots >= self.cap:
                self._grow()
            s = self.nslots
            self.nslots += 1
        self.iv_lo[s] = lo; self.iv_hi[s] = hi; self.iv_nxt[s] = nxt
        return s

    def reset_cell(self, c: int) -> None:
        """Re-seed cell ``c``'s list to the single free interval ``[0, MAXS]`` (removal-mode cell
        rebuild); callers re-apply the cell's surviving claims immediately after.

        The old chain's overflow slots are RECLAIMED onto the free list. Abandoning them is harmless
        per call but not per run: `_alloc` is otherwise a pure bump allocator, so under LNS — which
        resets and re-applies the same hot cells every iteration — `nslots` would grow without bound
        and drag `cap` through repeated doubling for a working set that never grows. Slot indices are
        pure storage (every reader walks the chain), so which slot holds an interval never affects an
        answer."""
        free = self._free
        slot = int(self.iv_nxt[c])
        while slot != -1:                      # walk only the overflow tail; head slot `c` is re-seeded
            free.append(slot)
            slot = int(self.iv_nxt[slot])
        self.iv_lo[c] = 0
        self.iv_hi[c] = self.MAXS
        self.iv_nxt[c] = -1

    def cell_id(self, q: int, r: int, L: int) -> int:
        iq, ir = q - self.qmin, r - self.rmin
        if iq < 0 or iq >= self.qspan or ir < 0 or ir >= self.rspan or L < 0 or L >= self.nlevels:
            return -1
        return (iq * self.rspan + ir) * self.nlevels + L

    def qr_index(self, q: int, r: int) -> int:
        """Level-less ``(iq*rspan+ir)`` index (``-1`` if out of box); the kernel's ``lane_qr`` — it
        completes it with the flight level as ``qr_index*nlevels + L`` (== :meth:`cell_id`)."""
        iq, ir = q - self.qmin, r - self.rmin
        if iq < 0 or iq >= self.qspan or ir < 0 or ir >= self.rspan:
            return -1
        return iq * self.rspan + ir

    # ---------- commit hook (mirrors SafeIntervalIndex) ----------
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
                self._rows[flight_id] = array("q", rows)
            else:
                entry.extend(rows)     # `_absorb`'s groupby can yield a second group for one fid

    def _inside_a_column(self, q, r, cols) -> bool:
        c = hg.hex_center(q, r, self.R)
        return any((c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= rad * rad for cx, cy, rad in cols)

    def _add(self, vol, own_cols, _rows=None) -> None:
        tid = vol.terminal_id
        is_column = tid is not None and isinstance(vol.shape, CylinderSpec)
        # Range-blocked, and deliberately the SHARED producer (issue #114): `rasterize_ranges` memoizes
        # the geometry sweep per (id(vol), R, infl_*), and this structure's inflation radii are
        # identical to `HexOccupancyService`'s, so the sweep the hex image just did on this same commit
        # is reused rather than repeated. The per-STEP `rasterize_volume_dual` this replaced did
        # neither: it missed the memo AND yielded ~8x the rows (span median 7), one pool walk each.
        for q, r, L, s_lo, s_hi, in_blk in hg.rasterize_ranges(
            vol, self.cfg, self.R, self.infl_blocked, self.infl_pad
        ):
            if not in_blk:
                continue
            # corridor cell inside the COMMITTING flight's own column = its unreserved interior (skip);
            # everything else — corridors AND all terminal columns — blocks the global pool. A column is
            # foreign-to-everyone here; the planning flight's own columns are exempted per-flight (overlay).
            if not is_column and own_cols and self._inside_a_column(q, r, own_cols):
                continue
            c = self.cell_id(q, r, L)
            if c < 0:
                if is_column:
                    continue                              # a column footprint cell just past the box edge
                if not self._warned_oob:
                    warnings.warn(
                        "CompiledOccupancy: a committed corridor cell fell outside the kernel box — "
                        "skipped (its flights fall back via FB_OOB). Consider widening `margin`.",
                        RuntimeWarning, stacklevel=2,
                    )
                    self._warned_oob = True
                self.oob_corridor_cells += 1
                continue
            self.block_range(c, int(s_lo), int(s_hi))
            if self.track_removal and c not in self._static_cells:
                # A walled cell is never rebuilt from claims (see `on_release`), so journaling one
                # would only cost memory. Necessary but NOT sufficient on its own — `_scompiled_occ`
                # binds as subscribe -> _absorb -> subscribe_static, so `_absorb` records claims on
                # cells that only become walled a moment later. `on_release` handles those.
                self._record(c, int(s_lo), int(s_hi), _rows)

    def _record(self, c: int, s0: int, s1: int, _rows: list | None) -> None:
        """Journal one applied span. Port of ``CompiledHexOccupancy._record`` with two deliberate
        differences: no pool index (this structure has one pool, so the key is the cell id), and the
        span is clamped to what ``block_range`` will ACTUALLY apply.

        The clamp is what makes the journal exactly reversible — what was applied is what is
        replayed — and it is also why there is no per-row ``_SPAN_LIMIT`` raise here, unlike the A*
        twin. A* records the RAW span, so a volume that outlives the box (a late return commits past
        MAXS and box-guards to the reference) can carry ``s1 >= _SPAN_LIMIT`` past the constructor's
        check and corrupt the packing; its per-row raise is load-bearing. Post-clamp,
        ``0 <= s0 <= s1 <= MAXS < _SPAN_LIMIT`` holds by construction (the constructor rejects a
        horizon that deep), so the same guard could never fire."""
        if s0 < 0:
            s0 = 0
        if s1 > self.MAXS:
            s1 = self.MAXS
        if s0 > s1:
            return
        packed = (s0 << _SPAN_BITS) | s1
        lst = self._claims.get(c)
        if lst is None:
            self._claims[c] = [packed]
        else:
            lst.append(packed)
        if _rows is not None:
            _rows.append(c)
            _rows.append(packed)

    def on_release(self, flight_id, volumes) -> None:
        """Ledger release subscriber (removal mode): drop the flight's recorded claims and rebuild
        exactly the cells it touched — reset each cell's interval list and re-apply the surviving
        claims (short per-cell lists) — in O(released rows) instead of a whole-pool reset+reabsorb.
        Keeps ``n_added`` in lockstep so the shrink tripwire stays silent.

        THE STATIC-WALL BRANCH IS NOT AN OPTIMISATION. ``reset_cell``'s blank slate is ``[0, MAXS]``
        — fully FREE — while a statically walled cell's correct blank slate is ``[0, -1]``, fully
        blocked. The claim journal only ever describes commit-derived blocks, so rebuilding a walled
        cell from it silently UNWALLS the hub for the rest of the run, and repairs then route foreign
        traffic through its terminal airspace. (`verify` would catch it, but `verify_every` defaults
        to 0.) Note the asymmetry that lets it hide: `block_range` on an already-walled cell reads
        head ``a=0, b=-1``, tests ``b < s0``, follows ``nxt == -1`` and returns — blocking is
        idempotent against a wall; un-blocking is not."""
        rows = self._rows.pop(flight_id)
        claims = self._claims
        touched: set[int] = set()
        for i in range(0, len(rows), 2):              # flat (cell, claim) pairs; see `__init__`
            c = rows[i]
            claims[c].remove(rows[i + 1])             # ValueError here IS the drift signal
            touched.add(c)
        for c in touched:
            self.reset_cell(c)                        # reclaims the overflow tail onto `_free`
            survivors = claims.get(c)
            if not survivors:
                claims.pop(c, None)                   # drop the empty list, both branches
            if c in self._static_cells:
                # Re-wall rather than rebuild. The `survivors` are deliberately KEPT (only an empty
                # list is dropped above): other flights still hold journal rows pointing at this cell,
                # and discarding their claims here would KeyError when THEY are released. A walled
                # cell simply never reads them.
                self.iv_lo[c] = 0; self.iv_hi[c] = -1; self.iv_nxt[c] = -1
                continue                              # a replay would also leak the slots it allocs
            for packed in survivors or ():
                self.block_range(c, packed >> _SPAN_BITS, packed & _FIELD_MASK)
        self.n_added -= len(volumes)

    def block_range(self, c: int, s0: int, s1: int) -> None:
        """Remove the whole contiguous span ``[s0, s1]`` from cell ``c``'s free intervals in one pass.

        A committed volume occupies each cell over a contiguous step range, so this replaces ``S``
        single-step calls with one walk — the SoA twin of ``CompiledHexOccupancy.block_range`` (issue
        #8 Phase E), which the hex image has had since the A* commit floor was profiled. The free-STEP
        set is identical to blocking ``s0, s0+1, …, s1`` one at a time, so the kernel is byte-unaffected.

        The chain (head = slot ``c``) is sorted ascending and every adjacent pair is separated by at
        least one blocked step, an invariant this method preserves: a split inserts the
        right remainder immediately after the left. The span may straddle several intervals once
        earlier commits punched holes — the first keeps a left remainder ``[a, s0-1]``, the last a
        right remainder ``[s1+1, b]``, and anything fully inside is marked empty (``lo>hi``) rather
        than unlinked, since the flat pool cannot cheaply drop a fixed head/middle slot and every
        reader (the kernel walk, ``free_intervals_py``) already skips ``lo > hi``.

        The ``a > s1`` early return is safe in the presence of those empty slots: an emptied interval
        keeps ``lo`` no greater than its successor's, so if it sits right of the span, so does
        everything after it.

        Every access re-reads ``self.iv_*``: ``_alloc`` can ``_grow``, which REPLACES all three arrays,
        so a hoisted reference would write into the dead buffer."""
        if s0 < 0:
            s0 = 0
        if s1 > self.MAXS:
            s1 = self.MAXS
        if s0 > s1:
            return
        slot = c
        while slot != -1:
            a, b = int(self.iv_lo[slot]), int(self.iv_hi[slot])
            nxt = int(self.iv_nxt[slot])
            if b < s0:                                    # wholly left of the span → keep walking
                slot = nxt
                continue
            if a > s1:                                    # wholly right (list sorted) → done
                return
            keep_left = a <= s0 - 1
            keep_right = s1 + 1 <= b
            if keep_left and keep_right:                  # span sits inside one interval → split once
                self.iv_hi[slot] = s0 - 1
                ns = self._alloc(s1 + 1, b, nxt)
                self.iv_nxt[slot] = ns
                return
            if keep_right:                                # right remainder is > s1 → nothing past it
                self.iv_lo[slot] = s1 + 1
                return
            if keep_left:                                 # left remainder kept; span may reach further
                self.iv_hi[slot] = s0 - 1
            else:                                         # interval fully covered → empty (lo>hi)
                self.iv_lo[slot] = 1
                self.iv_hi[slot] = 0
            slot = nxt
        # span already blocked (no interval overlaps it) → no-op

    def evict_before(self, step) -> None:
        if self.evicted_before is None or step > self.evicted_before:
            self.evicted_before = step             # queries read steps >= request clock; reclaim is TODO

    def reset(self) -> None:
        self.n_added = 0
        self.evicted_before = None
        self.oob_corridor_cells = 0
        # The journal describes the pool `_init_pool` is about to discard, so it goes with it —
        # otherwise `_claims[c].remove(...)` on the next release either raises (misread as drift) or
        # succeeds and re-blocks spans the fresh `_absorb` has already applied. `_static_cells` is
        # repopulated by the `_static_hubs` replay below; `_free` by `_init_pool`.
        self._claims.clear()
        self._rows.clear()
        self._init_pool()
        for center, term in self._static_hubs:      # _init_pool cleared the walls; re-wall each hub
            self._wall_static_terminal(center, term)

    def register_static_terminal(self, center, term) -> None:
        self._static_hubs.append((center, term))
        self._wall_static_terminal(center, term)

    _on_static = register_static_terminal   # ledger.subscribe_static hook name (main's A* contract)

    def _wall_static_terminal(self, center, term) -> None:
        """Permanently wall a hub's terminal airspace (column + exit lanes) off from FOREIGN traffic
        (``cfg.terminal_airspace_always_active``): force each footprint cell's pool interval EMPTY, so the
        kernel finds no free interval there and routes around. The planning flight's OWN hub lanes are
        restored per-flight by the overlay (built from ``SafeIntervalIndex``, which exempts own walls).
        Call AFTER ``_init_pool``/absorb; idempotent (re-emptying an empty interval is a no-op)."""
        for (q, r) in hg.terminal_cells(center, term, self.cfg):
            for L in range(self.nlevels):                # always-active walls the column at EVERY level
                c = self.cell_id(q, r, L)
                if c >= 0:
                    self.iv_lo[c] = 0; self.iv_hi[c] = -1; self.iv_nxt[c] = -1   # empty (lo>hi) ⇒ blocked
                    self._static_cells.add(c)            # `on_release` must re-wall, not rebuild

    # ---------- pure-Python reader (kernel parity oracle + tests) ----------
    def free_intervals_py(self, q: int, r: int, L: int, base: int, max_step: int):
        """Cell ``(q, r, L)``'s free intervals clipped to ``[base, max_step]`` — the exact view the kernel
        walks. Returns ``None`` only if out-of-box (the kernel would fall back to the reference)."""
        c = self.cell_id(q, r, L)
        if c < 0:
            return None
        out = []
        slot = c
        while slot != -1:
            lo = max(int(self.iv_lo[slot]), base)
            hi = min(int(self.iv_hi[slot]), max_step)
            if lo <= hi:
                out.append((lo, hi))
            slot = int(self.iv_nxt[slot])
        return out
