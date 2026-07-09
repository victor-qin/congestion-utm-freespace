"""Dense per-cell free-interval pool for the compiled SIPP kernel (issue #8, Track B).

The numba kernel (:mod:`sipp_kernel`) needs O(1) array reads of each hex cell's safe intervals. At
density a cell fragments into *many* disjoint free runs (many flights cross it at different times), so a
fixed-width ``[cell, K]`` table overflows — instead each interval is a **slot in a flat linked-list
pool**, maintained incrementally from the ledger commit hook:

    slot s:  iv_lo[s], iv_hi[s]  (free step-run, clipped per-flight to [base, max_step]),  iv_nxt[s]
    cell c's first interval is slot ``c`` (slots [0, NC) are pre-seeded, one per cell = [0, MAXS]);
    further intervals of c live in slots [NC, nslots) and are linked via iv_nxt from slot c.

So a cell's intervals are walked from slot ``c`` along ``iv_nxt``; the **slot index is the kernel's
frontier node id** (unique per (cell, interval), no wasted width, no overflow). A blocked step splits
the containing interval in place (+ maybe one new slot) — O(intervals in the cell) per (cell, step).
Degenerate slots (``lo > hi``, from single-step blocks) are left in the chain and skipped by the kernel.

The global pool encodes the **non-terminal corridor** occupancy (flight-independent). Terminal cells
(shared columns) are own-dependent — recorded in ``self.cols`` for the Phase-2 per-flight patch, unused
by the Phase-1 non-terminal kernel.

Cells are encoded over a bounding box from the region corners + a reroute ``margin``; a committed cell
outside the box is an error (widen ``margin``). The kernel bound-checks *query* cells and falls back to
the pure-Python reference for any out-of-box stray (the planner never enforced region bounds), so the
margin only needs to cover realistic edge-skirting detours.
"""
from __future__ import annotations

import numpy as np

from ..geometry import CylinderSpec
from . import hexgrid as hg


class CompiledOccupancy:
    """Incremental flat-pool free-interval store feeding the numba SIPP kernel. Commit-hook driven."""

    def __init__(self, cfg, margin: int = 48):
        self.cfg = cfg
        self.R = hg.circumradius(cfg)
        self.infl_blocked = cfg.corridor_width_m / 2.0 + self.R
        self.infl_pad = cfg.effective_hover_radius_m + self.R
        self.n_added = 0
        self.evicted_before: int | None = None

        self.nlevels = cfg.n_levels                  # flight-level axis (multi-altitude): cell = (q, r, L)
        qmin, rmin, qspan, rspan, maxs = self._box(cfg, margin)
        self.qmin, self.rmin, self.qspan, self.rspan = qmin, rmin, qspan, rspan
        self.NC = qspan * rspan * self.nlevels        # one pre-seeded slot per (q, r, L) cell
        self.MAXS = maxs
        self._init_pool()

    def _box(self, cfg, margin):
        w, h = cfg.region_size_m
        R = self.R
        qs, rs = [], []
        for x, y in ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h)):
            q, r = hg.enu_to_axial(x, y, R)
            qs.append(q); rs.append(r)
        qmin, qmax = min(qs) - margin, max(qs) + margin
        rmin, rmax = min(rs) - margin, max(rs) + margin
        dt = cfg.dt_s
        maxs = int(np.ceil(cfg.horizon_s / dt) + np.ceil(cfg.max_ground_delay_s / dt)) + 64
        return qmin, rmin, qmax - qmin + 1, rmax - rmin + 1, maxs

    def _init_pool(self):
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
        if self.nslots >= self.cap:
            self._grow()
        s = self.nslots
        self.iv_lo[s] = lo; self.iv_hi[s] = hi; self.iv_nxt[s] = nxt
        self.nslots += 1
        return s

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
        for q, r, L, s, in_blk in hg.rasterize_volume_dual(
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
                raise IndexError(
                    f"committed corridor cell ({q},{r},L={L}) outside kernel box — widen margin")
            self._block(c, int(s))

    def _block(self, c: int, s: int) -> None:
        """Split cell ``c``'s free interval containing step ``s`` (linked-list pool, in place)."""
        if s < 0 or s > self.MAXS:
            return
        slot = c
        while slot != -1:
            a, b = int(self.iv_lo[slot]), int(self.iv_hi[slot])
            if a <= s <= b:                               # interval [a,b] contains s → split out s
                if s + 1 <= b:
                    if a <= s - 1:                        # both sides survive
                        self.iv_hi[slot] = s - 1
                        ns = self._alloc(s + 1, b, int(self.iv_nxt[slot]))
                        self.iv_nxt[slot] = ns
                    else:                                  # s == a → slot becomes the right part
                        self.iv_lo[slot] = s + 1
                elif a <= s - 1:                           # s == b → slot becomes the left part
                    self.iv_hi[slot] = s - 1
                else:                                      # s == a == b → degenerate (lo>hi), skipped
                    self.iv_lo[slot] = s + 1
                return
            slot = int(self.iv_nxt[slot])
        # s already blocked (no interval contains it) → no-op

    def evict_before(self, step) -> None:
        if self.evicted_before is None or step > self.evicted_before:
            self.evicted_before = step             # queries read steps >= request clock; reclaim is TODO

    def reset(self) -> None:
        self.n_added = 0
        self.evicted_before = None
        self._init_pool()

    def register_static_terminal(self, center, term) -> None:
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
