"""Dense per-(cell, level) free-interval pools for the compiled A* kernel (issue #8 Track B, A* port).

The numba kernel (:mod:`astar_kernel`) needs O(1) array reads to answer ``is (q, r, L) blocked at step
s?`` — A*'s per-node obstacle test — reproducing :meth:`HexOccupancyService.is_blocked` exactly. The
"cell" is a **(q, r, L)** triple (a hex at a flight level). Two flat interval pools, both maintained
incrementally from the ledger commit hook via :func:`hexgrid.rasterize_volume_dual`:

  * **corridor pool** (``corr``) — ordinary corridor cells (``in_blk`` from non-column volumes, minus the
    committing flight's own-column interior). Equals ``HexOccupancyService.blocked`` cell-for-cell.
  * **column pool** (``col``) — every terminal column's inner footprint. Equals ``term_cells`` (which
    hubs, dropped — only presence matters; own/foreign is resolved per-flight).

``is_blocked(q,r,L,s,own)`` then folds to (kernel ``_blocked``):

    colb = column-blocked(cell,s);  corb = corridor-blocked(cell,s)
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

import numpy as np

from ..geometry import CylinderSpec
from . import hexgrid as hg


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


class _Pool:
    """Flat linked-list free-interval pool: cell ``c``'s intervals are walked from slot ``c`` along
    ``nxt``; a blocked step splits the containing interval in place. Slot 0..NC-1 pre-seeded ``[0, MAXS]``."""

    def __init__(self, NC: int, MAXS: int):
        self.NC = NC
        self.MAXS = MAXS
        cap = max(2 * NC, 1 << 18)
        self.cap = cap
        self.lo = np.empty(cap, np.int32)
        self.hi = np.empty(cap, np.int32)
        self.nxt = np.empty(cap, np.int32)
        self.lo[:NC] = 0
        self.hi[:NC] = MAXS
        self.nxt[:NC] = -1
        self.nslots = NC

    def reset(self):
        self.lo[: self.NC] = 0
        self.hi[: self.NC] = self.MAXS
        self.nxt[: self.NC] = -1
        self.nslots = self.NC

    def _grow(self):
        cap = self.cap * 2
        for name in ("lo", "hi", "nxt"):
            a = np.empty(cap, np.int32)
            a[: self.cap] = getattr(self, name)
            setattr(self, name, a)
        self.cap = cap

    def _alloc(self, lo, hi, nxt) -> int:
        if self.nslots >= self.cap:
            self._grow()
        s = self.nslots
        self.lo[s] = lo; self.hi[s] = hi; self.nxt[s] = nxt
        self.nslots += 1
        return s

    def block(self, c: int, s: int) -> None:
        """Split cell ``c``'s free interval containing ``s`` (in place)."""
        if s < 0 or s > self.MAXS:
            return
        slot = c
        while slot != -1:
            a, b = int(self.lo[slot]), int(self.hi[slot])
            if a <= s <= b:
                if s + 1 <= b:
                    if a <= s - 1:
                        self.hi[slot] = s - 1
                        ns = self._alloc(s + 1, b, int(self.nxt[slot]))
                        self.nxt[slot] = ns
                    else:
                        self.lo[slot] = s + 1
                elif a <= s - 1:
                    self.hi[slot] = s - 1
                else:
                    self.lo[slot] = s + 1
                return
            slot = int(self.nxt[slot])

    def blocked_at(self, c: int, s: int) -> bool:
        """True iff step ``s`` is in NO free interval of cell ``c``."""
        slot = c
        while slot != -1:
            if int(self.lo[slot]) <= s <= int(self.hi[slot]):
                return False
            slot = int(self.nxt[slot])
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
        # cell → {terminal ids whose column ever covers it}. Lets the host detect an own∩foreign shared
        # cell (issue #3) and fall back to the reference, instead of the overlay boolean silently treating a
        # foreign column as transparent. Column cells only (a small footprint), so the memory is tiny.
        self.col_owners: dict[int, set] = {}
        # committed corridor cells that fell outside the box: skipped (never a crash); any query to such a
        # cell gets cell_id < 0 and the kernel falls back via FB_OOB. Non-zero ⇒ consider widening `margin`.
        self.oob_corridor_cells = 0

    def _box(self, cfg, margin):
        w, h = cfg.region_size_m
        R = self.R
        qs, rs = [], []
        for x, y in ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h)):
            q, r = hg.enu_to_axial(x, y, R)
            qs.append(q); rs.append(r)
        qmin, qmax = min(qs) - margin, max(qs) + margin
        rmin, rmax = min(rs) - margin, max(rs) + margin
        # MAXS = the worst-case per-flight search_horizon (latest departure + region-DIAGONAL flight; >=
        # every flight's max_step since search_horizon is monotone) + the committed landing hover tail.
        dt = cfg.dt_s
        pitch = cfg.nominal_speed_mps * dt
        levels = cfg.flight_levels_m
        base_max = int(math.ceil(cfg.horizon_s / dt))
        takeoff_max = max(cfg.climb_steps_to(z) for z in levels)
        n_hops_max = int(math.ceil(math.hypot(w, h) / max(pitch, 1e-9)))
        climb_span = (int(math.ceil((levels[-1] - levels[0]) / (cfg.climb_rate_mps * dt)))
                      if cfg.n_levels > 1 else 0)
        maxs = search_horizon(base_max, takeoff_max, n_hops_max, climb_span, cfg) + hover_tail_steps(cfg)
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
        for q, r, L, s, in_blk in hg.rasterize_volume_dual(
            vol, self.cfg, self.R, self.infl_blocked, self.infl_pad
        ):
            if not in_blk:
                continue
            if self.evicted_before is not None and s < self.evicted_before:
                continue
            c = self.cell_id(q, r, L)
            if is_column:                               # → column pool (all columns; own/foreign per plan)
                if c >= 0:
                    self.col.block(c, int(s))
                    self.col_owners.setdefault(c, set()).add(tid)
            else:                                       # → corridor pool (minus committing own interior)
                if own_cols and self._inside_a_column(q, r, own_cols):
                    continue
                if c < 0:                               # outside the box → skip (never crash on commit);
                    self.oob_corridor_cells += 1        # a query to this cell gets cell_id<0 → kernel FB_OOB
                    continue
                self.corr.block(c, int(s))

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

    # ---------- pure-Python oracle (kernel parity + tests) ----------
    def blocked_py(self, q: int, r: int, L: int, s: int, own_cells=None) -> bool:
        """Point query reproducing the kernel ``_blocked`` (and thus ``HexOccupancyService.is_blocked``).

        ``own_cells``: a set of ``cell_id``s that are the planning flight's OWN column footprint (empty /
        ``None`` for ``own=∅`` — the occupancy-parity contract vs ``is_blocked(..., own=())``). Out-of-box ⇒
        ``True`` (the kernel would FALLBACK)."""
        c = self.cell_id(q, r, L)
        if c < 0:
            return True
        colb = self.col.blocked_at(c, s)
        if colb and (own_cells is None or c not in own_cells):
            return True                                 # foreign column → wall
        return self.corr.blocked_at(c, s)               # corridor / own-column fixed-lane sibling
