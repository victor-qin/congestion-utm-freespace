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

Cells live in a box from the region corners + a reroute ``margin`` (a committed corridor cell outside is
an error — widen ``margin``). ``MAXS`` covers the worst-case per-flight ``max_step`` (a region-diagonal
flight's search horizon), so the pools' free intervals never end before a reachable step. The kernel
bound-checks query cells/steps and falls back to the reference on any stray.
"""
from __future__ import annotations

import math

import numpy as np

from ..geometry import CylinderSpec
from . import hexgrid as hg


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

    def _box(self, cfg, margin):
        w, h = cfg.region_size_m
        R = self.R
        qs, rs = [], []
        for x, y in ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h)):
            q, r = hg.enu_to_axial(x, y, R)
            qs.append(q); rs.append(r)
        qmin, qmax = min(qs) - margin, max(qs) + margin
        rmin, rmax = min(rs) - margin, max(rs) + margin
        # MAXS must be >= the largest per-flight max_step (astar.py: base + takeoff + ground_delay +
        # 3*n_hops + 2*climb_span + 6). Worst case: latest departure and a region-DIAGONAL flight.
        dt = cfg.dt_s
        pitch = cfg.nominal_speed_mps * dt
        levels = cfg.flight_levels_m
        base_max = int(math.ceil(cfg.horizon_s / dt))
        takeoff_max = max(cfg.climb_steps_to(z) for z in levels)
        n_hops_max = int(math.ceil(math.hypot(w, h) / max(pitch, 1e-9)))
        climb_span = (int(math.ceil((levels[-1] - levels[0]) / (cfg.climb_rate_mps * dt)))
                      if cfg.n_levels > 1 else 0)
        maxs = (base_max + takeoff_max + int(math.ceil(cfg.max_ground_delay_s / dt))
                + 3 * n_hops_max + 2 * climb_span + 6 + 16)
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
            else:                                       # → corridor pool (minus committing own interior)
                if own_cols and self._inside_a_column(q, r, own_cols):
                    continue
                if c < 0:
                    raise IndexError(
                        f"committed corridor cell ({q},{r},L={L}) outside kernel box — widen margin")
                self.corr.block(c, int(s))

    def evict_before(self, step) -> None:
        if self.evicted_before is None or step > self.evicted_before:
            self.evicted_before = step

    def reset(self) -> None:
        self.n_added = 0
        self.evicted_before = None
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
