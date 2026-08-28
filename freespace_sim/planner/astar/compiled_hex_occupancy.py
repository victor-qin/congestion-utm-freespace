"""Dense per-(cell, level) free-interval pools for the compiled A* kernel (issue #8 Track B, A* port).

The numba kernel (:mod:`kernel`) needs O(1) array reads to answer ``is (q, r, L) blocked at step
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
hover tail) are recorded but never read: every kernel query is ``≤ max_step ≤ MAXS`` (guarded in
``_plan_compiled``).
"""
from __future__ import annotations

import math
import warnings
from array import array

import numpy as np

from ...geometry import CylinderSpec
from ...types import as_terminal
from .. import hexgrid as hg
from .claim_arena import ClaimArena


def ground_delay_steps(cfg) -> int:
    """Largest number of whole ``dt`` waits whose reported delay stays within the configured cap.

    Ground delay is emitted in whole timesteps, so rounding the cap upward can produce an intent whose
    ``ground_delay_s`` exceeds ``max_ground_delay_s`` (for example, 8 s at ``dt_s=4`` under a 5 s cap).
    Keep the tiny tolerance used by colgen so an exactly integral floating-point ratio is not rounded down.
    """
    return int(math.floor(cfg.max_ground_delay_s / cfg.dt_s + 1e-12))


def search_horizon(base: int, takeoff_steps_max: int, n_hops: int, climb_span: int, cfg) -> int:
    """The largest ``step`` an A* plan can reach: takeoff + a 3× lateral detour budget + a full ground-
    delay allowance + the mid-route climb span. ONE definition (issue #5) — ``_plan_reference``,
    ``_plan_compiled``, and ``CompiledHexOccupancy._box`` (with worst-case args) all call it, so the
    kernel's search bound, the box guard, and ``MAXS`` cannot drift apart. Monotone in ``base``/``n_hops``,
    so ``_box``'s worst-case value bounds every per-flight one."""
    return (base + takeoff_steps_max + ground_delay_steps(cfg)
            + 3 * n_hops + 2 * climb_span + 6)


def hover_tail_steps(cfg) -> int:
    """Extra steps a committed landing column occupies PAST the arrival step — hover dwell + climb to the
    top level + the ASTM time buffer, in dt units (mirrors ``volumes.hover_reservation`` /
    ``hexgrid._step_range``). ``MAXS`` adds this so the box covers every committed step;
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


# Packed-claim field layout (see `claim_arena`): three 20-bit fields in an int64,
# s0 | s1 | fid_code. 20 bits is ~1e6 — four orders of magnitude past any realistic step horizon or
# flight count — and both are range-checked where they enter, so a field can never silently wrap.
_SPAN_BITS = 20
_S0_SHIFT = 2 * _SPAN_BITS
_SPAN_LIMIT = 1 << _SPAN_BITS
_FIELD_MASK = _SPAN_LIMIT - 1


class CompiledHexOccupancy:
    """Two incremental flat pools (corridor + column) feeding the numba A* kernel. Commit-hook driven."""

    def __init__(self, cfg, margin: int = 64, track_removal: bool = False):
        self.cfg = cfg
        self.R = hg.circumradius(cfg)
        self.infl_blocked = cfg.corridor_width_m / 2.0 + self.R
        self.infl_pad = cfg.effective_hover_radius_m + self.R
        self.n_levels = cfg.n_levels
        self.n_added = 0
        self.evicted_before: int | None = None
        # Kept for compatibility with callers that still pass it; the per-owner row stream it used to
        # gate is now unconditional, because it is how the arena is fed rather than an opt-in journal.
        self.track_removal = track_removal
        # THE occupancy. One claim is one int64, in a flat arena (see `claim_arena`) constructed below
        # once NC is known:
        #     key    = c << 1 | pool_idx                 (corridor or column, which cell)
        #     claim  = s0 << 40 | s1 << 20 | fid_code    (`_fids[fid_code]` recovers the owner)
        # Packed because it is per-claim and therefore linear in schedule size (the tuple form
        # measured 68 MB at 290 flights). Ranges are checked in `_record`; the constructor rejects a
        # horizon too deep to pack. `_rows[fid]` is the same pairs per owner, which is what makes a
        # release O(the flight's own footprint).
        self._arena: ClaimArena
        self._rows: dict[int, array] = {}            # fid -> flat int64 (key, claim) pairs
        self._nvol: dict[int, int] = {}              # fid -> volumes absorbed
        self._fids: list = []                        # fid_code -> flight id
        self._fid_codes: dict = {}                   # flight id -> fid_code

        qmin, rmin, qspan, rspan, maxs = self._box(cfg, margin)
        self.qmin, self.rmin, self.qspan, self.rspan = qmin, rmin, qspan, rspan
        self.NC = qspan * rspan * self.n_levels
        self.MAXS = maxs
        if maxs >= _SPAN_LIMIT:       # see the packed-claim layout: s0/s1 get 20 bits each. No longer
            #                           `track_removal`-gated: every mode records claims now, so a
            #                           too-deep horizon must fail HERE, with a readable message,
            #                           rather than mid-commit inside `_record`.
            raise ValueError(
                f"CompiledHexOccupancy: horizon of {maxs} steps exceeds the removal journal's "
                f"{_SPAN_LIMIT}-step packing limit")
        self._arena = ClaimArena(2 * self.NC, _S0_SHIFT, _SPAN_BITS, _FIELD_MASK)
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
    def on_commit(self, flight_id, volumes) -> None:
        """Ledger commit subscriber: rasterize the flight's volumes into claims and add them.

        The per-owner row stream is no longer optional. It was `track_removal`-gated when it existed
        only to let `on_release` un-absorb; now it is how the arena — the occupancy itself — is fed,
        so every mode records. `rows` is exactly THIS commit's pairs, where `_rows[fid]` accumulates
        across commits for the same flight, so the arena must be fed from the former."""
        hg.prepare_range_cache_for_commit(volumes)   # main (#117): size the shared raster LRU so
        #                                              every observer of THIS commit reuses one sweep
        own_cols = tuple((v.shape.cx, v.shape.cy, v.shape.radius) for v in volumes
                         if v.terminal_id is not None and isinstance(v.shape, CylinderSpec))
        rows: list = []
        for v in volumes:
            self._add(v, own_cols, flight_id, rows)
        self.n_added += len(volumes)
        entry = self._rows.get(flight_id)
        if entry is None:
            self._rows[flight_id] = array("q", rows)
        else:
            entry.extend(rows)
        self._nvol[flight_id] = self._nvol.get(flight_id, 0) + len(volumes)
        if rows:
            flat = np.asarray(rows, np.int64).reshape(-1, 2)
            self._arena.add(flat[:, 0], flat[:, 1])

    def on_release(self, flight_id, volumes) -> None:
        """Ledger release subscriber: drop the flight's claims. O(ITS OWN footprint).

        This is the whole point of the pool-less occupancy. The interval pools stored FREE intervals,
        which can absorb a block but cannot subtract one, so this method used to reset every cell the
        flight touched and re-apply all the SURVIVING claims on it — measured at 12.2x the flight's
        own footprint at density_faa scale, and growing with congestion, because the multiplier is how
        many OTHER flights share those cells. A claim is a blocked span, so removing one is removing
        one. ``col_owners`` is still deliberately NOT pruned (documented conservative superset)."""
        rows = self._rows.pop(flight_id)
        if rows:
            flat = np.frombuffer(rows, dtype=np.int64).reshape(-1, 2)
            self._arena.remove(np.ascontiguousarray(flat[:, 0]),
                               np.ascontiguousarray(flat[:, 1]))
        self.n_added -= self._nvol.pop(flight_id)

    def _inside_a_column(self, q, r, cols) -> bool:
        c = hg.hex_center(q, r, self.R)
        return any((c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= rad * rad for cx, cy, rad in cols)

    def _add(self, vol, own_cols, fid=None, _rows: list | None = None) -> None:
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
            if is_column:                               # column claims: own/foreign resolved per plan
                if c >= 0:
                    self.col_owners.setdefault(c, set()).add(tid)
                    self._record(1, c, int(s_lo), int(s_hi), fid, _rows)
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
                self._record(0, c, int(s_lo), int(s_hi), fid, _rows)

    def _record(self, pool_idx: int, c: int, s0: int, s1: int, fid, _rows: list | None) -> None:
        if s1 >= _SPAN_LIMIT:
            # A committed volume can outlive the box (a late return commits past MAXS and box-guards
            # to the reference), so the constructor's MAXS check does not bound this. One compare
            # against a field overflow that would silently corrupt a survivor's span on release.
            raise ValueError(f"CompiledHexOccupancy: step {s1} exceeds the removal journal's "
                             f"{_SPAN_LIMIT}-step packing limit")
        code = self._fid_codes.get(fid)
        if code is None:
            code = self._fid_codes[fid] = len(self._fids)
            if code >= _SPAN_LIMIT:
                raise ValueError("CompiledHexOccupancy: too many distinct flights for the "
                                 "removal journal's packing")
            self._fids.append(fid)
        _rows.append((c << 1) | pool_idx)
        _rows.append((s0 << _S0_SHIFT) | (s1 << _SPAN_BITS) | code)

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
        self._rows.clear()
        self._nvol.clear()
        self._arena.reset()
        # `_fids` is an interning pool — value-identical across a rebuild, so it survives reset().
        # Static terminals are NOT ledger-derived (a shrink rebuild must keep them) — re-mark them into the
        # freshly-cleared col_owners (static_col was never cleared, so this is idempotent). Mirrors
        # HexOccupancyService.reset() leaving static_term_cells intact.
        for center, term in self._static_terms:
            self._mark_static(center, term)

    # ---------- pure-Python oracle (kernel parity + tests) ----------
    def blocked_py(self, q: int, r: int, L: int, s: int, own_cells=None) -> bool:
        """Point query reproducing the kernel's fold — the pure-Python oracle every compiled-path
        parity test compares against.

        Now a scan of the cell's claim slab. That is ~3.3x more work per probe than the interval
        walk it replaces, which is why the per-plan window is no longer optional: the kernel reads
        the window, not this, and this exists for tests and diagnostics.

        ``own_cells``: cell ids that are the planning flight's OWN column footprint (empty / ``None``
        for ``own=∅``). Out-of-box ⇒ ``True`` (the kernel would fall back)."""
        c = self.cell_id(q, r, L)
        if c < 0:
            return True
        colb = self._arena.blocked((c << 1) | 1, s) or bool(self.static_col[c])
        if colb and (own_cells is None or c not in own_cells):
            return True                                 # foreign column → wall
        return self._arena.blocked(c << 1, s)           # corridor / own-column fixed-lane sibling
