"""Incremental hex-occupancy service for the space-time A* planner.

A*'s search needs two cell maps derived from the committed volumes: ``blocked`` (the corridor
footprint a flight must avoid) and ``pad`` (the wider hover-cylinder footprint used for the
takeoff/landing dwell check). These maps are **global and flight-independent** — a cell's
membership depends only on the committed volumes and ``cfg``, never on who is planning — so rather
than rebuild them from scratch every plan (O(committed) per plan → O(N²) per run), this service
maintains them incrementally: each committed volume is rasterized **exactly once** (the dual sweep
in :func:`hexgrid.rasterize_volume_dual`) when the ledger publishes its commit, and cells older than
the request clock are evicted so memory stays bounded to the active time window.

**Shared terminal columns.** A committed *terminal column* (``vol.terminal_id is not None``) is NOT
an ordinary obstacle — it's a multi-pad vertiport shared by its own hub's flights and walled off from
everyone else. Such volumes are kept out of the binary ``blocked``/``pad`` sets and instead recorded
in ``term_cells`` (``step -> cell -> {terminal_id}``) — a per-cell SET of the hubs whose columns cover
that cell. :meth:`is_blocked` uses it for the **own-hub cruise exemption**: a cell occupied *only* by
the flight's own terminal(s) is transparent (a hub's flights fly through their shared column), while a
*foreign* hub's column is a wall (cruise reroutes around busy vertiports). **Pad capacity is NOT
counted here** — up-to-``capacity`` concurrent same-hub dwells are gated temporally by
:class:`~freespace_sim.planner.terminal_capacity.TerminalCapacity`, which the A* planner consults at
the takeoff/landing gate. A run with no terminals never touches ``term_cells`` (it stays empty → zero
overhead), so the binary maps are byte-identical to before — the property the occupancy tests pin.

ASTM framing: the planner's USS holds this as the local picture fed by DSS commit notifications
(F3548-21 Subscriptions) — see ``ReservationLedger.subscribe`` (the publish hook).

Two invariants this relies on (both true in the current single-USS, single-thread, FCFS sim, and
both guarded):
  * **monotonic time** — requests are processed in non-decreasing ``t_request`` order
    (``scenario.py`` sorts events), so a future flight only ever occupies steps ``>= now``; evicting
    earlier steps can never drop a cell anyone will query.
  * **add-only** — commits only add volumes (``ledger.release`` is test-only). A ledger *shrink*
    (a release) is detected by the planner, which rebuilds the service from scratch and warns.

Cells are bucketed by step (``step -> {(q, r)}``); volumes themselves are NOT retained.
"""

from __future__ import annotations

from array import array
from collections.abc import Collection, Hashable

from .. import hexgrid as hg
from ...config import SimConfig
from ...geometry import CylinderSpec
from ...types import as_terminal
from ...volumes import Volume4D

_EMPTY: dict = {}


class HexOccupancyService:
    def __init__(self, cfg: SimConfig, track_removal: bool = False):
        self.cfg = cfg
        # Removal mode (LNS destroy): step buckets hold per-cell REFCOUNTS instead of sets (two
        # flights' inflated rasters can legitimately cover the same (step, cell), so removing one
        # must not free the cell), and each committed flight's applied rows are recorded so
        # `on_release` can reverse them exactly. Membership queries are unchanged (`in` works on
        # dict keys); flag off ⇒ the original set-based structures, byte-for-byte.
        # Journal layout (mode on): `_rows[fid]` is a FLAT int64 array of 4-slot rows
        # ``(cell_id, s_lo, s_hi, code)`` rather than a list of ``(kind, cell, s_lo, s_hi, extra)``
        # tuples — measured 185 B/row against 32 B here, and at ~900 rows per flight the tuple form
        # cost 49 MB at 290 flights (linear in schedule size). ``cell_id`` indexes `_cells`, which
        # interns each ``(q, r, L)`` ONCE, so a reversal reads back the very tuple it inserted with:
        # no per-row allocation, unlike re-packing ``(q, r, L)`` out of three stored ints.
        # ``code``: -1 = corridor pad-only, -2 = corridor pad+blocked, >= 0 = terminal column whose
        # hub id is ``_tids[code]``. The release callback supplies the exact committed volumes.
        self.track_removal = track_removal
        self._rows: dict[int, array] = {}                         # fid -> flat rows (mode on)
        self._cells: list[tuple[int, int, int]] = []              # cell_id -> (q, r, L)
        self._cell_ids: dict[tuple[int, int, int], int] = {}      # (q, r, L) -> cell_id
        self._tids: list = []                                     # code -> terminal id
        self._tid_ids: dict = {}                                  # terminal id -> code
        self.R = hg.circumradius(cfg)
        self.infl_blocked = cfg.corridor_width_m / 2.0 + self.R   # corridor footprint
        self.infl_pad = cfg.effective_hover_radius_m + self.R     # wider hover-cylinder footprint
        self.blocked: dict[int, set[tuple[int, int, int]]] = {}   # step -> {(q, r, L)}  (non-terminal)
        self.pad: dict[int, set[tuple[int, int, int]]] = {}       # step -> {(q, r, L)}  (non-terminal)
        # shared terminal columns: step -> (q, r, L) -> {terminal_id}  (which hubs' columns cover the cell)
        self.term_cells: dict[int, dict[tuple[int, int, int], set[Hashable]]] = {}
        # always-active terminals (cfg.terminal_airspace_always_active): permanent FOREIGN walls, step- AND
        # level-independent (the column is the [ground, ceiling] tube), keyed by (q, r) only. Derived from the
        # ledger's PERMANENT terminal volumes via the `subscribe_static` hook (`_on_static`), NOT from
        # committed corridor volumes — so `reset()` (a from-scratch rebuild on ledger shrink) leaves it intact
        # (the hub set doesn't change). Empty ⇒ zero overhead when the flag is off.
        self.static_term_cells: dict[tuple[int, int], set[Hashable]] = {}
        self.n_added = 0                  # committed volumes absorbed (shrink tripwire)
        self.evicted_before: int | None = None   # lowest retained step

    # ----- maintenance -----
    @staticmethod
    def _bump(bucket: dict, s: int, key, tid=None) -> None:
        """Refcounted insert (removal mode): blocked/pad hold cell->count; term_cells holds
        cell -> {tid: count}."""
        d = bucket.setdefault(s, {})
        if tid is None:
            d[key] = d.get(key, 0) + 1
        else:
            e = d.setdefault(key, {})
            e[tid] = e.get(tid, 0) + 1

    @staticmethod
    def _drop(bucket: dict, s: int, key, tid=None) -> None:
        """Exact reverse of `_bump`; raises KeyError on drift (a row removed twice or never added)."""
        d = bucket[s]
        if tid is None:
            n = d[key] - 1
            if n:
                d[key] = n
            else:
                del d[key]
        else:
            e = d[key]
            n = e[tid] - 1
            if n:
                e[tid] = n
            else:
                del e[tid]
                if not e:
                    del d[key]
        if not d:
            del bucket[s]

    def add_volume(self, vol: Volume4D, own_cols: tuple = (), _rows: list | None = None) -> None:
        """Rasterize one committed volume (once). Ordinary corridor cells feed the binary blocked/pad
        step-buckets; a shared terminal column instead records its hub id in the per-cell set.

        ``own_cols`` is the committing flight's own terminal columns ``(cx, cy, radius)``. A corridor
        cell falling INSIDE one of them is the vertiport's unreserved tactical interior (the flight's
        exit lane proper lies outside the column and is still recorded), so it's skipped — leaving only
        *foreign* corridors inside any hub's column for a launch to detect and wait out (see pad_clear).
        """
        tid = vol.terminal_id
        # Only a tagged *column* (hover cylinder) feeds the per-cell hub set; a tagged *corridor*
        # box (an in-terminal exit lane) is still a corridor — it goes to blocked/pad like any other,
        # so it is never mistaken for a column cell. ("column ⟺ cylinder"; stored kind is issue #11.)
        is_column = tid is not None and isinstance(vol.shape, CylinderSpec)
        # The hex service is step-keyed (dict[s] → cell set), so it expands each cell's range back to
        # its steps; the shared geometry sweep is still done once (via rasterize_ranges' memo), which
        # is what the compiled image reuses. `own`/`in_blk`/`is_column` are per-cell, hoisted out of
        # the step loop.
        track = self.track_removal
        pad_b, blk_b, floor = self.pad, self.blocked, self.evicted_before
        for q, r, L, s_lo, s_hi, in_blk in hg.rasterize_ranges(
            vol, self.cfg, self.R, self.infl_blocked, self.infl_pad
        ):
            if floor is not None and s_lo < floor:
                s_lo = floor                 # guard: never resurrect an already-evicted past step
            if not is_column:
                own = own_cols and self._inside_a_column(q, r, own_cols)
                if own:
                    continue             # the committing flight's own terminal interior — unreserved
                cell = (q, r, L)
                # `_bump`/`setdefault(s, <new container>)` inlined over the step run: this is the
                # service's hottest loop (8.4 M bumps in a 60-iteration LNS run) and `setdefault`'s
                # default argument is built EAGERLY, so the dict/set it allocated was thrown away on
                # every hit. `.get` + a None test allocates only on a genuinely new step.
                if track:
                    # Intern whenever the journal is on, `_rows` or not: the bucket key must be the
                    # canonical object either way, or the sharing that makes the packing pay is lost.
                    cell, cid = self._intern(cell)
                    if _rows is not None:
                        _rows.append(cid)
                        _rows.append(s_lo)
                        _rows.append(s_hi)
                        _rows.append(-2 if in_blk else -1)
                    for s in range(s_lo, s_hi + 1):
                        d = pad_b.get(s)
                        if d is None:
                            d = pad_b[s] = {}
                        d[cell] = d.get(cell, 0) + 1
                        if in_blk:
                            d = blk_b.get(s)
                            if d is None:
                                d = blk_b[s] = {}
                            d[cell] = d.get(cell, 0) + 1
                else:
                    for s in range(s_lo, s_hi + 1):
                        d = pad_b.get(s)
                        if d is None:
                            d = pad_b[s] = set()
                        d.add(cell)
                        if in_blk:
                            d = blk_b.get(s)
                            if d is None:
                                d = blk_b[s] = set()
                            d.add(cell)
            elif in_blk:
                # shared terminal column: record `tid` over its inner (blocked-strength) footprint at
                # level L — the cells A* queries for the own-hub cruise exemption (capacity lives in
                # TerminalCapacity).
                cell = (q, r, L)
                if track:
                    cell, cid = self._intern(cell)
                    if _rows is not None:
                        code = self._tid_ids.get(tid)
                        if code is None:
                            code = self._tid_ids[tid] = len(self._tids)
                            self._tids.append(tid)
                        _rows.append(cid)
                        _rows.append(s_lo)
                        _rows.append(s_hi)
                        _rows.append(code)
                    for s in range(s_lo, s_hi + 1):
                        self._bump(self.term_cells, s, cell, tid)
                else:
                    for s in range(s_lo, s_hi + 1):
                        self.term_cells.setdefault(s, {}).setdefault(cell, set()).add(tid)
        self.n_added += 1

    def _intern(self, cell: tuple[int, int, int]) -> tuple[tuple[int, int, int], int]:
        """``(canonical_cell, cell_id)``, registering the cell if new. Returning the canonical TUPLE
        (not just its id) is what lets the bucket keys, the journal and `on_release` all share ONE
        tuple per distinct cell — the difference between 80 bytes per row and 80 bytes per cell."""
        cid = self._cell_ids.get(cell)
        if cid is None:
            cid = self._cell_ids[cell] = len(self._cells)
            self._cells.append(cell)
            return cell, cid
        return self._cells[cid], cid

    def _inside_a_column(self, q: int, r: int, cols: tuple) -> bool:
        c = hg.hex_center(q, r, self.R)
        return any((c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= rad * rad for cx, cy, rad in cols)

    def on_commit(self, flight_id, volumes) -> None:
        """Ledger commit subscriber (the publish hook): absorb a newly committed flight's volumes,
        dropping the corridor cells inside its own terminal columns (the unreserved tactical interior)."""
        hg.prepare_range_cache_for_commit(volumes)
        own_cols = tuple((v.shape.cx, v.shape.cy, v.shape.radius) for v in volumes
                         if v.terminal_id is not None and isinstance(v.shape, CylinderSpec))
        rows = [] if self.track_removal else None
        for v in volumes:
            self.add_volume(v, own_cols=own_cols, _rows=rows)
        if self.track_removal:
            entry = self._rows.get(flight_id)
            if entry is None:
                self._rows[flight_id] = array("q", rows)   # 'q' = int64: cell ids and steps cannot
                #                                            overflow it, so packing stays total
            else:
                entry.extend(rows)

    def on_release(self, flight_id, volumes) -> None:
        """Ledger release subscriber (removal mode): reverse the flight's recorded rows so the
        maps stay exact without a rebuild — and keep ``n_added`` in lockstep with the ledger so
        the shrink tripwire stays silent. Steps already evicted are skipped (eviction dropped
        them; the same clamp `add_volume` applies on insert)."""
        rows = self._rows.pop(flight_id)
        floor = self.evicted_before
        cells, pad_b, blk_b = self._cells, self.pad, self.blocked
        for i in range(0, len(rows), 4):               # flat 4-slot rows; see `_rows`
            s_lo, s_hi, code = rows[i + 1], rows[i + 2], rows[i + 3]
            if floor is not None and s_lo < floor:
                s_lo = floor
            cell = cells[rows[i]]
            if code < 0:                              # corridor: -1 pad-only, -2 pad + blocked
                in_blk = code == -2
                for s in range(s_lo, s_hi + 1):       # `_drop` inlined, mirroring `add_volume`
                    d = pad_b[s]
                    n = d[cell] - 1
                    if n:
                        d[cell] = n
                    else:
                        del d[cell]
                        if not d:
                            del pad_b[s]
                    if in_blk:
                        d = blk_b[s]
                        n = d[cell] - 1
                        if n:
                            d[cell] = n
                        else:
                            del d[cell]
                            if not d:
                                del blk_b[s]
            else:                                     # shared terminal column
                tid = self._tids[code]
                for s in range(s_lo, s_hi + 1):
                    self._drop(self.term_cells, s, cell, tid)
        self.n_added -= len(volumes)

    def _on_static(self, center, term) -> None:
        """Derive this hub's discrete routing wall from a ledger static-terminal registration — the
        ``ReservationLedger.subscribe_static`` hook target (bound in ``AStarPlanner._occupancy``). Records
        the whole terminal airspace (column + exit lanes) in ``static_term_cells`` keyed by ``(q, r)``:
        step- and level-independent (the column is the [ground, ceiling] tube), so :meth:`is_blocked` walls
        it at every step and every flight level while the hub's own flights pass through (own-hub
        exemption). Idempotent per hub (set-based). The *authoritative* wall is the ledger's permanent
        volume (seen by ``any_conflict``/verify); this is the derived view A* routes around proactively."""
        tid = as_terminal(term).id
        for cell in hg.terminal_cells(center, term, self.cfg):
            self.static_term_cells.setdefault(cell, set()).add(tid)

    def evict_before(self, step: int) -> None:
        """Drop all cells at steps < ``step`` (cells the sim clock has passed; no future plan can
        query them). Monotonic — calls with an earlier ``step`` are no-ops."""
        if self.evicted_before is not None and step <= self.evicted_before:
            return
        for bucket in (self.blocked, self.pad, self.term_cells):
            for s in [s for s in bucket if s < step]:
                del bucket[s]
        self.evicted_before = step

    def reset(self) -> None:
        self.blocked.clear()
        self.pad.clear()
        self.term_cells.clear()
        self._rows.clear()
        # `_cells` / `_tids` are pure interning pools — value-identical across a rebuild and never
        # read except through a live row, so keeping them saves re-interning the same cells.
        self.n_added = 0
        self.evicted_before = None

    # ----- queries (the A* search hot path) -----
    def is_blocked(self, q: int, r: int, L: int, s: int, own: Collection[Hashable] = ()) -> bool:
        """Is hex (q, r) at flight level ``L`` an obstacle at step ``s``?

        A flight owns its vertiports: a cell inside its **own** terminal column is passable — the flight
        climbs/descends through its shared column. A cell under a *foreign* terminal column is a hard
        wall (cruise reroutes around busy vertiports). Otherwise it's an ordinary corridor obstacle for
        everyone.

        **Same-hub exit lanes (issue #18, ``fixed_exit_lanes``).** A hub's own-column footprint inflates
        ~99 m past the 90 m column, swallowing the exit-lane cells (120-205 m out). That own-column
        transparency is what lets a hub's flights share their climb space — but it also hid *committed
        sibling exit corridors* sitting in that footprint, so two same-hub launches into the same cruise
        corridor only collided at commit (``conflict_filed``). The bearing graze-set could not express
        that conflict (it conflated hub-bearing with cruise direction). So under the flag we do NOT
        blanket-transparent the footprint: an own-only column cell that also carries a committed corridor
        (a sibling's tagged exit lane, recorded in ``blocked`` outside the 90 m interior) still blocks —
        the exact same-hub cell occupancy. The flight's own (uncommitted) corridor is absent during its
        plan, so this never self-blocks; the 90 m interior is skipped from ``blocked`` (``add_volume``
        ``own_cols``), so the climb stays clear. Flag off ⇒ ``False`` here, i.e. unchanged."""
        # ``static_term_cells`` (always-active terminals) adds step- and level-independent foreign walls
        # (the [ground, ceiling] tube), merged with the per-step/level ``term_cells`` so a cell covered by
        # EITHER (foreign) blocks. Both empty ⇒ unchanged (zero overhead).
        if self.term_cells or self.static_term_cells:      # zero-overhead when no terminals exist
            here = self.term_cells.get(s, _EMPTY).get((q, r, L))
            stat = self.static_term_cells.get((q, r)) if self.static_term_cells else None
            if here is not None or stat is not None:
                if (here is not None and any(tid not in own for tid in here)) or \
                        (stat is not None and any(tid not in own for tid in stat)):
                    return True                  # foreign column (transient or always-active) → wall
                # own-only column: transparent for the climb, unless (fixed lanes) a committed sibling
                # corridor occupies this footprint cell at this level — the same-hub serialisation A* sees.
                return self.cfg.fixed_exit_lanes and (q, r, L) in self.blocked.get(s, ())
        return (q, r, L) in self.blocked.get(s, ())

    def pad_clear(self, q: int, r: int, s0: int, dwell_steps: int) -> bool:
        """Is the ordinary (non-terminal) pad at hex (q, r) free for the whole dwell window
        [s0, s0 + dwell_steps]? The takeoff/landing hover column spans the full tube [ground, ceiling],
        so the pad is clear iff NO committed corridor sweeps its cell at ANY flight level AND it does not
        sit under any hub's shared column. Shared-terminal dwells are gated *temporally* by
        :class:`~freespace_sim.planner.terminal_capacity.TerminalCapacity`, not here. (One level ⇒ the
        legacy single-cell check.)"""
        # Build the (q, r, L) column once and hoist the dict handles out of the window loop: n_levels was a
        # per-step @property hit (the profile's 1.18M-call line) and (q, r, L) was rebuilt per step*level.
        # Same k-major, level-ascending, pad-before-term short-circuit ⇒ byte-identical result.
        cells = [(q, r, L) for L in range(self.cfg.n_levels)]
        pad = self.pad
        term_cells = self.term_cells
        for k in range(s0, s0 + dwell_steps + 1):
            padk = pad.get(k, ())
            tck = term_cells.get(k, _EMPTY) if term_cells else _EMPTY
            for cell in cells:
                if cell in padk:
                    return False                 # a committed corridor sweeps the pad at level L
                if cell in tck:
                    return False                 # an ordinary pad sitting under some hub's column
        return True
