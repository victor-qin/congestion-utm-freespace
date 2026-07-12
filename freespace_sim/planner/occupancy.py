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

from collections.abc import Collection, Hashable

from . import hexgrid as hg
from ..config import SimConfig
from ..geometry import CylinderSpec
from ..types import as_terminal
from ..volumes import Volume4D

_EMPTY: dict = {}


class HexOccupancyService:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
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
    def add_volume(self, vol: Volume4D, own_cols: tuple = ()) -> None:
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
        for q, r, L, s, in_blk in hg.rasterize_volume_dual(
            vol, self.cfg, self.R, self.infl_blocked, self.infl_pad
        ):
            if self.evicted_before is not None and s < self.evicted_before:
                continue                 # guard: never resurrect an already-evicted past step
            if not is_column:
                if own_cols and self._inside_a_column(q, r, own_cols):
                    continue             # the committing flight's own terminal interior — unreserved
                self.pad.setdefault(s, set()).add((q, r, L))
                if in_blk:
                    self.blocked.setdefault(s, set()).add((q, r, L))
            elif in_blk:
                # shared terminal column: record `tid` over its inner (blocked-strength) footprint at
                # level L — the cells A* queries for the own-hub cruise exemption (capacity lives in
                # TerminalCapacity).
                self.term_cells.setdefault(s, {}).setdefault((q, r, L), set()).add(tid)
        self.n_added += 1

    def _inside_a_column(self, q: int, r: int, cols: tuple) -> bool:
        c = hg.hex_center(q, r, self.R)
        return any((c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= rad * rad for cx, cy, rad in cols)

    def on_commit(self, _flight_id, volumes) -> None:
        """Ledger commit subscriber (the publish hook): absorb a newly committed flight's volumes,
        dropping the corridor cells inside its own terminal columns (the unreserved tactical interior)."""
        own_cols = tuple((v.shape.cx, v.shape.cy, v.shape.radius) for v in volumes
                         if v.terminal_id is not None and isinstance(v.shape, CylinderSpec))
        for v in volumes:
            self.add_volume(v, own_cols=own_cols)

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
        for k in range(s0, s0 + dwell_steps + 1):
            padk = self.pad.get(k, ())
            tck = self.term_cells.get(k, _EMPTY) if self.term_cells else _EMPTY
            for L in range(self.cfg.n_levels):
                if (q, r, L) in padk:
                    return False                 # a committed corridor sweeps the pad at level L
                if (q, r, L) in tck:
                    return False                 # an ordinary pad sitting under some hub's column
        return True
