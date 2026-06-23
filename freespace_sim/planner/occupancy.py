"""Incremental hex-occupancy service for the space-time A* planner.

A*'s search needs two cell maps derived from the committed volumes: ``blocked`` (the corridor
footprint a flight must avoid) and ``pad`` (the wider hover-cylinder footprint used for the
takeoff/landing dwell check). These maps are **global and flight-independent** — a cell's
membership depends only on the committed volumes and ``cfg``, never on who is planning — so rather
than rebuild them from scratch every plan (O(committed) per plan → O(N²) per run), this service
maintains them incrementally: each committed volume is rasterized **exactly once** (the dual sweep
in :func:`hexgrid.rasterize_volume_dual`) when the ledger publishes its commit, and cells older than
the request clock are evicted so memory stays bounded to the active time window.

**Shared terminal columns (Phase B).** A committed *terminal column* (``vol.terminal_id is not
None``) is NOT an ordinary obstacle — it's a multi-pad vertiport that admits up to ``capacity``
concurrent dwells of *its own* flights while still walling off everyone else. Such volumes are kept
out of the binary ``blocked``/``pad`` sets and instead counted in ``term_cells`` (``step -> cell ->
{terminal_id: dwell_count}``). The A* queries then:
  * **own-hub exemption** — :meth:`is_blocked` ignores a cell occupied *only* by the flight's own
    terminal(s), so a hub's flights fly through their shared column; a *foreign* hub's column still
    blocks (cruise reroutes around busy vertiports);
  * **capacity gate** — :meth:`pad_clear` admits a takeoff/landing dwell iff fewer than ``capacity``
    same-hub dwells already cover the window. **Capacity 1 ⟺ the old binary check** (count 0).
A run with no terminals never touches ``term_cells`` (it stays empty → zero overhead), so the maps
are byte-identical to before — the property the occupancy tests pin.

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
from ..volumes import Volume4D

_EMPTY: dict = {}


class HexOccupancyService:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.R = hg.circumradius(cfg)
        self.infl_blocked = cfg.corridor_width_m / 2.0 + self.R   # corridor footprint
        self.infl_pad = cfg.effective_hover_radius_m + self.R     # wider hover-cylinder footprint
        self.blocked: dict[int, set[tuple[int, int]]] = {}        # step -> {(q, r)}  (non-terminal)
        self.pad: dict[int, set[tuple[int, int]]] = {}            # step -> {(q, r)}  (non-terminal)
        # shared terminal columns: step -> cell -> {terminal_id: concurrent-dwell count}
        self.term_cells: dict[int, dict[tuple[int, int], dict[Hashable, int]]] = {}
        self.n_added = 0                  # committed volumes absorbed (shrink tripwire)
        self.evicted_before: int | None = None   # lowest retained step

    # ----- maintenance -----
    def add_volume(self, vol: Volume4D, own_cols: tuple = ()) -> None:
        """Rasterize one committed volume (once). Ordinary corridor cells feed the binary blocked/pad
        step-buckets; a shared terminal column instead increments its per-hub dwell counter.

        ``own_cols`` is the committing flight's own terminal columns ``(cx, cy, radius)``. A corridor
        cell falling INSIDE one of them is the vertiport's unreserved tactical interior (the flight's
        exit lane proper lies outside the column and is still recorded), so it's skipped — leaving only
        *foreign* corridors inside any hub's column for a launch to detect and wait out (see pad_clear).
        """
        tid = vol.terminal_id
        # Only a tagged *column* (hover cylinder) feeds the per-hub dwell counter; a tagged *corridor*
        # box (an in-terminal exit lane) is still a corridor — it goes to blocked/pad like any other,
        # so it is never miscounted as a pad dwell. ("column ⟺ cylinder"; stored kind is issue #11.)
        is_column = tid is not None and isinstance(vol.shape, CylinderSpec)
        for q, r, s, in_blk in hg.rasterize_volume_dual(
            vol, self.cfg, self.R, self.infl_blocked, self.infl_pad
        ):
            if self.evicted_before is not None and s < self.evicted_before:
                continue                 # guard: never resurrect an already-evicted past step
            if not is_column:
                if own_cols and self._inside_a_column(q, r, own_cols):
                    continue             # the committing flight's own terminal interior — unreserved
                self.pad.setdefault(s, set()).add((q, r))
                if in_blk:
                    self.blocked.setdefault(s, set()).add((q, r))
            elif in_blk:
                # shared terminal column: count one dwell of `tid` over its inner (blocked-strength)
                # footprint — the cells A* queries for takeoff/landing and own-hub exemption.
                cell = self.term_cells.setdefault(s, {}).setdefault((q, r), {})
                cell[tid] = cell.get(tid, 0) + 1
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
    def is_blocked(self, q: int, r: int, s: int, own: Collection[Hashable] = ()) -> bool:
        """Is hex (q, r) an obstacle at step ``s``?

        A flight owns its vertiports: a cell inside its **own** terminal column is passable even if a
        (strict) corridor's inflated footprint spills onto it — the flight climbs/descends through its
        shared column, and any real corridor-vs-corridor overlap is caught by the ledger, not the
        search grid. A cell under a *foreign* terminal column is a hard wall (cruise reroutes around
        busy vertiports). Otherwise it's an ordinary corridor obstacle for everyone."""
        if self.term_cells:                      # zero-overhead when no terminals exist
            here = self.term_cells.get(s, _EMPTY).get((q, r))
            if here is not None:
                # own column → transparent (overrides any corridor spillover); foreign → wall
                return any(tid not in own for tid in here)
        return (q, r) in self.blocked.get(s, ())

    def pad_clear(self, q: int, r: int, s0: int, dwell_steps: int,
                  terminal_id: Hashable = None, capacity: int = 1, footprint: tuple = None) -> bool:
        """Is the pad at hex (q, r) free for the whole dwell window [s0, s0 + dwell_steps]?

        An ordinary pad (``terminal_id is None``) is exclusive: clear iff no committed footprint touches
        it (capacity 1). A shared vertiport pad is gated by (a) **capacity** at the hub-centre cell and
        (b) the rule that *no foreign corridor occupies the column*: ``footprint`` (the column's hex
        cells) is scanned against the binary ``pad`` map — which inside a hub's column holds only foreign
        corridors, since a flight's own exit-lane interior is dropped on commit. So a launch ground-delays
        while a corridor transits its vertiport, instead of taking off and being denied by the ledger.
        """
        cells = footprint if footprint is not None else ((q, r),)
        for k in range(s0, s0 + dwell_steps + 1):
            padk = self.pad.get(k, ())
            for cell in cells:
                if cell in padk:
                    return False                 # a foreign corridor (or ordinary pad) sweeps the column
            here = self.term_cells.get(k, _EMPTY).get((q, r)) if self.term_cells else None
            if terminal_id is None:
                if here is not None:
                    return False                 # an ordinary pad sitting under some hub's column
            elif here is not None:
                if here.get(terminal_id, 0) >= capacity:
                    return False                 # all N pads busy this step → take ground delay
                if any(tid != terminal_id for tid in here):
                    return False                 # another hub's column overlaps this pad
        return True
