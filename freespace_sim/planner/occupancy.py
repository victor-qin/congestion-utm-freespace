"""Incremental hex-occupancy service for the space-time A* planner.

A*'s search needs two cell maps derived from the committed volumes: ``blocked`` (the corridor
footprint a flight must avoid) and ``pad`` (the wider hover-cylinder footprint used for the
takeoff/landing dwell check). These maps are **global and flight-independent** — a cell's
membership depends only on the committed volumes and ``cfg``, never on who is planning — so rather
than rebuild them from scratch every plan (O(committed) per plan → O(N²) per run), this service
maintains them incrementally: each committed volume is rasterized **exactly once** (the dual sweep
in :func:`hexgrid.rasterize_volume_dual`) when the ledger publishes its commit, and cells older than
the request clock are evicted so memory stays bounded to the active time window.

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

from . import hexgrid as hg
from ..config import SimConfig
from ..volumes import Volume4D


class HexOccupancyService:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.R = hg.circumradius(cfg)
        self.infl_blocked = cfg.corridor_width_m / 2.0 + self.R   # corridor footprint
        self.infl_pad = cfg.effective_hover_radius_m + self.R     # wider hover-cylinder footprint
        self.blocked: dict[int, set[tuple[int, int]]] = {}        # step -> {(q, r)}
        self.pad: dict[int, set[tuple[int, int]]] = {}            # step -> {(q, r)}  (superset)
        self.n_added = 0                  # committed volumes absorbed (shrink tripwire)
        self.evicted_before: int | None = None   # lowest retained step

    # ----- maintenance -----
    def add_volume(self, vol: Volume4D) -> None:
        """Rasterize one committed volume (once) into the blocked/pad step-buckets."""
        for q, r, s, in_blk in hg.rasterize_volume_dual(
            vol, self.cfg, self.R, self.infl_blocked, self.infl_pad
        ):
            if self.evicted_before is not None and s < self.evicted_before:
                continue                 # guard: never resurrect an already-evicted past step
            self.pad.setdefault(s, set()).add((q, r))
            if in_blk:
                self.blocked.setdefault(s, set()).add((q, r))
        self.n_added += 1

    def on_commit(self, _flight_id, volumes) -> None:
        """Ledger commit subscriber (the publish hook): absorb a newly committed flight's volumes."""
        for v in volumes:
            self.add_volume(v)

    def evict_before(self, step: int) -> None:
        """Drop all cells at steps < ``step`` (cells the sim clock has passed; no future plan can
        query them). Monotonic — calls with an earlier ``step`` are no-ops."""
        if self.evicted_before is not None and step <= self.evicted_before:
            return
        for bucket in (self.blocked, self.pad):
            for s in [s for s in bucket if s < step]:
                del bucket[s]
        self.evicted_before = step

    def reset(self) -> None:
        self.blocked.clear()
        self.pad.clear()
        self.n_added = 0
        self.evicted_before = None

    # ----- queries (the A* search hot path) -----
    def is_blocked(self, q: int, r: int, s: int) -> bool:
        return (q, r) in self.blocked.get(s, ())

    def pad_clear(self, q: int, r: int, s0: int, dwell_steps: int) -> bool:
        """Is the pad at hex (q, r) free for the whole dwell window [s0, s0 + dwell_steps]?"""
        return all((q, r) not in self.pad.get(k, ()) for k in range(s0, s0 + dwell_steps + 1))
