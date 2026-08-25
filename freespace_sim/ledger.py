"""ReservationLedger — the committed airspace, with a fast conflict query.

Holds every committed flight's `Volume4D`s and answers "does this candidate intent conflict with
anything already committed?" To stay fast for thousands of flights it prunes in two cheap stages
before the exact FCL test:

1. **Time bucketing** by discrete step — only volumes sharing a timestep are candidates.
2. **AABB overlap** — reject candidates whose world bounding boxes miss in any axis.

Survivors get the exact `volumes_conflict` (time + FCL narrowphase). Under FCFS, earlier-committed
intents are obstacles the newcomer must avoid; the ledger never mutates them.
"""

from __future__ import annotations

import numpy as np

from .config import SimConfig
from .conflict import volumes_conflict
from .volumes import Volume4D, permanent_terminal_reservation

_STATIC_GRID_CELL_M = 1024.0   # coarse xy-cell edge for the always-active-wall spatial index (_StaticWallGrid)
_DYNAMIC_GRID_CELL_M = 1024.0  # coarse xy-cell edge for the per-step committed-volume sub-index (commit/_candidate_indices)


def _xy_cell_span(aabb, cell):
    """The (cx, cy) xy-grid cells a flat AABB ``(xmin, ymin, zmin, xmax, ymax, zmax)`` touches.

    Shared by the always-active-wall grid (:class:`_StaticWallGrid`) and the per-step committed-volume xy
    sub-index (:meth:`ReservationLedger.commit` / :meth:`ReservationLedger._candidate_indices`). Floor-division
    so the slightly-negative coords a boundary corridor box can produce map to the correct (negative) cell — and
    so a box is indexed in, and a query visits, EVERY cell its xy-AABB overlaps. That is the no-false-negative
    property both indexes rely on: if two xy-AABBs overlap they cannot be cell-separated, so they share a cell."""
    for cx in range(int(aabb[0] // cell), int(aabb[3] // cell) + 1):
        for cy in range(int(aabb[1] // cell), int(aabb[4] // cell) + 1):
            yield (cx, cy)


class _StaticWallGrid:
    """Coarse uniform xy-grid over the always-active terminal walls — the SPATIAL analogue of the ledger's
    per-step time buckets, for obstacles that are fixed in space and permanent in time (so neither a time
    bucket nor a rebuild helps them). Built once as hubs register; a query returns only the walls whose
    xy-cell the query box touches, pruning the wall scan from O(all hubs) to O(hubs near the box).

    Exact — no false negatives: a wall is indexed in every cell its xy-AABB overlaps and a query visits
    every cell its xy-AABB overlaps, so any (wall, query) pair whose AABBs overlap in xy necessarily shares
    a cell. The candidate set is therefore a superset of everything the downstream ``_aabb_miss`` would keep,
    so the conflict result is byte-identical to the old full linear scan.
    """

    __slots__ = ("_cell", "_cells")

    def __init__(self, cell: float):
        self._cell = cell
        self._cells: dict[tuple[int, int], list[int]] = {}

    def _span(self, aabb):        # (xmin,ymin,zmin,xmax,ymax,zmax) → the xy-cells the box touches
        yield from _xy_cell_span(aabb, self._cell)

    def insert(self, idx: int, aabb) -> None:
        for key in self._span(aabb):
            self._cells.setdefault(key, []).append(idx)

    def candidates(self, aabb) -> list[int]:
        if not self._cells:
            return []
        hit: set[int] = set()
        for key in self._span(aabb):
            hit.update(self._cells.get(key, ()))
        return sorted(hit)        # ascending index order == the old enumerate order (stable conflicts() output)


class ReservationLedger:
    # Partner-id sentinel reported by `conflicts` for an always-active terminal WALL (a permanent
    # `_static_vols` entry owns no flight). Callers treat it as "static wall", never a real flight id.
    STATIC_WALL_FID = -1
    # Owner sentinel for a tombstoned (released-in-place) volume — see release_many. Never yielded by
    # iter_committed, never a candidate in conflicts (its AABB is the empty box below).
    TOMBSTONE_FID = -2

    # Empty AABB carried by tombstoned volumes: min > max on every axis, so `_aabb_miss` rejects it
    # against ANY query box and every AABB-pruned read path (conflicts/any_conflict/column_clear) skips
    # the dead entry without knowing tombstones exist.
    _DEAD_AABB = (np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf)

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        # `_vols` KEEPS a tombstoned volume's object (only its `_fids` owner and `_aabb` box are
        # overwritten — see release_many), so a raw `zip(_fids, _vols)` walk sees dead geometry as if it
        # were committed. Every read path either prunes by AABB or filters the owner; iterate with
        # `iter_committed()`, never over these lists directly.
        self._vols: list[Volume4D] = []
        self._fids: list[int] = []
        self._aabb: list[tuple[float, float, float, float, float, float]] = []  # flat per-volume AABB
        self._n_dead = 0                     # tombstoned entries in _vols (release_many); compacted lazily
        self._release_subs: list = []        # release_many subscribers (removal publish hook)
        # committed-volume index keyed by (step, cell_x, cell_y): a TIME bucket (discrete step) CROSSED with an
        # xy SPATIAL sub-index, so a query scans only volumes sharing its timestep AND near its xy — not every
        # volume metro-wide that merely shares the step (issue #30). See commit / _candidate_indices.
        self._buckets: dict[tuple[int, int, int], list[int]] = {}
        self._observers: list = []   # commit subscribers (publish hook); see subscribe()
        # Always-active terminal walls (cfg.terminal_airspace_always_active): PERMANENT, whole-horizon
        # reservations filed once at sim setup. Kept OUT of the per-step _buckets (time-invariant — bucketing
        # a whole-horizon volume would flood every step); instead they get their own xy spatial index
        # (_static_grid) and are scanned grid-pruned in conflicts/any_conflict. The (center, term) pairs feed
        # the A* occupancy-derivation hook (subscribe_static). Empty unless register_static_terminal is
        # called ⇒ zero overhead when the flag is off.
        self._static_vols: list[Volume4D] = []
        self._static_aabb: list[tuple[float, float, float, float, float, float]] = []  # per-wall flat AABB
        self._static_grid = _StaticWallGrid(_STATIC_GRID_CELL_M)   # xy prune over the (fixed) hub walls
        self._static_terms: list[tuple] = []   # (center, term) pairs, for the occupancy-derivation replay
        self._static_terminal_ids: set = set()  # O(1) proof that a queried hub has its permanent wall
        self._static_subs: list = []           # static-terminal subscribers (occupancy routing-wall hook)
        # Bumped by `detach_subscribers`. A service binds itself to (ledger, epoch); a mismatch means
        # commits happened that it could not observe, so its incremental state is unrecoverable and it
        # must REBIND (re-subscribe + rebuild), not merely rebuild. See detach_subscribers.
        self._epoch = 0

    @property
    def epoch(self) -> int:
        """Subscription generation — see :meth:`detach_subscribers`. Services store it at bind time and
        rebind when it changes; a ledger that never changes hands stays at 0 (zero overhead)."""
        return self._epoch

    def detach_subscribers(self) -> None:
        """Drop EVERY subscriber — commit, release and static — and bump :attr:`epoch`. The ownership
        transfer a new solver performs when it takes over a completed run's ledger (LNS
        destroy/repair), and the teardown it performs when it hands the ledger back.

        Clearing alone is not enough: a planner whose services were subscribed still holds
        ``_svc_ledger is ledger``, so it would neither re-subscribe nor rebuild, and would plan against
        an occupancy frozen at the moment of the takeover. The shrink tripwire (``n_volumes <
        n_added``) cannot catch that — a release/re-commit pair nets to the same count. The epoch does,
        deterministically.

        ``_static_subs`` goes too, and for the same reason the other two do. It holds BOUND METHODS of
        the detached services, so keeping it pins the very objects the transfer exists to release (a
        measured 8.4 MB of pool arrays on a 6 km single-level box, and the full occupancy image on a
        density scenario) and every later rebind appends another pair without removing the dead one.
        Nothing is lost by dropping them: ``subscribe_static`` REPLAYS every registered hub to each new
        subscriber, which is exactly why re-deriving the walls after a takeover is free.

        Deliberately returns nothing. An earlier version handed back the removed callbacks "so a caller
        can restore them", which cannot work: ``_epoch`` only ever increments, so re-subscribed
        services rebind and discard their state on the next ``plan()``. Re-binding is the supported
        way back, not restoration."""
        self._observers.clear()
        self._release_subs.clear()
        self._static_subs.clear()
        self._epoch += 1

    def subscribe(self, callback) -> None:
        """Register ``callback(flight_id, volumes)``, fired after each successful commit — the
        publish hook (ASTM F3548-21 Subscription/notification analogue). Used by the A* planner's
        incremental hex-occupancy service to absorb new volumes without rebuilding from scratch."""
        self._observers.append(callback)

    def subscribe_release(self, callback) -> None:
        """Register ``callback(flight_id, volumes)``, fired by ``release_many`` for each removed
        flight — the removal analogue of ``subscribe``. Services that track per-owner rows use it
        to un-absorb a flight in O(its volumes) and keep ``n_added`` in lockstep with
        ``n_volumes``, so the shrink tripwire (their safety net) stays silent. Legacy ``release``
        predates this hook and delegates to ``release_many`` whenever such a subscriber exists — its
        own rebuild re-feeds commit observers volume-by-volume, which would desync per-owner rows."""
        self._release_subs.append(callback)

    def register_static_terminal(self, center, term) -> None:
        """File a hub's always-active terminal airspace as a PERMANENT ledger volume (whole horizon), so
        ``any_conflict`` / ``verify`` / the ledger-only refiners see it — instead of an off-ledger
        occupancy side-structure. NOT bucketed (time-invariant); scanned separately in
        ``conflicts``/``any_conflict``. Fires the static-subscribe hook so the A* occupancy services derive
        their (discrete) routing walls from the same source. Called once per hub at sim setup."""
        self._static_terms.append((center, term))
        vol = permanent_terminal_reservation(center, term, self.cfg)
        self._static_vols.append(vol)
        self._static_terminal_ids.add(vol.terminal_id)
        self._static_aabb.append(self._flat_aabb(vol))
        self._static_grid.insert(len(self._static_vols) - 1, self._static_aabb[-1])
        for cb in self._static_subs:
            cb(center, term)

    def subscribe_static(self, callback) -> None:
        """Register ``callback(center, term)`` for static-terminal registrations — and REPLAY it
        immediately for every already-registered hub. The replay is essential (unlike ``subscribe``): the
        A* occupancy services bind lazily on their first plan, i.e. AFTER ``sim.run`` has already registered
        every hub, so a subscribe-only hook would miss them all and the routing walls would be empty."""
        self._static_subs.append(callback)
        for center, term in self._static_terms:
            callback(center, term)

    # ----- internals -----
    def _steps(self, vol: Volume4D) -> range:
        s0 = int(np.floor(vol.t_start / self.cfg.dt_s))
        s1 = int(np.floor(vol.t_end / self.cfg.dt_s))
        return range(s0, s1 + 1)

    def _candidate_indices(self, vol: Volume4D, vbb: tuple[float, ...]) -> set[int]:
        """Committed volumes sharing BOTH a timestep and an xy-cell with the query box ``vol`` (flat AABB
        ``vbb``). A superset of the true 3D overlaps — a real conflict needs xy-AABB overlap AND time overlap,
        so it must share a (step, cell) key — which ``_aabb_miss`` + ``volumes_conflict`` then filter exactly."""
        seen: set[int] = set()
        cells = list(_xy_cell_span(vbb, _DYNAMIC_GRID_CELL_M))
        buckets = self._buckets              # hoist the dict handle out of the step*cell loop
        for s in self._steps(vol):
            for cx, cy in cells:             # same FULL cross product as commit (a diagonal would miss conflicts)
                seen.update(buckets.get((s, cx, cy), ()))
        return seen

    @staticmethod
    def _flat_aabb(vol: Volume4D) -> tuple[float, float, float, float, float, float]:
        """A volume's world AABB as six plain floats ``(xmin, ymin, zmin, xmax, ymax, zmax)``.

        Delegates to ``vol.flat_aabb()``, which the shape computes directly as scalars — no ``np.array``
        allocation and no ``float(...)`` unpack (the old ``vol.aabb()`` built two length-3 arrays here just
        to read six floats back out; it was the profile's #1 self-time line via this path). Flattening lets
        the per-pair overlap prune below run as scalar comparisons; ``_aabb_miss`` is the ledger's single
        hottest line (tens of millions of calls per run). Bit-for-bit identical to the prior
        ``float(vol.aabb()[...])`` — pinned in ``tests/test_geometry.py``."""
        return vol.flat_aabb()

    @staticmethod
    def _aabb_miss(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        """True iff the two flat AABBs are separated on some axis (so they cannot intersect). Scalar
        equivalent of ``np.any(amax < bmin) or np.any(bmax < amin)`` — see :meth:`_flat_aabb`."""
        return (a[3] < b[0] or b[3] < a[0]      # x: amax < bmin or bmax < amin
                or a[4] < b[1] or b[4] < a[1]   # y
                or a[5] < b[2] or b[5] < a[2])  # z

    # ----- writes -----
    def _append(self, flight_id: int, v: Volume4D) -> None:
        """Insert one volume into the arrays and the (step, cell) buckets — the commit loop body,
        shared with `_compact` (which must NOT re-fire observers)."""
        idx = len(self._vols)
        self._vols.append(v)
        self._fids.append(flight_id)
        vbb = self._flat_aabb(v)
        self._aabb.append(vbb)
        cells = list(_xy_cell_span(vbb, _DYNAMIC_GRID_CELL_M))   # xy-cells this box touches (enumerate once)
        for s in self._steps(v):
            for key in cells:            # FULL cross product steps × cells — a diagonal pairing would miss conflicts
                self._buckets.setdefault((s, *key), []).append(idx)

    def commit(self, flight_id: int, volumes: list[Volume4D]) -> None:
        """Add a flight's volumes to the ledger (FCFS: it becomes an obstacle for later flights)."""
        for v in volumes:
            self._append(flight_id, v)
        for cb in self._observers:           # publish hook: notify subscribers of the new volumes
            cb(flight_id, volumes)

    def release(self, flight_id: int) -> None:
        """Remove a flight (operator-initiated replanning). Rare in v0; rebuilds the index.

        The rebuild re-commits every surviving volume, which re-feeds commit observers one volume at a
        time — how a commit-only service (no removal hook) stays in sync. That re-feed would desync a
        service that DOES track per-owner rows, so once release subscribers exist this delegates to
        ``release_many``, whose removal publish is exact for them (and whose shrink leaves the
        commit-only services' tripwire to heal them). Same live content either way."""
        if self._release_subs:
            self.release_many([flight_id])
            return
        keep = [(f, v) for f, v in zip(self._fids, self._vols)
                if f != flight_id and f != self.TOMBSTONE_FID]
        self._vols, self._fids, self._aabb, self._buckets = [], [], [], {}
        self._n_dead = 0
        for f, v in keep:
            self.commit(f, [v])

    def release_many(self, flight_ids) -> int:
        """Remove several flights by tombstoning their volumes in place — the LNS destroy primitive.

        O(current volumes) flag pass, no bucket rebuild, and — unlike ``release`` — **no observer
        re-feed**: the planners' incremental occupancy services notice the shrink themselves
        (``ledger.n_volumes < svc.n_added``) on their next ``plan()`` and rebuild from
        ``iter_committed``. A tombstone keeps its slot in ``_vols``/``_buckets`` but its AABB becomes
        the empty box, so every AABB-pruned read (``conflicts``/``any_conflict``/the terminal-capacity
        column scan) skips it; ``iter_committed`` skips it by owner id. Arrays are compacted once dead
        entries outnumber live ones. Returns the number of volumes tombstoned."""
        doomed = set(flight_ids)
        removed: dict[int, list[Volume4D]] = {}
        n = 0
        for i, f in enumerate(self._fids):
            if f in doomed:
                self._fids[i] = self.TOMBSTONE_FID
                self._aabb[i] = self._DEAD_AABB
                if self._release_subs:
                    removed.setdefault(f, []).append(self._vols[i])
                n += 1
        self._n_dead += n
        if self._n_dead > len(self._vols) - self._n_dead:
            self._compact()
        for fid, vols in removed.items():   # removal publish hook (one call per flight, like commit)
            for cb in self._release_subs:
                cb(fid, vols)
        return n

    def _compact(self) -> None:
        """Drop tombstoned entries and rebuild the buckets. Silent to observers by design: a
        compaction changes indices, not content, and the services never hold ledger indices."""
        keep = [(f, v) for f, v in zip(self._fids, self._vols) if f != self.TOMBSTONE_FID]
        self._vols, self._fids, self._aabb, self._buckets = [], [], [], {}
        self._n_dead = 0
        for f, v in keep:
            self._append(f, v)

    # ----- reads -----
    def conflicts(self, volumes: list[Volume4D]) -> list[tuple[int, Volume4D]]:
        """Every committed (flight_id, volume) that conflicts with any of ``volumes``. A permanent
        always-active terminal wall has no flight id — it surfaces as ``(-1, static_vol)``, the documented
        sentinel (callers treat ``-1`` as 'static wall', never a real flight id)."""
        out: list[tuple[int, Volume4D]] = []
        for v in volumes:
            vbb = self._flat_aabb(v)
            for idx in self._candidate_indices(v, vbb):
                if self._aabb_miss(vbb, self._aabb[idx]):
                    continue
                cv = self._vols[idx]
                if volumes_conflict(v, cv):
                    out.append((self._fids[idx], cv))
            for i in self._static_grid.candidates(vbb):       # always-active walls: xy-grid-pruned, time-invariant
                sv = self._static_vols[i]
                if self._aabb_miss(vbb, self._static_aabb[i]):
                    continue
                if volumes_conflict(v, sv):
                    out.append((self.STATIC_WALL_FID, sv))
        return out

    def any_conflict(self, volumes: list[Volume4D]) -> bool:
        """Fast feasibility check: True as soon as one committed volume — or an always-active terminal wall
        — conflicts (planner hot path)."""
        for v in volumes:
            vbb = self._flat_aabb(v)
            for idx in self._candidate_indices(v, vbb):
                if self._aabb_miss(vbb, self._aabb[idx]):
                    continue
                if volumes_conflict(v, self._vols[idx]):
                    return True
            for i in self._static_grid.candidates(vbb):       # always-active walls: xy-grid-pruned, time-invariant
                sv = self._static_vols[i]
                if self._aabb_miss(vbb, self._static_aabb[i]):
                    continue
                if volumes_conflict(v, sv):
                    return True
        return False

    def conflicting_flights(self, volumes: list[Volume4D]) -> set[int]:
        """The set of committed flight_ids that block ``volumes`` (for reroute targeting). Excludes the
        static-wall sentinel (a permanent terminal wall owns no flight and cannot be a reroute target)."""
        return {fid for fid, _ in self.conflicts(volumes) if fid != self.STATIC_WALL_FID}

    def iter_committed(self):
        """Yield every committed (flight_id, Volume4D) — used by verify and viz. Tombstoned entries
        (release_many) are skipped, so consumers see only live volumes, in commit order — each
        flight's volumes stay CONTIGUOUS (``_absorb`` groups by adjacent runs and relies on this)."""
        tomb = self.TOMBSTONE_FID
        yield from ((f, v) for f, v in zip(self._fids, self._vols) if f != tomb)

    def static_volumes(self) -> tuple:
        """The permanent always-active terminal walls (read-only view; empty unless registered)."""
        return tuple(self._static_vols)

    def static_terminals(self) -> tuple:
        """The ``(center, term)`` pairs actually registered as permanent walls — the record of what this
        ledger was BUILT with, for replaying the same world (``verify.find_interflight_conflict``'s
        ``static_terminals``, a re-derived unimpeded ledger). Re-deriving them from the demand model
        instead is wrong in both directions: it invents walls the run never filed when
        ``terminal_airspace_always_active`` is off, and it misses the scenario-collected fallback
        ``sim.run`` uses for demand models without ``terminals``. Empty unless registered."""
        return tuple(self._static_terms)

    def has_static_terminal(self, terminal_id) -> bool:
        """Whether ``terminal_id`` has a registered permanent ground-to-ceiling ledger wall.

        The always-active ``TerminalCapacity`` shortcut relies on this concrete registration, not merely
        on the config flag expressing that walls are intended. Terminal IDs uniquely identify fixed hubs,
        so membership is the O(1) proof that the queried column is backed by its permanent reservation.
        """
        return terminal_id in self._static_terminal_ids

    @property
    def n_volumes(self) -> int:
        """LIVE committed volumes (tombstones excluded) — the count the occupancy services compare
        against their `n_added` to detect a shrink, so it must drop the moment volumes are released."""
        return len(self._vols) - self._n_dead
