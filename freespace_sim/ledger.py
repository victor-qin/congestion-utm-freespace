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
        c = self._cell
        for cx in range(int(aabb[0] // c), int(aabb[3] // c) + 1):
            for cy in range(int(aabb[1] // c), int(aabb[4] // c) + 1):
                yield (cx, cy)

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

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._vols: list[Volume4D] = []
        self._fids: list[int] = []
        self._aabb: list[tuple[float, float, float, float, float, float]] = []  # flat per-volume AABB
        self._buckets: dict[int, list[int]] = {}
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
        self._static_subs: list = []           # static-terminal subscribers (occupancy routing-wall hook)

    def subscribe(self, callback) -> None:
        """Register ``callback(flight_id, volumes)``, fired after each successful commit — the
        publish hook (ASTM F3548-21 Subscription/notification analogue). Used by the A* planner's
        incremental hex-occupancy service to absorb new volumes without rebuilding from scratch."""
        self._observers.append(callback)

    def register_static_terminal(self, center, term) -> None:
        """File a hub's always-active terminal airspace as a PERMANENT ledger volume (whole horizon), so
        ``any_conflict`` / ``verify`` / the ledger-only refiners see it — instead of an off-ledger
        occupancy side-structure. NOT bucketed (time-invariant); scanned separately in
        ``conflicts``/``any_conflict``. Fires the static-subscribe hook so the A* occupancy services derive
        their (discrete) routing walls from the same source. Called once per hub at sim setup."""
        self._static_terms.append((center, term))
        vol = permanent_terminal_reservation(center, term, self.cfg)
        self._static_vols.append(vol)
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

    def _candidate_indices(self, vol: Volume4D) -> set[int]:
        seen: set[int] = set()
        for s in self._steps(vol):
            seen.update(self._buckets.get(s, ()))
        return seen

    @staticmethod
    def _flat_aabb(vol: Volume4D) -> tuple[float, float, float, float, float, float]:
        """A volume's world AABB as six plain floats ``(xmin, ymin, zmin, xmax, ymax, zmax)``.

        Flattening once (here, at commit/query time) lets the per-pair overlap prune below run as
        scalar comparisons. ``np.any`` on the 3-vector form costs ~34x more PER CALL — the arrays are
        length 3, so the work is dwarfed by numpy's dispatch/alloc/box overhead — and ``_aabb_miss``
        is the ledger's single hottest line (tens of millions of calls per run)."""
        lo, hi = vol.aabb()
        return (float(lo[0]), float(lo[1]), float(lo[2]),
                float(hi[0]), float(hi[1]), float(hi[2]))

    @staticmethod
    def _aabb_miss(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
        """True iff the two flat AABBs are separated on some axis (so they cannot intersect). Scalar
        equivalent of ``np.any(amax < bmin) or np.any(bmax < amin)`` — see :meth:`_flat_aabb`."""
        return (a[3] < b[0] or b[3] < a[0]      # x: amax < bmin or bmax < amin
                or a[4] < b[1] or b[4] < a[1]   # y
                or a[5] < b[2] or b[5] < a[2])  # z

    # ----- writes -----
    def commit(self, flight_id: int, volumes: list[Volume4D]) -> None:
        """Add a flight's volumes to the ledger (FCFS: it becomes an obstacle for later flights)."""
        for v in volumes:
            idx = len(self._vols)
            self._vols.append(v)
            self._fids.append(flight_id)
            self._aabb.append(self._flat_aabb(v))
            for s in self._steps(v):
                self._buckets.setdefault(s, []).append(idx)
        for cb in self._observers:           # publish hook: notify subscribers of the new volumes
            cb(flight_id, volumes)

    def release(self, flight_id: int) -> None:
        """Remove a flight (operator-initiated replanning). Rare in v0; rebuilds the index."""
        keep = [(f, v) for f, v in zip(self._fids, self._vols) if f != flight_id]
        self._vols, self._fids, self._aabb, self._buckets = [], [], [], {}
        for f, v in keep:
            self.commit(f, [v])

    # ----- reads -----
    def conflicts(self, volumes: list[Volume4D]) -> list[tuple[int, Volume4D]]:
        """Every committed (flight_id, volume) that conflicts with any of ``volumes``. A permanent
        always-active terminal wall has no flight id — it surfaces as ``(-1, static_vol)``, the documented
        sentinel (callers treat ``-1`` as 'static wall', never a real flight id)."""
        out: list[tuple[int, Volume4D]] = []
        for v in volumes:
            vbb = self._flat_aabb(v)
            for idx in self._candidate_indices(v):
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
            for idx in self._candidate_indices(v):
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
        """Yield every committed (flight_id, Volume4D) — used by verify and viz."""
        yield from zip(self._fids, self._vols)

    @property
    def n_volumes(self) -> int:
        return len(self._vols)
