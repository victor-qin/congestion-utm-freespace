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
from .volumes import Volume4D


class ReservationLedger:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._vols: list[Volume4D] = []
        self._fids: list[int] = []
        self._aabb: list[tuple[np.ndarray, np.ndarray]] = []
        self._buckets: dict[int, list[int]] = {}

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
    def _aabb_miss(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> bool:
        amin, amax = a
        bmin, bmax = b
        return bool(np.any(amax < bmin) or np.any(bmax < amin))

    # ----- writes -----
    def commit(self, flight_id: int, volumes: list[Volume4D]) -> None:
        """Add a flight's volumes to the ledger (FCFS: it becomes an obstacle for later flights)."""
        for v in volumes:
            idx = len(self._vols)
            self._vols.append(v)
            self._fids.append(flight_id)
            self._aabb.append(v.aabb())
            for s in self._steps(v):
                self._buckets.setdefault(s, []).append(idx)

    def release(self, flight_id: int) -> None:
        """Remove a flight (operator-initiated replanning). Rare in v0; rebuilds the index."""
        keep = [(f, v) for f, v in zip(self._fids, self._vols) if f != flight_id]
        self._vols, self._fids, self._aabb, self._buckets = [], [], [], {}
        for f, v in keep:
            self.commit(f, [v])

    # ----- reads -----
    def conflicts(self, volumes: list[Volume4D]) -> list[tuple[int, Volume4D]]:
        """Every committed (flight_id, volume) that conflicts with any of ``volumes``."""
        out: list[tuple[int, Volume4D]] = []
        for v in volumes:
            vbb = v.aabb()
            for idx in self._candidate_indices(v):
                if self._aabb_miss(vbb, self._aabb[idx]):
                    continue
                cv = self._vols[idx]
                if volumes_conflict(v, cv):
                    out.append((self._fids[idx], cv))
        return out

    def any_conflict(self, volumes: list[Volume4D]) -> bool:
        """Fast feasibility check: True as soon as one committed volume conflicts (planner hot path)."""
        for v in volumes:
            vbb = v.aabb()
            for idx in self._candidate_indices(v):
                if self._aabb_miss(vbb, self._aabb[idx]):
                    continue
                if volumes_conflict(v, self._vols[idx]):
                    return True
        return False

    def conflicting_flights(self, volumes: list[Volume4D]) -> set[int]:
        """The set of committed flight_ids that block ``volumes`` (for reroute targeting)."""
        return {fid for fid, _ in self.conflicts(volumes)}

    def iter_committed(self):
        """Yield every committed (flight_id, Volume4D) — used by verify and viz."""
        yield from zip(self._fids, self._vols)

    @property
    def n_volumes(self) -> int:
        return len(self._vols)
