"""Mutable MAPF-LNS solution state over the reservation ledger.

Owns the incumbent schedule (fid -> OperationalIntent), the per-cell claim index the
destroy heuristics read (implements ``neighborhood.DestroyContext``), and the
destroy -> PP-repair -> accept/revert transaction against the shared
``ReservationLedger``.

Transaction contract (why this is exact — the seams are documented in
context/lns_plan.md):

* Destroy uses ``ledger.release_many`` — tombstones, no observer re-feed. The repair
  planner's occupancy services detect the shrink (``n_volumes < n_added``) on the
  first ``plan()`` of the iteration and rebuild themselves from ``iter_committed``,
  so every iteration starts from an exact occupancy of "everyone but the victims".
* The repair planner runs with ``evict_floor = 0.0`` (the Track A out-of-order
  dispatch knob), so the eviction watermark never advances and victims can be
  replanned in ANY priority order — random PP orderings stay exact.
* On reject/failure the new plans are tombstoned and the old volumes re-committed
  verbatim, one ``commit`` per flight (``_absorb`` needs each flight's volumes
  contiguous). The occupancy services are left stale-with-ghosts, which is safe:
  nothing reads them until the next iteration's first ``plan()``, whose shrink
  tripwire rebuilds them (every iteration begins with a ``release_many``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.sim import realized_release_s
from freespace_sim.types import OperationalIntent

log = logging.getLogger("freespace_sim.lns")

Cell = tuple[int, int, int]


@dataclass
class RepairOutcome:
    """What one destroy->repair transaction did."""

    accepted: bool
    reason: str  # "improved" | "no_improvement" | "denied" | "anchor"
    cost_old: float
    cost_new: float  # inf when the repair never produced a complete candidate
    n_planned: int

    @property
    def improvement(self) -> float:
        return self.cost_old - self.cost_new if self.accepted else 0.0


class LNSState:
    """Incumbent schedule + claim index + ledger transaction (DestroyContext impl)."""

    def __init__(
        self,
        cfg: SimConfig,
        ledger: ReservationLedger,
        intents: list[OperationalIntent],
        *,
        static_terms: tuple = (),
        frozen_flight_ids: frozenset[int] = frozenset(),
        movable_uss_ids: frozenset[str] | None = None,
        turnaround_s: float | None = None,
        repair_planner: AStarPlanner | None = None,
        incremental_release: bool = True,
    ) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.dt = cfg.dt_s
        self.n_levels = len(cfg.flight_levels_m)
        self.rng: np.random.Generator = np.random.default_rng(0)  # solver re-seeds per iteration

        self.order = [it.request.flight_id for it in intents]
        self.incumbent: dict[int, OperationalIntent] = {it.request.flight_id: it for it in intents}

        movable = [
            it.request.flight_id
            for it in intents
            if it.accepted
            and it.request.flight_id not in frozen_flight_ids
            and (movable_uss_ids is None or it.request.uss_id in movable_uss_ids)
        ]
        self._movable = sorted(movable)
        self._movable_set = set(self._movable)
        self.total_cost = float(sum(it.cost for it in intents if it.accepted))

        # LNS takes ownership of the ledger: the FCFS run's planner services stay subscribed
        # otherwise, silently absorbing (and retaining the memory of) every repair commit.
        ledger._observers.clear()
        ledger._release_subs.clear()

        if repair_planner is not None and (
            getattr(repair_planner, "_svc_ledger", None) is ledger
            or getattr(repair_planner, "_cocc_ledger", None) is ledger
        ):
            # Its subscriptions were just cleared; re-binding would miss repair commits (PP
            # would plan later victims blind to earlier victims' new claims).
            raise ValueError("repair_planner must not already be bound to this ledger")
        # incremental_release=True: the planner's occupancy/capacity services subscribe to
        # `release_many` and un-absorb victims in O(their volumes), so the per-iteration shrink
        # rebuild (measured 94% of iteration wall) never happens. False keeps the rebuild path
        # (the byte-parity reference for A/Bs).
        self.repair_planner = repair_planner or AStarPlanner(incremental_release=incremental_release)
        self.repair_planner.evict_floor = 0.0  # random repair orders need the full-horizon occupancy

        # Paired-return anchor guard (only when the baseline ran return_anchor="realized"):
        # outbound fid -> the committed return's desired departure. We never re-time returns, so
        # an outbound repair must keep its realized release <= anchor - turnaround.
        self._turnaround_s = turnaround_s
        self._return_anchor: dict[int, float] = {}
        if turnaround_s is not None:
            for it in intents:
                pid = it.request.paired_outbound_id
                if pid is not None and it.accepted:
                    self._return_anchor[pid] = float(it.request.t_departure)

        # Unimpeded weighted cost per movable flight — the paper's d(s_i, g_i) analogue, so
        # delay(fid) = incumbent cost - unimpeded cost. One plan per flight on a static-walls-only
        # ledger (its own planner instance: one planner per ledger).
        self._unimp_cost: dict[int, float] = {}
        free = ReservationLedger(cfg)
        for center, term in static_terms:
            free.register_static_terminal(center, term)
        planner_u = AStarPlanner()
        planner_u.evict_floor = 0.0
        for k, fid in enumerate(self._movable):
            u = planner_u.plan(self.incumbent[fid].request, free, cfg)
            if u.accepted:
                self._unimp_cost[fid] = float(u.cost)
            else:  # can't even place it alone (cap artifact): treat as undelayed, never seed a walk
                self._unimp_cost[fid] = float(self.incumbent[fid].cost)
                log.warning("lns: unimpeded plan denied for flight %d (%s)", fid, u.denial_reason)
            if (k + 1) % 1000 == 0:
                log.info("lns: unimpeded baseline %d/%d", k + 1, len(self._movable))

        # Claim index for the destroy heuristics: cell -> [(s_lo, s_hi, fid)] over the same
        # inflated corridor/pad raster A* deconflicts against (blocked rows only; the
        # terminal-tagged capacity cylinders are counting constraints, not cells).
        self._R = hg.circumradius(cfg)
        self._infl_b = cfg.corridor_width_m / 2.0 + self._R
        self._infl_p = cfg.effective_hover_radius_m + self._R
        self._claims: dict[Cell, list[tuple[int, int, int]]] = {}
        self._cells_of: dict[int, list[tuple[Cell, int, int]]] = {}
        self._contended: set[Cell] = set()
        self._contended_list: list[Cell] | None = None
        self._visits: dict[int, list[tuple[int, Cell]]] = {}
        for fid in self._movable:
            self._index_add(fid, self.incumbent[fid].volumes, refresh=False)
        for cell in self._claims:  # one contention sweep instead of a refresh per row
            self._refresh_contention(cell)

    # ------------------------------------------------------------------ claim index
    def _index_add(self, fid: int, volumes, refresh: bool = True) -> None:
        rows = self._cells_of.setdefault(fid, [])
        for v in volumes:
            if v.terminal_id is not None and isinstance(v.shape, CylinderSpec):
                continue  # capacity-gated own column, not a blocked cell
            for q, r, level, s_lo, s_hi, in_blk in hg.rasterize_ranges(
                v, self.cfg, self._R, self._infl_b, self._infl_p
            ):
                if not in_blk:
                    continue
                cell = (q, r, level)
                self._claims.setdefault(cell, []).append((s_lo, s_hi, fid))
                rows.append((cell, s_lo, s_hi))
                if refresh:
                    self._refresh_contention(cell)

    def _index_remove(self, fid: int) -> None:
        for cell, _s_lo, _s_hi in self._cells_of.pop(fid, ()):
            entries = self._claims.get(cell)
            if not entries:
                continue
            kept = [e for e in entries if e[2] != fid]
            if kept:
                self._claims[cell] = kept
            else:
                del self._claims[cell]
            self._refresh_contention(cell)

    def _refresh_contention(self, cell: Cell) -> None:
        entries = self._claims.get(cell, ())
        owners = {e[2] for e in entries}
        contended = len(owners) >= 2
        if contended != (cell in self._contended):
            if contended:
                self._contended.add(cell)
            else:
                self._contended.discard(cell)
            self._contended_list = None

    # ------------------------------------------------------------ DestroyContext API
    def movable_ids(self):
        return self._movable

    def is_movable(self, fid: int) -> bool:
        return fid in self._movable_set

    def delay(self, fid: int) -> float:
        return max(0.0, float(self.incumbent[fid].cost) - self._unimp_cost[fid])

    def visits(self, fid: int):
        cached = self._visits.get(fid)
        if cached is None:
            cached = self._visits[fid] = self._extract_visits(self.incumbent[fid])
        return cached

    def unimpeded_launch_step(self, fid: int) -> int:
        vis = self.visits(fid)
        if not vis:
            return 0
        hold = int(round(float(self.incumbent[fid].ground_delay_s) / self.dt))
        return vis[0][0] - hold

    def owners_over(self, cell: Cell, s_lo: int, s_hi: int):
        return {f for a, b, f in self._claims.get(cell, ()) if a <= s_hi and b >= s_lo}

    def claim_span(self, cell: Cell) -> tuple[int, int]:
        entries = self._claims[cell]
        return min(e[0] for e in entries), max(e[1] for e in entries)

    def contention_cells(self):
        if self._contended_list is None:
            self._contended_list = sorted(self._contended)
        return self._contended_list

    def _extract_visits(self, it: OperationalIntent) -> list[tuple[int, Cell]]:
        """Per-step (step, cell) samples of the centerline at a flight level — the airborne
        lateral path the random walk explores. Climb/descend samples (between levels) are skipped."""
        cl = it.centerline
        if not cl:
            return []
        levels = self.cfg.flight_levels_m
        pts = [(np.asarray(p, float), float(t)) for p, t in cl]
        out: list[tuple[int, Cell]] = []
        s_lo = int(math.ceil(pts[0][1] / self.dt))
        s_hi = int(math.floor(pts[-1][1] / self.dt))
        k = 0
        for s in range(s_lo, s_hi + 1):
            t = s * self.dt
            while k + 1 < len(pts) and pts[k + 1][1] < t:
                k += 1
            if k + 1 >= len(pts):
                break
            (p0, t0), (p1, t1) = pts[k], pts[k + 1]
            a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            p = p0 + a * (p1 - p0)
            level = min(range(len(levels)), key=lambda i: abs(float(p[2]) - levels[i]))
            if abs(float(p[2]) - levels[level]) < 0.5:
                q, r = hg.enu_to_axial(float(p[0]), float(p[1]), self._R)
                out.append((s, (q, r, level)))
        return out

    # ------------------------------------------------------------------- transaction
    def try_repair(
        self,
        victims,
        rng: np.random.Generator,
        accept_epsilon: float = 0.0,
        order_mode: str = "premium",
    ) -> RepairOutcome:
        """One LNS iteration body: release the victims, PP-replan them in priority order,
        accept iff the neighborhood's summed weighted cost strictly improves, else restore the
        old plans verbatim.

        ``order_mode="premium"`` (default) plans the most-delayed victims first with random
        tie-breaking among equal premiums — measured on 150 agent-based neighborhoods to
        capture +51% improvement mass over uniform random order and to collapse the
        exact-no-op rate 47% -> 11.5% (FCFS re-lock: an undelayed victim planned first
        deterministically re-picks its old geodesic and walls the delayed one back out; see
        context/lns_plan.md §4). ``"random"`` is the paper's uniform order, kept for A/Bs."""
        victims = sorted(victims)
        assert all(f in self._movable_set for f in victims)
        old = {f: self.incumbent[f] for f in victims}
        cost_old = float(sum(it.cost for it in old.values()))

        self.ledger.release_many(victims)
        if order_mode == "premium":
            jitter = rng.random(len(victims))
            order = [f for _, _, f in sorted((-self.delay(f), jitter[i], f)
                                             for i, f in enumerate(victims))]
        elif order_mode == "random":
            order = [victims[i] for i in rng.permutation(len(victims))]
        else:
            raise ValueError(f"unknown order_mode {order_mode!r} (want 'premium' or 'random')")
        new: dict[int, OperationalIntent] = {}
        reason = "improved"
        for fid in order:
            it = self.repair_planner.plan(old[fid].request, self.ledger, self.cfg)
            if not it.accepted:
                reason = "denied"
                break
            self.ledger.commit(fid, it.volumes)
            new[fid] = it

        if reason == "improved" and self._turnaround_s is not None:
            for fid, it in new.items():
                anchor = self._return_anchor.get(fid)
                if anchor is None:
                    continue
                rel = realized_release_s(it)
                if rel is not None and rel + self._turnaround_s > anchor + 1e-6:
                    reason = "anchor"
                    break

        cost_new = float(sum(it.cost for it in new.values())) if reason == "improved" else math.inf
        if reason == "improved" and cost_new < cost_old - accept_epsilon:
            for fid, it in new.items():
                self.total_cost += float(it.cost) - float(self.incumbent[fid].cost)
                self.incumbent[fid] = it
                self._index_remove(fid)
                self._index_add(fid, it.volumes)
                self._visits.pop(fid, None)
            return RepairOutcome(True, "improved", cost_old, cost_new, len(new))

        if reason == "improved":
            reason = "no_improvement"
        if new:
            self.ledger.release_many(new.keys())
        for fid in victims:
            self.ledger.commit(fid, old[fid].volumes)
        return RepairOutcome(False, reason, cost_old, cost_new, len(new))

    # ---------------------------------------------------------------------- readout
    def final_intents(self) -> list[OperationalIntent]:
        """The incumbent schedule in the original request order (drop-in for SimResult.intents)."""
        return [self.incumbent[fid] for fid in self.order]
