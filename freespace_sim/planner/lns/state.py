"""Mutable MAPF-LNS solution state over the reservation ledger.

Owns the incumbent schedule (fid -> OperationalIntent), the per-cell claim index the
destroy heuristics read (implements ``neighborhood.DestroyContext``), and the
destroy -> PP-repair -> accept/revert transaction against the shared
``ReservationLedger``.

Transaction contract (why this is exact — the seams are documented in
context/lns_plan.md):

* Destroy uses ``ledger.release_many`` — tombstones, never an observer re-feed. With
  ``incremental_release`` (the default) the repair planner's services subscribed to the
  removal hook and un-absorb the victims exactly, in O(their volumes); with it off they
  instead notice the shrink (``n_volumes < n_added``) on the iteration's first ``plan()``
  and rebuild from ``iter_committed``. Either way the repair sees an exact occupancy of
  "everyone but the victims" — the two paths are pinned byte-identical.
* The repair planner runs with ``evict_floor = 0.0`` (the Track A out-of-order
  dispatch knob), so the eviction watermark never advances and victims can be
  replanned in ANY priority order — random PP orderings stay exact.
* On reject/failure/exception the new plans are tombstoned and the old volumes re-committed
  verbatim, one ``commit`` per flight (``_absorb`` needs each flight's volumes contiguous)
  — see ``_rewind``. It runs on EVERY exit from the destroyed state, including an
  exception thrown mid-repair: the ledger is the run's only copy of the schedule, so a
  transaction that unwinds without it loses flights outright.
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
from freespace_sim.planner.lns.unimpeded import resolve_workers, unimpeded_costs
from freespace_sim.sim import realized_release_s
from freespace_sim.types import OperationalIntent

log = logging.getLogger("freespace_sim.lns")

Cell = tuple[int, int, int]

# Baselines whose per-flight costs a plain ``AStarPlanner`` reproduces, so the unimpeded ruler and the
# incumbent are denominated in the same currency. ``astar_ref`` is the same search without the compiled
# kernel (byte-identical by contract); the shortcut/MILP/colgen families are NOT — see LNSState.
_REPRODUCIBLE_PLANNERS = frozenset({"astar", "astar_ref"})
_MISSING = object()


def _same_committed_schedule(
    ledger: ReservationLedger, intents: list[OperationalIntent]
) -> bool:
    """Whether the ledger holds the exact volume objects owned by the accepted intents.

    Commits retain the immutable ``Volume4D`` objects from each intent. Comparing those references is
    exact without hashing geometry, and the ledger's per-flight runs let this use O(flights) memory.
    Flight commit order may change after a repair; volume order within each flight may not.
    """
    expected: dict[int, list] = {}
    for intent in intents:
        if not intent.accepted:
            continue
        fid = intent.request.flight_id
        if fid in expected:  # duplicate ownership cannot describe one ledger schedule
            return False
        expected[fid] = intent.volumes or []

    seen: set[int] = set()
    current_fid = _MISSING
    current_volumes: list = []
    position = 0
    for fid, volume in ledger.iter_committed():
        if fid != current_fid:
            if current_fid is not _MISSING and position != len(current_volumes):
                return False
            if fid in seen or fid not in expected:
                return False
            seen.add(fid)
            current_fid = fid
            current_volumes = expected[fid]
            position = 0
        if position >= len(current_volumes) or volume is not current_volumes[position]:
            return False
        position += 1

    if current_fid is not _MISSING and position != len(current_volumes):
        return False
    return len(seen) == len(expected)


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
        # Safe for direct construction too: None is an explicit opt-in to automatic multiprocessing.
        unimpeded_workers: int | None = 1,
    ) -> None:
        self.cfg = cfg
        self.ledger = ledger
        self.dt = cfg.dt_s
        self.n_levels = len(cfg.flight_levels_m)
        self.rng: np.random.Generator = np.random.default_rng(0)  # solver re-seeds per iteration

        self.order = [it.request.flight_id for it in intents]
        self.incumbent: dict[int, OperationalIntent] = {it.request.flight_id: it for it in intents}

        # `run_lns` mutates the ledger in place, so its intents and ledger must still be the exact pair
        # produced by the same run (use LNSResult.intents after an earlier pass).
        n_intent_vols = sum(len(it.volumes) for it in intents if it.accepted)
        if n_intent_vols != ledger.n_volumes or not _same_committed_schedule(ledger, intents):
            raise ValueError(
                f"intents describe {n_intent_vols} live volumes but the ledger holds "
                f"{ledger.n_volumes}, or their owners/content differ — they are not the same schedule. "
                f"Re-run sim.run for a matching (intents, ledger) pair; an LNS pass mutates the ledger "
                f"in place and supersedes the intents it was given (use LNSResult.intents afterwards).")

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

        # Baseline, unimpeded, and repaired costs must come from compatible planners; otherwise delay
        # premiums and acceptance comparisons use different currencies.
        if repair_planner is None and cfg.planner not in _REPRODUCIBLE_PLANNERS:
            raise ValueError(
                f"LNS cannot measure a {cfg.planner!r} baseline: its unimpeded ruler and its repair "
                f"planner are plain A*, so delay premiums would compare two different planners. Pass "
                f"repair_planner= a planner that reproduces {cfg.planner!r}, or re-run the baseline "
                f"with planner='astar'.")

        # Vet a borrowed planner BEFORE taking the ledger over: a constructor that raises must not
        # leave the caller's ledger stripped of its subscribers.
        if repair_planner is not None:
            if (getattr(repair_planner, "_svc_ledger", None) is ledger
                    or getattr(repair_planner, "_cocc_ledger", None) is ledger):
                # It would rebind on the epoch bump below (correct, but it silently throws away the
                # warm services the caller built) — refuse rather than surprise them.
                raise ValueError("repair_planner must not already be bound to this ledger")
            # `evict_floor = 0.0` freezes the monotone eviction watermark, so victims can be replanned
            # in ANY priority order (a later victim planned first must not evict an earlier one's
            # obstacles). Required, never written here: silently rewriting a caller's planner would
            # outlive this state and change that planner's behavior everywhere else it is used.
            # `getattr` because a planner WRAPPER (ShortcutRefiner) has no such attribute of its own —
            # the floor belongs to the inner planner, and the caller has to have set it there.
            if getattr(repair_planner, "evict_floor", None) != 0.0:
                raise ValueError("repair_planner.evict_floor must be 0.0 — random/premium repair orders "
                                 "need the full-horizon occupancy, and the floor is the caller's to set "
                                 "(on the inner planner, for a wrapper)")

        # LNS takes ownership of the ledger: the FCFS run's planner services stay subscribed
        # otherwise, silently absorbing (and retaining the memory of) every repair commit. The epoch
        # bump is what makes the takeover safe for the DETACHED planner too — it rebinds instead of
        # planning against an occupancy frozen at this instant (see ReservationLedger.epoch).
        ledger.detach_subscribers()

        # incremental_release=True: the planner's occupancy/capacity services subscribe to
        # `release_many` and un-absorb victims in O(their volumes), so the per-iteration shrink
        # rebuild (measured 94% of iteration wall) never happens. False keeps the rebuild path
        # (the byte-parity reference for A/Bs).
        self.repair_planner = repair_planner
        if self.repair_planner is None:
            self.repair_planner = AStarPlanner(incremental_release=incremental_release)
            self.repair_planner.evict_floor = 0.0

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
        # ledger, which nothing is ever committed to: the plans cannot see each other, so
        # `unimpeded_costs` may shard them across processes without changing a cost (see its
        # docstring). This is the whole of the state build, and it grows with the schedule.
        self._unimp_cost: dict[int, float] = {}
        # Kept so the solver's verify replays the SAME world the unimpeded baseline was measured in —
        # one owner for "which walls is this run about", not two that can drift apart.
        self.static_terms = tuple(static_terms)
        rows = unimpeded_costs(
            cfg, self.static_terms, [self.incumbent[fid].request for fid in self._movable],
            n_workers=resolve_workers(unimpeded_workers),
        )
        for fid, cost, denial in rows:
            if cost is not None:
                self._unimp_cost[fid] = cost
            else:  # can't even place it alone (cap artifact): treat as undelayed, never seed a walk
                self._unimp_cost[fid] = float(self.incumbent[fid].cost)
                log.warning("lns: unimpeded plan denied for flight %d (%s)", fid, denial)

        # Claim index for the destroy heuristics: cell -> [(s_lo, s_hi, fid)] over the same
        # inflated corridor/pad raster A* deconflicts against (blocked rows only; the
        # terminal-tagged capacity cylinders are counting constraints, not cells).
        self._R = hg.circumradius(cfg)
        self._infl_b = cfg.corridor_width_m / 2.0 + self._R
        self._infl_p = cfg.effective_hover_radius_m + self._R
        self._claims: dict[Cell, list[tuple[int, int, int]]] = {}
        # fid -> the DISTINCT cells it claims. `_index_remove` filters each cell's row list by owner
        # and re-derives its contention, both of which are per-cell — so a per-ROW list did the same
        # work once per (cell, span) instead of once per cell, and stored a `(cell, s_lo, s_hi)`
        # triple whose spans nothing ever read (measured 44 MB at 290 flights).
        self._cells_of: dict[int, set[Cell]] = {}
        self._contended: set[Cell] = set()
        self._contended_list: list[Cell] | None = None
        self._visits: dict[int, list[tuple[int, Cell]]] = {}
        self._rebuild_claim_index()

    # ------------------------------------------------------------------ claim index
    def _rebuild_claim_index(self) -> None:
        """Rebuild all destroy-heuristic claims from the incumbent schedule."""
        self._claims.clear()
        self._cells_of.clear()
        self._contended.clear()
        self._contended_list = None
        self._visits.clear()
        for fid in self._movable:
            self._index_add(fid, self.incumbent[fid].volumes, refresh=False)
        for cell in self._claims:  # one contention sweep instead of a refresh per row
            self._refresh_contention(cell)

    def _index_add(self, fid: int, volumes, refresh: bool = True) -> None:
        rows = self._cells_of.get(fid)
        if rows is None:
            rows = self._cells_of[fid] = set()
        # The flight's own terminal columns. A corridor cell INSIDE one of them is the vertiport's
        # unreserved tactical interior — the occupancy services drop it (HexOccupancyService.add_volume
        # / CompiledHexOccupancy._add), so A* never deconflicts there. Indexing it anyway would make
        # every pair of same-hub flights look mutually contended at their shared hub, and the map-based
        # destroy operator picks cells BY contention: it would spend its neighborhoods on hub interiors
        # where no conflict can exist.
        own_cols = tuple((v.shape.cx, v.shape.cy, v.shape.radius) for v in volumes
                         if v.terminal_id is not None and isinstance(v.shape, CylinderSpec))
        for v in volumes:
            if v.terminal_id is not None and isinstance(v.shape, CylinderSpec):
                continue  # capacity-gated own column, not a blocked cell
            for q, r, level, s_lo, s_hi, in_blk in hg.rasterize_ranges(
                v, self.cfg, self._R, self._infl_b, self._infl_p
            ):
                if not in_blk:
                    continue
                if own_cols and self._inside_own_column(q, r, own_cols):
                    continue
                cell = (q, r, level)
                entries = self._claims.get(cell)
                if entries is None:
                    self._claims[cell] = [(s_lo, s_hi, fid)]
                else:
                    entries.append((s_lo, s_hi, fid))
                if cell not in rows:
                    # A flight claims the same cell in several volumes (adjacent corridor boxes
                    # overlap). Contention is the cell's OWNER SET, which the second and later
                    # claims cannot change — so refresh on the first only. Same final state.
                    rows.add(cell)
                    if refresh:
                        self._refresh_contention(cell)

    def _inside_own_column(self, q: int, r: int, cols) -> bool:
        """Same test as the occupancy services' ``_inside_a_column`` — hex centre inside any of the
        flight's own terminal column discs."""
        c = hg.hex_center(q, r, self._R)
        return any((c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= rad * rad for cx, cy, rad in cols)

    def _index_remove(self, fid: int) -> None:
        for cell in self._cells_of.pop(fid, ()):
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
        """Release and PP-repair victims; accept only a strict weighted-cost improvement.

        ``premium`` repairs the most-delayed first with random ties; ``random`` retains the paper's
        order for A/B comparisons. See ``context/lns_plan.md`` §4.
        """
        victims = sorted(victims)
        assert all(f in self._movable_set for f in victims)
        old = {f: self.incumbent[f] for f in victims}
        cost_old = float(sum(it.cost for it in old.values()))

        # Build the repair order BEFORE the destroy: everything it reads (delays, the incumbent) is
        # ledger-independent, and a bad `order_mode` must raise while the schedule is still intact —
        # validating after the release left the victims tombstoned with nobody to re-commit them.
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
        cost_at_entry = self.total_cost
        applied: list[int] = []   # fids whose accept-side in-memory rewrite has started
        # EVERYTHING that can leave the schedule half-destroyed lives inside this block — the destroy
        # itself included. `release_many` tombstones every victim volume BEFORE it notifies removal
        # subscribers, and each of those hooks can raise, so a destroy that dies part-way is exactly
        # the "ledger missing k flights" state the handler exists to undo.
        try:
            self.ledger.release_many(victims)
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
                # Inside the try as well: this rewrites the incumbent, the running cost and the claim
                # index, so a raise part-way (rasterize, MemoryError, a second SIGINT) would otherwise
                # leave them describing a schedule the ledger does not hold, with `old` already gone.
                for fid, it in new.items():
                    applied.append(fid)
                    self.total_cost += float(it.cost) - float(self.incumbent[fid].cost)
                    self.incumbent[fid] = it
                    self._index_remove(fid)
                    self._index_add(fid, it.volumes)
                    self._visits.pop(fid, None)
                return RepairOutcome(True, "improved", cost_old, cost_new, len(new))
        except BaseException:
            self._rewind(victims, old, cost_at_entry, applied)
            raise

        if reason == "improved":
            reason = "no_improvement"
        self._rewind(victims, old, cost_at_entry)
        return RepairOutcome(False, reason, cost_old, cost_new, len(new))

    def _rewind(self, victims, old, cost_at_entry, applied=()) -> None:
        """Restore the ledger and any partially applied in-memory acceptance.

        Release every victim because ``commit`` can append before a subscriber raises, leaving live
        volumes absent from the repair's bookkeeping. Every victim gets a restore attempt even if an
        earlier one fails. If acceptance had begun, rebuild the full claim index from the restored
        incumbent; that exceptional O(all claims) path also heals partial index mutations while the
        ordinary rejection path remains one release plus k commits.
        """
        self.ledger.release_many(victims)
        failures = []
        for fid in victims:
            try:
                self.ledger.commit(fid, old[fid].volumes)
            except BaseException as exc:            # noqa: BLE001 - re-raised below, after the rest
                failures.append((fid, exc))
        if applied:
            self.total_cost = cost_at_entry
            for fid in applied:
                self.incumbent[fid] = old[fid]
            self._rebuild_claim_index()
        if failures:
            fids = [f for f, _ in failures]
            raise RuntimeError(
                f"LNS restore could not re-commit flight(s) {fids} — the ledger no longer holds the "
                f"incumbent schedule for them") from failures[0][1]

    # ---------------------------------------------------------------------- readout
    def final_intents(self) -> list[OperationalIntent]:
        """The incumbent schedule in the original request order (drop-in for SimResult.intents)."""
        return [self.incumbent[fid] for fid in self.order]
