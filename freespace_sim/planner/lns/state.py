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
from freespace_sim.sim import realized_release_s
from freespace_sim.types import OperationalIntent

log = logging.getLogger("freespace_sim.lns")

Cell = tuple[int, int, int]

# Baselines whose per-flight costs a plain ``AStarPlanner`` reproduces, so the unimpeded ruler and the
# incumbent are denominated in the same currency. ``astar_ref`` is the same search without the compiled
# kernel (byte-identical by contract); the shortcut/MILP/colgen families are NOT — see LNSState.
_REPRODUCIBLE_PLANNERS = frozenset({"astar", "astar_ref"})


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

        # The intents must BE the schedule this ledger holds. They are two views of one thing and
        # nothing else re-checks it: `run_lns` mutates the ledger in place and never writes back to the
        # caller's intent list, so feeding the same SimResult to a second pass (the natural A/B shape)
        # measures a stale baseline against an already-improved ledger and returns a genuinely
        # conflicting schedule — measured: a real inter-flight 4D conflict, `verified False`, and the
        # runner writes the JSON anyway. A live-volume count is O(1) here and turns that into an error.
        n_intent_vols = sum(len(it.volumes) for it in intents if it.accepted)
        if n_intent_vols != ledger.n_volumes:
            raise ValueError(
                f"intents describe {n_intent_vols} live volumes but the ledger holds "
                f"{ledger.n_volumes} — they are not the same schedule. Re-run sim.run for a matching "
                f"(intents, ledger) pair; an LNS pass mutates the ledger in place and supersedes the "
                f"intents it was given (use LNSResult.intents afterwards).")

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

        # Both the unimpeded ruler below and the repair planner are plain A*. If the baseline was flown
        # by a different planner, `delay(fid) = incumbent.cost - unimpeded.cost` subtracts one
        # planner's cost from another's: measured on an `astar_shortcut` baseline (70 flights), 18 have
        # a NEGATIVE raw premium that `delay`'s max(0.0, ...) silently clamps to zero, and 7 of the 58
        # flights actually held on the ground report zero delay — so the agent seed and the premium
        # repair order go blind to genuinely delayed flights, and accepted repairs swap refined plans
        # for unrefined ones while cost_new < cost_old compares the two rulers. Refuse rather than
        # report a wrong number; pass `repair_planner=` explicitly to override.
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
        # ledger (its own planner instance: one planner per ledger).
        self._unimp_cost: dict[int, float] = {}
        # Kept so the solver's verify replays the SAME world the unimpeded baseline was measured in —
        # one owner for "which walls is this run about", not two that can drift apart.
        self.static_terms = tuple(static_terms)
        free = ReservationLedger(cfg)
        for center, term in self.static_terms:
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
                self._claims.setdefault(cell, []).append((s_lo, s_hi, fid))
                rows.append((cell, s_lo, s_hi))
                if refresh:
                    self._refresh_contention(cell)

    def _inside_own_column(self, q: int, r: int, cols) -> bool:
        """Same test as the occupancy services' ``_inside_a_column`` — hex centre inside any of the
        flight's own terminal column discs."""
        c = hg.hex_center(q, r, self._R)
        return any((c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= rad * rad for cx, cy, rad in cols)

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
        applied: list[int] = []   # fids the accept loop has already rewritten in memory
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
                    self.total_cost += float(it.cost) - float(self.incumbent[fid].cost)
                    self.incumbent[fid] = it
                    self._index_remove(fid)
                    self._index_add(fid, it.volumes)
                    self._visits.pop(fid, None)
                    applied.append(fid)
                return RepairOutcome(True, "improved", cost_old, cost_new, len(new))
        except BaseException:
            self._rewind(victims, old, cost_at_entry, applied)
            raise

        if reason == "improved":
            reason = "no_improvement"
        self._rewind(victims, old, cost_at_entry)
        return RepairOutcome(False, reason, cost_old, cost_new, len(new))

    def _rewind(self, victims, old, cost_at_entry, applied=()) -> None:
        """Put the incumbent back — ledger AND in-memory state — after a rejected or aborted
        transaction.

        Releases the VICTIMS rather than only what the repair recorded. The two differ precisely when a
        commit dies part-way: `ledger.commit` appends the volumes and only then fires observers, so an
        observer that raises leaves those volumes live while the caller's `new[fid] = it` never ran.
        Every fid the repair can have committed is a victim, and re-releasing an already-tombstoned
        flight is a no-op, so releasing `victims` covers the torn case for free — where releasing
        `new.keys()` would leave the flight double-booked against its own restored plan.

        Each victim is re-committed in ONE call (`_absorb` groups by adjacent runs, so a flight's
        volumes must stay contiguous), and every victim gets its attempt even if one fails: the
        restoring commit re-fires the same observers that may have just raised, and aborting the loop
        there would strand every victim after it. The first failure is re-raised once the rest are
        home.

        ``applied`` names the fids the ACCEPT loop had already rewritten in memory before dying — the
        only case where the incumbent, the running cost and the claim index need rolling back too. It
        is empty on the ordinary reject path, which therefore costs exactly what it always did (one
        release plus k commits) and never re-rasterizes an index that did not change."""
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
                self._index_remove(fid)
                self._index_add(fid, old[fid].volumes)
                self._visits.pop(fid, None)
        if failures:
            fids = [f for f, _ in failures]
            raise RuntimeError(
                f"LNS restore could not re-commit flight(s) {fids} — the ledger no longer holds the "
                f"incumbent schedule for them") from failures[0][1]

    # ---------------------------------------------------------------------- readout
    def final_intents(self) -> list[OperationalIntent]:
        """The incumbent schedule in the original request order (drop-in for SimResult.intents)."""
        return [self.incumbent[fid] for fid in self.order]
