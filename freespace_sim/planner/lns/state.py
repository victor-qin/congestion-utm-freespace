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
* On reject/failure/exception—or a worker's report-only success—the new plans are tombstoned
  and the old volumes re-committed verbatim, one ``commit`` per flight (``_absorb`` needs each
  flight's volumes contiguous)—see ``_rewind``. It runs on EVERY non-adopting exit from the
  destroyed state: the ledger is the run's only copy of the schedule, so a transaction that
  unwinds without it loses flights outright.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import hexgrid as hg
from freespace_sim.planner.astar import AStarPlanner
from freespace_sim.planner import chain_attr, iter_planner_chain
from freespace_sim.planner.lns.unimpeded import resolve_workers, unimpeded_costs
from freespace_sim.sim import realized_release_s
from freespace_sim.types import OperationalIntent

log = logging.getLogger("freespace_sim.lns")

Cell = tuple[int, int, int]

# Baselines whose per-flight costs a plain ``AStarPlanner`` reproduces, so the unimpeded ruler and the
# incumbent are denominated in the same currency. ``astar_ref`` is the same search without the compiled
# kernel (byte-identical by contract); the shortcut/MILP/colgen families are NOT — see LNSState.
#
# The SIPP pair is here on measured evidence, not by analogy. A* and SIPP are exact optimizers of the
# SAME weighted cost over the same lattice, so they must agree on the optimum even though they break
# ties differently and file different routes — and `tests/test_lns_sipp.py` pins that both on the
# empty ruler world (1e-9) and on a congested A*-committed ledger (1e-6), which is the check that was
# missing when this set was written. What that buys: a `planner="sipp"` baseline may be repaired and
# ruled by A*, and vice versa, without the delay premium comparing two currencies.
_REPRODUCIBLE_PLANNERS = frozenset({"astar", "astar_ref", "sipp", "sipp_ref"})
#: Planners LNS may construct as its repair planner. Deliberately a small ALLOWLIST rather than
#: `get_planner`: that registry also holds `ShortcutRefiner` wrappers (a wrapper has no `evict_floor`
#: of its own — see the vet block below) and the whole-schedule `colgen`, neither of which meets the
#: repair contract. An explicit list fails loudly on `astar_shortcut` instead of three frames later.
LNS_REPAIR_PLANNERS = ("astar", "astar_ref", "sipp", "sipp_ref")


def _new_repair_planner(name, *, incremental_release, kernel_log2_min=None,
                        record_envelope=False, window_bytes=None):
    """The ONE construction site for an LNS repair planner, so the sequential path, `LNSState`'s
    default and `LNSState.replica` cannot drift.

    `evict_floor = 0.0` is set HERE because this constructor is the owner: `LNSState`'s vet block
    only runs for a BORROWED planner, so a constructed one is never checked. Callers must validate
    `name` BEFORE the ledger is taken over (`solver._validate_lns_config`) — see `LNSState.__init__`.
    """
    # `window_bytes` is the #124 dense-window budget; omitted rather than passed as None so each
    # planner keeps its own default.
    kw = {} if window_bytes is None else {"window_bytes": window_bytes}
    if name in ("astar", "astar_ref"):
        planner = AStarPlanner(compiled=name == "astar", kernel_log2_min=kernel_log2_min,
                               incremental_release=incremental_release, **kw)
    elif name in ("sipp", "sipp_ref"):
        from freespace_sim.planner.sipp import SIPPPlanner

        planner = SIPPPlanner(compiled=name == "sipp", kernel_log2_min=kernel_log2_min,
                              incremental_release=incremental_release, **kw)
    else:
        raise ValueError(
            f"repair_planner {name!r} is not a supported LNS repair planner "
            f"(want one of {LNS_REPAIR_PLANNERS})")
    planner.evict_floor = 0.0   # random/premium repair orders need the full-horizon occupancy
    planner.record_envelope = record_envelope
    return planner


#: Absent-value sentinel for `_same_committed_schedule`'s per-owner walk, where None is a real value.
_MISSING = object()
#: Premium slack absorbing cost-accumulation float noise, well below one dt of ground delay.
_PREMIUM_TOL = 1e-6
#: Share of movable flights that may go negative from refiner tie-break noise before it reads as a
#: currency mismatch. Measured regimes are 0.3% (artifact) against 17-85% (bug) — nothing in between.
_PREMIUM_ARTIFACT_FRAC = 0.02


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
    # The repaired schedule, populated ONLY on the accept return: a parallel worker has to hand
    # these back to the coordinator, which owns the incumbent. The reject path (79% of iterations)
    # must stay free, so nothing is built for it.
    new_intents: dict[int, OperationalIntent] = field(default_factory=dict)
    # One `AStarPlanner.last_envelope` per repaired flight, in repair order, when the planner was
    # built with `record_envelope`. Entries may be None (the planner resets it per plan and only
    # `_mk_envelope` sets it, so a host-side early denial leaves it unset) — a consumer must treat
    # None as "read set unknown", i.e. always dirty.
    envelopes: tuple = ()

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
        repair_planner_name: str = "astar",
        incremental_release: bool = True,
        # Safe for direct construction too: None is an explicit opt-in to automatic multiprocessing.
        unimpeded_workers: int | None = 1,
        unimpeded_cost: dict[int, float | None] | None = None,
        maintain_claim_index: bool = True,
        window_bytes: int | None = None,
        shortcut_repair: str = "none",
        shortcut_ruler: bool = False,
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
        # premiums and acceptance comparisons use different currencies. `repair_planner is None` is
        # NOT on its own evidence of a bare repair: the deferred arms keep the default A* and refine
        # after the PP loop, so they reproduce a refined baseline just as a wrapper does. Testing
        # only the planner slot refused `astar_shortcut` + shortcut_repair, which is the one
        # configuration where an in-loop cut is being asked a fair question. This stays a cheap
        # string pre-check either way; the authority is the numeric premium invariant below, which
        # reads the actual costs and cannot be fooled by a planner name.
        if (repair_planner is None and shortcut_repair == "none"
                and cfg.planner not in _REPRODUCIBLE_PLANNERS):
            raise ValueError(
                f"LNS cannot measure a {cfg.planner!r} baseline: its unimpeded ruler is a plain A*, "
                f"and the repair planner is one of {sorted(_REPRODUCIBLE_PLANNERS)}, so delay premiums "
                f"would compare two different planners. Pass repair_planner= a planner object that "
                f"reproduces {cfg.planner!r}, set a shortcut_repair arm, or re-run the baseline with "
                f"a planner in that set.")

        # ---- everything below runs BEFORE `ledger.detach_subscribers()` -----------------------
        # That call is irreversible (it bumps the epoch, stripping the caller's ledger), and the
        # unimpeded fan-out just after it IS the state build (measured 32-83 s). So every argument
        # error has to be caught here, by a check that reads only the arguments.
        # `solver._validate_lns_config` applies the same rule one level up, to the config fields.
        if shortcut_repair not in ("none", "deferred", "post_accept"):
            raise ValueError(f"unknown shortcut_repair {shortcut_repair!r} (want 'none', 'deferred' "
                             "or 'post_accept'; the INTERLEAVED arm is a ShortcutRefiner passed as "
                             "repair_planner, not a mode)")
        # The numeric premium invariant below cannot catch this one: at build time the incumbent is
        # still the UNREFINED baseline, and the refinement that breaks the currency happens later,
        # per transaction.
        if shortcut_repair != "none" and not shortcut_ruler:
            raise ValueError(
                f"shortcut_repair={shortcut_repair!r} with shortcut_ruler=False: the repaired "
                "incumbent would be measured against a bare-A* unimpeded plan, so every refined "
                "flight's delay premium goes negative and `delay()` clamps it to a silent 0")
        # These arms release a flight and re-check pad capacity WITHOUT replanning, so the shrink
        # tripwire never fires for them and a `track_removal=False` authority never subtracts the
        # released flight's own dwell (measured: it stays in `TerminalCapacity.dwells`). It then
        # refuses that flight's own shortcut. A quality loss is invisible, so refuse instead.
        releases = (chain_attr(repair_planner, "incremental_release") if repair_planner is not None
                    else [bool(incremental_release)])
        if shortcut_repair != "none" and not (releases and all(releases)):
            raise ValueError(
                f"shortcut_repair={shortcut_repair!r} needs incremental_release=True: it releases a "
                "flight and re-checks pad capacity without replanning, and a capacity authority "
                "built with track_removal=False cannot un-absorb the released flight's own dwell")

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
            # The floor belongs to the SEARCH planner, so a wrapper (ShortcutRefiner) never carries
            # one — walk the chain. Reading it off the wrapper made every correctly-configured
            # wrapper raise, which is why no shortcut repair arm could be constructed at all.
            # Demand it of every planner in the chain that has the attribute: a diamond
            # (astar_milp_shortcut) holds several searches and ANY of them advancing the watermark
            # would evict an earlier victim's obstacles out from under a later one.
            floors = chain_attr(repair_planner, "evict_floor")
            if not floors or any(f != 0.0 for f in floors):
                raise ValueError("repair_planner.evict_floor must be 0.0 on every search planner in "
                                 "its wrapper chain — random/premium repair orders need the "
                                 "full-horizon occupancy, and the floor is the caller's to set "
                                 "(on the inner planner, for a wrapper)")
        else:
            # Construct before taking ownership of the caller's ledger. `run_lns` validates its
            # config first, but LNSState is also directly constructible; an invalid window budget or
            # a guarded JIT failure must not strip observers before the planner constructor reports it.
            #
            # incremental_release=True: the planner's occupancy/capacity services subscribe to
            # `release_many` and un-absorb victims in O(their volumes), so the per-iteration shrink
            # rebuild (measured 94% of iteration wall) never happens. False keeps the rebuild path
            # (the byte-parity reference for A/Bs).
            repair_planner = _new_repair_planner(
                repair_planner_name, incremental_release=incremental_release,
                window_bytes=window_bytes)

        # LNS takes ownership of the ledger: the FCFS run's planner services stay subscribed
        # otherwise, silently absorbing (and retaining the memory of) every repair commit. The epoch
        # bump is what makes the takeover safe for the DETACHED planner too — it rebinds instead of
        # planning against an occupancy frozen at this instant (see ReservationLedger.epoch).
        ledger.detach_subscribers()

        self.shortcut_repair = shortcut_repair
        self.repair_planner = repair_planner

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
        if unimpeded_cost is not None:
            # Injected by a parallel replica: the ruler is a pure function of
            # (request, cfg, static_terms), so it is computed ONCE by the coordinator and broadcast
            # rather than re-planned per worker — which would also have each worker stand up a ruler
            # pool of its own. `None` still means "denied" and falls through to the SAME fan-out
            # below, so the injected and computed paths cannot disagree about `delay()`.
            missing = [f for f in self._movable if f not in unimpeded_cost]
            if missing:
                # Silent otherwise: `delay()` would KeyError mid-walk, potentially an hour in.
                raise ValueError(
                    f"unimpeded_cost is missing {len(missing)} movable flight(s) (first: "
                    f"{missing[:5]}) — it must cover every movable id, and the caller's movable "
                    f"rule must match this state's (accepted, not frozen, uss-filtered)")
            rows = [(fid, unimpeded_cost[fid], "upstream ruler") for fid in self._movable]
        else:
            rows = unimpeded_costs(
                cfg, self.static_terms, [self.incumbent[fid].request for fid in self._movable],
                n_workers=resolve_workers(unimpeded_workers), shortcut=shortcut_ruler,
            )
        for fid, cost, denial in rows:
            if cost is not None:
                self._unimp_cost[fid] = cost
            else:  # can't even place it alone (cap artifact): treat as undelayed, never seed a walk
                self._unimp_cost[fid] = float(self.incumbent[fid].cost)
                log.warning("lns: unimpeded plan denied for flight %d (%s)", fid, denial)
        # A premium below zero means the incumbent beat a flight's own ALONE plan. Two very different
        # things produce that, and `delay()`'s `max(0.0, ...)` used to hide both — the flight then
        # reads as perfectly unimpeded and drops out of the agent-based destroy seed AND the
        # `premium` repair order.
        #
        # (1) CURRENCY MISMATCH — the ruler and the incumbent came from different planners. Systemic:
        #     measured 34/40 flights (85%) on a dallas_hub_2uss x240 `astar_shortcut` baseline ruled
        #     by bare A*, 76/451 (17%) at x1800. This is a bug and must raise.
        # (2) TIE-BREAK ARTIFACT — both sides ARE the same planner, but a greedy refiner is in the
        #     currency and the ruler is no longer a lower bound. A* returns equal-cost geodesics of
        #     different SHAPE in the empty and the congested world, and `shortcut_corners` is a
        #     shape-dependent greedy fixpoint, so the solo plan can refine WORSE. Measured on
        #     density_faa: flight 112, zero ground delay and zero air hold, identical flight level
        #     and altitude change, `air_detour_m` 58.6 m in traffic against 201.0 m alone. Rare:
        #     1/290. This is not a bug and must not raise.
        #
        # Prevalence separates them cleanly (85% / 17% against 0.3%), so that is what is tested.
        # The residual is then clamped by taking the MINIMUM rather than by `max(0.0, ...)` on the
        # difference: the flight demonstrably achieved `incumbent.cost` in a world holding traffic,
        # so its true solo optimum is at most that. Tightening a bound with evidence already in hand,
        # not a fudge. (`delay()` keeps its own `max(0.0, ...)`: the incumbent moves during the run,
        # so this fixes the ruler at build time, it does not make the clamp unreachable.)
        premium = {fid: float(self.incumbent[fid].cost) - self._unimp_cost[fid]
                   for fid in self._movable}
        negative = sorted((f for f, d in premium.items() if d < -_PREMIUM_TOL),
                          key=lambda f: premium[f])
        if negative and len(negative) > max(1, int(_PREMIUM_ARTIFACT_FRAC * len(self._movable))):
            raise ValueError(
                f"{len(negative)} of {len(self._movable)} movable flights "
                f"({100.0 * len(negative) / len(self._movable):.1f}%) have a NEGATIVE delay premium "
                f"(worst {premium[negative[0]]:.2f} on flight {negative[0]}, first ids "
                f"{negative[:5]}) — too many to be refiner tie-break noise, so the incumbent and the "
                f"unimpeded ruler were built by different planners. Pass shortcut_ruler=True when "
                f"the schedule was refined, or rebuild the baseline with the ruler's planner.")
        for fid in negative:
            log.warning("lns: flight %d beat its own unimpeded plan by %.2f (refiner tie-break); "
                        "clamping its ruler to the incumbent", fid, -premium[fid])
            self._unimp_cost[fid] = float(self.incumbent[fid].cost)

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
        # Where an iteration's wall goes, split the only two ways that matter for choosing a repair
        # planner: SEARCH (what SIPP makes cheaper) against LEDGER MAINTENANCE (what SIPP makes more
        # expensive, by keeping a fourth subscribed structure on terminal legs). Two counters over a
        # whole run, read once at the end — deliberately NOT per-flight attribution, which
        # `analysis/ab_column_clear.py` showed is dominated by the `perf_counter` calls themselves.
        self.t_plan_s = 0.0
        self.t_ledger_s = 0.0
        self._visits: dict[int, list[tuple[int, Cell]]] = {}
        self._maintain_claim_index = maintain_claim_index
        self._rebuild_claim_index()

    # ------------------------------------------------------------------ parallel replica
    @classmethod
    def replica(
        cls,
        cfg: SimConfig,
        intents: list[OperationalIntent],
        *,
        static_terms: tuple,
        unimpeded_cost: dict[int, float | None],
        turnaround_s: float | None = None,
        frozen_flight_ids: frozenset[int] = frozenset(),
        movable_uss_ids: frozenset[str] | None = None,
        incremental_release: bool = True,
        kernel_log2_min: int | None = None,
        record_envelope: bool = True,
        window_bytes: int | None = None,
        repair_planner_name: str = "astar",
    ) -> "LNSState":
        """A private copy of the incumbent for a parallel worker: own ledger, own planner.

        The ledger is rebuilt by committing the intents' own ``Volume4D`` objects (the recipe
        ``verify.find_interflight_conflict`` and ``parallel._worker_main`` both use), which also
        keeps the constructor's object-identity schedule check happy.

        **Every keyword here changes what a repair is ALLOWED to do**, so each must be forwarded
        or the worker silently runs a different algorithm than the coordinator believes it does:

        * ``turnaround_s`` builds ``_return_anchor``; without it ``try_repair``'s anchor guard is
          disarmed, so a repair may re-time an outbound past its return's departure — and
          ``verify.find_interflight_conflict`` checks 4D conflicts ONLY, so the run would still
          report ``verified``.
        * ``frozen_flight_ids`` / ``movable_uss_ids`` derive ``_movable``; without them the worker
          treats every accepted flight as movable, the destroy operators may select frozen
          flights, and ``try_repair``'s membership assert passes because it tests the worker's own
          (wrong) set.
        * ``incremental_release`` is the rebuild-path byte-parity reference (``--no-incremental``);
          hardcoding it would make that A/B inexpressible under parallelism.
        * ``window_bytes`` controls per-planner bitmap allocation and fallback frequency, so a worker
          that did not receive it would use a different resource policy from the coordinator.

        ``record_envelope`` is needed only when multiple DROP workers can return stale repairs;
        SYNC skips that bookkeeping, and effective widths below two stay in-process.
        ``unimpeded_workers=1`` is pinned because
        this constructor runs INSIDE a search worker — the ruler's own pool would otherwise fan out
        to m x m processes.
        """
        led = ReservationLedger(cfg)
        for center, term in static_terms:
            led.register_static_terminal(center, term)
        for it in intents:
            if it.accepted and it.volumes:
                led.commit(it.request.flight_id, it.volumes)
        # `repair_planner_name` is as load-bearing as every other keyword in this docstring: a worker
        # that builds A* while the coordinator believes it is running SIPP differs silently, and
        # `verify` (4D conflicts only) would still report the run as verified.
        planner = _new_repair_planner(repair_planner_name, incremental_release=incremental_release,
                                      kernel_log2_min=kernel_log2_min,
                                      record_envelope=record_envelope,
                                      window_bytes=window_bytes)
        return cls(
            cfg, led, intents,
            static_terms=static_terms,
            frozen_flight_ids=frozen_flight_ids,
            movable_uss_ids=movable_uss_ids,
            turnaround_s=turnaround_s,
            repair_planner=planner,
            incremental_release=incremental_release,
            unimpeded_cost=unimpeded_cost,
            unimpeded_workers=1,   # NO NESTED POOLS: this runs INSIDE a search worker, and the
            #                        ruler's own default (min(8, cpu-2)) would fan out again to
            #                        m x m processes. Belt and braces — `unimpeded_cost` is
            #                        supplied, so the ruler never runs at all.
        )

    # ------------------------------------------------------------------ claim index
    def _rebuild_claim_index(self) -> None:
        """Rebuild all destroy-heuristic claims from the incumbent schedule."""
        self._claims.clear()
        self._cells_of.clear()
        self._contended.clear()
        self._contended_list = None
        self._visits.clear()
        if not self._maintain_claim_index:
            return
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
        # Resolved once per flight, and shared with the two occupancy services through the memo in
        # `hg.column_hexes` — this index is the third consumer of the identical test.
        own_hexes = hg.column_hexes(own_cols, self._R) if own_cols else None
        for v in volumes:
            if v.terminal_id is not None and isinstance(v.shape, CylinderSpec):
                continue  # capacity-gated own column, not a blocked cell
            for q, r, level, s_lo, s_hi, in_blk in hg.rasterize_ranges(
                v, self.cfg, self._R, self._infl_b, self._infl_p
            ):
                if not in_blk:
                    continue
                if own_hexes is not None and (q, r) in own_hexes:
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

    # ------------------------------------------------------------------- shortcut arms
    def _shortcut_repaired(self, new: dict, order) -> None:
        """Refine each freshly-repaired flight against the OTHER repaired plans; mutate ``new``.

        A committed flight conflicts with itself, so each one is released, refined, and re-committed
        in repair order. The refiner's own contract does the rest: it re-verifies the rebuilt geometry
        against this ledger and returns the original when the splice does not hold, so a failure here
        costs a probe, never feasibility.
        """
        from freespace_sim.planner.shortcut import can_refine, refine_intent, terminal_capacity_for

        tcap = terminal_capacity_for(self.repair_planner, self.ledger)
        for fid in order:
            it = new.get(fid)
            # `can_refine` and not a local length test: skipping the release/commit round trip is an
            # optimisation, so its condition must be the refiner's own or the two can disagree.
            if it is None or not can_refine(it):
                continue
            self.ledger.release_many([fid])
            # Commit SOMETHING for this flight on every path. Between the release and the commit the
            # ledger is a flight short, and `_rewind` re-commits the OLD plan for every victim — over
            # a hole, that is still correct, but a raise from `refine_intent` would otherwise leave
            # this flight's slot doubly written on the rewind. Restoring the unrefined plan first
            # keeps the invariant "every victim has exactly one live commit" true at every yield.
            try:
                keep = refine_intent(it, self.ledger, self.cfg, tcap=tcap)
            except BaseException:
                self.ledger.commit(fid, it.volumes)
                raise
            self.ledger.commit(fid, keep.volumes)
            if keep is not it:
                new[fid] = keep

    # ------------------------------------------------------------------- transaction
    def try_repair(
        self,
        victims,
        rng: np.random.Generator,
        accept_epsilon: float = 0.0,
        order_mode: str = "premium",
        *,
        report_only: bool = False,
    ) -> RepairOutcome:
        """Release and PP-repair victims; accept only a strict weighted-cost improvement.

        ``premium`` repairs the most-delayed first with random ties; ``random`` retains the paper's
        order for A/B comparisons. ``report_only`` returns an accepted candidate but restores the
        incumbent ledger without adopting it; parallel workers use this because only the coordinator
        may commit a result. See ``context/lns_plan.md`` §4.
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
        # Read set per repaired flight, in repair order, for a parallel coordinator's staleness
        # test. Only collected when the planner was asked to record it, so the sequential path
        # builds nothing; `record_envelope` off leaves `last_envelope` None for every plan anyway.
        # Found by chain walk, and kept as the OBJECT so the flag and the read cannot disagree about
        # which planner they mean: a wrapper has neither attribute, and the resulting empty
        # `envelopes` is not a safe default — `parallel._read_set_is_clean` returns True for an empty
        # one, so "recorded nothing" would read as "read nothing" and every stale repair as clean.
        rec_src = next((pl for pl in iter_planner_chain(self.repair_planner)
                        if getattr(pl, "record_envelope", False)), None)
        envelopes: list = []
        candidate: RepairOutcome | None = None
        # EVERYTHING that can leave the schedule half-destroyed lives inside this block — the destroy
        # itself included. `release_many` tombstones every victim volume BEFORE it notifies removal
        # subscribers, and each of those hooks can raise, so a destroy that dies part-way is exactly
        # the "ledger missing k flights" state the handler exists to undo.
        try:
            t0 = time.perf_counter()
            self.ledger.release_many(victims)
            self.t_ledger_s += time.perf_counter() - t0
            for fid in order:
                t0 = time.perf_counter()
                it = self.repair_planner.plan(old[fid].request, self.ledger, self.cfg)
                self.t_plan_s += time.perf_counter() - t0
                if not it.accepted:
                    reason = "denied"
                    break
                t0 = time.perf_counter()
                self.ledger.commit(fid, it.volumes)
                self.t_ledger_s += time.perf_counter() - t0
                new[fid] = it
                if rec_src is not None:
                    envelopes.append(rec_src.last_envelope)

            if reason == "improved" and self._turnaround_s is not None:
                for fid, it in new.items():
                    anchor = self._return_anchor.get(fid)
                    if anchor is None:
                        continue
                    rel = realized_release_s(it)
                    if rel is not None and rel + self._turnaround_s > anchor + 1e-6:
                        reason = "anchor"
                        break

            # A2: tighten the whole neighborhood BEFORE the accept test, so the geometry the test
            # weighs is the geometry that would be adopted. The anchor guard above ran on the
            # unrefined plans; a refine only shortens the path, so it can only pull the realized
            # release EARLIER, and an anchor that held cannot be broken by it.
            if reason == "improved" and self.shortcut_repair == "deferred":
                self._shortcut_repaired(new, order)

            cost_new = float(sum(it.cost for it in new.values())) if reason == "improved" else math.inf
            if reason == "improved" and cost_new < cost_old - accept_epsilon:
                # A3: the test has already passed on un-refined cost — polish only the winner, so a
                # rejected repair never pays for a cut. `cost_new` is then RESTATED from the refined
                # plans; leaving the pre-cut figure would make `total_cost` disagree with the sum of
                # the incumbent's own costs, which is the number every later accept test reads.
                if self.shortcut_repair == "post_accept":
                    self._shortcut_repaired(new, order)
                    cost_new = float(sum(it.cost for it in new.values()))
                candidate = RepairOutcome(
                    True, "improved", cost_old, cost_new, len(new),
                    new_intents=dict(new), envelopes=tuple(envelopes),
                )
                if not report_only:
                    # Inside the try as well: this rewrites the incumbent, running cost, and claim
                    # index, so a raise part-way would otherwise leave them describing a schedule
                    # the ledger does not hold. LEDGER-FREE: the loop already committed the plans.
                    self._apply_in_memory(new, applied)
                    return candidate
        except BaseException:
            self._rewind(victims, old, cost_at_entry, applied)
            raise

        if candidate is not None:
            # Leave in-memory state (including claim/visit indexes) untouched. Restore once outside
            # the exception handler so a failed restore cannot trigger a second rewind and mask it.
            self._rewind(victims, old, cost_at_entry)
            return candidate
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
        t0 = time.perf_counter()
        self.ledger.release_many(victims)
        failures = []
        for fid in victims:
            try:
                self.ledger.commit(fid, old[fid].volumes)
            except BaseException as exc:            # noqa: BLE001 - re-raised below, after the rest
                failures.append((fid, exc))
        self.t_ledger_s += time.perf_counter() - t0    # the rewind is ledger work too, and on the
        #                                                rejection path (79%) it is HALF of it
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

    # ------------------------------------------------------------------ incumbent moves
    def _apply_in_memory(self, changes: dict[int, OperationalIntent], applied: list[int]) -> None:
        """Move the incumbent, the running cost and the claim index onto ``changes``.

        Assumes the LEDGER already holds them — true straight out of the repair loop, which
        commits as it plans. ``applied`` is filled as it goes so a caller's rollback knows exactly
        which fids had been rewritten when a raise landed part-way.
        """
        for fid, it in changes.items():
            applied.append(fid)
            self.total_cost += float(it.cost) - float(self.incumbent[fid].cost)
            self.incumbent[fid] = it
            if self._maintain_claim_index:
                self._index_remove(fid)
                self._index_add(fid, it.volumes)
                self._visits.pop(fid, None)

    def apply_delta(self, changes: dict[int, OperationalIntent]) -> None:
        """Adopt someone else's accepted repair: move the LEDGER **and** the in-memory views.

        The replica-sync counterpart of ``try_repair``'s accept branch. The difference is the
        ledger: ``try_repair`` has already committed its own plans and only the in-memory views
        are behind, while a worker told "the incumbent moved" still holds the old volumes and must
        release them first. O(the changed flights), not O(the schedule) — which is what makes a
        parallel worker's "take a private copy of P_min" affordable at all.

        One ``commit`` per flight, because ``_absorb`` groups a flight's volumes by adjacent runs
        and needs them contiguous. Any failure rewinds the whole delta before propagating.
        """
        # The CALLER's order, not sorted. `changes` comes out of a repair in PP priority order, and
        # replaying it in that order lands the ledger's `_vols`/`_fids` in the same layout an
        # in-process repair would have produced. Deterministic either way: every producer of this
        # dict builds it from a seeded order.
        fids = list(changes)
        if not fids:
            return
        old = {f: self.incumbent[f] for f in fids}
        cost_at_entry = self.total_cost
        applied: list[int] = []
        try:
            self.ledger.release_many(fids)
            for fid in fids:
                self.ledger.commit(fid, changes[fid].volumes)
            self._apply_in_memory(changes, applied)
        except BaseException:
            self._rewind(fids, old, cost_at_entry, applied)
            raise

    # ---------------------------------------------------------------------- readout
    def final_intents(self) -> list[OperationalIntent]:
        """The incumbent schedule in the original request order (drop-in for SimResult.intents)."""
        return [self.incumbent[fid] for fid in self.order]
