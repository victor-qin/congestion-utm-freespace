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
from freespace_sim.planner.lns.unimpeded import resolve_workers, unimpeded_costs
from freespace_sim.verify import pair_precedence_shortfall
from freespace_sim.types import OperationalIntent

log = logging.getLogger("freespace_sim.lns")

Cell = tuple[int, int, int]

# Baselines whose per-flight costs a plain ``AStarPlanner`` reproduces, so the unimpeded ruler and the
# incumbent are denominated in the same currency. ``astar_ref`` is the same search without the compiled
# kernel (byte-identical by contract); the shortcut/MILP/colgen families are NOT — see LNSState.
#
# The SIPP pair belongs on measured evidence: A* and SIPP are exact optimizers of the same
# weighted cost over the same lattice, so they agree on the optimum even though they break ties
# differently and file different routes. `tests/test_lns_sipp.py` pins that agreement on the
# empty ruler world and on a congested A*-committed ledger. So a `planner="sipp"` baseline may be
# repaired and ruled by A* (and vice versa) without the delay premium comparing two currencies.
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

    Parameters
    ------------
    - name (str): repair planner name; one of ``astar``/``astar_ref``/``sipp``/``sipp_ref`` (the
      ``*_ref`` variants build the pure-Python planner), else ``ValueError``.
    - incremental_release (bool): planner incremental-release mode (keyword-only).
    - kernel_log2_min (int | None): starting g-hash/heap size exponent (keyword-only); ``None``
      uses the planner default.
    - record_envelope (bool): when True, the planner records its per-plan read envelope
      (keyword-only).
    - window_bytes (int | None): dense-window byte budget (keyword-only); ``None`` keeps each
      planner's own default.

    Return
    --------
    - output (AStarPlanner | SIPPPlanner): the constructed planner with ``evict_floor = 0.0`` and
      ``record_envelope`` set.
    """
    # `window_bytes` is the dense-window byte budget; omitted rather than passed as None so each
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
_MISSING = object()


def _same_committed_schedule(
    ledger: ReservationLedger, intents: list[OperationalIntent]
) -> bool:
    """Whether the ledger holds the exact volume objects owned by the accepted intents.

    Commits retain the immutable ``Volume4D`` objects from each intent. Comparing those references is
    exact without hashing geometry, and the ledger's per-flight runs let this use O(flights) memory.
    Flight commit order may change after a repair; volume order within each flight may not.

    Parameters
    ------------
    - ledger (ReservationLedger): the ledger whose committed volumes are compared, via
      ``iter_committed``.
    - intents (list[OperationalIntent]): the schedule to match; only ``accepted`` intents count.

    Return
    --------
    - output (bool): True iff the ledger's committed volumes are exactly the objects owned by the
      accepted intents, per flight and in order.
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
    # these back to the coordinator, which owns the incumbent. The reject path is the common case
    # and must stay free, so nothing is built for it.
    new_intents: dict[int, OperationalIntent] = field(default_factory=dict)
    # One `AStarPlanner.last_envelope` per repaired flight, in repair order, when the planner was
    # built with `record_envelope` — the read set a parallel coordinator tests commits against (see
    # context/figures/read_envelope.png). Entries may be None (the planner resets it per plan and
    # only `_mk_envelope` sets it, so a host-side early denial leaves it unset) — a consumer must
    # treat None as "read set unknown", i.e. always dirty.
    envelopes: tuple = ()

    @property
    def improvement(self) -> float:
        """The weighted-cost drop this repair achieved, or 0.0 when it was not accepted."""
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
    ) -> None:
        """Take ownership of ``ledger`` and build the incumbent, ruler, and destroy claim index.

        Detaches the ledger's existing subscribers and bumps its epoch, so the repair planner
        rebinds instead of planning against a frozen occupancy. Validates that ``intents`` and
        ``ledger`` describe the same schedule before touching anything.

        Parameters
        ------------
        - cfg (SimConfig): sim config (planner, ``dt_s``, flight levels, corridor/pad geometry).
        - ledger (ReservationLedger): the committed schedule LNS takes over (subscribers detached).
        - intents (list[OperationalIntent]): the incumbent; must be the exact pair produced with
          ``ledger`` by one run (an LNS pass mutates the ledger in place).
        - static_terms (tuple): the (center, terminal) walls this run is about; kept so verify
          replays the same world the ruler was measured in.
        - frozen_flight_ids (frozenset[int]): flights excluded from the movable set.
        - movable_uss_ids (frozenset[str] | None): if set, only these USS ids are movable.
        - turnaround_s (float | None): enables the paired-return anchor guard when not None.
        - repair_planner (AStarPlanner | None): a borrowed repair planner (must have
          ``evict_floor == 0.0`` and not already be bound to this ledger); None constructs one.
        - repair_planner_name (str): which planner to construct when ``repair_planner`` is None.
        - incremental_release (bool): un-absorb victims incrementally (default), or take the
          shrink-rebuild path (the byte-parity reference for A/Bs).
        - unimpeded_workers (int | None): worker count for the unimpeded ruler; None opts into
          automatic multiprocessing.
        - unimpeded_cost (dict[int, float | None] | None): a precomputed ruler broadcast by a
          coordinator; None computes it here, and a None entry means "denied".
        - maintain_claim_index (bool): build the destroy-heuristic claim index (off for consumers
          that never destroy by cell).
        - window_bytes (int | None): dense-window byte budget forwarded to a constructed planner.

        Return
        --------
        - output (None): builds the state in place; raises ValueError on a mismatched
          ``(intents, ledger)`` pair, an unrepeatable baseline planner, or a borrowed planner that
          fails the vet.
        """
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
                f"LNS cannot measure a {cfg.planner!r} baseline: its unimpeded ruler is a plain A*, "
                f"and the repair planner is one of {sorted(_REPRODUCIBLE_PLANNERS)}, so delay premiums "
                f"would compare two different planners. Pass repair_planner= a planner object that "
                f"reproduces {cfg.planner!r}, or re-run the baseline with a planner in that set.")

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
        else:
            # Construct before taking ownership of the caller's ledger. `run_lns` validates its
            # config first, but LNSState is also directly constructible; an invalid window budget or
            # a guarded JIT failure must not strip observers before the planner constructor reports it.
            repair_planner = _new_repair_planner(
                repair_planner_name, incremental_release=incremental_release,
                window_bytes=window_bytes)

        # LNS takes ownership of the ledger: the FCFS run's planner services stay subscribed
        # otherwise, silently absorbing (and retaining the memory of) every repair commit. The epoch
        # bump is what makes the takeover safe for the DETACHED planner too — it rebinds instead of
        # planning against an occupancy frozen at this instant (see ReservationLedger.epoch).
        ledger.detach_subscribers()

        # incremental_release=True: the planner's occupancy/capacity services subscribe to
        # `release_many` and un-absorb victims in O(their volumes), so the per-iteration shrink
        # rebuild (the dominant cost of an iteration) never happens. False keeps the rebuild path
        # (the byte-parity reference for A/Bs).
        self.repair_planner = repair_planner

        # Paired-return PRECEDENCE: a return cannot depart before the aircraft flying it has landed.
        # This is a different property from `verify`'s separation check and invisible to it — the two
        # legs hold DISJOINT windows at the same pad, so there is no 4D overlap to find. `try_repair`
        # is the only place it can be PREVENTED, because by the time a whole-schedule replay sees it
        # the repair has already been accepted.
        self._turnaround_s = turnaround_s
        from freespace_sim import verify as _verify
        # Per-pair, not a count. A nominal-anchor schedule arrives with violations already in it, so
        # the rule is "no pair gets worse", not "no pair is bad" — and a COUNT cannot express that:
        # LNS can repair pair A and break pair B in one iteration with the count unchanged.
        self._pair_shortfall = _verify.pair_shortfalls(intents, float(turnaround_s or 0.0))
        self._precedence_baseline = sum(1 for v in self._pair_shortfall.values() if v > 1e-6)
        # Round-trip partners, BOTH directions: the guard has to reach the leg this repair did NOT
        # touch. Built from the requests, so it is populated under nominal anchoring too.
        self._pair_of: dict[int, int] = {}
        self._outbound_of_pair: dict[int, int] = {}   # either leg's fid -> the OUTBOUND leg's fid
        for it in intents:
            pid = it.request.paired_outbound_id
            if pid is not None:
                fid = it.request.flight_id
                self._pair_of[fid] = pid
                self._pair_of[pid] = fid
                self._outbound_of_pair[fid] = self._outbound_of_pair[pid] = pid

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
        # fid -> the DISTINCT cells it claims. `_index_remove` and contention refresh are both
        # per-cell, so storing distinct cells (not one entry per (cell, span)) is what keeps this
        # index O(cells) rather than O(spans).
        self._cells_of: dict[int, set[Cell]] = {}
        self._contended: set[Cell] = set()
        self._contended_list: list[Cell] | None = None
        # Where an iteration's wall goes, split the only two ways that matter for choosing a repair
        # planner: SEARCH (what SIPP makes cheaper) vs LEDGER MAINTENANCE (what SIPP makes more
        # expensive, keeping a fourth subscribed structure on terminal legs). Two whole-run counters
        # read once at the end — deliberately NOT per-flight attribution, which is dominated by the
        # `perf_counter` calls themselves.
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

        Every keyword below changes what a repair is ALLOWED to do, so each must be forwarded or
        the worker silently runs a different algorithm than the coordinator believes it does — and
        ``verify.find_interflight_conflict`` checks 4D conflicts ONLY, so such a run still reports
        ``verified``.

        Parameters
        ------------
        - cfg (SimConfig): sim config, shared with the coordinator.
        - intents (list[OperationalIntent]): the incumbent to copy; accepted intents are committed
          to the fresh ledger.
        - static_terms (tuple): the (center, terminal) walls to re-register, so the worker measures
          the same world as the ruler.
        - unimpeded_cost (dict[int, float | None]): the broadcast ruler; a None entry means denied.
        - turnaround_s (float | None): arms ``try_repair``'s paired-leg precedence guard; without it
          a repair may land an outbound after its return has departed, or shed a return's hold until
          it lifts off before its own aircraft is back (a precedence break ``verify`` cannot see).
        - frozen_flight_ids (frozenset[int]): non-movable flights; omitting them lets destroy pick
          frozen flights while the membership assert still passes on the worker's own (wrong) set.
        - movable_uss_ids (frozenset[str] | None): USS movability filter, forwarded for the same
          reason as ``frozen_flight_ids``.
        - incremental_release (bool): the rebuild-path byte-parity reference (``--no-incremental``);
          hardcoding it would make that A/B inexpressible under parallelism.
        - kernel_log2_min (int | None): the repair planner's compiled-kernel size threshold.
        - record_envelope (bool): record per-plan read envelopes; needed only when multiple DROP
          workers can return stale repairs (SYNC and widths below two skip it).
        - window_bytes (int | None): per-planner bitmap budget; without it a worker uses a
          different resource policy from the coordinator.
        - repair_planner_name (str): which planner to build; a worker building A* while the
          coordinator believes it runs SIPP diverges silently.

        Return
        --------
        - output (LNSState): a standalone state over a private ledger, ready to ``try_repair``.
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
        """Index ``fid``'s blocked-cell claims (own-hub column interiors are skipped).

        A flight's own terminal-column interior is exempt from deconfliction, so those cells are
        not indexed (else same-hub flights would look mutually contended at their shared hub).

        Parameters
        ------------
        - fid (int): flight whose claims are added to the index.
        - volumes (Sequence): the flight's committed volumes to rasterize into blocked cells.
        - refresh (bool): when True, recompute contention for each newly claimed cell; pass False
          to batch a single contention sweep afterwards.

        Return
        --------
        - output (None): mutates ``self._cells_of``, ``self._claims`` and (when ``refresh``) the
          contention set.
        """
        rows = self._cells_of.get(fid)
        if rows is None:
            rows = self._cells_of[fid] = set()
        # A flight's own terminal-column interior is exempt from deconfliction — the occupancy
        # services drop those cells (HexOccupancyService.add_volume / CompiledHexOccupancy._add), so
        # A* never conflicts there (see context/figures/cell_blocking.png). Indexing them anyway
        # would make same-hub flights look mutually contended at their shared hub, and
        # the map destroy operator picks cells BY contention, spending neighborhoods on
        # hub interiors where no conflict can exist.
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
        """Drop fid's claims from the index and refresh each freed cell's contention."""
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
        """Recompute whether cell is contended (2+ owners) and update the contended set."""
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
        """The sorted movable flight ids — the destroy operators' selection pool."""
        return self._movable

    def is_movable(self, fid: int) -> bool:
        """Whether ``fid`` is movable (accepted, not frozen, and USS-allowed)."""
        return fid in self._movable_set

    def delay(self, fid: int) -> float:
        """Flight ``fid``'s delay: incumbent cost minus its unimpeded ruler, floored at 0."""
        return max(0.0, float(self.incumbent[fid].cost) - self._unimp_cost[fid])

    def visits(self, fid: int):
        """Cached per-step ``(step, cell)`` samples of ``fid``'s airborne centreline.

        Memoizes :meth:`_extract_visits`; the cache entry is dropped whenever the flight is
        re-planned.

        Parameters
        ------------
        - fid (int): flight id to sample.

        Return
        --------
        - output (list[tuple[int, Cell]]): airborne lateral samples (empty when the flight has no
          centreline).
        """
        cached = self._visits.get(fid)
        if cached is None:
            cached = self._visits[fid] = self._extract_visits(self.incumbent[fid])
        return cached

    def unimpeded_launch_step(self, fid: int) -> int:
        """The step ``fid`` would first go airborne with its ground hold removed.

        Parameters
        ------------
        - fid (int): flight id.

        Return
        --------
        - output (int): first airborne step minus the held ground-delay steps (0 when the flight
          has no visits).
        """
        vis = self.visits(fid)
        if not vis:
            return 0
        hold = int(round(float(self.incumbent[fid].ground_delay_s) / self.dt))
        return vis[0][0] - hold

    def owners_over(self, cell: Cell, s_lo: int, s_hi: int):
        """The flights whose claim on ``cell`` overlaps the step span ``[s_lo, s_hi]``."""
        return {f for a, b, f in self._claims.get(cell, ()) if a <= s_hi and b >= s_lo}

    def claim_span(self, cell: Cell) -> tuple[int, int]:
        """The step span covered by every claim on ``cell``.

        Parameters
        ------------
        - cell (Cell): a (q, r, level) cell that has at least one claim.

        Return
        --------
        - output (tuple[int, int]): the earliest claim start and latest claim end over the cell.
        """
        entries = self._claims[cell]
        return min(e[0] for e in entries), max(e[1] for e in entries)

    def contention_cells(self):
        """The contended cells (2+ owners), sorted; cached until the index changes."""
        if self._contended_list is None:
            self._contended_list = sorted(self._contended)
        return self._contended_list

    def _extract_visits(self, it: OperationalIntent) -> list[tuple[int, Cell]]:
        """Per-step ``(step, cell)`` samples of the centerline at a flight level — the airborne
        lateral path the random walk explores.

        Interpolates the centerline at each integer step and maps it to a hex cell, keeping only
        steps whose altitude is within 0.5 m of a flight level (climb/descend samples between
        levels are skipped).

        Parameters
        ------------
        - it (OperationalIntent): the flight whose centerline is sampled; an empty centerline
          yields ``[]``.

        Return
        --------
        - output (list[tuple[int, Cell]]): ordered ``(step, (q, r, level))`` samples, one per
          cruise-level timestep.
        """
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
        *,
        report_only: bool = False,
    ) -> RepairOutcome:
        """Release and PP-repair ``victims``; accept only a strict weighted-cost improvement.

        The whole destroy->repair->accept/revert transaction: release the victims, replan them in
        priority order, then adopt a strict improvement or rewind to the incumbent. Every
        non-adopting exit restores the ledger, so a rejected or failed repair leaves the schedule
        unchanged. See ``context/lns_plan.md`` §4.

        Parameters
        ------------
        - victims (Iterable[int]): movable flight ids to destroy and replan (sorted internally).
        - rng (np.random.Generator): source for the repair-order tie-break or permutation.
        - accept_epsilon (float): minimum weighted-cost drop required to accept.
        - order_mode (str): ``"premium"`` repairs the most-delayed first with random ties;
          ``"random"`` uses the paper's random order (for A/Bs). Any other value raises.
        - report_only (bool): when True, return an accepted candidate but restore the incumbent
          ledger without adopting it — parallel workers use this; only the coordinator commits.

        Return
        --------
        - output (RepairOutcome): the outcome; ``accepted``/``reason`` say what happened,
          and ``new_intents``/``envelopes`` are populated only on an accepted return.
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
        rec_env = bool(getattr(self.repair_planner, "record_envelope", False))
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
                if rec_env:
                    envelopes.append(self.repair_planner.last_envelope)

            if reason == "improved" and self._turnaround_s is not None:
                # One predicate over the PAIR, not two branches keyed by which leg moved. Whichever
                # leg this repair touched, the pair is re-scored the same way `verify` scores it, and
                # the partner is read from `new` when the same transaction moved it too — so the
                # verdict does not depend on repair order, and a pair with both legs repaired is
                # judged once, from both new plans.
                seen: set[int] = set()
                for fid in new:
                    out_fid = self._outbound_of_pair.get(fid)
                    if out_fid is None or out_fid in seen:
                        continue
                    seen.add(out_fid)
                    ret_fid = self._pair_of[out_fid]
                    outbound = new.get(out_fid) or self.incumbent.get(out_fid)
                    ret = new.get(ret_fid) or self.incumbent.get(ret_fid)
                    if outbound is None or ret is None:
                        continue          # partner denied or not in this state: nothing to preserve
                    short = pair_precedence_shortfall(outbound, ret, self._turnaround_s)
                    if short > self._pair_shortfall.get((out_fid, ret_fid), 0.0) + 1e-6:
                        reason = "anchor"
                        break

            cost_new = float(sum(it.cost for it in new.values())) if reason == "improved" else math.inf
            if reason == "improved" and cost_new < cost_old - accept_epsilon:
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

        Parameters
        ------------
        - victims (Sequence[int]): flight ids to release and re-commit from ``old``.
        - old (dict[int, OperationalIntent]): the pre-repair intents to restore each victim to.
        - cost_at_entry (float): the running total cost to restore when acceptance had begun.
        - applied (Sequence[int]): fids already moved onto the new schedule in memory; when
          non-empty, the incumbent, ``total_cost`` and the full claim index are restored.

        Return
        --------
        - output (None): mutates the ledger, ``self.total_cost``, ``self.incumbent`` and the claim
          index; raises ``RuntimeError`` if any victim cannot be re-committed.
        """
        t0 = time.perf_counter()
        self.ledger.release_many(victims)
        failures = []
        for fid in victims:
            try:
                self.ledger.commit(fid, old[fid].volumes)
            except BaseException as exc:            # noqa: BLE001 - re-raised below, after the rest
                failures.append((fid, exc))
        self.t_ledger_s += time.perf_counter() - t0    # the rewind is ledger work too, and is a
        #                                                big share of it on the rejection path
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
        """Adopt someone else's accepted repair: move the LEDGER and the in-memory views together.

        The replica-sync counterpart of ``try_repair``'s accept branch. The difference is the
        ledger: ``try_repair`` has already committed its own plans and only the in-memory views
        are behind, while a worker told "the incumbent moved" still holds the old volumes and must
        release them first. One ``commit`` per flight, because ``_absorb`` groups a flight's
        volumes by adjacent runs and needs them contiguous. Costs O(changed flights), not
        O(schedule) — which is what makes a parallel worker's private copy of the incumbent
        affordable at all.

        Parameters
        ------------
        - changes (dict[int, OperationalIntent]): fid -> its adopted intent, in the producer's PP
          priority order (see below); an empty dict is a no-op.

        Return
        --------
        - output (None): mutates the ledger and in-memory views in place; on any failed commit it
          rewinds the whole delta and re-raises.
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
