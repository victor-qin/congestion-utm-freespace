"""The simulator — FCFS event loop tying the strategic layer together.

Build the world (ledger, DSS, USSs), process demand events in FCFS order (each USS plans a
conflict-free reservation and commits it through the DSS), then verify the core invariant. v0
execution is perfect conformance, so there is no separate tactical step — the reserved centerline
*is* the flown path. The `ExecutionBackend` seam for BlueSky is noted for a later phase.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from . import verify
from .config import SimConfig
from .demand import DemandModel, UniformPoissonDemand
from .dss import DSS
from .ledger import ReservationLedger
from .mechanism import FCFSMechanism, Mechanism
from .planner import get_planner
from .progress import RollingRate
from .scenario import Scenario, scenario_from_requests
from .telemetry import TelemetryCollector, build_terminal_snapshot
from .types import FlightRequest, IntentStatus, OperationalIntent, as_terminal
from .uss import USS

log = logging.getLogger(__name__)

# Called after each flight is planned: (done, total, latest_intent). Return value ignored.
ProgressCallback = Callable[[int, int, OperationalIntent], None]


class ConsoleProgress:
    """A throttled, single-line progress reporter for long simulations.

    Prints at most every ``every_s`` seconds (and once at the end) to ``stream`` (stderr by default),
    showing flights done/total, running accepted/denied counts, elapsed wall time, the per-flight
    rate, and a linear ETA. Uses a carriage return so it updates in place.
    """

    def __init__(self, total: int, every_s: float = 2.0, stream=None, window: int = 100):
        self.total = total
        self.every_s = every_s
        self.stream = stream if stream is not None else sys.stderr
        self.t0 = time.monotonic()
        self.prev = self.t0              # wall clock at the previous flight, for per-flight deltas
        self.last = 0.0
        self.acc = 0
        self.den = 0
        self.rate = RollingRate(window)

    def __call__(self, done: int, total: int, intent: OperationalIntent) -> None:
        if intent.accepted:
            self.acc += 1
        elif intent.status is IntentStatus.REJECTED:
            self.den += 1
        now = time.monotonic()
        self.rate.add(now - self.prev)   # this flight's WALL time — accrued EVERY flight so the rolling
        self.prev = now                  # mean/ETA stay right even though we only print every every_s
        if done < total and now - self.last < self.every_s:
            return
        self.last = now
        elapsed = now - self.t0
        roll, eta = self.rate.roll_ms(), self.rate.eta_s(done, total)
        roll_str = f"{roll:.0f}ms" if roll is not None else "n/a"
        eta_str = f"{eta:.0f}s" if eta is not None else "n/a"
        end = "\n" if done >= total else ""
        # ETA rides the rolling per-flight time, so it re-forecasts through a slowdown instead of trusting
        # the whole-run average; avg (cumulative) is kept alongside so the two diverging shows saturation.
        print(f"\r  [{done:>4}/{total}] acc={self.acc} den={self.den}  elapsed={elapsed:5.0f}s  "
              f"wall/flight avg={self.rate.avg_ms(done):.0f}ms roll[{self.rate.window}]={roll_str}  "
              f"ETA {eta_str}   ",
              end=end, file=self.stream, flush=True)


def _resolve_progress(progress, total: int) -> ProgressCallback | None:
    """Map the ``progress`` arg to a callback: None/False → off, True → ConsoleProgress, else passthrough."""
    if not progress:
        return None
    if progress is True:
        return ConsoleProgress(total)
    return progress


class _MilestoneLog:
    """Discrete INFO status milestones through a run — independent of the live ``progress`` ticker.

    Two cadences, both observer-only (results are byte-identical):
      • every ``every_n`` planned flights (accepted or denied), and
      • a "recording" at each ``every_frac`` of the horizon, carried by the first flight filing AT or
        after the mark — events are FCFS-sorted by ``t_request`` so a single advancing cursor suffices,
        and a sparse stretch makes one flight carry several consecutive marks (one line each).
    Each line reports the flight id, the sim time it appears (``t_request``), elapsed wall clock, and
    running planned/accepted/denied counts. Emitted at INFO via ``logging``: ``experiments.run``
    configures INFO→stderr so every batch-script run shows them; bare library/test use has no handler
    and stays silent (the level check short-circuits, so the quiet path costs nothing)."""

    def __init__(self, total: int, horizon_s: float, every_n: int = 1000, every_frac: float = 0.05,
                 roll_window: int = 100):
        self.total = total
        self.every_n = every_n
        self.t0 = time.monotonic()
        self.acc = 0
        self.den = 0
        # Per-flight PLANNER time (``intent.solve_time_s``) as a cumulative avg + rolling mean + ETA — the
        # same `RollingRate` treatment as the live ConsoleProgress ticker (which feeds it wall time), so a
        # saturation slowdown pulls roll above avg here too instead of hiding in the whole-run average.
        self.plan = RollingRate(roll_window)
        n_marks = max(1, round(1.0 / every_frac))
        # k/n_marks division, NOT horizon*every_frac*k: 0.05 is not float-representable and the
        # product overshoots the true fraction for ~a third of (horizon, k) pairs (1.0*0.05*3 =
        # 0.15000000000000002), silently deferring a mark past a flight that files EXACTLY on it.
        self.marks = [horizon_s * k / n_marks for k in range(1, n_marks + 1)]
        self.pcts = [round(100.0 * k / n_marks) for k in range(1, n_marks + 1)]
        self.mi = 0                                     # next un-recorded horizon mark

    def __call__(self, done: int, req: FlightRequest, intent: OperationalIntent) -> None:
        if intent.accepted:
            self.acc += 1
        elif intent.status is IntentStatus.REJECTED:
            self.den += 1
        self.plan.add(intent.solve_time_s)
        wall = time.monotonic() - self.t0
        while self.mi < len(self.marks) and req.t_request >= self.marks[self.mi]:
            log.info("recording @%d%% horizon (mark %.0fs): flight=%d sim_t=%.1fs wall=%.1fs "
                     "planned=%d/%d acc=%d den=%d %s",
                     self.pcts[self.mi], self.marks[self.mi], req.flight_id, req.t_request, wall,
                     done, self.total, self.acc, self.den, self._perf(done))
            self.mi += 1
        if done % self.every_n == 0:
            log.info("planned %d/%d: flight=%d sim_t=%.1fs wall=%.1fs acc=%d den=%d %s",
                     done, self.total, req.flight_id, req.t_request, wall, self.acc, self.den,
                     self._perf(done))

    def _perf(self, done: int) -> str:
        """This milestone's plan-time readout — cumulative avg always, plus the rolling mean and ETA once
        the window fills (``n/a`` while warming up). Built from the shared :class:`~freespace_sim.progress.RollingRate`, so it
        mirrors the live ConsoleProgress ticker; it's solve-time based, so its ETA slightly undershoots the
        wall clock by the per-flight commit overhead the ticker's wall-based ETA does capture."""
        roll, eta = self.plan.roll_ms(), self.plan.eta_s(done, self.total)
        if roll is None:
            return f"solve/flight avg={self.plan.avg_ms(done):.0f}ms roll[{self.plan.window}]=n/a ETA=n/a"
        return (f"solve/flight avg={self.plan.avg_ms(done):.0f}ms "
                f"roll[{self.plan.window}]={roll:.0f}ms ETA={eta:.0f}s")


@dataclass
class SimResult:
    config: SimConfig
    intents: list[OperationalIntent]
    ledger: ReservationLedger
    verified: bool
    telemetry: TelemetryCollector | None = None   # observer-only congestion capture (default off)
    # Whole-schedule solver diagnostics (colgen), or None for per-flight planners. Carried on the
    # result rather than only logged because the intents cannot answer "did this solve converge":
    # a run that stopped at iteration 1 files a complete, feasible, ordinary-looking accepted set.
    # `runs.save_run` persists this as planner_stats.json.
    planner_stats: dict | None = None
    # The `return_anchor` mode this run flew ("nominal" | "realized"). Carried because it is a property
    # of the SCHEDULE that the intents cannot answer: under "realized" a paired return's departure was
    # anchored to its outbound's realized arrival, so any post-hoc re-timing of that outbound (LNS) must
    # respect the turnaround — and a re-timer that defaults to "nominal" would silently skip the guard.
    return_anchor: str = "nominal"

    @property
    def accepted(self) -> list[OperationalIntent]:
        return [i for i in self.intents if i.accepted]

    @property
    def denied(self) -> list[OperationalIntent]:
        return [i for i in self.intents if i.status == IntentStatus.REJECTED]

    def summary(self) -> dict:
        from collections import Counter

        acc = self.accepted
        delays = [i.ground_delay_s for i in acc]
        detours = [i.air_detour_m for i in acc]
        reasons = Counter(i.denial_reason.value for i in self.denied)
        return {
            "n_requests": len(self.intents),
            "n_accepted": len(acc),
            "n_denied": len(self.denied),
            "denial_rate": len(self.denied) / max(1, len(self.intents)),
            # split real congestion (budget) from compute artifact (search) — see DenialReason
            "denials_by_reason": dict(reasons),
            "mean_ground_delay_s": float(np.mean(delays)) if delays else 0.0,
            "max_ground_delay_s": float(np.max(delays)) if delays else 0.0,
            "mean_air_detour_m": float(np.mean(detours)) if detours else 0.0,
            "verified": self.verified,
        }


def _wall_aware(planner) -> bool:
    """True if ``planner``'s committed corridor is wall-aware under always-active terminal airspace:
    it TAGS its terminal columns and gates pad capacity itself (A*, or any planner declaring
    ``plans_terminal_airspace`` — the MILP family), or it reaches such a planner through its
    ``inner``/``warm_planner`` chain and so rebuilds or falls back to a tagged intent. Walks the chain
    (the A* shortcut variants → inner, astar_milp → warm_planner,
    astar_milp_shortcut → both). Used only to gate
    ``terminal_airspace_always_active`` (see ``run``): tagged columns are exempt from their own hub's
    permanent wall, whereas a planner that builds untagged near-hub columns would collide with it."""
    from .planner.astar import AStarPlanner
    seen: set = set()
    stack = [planner]
    while stack:
        p = stack.pop()
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        if isinstance(p, AStarPlanner) or getattr(p, "plans_terminal_airspace", False):
            return True
        stack.extend((getattr(p, "inner", None), getattr(p, "warm_planner", None)))
    return False


def _astar_planners(planner) -> list:
    """Every ``AStarPlanner`` reachable from ``planner`` via the inner/warm_planner chain — so telemetry
    attaches to the A* inside any shortcut refiner or a warm-start wrapper (astar_milp, …), not just
    a bare top-level planner."""
    from .planner.astar import AStarPlanner
    out, seen, stack = [], set(), [planner]
    while stack:
        p = stack.pop()
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        if isinstance(p, AStarPlanner):
            out.append(p)
        stack.extend((getattr(p, "inner", None), getattr(p, "warm_planner", None)))
    return out


RETURN_ANCHORS = ("nominal", "realized")


def demand_turnaround_s(demand) -> float:
    """The ground turnaround a demand model budgeted between an outbound and its paired return.

    One reader, because two would be worse than none: the realized-anchor coupling in :func:`run` uses
    it to place each return, and the LNS paired-return guard uses it to decide whether a re-timed
    outbound still lands in time for that return. If those two disagreed, the guard would either reject
    valid repairs or — the dangerous direction — admit an outbound that lands after its return has
    already departed, with no error and no log line. Models without the attribute budget nothing."""
    return float(getattr(demand, "turnaround_s", 0.0) or 0.0)


def realized_release_s(intent: OperationalIntent) -> float | None:
    """When an accepted flight's landing column clears (touchdown + pad dwell) — the earliest its
    aircraft can leave again, before turnaround. That is the destination cylinder's ``t_end``, hence
    the largest the intent holds. ``None`` if nothing landed: denied, or no volumes reserved.

    NOT ``centerline[-1]``: the corridor stops at the column's EDGE at cruise altitude because the
    descent inside is flown but unreserved (``astar._build``), so the last waypoint precedes touchdown
    by ``climb_time_to(z_land)`` — 16.7 s at the density scenarios' 100 m. Anchoring returns there
    launched them mid-descent, and the shared pad cylinder billed the overlap back as ground delay.
    """
    if not intent.accepted or not intent.volumes:
        return None
    return float(max(v.t_end for v in intent.volumes))


def run(
    cfg: SimConfig,
    *,
    scenario: Scenario | None = None,
    requests: list[FlightRequest] | None = None,
    demand: DemandModel | None = None,
    planner_name: str | None = None,
    planner_params=None,
    mechanism: Mechanism | None = None,
    progress: bool | ProgressCallback | None = None,
    telemetry: bool | TelemetryCollector = False,
    parallel=None,
    return_anchor: str = "nominal",
    static_terminals: list | None = None,
) -> SimResult:
    """Run one strategic-layer simulation. Provide a scenario, an explicit request list, a `demand`
    model, or none (a default `UniformPoissonDemand` is then generated from `cfg`).

    ``static_terminals`` overrides the walled-hub set derived below, and exists because that
    derivation is NOT a function of the requests alone: with a `demand` model it is every
    PLACED hub, and without one it is only the hubs some request actually touches. A caller
    re-running the same flights through a second planner therefore gets a DIFFERENT airspace
    unless it says otherwise -- measured on `density_faa_wing_zipline` truncated to 600 s,
    182 placed hubs against 180 flight-carrying ones, so two hubs are solid for one run and
    open for the other. Pass the set the first run used to keep the two comparable.

    ``progress`` gives live feedback through long runs: ``True`` prints a throttled status line
    (done/total, accepted/denied, elapsed, ETA); a callable is invoked as ``progress(done, total,
    intent)`` after each flight; ``None``/``False`` (default) stays silent. Independent of it,
    :class:`_MilestoneLog` emits INFO status milestones (every 1000 planned flights + each 5% of the
    horizon) via ``logging`` — visible when the host configures logging (``experiments.run`` does),
    silent otherwise.

    ``telemetry`` (default off → byte-identical to today) attaches an observer-only
    :class:`~freespace_sim.telemetry.TelemetryCollector` capturing the non-recoverable congestion streams
    (filed-but-rejected corridors, `conflict_filed` culprits, per-hub metadata) onto ``SimResult.telemetry``
    for `save_run` to persist. Pass ``True`` or a preexisting collector.

    ``planner_params`` configures the selected planner where it takes a configuration object —
    today only ``colgen`` (a :class:`~freespace_sim.planner.colgen.ColGenParams`, carrying the solver
    backend, iteration cap, whole-solve time budget and objective). ``None`` keeps that planner's
    own defaults.

    ``parallel`` (default off → the serial FCFS loop, byte-identical to today) runs the speculative
    worker-pool sim (issue #8 Track A): a :class:`~freespace_sim.parallel.ParallelConfig`, or an int
    as an ``n_workers`` shorthand. ``mode="exact"`` (default) is byte-identical to the serial run;
    ``mode="relaxed"`` is a documented FCFS-class relaxation. Needs an envelope-recording planner
    (``astar``/``astar_ref``/``astar_shortcut``/``astar_heading_shortcut``/
    ``astar_batched_shortcut``). Composes with ``telemetry`` (worker streams are
    merged in commit order).

    ``return_anchor`` decides what a round-trip return's desired departure waits on:

    - ``"nominal"`` (default → byte-identical to today) keeps whatever the demand model set. Demand is
      materialized before anything is planned, so that can only ever be a straight-line, undelayed
      estimate of when the outbound lands — under congestion it schedules the return before its
      aircraft is back.
    - ``"realized"`` re-anchors each return to ``realized_release_s(outbound) + turnaround`` — the
      arrival that actually happened. Exact and free: FCFS already plans the outbound first (a paired
      return shares its filing time and takes the next flight_id), so the arrival is always in hand.
      Filing times never move, so FCFS order and the monotonic-``t_request`` eviction are untouched.
      A return whose outbound was DENIED keeps its nominal anchor — dropping it would make the flight
      set depend on congestion and break paired comparisons across runs. The return still pays the
      ordinary pad-reuse separation on top (~8 s: ASTM buffer + ``dt`` rounding), as any flight taking
      over that pad would; the anchor says when the aircraft is ready, not who gets the pad.

    ``turnaround_s`` comes from the demand model, so the realized anchor uses exactly the turnaround
    the nominal one budgeted for.
    """
    if return_anchor not in RETURN_ANCHORS:
        raise ValueError(f"unknown return_anchor {return_anchor!r} (want one of {RETURN_ANCHORS})")
    if return_anchor == "realized" and parallel is not None:
        # A worker speculating on the return would read a t_departure its outbound has not fixed yet,
        # and exact mode could not catch it: the envelope records LEDGER reads, and the stale value is
        # request data, so the speculation would be accepted and silently diverge from sequential.
        raise ValueError(
            "return_anchor='realized' needs the sequential loop: it re-anchors each return to its "
            "outbound's committed arrival, which a speculative worker may not have yet — and the "
            "exact-mode envelope check cannot detect that (it tracks ledger reads, not request "
            "fields). Run with parallel=None, or use return_anchor='nominal'.")
    if scenario is None:
        if requests is None:
            model = demand if demand is not None else UniformPoissonDemand()
            requests = model.generate(cfg, np.random.default_rng(cfg.seed))
        scenario = scenario_from_requests(requests)

    ledger = ReservationLedger(cfg)
    dss = DSS(ledger=ledger, mechanism=mechanism or FCFSMechanism())
    pname = planner_name or cfg.planner
    usses = {uid: USS(uid, dss, cfg, get_planner(pname, planner_params)) for uid in scenario.uss_ids}
    default_uss = next(iter(usses.values()))

    static_terms: list = []                              # (center, term) per walled hub; [] unless always-active
    if cfg.terminal_airspace_always_active:
        # Wall EVERY placed hub's terminal off from foreign cruise traffic for the whole horizon. Prefer
        # the demand model's FULL placed-hub set (permanent infrastructure — a vertiport is walled even
        # when it draws no request this horizon, matching the demand foreign-column filter which drops
        # against ALL placed hubs); fall back to the flight-carrying hubs from the scenario otherwise.
        if static_terminals is not None:
            # An explicit set wins over both branches below: the caller already knows which
            # walls this run must share with another one.  See the argument's docstring.
            static_terms = list(static_terminals)
        elif demand is not None and hasattr(demand, "terminals"):
            static_terms = list(demand.terminals(cfg))
        else:
            terms: dict = {}
            for ev in scenario.events:
                rq = ev.request
                for pt, t in ((rq.origin, rq.origin_terminal), (rq.dest, rq.dest_terminal)):
                    term = as_terminal(t)
                    if term is not None:
                        terms.setdefault(term.id, (pt, term))
            static_terms = list(terms.values())
        # File each hub's terminal airspace as a PERMANENT ledger volume (whole horizon). any_conflict /
        # verify / the ledger-only refiners now ALL see the walls, and the A* occupancy services derive their
        # discrete routing walls from the ledger (subscribe_static).
        for center, term in static_terms:
            ledger.register_static_terminal(center, term)
        # The walls are per-hub TAGGED CylinderSpecs; a flight's own-hub column is exempt from its own hub's
        # wall only if it too is tagged (conflict.volumes_conflict same-tid+cylinder). Wall-aware planners:
        #   • astar and all shortcut variants TAG their terminal columns
        #     (astar._build / shortcut pass the terminal id),
        #     so they refine fully under always-active.
        #   • the MILP family (plans_terminal_airspace) folds its corners to the column edge, TAGS the rebuilt
        #     columns/near-hub boxes, and gates pad capacity through its own TerminalCapacity — tagging is safe
        #     for it for the same reason it is for astar: the capacity authority serialises the pad, so the
        #     same-tid exemption cannot pull a flight into a same-hub pad overlap.
        #   • refiners/warm-start wrappers qualify through their chain (they rebuild or fall back to a tagged
        #     intent).
        # A planner that is none of these (bare straight / decoupled) has no wall-respecting geometry, so it
        # would commit untagged near-hub columns that collide with the wall (or ignore it) and deny / mis-plan
        # every hub flight — refused LOUDLY below rather than allowed to silently mis-plan.
        for u in usses.values():
            if not _wall_aware(u.planner):
                raise ValueError(
                    f"terminal_airspace_always_active=True needs a wall-aware planner (tagged terminal "
                    f"columns — A*-reaching or terminal-aware MILP), but {pname!r} is neither and would "
                    f"commit untagged near-hub columns that collide with the wall and deny every hub flight.")

    collector: TelemetryCollector | None = None
    if telemetry:
        collector = telemetry if isinstance(telemetry, TelemetryCollector) else TelemetryCollector()
        collector.terminals = build_terminal_snapshot(cfg, demand, scenario.events)
        for u in usses.values():                     # observer-only; reaches A* inside wrapper planners too
            for p in _astar_planners(u.planner):
                p._tele = collector

    total = len(scenario.events)
    planner_stats: dict | None = None
    report = _resolve_progress(progress, total)
    status = _MilestoneLog(total, cfg.horizon_s)        # INFO milestones; silent without a log handler
    batch_planners = [
        u.planner for u in usses.values() if getattr(u.planner, "plans_whole_schedule", False)
    ]
    if parallel is not None:
        from .parallel import PARALLEL_PLANNERS, ParallelConfig, run_parallel

        pcfg = parallel if isinstance(parallel, ParallelConfig) else ParallelConfig(n_workers=int(parallel))
        if pname not in PARALLEL_PLANNERS:
            raise ValueError(
                f"parallel mode needs an envelope-recording planner {PARALLEL_PLANNERS}, got {pname!r} "
                f"(the MILP/opt refiners optimize outside any recorded read set — not supported in v1).")
        # telemetry: workers capture per-flight on_deny rows and the coordinator merges them in
        # commit order into `collector`; serial replans write into it directly.
        intents = run_parallel(scenario, cfg, pcfg, ledger, dss, pname, static_terms, status, report,
                               collector=collector)
    elif batch_planners:
        from .planner.colgen import run_batch

        if return_anchor == "realized":
            # A batch solve never enters the FCFS loop the coupling lives in, so the flag would do
            # NOTHING — and since return_anchor is absent from index.parquet, the run would look
            # coupled on disk. Refuse rather than no-op.
            raise ValueError(
                f"return_anchor='realized' is not implemented for whole-schedule planners: {pname!r} "
                "solves every flight at once, so there is no moment at which an outbound has "
                "committed and its return has not, and the per-flight coupling loop never runs. Use "
                "a per-flight planner (astar*/milp), or return_anchor='nominal'.")
        # A whole-schedule planner solves for every flight at once, so it cannot share a
        # run with per-flight planners: the FCFS loop below would file some flights against
        # a ledger the batch solve already reserved against.
        if len(batch_planners) != len(usses):
            raise ValueError("whole-schedule and per-flight planners cannot share one simulation")
        params = batch_planners[0].params
        if any(planner.params != params for planner in batch_planners[1:]):
            raise ValueError("all whole-schedule planners must use identical parameters")
        intents, planner_stats = run_batch(
            scenario,
            cfg,
            ledger,
            dss,
            static_terms,
            status,
            report,
            collector,
            params=params,
        )
    else:
        intents = []
        # Round-trip coupling. `anchors[outbound_id]` = when that outbound's landing column cleared;
        # its return pops it just before being planned. FCFS plans the outbound first, so the entry is
        # always in hand, and popping keeps the dict near one entry (a paired return is the next event).
        couple = return_anchor == "realized"
        turnaround_s = 0.0
        if couple:
            # The turnaround has to match the one the NOMINAL anchor budgeted for, and only the demand
            # model knows it. Without a model there is no way to recover it, and defaulting to 0 would
            # quietly shorten every turnaround — so name the assumption instead of absorbing it.
            if demand is None:
                log.warning("return_anchor='realized' without a demand model: assuming turnaround_s=0. "
                            "Pass demand= (alongside requests=) so the realized anchor uses the same "
                            "turnaround the requests were generated with.")
            else:
                turnaround_s = demand_turnaround_s(demand)
        awaited = ({ev.request.paired_outbound_id for ev in scenario.events} - {None}) if couple else set()
        # Only HubRadiusDemand links its legs, so asking for the realized anchor anywhere else is a
        # no-op. Say so: silently doing nothing is exactly the failure this option exists to prevent.
        if couple and not awaited:
            log.warning("return_anchor='realized' but no request carries paired_outbound_id — nothing "
                        "to re-anchor. Only the hub_radius demand model emits linked round-trip legs.")
        elif couple:
            # A link is only usable if its outbound is planned FIRST. Both shipped return modes
            # guarantee that (a paired return shares its outbound's filing time and takes the next
            # flight_id; a legacy one files strictly later), but a hand-built request list can point at
            # a flight that is absent, or that FCFS orders after the return — in which case that leg
            # silently keeps its nominal anchor. Check once, up front, rather than let it pass quietly.
            pos = {ev.request.flight_id: k for k, ev in enumerate(scenario.events)}
            unusable = sum(1 for k, ev in enumerate(scenario.events)
                           if ev.request.paired_outbound_id is not None
                           and pos.get(ev.request.paired_outbound_id, len(pos)) >= k)
            if unusable:
                log.warning(
                    "return_anchor='realized': %d/%d linked return(s) name an outbound that is absent "
                    "from the scenario or that FCFS orders no earlier than the return itself — their "
                    "arrival is not known in time, so they keep the nominal anchor.",
                    unusable, len(awaited))
        anchors: dict[int, float] = {}
        for done, ev in enumerate(scenario.events, 1):
            req = ev.request
            if couple and req.paired_outbound_id is not None:
                released = anchors.pop(req.paired_outbound_id, None)
                if released is not None:               # None ⇒ outbound denied; keep the nominal anchor
                    # `replace`, NOT in-place: `requests` may be caller-owned, and mutating it leaks the
                    # coupled departures into any later run over the same list — corrupting exactly the
                    # anchor A/B this option invites. It also re-validates t_departure >= t_request via
                    # __post_init__, which the max() keeps satisfied (unreachable for both shipped return
                    # modes, but a future one that filed later would otherwise violate it silently).
                    # Only turnaround is added: `released` already includes the pad dwell.
                    req = replace(req, t_departure=max(req.t_request, released + turnaround_s))
            uss = usses.get(req.uss_id, default_uss)
            intent = uss.handle_request(req)
            if req.flight_id in awaited:
                free_at = realized_release_s(intent)
                if free_at is not None:
                    anchors[req.flight_id] = free_at
            intents.append(intent)
            status(done, req, intent)
            if report:
                report(done, total, intent)

    verified = verify.find_interflight_conflict(intents, cfg, static_terminals=static_terms) is None
    # Carry the planner that ACTUALLY flew: a planner_name= override must be reflected in the stored
    # config, or downstream metrics/aggregate (which key on cfg.planner — e.g. the altitude baseline)
    # and the reported planner label would describe cfg.planner, not the planner that ran.
    result_cfg = cfg if pname == cfg.planner else replace(cfg, planner=pname)
    return SimResult(config=result_cfg, intents=intents, ledger=ledger, verified=verified,
                     telemetry=collector, planner_stats=planner_stats, return_anchor=return_anchor)
