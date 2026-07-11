"""The simulator — FCFS event loop tying the strategic layer together.

Build the world (ledger, DSS, USSs), process demand events in FCFS order (each USS plans a
conflict-free reservation and commits it through the DSS), then verify the core invariant. v0
execution is perfect conformance, so there is no separate tactical step — the reserved centerline
*is* the flown path. The `ExecutionBackend` seam for BlueSky is noted for a later phase.
"""

from __future__ import annotations

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
from .scenario import Scenario, scenario_from_requests
from .types import FlightRequest, IntentStatus, OperationalIntent, as_terminal
from .uss import USS

# Called after each flight is planned: (done, total, latest_intent). Return value ignored.
ProgressCallback = Callable[[int, int, OperationalIntent], None]


class ConsoleProgress:
    """A throttled, single-line progress reporter for long simulations.

    Prints at most every ``every_s`` seconds (and once at the end) to ``stream`` (stderr by default),
    showing flights done/total, running accepted/denied counts, elapsed wall time, the per-flight
    rate, and a linear ETA. Uses a carriage return so it updates in place.
    """

    def __init__(self, total: int, every_s: float = 2.0, stream=None):
        self.total = total
        self.every_s = every_s
        self.stream = stream if stream is not None else sys.stderr
        self.t0 = time.monotonic()
        self.last = 0.0
        self.acc = 0
        self.den = 0

    def __call__(self, done: int, total: int, intent: OperationalIntent) -> None:
        if intent.accepted:
            self.acc += 1
        elif intent.status is IntentStatus.REJECTED:
            self.den += 1
        now = time.monotonic()
        if done < total and now - self.last < self.every_s:
            return
        self.last = now
        elapsed = now - self.t0
        rate = elapsed / max(done, 1)
        eta = rate * (total - done)
        end = "\n" if done >= total else ""
        print(f"\r  [{done:>4}/{total}] acc={self.acc} den={self.den}  "
              f"elapsed={elapsed:5.0f}s  {rate * 1000:5.0f}ms/flight  ETA {eta:4.0f}s   ",
              end=end, file=self.stream, flush=True)


def _resolve_progress(progress, total: int) -> ProgressCallback | None:
    """Map the ``progress`` arg to a callback: None/False → off, True → ConsoleProgress, else passthrough."""
    if not progress:
        return None
    if progress is True:
        return ConsoleProgress(total)
    return progress


@dataclass
class SimResult:
    config: SimConfig
    intents: list[OperationalIntent]
    ledger: ReservationLedger
    verified: bool

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


def _reaches_astar(planner) -> bool:
    """True if ``planner``'s committed corridor originates from A* — directly, or via an inner/warm-start
    A* whose (terminal-TAGGED) intent it rebuilds or falls back to. Walks the ``inner``/``warm_planner``
    chain (astar_shortcut → inner, opt_astar/astar_milp → warm_planner, astar_milp_shortcut → both). Used
    only to gate ``terminal_airspace_always_active`` (see ``run``): A* tags its terminal columns so they are
    exempt from their own hub's permanent wall, whereas a planner that builds untagged near-hub columns
    would collide with it."""
    from .planner.astar import AStarPlanner
    seen: set = set()
    stack = [planner]
    while stack:
        p = stack.pop()
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        if isinstance(p, AStarPlanner):
            return True
        stack.extend((getattr(p, "inner", None), getattr(p, "warm_planner", None)))
    return False


def run(
    cfg: SimConfig,
    *,
    scenario: Scenario | None = None,
    requests: list[FlightRequest] | None = None,
    demand: DemandModel | None = None,
    planner_name: str | None = None,
    mechanism: Mechanism | None = None,
    progress: bool | ProgressCallback | None = None,
) -> SimResult:
    """Run one strategic-layer simulation. Provide a scenario, an explicit request list, a `demand`
    model, or none (a default `UniformPoissonDemand` is then generated from `cfg`).

    ``progress`` gives live feedback through long runs: ``True`` prints a throttled status line
    (done/total, accepted/denied, elapsed, ETA); a callable is invoked as ``progress(done, total,
    intent)`` after each flight; ``None``/``False`` (default) stays silent.
    """
    if scenario is None:
        if requests is None:
            model = demand if demand is not None else UniformPoissonDemand()
            requests = model.generate(cfg, np.random.default_rng(cfg.seed))
        scenario = scenario_from_requests(requests)

    ledger = ReservationLedger(cfg)
    dss = DSS(ledger=ledger, mechanism=mechanism or FCFSMechanism())
    pname = planner_name or cfg.planner
    usses = {uid: USS(uid, dss, cfg, get_planner(pname)) for uid in scenario.uss_ids}
    default_uss = next(iter(usses.values()))

    static_terms: list = []                              # (center, term) per walled hub; [] unless always-active
    if cfg.terminal_airspace_always_active:
        # Wall EVERY placed hub's terminal off from foreign cruise traffic for the whole horizon. Prefer
        # the demand model's FULL placed-hub set (permanent infrastructure — a vertiport is walled even
        # when it draws no request this horizon, matching the demand foreign-column filter which drops
        # against ALL placed hubs); fall back to the flight-carrying hubs from the scenario otherwise.
        if demand is not None and hasattr(demand, "terminals"):
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
        # The walls are per-hub TAGGED CylinderSpecs, so a flight's own-hub column must also be tagged to be
        # exempt from its own hub's wall (conflict.volumes_conflict same-tid+cylinder). A*-based planners tag
        # their terminal columns (astar, astar_shortcut, opt_astar, astar_milp, ...); a planner that builds
        # UNTAGGED near-hub columns (bare rrt / opt / milp / straight / lazy) would collide with its OWN hub's
        # wall and deny every hub flight. This lifts the old bare-'astar'-only ban to allow A*-based refiners,
        # but still refuses untagged planners LOUDLY rather than let them silently mis-plan.
        for u in usses.values():
            if not _reaches_astar(u.planner):
                raise ValueError(
                    f"terminal_airspace_always_active=True needs an A*-based planner (its terminal columns "
                    f"are tagged, hence exempt from their own hub's permanent wall), but {pname!r} builds "
                    f"untagged near-hub columns that would collide with the wall and deny every hub flight.")

    total = len(scenario.events)
    report = _resolve_progress(progress, total)
    intents: list[OperationalIntent] = []
    for done, ev in enumerate(scenario.events, 1):
        uss = usses.get(ev.request.uss_id, default_uss)
        intent = uss.handle_request(ev.request)
        intents.append(intent)
        if report:
            report(done, total, intent)

    verified = verify.find_interflight_conflict(intents, cfg, static_terminals=static_terms) is None
    # Carry the planner that ACTUALLY flew: a planner_name= override must be reflected in the stored
    # config, or downstream metrics/aggregate (which key on cfg.planner — e.g. the altitude baseline)
    # and the reported planner label would describe cfg.planner, not the planner that ran.
    result_cfg = cfg if pname == cfg.planner else replace(cfg, planner=pname)
    return SimResult(config=result_cfg, intents=intents, ledger=ledger, verified=verified)
