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

    if cfg.terminal_airspace_always_active:
        # Hand every A* planner the full hub set so it can permanently wall each terminal off from
        # foreign cruise traffic. A hub centre = the request endpoint that carries its terminal (a
        # delivery's origin / a return's dest). Normalize (id, capacity) tuple terminals to Terminal so
        # downstream register_static_terminal / terminal_cells get a real column radius.
        terms: dict = {}
        for ev in scenario.events:
            rq = ev.request
            for pt, t in ((rq.origin, rq.origin_terminal), (rq.dest, rq.dest_terminal)):
                term = as_terminal(t)
                if term is not None:
                    terms.setdefault(term.id, (pt, term))
        static_terms = list(terms.values())
        reached = False
        for u in usses.values():
            p = u.planner
            while p is not None:                        # descend refiners (.inner) AND opt/milp warm A*
                if hasattr(p, "static_terminals"):      # (.warm_planner) so every A* layer is reached
                    p.static_terminals = static_terms
                    reached = True
                p = getattr(p, "inner", None) or getattr(p, "warm_planner", None)
        if not reached:                                 # loud failure beats a silent wrong result: the
            raise ValueError(                           # demand filter already dropped foreign customers,
                f"terminal_airspace_always_active=True but planner {pname!r} exposes no A* layer to "
                "install static terminal walls (looked through .inner/.warm_planner) — foreign-column "
                "customers would be dropped with no compensating walls.")

    total = len(scenario.events)
    report = _resolve_progress(progress, total)
    intents: list[OperationalIntent] = []
    for done, ev in enumerate(scenario.events, 1):
        uss = usses.get(ev.request.uss_id, default_uss)
        intent = uss.handle_request(ev.request)
        intents.append(intent)
        if report:
            report(done, total, intent)

    verified = verify.find_interflight_conflict(intents, cfg) is None
    # Carry the planner that ACTUALLY flew: a planner_name= override must be reflected in the stored
    # config, or downstream metrics/aggregate (which key on cfg.planner — e.g. the altitude baseline)
    # and the reported planner label would describe cfg.planner, not the planner that ran.
    result_cfg = cfg if pname == cfg.planner else replace(cfg, planner=pname)
    return SimResult(config=result_cfg, intents=intents, ledger=ledger, verified=verified)
