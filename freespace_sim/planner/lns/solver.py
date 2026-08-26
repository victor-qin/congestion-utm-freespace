"""Anytime MAPF-LNS driver (Li et al., IJCAI-21) seeded by a completed FCFS A* schedule.

The incumbent is a complete, ledger-feasible schedule at every instant: destroy/repair
transactions only replace it when the repaired neighborhood is conflict-free AND cheaper
(utilitarian, weighted-cost currency), so the cost trajectory is monotone non-increasing
and the run can be stopped after any iteration. The iteration budget — not wall clock —
is the reproducible knob; ``time_limit_s`` exists for cluster jobs but breaks
reproducibility across machines by design of the machines, not of this loop.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from dataclasses import dataclass

import numpy as np

from freespace_sim import verify
from freespace_sim.config import SimConfig
from freespace_sim.ledger import ReservationLedger
from freespace_sim.sim import demand_turnaround_s
from freespace_sim.planner.lns.neighborhood import (
    AdaptiveSelector,
    agent_based_neighborhood,
    map_based_neighborhood,
    random_neighborhood,
)
from freespace_sim.planner.lns.state import LNSState
from freespace_sim.types import OperationalIntent

log = logging.getLogger("freespace_sim.lns")


@dataclass
class LNSConfig:
    seed: int = 0
    max_iterations: int = 2000
    neighborhood_size: int = 8          # paper N in {2,4,8,16}; larger favors less-congested instances
    operators: tuple[str, ...] = ("agent", "map", "random")
    adaptive: bool = True               # ALNS roulette; False -> uniform random operator choice
    gamma: float = 0.01                 # ALNS reaction factor (paper value)
    accept_epsilon: float = 0.0         # required strict improvement in weighted seconds
    repair_order: str = "premium"       # PP priority: "premium" (most-delayed first) | "random" (paper)
    max_walks: int = 10                 # agent-based: walk restarts before giving up on size N
    map_max_cells: int = 4096           # map-based: BFS exploration bound
    frozen_flight_ids: frozenset = frozenset()      # never destroyed (USS-restriction hook)
    movable_uss_ids: frozenset | None = None        # None -> system operator may move every USS's intents
    incremental_release: bool = True     # O(victims) occupancy removal; False = rebuild path (parity ref)
    # Processes used for the unimpeded delay ruler at state build (None -> min(8, cpu-2), 1 -> in
    # process). Throughput only: the ruler's plans are independent, so the count cannot move a cost.
    unimpeded_workers: int | None = None
    time_limit_s: float | None = None
    verify_every: int = 0               # independent conflict replay every n accepted iterations (tests)
    log_every: int = 200


@dataclass
class LNSResult:
    intents: list[OperationalIntent]    # incumbent schedule, original request order
    trajectory: list[dict]              # one row per iteration (anytime curve)
    cost_before: float
    cost_after: float
    n_iterations: int
    n_accepted: int
    wall_s: float
    init_wall_s: float                  # state build incl. the unimpeded baseline pass
    weights: dict[str, float]
    verified: bool

    def summary(self) -> dict:
        return {
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "improvement": self.cost_before - self.cost_after,
            "improvement_pct": 100.0 * (self.cost_before - self.cost_after) / max(1e-9, self.cost_before),
            "n_iterations": self.n_iterations,
            "n_accepted": self.n_accepted,
            "wall_s": self.wall_s,
            "init_wall_s": self.init_wall_s,
            "weights": dict(self.weights),
            "verified": self.verified,
        }


def run_lns(
    cfg: SimConfig,
    ledger: ReservationLedger,
    intents: list[OperationalIntent],
    lns: LNSConfig,
    *,
    static_terms: tuple | None = None,
    turnaround_s: float | None = None,
) -> LNSResult:
    """Improve a committed schedule in place. ``ledger``/``intents`` are a completed run's
    (the ledger is mutated; the returned intents supersede the input list).

    ``static_terms`` defaults to the walls the LEDGER actually holds. Passing ``()`` explicitly means
    "a world with no always-active terminal airspace", which is a different claim: it makes the
    unimpeded baseline free of walls (so every delay premium — the ranking that picks victims and
    orders the repair — is inflated) and it makes the closing ``verify`` replay a world the schedule
    was never planned against, so ``verified`` can come back True for an infeasible schedule.
    ``turnaround_s=None`` likewise disables the paired-return guard; supply it whenever the baseline
    ran ``return_anchor="realized"``. ``run_lns_on_result`` derives both correctly."""
    # Validate the operator set BEFORE constructing LNSState: the constructor detaches the caller's
    # ledger subscribers irrecoverably and spends one A* plan per movable flight (minutes at scenario
    # scale), and an argument error must not cost either.
    known = {"agent", "map", "random"}
    unknown = [name for name in lns.operators if name not in known]
    if unknown:
        raise ValueError(f"unknown LNS operators {unknown!r} (want subset of agent/map/random)")
    if not lns.operators:   # else the first pick indexes an empty wheel (numpy: "high <= 0")
        raise ValueError("LNSConfig.operators is empty — need at least one of agent/map/random")
    if len(set(lns.operators)) != len(lns.operators):
        # `ops` would collapse the duplicate while AdaptiveSelector.names keeps it, silently handing
        # that operator a double share of the roulette and reporting a weights dict shorter than the
        # configuration — a run that is quietly not the experiment that was asked for.
        raise ValueError(f"duplicate LNS operators {list(lns.operators)!r} — each name at most once")

    t0 = time.monotonic()
    state = LNSState(
        cfg,
        ledger,
        intents,
        static_terms=ledger.static_terminals() if static_terms is None else static_terms,
        frozen_flight_ids=lns.frozen_flight_ids,
        movable_uss_ids=lns.movable_uss_ids,
        turnaround_s=turnaround_s,
        incremental_release=lns.incremental_release,
        unimpeded_workers=lns.unimpeded_workers,
    )
    static_terms = state.static_terms
    init_s = time.monotonic() - t0
    cost_before = state.total_cost

    tabu: set[int] = set()
    ops = {
        "agent": lambda ctx, n: agent_based_neighborhood(ctx, n, tabu, lns.max_walks),
        "map": lambda ctx, n: map_based_neighborhood(ctx, n, lns.map_max_cells),
        "random": random_neighborhood,
    }
    ops = {name: ops[name] for name in lns.operators}
    selector = AdaptiveSelector(tuple(lns.operators), lns.gamma)

    trajectory: list[dict] = []
    n_accepted = 0
    n_iter = 0
    try:
        # Shrink warnings are expected only in the reference rebuild path. Keep the filter scoped and
        # prepend it so stricter caller filters do not turn the expected warning into an exception.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="ReservationLedger shrank")
            for i in range(lns.max_iterations):
                if lns.time_limit_s is not None and time.monotonic() - t0 > lns.time_limit_s:
                    break
                n_iter = i + 1
                rng_i = np.random.default_rng(np.random.SeedSequence([lns.seed, i]))
                state.rng = rng_i
                if lns.adaptive:
                    name = selector.pick(rng_i)
                else:
                    name = lns.operators[int(rng_i.integers(len(lns.operators)))]
                victims = ops[name](state, lns.neighborhood_size)
                if not victims:
                    if lns.adaptive:
                        selector.update(name, 0.0)
                    trajectory.append(dict(
                        iter=i, op=name, n=0, victims=[], accepted=False, reason="empty",
                        cost_old=0.0, cost_new=0.0,
                        incumbent_cost=state.total_cost, wall_s=time.monotonic() - t0,
                    ))
                    continue
                out = state.try_repair(
                    victims, rng_i, lns.accept_epsilon, order_mode=lns.repair_order
                )
                if lns.adaptive:
                    selector.update(name, out.improvement)
                n_accepted += int(out.accepted)
                trajectory.append(dict(
                    iter=i, op=name, n=len(victims), victims=sorted(victims),
                    accepted=out.accepted, reason=out.reason,
                    cost_old=out.cost_old,
                    cost_new=None if math.isinf(out.cost_new) else out.cost_new,
                    incumbent_cost=state.total_cost, wall_s=time.monotonic() - t0,
                ))
                if out.accepted and lns.verify_every and n_accepted % lns.verify_every == 0:
                    bad = verify.find_interflight_conflict(
                        state.final_intents(), cfg, static_terminals=static_terms
                    )
                    if bad is not None:
                        raise AssertionError(f"LNS incumbent has an interflight conflict: {bad}")
                if lns.log_every and (i + 1) % lns.log_every == 0:
                    log.info(
                        "lns %d/%d: cost %.0f (%.2f%% below start), %d accepted, weights %s",
                        i + 1, lns.max_iterations, state.total_cost,
                        100.0 * (cost_before - state.total_cost) / max(1e-9, cost_before),
                        n_accepted, {k: round(v, 3) for k, v in selector.weights.items()},
                    )

        final = state.final_intents()
        bad = verify.find_interflight_conflict(final, cfg, static_terminals=static_terms)
        return LNSResult(
            intents=final,
            trajectory=trajectory,
            cost_before=cost_before,
            cost_after=state.total_cost,
            n_iterations=n_iter,
            n_accepted=n_accepted,
            wall_s=time.monotonic() - t0,
            init_wall_s=init_s,
            weights=dict(selector.weights),
            verified=bad is None,
        )
    except BaseException:
        log.exception("lns aborted; detaching repair-planner subscribers before propagating")
        raise
    finally:
        # End the ownership transfer even if repair or verification fails.
        ledger.detach_subscribers()


def run_lns_on_result(res, demand, lns: LNSConfig, *, return_anchor: str | None = None) -> LNSResult:
    """Convenience entry over a ``sim.run`` result: takes the static terminals and the return-anchor
    mode from the RESULT (what the baseline actually flew), and, when that mode was ``"realized"``,
    the turnaround the paired-return guard must respect from the demand model.

    Both are read off the result rather than re-derived, because both are silent when wrong:

    * ``ledger.static_terminals()`` is the set of permanent walls the run really filed. Re-deriving
      it as ``demand.terminals(cfg)`` invents walls whenever ``terminal_airspace_always_active`` is
      off (the unimpeded baseline then over-charges every flight, distorting delay premiums, and the
      final ``verify`` replays a world the schedule was never planned against), crashes outright on a
      demand model without a ``terminals`` method, and misses ``sim.run``'s scenario-collected
      fallback for those models.
    * ``res.return_anchor`` decides whether the paired-return guard runs at all. Defaulting it to
      ``"nominal"`` disabled the guard for exactly the runs that need it, with no error and no log
      line — the LNS would happily re-time an outbound past its return's departure. Pass
      ``return_anchor=`` only to assert the mode; disagreeing with the result is an error, not an
      override.
    """
    try:
        recorded = res.return_anchor
    except AttributeError:                     # never default: "nominal" is the value that DISARMS
        raise TypeError(                       # the guard, so guessing it is the unsafe direction
            f"{type(res).__name__} carries no return_anchor — run_lns_on_result needs the anchor mode "
            "the baseline actually flew. Pass a sim.run() SimResult, or call run_lns directly with an "
            "explicit turnaround_s.") from None
    if return_anchor is not None and return_anchor != recorded:
        raise ValueError(
            f"return_anchor={return_anchor!r} contradicts the baseline's {recorded!r} — the anchor mode "
            "is a property of the schedule being improved, not a knob of the improvement pass")
    turnaround_s = None
    if recorded == "realized":
        if demand is None:
            log.warning("lns: return_anchor='realized' without a demand model — assuming turnaround_s=0 "
                        "(the paired-return guard then only enforces release <= the return's departure)")
        turnaround_s = demand_turnaround_s(demand)
    return run_lns(
        res.config, res.ledger, res.intents, lns,
        static_terms=res.ledger.static_terminals(), turnaround_s=turnaround_s,
    )
