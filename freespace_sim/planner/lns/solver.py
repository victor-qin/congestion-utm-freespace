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
    enforce_return_anchor: bool = False  # set when the baseline ran return_anchor="realized"
    incremental_release: bool = True     # O(victims) occupancy removal; False = rebuild path (parity ref)
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
    static_terms: tuple = (),
    turnaround_s: float | None = None,
) -> LNSResult:
    """Improve a committed schedule in place. ``ledger``/``intents`` are a completed run's
    (the ledger is mutated; the returned intents supersede the input list)."""
    t0 = time.monotonic()
    state = LNSState(
        cfg,
        ledger,
        intents,
        static_terms=static_terms,
        frozen_flight_ids=lns.frozen_flight_ids,
        movable_uss_ids=lns.movable_uss_ids,
        turnaround_s=turnaround_s,
        incremental_release=lns.incremental_release,
    )
    init_s = time.monotonic() - t0
    cost_before = state.total_cost

    tabu: set[int] = set()
    ops = {}
    if "agent" in lns.operators:
        ops["agent"] = lambda ctx, n: agent_based_neighborhood(ctx, n, tabu, lns.max_walks)
    if "map" in lns.operators:
        ops["map"] = lambda ctx, n: map_based_neighborhood(ctx, n, lns.map_max_cells)
    if "random" in lns.operators:
        ops["random"] = random_neighborhood
    unknown = [name for name in lns.operators if name not in ops]
    if unknown:
        raise ValueError(f"unknown LNS operators {unknown!r} (want subset of agent/map/random)")
    selector = AdaptiveSelector(tuple(lns.operators), lns.gamma)

    trajectory: list[dict] = []
    n_accepted = 0
    n_iter = 0
    # Every iteration's first plan() rebuilds the occupancy from the shrunk ledger — that is the
    # designed heal, so the reference path's per-shrink RuntimeWarning is pure noise here.
    warnings.filterwarnings("ignore", message="ReservationLedger shrank", append=True)
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
        out = state.try_repair(victims, rng_i, lns.accept_epsilon, order_mode=lns.repair_order)
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
            bad = verify.find_interflight_conflict(state.final_intents(), cfg, static_terminals=static_terms)
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


def run_lns_on_result(res, demand, lns: LNSConfig, *, return_anchor: str = "nominal") -> LNSResult:
    """Convenience entry over a ``sim.run`` result: derives the static terminals from the
    demand model and, when the baseline used realized return anchors, the turnaround the
    paired-return guard must respect."""
    static_terms = tuple(demand.terminals(res.config)) if demand is not None else ()
    turnaround_s = None
    if return_anchor == "realized" or lns.enforce_return_anchor:
        turnaround_s = float(getattr(demand, "turnaround_s", 0.0) or 0.0)
    return run_lns(
        res.config, res.ledger, res.intents, lns,
        static_terms=static_terms, turnaround_s=turnaround_s,
    )
