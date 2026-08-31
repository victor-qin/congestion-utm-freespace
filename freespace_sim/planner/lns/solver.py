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
import operator
import os
import time
import warnings
from dataclasses import dataclass, field, replace
from numbers import Real

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


@dataclass(kw_only=True)
class LNSConfig:
    seed: int = 0
    max_iterations: int = 2000           # non-negative task budget
    neighborhood_size: int = 8          # paper N in {2,4,8,16}; larger favors less-congested instances
    operators: tuple[str, ...] = ("agent", "map", "random")
    adaptive: bool = True               # ALNS roulette; False -> uniform random operator choice
    gamma: float = 0.01                 # ALNS reaction factor (paper value)
    accept_epsilon: float = 0.0         # non-negative required improvement; +inf rejects all
    repair_order: str = "premium"       # PP priority: "premium" (most-delayed first) | "random" (paper)
    max_walks: int = 10                 # agent-based: walk restarts before giving up on size N
    map_max_cells: int = 4096           # map-based: BFS exploration bound
    frozen_flight_ids: frozenset = frozenset()      # never destroyed (USS-restriction hook)
    movable_uss_ids: frozenset | None = None        # None -> system operator may move every USS's intents
    incremental_release: bool = True     # O(victims) occupancy removal; False = rebuild path (parity ref)
    # Processes used for the unimpeded delay ruler at state build. The library default is in-process:
    # implicit `spawn` can re-execute an unguarded caller's top-level code. Guarded applications may
    # explicitly pass None for min(8, cpu-2) automatic parallelism.
    unimpeded_workers: int | None = 1
    # Byte cap on the repair planner's per-plan dense occupancy window (`astar.window`); None keeps
    # AStarPlanner's default, 0 turns it off. ANSWER-NEUTRAL — the window caches the interval pools
    # `_blocked` would otherwise walk, and anything outside it takes that walk — so this is a pure
    # speed knob and the byte-parity gates hold at any value. Exposed because the LNS loop is the
    # window's hardest case (destroy/repair re-fragments the same congested cells thousands of
    # times, which is exactly the list-walk growth `analysis/prof_ledger_scaling.py` measures), so
    # the A/B has to be runnable end to end: `analysis/ab_dense_window.py`.
    window_bytes: int | None = None
    time_limit_s: float | None = None
    verify_every: int = 0               # independent conflict replay every n accepted iterations (tests)
    log_every: int = 200
    # Parallel destroy/repair (DROP-LNS). The conservative default stays in-process because the
    # crossover is instance-dependent and every search worker holds a full schedule replica.
    # `unimpeded_workers` remains a separate, parent-only knob for the one-off delay ruler.
    search_workers: int = 1
    parallel_mode: str = "drop"          # "drop": async throughput | "sync": deterministic barrier
    worker_kernel_log2: int | None = None  # non-negative AStarPlanner.kernel_log2_min; oversized
    #                                        kernel arrays were measured to slow CONCURRENT plans
    #                                        ~1.75x at 8 workers while a lone worker matched serial


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
    # --- parallel only; defaulted so every existing construction site is untouched ---
    search_workers: int = 1
    parallel_mode: str = "sequential"
    auc: float = 0.0        # paper §5.2 AUC: best-known cost integrated over WALL CLOCK
    pool_spawn_s: float = 0.0
    parallel_stats: dict = field(default_factory=dict)   # mode counters + dirty/accept rates

    @property
    def npo(self) -> int:
        """Paper §5.2 NPO*: destroy/repair operations actually run."""
        return self.n_iterations

    def summary(self) -> dict:
        return {
            "search_workers": self.search_workers,
            "parallel_stats": dict(self.parallel_stats),
            "parallel_mode": self.parallel_mode,
            "npo": self.npo,
            "auc": self.auc,
            "pool_spawn_s": self.pool_spawn_s,
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


def _integer_config_value(name: str, value, *, minimum: int) -> int:
    """Normalize an integer config value without truncating floats or parsing strings."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"LNSConfig.{name} must be an int >= {minimum}, got {value!r}")
    try:
        normalized = operator.index(value)
    except TypeError:
        raise ValueError(
            f"LNSConfig.{name} must be an int >= {minimum}, got {value!r}") from None
    if normalized < minimum:
        raise ValueError(f"LNSConfig.{name} must be an int >= {minimum}, got {value!r}")
    return int(normalized)


def _float_config_value(
    name: str,
    value,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_positive_infinity: bool = False,
) -> float:
    """Normalize a bounded real config value without accepting booleans or NaNs."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"LNSConfig.{name} must be a real number, got {value!r}")
    normalized = float(value)
    invalid_infinity = math.isinf(normalized) and not (
        allow_positive_infinity and normalized > 0.0
    )
    if (math.isnan(normalized) or invalid_infinity
            or (minimum is not None and normalized < minimum)
            or (maximum is not None and normalized > maximum)):
        bounds = (
            f" between {minimum} and {maximum}" if maximum is not None
            else f" >= {minimum}" if minimum is not None
            else ""
        )
        raise ValueError(f"LNSConfig.{name} must be a real number{bounds}, got {value!r}")
    return normalized


def _boolean_config_value(name: str, value) -> bool:
    """Normalize Python/NumPy booleans without treating arbitrary truthy values as flags."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"LNSConfig.{name} must be a bool, got {value!r}")
    return bool(value)


def _iterable_config_value(name: str, value) -> list:
    """Materialize a collection field once, rejecting strings masquerading as collections."""
    if isinstance(value, (str, bytes)):
        raise ValueError(f"LNSConfig.{name} must be a collection, got {value!r}")
    try:
        return list(value)
    except TypeError:
        raise ValueError(f"LNSConfig.{name} must be a collection, got {value!r}") from None


def _validate_lns_config(lns: LNSConfig) -> LNSConfig:
    """Validate and normalize arguments before ``LNSState`` takes over the ledger.

    Normalized execution-control fields use ordinary Python scalars so pool widths and result
    summaries remain JSON-serializable when callers supplied compatible NumPy scalar types.
    """
    if isinstance(lns.operators, str) or not isinstance(lns.operators, (tuple, list)):
        raise ValueError(
            f"LNSConfig.operators must be a sequence of operator names, got {lns.operators!r}")
    operators = tuple(lns.operators)
    if any(not isinstance(name, str) for name in operators):
        raise ValueError(f"LNSConfig.operators must contain only strings, got {operators!r}")
    known = {"agent", "map", "random"}
    unknown = [name for name in operators if name not in known]
    if unknown:
        raise ValueError(f"unknown LNS operators {unknown!r} (want subset of agent/map/random)")
    if not operators:
        raise ValueError("LNSConfig.operators is empty — need at least one of agent/map/random")
    if len(set(operators)) != len(operators):
        raise ValueError(
            f"duplicate LNS operators {list(operators)!r} — each name at most once")
    if lns.repair_order not in ("premium", "random"):
        raise ValueError(
            f"unknown LNSConfig.repair_order {lns.repair_order!r} (want 'premium' or 'random')")

    integer_minima = {
        "seed": 0,
        "search_workers": 1,
        "max_iterations": 0,
        "neighborhood_size": 1,
        "max_walks": 0,
        "map_max_cells": 0,
        "verify_every": 0,
        "log_every": 0,
    }
    integers = {
        name: _integer_config_value(name, getattr(lns, name), minimum=minimum)
        for name, minimum in integer_minima.items()
    }
    optional_integers = {}
    for name, minimum in {
        "unimpeded_workers": 1,
        "worker_kernel_log2": 0,
        "window_bytes": 0,
    }.items():
        value = getattr(lns, name)
        optional_integers[name] = (
            None if value is None
            else _integer_config_value(name, value, minimum=minimum)
        )
    booleans = {
        name: _boolean_config_value(name, getattr(lns, name))
        for name in ("adaptive", "incremental_release")
    }
    frozen_flight_ids = frozenset(
        _integer_config_value("frozen_flight_ids", fid, minimum=0)
        for fid in _iterable_config_value("frozen_flight_ids", lns.frozen_flight_ids)
    )
    if lns.movable_uss_ids is None:
        movable_uss_ids = None
    else:
        uss_ids = _iterable_config_value("movable_uss_ids", lns.movable_uss_ids)
        if any(not isinstance(uss_id, str) for uss_id in uss_ids):
            raise ValueError(
                f"LNSConfig.movable_uss_ids must contain only strings, got {lns.movable_uss_ids!r}")
        movable_uss_ids = frozenset(uss_ids)
    gamma = _float_config_value("gamma", lns.gamma, minimum=0.0, maximum=1.0)
    accept_epsilon = _float_config_value(
        "accept_epsilon", lns.accept_epsilon,
        minimum=0.0, allow_positive_infinity=True,
    )
    time_limit_s = (
        None if lns.time_limit_s is None
        else _float_config_value(
            "time_limit_s", lns.time_limit_s,
            minimum=0.0, allow_positive_infinity=True,
        )
    )
    ceiling = 4 * (os.cpu_count() or 1)
    search_workers = integers["search_workers"]
    if search_workers > ceiling:
        raise ValueError(
            f"LNSConfig.search_workers={search_workers} exceeds {ceiling} (4x cores) — each "
            f"worker holds its own replica of the schedule (ledger + occupancy pools + claim "
            f"index), so memory is linear in workers, not amortised across them")
    if lns.parallel_mode not in ("sync", "drop"):
        raise ValueError(
            f"unknown LNSConfig.parallel_mode {lns.parallel_mode!r} (want 'sync' or 'drop')")
    return replace(
        lns,
        operators=operators,
        gamma=gamma,
        accept_epsilon=accept_epsilon,
        frozen_flight_ids=frozen_flight_ids,
        movable_uss_ids=movable_uss_ids,
        time_limit_s=time_limit_s,
        **integers,
        **optional_integers,
        **booleans,
    )


def _effective_search_workers(lns: LNSConfig) -> int:
    """Processes that can receive work under this configuration's task budget."""
    return min(lns.search_workers, lns.max_iterations)


def _trajectory_row(
    i,
    op,
    victims,
    accepted,
    reason,
    cost_old,
    cost_new,
    incumbent_cost,
    wall_s,
    *,
    incumbent_before=None,
    audit: dict | None = None,
) -> dict:
    """Build the common trajectory schema without coupling it to an execution mode."""
    row = dict(
        iter=i, op=op, n=len(victims), victims=list(victims),
        accepted=accepted, reason=reason, cost_old=cost_old,
        cost_new=None if (cost_new is None or math.isinf(cost_new)) else cost_new,
        incumbent_cost=incumbent_cost, wall_s=wall_s,
    )
    if incumbent_before is not None:
        row["realized_improvement"] = float(incumbent_before) - float(incumbent_cost)
    if audit:
        row.update(audit)
    return row


def _trajectory_auc(trajectory: list[dict], cost_before: float, horizon_s: float) -> float:
    """Integrate the best-known cost as a right-continuous step function over ``[0, horizon]``."""
    horizon = max(0.0, float(horizon_s))
    area = 0.0
    previous_t = 0.0
    previous_cost = float(cost_before)
    for row in trajectory:
        # Completion timestamps should already be monotone. Clamp defensively so telemetry can
        # never manufacture negative area or integrate beyond the reported run horizon.
        current_t = min(horizon, max(previous_t, float(row["wall_s"])))
        area += (current_t - previous_t) * previous_cost
        previous_t = current_t
        previous_cost = float(row["incumbent_cost"])
        if current_t >= horizon:
            break
    area += (horizon - previous_t) * previous_cost
    return float(area)


def _build_lns_state(
    cfg: SimConfig,
    ledger: ReservationLedger,
    intents: list[OperationalIntent],
    lns: LNSConfig,
    *,
    static_terms: tuple | None,
    turnaround_s: float | None,
    maintain_claim_index: bool = True,
) -> LNSState:
    """One construction path for the sequential runner and the parallel coordinator."""
    return LNSState(
        cfg,
        ledger,
        intents,
        static_terms=ledger.static_terminals() if static_terms is None else static_terms,
        frozen_flight_ids=lns.frozen_flight_ids,
        movable_uss_ids=lns.movable_uss_ids,
        turnaround_s=turnaround_s,
        incremental_release=lns.incremental_release,
        unimpeded_workers=lns.unimpeded_workers,
        maintain_claim_index=maintain_claim_index,
        window_bytes=lns.window_bytes,
    )


def _finalize_lns_result(
    state: LNSState,
    trajectory: list[dict],
    cost_before: float,
    n_iter: int,
    n_accepted: int,
    t0: float,
    init_s: float,
    selector: AdaptiveSelector,
    *,
    search_workers: int = 1,
    parallel_mode: str = "sequential",
    pool_spawn_s: float = 0.0,
    parallel_stats: dict | None = None,
) -> LNSResult:
    """Verify and build the common sequential/parallel result without metric drift."""
    final = state.final_intents()
    bad = verify.find_interflight_conflict(
        final, state.cfg, static_terminals=state.static_terms)
    wall_s = time.monotonic() - t0
    return LNSResult(
        intents=final,
        trajectory=trajectory,
        cost_before=cost_before,
        cost_after=state.total_cost,
        n_iterations=n_iter,
        n_accepted=n_accepted,
        wall_s=wall_s,
        init_wall_s=init_s,
        weights=dict(selector.weights),
        verified=bad is None,
        search_workers=search_workers,
        parallel_mode=parallel_mode,
        auc=_trajectory_auc(trajectory, cost_before, wall_s),
        pool_spawn_s=pool_spawn_s,
        parallel_stats={} if parallel_stats is None else dict(parallel_stats),
    )


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
    # Validate BEFORE constructing LNSState: it detaches the caller's subscribers and may spend
    # minutes building the unimpeded ruler, so an argument error must not cost either.
    lns = _validate_lns_config(lns)

    if _effective_search_workers(lns) > 1:
        # Deferred import: keeps multiprocessing (and the pool module) off the sequential path,
        # while parallel.py can import the shared solver helpers only after this module is loaded.
        from freespace_sim.planner.lns.parallel import run_lns_parallel

        return run_lns_parallel(cfg, ledger, intents, lns,
                                static_terms=static_terms, turnaround_s=turnaround_s)

    t0 = time.monotonic()
    state = _build_lns_state(
        cfg, ledger, intents, lns,
        static_terms=static_terms, turnaround_s=turnaround_s,
        maintain_claim_index=lns.max_iterations > 0,
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
                    trajectory.append(_trajectory_row(
                        i, name, (), False, "empty", 0.0, 0.0,
                        state.total_cost, time.monotonic() - t0,
                    ))
                    continue
                out = state.try_repair(
                    victims, rng_i, lns.accept_epsilon, order_mode=lns.repair_order
                )
                if lns.adaptive:
                    selector.update(name, out.improvement)
                n_accepted += int(out.accepted)
                trajectory.append(_trajectory_row(
                    i, name, sorted(victims), out.accepted, out.reason,
                    out.cost_old, out.cost_new,
                    state.total_cost, time.monotonic() - t0,
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

        return _finalize_lns_result(
            state, trajectory, cost_before, n_iter, n_accepted, t0, init_s, selector,
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
