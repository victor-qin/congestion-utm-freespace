"""Configuration for the column-generation planner."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ColGenParams:
    """Network and solver controls for one column-generation run.

    The gaps use relative objective units (``1e-4`` means 0.01%).  ``M`` is the
    per-flight benefit in the maximize master objective ``M - delay_s``; it is
    deliberately much larger than the shipped ground-delay budget so a usable
    trajectory dominates cancellation.  ``time_limit_s`` is a best-effort
    whole-solve wall budget: pricing and native LP/IP calls receive the remaining
    time, with checks between graph, seed, and lazy-separation rounds.  A single
    synchronous geometry/backend call cannot be preempted and may overrun slightly.
    """

    detour_slack_hops: int = 12
    solver: str = "auto"
    max_iterations: int = 30
    time_limit_s: float = 120.0
    lp_gap: float = 1e-4
    ip_gap: float = 1e-3
    M: float = 1_000_000.0
    epsilon: float = 1e-6
    n_heuristic_tries: int = 32
    objective: str = "total_delay"
    # Which scale the lp_gap / ip_gap thresholds are measured on.
    #   "revenue" -- the paper's equations (10) and (11): (UB - RMP)/RMP on the maximize
    #                objective, whose scale includes n*M.
    #   "cost"    -- the same absolute gap normalized by total cost instead, which is
    #                what this repo used before and is far stricter when M >> cost.
    gap_metric: str = "revenue"
    # Pure clock translations of each flight's seed, offered to the master before the
    # first LP.  A shift is arithmetic, not a search, and pricing otherwise spends its
    # early iterations rediscovering exactly these -- 91% of the columns added in
    # iterations 2-11 of a converged 100-flight solve were time shifts of a route already
    # in the pool.  See :func:`solver._add_departure_ladder` for the depth measurements;
    # 0 disables.
    seed_ladder_steps: int = 20
    shortcut: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.detour_slack_hops, bool):
            raise TypeError("detour_slack_hops must be an integer")
        try:
            slack = operator.index(self.detour_slack_hops)
        except TypeError as exc:
            raise TypeError("detour_slack_hops must be an integer") from exc
        if slack < 0:
            raise ValueError("detour_slack_hops must be non-negative")
        object.__setattr__(self, "detour_slack_hops", slack)

        if not isinstance(self.solver, str):
            raise TypeError("solver must be a string")
        solver = self.solver.lower()
        if solver not in {"auto", "gurobi", "highs"}:
            raise ValueError("solver must be one of 'auto', 'gurobi', or 'highs'")
        object.__setattr__(self, "solver", solver)

        if not isinstance(self.objective, str):
            raise TypeError("objective must be a string")
        # ``total_delay`` sums ground and excess-air seconds unweighted, which is what
        # colgen has always done.  ``total_cost`` weights them by the config's
        # per-second dials (1:3), matching the cost model the A* planner uses -- see
        # :mod:`freespace_sim.planner.colgen.objective`.
        if self.objective not in {"total_delay", "total_cost"}:
            raise ValueError("objective must be 'total_delay' or 'total_cost'")
        if not isinstance(self.gap_metric, str):
            raise TypeError("gap_metric must be a string")
        if self.gap_metric not in {"revenue", "cost"}:
            raise ValueError("gap_metric must be 'revenue' or 'cost'")
        if not isinstance(self.shortcut, bool):
            raise TypeError("shortcut must be a boolean")

        if isinstance(self.seed_ladder_steps, bool):
            raise TypeError("seed_ladder_steps must be an integer")
        try:
            ladder = operator.index(self.seed_ladder_steps)
        except TypeError as exc:
            raise TypeError("seed_ladder_steps must be an integer") from exc
        if ladder < 0:
            raise ValueError("seed_ladder_steps must be non-negative")
        object.__setattr__(self, "seed_ladder_steps", ladder)

        for name in ("max_iterations", "n_heuristic_tries"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            try:
                normalized = operator.index(value)
            except TypeError as exc:
                raise TypeError(f"{name} must be an integer") from exc
            if normalized < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, normalized)

        for name in ("time_limit_s", "lp_gap", "ip_gap", "M", "epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a real number")
            try:
                normalized = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name} must be a real number") from exc
            if not math.isfinite(normalized):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, normalized)

        if self.time_limit_s <= 0.0:
            raise ValueError("time_limit_s must be positive")
        if not 0.0 <= self.lp_gap < 1.0:
            raise ValueError("lp_gap must be in [0, 1)")
        if not 0.0 <= self.ip_gap < 1.0:
            raise ValueError("ip_gap must be in [0, 1)")
        if self.M <= 0.0:
            raise ValueError("M must be positive")
        if not 0.0 <= self.epsilon < 0.5:
            raise ValueError("epsilon must be in [0, 0.5)")
