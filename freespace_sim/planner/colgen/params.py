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

    # Sizes the spatial ellipse each flight is priced over, so it is the dominant term in
    # how much search a sweep does -- and it is NOT a route-length budget (see
    # `_best_column`: ordinary pricing may spend the clock slack on wide loops).
    #
    # Measured on colgen_test's first 50 flights, no time limit, everything else equal:
    #
    #     slack   wall    iters  termination       objective   labels expanded   arc nodes
    #         1   645 s      30  iteration_limit        724.8    35,017,481         3,554
    #         2   734 s      30  iteration_limit        724.8    47,583,416         4,269
    #         3   762 s      29  lp_gap                 724.8    60,166,158         4,909
    #        12   971 s      30  iteration_limit        724.8   124,599,822        11,230
    #
    # Labels grow 3.56x from slack 1 to 12 and every setting reaches the SAME objective;
    # slack 1 and 2 placed all 50 flights with zero denials. Wall grows only 1.51x across
    # that range, because much of a solve is per-iteration overhead -- LP solves,
    # canonicalization, the greedy stage -- that the ellipse does not touch.
    #
    # Do not read the termination column as "slack 3 converges and the others do not":
    # every arm sat at or one below the 30-iteration cap, so which side of it they landed
    # on is a threshold artifact, not a property of the slack.
    #
    # Left at 12 deliberately. This is one uncongested 50-flight world, and the slack exists
    # so pricing can route AROUND congestion -- exactly what a denser instance has more of,
    # and exactly what this one cannot test. But anyone tuning for speed starts here: it is
    # the dominant term in how much search a sweep does.
    detour_slack_hops: int = 12
    # How far over the LATTICE geodesic a priced route may fly, as a fraction of it: a
    # flight whose shortest path is 17 hops may fly ceil(0.10 * 17) = 2 extra. This is an
    # air-time cap, so ground delay is unaffected -- holding on the pad is a lever, wandering
    # in the air to dodge a dual is the thing being bounded.
    #
    # Measured against the LATTICE geodesic and not `enroute_reference_m`, because hex
    # quantization alone already puts 49 of colgen_test's first 50 flights over 1.10x the
    # Euclidean straight line (median 1.26x, max 2.32x) -- a fractional cap on that base
    # would deny almost every flight before congestion entered the picture.
    #
    # Deliberately suboptimal, in the same way `detour_slack_hops` is: a route needing more
    # than this becomes unreachable even if it is the true optimum. `translate.py`'s
    # `max_detour_factor` gate expresses a similar idea but fires at translation, after the
    # label has been expanded and certified -- this one prunes during the search, which is
    # where the work is. Set to 0 for the geodesic only; set high to disable.
    #
    # Measured on colgen_test's first 50 flights, no time limit:
    #
    #     slack  ceiling   wall   iters  objective     labels    arc nodes  denied
    #        12      off   936 s     30      724.8  124,615,013     11,230       0
    #        12      10%   628 s     30      724.8   35,878,413      5,751       0
    #         3      off   739 s     29      724.8   60,178,713      4,909       0
    #         3      10%   611 s     30      724.8   33,246,072      4,771       0
    #
    # At the shipped slack of 12 the ceiling cuts labels 3.47x and wall 1.49x for the same
    # objective and no denials, and it also halves `arc nodes` -- cells unreachable within
    # the hop budget are never expanded at all, so it shrinks the reachable corridor rather
    # than only the label count. It is the stronger of the two levers here: slack 12 WITH
    # the ceiling searches less than slack 3 without it, so the wide ellipse is affordable.
    #
    # Read the identical objective with care. This world is uncongested -- every arm places
    # all 50 flights and reaches 724.8 -- so it cannot show what the ceiling costs. For a
    # case where it does, see `test_the_air_time_ceiling_forbids_the_loop_the_ellipse_allows`.
    max_air_overrun_frac: float = 0.10
    solver: str = "auto"
    max_iterations: int = 30
    time_limit_s: float = 120.0
    lp_gap: float = 1e-4
    ip_gap: float = 1e-3
    M: float = 1_000_000.0
    epsilon: float = 1e-6
    # Two consumers, and they scale differently.
    #
    # `master.round_heuristic` runs this many randomized-rounding restarts per iteration:
    # ~4.7 ms per try on a 425-column colgen_test master, against pricing sweeps of ~100 s.
    # Measured on 98 flights, 16 / 32 / 64 restarts all returned the IDENTICAL incumbent
    # (heuristic 3260.4, IP 3152.4 after one iteration), so the extra restarts bought
    # nothing on that world and the low value is the honest default; their value is on a
    # more fractional LP than this one produces.
    #
    # `_greedy_feasible_selection` ALSO derives its candidate cap from this (x16), which has
    # nothing to do with rounding and is worth knowing about before changing the field. That
    # stage runs once, after the first LP, and walks flights in order asking pricing for a
    # better column for each; the cap truncates how many it reaches. At 16 the cap is 256.
    #
    # The stage is bounded by `min(60 s, 0.55 * time_limit_s)` and splits what remains
    # evenly across the flights still to try, so a longer list means a SMALLER slice each
    # and hard flights hit their local timeout instead of eating the stage. Measured on 780
    # flights at the 60 s budget, a cap of 512 ran 32.2 s and improved 217 flights where
    # 1024 ran 19.6 s and improved 265 -- reaching every flight shallowly beat reaching two
    # thirds of them deeply. Below the cap none of this applies.
    n_heuristic_tries: int = 16
    objective: str = "total_delay"
    # Which scale the lp_gap / ip_gap thresholds are measured on.
    #   "revenue" -- the paper's equations (10) and (11): (UB - RMP)/RMP on the maximize
    #                objective, whose scale includes n*M.
    #   "cost"    -- the same absolute gap normalized by total cost instead, which is
    #                what this repo used before and is far stricter when M >> cost.
    gap_metric: str = "revenue"

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

        for name in ("time_limit_s", "lp_gap", "ip_gap", "M", "epsilon", "max_air_overrun_frac"):
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

        if self.max_air_overrun_frac < 0.0:
            raise ValueError("max_air_overrun_frac must be non-negative")
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
