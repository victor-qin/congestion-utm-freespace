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

    # THE knob that sizes the pricing search. It does two jobs that used to be two fields,
    # because they are one degree of freedom:
    #
    #   1. How far over the LATTICE geodesic a priced route may fly, in hops. One hop advances
    #      the clock by `dt_s`, so this is equally "how many steps of air time over nominal": a
    #      flight whose shortest path is 17 hops may fly 17 + this. Ground delay is unaffected --
    #      holding on the pad is a lever, wandering in the air to dodge a dual is what is bounded.
    #   2. The radius of the spatial O-D ellipse the flight is priced over -- the corridor.
    #
    # (2) FOLLOWS FROM (1) and is not an independent choice. For any path o = c0 ... cL = d and
    # any cell ck on it, hex_distance(o,ck) <= k and hex_distance(ck,d) <= L - k, so their sum
    # is <= L. A route within `shortest + overrun` hops can therefore only touch cells inside
    # the ellipse of that radius, whatever a wider corridor would have allowed. The converse
    # does NOT hold -- an ellipse permits circling inside itself indefinitely -- which is why
    # the hop budget is the one that has to be primary.
    #
    # This was two knobs (`detour_slack_hops` for the ellipse, this one for the budget) until
    # issue #78. Sized apart they were either waste (a corridor band no legal route can reach)
    # or a silent takeover (a narrow corridor quietly overriding the budget the user set), and
    # `9816f61` had already had to fix one derived quantity denominated in the wrong one of the
    # pair. One number, one currency.
    #
    # ABSOLUTE rather than a fraction of the flight, which is a deliberate choice about who
    # pays. A fractional cap gives a 2-hop flight one extra hop and a 23-hop flight three,
    # so it is harshest exactly where the absolute room is smallest; a fixed budget gives
    # every flight the same room to route around a busy cell regardless of how far it is
    # going. Neither is obviously right, but the fixed one is easier to reason about and
    # does not silently pin short flights to their geodesic.
    #
    # Measured against the LATTICE geodesic and not `enroute_reference_m`: hex quantization
    # alone already puts 49 of colgen_test's first 50 flights over 1.10x the Euclidean
    # straight line (median 1.26x, max 2.32x), so a budget measured on that base would be
    # spent before congestion entered the picture.
    #
    # Deliberately suboptimal: a route needing more than this becomes unreachable even if it
    # is the true optimum. `translate.py`'s `max_detour_factor` gate expresses a similar idea
    # but fires at translation, after the label has been expanded and certified -- this one
    # prunes during the search, which is where the work is. 0 pins the geodesic; a large
    # value effectively disables both bounds.
    #
    # ---- MEASUREMENTS, AND WHAT THEY NO LONGER DESCRIBE -------------------------------
    #
    # Both tables below were taken on colgen_test's FIRST 50 FLIGHTS via `ColGenSolver().solve`
    # directly: HiGHS, objective=total_delay, gap_metric=cost, n_heuristic_tries=16,
    # time_limit_s=86400 (so the 30-iteration cap is what stops them, not the clock). They share
    # that harness, so they compare to each other and to nothing else -- in particular NOT to the
    # end-to-end figures in the PR/README, which are 98 flights of weighted total_cost under
    # Gurobi on a 600 s budget. Those objectives are different quantities, not the same one
    # measured twice.
    #
    # BOTH ARE HISTORICAL RECORDS OF THE TWO-KNOB ERA. Neither is a sweep of the knob that now
    # exists, and neither can be reproduced by this code -- read them for the shape of the
    # effect, not as a menu.
    #
    # Table 1 -- corridor swept with the CEILING OFF (unreproducible: the ceiling is the
    # corridor now, so it cannot be turned off while the corridor stays finite):
    #
    #     corridor   wall    iters  termination       objective   labels expanded   arc nodes
    #            1   645 s      30  iteration_limit        724.8    35,017,481         3,554
    #            2   734 s      30  iteration_limit        724.8    47,583,416         4,269
    #            3   762 s      29  lp_gap                 724.8    60,166,158         4,909
    #           12   971 s      30  iteration_limit        724.8   124,599,822        11,230
    #
    # Table 2 -- ceiling swept at a FIXED corridor of 3 (unreproducible below/above the top row
    # for the same reason: "6" would now mean corridor 6 AND ceiling 6, which is neither arm):
    #
    #     ceiling   wall    iters  termination       objective   labels expanded   arc nodes
    #           3   685 s      30  iteration_limit        724.8    36,702,755          4,982
    #           6   788 s      30  iteration_limit        724.8    57,745,819          4,907
    #           9   756 s      29  lp_gap                 724.8    59,448,491          4,907
    #         off   757 s      29  lp_gap                 724.8    60,169,941          4,909
    #
    # Table 2's first row is the shipped 3 == 3 pairing, so it is the closest either table comes
    # to the configuration below -- but it too was taken on pre-#78 code and has been superseded.
    # Re-measured on the same harness, `main` at `31891be` against this build, at the default:
    #
    #     build           wall    iters  termination       objective   labels        arc nodes
    #     31891be        686 s      30  iteration_limit   724.840287   36,269,194        4,983
    #     this (issue78) 454 s      30  iteration_limit   724.840287   16,680,454        4,983
    #
    # BYTE-IDENTICAL schedule: all 50 columns agree on flight_id, departure_step, both lane
    # indices and full cell path. The 2.17x is `completion_envelope`'s ceiling cap, which stops
    # `completion_can_compete` keeping labels alive on completions the ceiling forbids -- see
    # the comment on `max_total_hops` in pricing.py. The 36.3M reproduces table 2's 36.7M to
    # within 1.2%, so its `labels` column and this one are the same quantity.
    #
    # (Wall was taken under uneven load and the counter itself is instrumentation, so read the
    # label column, not the seconds. Both arms carried the same instrumentation.)
    #
    # Every arm of both tables placed all 50 flights with zero denials at the SAME objective,
    # so neither bound cost anything on this world. Do not read the termination column as
    # "3 converges and the others do not": every arm sat at or one below the 30-iteration cap,
    # so which side of it they landed on is a threshold artifact.
    #
    # THE CAVEAT THAT COMES WITH ALL OF IT: both tables are one UNCONGESTED 50-flight world,
    # and this knob exists so pricing can route AROUND congestion -- exactly what a denser
    # instance has more of, and exactly what this one cannot test. If a congested scenario
    # starts denying flights pricing ought to be able to place, this is the first knob to
    # widen, and `--colgen-max-air-overrun` widens it without a code change. Symmetrically, it
    # is the dominant term in how much search a sweep does, so it is also where speed tuning
    # starts.
    max_air_overrun_hops: int = 3
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
        if isinstance(self.max_air_overrun_hops, bool):
            raise TypeError("max_air_overrun_hops must be an integer")
        try:
            overrun = operator.index(self.max_air_overrun_hops)
        except TypeError as exc:
            raise TypeError("max_air_overrun_hops must be an integer") from exc
        if overrun < 0:
            raise ValueError("max_air_overrun_hops must be non-negative")
        object.__setattr__(self, "max_air_overrun_hops", overrun)

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
