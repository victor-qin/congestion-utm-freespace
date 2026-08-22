"""Configuration for the column-generation planner."""

from __future__ import annotations

import math
import operator
import os
from dataclasses import dataclass


# Planners whose schedule can be translated into a colgen warm start.  Kept as a module
# constant so `params` validates a name without importing the planner registry -- colgen is
# imported BY the sim, so reaching back for it here would be circular.
WARM_START_PLANNERS = frozenset({"astar"})


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
    # the comment on `max_total_hops` in pricing.py.
    #
    # The 2.17x is like-for-like WITHIN this pair -- one machine, one counter
    # (`analysis/colgen_bounds_digest.py --label-counts`, which wraps `_prefer`), both arms
    # instrumented identically. Do NOT read it against table 2's 36,702,755. That is the same
    # order of magnitude as the 36,269,194 here, which is reassuring, but a deterministic solve
    # under one counter would reproduce EXACTLY, and 1.2% apart does not: the two differ in
    # counter definition, in the build the table was taken on, or both. Cross-table label
    # comparisons are not supported; comparisons within a pair are.
    #
    # Wall was taken under uneven load and with the counter attached, so read the label column
    # rather than the seconds.
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
    # Gurobi by default, NOT "auto".  The difference is what happens when gurobipy is
    # missing: "auto" falls back to HiGHS silently, so a run that believes it is on Gurobi
    # can quietly be on the other backend -- and the backend is answer-affecting, not just
    # a speed knob (Gurobi's duals close the default revenue gap at iteration 1 where HiGHS
    # runs 17).  "gurobi" raises instead, naming the missing extra.
    #
    # It is also the backend that scales.  HiGHS reaches the master through
    # `scipy.optimize.milp`, which has no incumbent parameter, so the final IP runs COLD --
    # measured on `density_faa_wing_zipline` x1000, a 23.5k-column / 46k-row pool: over 30
    # minutes and still going, more than all three pricing sweeps put together (610 + 459 +
    # 190 s).  Gurobi takes the rounding incumbent as a warm start and is multi-threaded.
    #
    # `pyproject.toml` deliberately keeps gurobipy an opt-in extra because the PyPI wheel
    # ships a size-limited trial that silently caps model size; this default therefore makes
    # `uv sync --extra gurobi` (and a real licence) a requirement for colgen, and says so
    # loudly rather than degrading.  Set `solver="highs"` to opt out, or "auto" for the old
    # fall-back behaviour.
    solver: str = "gurobi"
    max_iterations: int = 30
    # 20 minutes, raised from 120 s. The old default could not finish a single pricing
    # sweep on a real instance: measured on `density_faa_wing_zipline` x100, iteration 1
    # alone is 147 s and the sweeps LENGTHEN as the pool grows -- 147, 213, 222, 234, 235,
    # 397, 521, 555 s -- so a 120 s budget terminated inside the first sweep and reported
    # `time_limit` with a schedule built entirely by the rounding heuristic. 1200 s buys
    # roughly three iterations at that size and ten or more at 50 flights.
    #
    # `ip_reserve_s = min(5, 0.05 * t)` is already at its cap by 1200 s, so the tail left
    # for the final IP does not move.  The greedy's budget is no longer derived from this
    # at all -- see `greedy_budget_s_per_flight`, which now ships at 0 and therefore skips
    # the stage entirely -- but an ENABLED rate is still CLAMPED by `pricing_deadline`, so
    # raising this lifts that stage's effective ceiling on a batch large enough for
    # `rate * n_flights` to reach it (about 1,700 flights at the old 0.7 rate).
    time_limit_s: float = 1200.0
    # A cap on the FINAL restricted-master IP, in seconds of its own -- not a share of
    # `time_limit_s`.  Without it the IP simply inherits whatever is left of the whole-solve
    # deadline, which is unbounded from the IP's point of view precisely when the loop went
    # WELL: a solve that converges on `lp_gap` or hits `max_iterations` early hands the MILP
    # every remaining second. Measured on `density_faa_wing_zipline` x1000 with
    # `time_limit_s=10800`: the CG loop finished three iterations in ~21 minutes of pricing,
    # then the IP ran over 30 minutes and was still going, with ~2 hours of budget left to
    # burn -- longer than every pricing sweep combined, and single-threaded on HiGHS, whose
    # `scipy.optimize.milp` takes no incumbent so it cannot even start from the rounding
    # solution the solver already holds.
    #
    # Exceeding it is not a failure: `solve_ip` falls back to the independently validated
    # rounding incumbent and reports `time_limit_separation` / a non-optimal `ip_status`, so
    # the run still produces a claim-feasible schedule -- just an uncertified one.  That is
    # the right trade for a stage whose marginal value is proving what the heuristic already
    # found; raise it when the IP's certificate is what you are after.
    #
    # It composes with, rather than replaces, `ip_reserve_s = min(5, 0.05 * time_limit_s)`:
    # that is how much the pricing loop holds BACK, this is how long the IP may then run.
    # When the loop consumes its whole budget the reserve still binds and the IP gets the
    # smaller tail.
    ip_time_limit_s: float = 120.0
    lp_gap: float = 1e-4
    ip_gap: float = 1e-3
    # Revenue per served flight in the set-packing objective `sum (M - delay_s) x`.  Its ONLY
    # requirement is that serving a flight always beats denying it -- `M > max delay_s` -- and
    # everything above that threshold is pure numerical harm.
    #
    # `delay_s = 1*ground + 3*(air beyond reference)`.  Ground is capped by
    # `cfg.max_ground_delay_s` (3600 on the density scenarios) and air by
    # `max_air_overrun_hops` plus lattice overhead (~35 s typical, a few hundred at worst), so
    # the bound is ~4,600 and 1e4 clears it ~2.2x.  Exceeding 1e4 would need >2,100 s of extra
    # flying, which the 3-hop cap makes impossible.
    #
    # 1e6 cost a factor of 100 in conditioning for nothing.  Coefficients are `M - delay_s`:
    # at 1e6 that is ~999,855 varying by ~1,000, so the quantity being optimized is 0.1% of
    # the number carrying it and lives in the last three digits.  At 1e4 the same variation is
    # 10% of the coefficient.  The `LB -1e9` dual escape recorded on the density runs is what
    # that looks like from the outside.
    #
    # Reported figures do NOT move with M: runs report `cost = n*M - ip_objective`, and
    # `Column.delay_s` has no M in it, so archived cost numbers stay on this same ruler.  The
    # SOLUTIONS may differ -- a differently conditioned search takes a different path -- so
    # this is answer-affecting even though the metric is not.
    M: float = 10_000.0
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
    # The stage is bounded by `greedy_budget_s_per_flight * n_flights` and splits what
    # remains evenly across the flights still to try, so a longer list means a SMALLER slice
    # each and hard flights hit their local timeout instead of eating the stage. Measured on
    # 780 flights at the old flat 60 s budget, a cap of 512 ran 32.2 s and improved 217 where
    # 1024 ran 19.6 s and improved 265 -- reaching every flight shallowly beat reaching two
    # thirds of them deeply. Below the cap none of this applies.
    n_heuristic_tries: int = 16
    # What the master minimises.  ``total_cost`` weights the two currencies the way the
    # config does -- `cost_ground_delay_per_s = 1`, `cost_air_lateral_per_s = 3`
    # (config.py:83-85) -- so one step of ground costs `dt` and one hop of air costs `3*dt`.
    # ``total_delay`` weights them EQUALLY, and that is not merely a different answer, it is
    # a degenerate one: `ground + flown` is invariant under a ground-for-air swap, so an
    # enormous set of columns are EXACTLY tied and the label DP's dominance cannot separate
    # labels the real objective strictly orders (pricing.py:1765-1772 spells this out).
    #
    # It is also the reason this default moved.  The tie plateau, plus a `delay_lbs[hops]`
    # that rises `w_air*dt` per hop -- 4 under `total_delay`, 12 here -- makes the
    # completion envelope roughly 3x longer and prunes far less.  Measured on
    # `density_faa_wing_zipline` x12, 2 iterations, sequential, bootstrap off:
    #
    #     objective      WALL       fell_back   peak labels   straggler
    #     total_delay    975.42 s   2 of 24     33.5M         456.9 s (87.6% of sweep 1)
    #     total_cost      32.11 s   0 of 24      3.4M          20.6 s
    #
    # 30x, and issue #90's label-ceiling straggler does not occur at all.  Benchmarking the
    # pricing search under `total_delay` was measuring the weakest pruning regime the code
    # has, which is not the one that ships.
    objective: str = "total_cost"
    # Which scale the lp_gap / ip_gap thresholds are measured on.
    #   "revenue" -- the paper's equations (10) and (11): (UB - RMP)/RMP on the maximize
    #                objective, whose scale includes n*M.
    #   "cost"    -- the same absolute gap normalized by total cost instead, which is
    #                what this repo used before and is far stricter when M >> cost.
    #
    # Pinned to "cost" because "revenue" is not scale-free in M and M just moved.  The revenue
    # gate reduces to `tau*M` -- `n` cancels -- so it literally reads "stop unless the average
    # flight can still save more than `tau*M` seconds": 100 s at M=1e6, but 1 s at M=1e4.
    # Leaving the default on "revenue" would have turned the M change into a silent 100x
    # tightening of the stopping rule, converting ordinary runs into time-limit runs.  "cost"
    # measures the gap against the quantity being optimized and does not move with M at all,
    # which is the property a default needs.  Every analysis script in this investigation was
    # already passing "cost" explicitly for exactly this reason.
    gap_metric: str = "cost"
    # Optionally run another planner first and hand colgen its schedule -- as pool columns
    # AND as the starting incumbent.  ``None`` (the shipped default) means colgen opens
    # from its own geodesic seeds; ``"astar"`` seeds from FCFS A*.
    #
    # OFF BY DEFAULT ON PURPOSE, and the reason is reporting rather than cost.  Measured on
    # `density_faa_wing_zipline` x1500: unaided colgen is 235,388 (+11.3% against A*), and
    # A*-seeded colgen is 196,398 (-7.1%).  Turning this on silently would make "colgen" in
    # every future result mean "A* plus colgen refinement", so "colgen beats A*" would
    # quietly become "refining A* beats A*" -- a different and much weaker claim, with no
    # way to tell archived runs apart afterwards.  It is reported in `stats` for the same
    # reason.  A* itself costs 64 s against a 3,750 s solve, so the price is not the issue.
    warm_start_planner: str | None = None
    # Pure clock translations of each flight's seed, offered to the master before the
    # first LP.  A shift is arithmetic, not a search, and pricing otherwise spends its
    # early iterations rediscovering exactly these -- 91% of the columns added in
    # iterations 2-11 of a converged 100-flight solve were time shifts of a route already
    # in the pool.  See :func:`solver._add_departure_ladder` for the depth measurements;
    # 0 disables.
    seed_ladder_steps: int = 20
    # Distinct ROUTES seeded per flight, against the ladder's distinct DEPARTURES.  1 (the
    # shipped default) is today's behaviour exactly: one geodesic, re-timed.
    #
    # The two knobs span the two axes of the same pool, and until now only one of them was
    # open.  Measured on the x1500 barrier pool: seeds give 1.000 distinct routes per
    # flight, and 99.3% of what PRICING then returns is a new route -- 84.4% of which
    # shares the seed's endpoint lanes AND its hop count, making it delay-identical to the
    # seed (all 9,592 such columns, to the millisecond).  Pricing is spending 1,686 s of a
    # 4,561 s solve on alternatives that differ only in which rows they claim, and a hex
    # lattice hands those out for free: 97.8% of flights have many geodesics, median 10^19.
    #
    # `pricing.seed_route_fan` enumerates them by minimum cell overlap, so the variants
    # spread across the lattice instead of shuffling one or two cells.  Flights whose seed
    # is longer than `hex_distance` -- an ellipse or block forced a detour -- get no fan,
    # because the equal-cost argument does not hold for them.
    #
    # TREAT THE POOL AS A BUDGET.  `(variants) x (ladder + 1)` columns per flight all
    # become binary variables in the final MILP, and pool size is not monotone in quality:
    # at x1500 the marks between 35,790 and 39,446 columns returned the incumbent unchanged
    # where both 33,425 and 41,954 beat A*.  Widen the fan by shortening the ladder.
    seed_route_variants: int = 1
    # Break the fan's ties by PREDICTED CONGESTION rather than lexicographically.
    #
    # The fan picks each variant by minimum cell overlap, and on a lattice offering ~10^19
    # geodesics the set achieving the minimum is enormous -- so what breaks that tie decides
    # which routes actually enter the pool.  Left alone it is the lexicographically smallest
    # path, which is arbitrary.  With this on, `seed_row_load` counts how many flights'
    # nominal seeds want each capacity row and the fan prefers the rows fewer of them want.
    #
    # That is the answer to "must a seed fan be blind": no.  The duals are large exactly
    # where demand exceeds a cap-1 row, and "how many flights would claim this row if nobody
    # moved" is the zeroth-order version of the same quantity -- available before the first
    # LP, for the cost of a `Counter` over claims the solver already holds.  It is a ranking
    # prior over a tie class, not a feasibility test, and nothing downstream trusts it.
    #
    # Requires `seed_route_variants > 1`; inert otherwise.  ON by default because the
    # alternative it replaces is an arbitrary tie-break, not a considered choice -- but the
    # two are separable on purpose, so an A/B can say how much of the fan's value is the
    # breadth and how much is aiming it.
    seed_fan_congestion_prior: bool = True
    # Columns to take from EACH pricing subproblem, where 1 (the shipped default) is the
    # one-column-per-flight-per-sweep that column generation has always done.
    #
    # It has always done it on an assumption that is measurably false.  Over 1,764
    # certifications (`.context/probe_rc_plateau.py`), the gap from the column pricing
    # RETURNS to the runner-up is **0.000000 s at p10, at the median and at p90**: the
    # search is not selecting a unique best, it is picking one arbitrarily out of a tie
    # ~16 wide -- and that width is CENSORED at the recorder's cap of 20, and it grows as
    # the duals mature rather than closing.  99.8% of the tied set shares the winner's hop
    # count, so they are lateral swaps: identical cost, different rows.
    #
    # So this is not "generate more columns"; it is "stop throwing away fifteen of the
    # sixteen the label search already paid for".  The extra cost is one
    # `_canonical_candidate` per candidate examined, against a search that is 60-86% of the
    # sweep.
    #
    # NOT the rejected top-k proposal, and the distinction is the measurement above.  Top-k
    # returned the k highest reduced costs, which are mostly strictly worse columns priced
    # under duals that had already moved; this returns only EXACT ties, chosen among
    # themselves for minimum row overlap because the median tied alternative still shares
    # 55% of the winner's rows and k near-duplicates are excluded by the same conflict.
    pricing_tied_columns: int = 1
    # Wall clock for the post-first-LP greedy, PER FLIGHT.  **NOW 0, WHICH DISABLES THE
    # STAGE.**  It was 0.7, and the history matters: the value scales with the batch because
    # the stage splits its budget across up to `max(64, n_heuristic_tries * 16)` candidates,
    # so the previous `min(60.0, 0.55 * time_limit_s)` gave each candidate ~0.23 s at 500
    # flights and 168 of 202 searches were cut off mid-kernel.  Scaling fixed the starvation;
    # it did not make the stage worth running.
    #
    # THE STAGE'S STATED JOB IS MEASURABLY WORTHLESS.  It produces `best_heuristic`, which
    # `solver.py` hands pricing as `known_column` -- the cutoff each subproblem prunes
    # against.  That cutoff's reduced cost is `entry_rc` and it is **exactly 0.0000 on every
    # flight of every sweep measured**, because LP duality forces it: every column already in
    # the master's pool has rc <= 0 by optimality, and complementary slackness pins the basic
    # one at 0.  Removing the stage leaves pricing's time UNCHANGED on three instances --
    # 28.98 -> 29.02 s, 40.40 -> 42.54 s, 99.23 -> 99.89 s -- which is the direct measurement
    # of that argument.  (The bootstrap, `bootstrap_roots`, exists precisely because a real
    # cutoff has to be SEARCHED for; nothing free can supply one.)
    #
    # What it costs, at 4 workers, objective=total_cost, K=1/bound:
    #
    #     instance             greedy on    greedy off   saved   pricing (on -> off)
    #     density_faa   x100     60.73 s      37.80 s    1.61x   28.98 -> 29.02 s
    #     density_future x100    87.89 s      54.98 s    1.60x   40.40 -> 42.54 s
    #     density_future x200   221.04 s     129.00 s    1.71x   99.23 -> 99.89 s
    #
    # ~40% of the solve, and RSS roughly halves.  What it buys at these sizes is a COIN FLIP
    # of +/-0.16% on the objective, with no consistent sign -- `density_faa` x100 is
    # BIT-IDENTICAL without it (same objective, same 2178 columns), `density_future` x100 is
    # 0.155% worse without, and `density_future` x200 is 0.013% BETTER without.  That is path
    # dependence: the stage perturbs the master's column pool, which moves the duals, which
    # lands the search in a different local outcome.  No arm lost a flight --
    # `selected_flights` is unchanged in every one.
    #
    # AT SCALE, RUN TO CONVERGENCE, IT IS A HEAD START THAT COLUMN GENERATION CLOSES.  Twin
    # run, `density_future_wing_zipline` x500, 16 workers, K=1/bound, `gap_metric=revenue`,
    # both arms terminating on `lp_gap` at iteration 6, one harness invocation:
    #
    #     arm          WALL       pricing    objective            cols     parent RSS
    #     greedy 0.0   508.75 s   372.67 s   64373.6808602346     11,917   1936 M
    #     greedy 0.7   799.63 s   432.64 s   64290.3680883999     11,970   3201 M
    #
    # 0.129% of objective for +57.2% of wall.  The `cost_upper_bound` gap by iteration is
    # 0, -93.77, -63.43, -1.98, -2.74, -39.77: **iteration 1 is bit-identical** across the
    # arms -- same `lp_objective`, same `gap_cost`, same `gap_revenue`, same `cost_ub` --
    # because `round_heuristic` sets `best_heuristic` regardless, and the lead peaks at
    # iteration 2 and is gone by iteration 4.
    #
    # Pricing was 16% SLOWER with the stage on (372.67 -> 432.64 s), so it does not pay for
    # itself even in the stage its cutoff exists to accelerate.  Some of that is the
    # 1.9 -> 3.2 GB parent RSS under 16 workers, but the sign is the point.
    #
    # BEWARE THE TRUNCATED-ITERATION A/B, which got this call wrong twice.  The same pair at
    # **2 iterations** reads greedy-on 2.03% BETTER -- squarely inside the head start.  A
    # truncated comparison on a column-generation solve measures CONVERGENCE RATE, not
    # solution quality; both arms must run to the same TERMINATION CONDITION.
    #
    # TURNING IT OFF DOES NOT REMOVE THE FALLBACK SCHEDULE, which was the real objection.
    # `best_heuristic` is set from `master.round_heuristic` BEFORE this stage runs
    # (`solver.py:991-993`), and the greedy only replaces it when `_better_selection` says
    # it improved (`solver.py:1038-1044`).  So a timed-out solve still has a feasible
    # answer; it is
    # simply the LP-rounding one rather than a refined one.
    #
    # WHERE THIS DEFAULT IS MOST LIKELY WRONG: a solve whose `time_limit_s` genuinely binds,
    # where a better heuristic is the answer rather than a starting point.  Every measurement
    # above pinned `time_limit_s=86400`, so that regime is UNTESTED -- and the head-start
    # finding is precisely what makes it the risk.  A binding limit stops the solve INSIDE
    # the head start, at the iteration where the twin above reads 2.03% rather than 0.129%;
    # the shipped `time_limit_s = 1200.0` is 2.4x the x500 wall, which is margin but not
    # much.  If it matters, the right shape is budget-conditional -- skip the stage when the
    # projected solve fits inside the limit, run it when it might not -- rather than a fixed
    # number here.
    greedy_budget_s_per_flight: float = 0.0
    # Worker processes for the per-iteration pricing sweep; 0 keeps it in-process.  A pure
    # performance knob -- `pricing_pool.price_sweep` reproduces the sequential loop's
    # accepted prefix and index order -- but NOT a free one: each worker rebuilds every
    # graph and carries its own label pool, so memory is linear in this number.
    #
    # STAYS 0 -- OPT-IN -- AND THE ATTEMPT TO DEFAULT IT TO 4 IS RECORDED HERE BECAUSE THE
    # EVIDENCE FOR IT WAS AN ARTIFACT.  Speed was never the question: the pool measures 2.62x
    # at 4 workers on `density_faa` x100 (2026-08-04) and 2.36x / 2.15x on the two density
    # scenarios at x50.  Memory is, per the paragraph above, and the 2026-08-04 table put
    # tree peak RSS at 3,501 MB sequential against 7,888 MB at 4 workers and 22,592 MB at 16
    # -- roughly linear, and what forecloses a 4 GB/core cluster node.
    #
    # The claim that the lazily-mapped arena had lifted that rested on `rss_children` reading
    # FLAT from 2 to 16 workers.  It does, and it means nothing: `getrusage(RUSAGE_CHILDREN)`
    # defines `ru_maxrss` as the largest SINGLE child, never the sum across the tree, so flat
    # is the only answer that metric can give however many workers run.  Demonstrated
    # directly -- 1, 2 and 4 concurrent children each touching 150 MiB all report 172 MiB.
    #
    # Measured properly since, by sampling summed RSS across the process TREE
    # (`sweep_pricing_workers.py`), `density_faa_wing_zipline` x50, 2 iterations, greedy off:
    #
    #     workers   peak tree RSS   marginal/worker   speedup   schedule sha
    #           0        3,953 MB              --       1.00x   b6ddd8d9af579126
    #           2        7,100 MB        1,573 MB       2.27x   b6ddd8d9af579126
    #           4       12,536 MB        2,146 MB       3.50x   b6ddd8d9af579126
    #           8       22,688 MB        2,342 MB       3.89x   b6ddd8d9af579126
    #
    # LINEAR, and slightly worse than linear per worker.  Nothing about the lazy arena made
    # the pool cheap; it made one METRIC stop reporting the cost.  4 workers is 12.5 GB at
    # FIFTY flights, so on the 4 GB/core node this has to run on, the default cannot be 4 --
    # and an OOM-killed worker does not fail the sweep, it HANGS it (see `pricing_pool`), so
    # the failure mode is silence rather than a traceback.
    #
    # Speed is not the obstacle and never was: 3.50x at 4 workers here, better than the 2.36x
    # measured while the greedy still ran, because turning that serial stage off lifted the
    # Amdahl cap.  The schedule sha is identical across all four arms, so the pool is
    # answer-identical on a sweep that finishes, exactly as claimed.  This is purely a memory
    # budget, and raising the default needs a cap on aggregate live RSS -- not another
    # speedup table.
    #
    # DO NOT cite the 0.66-0.74x "parallel loses on density" figures here, as an earlier
    # draft of this comment did.  Those measure the A* SPECULATIVE PARALLEL RUNNER
    # (`analysis/bench_parallel.py`, `--mode exact`, where the serial commit floor dominates
    # a compiled per-flight plan) -- a different mechanism with a different bottleneck.  The
    # colgen pricing pool has no such result.
    #
    # ONE CAVEAT THAT SURVIVES.  A pool is answer-identical to the sequential loop only on a
    # sweep that FINISHES.  `pricing_deadline` is a wall clock, so a pool gets further
    # through `pricing_order` before the same absolute instant and keeps a LONGER accepted
    # prefix -- more pricing inside the same budget, but a different column set.  With
    # `time_limit_s = 1200.0` shipped and a 500-flight solve at ~509 s, there is margin, but
    # a parity comparison must still pin this to 0 on both arms.  See `pricing_pool`.
    #
    # THIS IS THE ONLY HOME FOR THE SETTING.  It used to be mirrored by a separate
    # `ParallelPricingConfig` dataclass that `solve` took as a `parallel=` keyword, so the
    # count had two defaults that could disagree -- and did: `batch.py` turned an explicit
    # 0 into `None`, which `price_sweep` resolved as "whatever the dataclass defaults to"
    # rather than "sequential".  `price_sweep` already receives this params object, so the
    # second one carried no information.
    n_pricing_workers: int = 0
    # `mp.Pool.imap`'s third argument.  1 is almost certainly right and raising it is a
    # trap: chunking amortizes DISPATCH, and a pricing task ships one int in and ~14 KB out
    # against tens of seconds of compute, so there is nothing to amortize.  What it costs
    # instead is load balance -- `mp.Pool` pre-partitions the iterable, per-flight cost
    # varies several-fold, and n/chunksize chunks across n_pricing_workers lanes leaves
    # nothing to rebalance a straggler against.  Kept configurable so that claim stays
    # measurable rather than merely asserted.
    pricing_chunksize: int = 1
    # How many ROOTS the pricing BOOTSTRAP searches before the real search starts, taken in
    # descending order of `PreparedVariants.score`.  0 disables it; PR #76 uses 1.  A root is
    # one `(departure_step, origin lane)` pair, so this is a count of start options and not
    # of departure times.
    #
    # Ranked rather than truncated, which is the whole difference between this and the
    # departure PREFIX it replaces.  A prefix takes the earliest N departures and so misses
    # an optimum that departs late -- measured: at a 4- and 8-step prefix the bootstrap
    # returned rc 4.4590 against an 8.4590 optimum on the second sweep and the main search
    # declined anyway, while `score` is the root's own upper bound and ranks the promising
    # departure first whenever it is.  It is also far cheaper: a prefix of N covers
    # `N * n_origin_lanes` roots (240 at N=16 on the density straggler) where this covers
    # exactly `bootstrap_roots`.
    #
    # It exists because the cutoff pricing enters with is not merely weak, it is
    # structurally ZERO -- and that is a property of column generation, not a bug.  Every
    # column `price_flight` can reach for free (the geodesic seed, its time translations,
    # the master's `known_column`) is already in the master's pool, and LP optimality over
    # that pool forces every pool column's reduced cost <= 0; complementary slackness pins
    # the basic one at exactly 0.  Measured on `density_faa_wing_zipline` x12: `entry_rc` is
    # 0.0000 for all twelve flights in both sweeps, against optima of 8.5-20.5.  Re-scoring
    # a flight's own previous PRICED column under the next iteration's duals gives 0.0 or
    # negative (-5.9211 observed), so mining the pool cannot help either.  The only way to a
    # positive cutoff before the search is to SEARCH for one.
    #
    # What that cutoff is worth, measured by handing each flight its own optimum (the oracle
    # bound, `analysis/ab_colgen_oracle_cutoff.py`): sweep-1 task total 519.7 s -> 5.3 s,
    # 98x.  The flight that exhausts the label pool goes from 33.5M labels plus a 418 s
    # Python fallback to 4.1M labels in 4.53 s, i.e. it then fits under the SHIPPED ceiling
    # with 8x headroom.  Two flights are pruned to zero labels at the root gate.
    #
    # DEPARTURES is the only restriction axis, and that is a choice.  The other candidate --
    # a tighter hop budget -- sizes the corridor at `build_flight_graph`, so varying it means
    # rebuilding the graph per bootstrap, which costs more than the search it saves.
    # Departures are a slice of an array that already exists, and they are where the leverage
    # is: 157 of ~901 possible departures survive the root gate on the straggler, and the
    # oracle collapses that to between 0 and 16.
    #
    # Answer-affecting, and optimality-safe.  Pruning against a certified achievable score
    # never discards anything strictly better, so the optimum is unchanged -- but under
    # dominance a tighter cutoff can return a DIFFERENT equally-optimal column, so this moves
    # the `ab_colgen_parity.py` sha and has to be re-baselined deliberately.
    #
    # NOW 1, RE-BASELINED 2026-08-13.  Measured x50, 2 iterations, sequential, `total_cost`,
    # greedy OFF -- i.e. at the shipped configuration:
    #
    #     instance            off       K=2/score      K=1/bound
    #     density_faa      263.40 s      197.87 s       80.79 s     3.26x over off
    #     density_future   231.25 s      179.49 s      103.32 s     2.24x over off
    #
    # K=1 is enough ONLY with `bootstrap_ranking="bound"`; at `"score"` it provably fails
    # (`entry_rc` stays at exactly 0.0000) and 2 is the floor, which is why the two defaults
    # move together.  PR #76's literal K=1 is too weak on a 157-root instance ranked by
    # score -- the ranking is what makes one root sufficient.
    #
    # THE BOOTSTRAP AND THE GREEDY ARE SUBSTITUTES, and that is why this is worth having.
    # Both exist to hand pricing a `known_column` cutoff.  The greedy costs ~57% of wall for
    # 0.129% of objective and is off by default; the pool cannot supply one at all
    # (`entry_rc` is structurally 0, forced by LP duality).  With the greedy off the
    # bootstrap is the ONLY cutoff source, which is where the 3.26x comes from -- and it is
    # why a harness that pins the greedy on measures this change at roughly zero.
    # `ab_colgen_parity.py` does exactly that (`--greedy-budget` defaults to 1e6), so its
    # timings are not the ones to read for this knob.
    #
    # ANSWER-AFFECTING IN PRINCIPLE, ANSWER-NEUTRAL IN MEASUREMENT.  Pruning against a
    # certified achievable score cannot discard anything strictly better, so the optimum is
    # safe by construction; what a tighter cutoff CAN do under dominance is return a
    # different equally-optimal column.  On every arm measured it does not: objective and
    # full schedule sha are identical across the whole grid above, and the parity shas are
    # unchanged from the pre-bootstrap baseline (`1cb183616dceb2a4` / `cb1e9afa2f31bdf1` /
    # `45b35d203b1b8d47`).  Treat it as answer-affecting anyway -- the guarantee is about
    # optimality, not about column identity.
    bootstrap_roots: int = 1
    # WHAT the bootstrap sorts its roots on.  "score" is `PreparedVariants.score`, pure
    # cost-so-far; "bound" is `completion_can_compete`'s own `hop_rc_bound` at the root's
    # minimum feasible hop count -- `g + h` instead of `g`, already computed by the root
    # gate, so it costs nothing to read.  Inert on single-lane flights (the two rank
    # identically, correlation 1.000) and decisive on multi-lane ones.  Ordering only: it
    # cannot prune, because the bootstrap returns an INCUMBENT and the main search still
    # fans out over every root.
    #
    # NOW "bound", and it is what makes `bootstrap_roots=1` viable: `score` carries one live
    # bit, "depart earlier" -- its other two terms are inert, `start_dual_cost` constant
    # across roots and `origin_leg` correlating ~0 with quality -- so it cannot tell a short
    # remaining route from a long one.  `hop_rc_bound` can, because it is `g + h`.  Decisive
    # exactly where roots sit on lanes of differing length; on a single-lane flight the two
    # orderings are identical and this costs nothing.
    bootstrap_ranking: str = "bound"

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

        # Zero would make every IP a no-op that silently returns the rounding incumbent,
        # which is a legitimate thing to want but not a legitimate thing to reach by
        # accident, so it has to be asked for as a positive number.
        try:
            ip_limit = float(self.ip_time_limit_s)
        except (TypeError, ValueError) as exc:
            raise TypeError("ip_time_limit_s must be a number") from exc
        if not math.isfinite(ip_limit) or ip_limit <= 0.0:
            raise ValueError(
                f"ip_time_limit_s must be finite and positive, got {self.ip_time_limit_s!r}"
            )
        object.__setattr__(self, "ip_time_limit_s", ip_limit)

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
        # Validated against a whitelist rather than resolved lazily: an unknown name here
        # would otherwise surface as a silent no-op warm start, and "colgen ran unseeded"
        # is indistinguishable in the output from "colgen was seeded and it did not help".
        if self.warm_start_planner is not None:
            if not isinstance(self.warm_start_planner, str):
                raise TypeError("warm_start_planner must be a string or None")
            if self.warm_start_planner not in WARM_START_PLANNERS:
                raise ValueError(
                    f"warm_start_planner must be None or one of "
                    f"{sorted(WARM_START_PLANNERS)}, got {self.warm_start_planner!r}"
                )

        if isinstance(self.seed_ladder_steps, bool):
            raise TypeError("seed_ladder_steps must be an integer")
        try:
            ladder = operator.index(self.seed_ladder_steps)
        except TypeError as exc:
            raise TypeError("seed_ladder_steps must be an integer") from exc
        if ladder < 0:
            raise ValueError("seed_ladder_steps must be non-negative")
        object.__setattr__(self, "seed_ladder_steps", ladder)

        if isinstance(self.seed_route_variants, bool):
            raise TypeError("seed_route_variants must be an integer")
        try:
            variants = operator.index(self.seed_route_variants)
        except TypeError as exc:
            raise TypeError("seed_route_variants must be an integer") from exc
        # 1, not 0, is the floor: the nominal seed is not optional, so "zero routes" has no
        # meaning here the way "zero extra departures" does for the ladder.
        if variants < 1:
            raise ValueError("seed_route_variants must be at least 1")
        object.__setattr__(self, "seed_route_variants", variants)

        if not isinstance(self.seed_fan_congestion_prior, bool):
            raise TypeError("seed_fan_congestion_prior must be a bool")

        if isinstance(self.pricing_tied_columns, bool):
            raise TypeError("pricing_tied_columns must be an integer")
        try:
            tied = operator.index(self.pricing_tied_columns)
        except TypeError as exc:
            raise TypeError("pricing_tied_columns must be an integer") from exc
        if tied < 1:
            raise ValueError("pricing_tied_columns must be at least 1")
        object.__setattr__(self, "pricing_tied_columns", tied)

        if isinstance(self.n_pricing_workers, bool):
            raise TypeError("n_pricing_workers must be an integer")
        try:
            workers = operator.index(self.n_pricing_workers)
        except TypeError as exc:
            raise TypeError("n_pricing_workers must be an integer") from exc
        if workers < 0:
            raise ValueError("n_pricing_workers must be non-negative")
        # An upper bound, because the failure past it is not an error message.  Each worker
        # rebuilds every graph and carries its own label pool -- roughly 1.5 GB on density
        # -- so an over-large count from a config file OOMs the host rather than running
        # slowly.  Measured, more lanes stop paying long before here anyway: 8 and 12
        # workers were within noise of each other, because added lanes add memory-system
        # contention as fast as they add throughput.
        ceiling = 4 * (os.cpu_count() or 1)
        if workers > ceiling:
            raise ValueError(
                f"n_pricing_workers={workers} exceeds {ceiling} (4x this host's "
                f"{os.cpu_count()} cores); each worker holds its own label pool"
            )
        object.__setattr__(self, "n_pricing_workers", workers)

        if isinstance(self.pricing_chunksize, bool):
            raise TypeError("pricing_chunksize must be an integer")
        try:
            chunksize = operator.index(self.pricing_chunksize)
        except TypeError as exc:
            raise TypeError("pricing_chunksize must be an integer") from exc
        if chunksize < 1:
            raise ValueError("pricing_chunksize must be positive")
        object.__setattr__(self, "pricing_chunksize", chunksize)

        if isinstance(self.bootstrap_roots, bool):
            raise TypeError("bootstrap_roots must be an integer")
        try:
            bootstrap = operator.index(self.bootstrap_roots)
        except TypeError as exc:
            raise TypeError("bootstrap_roots must be an integer") from exc
        if not isinstance(self.bootstrap_ranking, str):
            raise TypeError("bootstrap_ranking must be a string")
        _ranking = self.bootstrap_ranking.lower()
        if _ranking not in {"score", "bound"}:
            raise ValueError("bootstrap_ranking must be 'score' or 'bound'")
        object.__setattr__(self, "bootstrap_ranking", _ranking)

        if bootstrap < 0:
            raise ValueError("bootstrap_roots must be non-negative")
        object.__setattr__(self, "bootstrap_roots", bootstrap)

        if isinstance(self.greedy_budget_s_per_flight, bool):
            raise TypeError("greedy_budget_s_per_flight must be a real number")
        try:
            per_flight = float(self.greedy_budget_s_per_flight)
        except (TypeError, ValueError) as exc:
            raise TypeError("greedy_budget_s_per_flight must be a real number") from exc
        if not math.isfinite(per_flight) or per_flight < 0.0:
            raise ValueError("greedy_budget_s_per_flight must be finite and non-negative")
        object.__setattr__(self, "greedy_budget_s_per_flight", per_flight)

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
