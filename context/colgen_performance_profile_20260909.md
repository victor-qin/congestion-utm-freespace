# Initialization and pricing performance diagnosis

Worktree: `/private/tmp/congestion-colgen-lns-20260905`, branch `codex/colgen-lns-main`.
Production implementation and defaults have not been changed for this diagnosis.

## Nominal initialization: isolated measurements

`analysis/profile_colgen_initialization.py` regenerates and checks the exact saved
1,537 outbound requests, configuration and 182 static terminals from the completed
1200-second FAA benchmark. It runs only nominal routes and departure ladders,
using the real Gurobi restricted master. It runs no pricing or integer solve.

The full replay measured:

| Stage | Seconds |
|---|---:|
| Lazy graph setup | 0.161 |
| Nominal spatial search and certification | 22.250 |
| Nominal recheck and insertion | 1.061 |
| 30,740 delayed copies: certification and insertion | 96.076 |

The last three stages total 119.387 seconds; delayed copies account for 80.47%.
These are new isolated timings, not a decomposition of the earlier run's clock.
The earlier full benchmark measured 107.000 seconds for the combined seed loop.
The replay produced 32,277 columns (21 per flight), before any greedy additions.
There were 1,828 spatial searches and zero zero-dual DAG seed fallbacks.

A separate unprofiled 128-flight sample gave 1.665 seconds for routes,
0.067 for their insertion/recheck, and 7.568 for delayed copies. A separate
32-flight cProfile pass attributed 5.546 of 6.700 seconds to the departure ladder;
`_canonical_column` accounts for 3.451 seconds cumulatively, and `_shift_column`
1.237 seconds. These cumulative times overlap their callees and must not be added.

Source chain:

- `solver.py:968`: serial per-flight seed loop, including `_add_departure_ladder`.
- `pricing.py:491`: `_shortest_cell_path` uses a Python heap-based A* traversal.
  Several older comments still call it BFS. It does not invoke the FCFS A* kernel.
- `solver.py:223`: each delayed copy is `_shift_column` followed by
  `_canonical_column`, then master insertion.
- `network.py:1682`: certification-cache identity includes departure step.
- `network.py:1744`: each uncached departure reconstructs the operational intent,
  checks permanent walls, and builds resource claims again.
- `_shift_column` already constructs shifted claims before those claims are rebuilt.
- `solver.py:1155`: pricing workers are started later, so 12 workers do not parallelize
  nominal initialization.

The first optimization to evaluate is reuse of spatial certification across pure
time shifts, while preserving temporal bounds and verifying identical claims
against the canonical implementation. Packed numeric claim generation could
reduce Python object allocation too. A compiled nominal search is secondary to
the measured ladder cost. No such production optimization has been implemented.

A literal continuous straight-line planner is not directly interchangeable with
the current hex-hop column domain. A direct hex-grid candidate with static
validation and spatial A* fallback is a separate possible initializer.

## Comparing initialization with A*

Do not compare the normal 107-second seed-loop timer (routes plus 20 delayed
alternatives) with the A* 53.962-second FCFS planner timer alone. The A* route
converter additionally took 15.364 seconds, and its centered alternatives were
built inside the main solver after the nominal timer was skipped.

Comparable elapsed time to the start of the first CG iteration, including the A*
prepass, is 117.403 seconds for normal initialization versus 137.392 for A* centered.
This is derived from `simulation_elapsed_s - iteration_wall_s` for round one.
The A* `initial_available_s` field is not this boundary: with no initial feasible
incumbent, its first `set_heuristic` occurs later inside the first iteration.

First pricing-stage wall time was 153.868 seconds for normal versus 27.313 for A*;
first iteration completion was 323.789 versus 181.529 seconds. The first A* IP
still left three flights uncovered, whereas the first normal IP covered all 1,537.
Thus earlier first-round output is not evidence of faster total initialization.

## Pricing averages from the completed runs

Normal initialization: 19,981 flight-round tasks across 13 exact sweeps,
7,058.065 summed worker-task seconds, and 877.969 pricing-stage wall seconds:
0.35324 seconds per flight task and 0.04394 amortized elapsed seconds per task.
The first sweep averaged 0.9833 seconds per task; the last averaged 0.2546 seconds.

A* centered: 46,110 flight-round tasks across 30 exact sweeps,
11,435.295 summed worker-task seconds and 1,481.062 pricing-stage wall seconds:
0.24800 seconds per task and 0.03212 amortized elapsed seconds per task.

Worker-task time is elapsed duration inside a worker, not OS CPU time. Pricing-stage
wall includes parent certification after the worker sweep; these amortized figures
are throughput measures, not latency of one flight's search. Both runs used 12
pricing workers; neither reported a compiled-kernel fallback. The initializers
produce different LPs and duals, so their per-task times describe different searches.

Detailed capture/replay analysis is being collected under
`analysis/colgen_pricing_profile_20260909` by `analysis/profile_colgen_pricing_current.py`.
The capture uses the exact production first-sweep settings and stops immediately
after that sweep. It is not a completed simulation or an IP benchmark.


## Completed pricing capture and replays

# Current CG pricing profile — completed

All work ran in `/private/tmp/congestion-colgen-lns-20260905`. No production changes were made.

## 1. Nominal initialization

An isolated replay of all 1,537 outbound flights measured 22.250 seconds for nominal
route finding/certification, 1.061 seconds for nominal insertion/rechecking and 96.076
seconds for 30,740 delayed copies. The delayed copies account for 80.47% of this stage.
These new measurements are distinct from the original run's combined 107.000-second seed timer.

The nominal path search is a serial Python heap-based A* at
[pricing.py:491](/private/tmp/congestion-colgen-lns-20260905/freespace_sim/planner/colgen/pricing.py:491), not the compiled FCFS
A* planner. The delayed-copy loop at
[solver.py:223](/private/tmp/congestion-colgen-lns-20260905/freespace_sim/planner/colgen/solver.py:223) shifts already certified
claims, then calls the full canonical validator again. Its cache key includes departure
step, so each departure reconstructs reservations, checks static walls and rebuilds claims.
The 12 pricing workers start after initialization.

Reuse spatial certification for time translations, preserve departure/arrival checks and
verify claim equivalence before removing the repeated geometry work. A packed claim
representation is another target. A direct hex-grid route candidate with spatial A* fallback
is possible; the continuous straight-line planner is not directly the same column domain.

## 2. The comparison with A*

In the original paired benchmark, elapsed time before the first CG iteration was **117.403
seconds for normal initialization versus 137.392 seconds for A* centered**. The standalone
A* planner's 53.962 seconds excluded conversion (15.364 seconds) and centered alternatives.
Normal's 107-second seed timer included its 20 alternatives per flight.

The first pricing stage took 153.868 seconds for normal versus 27.313 for A* centered.
That, and the first integer solve, account for the earlier A* first-round result; it was
not a faster complete initialization. The first A* IP left three flights uncovered.

## 3. Pricing per flight

From the original completed normal run: **0.35324 seconds per worker task**, averaged over
19,981 flight-round tasks in 13 sweeps. Amortized pricing-stage elapsed time with 12 workers
was **0.04394 seconds per task**. The first round averaged 0.9833 seconds per worker task;
round 13 averaged 0.2546 seconds. Worker-task durations are elapsed measurements, not OS CPU
accounting. Stage wall time also includes parent certification after worker results return.

The new first-sweep capture completed all 1,537 tasks with 12 workers in 151.347
seconds, accumulating 1577.249 worker-task seconds. Worker-task distribution:

| Metric | Seconds |
|---|---:|
| Mean | 1.026 |
| Median | 0.574 |
| 95th percentile | 3.450 |
| 99th percentile | 6.480 |
| Maximum | 13.147 |

The capture reproduced **all 1,227 inserted columns**, including route, lanes, departure and
cost, and the original reduced-cost sum 29469.18220693942. No kernel fallback occurred.
The capture stops after pricing; it is not a new full simulation or IP benchmark.

## 4. Where to target the DP

The first sweep spent **83.30%** of worker-task
time in the per-flight pricing bootstrap and **15.20%**
in the subsequent exact-search stage. This bootstrap is a restricted search for a good
reduced-cost incumbent, separate from nominal initialization and the greedy ground-delay pass.

The core `_price_dag` already runs in Numba. Direct wall probes on the warm slowest flight
934 measured 11.550 native-DP seconds out of 11.957 total seconds (96.60%). Its bootstrap
expanded 9,105,040 labels; the exact search after that cutoff expanded 169,132. This is a
real search-cost target, not merely Python setup. Flight 2692 spent 2.367 native seconds
out of 2.754 total (85.93%); median flight 1520 spent 0.208 out of 0.522 seconds (39.91%).

The first target is a cheaper goal-directed pricing heuristic for the bootstrap, or tighter
valid bounds before that bootstrap expands its states, retaining the final exact pricing
search. A pricing A* would use soft resource dual prices; the existing FCFS planner instead
avoids hard ledger conflicts and is not a drop-in replacement. Reusing common preparation
and reducing Python claim/candidate work would help ordinary flights more than the slowest.
No speedup from a new implementation has been demonstrated in this task.

`compiled_s` includes Python host preparation and validation. Direct wall probes around
`_price_dag` were used to distinguish native time: cProfile attributed some native work to
unrelated geometry/thread frames in this environment and is not authoritative for that split.

## 5. More spatial routes

Doubling the overrun/corridor parameter from 3 to 6 was tested on four captured first-round
subproblems, holding their duals and known columns fixed. These are one cold and one warm
replay per setting, not repeated timing estimates or a full larger-domain CG run.

| Flight | Warm +3 seconds | Warm +6 seconds | +3 bootstrap / main labels | +6 bootstrap / main labels | Reduced cost |
|---|---:|---:|---:|---:|---:|
| 1520 | 0.522 | 0.418 | 236,823 / 47,126 | 236,823 / 47,126 | 16.00000 → 16.00000 |
| 2692 | 2.754 | 2.789 | 2,126,270 / 173,140 | 2,126,270 / 173,140 | 29.50000 → 29.50000 |
| 934 | 11.957 | 13.097 | 9,105,040 / 169,132 | 10,244,533 / 169,132 | 64.00000 → 64.00000 |
| 1600 | 5.463 | 1.032 | 89,202 / 6,706,927 | 539,735 / 640,057 | -0.00000 → 81.71483 |

The first two flights did the same label work: their cost bounds pruned the extra domain.
The worst wall-time flight expanded 12.5% more bootstrap labels and ran about 9.5% longer.
The hardest main-search flight became about 5.3 times faster and found an improving column,
because the new route gave a much stronger cutoff. Thus bigger domains can cost more, but
runtime need not increase monotonically. Later duals and more generous domains remain untested.

Returning multiple columns from the existing domain differs from widening it. It may
amortize a search, but retaining extra candidates can increase state and master-IP costs;
this diagnostic did not implement or benchmark multi-column pricing.

### Confirmed corridor-pruning inconsistency

Flight 1600's newly found route has **127 hops**, below the original **128-hop ceiling**.
It is rejected by the original corridor because cells (180,223), (180,222), (180,221) have
terminal-center distance sum 129. The actual selected exit cell (176,260) lies two hops from
the terminal's center (176,262); the DP counts its cruise hops from that exit cell.

[network.py:1531](/private/tmp/congestion-colgen-lns-20260905/freespace_sim/planner/colgen/network.py:1531) claims that the
terminal-center ellipse excludes no route within the hop budget. This counterexample shows
that claim is false for terminal exit lanes. An analysis-only replay widened corridor
membership while retaining the **original 128-hop and original time caps**, and still
found the improving route with reduced cost **81.71483362667459**. Its claims and permanent
wall geometry were independently recertified on a fresh graph.

This should be corrected or explicitly treated as an additional route-domain restriction
before interpreting the hop cap as the sole spatial bound. Existing LP certificates apply
to the implemented corridor-restricted domain. No production fix has been made here.

Evidence: [corridor counterexample](/private/tmp/congestion-colgen-lns-20260905/analysis/colgen_pricing_profile_20260909/corridor_counterexample.json),
[new route](/private/tmp/congestion-colgen-lns-20260905/analysis/colgen_pricing_profile_20260909/flight_1600_wider_column.json),
[per-flight records](/private/tmp/congestion-colgen-lns-20260905/analysis/colgen_pricing_profile_20260909/flight_records.json),
[initialization timings](/private/tmp/congestion-colgen-lns-20260905/analysis/colgen_seed_profile_20260909/full/timings.json).
