# Full experiment: native DP path comparison

The full paired experiment reduced simulation runtime from **1,356.33 to 1,256.21
seconds**, saving **100.13 seconds (7.38%)**. Pricing fell from **730.70 to 641.64
seconds**, saving **89.06 seconds (12.19%)**. These are observed results from one
serial pair on one demand seed; they are not a statistical confidence interval.

## Controlled setup

Both runs generate 1,200 seconds of FAA density demand with seed 0 and retain the
same 1,537 outbound requests, preserving their original IDs, coordinates, and clocks.
The full demand, configuration, and algorithm-parameter hashes match. Both use:

- 12 pricing workers and 4 Gurobi threads.
- Six extra hops, one priced column per flight, and the current hybrid bootstrap.
- Nominal route initialization with 20 departure alternatives and the greedy pass.
- LP cost-gap tolerance 0.001, at most 30 CG rounds, and IP search for up to 30 seconds
  each round with eager capacity-row materialization.
- A final IP search allowance of 600 seconds, 660 seconds reserved for final IP work,
  and a total solver cap of 14,460 seconds.

The baseline is frozen from `6a4c815`; the candidate production files match `f51b8b0`.
Only `freespace_sim/planner/colgen/dp_kernel.py` differs between measured sources.
Both were run through the same frozen harness, sequentially, with no concurrent
tests or other agent benchmarks. Warmup and demand generation are excluded from
simulation time; filing, final conflict verification, and diagnostic overhead are included.

## Runtime and quality

| Metric | Baseline | New comparison |
|---|---:|---:|
| Full simulation, s | 1,356.333 | 1,256.206 |
| Pricing, s | 730.703 | 641.644 |
| All iteration IP phases, s | 348.361 | 345.446 |
| Final IP phase, s | 18.144 | 18.277 |
| Final congestion cost | 156,817.302553 | 156,817.302553 |
| Accepted / denied | 1,537 / 0 | 1,537 / 0 |
| CG rounds | 11 | 11 |
| Final column pool | 42,879 | 42,879 |
| LP cost gap | 0.0852% | 0.0852% |
| Global configured-domain integer cost gap | 1.1826% | 1.1826% |

Both final schedules pass conflict verification. Both stop on the LP gap target,
with no native fallbacks. Each final IP proves optimality over its generated column
pool; neither proves global integer optimality over all allowed routes. The final IP
finishes well before its 600-second allowance.

These phase measurements are nested within the simulation total and should not be
added to it. The tenth baseline pricing sweep was unusually slow (79.95 versus
49.40 seconds), contributing 30.54 seconds of the observed 89.06-second pricing saving.
The earlier repeated fixed-dual first-sweep benchmark measured a smaller 5.81% median
pricing reduction; this full-run result should not be read as a universal 12% speedup.

## Correctness interpretation

All 11 LP objectives, recorded pool counts and additions, and LP certificates match.
The complete pools contain exactly the same 42,879 timed route identities and costs
in the same insertion order. All 15,580 recorded final reduced costs match exactly.
The first five rounds also have identical recorded per-flight pricing work and reduced
cost data. Intermediate IP selections first differ in round 5 under their wall-clock
search limits; later pricing work can differ because it receives different incumbents.
The 1,511 differing reduced-cost records contain different entry cutoffs, not different
final reduced costs. Main label counts fell only 0.0028% overall, while bootstrap
label counts rose 0.0846%; the runtime saving did not come from substantially reducing
the amount of search.

The final IP selects different routes for 568 flights from the same pool, with exactly
the same total objective. Both schedules have the same aggregate ground delay, airborne
detour distance, and 95th-percentile delay. Total delay differs only by approximately
7.3e-12 seconds from floating-point aggregation. This establishes matching aggregate
quality, not identical final trajectories. The full-run trace does not contain capacity
claims or every dual vector, so it does not establish byte-for-byte claim parity.

The underlying comparator's exact same-input behavior was separately checked with
159 passing targeted tests and the repeated fixed-dual pricing replays described in
[the comparator report](colgen_dp_path_comparison_20260910.md).

Detailed evidence, source hashes, commands, per-flight diagnostics, IP logs, and the
full comparison are under
[the experiment report](../analysis/colgen_pathcmp_full_1200s_20260910/report.md).
