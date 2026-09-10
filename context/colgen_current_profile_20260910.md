# Current branch and pricing profile

This summarizes `codex/colgen-ip-profiling` at `30ffaf5` against `main`. Metrics
come from the completed 1200-second FAA outbound candidate experiment, not a new
simulation: 1,537 flights, 11 rounds, 12 pricing workers, four Gurobi threads.

## Branch changes

- Correct global cost-gap reporting, with separate restricted-pool IP gaps;
  IP every round is now enabled by default, and LP tolerance defaults to 0.001.
- Optional incumbent-preserving LNS, cheap pricing with exact certification sweeps,
  imported/FCFS initialization, centered departure alternatives, and multiple columns.
- Correct corridor pruning around terminal exit lanes, including the 127-hop regression;
  the extra-hop default is six.
- Reuse spatial certificates and shift capacity claims across departure alternatives,
  with timing and endpoint-rounding checks.
- Goal-directed bootstrap followed by restricted DP refinement and unrestricted exact
  DP; equal-depth native tie comparisons now walk parent chains without full path copies.
- Per-flight diagnostics, source snapshots, benchmarks, correctness audits, tests,
  explanations, and figures.
- Transferred documentation/annotation cleanup and two small non-CG reporting fixes:
  nonnegative deconfliction detour and attempted ground delay on rejected straight flights.

LNS and cheap pricing remain off by default; one column per flight and nominal routes
plus 20 departure alternatives remain the defaults. Twelve workers and the 600-second
final IP allowance are benchmark settings, not production defaults.

## What the worker's bootstrap does

For each flight and current LP dual prices, pricing first evaluates cached/shifted seed
routes and the existing incumbent. Bootstrap then prepares completion bounds and ranks
feasible `(departure step, origin lane)` starting options. The current setting keeps one
option (`bootstrap_roots=1`, ranking by completion bound).

1. A Python goal-directed search explores up to 20,000 labels from that option, using
   route cost and priced capacity claims to seek a useful feasible route.
2. Native DP restricted to the selected start refines that route and its reduced-cost
   cutoff. Restricting the start still allows many spatial paths.
3. The resulting route/cutoff is passed to the unrestricted native DP, which considers
   all eligible starts and certifies pricing.

Using the code's maximization convention, a feasible route establishes

\[
B=M-c_f(p)-\pi_f-\sum_r\lambda_r a_{rfp}.
\]

Here `M` is the benefit for covering the flight, `c` its cost, `pi` the flight dual,
and `lambda` the resource dual prices. The exact DP can prune a partial path whose
optimistic completion score is below `B`, subject to the existing tolerance/tie rules.
For example, an achieved score of +20 rules out a branch whose best possible score
is +15. A feasible bootstrap route is a bound; it is not an optimality certificate.

Bootstrap returns early when no useful search is needed, including at most one eligible
start, where the unrestricted search already covers the same domain. Preparation can
still cost time before this is discovered. There were 9,011 zero-expansion bootstrap
calls taking 372.12 summed worker-seconds in this run.

## Measured per-flight pricing latency

All rows below use the same 16,907 flight-round calls; skipped stages contribute zero.
Measurements are elapsed durations inside workers, not OS CPU accounting.

| Per flight-round call | Mean | Median | p95 |
|---|---:|---:|---:|
| Whole worker pricing task | 335 ms | 258 ms | 930 ms |
| Main compiled-search path | 171 ms | 156 ms | 334 ms |
| Bootstrap, including its preparation | 160 ms | 65 ms | 681 ms |

Whole-task p99 is 1.535 seconds and maximum is 4.208 seconds. There were 15,580 calls
entering the main search; the remaining 1,327 exited earlier. Among the 15,580 calls,
main search averaged 185 ms and bootstrap 174 ms. The first round averaged 604 ms per
worker task; the last averaged 280 ms.

`compiled_s` includes Python packing/preparation, candidate processing and validation.
It is not a pure native DP timer. Separate warm probes measure all `_price_dag` calls
for a flight, including bootstrap DP and final DP: flight 1520 took 0.080 s, 934 took
0.190 s, 1600 took 0.353 s, 1640 took 1.287 s, and 2282 took 1.972 s. These selected
cases do not establish a population mean or percentile for pure native DP.

## Where time goes

| Full simulation component | Wall seconds | Share |
|---|---:|---:|
| Pricing, including parent validation | 641.64 | 51.1% |
| Per-round and final IP phases | 363.72 | 29.0% |
| LP solving and row separation | 156.41 | 12.5% |
| Initial seeding | 56.98 | 4.5% |
| Other overhead and verification | 37.45 | 3.0% |

Within the 5,658.76 summed worker-seconds, main compiled-search work accounts for
51.0%, bootstrap for 47.9%, and other work for 1.1%. The Python goal search consumes
742.22 worker-seconds, or 13.1% of all worker time, inside the bootstrap share.
These worker totals overlap in wall time and must not be added to the table above.

Parent-side certification consumes 82.46 wall seconds inside pricing, about 6.6% of
the complete run. Worker bootstrap and parent validation are separate sources of cost.

In the six instrumented native probes, full path materialization is zero, but the new
comparator still visits 300.74 million parent pairs. There are also 32.75 million layer
comparisons and 30.85 million recent-history fills. These are operation counts, not
time percentages. Hash lookups average only 1.052 probes in this sample, making hash
collisions a less compelling target without contrary timing evidence.

## Next improvements to test

1. Reuse pricing preparation across bootstrap ranking, restricted DP, and full DP.
   Spatial topology is already cached; current-dual packing, completion envelopes and
   root preparation are still repeated. Separate reusable data from incumbent-dependent
   pruning so a tighter cutoff does not silently change correctness or tie ordering.
2. Compile the goal-directed bootstrap over packed arrays, reducing Python heap,
   full-path tuple copying and claim-object work. Retain restricted DP refinement:
   the earlier first-goal-only experiment failed on difficult flights.
3. Reduce duplicate parent certification using cached spatial results and faster claim
   construction, retaining canonical time and resource checks. The measured target is
   82.46 serial seconds, not the entire pricing stage.
4. Profile native layer sorting, recent-history comparisons and remaining parent walks
   before selecting another native data-structure change. Exact prefix ordering might
   shorten comparisons, but needs memory and deterministic-tie validation.
5. Separately test shorter early-round IP budgets while keeping IP every round and a
   strong final solve. IP uses 29% of wall time, but different incumbents can change
   later bootstrap work; this is a quality/runtime tradeoff, not a free optimization.

Wider corridors and three-column pricing have already been benchmarked. The wider
case gave no quality improvement in that sample; three columns gave a small timing
gain but slightly worse final cost. Neither is an established better default.

Evidence: [full experiment](colgen_dp_full_experiment_20260910.md),
[native probes](colgen_dp_path_comparison_20260910.md), and
[aggregate profile](../analysis/colgen_pathcmp_full_1200s_20260910/current_profile.json).
