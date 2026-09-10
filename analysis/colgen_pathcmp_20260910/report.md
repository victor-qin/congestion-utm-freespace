# Native path-comparison benchmark

Both sources use the current six-extra-hop corridor and hybrid A* plus restricted-DP bootstrap. They differ only in `dp_kernel.py`. Inputs are the same frozen corrected first-LP capture. This is a kernel/pricing benchmark, not a new master/IP simulation.

Each single-flight result is the median of three warm uninstrumented trials, preceded by one untimed solve on its own graph. Native timing wraps whole `_price_dag` calls. Instrumentation is a separate run and is excluded from these timings.

| Flight | Baseline total s | Candidate total s | Total change | Baseline native s | Candidate native s | Native change |
|---|---:|---:|---:|---:|---:|---:|
| 934 | 0.754 | 0.629 | -16.5% | 0.291 | 0.190 | -34.7% |
| 2692 | 0.719 | 0.594 | -17.4% | 0.293 | 0.193 | -34.1% |
| 1520 | 0.308 | 0.284 | -7.8% | 0.094 | 0.080 | -14.6% |
| 1600 | 1.075 | 0.937 | -12.9% | 0.433 | 0.353 | -18.5% |
| 1640 | 2.484 | 1.658 | -33.2% | 2.084 | 1.287 | -38.2% |
| 2282 | 3.948 | 2.507 | -36.5% | 3.408 | 1.972 | -42.1% |

The relevant baseline is the current hybrid implementation (flight 934 native about 0.29 seconds), not the older roughly 12-second full-DP-bootstrap replay in the earlier native report.

| Flight | Old materialized path nodes | New parent pairs visited | Candidate materialized nodes |
|---|---:|---:|---:|
| 934 | 37,884,680 | 12,152,362 | 0 |
| 2692 | 37,761,584 | 11,218,192 | 0 |
| 1520 | 5,995,332 | 2,143,476 | 0 |
| 1600 | 25,303,636 | 8,934,997 | 0 |
| 1640 | 286,482,868 | 113,892,967 | 0 |
| 2282 | 414,741,876 | 152,393,900 | 0 |

Path-comparison call counts and all non-path native work counters match for every compared instrumented flight: True.

A materialized node and a parent pair are different units of work; these counts explain the implementation change, not an elapsed-time attribution. The new comparison stops at a shared label ID and creates no full path arrays for these DP ties. The general unequal-depth path comparison remains available to the feasible-search frontier.

| Full sweep run | Flights | Wall s | Worker task sum s | Pool setup s | Fallbacks |
|---|---:|---:|---:|---:|---:|
| baseline_sweep | 1537 | 88.867 | 995.941 | 0.335 | 0 |
| candidate_sweep | 1537 | 83.260 | 928.822 | 0.339 | 0 |
| candidate_sweep_repeat | 1537 | 82.939 | 926.994 | 0.327 | 0 |
| baseline_sweep_repeat | 1537 | 87.593 | 981.570 | 0.326 | 0 |

Baseline sweep wall range: 87.593–88.867 seconds across 2 run(s); median 88.230 seconds.

Candidate sweep wall range: 82.939–83.260 seconds across 2 run(s); median 83.100 seconds.

Candidate sweep median change: -5.81%.

Sweep execution order is A–B–B–A (baseline, candidate, candidate repeat, baseline repeat). Regression tests ran between the first A and B, and separate instrumentation runs between the two B sweeps; none overlapped a measured run.

Sweep timings include 12-worker launch, preparation, and worker warmup; worker task sums are not wall time. Runs use the same capture and execute serially. Repeated samples expose observed variance but do not establish broad statistical confidence.

Exact parity artifacts compare reduced-cost and full-column hashes, timed path identity, canonical claim sets, cost/RC bits, and per-flight label/stage work counts. No tolerance is used for these kernel-change comparisons. Individual records also preserve arena allocation bytes; unchanged arena size does not establish unchanged whole-process RSS.

- baseline_counts: PASS, 6 flights compared.
- baseline_sweep_repeat: PASS, 1537 flights compared.
- candidate_counts: PASS, 6 flights compared.
- candidate_single: PASS, 6 flights compared.
- candidate_sweep: PASS, 1537 flights compared.
- candidate_sweep_repeat: PASS, 1537 flights compared.

Label arena bytes match on all compared single flights: True.

Reproduce using `analysis/benchmark_colgen_pathcmp.py --source-root SOURCE --capture analysis/colgen_bootstrap_tail_20260910/subproblems.pkl --out NEW_DIRECTORY`. Use `--reference BASELINE_DIRECTORY`, `--mode sweep` for the 1,537-flight pricing sweep, and `--instrument --repeats 1` for a separate work-counter run. Every output directory preserves source/capture hashes and the harness used.
