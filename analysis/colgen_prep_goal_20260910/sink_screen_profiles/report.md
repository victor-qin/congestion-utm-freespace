# Native goal sink screen: bounded warm comparison

Same captured round-11 inputs, six tails selected by prior baseline goal time; one warmup, three ordinary warm trials, then a separate nested-instrumentation trial per flight. Arms ran serially in isolated processes; compilation excluded. Frozen old combined versus screened sources.

Every final output, identity, claims, cost/reduced-cost and search-work signature matches exactly. Goal entry, result and 20,000-expansion count also match for every flight. Callback counts intentionally differ. This is a selected-tail result, not a full-solver speedup estimate.

| Flight | Total old → screened (s) | Goal old → screened (s) | Skipped / asked labels |
|---|---:|---:|---:|
| 1186 | 2.8214 → 0.2442 | 2.6033 → 0.0397 | 3751 / 0 |
| 3122 | 2.5563 → 0.2740 | 2.3429 → 0.0580 | 4684 / 10 |
| 1104 | 2.1835 → 0.2028 | 2.0275 → 0.0417 | 4271 / 0 |
| 592 | 1.9968 → 0.1594 | 1.8822 → 0.0450 | 4936 / 0 |
| 974 | 1.9770 → 0.1671 | 1.8623 → 0.0523 | 3185 / 1 |
| 1480 | 1.9833 → 0.2589 | 1.7784 → 0.0470 | 2387 / 4 |

Sum of per-flight ordinary medians: total -90.34%; goal -97.73%.

Separate instrumented inclusive goal components (seconds; nested rows must not be added twice):

| Component | Old combined | Screened |
|---|---:|---:|
| goal | 12.659154 | 0.283783 |
| canonical_sink_in_goal | 11.354693 | 0.007852 |
| path_claims_in_goal | 6.741824 | 0.004429 |
| path_delay_s_in_goal | 3.350692 | 0.002253 |
| claim_cost_in_goal | 0.726476 | 0.000776 |
| canonical_candidate_in_goal | 0.000000 | 0.000000 |
| native_advance_in_goal | 0.401910 | 0.232763 |
| pack_duals_in_goal | 0.039093 | 0.039492 |

Goal sink callbacks: 23229 → 15; canonical-candidate calls inside goal: 0 → 0.

The conservative native bound screens before host claim construction and delay computation. It does not compile canonical geometry, use queue priority as a pruning bound, or reduce expansions. The positive paid-cost lower bound remains zero; near-cutoff uncertainty still asks Python.

Validation: 32 goal-kernel tests passed in 2.16 s, including near-cutoff, negative dual, invalid sink/resume and strong-to-weak cached-envelope cases. Frozen source differs only in bootstrap_kernel.py (screen) and pricing.py (diagnostics).

Instrumentation wall/ordinary-median ratios: old_combined 1.001–1.018; screened 0.980–1.055. Ordinary timings are authoritative.

Reproduce each arm with profile_goal_phases.py --source-root <frozen-source> --out <fresh-output>; default capture and six-flight selection paths match this report.
