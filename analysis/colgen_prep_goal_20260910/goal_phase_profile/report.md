# Warm round-11 goal phase profile

All six selected flights have exact baseline/preparation/combined output, claim, cost/RC and work parity. Goal entry/output/expansion signatures also match. Every goal exhausts 20,000 expansions and returns its original incumbent. Every selected root has **zero paid rows**.

Selection: top six `bootstrap_goal_s` records from the frozen baseline round-11 replay. Each arm uses the same captured caller inputs; one untimed graph warm solve, three ordinary warm repeats, then one separate instrumented solve. No production source changed.

| Flight | Baseline / prep / combined ordinary goal medians (s) | Combined native (ms) | Host sink callback (ms / %) | Sink calls |
|---|---:|---:|---:|---:|
| 1186 | 2.6310 / 2.6153 / 2.6669 | 67.08 | 2444.42 / 90.05% | 3751 |
| 3122 | 2.3639 / 2.3689 / 2.3884 | 79.14 | 2149.76 / 89.25% | 4694 |
| 1104 | 2.0784 / 2.0656 / 2.0817 | 68.41 | 1847.42 / 89.65% | 4271 |
| 592 | 1.9346 / 1.9412 / 1.9034 | 76.08 | 1716.82 / 88.79% | 4936 |
| 974 | 1.9841 / 1.9738 / 1.9253 | 70.12 | 1731.07 / 89.30% | 3186 |
| 1480 | 1.8371 / 1.8320 / 1.8420 | 55.73 | 1660.66 / 89.58% | 2391 |

| Six-flight sum | Baseline | Preparation | Combined |
|---|---:|---:|---:|
| sum_median_price_s | 14.113692 | 14.017883 | 13.867727 |
| sum_median_goal_s | 12.829024 | 12.796863 | 12.807780 |
| goal_profile_s | 12.942440 | 12.927776 | 12.909846 |
| native_advance_in_goal_profile_s | 0.000000 | 0.000000 | 0.416572 |
| canonical_sink_in_goal_profile_s | 11.671804 | 11.650711 | 11.550159 |
| pack_duals_in_goal_profile_s | 0.000000 | 0.000000 | 0.038847 |
| pack_duals_outside_goal_profile_s | 0.081449 | 0.040575 | 0.000000 |
| destination_cost_in_goal_profile_s | 0.018899 | 0.019688 | 0.000013 |
| other_goal_host_profile_s | 1.251736 | 1.257377 | 0.904256 |
| sink_calls | 23229.000000 | 23229.000000 | 23229.000000 |

Across these selected tails, native `_advance` is **3.23%** of instrumented goal time; host sink callbacks are **89.47%**. Packing contributes 0.30%. The proposed paid-row range guard cannot address this bottleneck: the CSR is empty in every selected root, and the entire native expansion phase is already small.

The sink callback is `_sink_certifier`: endpoint/path claim construction, delay calculation, dual summation and incumbent screening, followed by canonicalization only if promising. Callback counts are **not** canonical translation counts. The profile does not separately attribute these internal screening steps.

Instrumentation wall/ordinary median ratios range 0.9671–1.0180; wrappers add overhead, especially to frequent baseline destination-cost queries. Ordinary medians are the timing evidence; instrumented phase times explain where time is spent. Inclusive goal time contains all its children, and exclusive host time is its remainder. Do not add inclusive totals to their components.

In the combined version, native goal dispatch moves packed-dual construction from the later DP stage into the goal phase; the inside/outside rows show this change in attribution. This six-flight expensive-tail sample does not establish population-wide phase percentages or statistical speedup confidence.

Reproduce: `.venv/bin/python analysis/colgen_prep_goal_20260910/profile_goal_phases.py --out <fresh-directory>`; fixed source roots and capture/selection defaults are recorded in the helper. Raw ordinary trials, separate profiles, hashes and exact parity are in `comparison.json` and each arm’s `profile.json`.
