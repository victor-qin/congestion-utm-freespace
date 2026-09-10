# Column-generation performance changes

Work is isolated on `codex/colgen-ip-profiling` in `/private/tmp/congestion-colgen-lns-20260905`. Main and its pre-existing edits are unchanged.

## Implemented behavior

1. **Terminal-lane corridor correction.** Corridor membership now measures reachability from actual cruise-lane endpoints. The hop and time ceilings and foreign terminal exclusions remain enforced. A real 127-hop route, previously excluded by the terminal-center ellipse despite its 128-hop budget, is a regression fixture in `tests/test_colgen_network.py`. See [the corridor figure](figures/colgen_lane_aware_corridor.png).
2. **Departure-alternative certificates.** A bounded graph-local cache validates each spatial route once. Integer departure shifts reuse its canonical claims when endpoint-window rounding is translation invariant. Otherwise claims are rebuilt using the actual endpoint windows while retaining the spatial certificate. This guard matters for nonbinary time steps and large clocks; naive unconditional shifts are not correct. Departure/arrival limits and route/configuration identity are still checked.
3. **Goal-directed bootstrap with DP refinement.** `bootstrap_method="astar"` first seeks a feasible route with at most 20,000 expanded heuristic labels, then supplies it to the existing restricted-root DP. The unrestricted DP still certifies pricing over all roots. `bootstrap_method="dp"` retains the comparison implementation. No heuristic label or dominance decision enters either DP.
4. **Optional multiple columns.** `columns_per_flight=3` returns the winning column plus at most two distinct, positive-reduced-cost, canonically certified surviving sink columns from the same completed search. Default is one. This is not an exhaustive k-best search. Extra columns enter the restricted master but are not double-counted in the pricing bound.
5. **Persistent diagnostics.** Benchmark callbacks atomically save per-round compressed flight records and summaries, including task/stage times, goal expansions, DP label counts, kernel status, and worker load. Raw records accompany aggregate timings. A run interrupted within a pricing sweep has no completed-sweep callback for that sweep.

## A performance regression found and fixed

The first implementation used the first goal from the bounded heuristic directly as the full DP's starting cutoff. Four sampled flights looked much faster, but the full benchmark exposed a tail regression: flight 1640 exhausted the 67-million-label arena and entered Python fallback. That attempt was stopped and its source/logs preserved.

The diagnostic capture found two failure modes. Flight 2282 exhausted the heuristic budget without improving its cutoff; restricted DP found reduced cost 23.113. Flight 1640 found a valid but weak first goal with reduced cost 1.537; restricted DP found 32.0. Falling back only when the heuristic finds nothing would miss the second case. Always refining through restricted DP fixes both cases while preserving much of the measured savings on expensive flights.

The revised first-round replay covered all 1,537 flights with 12 workers in 85.615 seconds, versus 164.826 seconds for the corrected-corridor DP benchmark. There were no native label restarts or fallbacks. All 1,246 recorded reduced costs and the positive-RC sum matched. All 1,226 returned columns passed independent cold-graph claim, cost, and RC checks; all 291 early exits were independently certified. Twenty-three routes had different claim sets but equal objective cost and reduced cost. Consequently later column pools, iterations, and schedules may differ. This first-sweep replay includes worker launch and graph preparation; it is not a repeated end-to-end experiment.

Evidence: `analysis/colgen_bootstrap_tail_20260910/{report.md,full_hybrid_sweep/correctness.json}`. The original identity-parity failure remains in `full_hybrid_sweep/parity.json`; certification and objective agreement are recorded separately.

## Native DP findings

The opt-in analysis profiler instruments a generated Numba twin. Instrumentation is absent from production execution. Four sampled pricing results matched the uninstrumented implementation exactly; six targeted native checks passed with the instrumentation.

On flight 934, native DP accounted for 11.999 of 12.420 seconds. Counters found approximately 10 million full-path comparisons, 1.54 billion materialized path nodes, 6.45 million dominance comparisons, and 9.27 million hash lookups. Hash probing averaged 1.01–1.72 probes per lookup across four samples. Repeated full-path reconstruction for ties is the clearest next optimization target. These counts do not establish a percentage of native runtime per operation. Instrumentation added 9–20% in these samples; reported baseline timings exclude it.

Evidence and reproduction command: `analysis/colgen_pricing_profile_20260909/native_report.md`.

## Controlled full-run protocol

FAA seed 0, 1,200 seconds of demand, paired stream filtered to 1,537 outbound flights while retaining IDs and desired times; 182 static terminals; 12 pricing workers and 4 Gurobi threads. Up to 30 CG iterations, LP gap tolerance 0.001, 30-second IP after every completed pricing round, and a final IP allocation of 600 seconds with 660 seconds reserved. Total solver cap is 14,460 seconds. A pool-optimal IP can finish before consuming its allocation. Each arm runs serially in a fresh process with immutable source hashes; JIT warmup is outside the simulation clock. There is one demand seed and one trial per arm.

`analysis/colgen_todos_1200s_20260910` preserves the completed baseline, corrected corridor, and certificate-reuse arms, plus the interrupted heuristic-only attempt. `analysis/colgen_todos_hybrid_1200s_20260910` holds the revised bootstrap, combined changes, wider corridor, and multiple-column arms. Wider6 changes the extra-hop budget from 3 to 6, enlarging both spatial and time domains. Multi3 keeps the usual corridor and requests up to three columns per flight.

Completed initial arms: baseline 1,687.113 seconds / cost 156,714.567; corrected corridor 1,467.333 seconds / cost 156,900.319; certificate reuse 1,374.086 seconds / the same cost 156,900.319. All accepted and verified every flight. Certificate reuse cut seeding from 128.920 to 58.896 seconds (54.3%) and total time by 6.36% against the corrected corridor. The corridor correction changes the feasible domain and is not a pure speed comparison. Final costs refer to the penalized congestion objective, excluding the constant mandatory-altitude cost.

Consolidate completed results, including preserved superseded attempts, with:

```sh
.venv/bin/python analysis/summarize_colgen_todos.py analysis/colgen_todos_1200s_20260910 --additional-directory analysis/colgen_todos_hybrid_1200s_20260910 --out analysis/colgen_todos_report_20260910
```

Use the main checkout's virtualenv when executing from the temporary worktree.

## Completed results

| Arm | Wall minutes | Congestion cost | CG rounds |
|---|---:|---:|---:|
| Original baseline | 28.12 | 156,714.57 | 13 |
| Corrected corridor | 24.46 | 156,900.32 | 11 |
| Certificate reuse only | 22.90 | 156,900.32 | 11 |
| Refined bootstrap only | 22.96 | 156,817.30 | 11 |
| Combined changes | 22.09 | 156,817.30 | 11 |
| Combined, six extra hops | 21.79 | 156,817.30 | 11 |
| Combined, up to three columns | 21.45 | 156,869.55 | 10 |

All seven completed arms accepted and conflict-verified all 1,537 flights, reached the 0.1% LP stopping tolerance, and proved their final restricted-pool IP optimal. Final IP wall time was 17–24 seconds, despite the 600-second allocation. The combined arm's gap to the full configured pricing-domain bound remains 1.18%; restricted-pool optimality is not global integer optimality.

The combined changes reduced total runtime by **9.69% against the corrected corridor**, with cost **0.053% lower**. Against the original domain, runtime was **21.45% lower**, while cost was **0.066% higher**. The original and corrected corridors reached the LP stopping tolerance with different finite column pools and iteration counts; expanding the domain does not ensure a better final finite-pool IP at the same LP stopping tolerance.

Certificate reuse is the clearest isolated gain: seeding fell from 128.920 to 58.896 seconds with identical final cost. The refined bootstrap alone cut total pricing time by 10.08% and total runtime by 6.11%. Its large first-round reduction did not translate proportionally to every later round.

**Current default: six extra hops, as explicitly requested by the user on 2026-09-10; one column per flight remains the default.** This supersedes the earlier recommendation to retain three hops. The historical measurements below retain their original explicit three- and six-hop settings. The wider run produced exactly the same 42,879 timed columns and costs; none exceeded the standard hop cap. It expanded 0.15% more bootstrap labels and 0.11% more main-search labels. Its 1.35% lower measured wall time is not evidence of an algorithmic speedup.

Three-column pricing grew the pool to 61,717 columns (44% larger) and 30,749 spatial routes, compared with 11,962 spatial routes normally. It needed one fewer round and finished 2.88% sooner, but final cost was 52.25 higher (0.033%). Of its timed columns, 35,950 were shared with the standard run; 25,767 were new and 6,929 standard-run columns were absent. This experiment is not a superset comparison because extra columns change subsequent duals and pricing decisions. It does not establish a clear overall improvement.

These are single trials. In combined rounds 2 and 4, every flight had exactly the same DP label counts as bootstrap-only, yet measured worker time was about 30% higher; small timing differences must be interpreted cautiously. All comparisons ran serially with identical demand hashes, 12 pricing workers, and 4 IP threads.

The benchmark IP log originally printed route cost without the rejection penalty. One intermediate multi3 incumbent covered 1,536 flights, so its true objective was 10,000 above that log value. Completed callbacks and final results correctly included the penalty. The tracing helper now labels route cost explicitly and records coverage and penalized cost. Recorded benchmark sources and original artifacts remain preserved; this logging clarification was made after measured runs.

The consolidated tables, stage timings, worker diagnostics, source hashes, and preserved failed attempt are in `analysis/colgen_todos_report_20260910/report.md` and `summary.json`. The exact pool comparison is in `domain_pool_audit.json`. The next native-DP optimization target is repeated path reconstruction during tie comparisons; this task profiled that work rather than changing the kernel.

Final validation: all **380 targeted tests passed** on the final implementation, including the 127-hop regression, shifted-certificate cases, heuristic failure recovery, and +9-hop native/Python parity. The only warning was emitted by the deliberate worker-death test. Lint and diff checks passed; `final_source_verification.json` confirms production matches the measured combined snapshot and main is unchanged.
