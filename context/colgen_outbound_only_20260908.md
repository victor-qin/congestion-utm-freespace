# Outbound-only CG/LNS comparison, 2026-09-08

User asked how column generation with embedded LNS performs when deliveries do
not need return flights. Work stays in `/private/tmp/congestion-colgen-lns-20260905`,
branch `codex/colgen-lns-main`; main checkout and production optimization code
are unchanged by this experiment.

The benchmark harness now supports `--outbound-only`: generate the original
paired demand first and filter requests whose `paired_outbound_id` is not None.
This preserves outbound IDs, every coordinate and clock, and random draws.
There are 763 outbound requests versus 1,526 original legs; all 182 terminals
remain. The generator flag itself is not toggled, because that can renumber IDs
or change legacy random draws. `sim.run` receives the explicit outbound subset.

Reference: `analysis/colgen_reserved_ip_20260905/long`, 600 seconds of FAA
Wing/Zipline demand, seed 0. Same 7,200-second horizon, solver parameters, package
versions, 4 pricing workers and 4 Gurobi master threads. Max 30 iterations,
14,460-second total budget, 660-second final-IP reserve, native IP cap 600 seconds,
zero IP gap target and LP target 0.01%. Ground-delay seed retained, zero rounding
calls. The historical reference uses nominal return anchoring; removing returns
removes their traffic, not a realized-return coupling constraint implemented in CG.

| Metric | With returns | Outbound only |
|---|---:|---:|
| Accepted / verified | 1,526 / 1,526 | 763 / 763 |
| Whole simulation | 8,511.12 s | 1,260.20 s |
| Complete pricing sweeps | 25 | 19 |
| Pricing wall | 5,890.15 s | 1,040.12 s |
| Native IP | 600.07 s, time limit | 2.74 s, pool optimum |
| IP stage incl. setup | 606.73 s | 4.34 s |
| LP gap | 0.00815% | 0.00447% |
| Final global gap | 1.9683% | 1.1783% |
| Restricted IP gap | 5.2423% | effectively 0% |
| Seed congestion cost | 260,047.88 | 94,127.94 |
| LNS congestion cost | 247,015.74 | 85,075.37 |
| LNS improvement vs seed | 5.01% | 9.62% |
| All LNS time | 23.13 s | 16.12 s |
| Final congestion cost | 183,520.46 | 77,927.88 |
| Final cost of same 763 outbound flights | 87,418.06 | 77,927.88 |
| Mean outbound ground delay | 13.54 s | 1.28 s |
| Mean outbound total delay | 47.21 s | 34.90 s |
| p95 outbound total delay | 105.15 s | 73.35 s |
| Final columns | 48,945 | 19,748 |
| LNS-selected priced columns | 105 | 374 |
| IP-selected priced columns | 961 | 344 |
| Median distinct LNS trials | 2/16 | 16/16 |
| Calls with just one distinct trial | 11/25 | 0/19 |
| Last improved LNS call | 16 | 17 |
| Final repair | one omitted return, 1518 | none |

Measured wall speedup is 6.75×, pricing 5.66×. Matched outbound congestion cost
decreases 10.86%; total cost across the differently sized fleets drops 57.54%.
Much of the matched cost decrease is ground delay: 10,328 → 980 cost/seconds.
Lateral cost changes only slightly: 77,090.06 → 76,947.88; neither has air holding.

LNS still only recombines pooled columns. Its 100-flight neighborhood now releases
13.1% of the fleet rather than 6.6%, so its improvement can reflect both less
traffic and a larger relative neighborhood. Distinct trials do not guarantee
optimality; the final IP improves LNS by about 8.4%. The final IP can see columns
from after the last LNS call. It solves the generated pool optimally but the
remaining 1.1783% global gap does not prove global integer optimality.

This is a single-seed comparison against an archived reference, not a repeated
contemporaneous timing trial. The sole production source difference is the
already documented warning-text correction in `batch.py`; all optimization code
is identical. Archive hashes, harness/tracer hashes, exact outbound subset,
parameter and infrastructure equality, costs of filed trajectories, and retained
IP route geometries were checked. Full simulation passed conflict verification;
no pricing-kernel fallback occurred. Ruff and `git diff --check` passed for the
benchmark/analysis changes; no new production changes require solver tests.

Artifacts and exact reproduction commands:
`analysis/colgen_outbound_only_20260908/README.md`.
Full report: `analysis/colgen_outbound_only_20260908/summary/report.md`.
Analyzer: `analysis/summarize_colgen_outbound_only.py`.

## Follow-up: LNS ablation and priced-route lengths

A matched run without LNS is complete in `analysis/colgen_outbound_only_20260908/cg_rounding`.
Use the same full command, changing `--versions fixed` to `--versions baseline`
and the output directory. Both versions import the same current worktree: baseline
only disables LNS (`lns_destroy_flights=0`) and enables existing 16-try randomized
rounding with greedy fill. It does not disable all primal heuristics or the seed.

CG+rounding takes 1,354.122 s vs CG+LNS 1,260.196 s (6.94% less with LNS in this pair).
Pricing is 1,090.227 vs 1,040.121 s; heuristic stages 53.492 vs 16.285 s;
canonicalization 12.364 vs 2.571 s; native IP 2.647 vs 2.741 s. Both native IPs are
pool-optimal and were given the full 600-second cap. Both complete 19 sweeps,
generate identical sets of 19,748 timed columns and identical LP-gap histories,
and return the exact same final trajectory SHA and cost 77,927.8754. All 763
flights verified; no repairs or pricing fallbacks. Global cost gap is 1.1783%.

Rounding never improves its complete 94,127.94 seed: all 304 trials cover only
701–743 flights. LNS preserves coverage and reaches 85,075.37 (9.62% better).
Thus LNS improves the incumbent available during pricing and saves heuristic
overhead, but does not improve the final result, reduce sweep count, or accelerate
the final IP in this case. Roughly 47 of the 94 seconds of wall difference are
heuristic/canonicalization savings; pricing also varies by 50 seconds, so do not
generalize the whole 6.94% as a robust speedup from one timing sample per arm.

Comparison script: `analysis/summarize_outbound_heuristics.py`. Audited sources,
input equality apart from the heuristic, archive/harness/tracer hashes, selected
route geometry, trajectory costs, pool equality, and final trajectory fingerprints.
Report/PNG/audit: `analysis/colgen_outbound_only_20260908/heuristic_comparison/`.

Route-length audit: `analysis/audit_colgen_route_options.py`, artifacts under
`analysis/colgen_outbound_only_20260908/route_options/`. Recomputed actual horizontal
flown distance using production terminal folding and customer endpoint connectors,
checking objective agreement for every column. Length tolerance 0.0001 m.

| Priced route type | Outbound no ground delay | Outbound delayed | Paired no ground delay | Paired delayed |
|---|---:|---:|---:|---:|
| Different geometry, equal seed length | 3283 | 77 | 6407 | 7491 |
| Longer than seed | 280 | 32 | 392 | 1319 |
| Original seed geometry | 0 | 1 | 0 | 811 |

Outbound: 3,673 priced columns, 97.00% zero ground delay; 3,656 new spatial families.
Paired: 16,420 priced columns, 58.59% delayed; 15,120 new spatial families.
No priced airborne holds in either trace. Initialization is excluded: outbound
starts pricing with 16,075 columns representing only 763 spatial families, from
one shortest seed plus 20 delayed copies at 4-second intervals through 80 seconds
and extra columns added by the feasible ground-delay bootstrap.

Equal length refers to the shortest discrete hex-grid seed, not a continuous
straight-line path. Different orders of directional grid steps can preserve total
length while occupying different space–time cells; capacity duals can favor these
alternatives. Actual plotted example: flight 1536, seed column 15729 vs column 19693
added in sweep 15. Both have zero ground delay and length 18,050.859864 m, while
the continuous en-route reference is 15,598.704767 m. This is an available column
example, not a claim it was selected by the final IP. Scientific PNGs were visually
inspected; Ruff and `git diff --check` passed. No production optimization changes.
