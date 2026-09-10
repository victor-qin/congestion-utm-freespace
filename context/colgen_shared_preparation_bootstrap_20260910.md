# Shared pricing preparation and compiled bootstrap

Status: the first two changes pass 273 regression cases, matched pricing replays and
the full 1200-second comparison. The later completed-route screen passes 32 updated
bootstrap tests, six warm profiles, three complete pricing sweeps and a timing repeat.

## Changes

`PricingPreparation` lives for one `price_flight` call. Bootstrap ranking, restricted
DP refinement, and full DP share its unchanged packed duals, exclusion mask, endpoint
claims, start prices, scalar delay costs, and destination-price calculations. No data survives into the
next pricing round. Existing graph-local topology reuse remains in place.

For flight \(f\), departure step \(d\), and exit lane \(\ell\), the initial claim set is

\[
S_{fd\ell}=E_f^{\mathrm{origin}}(d)\cup
V\!\left(q_{f\ell},d+t_f^{\mathrm{takeoff}}+t_{f\ell}^{\mathrm{lane}}\right).
\]

The cached quantities are that set's price \(p_{fd\ell}=\sum_{r\in S_{fd\ell}}\lambda_r\)
and its active rows \(P_{fd\ell}=\{r\in S_{fd\ell}:\lambda_r\ne0\}\). The implementation
uses the original exact-sum operation, preserving floating-point results.

The incumbent changes between stages. Each stage therefore gets **fresh completion
envelopes and root pruning**, even though it reads the same prices. The envelope's
length depends on the incumbent at first use and itself controls pruning. Sharing
that length could change labels, dominance and tie results. Restricted root lists
and paid-class identifiers are also rebuilt for each stage. Individual scalar delay
costs are reused without reordering their arithmetic; each stage still applies its
own cutoff to determine the envelope's length.

`bootstrap_kernel.py` compiles the bounded goal-directed queue expansion. Parent
indices replace complete path tuples during expansion; a path is reconstructed when
a candidate needs the existing canonical sink check. Queue priorities, insertion
order, revisit rules and the expansion cap match the retained Python oracle. A
rejected candidate resumes the same queue. Explicit unsupported inputs fall back
to Python; unexpected errors and timeouts propagate.

The heuristic still only supplies a feasible incumbent. Restricted DP refines it,
and unrestricted DP still certifies pricing. Route domains, objectives, seeding,
IP settings and stopping tolerances are unchanged.

## Validation plan and provenance

- Focused tests compare packed arrays, stage-specific pruning and rewind behavior,
  changed prices/exclusions/objectives, bounded search results, rejected candidates,
  exact queue order, deadlines and explicit fallback.
- Frozen baseline: commit `6540337`, copied into
  `/private/tmp/colgen-prep-goal-baseline-20260910`.
- Frozen preparation-only source: `/private/tmp/colgen-prep-only-20260910`.
- Frozen combined source: `/private/tmp/colgen-prep-goal-combined-20260910`.
- Capture rounds 1, 6 and 11 of the 1200-second outbound FAA scenario, then replay
  identical inputs against baseline, preparation-only and combined sources.
- Compare exact output, route identity, claims, float bits and search counts.
- Run the combined full scenario with 12 pricing workers, 4 Gurobi threads, IP each
  round, 0.1% LP tolerance, six extra hops and one column per flight.
- Raw inputs, source hashes, timing and per-flight records:
  `analysis/colgen_prep_goal_20260910/`.

The 24 new native-search tests pass. The broader 249-case regression set passes
across its main run and targeted corrections. Two new tests needed to include the
immediate deadline check inside their exception expectation. One older test also
failed on the frozen baseline because it counted time variants as separate spatial
certificates; it now verifies both time-shift reuse and eviction across three
distinct spatial paths. No production cache change was needed. See
`analysis/colgen_prep_goal_20260910/test_validation.json` and its linked raw evidence.

## Matched pricing results

All runs use 12 workers. Each replay starts fresh workers and includes their graph
preparation; these are not the warm later-round times of a persistent full solve.

| Captured round | Baseline | Preparation only | Both changes |
|---|---:|---:|---:|
| 1 | 85.007 s | 85.580 s | 84.072 s |
| 6 | 90.678 s | 86.576 s | 86.094 s |
| 11 | 91.051 s | 86.688 s | 84.998 s |
| Sum | 266.737 s | 258.844 s | 255.164 s |

Preparation alone reduced the three-round sum by 2.96%; adding the native goal
search reduced it by another 1.42%, for 4.34% combined. Round 1 was effectively flat
for preparation alone. Do not describe it as an initialization speedup.

The reverse-order round-11 repeat gave 89.875 / 87.009 / 84.165 seconds for baseline /
preparation / combined. Averaging the two orders gives 90.463 / 86.848 / 84.581 seconds:
4.00% from preparation, another 2.61% from native goal search, and 6.50% combined.
These are two measurements of one captured round, not a statistical confidence interval.

Every compared flight preserves exact route identity, claims, cost/reduced-cost
float bits and search-work counts: nine comparisons of 1,537 flights, covering
4,611 distinct captured flight-round cases. Every expected native call ran and
none declined. See `analysis/colgen_prep_goal_20260910/replay_matrix/report.md`.

Stage timing needs care: preparation-only reduces the main phase but adds some
bootstrap work, particularly in round 1. Native goal timing also includes the
first packed-dual preparation previously performed inside restricted DP. Therefore
`bootstrap_goal_s` is an inclusive phase timer, not native kernel time.

## Full 1200-second comparison

The serial runs use the same 1,537 outbound flights, seed, configuration and algorithm
parameters: 12 pricing workers, four Gurobi threads, six extra hops, one column per
flight, 0.1% LP tolerance, up to 30 rounds, a 30-second IP each round and a 600-second
final IP allowance. Both stopped at the LP tolerance after 11 rounds.

| Measure | Baseline | Preparation + compiled queue | Reduction |
|---|---:|---:|---:|
| Pricing wall time | 683.769 s | 595.712 s | 12.88% |
| Total simulation wall time | 1324.508 s | 1227.099 s | 7.35% |
| Capture overhead within total | 4.149 s | 4.056 s | — |
| Total excluding capture overhead | 1320.360 s | 1223.043 s | 7.37% |
| Final IP wall time | 21.018 s | 19.367 s | — |

Both accept all 1,537 flights, deny none, pass feasibility verification, and finish
with exactly the same congestion cost: 156817.30255254533. Each final IP proves
optimality over its restricted column pool; the remaining configured-domain integer
gap is about 1.1826%, so this is not a proof of global integer optimality. Each pool
contains 42,879 columns. Neither run falls back from the native DP.

The full experiment measures the first two changes together. Their separate effects
come from the matched replays above. This is one serial pair on one seed; the elapsed
time reductions are measurements, not statistical confidence bounds. The additional
native completed-route screen below is excluded from these full-run results.

## Warm bootstrap profile

The six most expensive goal searches in the captured round-11 replay were measured
separately: one untimed warm call, three ordinary repeats, then one instrumented call
per source version. All outputs, claims, cost/reduced-cost bits, goal entry/output
and expansion counts match across the three versions.

Every selected search exhausted the 20,000-expansion budget and retained its input
incumbent. Across these six combined-version instrumented calls, host sink screening
used 11.550 of 12.910 seconds (89.47%); native queue expansion used 0.417 seconds
(3.23%). These percentages describe this expensive-tail sample, not all flights.
The ordinary summed goal medians were 12.829 / 12.797 / 12.808 seconds for baseline /
preparation / combined: effectively unchanged for these tails.

The sink callback constructs endpoint/path claims, calculates delay, sums duals and
screens against the incumbent. It invokes full canonicalization only for a promising
candidate, so the 23,229 callback invocations are **not** 23,229 canonical translations.
All selected roots had zero paid rows; optimizing membership checks for those rows
would not address this bottleneck. A future improvement would need to reduce the
cost or number of sink screens while preserving their acceptance and tie behavior.
The goal queue estimate is not a certified pruning bound.

Raw trials, instrumentation and exact parity are recorded in
`analysis/colgen_prep_goal_20260910/goal_phase_profile/report.md` and its linked JSON.

## Additional native screen for completed routes

Implemented after freezing the full-run candidate; the full comparison above excludes
this addition. The 32 updated bootstrap tests pass, including near-cutoff improvements,
negative duals, rejected candidates and strong-to-weak cached-envelope transitions.

The bootstrap can apply the exact DP's existing conservative sink bound before
reconstructing a path or invoking Python. With delay lower bound \(L\), unavoidable
positive destination-claim cost \(D^+\), and total available negative-dual credit
\(N^-\), the test uses

\[
U = M-\pi_f-L-D^+ + N^-.
\]

If this upper bound is below the incumbent by the existing recomputation margin,
the route cannot pass the exact improvement check. The initial implementation
deliberately contributes zero as the lower bound on positive origin/path prices;
that weakens the test while avoiding reliance on their accumulated rounding error.
The queue estimate itself is never used to prune.

This screens completed heap entries only. It preserves expansions, dominance,
insertion order and the 20,000-expansion budget. Potential improvements still use
the existing Python claim/geometry and canonical checks. Per-flight diagnostics
count `bootstrap_goal_sinks_skipped` and `bootstrap_goal_sinks_asked`; these count
finished labels, whereas one asked label can require several destination-lane checks.

Before trusting a shortened envelope, the implementation recomputes its first omitted
scalar delay bound against the current incumbent using the original cutoff rule.
If that no longer justifies truncation, the screen is disabled for the root. This
handles envelopes that previously saw a stronger incumbent without changing their
queue-order estimates or introducing a new cache.

On the same six expensive round-11 flights, Python sink callbacks fell from 23,229
to 15 (23,214 finished labels skipped). The sum of per-flight warm median goal times
fell 97.73%; total per-flight pricing time fell 90.34%. Every output, claim, cost/RC,
goal entry/result and expansion count matched. Each goal still expanded 20,000 labels.
These are selected-tail timings, not a full-population or full-solver estimate.

The separate nested profile attributes the old 11.355-second sink total to claim
construction (6.742 seconds), delay calculation (3.351), claim pricing (0.726), and
callback overhead. None of these goal callbacks reached `_canonical_candidate`.
The new screen therefore avoids preliminary Python work; it does not compile geometry.
See `analysis/colgen_prep_goal_20260910/sink_screen_profiles/report.md`.

The subsequent complete captured sweeps preserve exact output, identity, claims,
cost/RC and search-work counts across all 4,611 flight-round cases, with every expected
goal call native and zero declines. These use fresh workers, unlike the persistent
workers in the full simulation.

| Captured round | Compiled queue, no sink screen | With sink screen | Reduction | Skipped / asked finished labels |
|---|---:|---:|---:|---:|
| 1 | 84.072 s | 80.194 s | 4.61% | 13,268 / 1,471 |
| 6 | 86.094 s | 77.003 s | 10.56% | 121,616 / 1,119 |
| 11 | 84.998 s | 77.033 s | 9.37% | 136,781 / 1,077 |

These measurements compare against the frozen candidate containing the first two
changes. No full CG+IP run includes the sink screen yet.

The fresh round-11 pair measured 84.753 seconds without screening and 77.299 seconds
with it (8.79% lower), again with exact output/claims/float/work parity and identical
skip/ask counts. This supports the initial 9.37% observation on that captured input.
The complete sweep evidence is in
`analysis/colgen_prep_goal_20260910/sink_screen_replays/`.
