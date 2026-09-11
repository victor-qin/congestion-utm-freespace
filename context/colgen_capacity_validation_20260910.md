# Capacity separation and pricing validation

Work is on `codex/colgen-ip-profiling`. The controlled baseline is commit
`40c25e9ea63165eda6e173c4095d31e1e903dede`. Both seed comparisons, component
replays, and shorter-IP trials are complete. The two code optimizations reduced
full simulation time by **11.08% and 10.29%** with identical final costs. The
30-second per-round IP default is retained because the 25-second trial was not
consistently faster across the two seeds. Protocol and source manifests are in
`analysis/colgen_validation_20260910/`.

## Full simulation results

| Seed | Outbound flights | Baseline wall | Optimized wall | Reduction | Final cost, both runs | CG rounds / columns, both runs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,537 | 1,198.65 s | 1,065.78 s | 11.08% | 156,817.302553 | 11 / 42,879 |
| 1 | 1,545 | 1,499.89 s | 1,345.49 s | 10.29% | 161,135.996334 | 14 / 45,568 |

All four runs accepted every flight and passed independent conflict verification.
Both versions reached the 0.1% LP stopping target on each seed: final LP gaps
were 0.085201% and 0.071748%, respectively. Final IPs proved optimality over their
generated pools; the configured-domain global integer cost gaps were still
1.1826% and 0.8955%. Each setting has one timing trial per seed.

Seed 0 produced byte-identical final trajectory traces. On seed 1, the capped
IP incumbents first diverged in round 6 and final trajectories differ, while
final cost, ground delay, air detour, total delay, and p95 delay match exactly.
The audit found identical ordered column identities, column costs/reduced costs,
and spatial path definitions for both seed pairs. Frozen source hashes and all
input hashes match their recorded manifests. The complete comparison, including
per-flight distributions and worker loads, is in
`analysis/colgen_validation_20260910/full_comparison/`.

| Stage, seconds | Seed 0 baseline | Seed 0 optimized | Seed 1 baseline | Seed 1 optimized |
|---|---:|---:|---:|---:|
| Cumulative pricing | 560.80 | 472.03 | 746.82 | 644.58 |
| Capacity scans | 73.68 | 9.74 | 90.67 | 3.15 |
| Parent priced-column validation, within pricing | 91.89 | 0.00 | 106.47 | 0.00 |
| Per-round IP, including selected-column checks | 352.58 | 374.12 | 408.96 | 446.66 |
| Final IP | 20.85 | 20.33 | 18.64 | 19.53 |

Seed-0 pricing was **15.83%** faster. On seed 1, pricing fell from 746.82 to
644.58 seconds (**13.69%**), capacity scans from 90.67 to 3.15 seconds, and parent
pricing validation from 106.47 seconds to zero. All 12,998 accepted priced
columns on seed 1 used worker certification, with zero parent revalidations.

Some savings move between stages: the parent no longer warms its geometry cache
by checking every priced column, so checking selected IP columns costs more.
That relocation contributes to the 21.54-second increase in the per-round IP
stage on seed 0. On seed 1, selected-column checks rose from 16.49 to 45.84
seconds, and the full per-round IP stage rose from 408.96 to 446.66 seconds.
The end-to-end comparisons include these costs and native IP timing variation.

On seed 0, all 10,457 accepted priced columns used worker certification in the optimized
run, with zero parent revalidations. Receipt checks totaled 0.0046 seconds;
issuing the receipts and worker certification also have costs inside the full
pricing measurement. Aggregate worker task time was nearly unchanged
(4,661.31 to 4,683.41 worker-seconds), consistent with this change removing parent
work rather than accelerating the DP kernel itself.

The same pattern holds on seed 1: aggregate worker task time was 5,878.90 versus
5,931.35 worker-seconds. Mean task times were 0.27570 versus 0.27701 seconds on
seed 0 and 0.27179 versus 0.27422 seconds on seed 1. These are complete pricing
task times, not measurements of the native DP kernel alone.

## Component checks

Round-6 pricing was replayed against identical captured duals and incumbents,
using fresh workers. Every result across 3,082 flights matched exactly, including
claims, costs, reduced costs, and recorded search work. Fresh parent graphs were
used for independent canonical checks after timing; these changed no returned
claims or costs.

| Seed | Pricing sweep, baseline → optimized | Parent acceptance, baseline → optimized | Positive columns certified |
|---:|---:|---:|---:|
| 0 | 76.947 → 77.977 s | 15.531928 → 0.000752 s | 1,299 / 1,299 |
| 1 | 90.633 → 91.017 s | 4.344251 → 0.000518 s | 340 / 340 |

The capacity-only replay used the captured incumbent from rounds 1 and 11 of
each seed: one feasible column per flight, with every selection value equal to
one. All four pairs had identical loads and no violated rows. The median of
three repeated scans fell from 0.290–0.296 seconds to 0.000531–0.000854 seconds.
This isolates the no-new-row case on real schedules; it is not a full
multi-column master benchmark. The full-run capacity stage totals above are the
relevant end-to-end evidence.

An output-path error in the replay runner was corrected before the final matrix.
The completed baseline replay was relocated without changing its bytes; the
interrupted candidate was retained and excluded. Harness hashes and interruption
provenance are retained under `analysis/colgen_validation_20260910/ablations_v2/`.

## Shorter per-round IP budget

The controlled policy trial uses **25 seconds**, retaining 600 seconds
for the final IP. On seed 0 it reduced total wall time from 1,065.78 to
1,020.09 seconds (**4.29%**), with identical final cost, 11 rounds, 42,879 columns,
and all 1,537 flights verified. Per-round IP time fell from 374.12 to 320.96
seconds; pricing rose from 472.03 to 479.23 seconds. The final IP took 18.32
seconds and proved optimality over the generated pool. The trial temporarily
had worse incumbents in rounds 7–10, but recovered the control's final cost.
On seed 1, the 25-second policy required **15 rounds instead of 14** and was
**0.95% slower** overall. It generated 46,885 columns instead of 45,568 and found
a slightly better final cost: 161,062.133254 versus 161,135.996334, a reduction of
73.863081 (**0.04584%**). All flights were verified, the final LP gap was 0.068976%,
and the final IP proved optimality over the generated pool in 16.95 seconds.

| Seed | Optimized code, 30 s IP | Same code, 25 s IP | Wall change | Final cost change | Rounds, 30 → 25 s |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,065.78 s | 1,020.09 s | −4.29% | 0 | 11 → 11 |
| 1 | 1,345.49 s | 1,358.25 s | +0.95% | −0.04584% | 14 → 15 |

The recommendation is to retain **30 seconds per round** and the **600-second
final IP allowance**. A smaller cap can change incumbents, subsequent pricing,
and the number of CG rounds. The policy audit verifies that the budget was the
only algorithm-parameter change, so these are observed whole-policy outcomes,
not isolated solver-call speedups. With one trial per seed, the small wall-time
differences do not establish a universal ranking. Detailed results are in
`analysis/colgen_validation_20260910/policy25/`.

One slow pricing round should not be attributed entirely to weaker incumbents:
round 10's bootstrap and main label counts rose only 0.1818% and 0.00437%, while
their measured times rose roughly 58% and 50%. There are small real differences
in search work, but these counters do not establish the cause of the much larger
timing change. The full-run result includes that variation.
Indeed, 1,521 of 1,537 flights in that round had exactly matching recorded work
counters and still took 52.5% longer. Across all rounds, matching-counter flights
were only 1.05% slower on seed 0 and 0.86% faster on seed 1. These checks do not
separate machine variation from other unrecorded work.

The budget choice came from the optimized 30-second seed-0 run: none of its 11
per-round searches
improved the incumbent within 10 seconds; first improvements took about 12–22
seconds, and one round had no improvement. At 25 seconds, 9 of those 11 searches
had already found their 30-second incumbent. These observations motivate the
trial, but do not predict the full counterfactual run: changing an incumbent
can change subsequent pricing work and the generated pool.

Rounding and LNS remain options for a subsequent policy comparison. In the
current solver they run before pricing, whereas the per-round IP runs after
new columns have entered the pool and also materializes potentially binding
capacity rows. Switching policies can therefore change the next LP's rows and
duals, the incumbent used by pricing, and stopping behavior. LNS recombines
existing columns; it does not generate new spatial routes. A positive per-round
IP budget disables both rounding and LNS, so enabling LNS requires disabling
that IP path explicitly. Historical heuristic benchmarks use different settings
and do not establish their performance on the present two-seed configuration.

## Capacity separation

The master already records which columns claim each resource row, and which rows
can be overloaded by distinct flights. Separation now uses that information
instead of reconstructing every resource load from every positive column.
Materialized rows are already constraints in the LP/IP and cannot be newly added
by separation, so they need no further scan here. Final schedule feasibility
checks remain in place.

For a row \(r\), let \(F_r\) be the flights with a column claiming it,
\(b_r\) its fixed load, and \(C_r\) its capacity. Define the positive mass of
flight \(f\) by

\[
m_f^+ = \sum_{p\in P_f}\max(0,x_{fp}).
\]

When every \(m_f^+\leq 1\), a row satisfies

\[
b_r+\sum_{f,p}a_{rfp}\max(0,x_{fp})\leq b_r+|F_r|.
\]

Consequently, rows with \(|F_r|\leq C_r-b_r\) need no explicit evaluation.
The implementation checks a conservative numerical bound on the positive flight
masses and on floating-point accumulation against the separator's actual tolerance.
This matters because the public separator also accepts vectors that violate the
flight constraints, contain negative values, or use zero tolerance.

When that proof is unavailable, separation evaluates every implicit row through
the existing row-to-column index. It preserves the old positive-only accumulation
and column order. Changed fixed loads invalidate the shortcut. No pending-row
cache or cached fractional loads are introduced.

## Worker certification

A pricing result can bypass parent reconstruction only when its exact immutable
column object has been certified under the current graph and cost model, and the
sweep's result receipt matches the parent solve context. Imported or changed
results retain canonical claim and cost validation. A weak identity registry
records certified objects without retaining the columns. Certificates remain
available for live winners even when pricing checks many tied alternatives.

Before enabling reuse, shifted seed incumbents must be corrected as well. A time
translation is not always a literal shift of every terminal claim: endpoint
rounding can change at large clocks or with nonbinary time steps. The pricing
prepass now checks the actual endpoint windows before using a translated column
as a cutoff. When translation is not exact, it rebuilds canonical claims and cost
before comparing scores or applying its zero-dual stopping rule.

The fast path reuses the certified spatial detour but evaluates the cost at the
new departure using the canonical objective's operation order. The parent cannot
repair an incorrect earlier cutoff, so this check belongs in the worker before
search pruning.

## Validation protocol

Capacity tests compare newly materialized rows against full load accumulation for
feasible and arbitrary vectors, overflow, tight tolerances, pool growth, eager
materialization, and changed fixed loads. Pricing tests check translated claims,
costs, receipts, and sequential/worker behavior. Full simulations verify final
flight conflicts independently.

An initial full-run attempt revealed that an eight-entry cache evicted live
winners while checking tied candidates. That attempt was stopped and retained as
incomplete evidence. Replacing eviction with weak references addresses this
performance defect; all 161 focused solver, pool, and certification cases passed
on the revised implementation, including a new tied-winner regression. The
94 network and 24 native/reference checks had passed before this registry change.
Scoped Ruff and whitespace checks also passed. Logs and JUnit results are retained
with the benchmark artifacts; the earlier run's two invalid test-grid fixtures
were corrected before the final focused run.

The main comparison uses seeds 0 and 1, 1,200 seconds of outbound FAA demand,
12 pricing workers, 4 Gurobi threads, a 0.1% LP stopping tolerance, and a 30-second
IP allowance each round. Matched component replays separate the capacity and
parent-validation effects. The two 25-second policy trials then used the same
optimized code, with only the per-round IP allowance changed.
