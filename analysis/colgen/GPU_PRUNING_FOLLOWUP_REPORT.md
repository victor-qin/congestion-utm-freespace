# Integer reuse, parallel expansion, safe merging and the approach bound

This report separates three mechanisms: representing existing numerical bounds
on the GPU, reducing the search with a stronger bound, and merging equivalent
partial routes. Only the stronger bound is newly implemented in this follow-up.
Caching and certified merging are investigated designs, not shipped features.

## Can the integer preparation be cached?

Yes. An unchanged immutable encoding remains a valid enclosure. Rebuilding it
on every invocation is a current ownership/invalidation choice, not a
mathematical requirement. A cache should own immutable operand arrays and
device buffers; it must not retain the whole invocation, predecessor arena,
cutoffs or certificate bookkeeping.

The integer preparation starts with original FP64 objective operands and exact
resource membership. It chooses a power-of-two scale, rounds interval endpoints
outward, packs them into checked multiword integers, and uploads the inputs used
by the GPU. Integer overflow produces an unavailable bound, not a deletion.
FP32 remains useful for ranking candidates but cannot authorize certified
pruning. Completed routes still receive canonical FP64 scoring and validation.

```text
immutable_key = snapshot + graph + clocks + objective + restrictions
                + ordered roots + integer scale + bound-table generation
operands = cache.lookup(immutable_key)
if operands are absent:
    operands = outward_encode(original_FP64_operands, exact_memberships)
    publish_only_after_complete_build(operands)
fresh_search_state = new_labels + new_cutoffs + new_certificate_accounting
```

Same-price reuse can avoid repeated price/root encoding. Price-independent
FP64 geometry operands can also survive a price change; their integer encoding
can survive only when its scale and inputs still agree. New rows, zero-to-positive
or sign-changing prices, assignment duals, objective/clocks/restrictions and
root filtering/reordering all matter. A byte budget and explicit ownership are
necessary because process memory is already higher in integer mode.

Measured preparation is about 3.43 seconds on warm hard88 and 1.30 seconds on
late18. These are potential targets, not promised cache savings: some generation
work, lookup and memory costs would remain. See
[the cache design and required functional checks](GPU_INTEGER_CACHE_REVIEW.md).

## What changed in CPU/GPU work?

The integer version does **not** expand substantially more labels in the
matched runs. In the first warm hard88 pass of repeat 6, the reference version
stored 511,170 labels and the integer version 511,172. Their final full-search
label counts were identical at 3,105. Both certified all 88 flights and returned
the same reduced costs and columns.

The key saving was about 1.13 million per-prune CPU numerical checks, whose
retention stage took roughly 12.6 seconds. Warm wall-time medians fell from
38.7924 to 25.8866 seconds, a 33.27% reduction. This is historical fixed-price
pricing replay, not full-solver convergence.

CPU/GPU copying has **not** disappeared. The preliminary search still uploads
frontiers and reads back candidate records, now with 60 additional bytes of
integer state per record. Accounted upload-plus-download copies increased from
about 855 MB to 1,913 MB. On the M1 Ultra these are explicit shared-buffer copies,
not measured PCIe traffic. CPU numerical decisions disappeared from the guarded
integer production path; all host involvement did not.

The late18 regression is also not extra search: both versions store 19,668
labels and count 84,152 candidates. Reference checks take about 0.378 seconds,
while integer preparation takes about 1.30 seconds even on the next warm call.
Changed-price wall time is 2.6744 versus 3.5369 seconds; the following warm call
is 1.7266 versus 2.6118 seconds. All 18 certify with identical reduced costs.
Timers overlap, so these explain the direction without being an additive
decomposition of wall time.

## Does GPU parallelism help?

Yes: one 2,048-parent expansion chunk exposes up to 12,288 parent/successor
invocations. Many candidates expand together. But the next time layer depends
on the previous layer, and the small preliminary beam still triggers many
host checkpoints. The measured hard88 pass used 15,706 preliminary-search
dispatches versus 284 resident command blocks, with one flight per batch.

Measured device command time was about 1.59 seconds within 25.85 seconds wall
time. This is not an occupancy measurement, but it shows why faster expansion
alone cannot remove most latency. Input preparation, command submission,
candidate readback and canonical completion work remain. Parallel hardware
reduces the time per expansion; a good bound reduces how many expansions are
needed. Both are useful.

For flight 2224, integer rounding added only about \(3.813\times10^{-11}\) to
the sampled existing upper. The roughly 0.771219 unresolved margin was already
present in the original FP64 bound. Its combinatorial growth is a shared bound
weakness, not an integer-rounding penalty.

## Why not merge equal-cost histories?

The user's principle is correct **if equal prefixes have equivalent legal
continuations and equal or dominating completed objectives**. A deterministic
tie-break is then sufficient. Exact route identity is unnecessary.

Same cell, time and scalar cost alone omit information. For example, prefix A
has already paid resource X at price 5, while B has paid Y at price 5. A common
suffix claiming X adds no charge to A but adds 5 to B. Recent-cell revisit
restrictions can also make that suffix legal for one history and illegal for
the other. Root/hop/role constraints and canonical FP64 geometry must be
represented or bounded as well.

This is not merely a hypothetical obstacle. A guarded canonical example has
equal stub price 10 and equal geometry cost 4.250984267224913 at the same
destination/time; adding the mandatory arrival union produces complete prices
10 and 20. The different paid identities explain the difference.

Conversely, flight 2224 is promising for a **richer** merge rule. Its 250,000
stored prefixes contain only 967 existing structural keys. All active prefix
prices are zero. We sampled 32 histories from the largest identical-key group,
appended the same legal 102-hop suffix, and obtained identical validated final
costs and reduced costs. That supports the intuition, but does not prove all
suffixes of every history equivalent. A canonical geometry/feasibility proof
or a bounded-difference certificate is still needed. Repeated approximate ties
must carry cumulative loss; independently discarding 1e-6 at each merge can
exceed the final 1e-6 tolerance.

No certified merging is enabled. The detailed sufficient-state proposal and
actual-record examples are in [the merging audit](GPU_HISTORY_MERGING_REVIEW.md).

## Implemented: jointly reachable final approach and arrival

The previous endpoint bound could choose cheap arrival claims without paying
the preceding visit that every sufficiently early arrival requires. Flight
2224 exposes that gap: four cheaper final approaches cannot arrive in 119 hops.
Joining claims without checking this reachability still leaves the gap.

The new opt-in bound enumerates arrival times and possible final graph edges,
not route histories. For each alternative it prices the exact union of origin,
penultimate visit, final visit and mandatory arrival claims. A lower bound on
distance rejects final edges that cannot be reached by that arrival time.
Positive charges round downward; the full negative credit and existing
canonical FP64 geometry/roundoff allowances remain.

```text
new_upper = -infinity
for arrival_time and final_edge in a superset of all legal endings:
    if minimum_hops_to_final_edge > available_hops:
        continue
    rows = exact_union(origin, penultimate_visit, final_visit, arrival)
    candidate_upper = conservative_RC_upper(rows, hop_cost, signed_credits)
    new_upper = max(new_upper, candidate_upper)
if enumeration was incomplete or arithmetic unavailable:
    new_upper = infinity
root_upper = min(existing_independent_uppers, new_upper)
```

The CPU prepares at most one new finite upper per departure/lane root, after
the first canonically validated preliminary incumbent. The integer GPU kernel
consumes its outward encoding. A later retry has only the remaining preparation
budget. There is no per-label CPU verifier, new descendant state, worker pool,
CPU/GPU overlap, or full CPU pricing fallback. Caps/deadlines still cannot
certify; incomplete optional preparation merely leaves the old bound in use.

This is a **new, stronger mathematical bound**, complementary to the integer
representation of the original bounds. Independent uppers combine by minimum,
never by addition. Proof and failure behavior: [GPU_APPROACH_BOUND.md](GPU_APPROACH_BOUND.md).

## Measurement and validation

Final-source results are recorded in `approach_bound_results.json`. Direct
flight-2224 runs use the actual subset-solver round-two problem; cohort replays
use original full-scenario frozen prices. These are different objectives.
All sustained runs use the shared advisory lock, with release between jobs.
No reference-route oracle seed is supplied.

The first implementation ran only after a capped first pass and future-table
preparation. It reduced labels dramatically but barely changed integer wall
time: preparation had already been paid. Moving the same valid bound after the
initial incumbent search also avoids that future-table work. The earlier trial
is preserved in `.agent/approach-late-scheduling/`, and pre-budget-fix trials in
`.agent/approach-before-budget-fix/`; neither supplies final-source claims.

At the failing flight-2224 prices, the original FP64 upper is 5.305845430572;
the new approach upper is 4.534625231176, only about \(1.27\times10^{-8}\)
above the best validated reduced cost 4.534625218508. Integer mode reports a
conservative complete certificate upper of 4.534626218508: its retained dispatch
cutoff envelope accounts for all discarded alternatives without a 64-bit GPU
atomic maximum. The reported 1e-6 gap does not measure the looseness of the new
approach upper alone.

| Actual flight 2224, same saved round-two problem | Bound off | Bound on |
| --- | ---: | ---: |
| Integer cold pricing | 2.030520 s | 1.253105 s |
| Integer warm passes | 1.207625 / 1.184059 s | 0.437065 / 0.440924 s |
| Labels across all attempts | 272,210 | 7,426 |
| Final full-search labels | 250,000 | 0 |
| Status | Label limit, uncertified | Complete, certified |
| Best reduced cost | 4.534625218508154 | 4.534625218508154 |
| Reference-mode cold / warm | 8.665658 / 7.840941 s | 1.472186 / 0.667504 s |

The final 250,000-label pass is avoided; all 7,426 remaining labels belong to
the initial preliminary search. Future-table preparation is also avoided.
The integer helper costs about 0.055 seconds and consumes 40,999 work units,
below the million-unit budget. Numeric storage remains 1,799,686 bytes; the
42 MB final resident arena is not needed. The separate actual-Metal audit
checks 23,594 deletions with zero deficit and observes the new root closure.

These direct timings show a faster completed result replacing an incomplete
one. They are not a matched-completion speedup or full-solver convergence.

| Matched frozen-price hard88, integer mode | Bound off | Bound on |
| --- | ---: | ---: |
| Cold pricing | 65.425619 s | 65.102653 s |
| Same-price warm pricing | 25.834491 s | 25.631639 s |
| Certified / selected | 88 / 88 | 88 / 88 |
| Labels | 511,172 | 501,273 |
| Final full-search labels | 3,105 | 1,398 |
| New bound preparation | 0 | 0.257 s |
| Warm process peak RSS | 5.075 GB | 4.880 GB |

All returned columns and reduced costs match exactly. Five root closures save
some future-table work, but this single pair's less-than-1% wall difference is
not a material or repeatable cohort-wide speedup. Warm explicit copy volume
barely changes, about 1.914 GB to 1.910 GB. The independent hard88 Metal audit
certifies all 88 flights and checks 1,108,055 deletions, including all five new
root closures.

| Matched late18, integer mode | Bound off | Bound on |
| --- | ---: | ---: |
| Early prices, cold | 16.813602 s | 16.840972 s |
| Early prices, warm | 4.685870 s | 4.736027 s |
| Changed late prices | 3.574162 s | 3.594494 s |
| Late prices, warm | 2.618888 s | 2.657389 s |
| Restored early prices | 4.707893 s | 4.748791 s |

All five phases certify 18/18 with identical columns and reduced costs. No
root is closed by the new bound here: its 0.044–0.049 seconds of optional work
adds overhead without reducing labels. This is a separate small overhead from
the larger existing integer-preparation regression discussed above. The option
therefore remains off by default.

The canonical ten-flight solver completes in both modes: 8.811578 versus
8.854833 seconds, with identical cost 1895.591897429123, identical lower bound,
10/10 certified pricing calls and independent conflict verification.

| Fresh hard88 subset solve, integer mode | Bound off | Bound on |
| --- | ---: | ---: |
| Wall time | 46.055591 s | 67.171721 s |
| Solver rounds | 2 | 4 |
| Certified pricing calls | 102 / 103 | 352 / 352 |
| Stopping condition | Flight 2224 label limit | Configured LP gap |
| Feasible cost | 8716.244488722932 | 8708.244488722932 |
| Cost lower bound | 8604.350157717476 | 8696.551502329414 |
| Reported remaining integer gap | 1.28374475% | 0.13427490% |
| Accepted, independently verified flights | 88 | 88 |

The stronger bound resolves flight 2224 in rounds two, three and four; all
pricing calls in the completed run certify. There are 14 approach root-closure
events across the solve. Total stored labels fall from 400,185 to 270,641 even
though the new run performs many more pricing calls. Both runs guard native
CPU pricing and per-prune CPU verification. The result reaches the configured
stopping rule; the residual integer gap is not zero, so this is not a claim of
exact global integer optimality.

This is a **completion and solution-quality improvement**, not a speedup
between equivalent solves. The longer new wall time includes two additional
rounds which the capped run never performed. It is one fresh paired solver
trial, not a statistical speed estimate.

### Validation scope and reproduction

The follow-up covered 332 distinct focused cases across integer arithmetic,
options, endpoint bounds, GPU platform, canonical solver and validation tests.
After the final work-accounting fix, the 146 integer/options/endpoint cases were
rerun successfully; the other 186 cover unchanged integration/default behavior.
Coverage includes exhaustive canonical completions by root, signed/overlapping
claims, terminal roles, tiny/extreme values, snapshot changes, rejection,
deadline/budget/overflow behavior, actual Metal cap consumption and unavailable
cap handling. Changed files pass Ruff and `git diff --check`. CUDA source and
interface syntax are checked; no NVIDIA hardware was available.

Two independent Extra High source/proof reviews covered the new bound. Their
findings were fixed: require a test that actually consumes the new GPU cap, and
charge every arrival's incoming-edge scan against the preparation budget. These
reviews do not replace an independent review of the whole original integer
implementation or PR144.

The original full-scenario captures, exact saved hard88/late18 ordering and
limits are unchanged. `approach_bound_report.py` checks current source hashes,
capture hashes, ordered cohorts and limits, all paired reduced costs/columns,
and identical solver inputs except the optional flag. Compact results include
raw-file hashes, stage timers, allocation/copy/wait counters, candidates/labels,
memory counters, certificates, stopping conditions and objective bounds.
Inclusive timers can overlap.

```sh
PYTHONPATH="$PWD" /Users/victor/Documents/Research/congestion-utm-freespace/.venv/bin/python \
  -m analysis.colgen.approach_bound_report --jobs .agent \
  --out analysis/colgen/approach_bound_results.json
```

The local reproducible job list is `.agent/run_approach_jobs.py`; it invokes
`analysis.colgen.execution_replay` with/without `--approach-bounds`, and
`analysis.colgen.gpu_validation solve` with `--integer-bounds`, the same flag
pair and a 300-second budget. It uses `COLGEN_GUROBI_THREADS=1` and releases the
shared advisory lock between every job. The exact round-two reproduction is
`.agent/replay_approach_2224.py` with its original known-column seed; saved input
SHA256 is `b7ec19de5033f76a94a8971fe6c08c466ea29fd8fd413c23a7ab2edbe0b258fe`.
No reference-route oracle seed is used.

Raw final evidence: `.agent/approach-2224-*.json`,
`.agent/approach-{hard88,late18,controls4}-*.json`, and
`.agent/approach-solver-{ten,hard88}-{before,after}/`. All final production
hashes match commit `a4205eca`; later commits contain reports only. No measurement
job remains running. PR146 and PR144 remain draft and unmerged.
