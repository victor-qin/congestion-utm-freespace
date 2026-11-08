# Bounded reuse of integer pricing operands

This implementation was prepared on PR146 at
`d7ad96861f866dededecd6d2c0ebee5ef02f0541`, including PR144. Both parent PRs have
since landed; the cache PR targets main. Measurements use the original stacked
source rather than the separate main-port experiment. Integer pruning remains
opt-in. The checked arithmetic, legal search domain, explicit 1e-6 closure rule
and original canonical completion validation are unchanged.

The original measured branch incorporated main at
`ebc60608e72de1afc5dcbd2e4b82094ab5464210` after all measurements finished.
Every production source was compared with the recorded measurement manifest;
the only difference is the parent's `_Batch.run` docstring clarification in
`gpu_pricing.py`. Python ASTs are identical after removing docstrings. The
evidence JSON therefore preserves the measured hashes rather than relabeling
them as the final source hashes.

The cache branch was subsequently rebased onto main at `c62e8bbe`, including
merged PR145's parallel CPU preparation and larger resident-search budget.
Those production changes are newer than the saved cache measurements below;
the original same-price timings must not be treated as measurements of this
combined stack. The root-reordering test now intercepts preparation in
`gpu_prepared`, where PR145 moved it.
The rebased integer/runtime/preparation-pool/pricing-platform/GPU-solver suite
passed **255 tests in 90.41 seconds**, including synchronous worker preparation
with integer bounds and canonical certificate parity.

## Retained data and exact dependencies

`ComputeRuntime.operands` owns a bounded cache of immutable numeric arrays and
device buffers. It never owns an `IntegerData`, batch, incumbent, predecessor
arena, cutoff, queue or discarded-upper certificate. Each pricing invocation
constructs those again. Root uppers are still evaluated by the real GPU on every
call. A cache hit skips construction/encoding and the corresponding retained
uploads; it does not substitute a saved pricing result.

| Retained block | Dependencies and reuse |
| --- | --- |
| Root inputs | Owned snapshot token; topology/configuration token; exact ordered departure, lane, cell and start descriptors; weighted ground and geometry operands; assignment dual, benefit, objective and scale; restrictions, clocks/hops, batch layout and options; selected endpoint rows, both root mask words and ordered complete mask entries; completed approach uppers |
| Continuation root constants | The same exact original-price/root context; independent of endpoint/future bounds. Uploaded only when continuation storage is needed, including on warm calls |
| Positive-price lookup | Exact snapshot/root context and local cell layout; sorted absolute resource times and original positive FP64 prices. New rows, zero-to-positive changes and sign changes require a new snapshot |
| Individual root cost encodings | Price-independent canonical root, geometry/configuration token, legal hop count, exact original FP64 coefficients and integer exponent. These can survive changed prices if those dependencies remain equal |

Root numbers alone are never a cache identity. A canonically validated incumbent
can filter departures before packing, and regrouping flights changes cell/root
offsets. Exact descriptors and ordered layout distinguish those cases. The
price-independent cost tier resolves each current packed root to its canonical
descriptor and original operands before lookup; its metadata and aggregate
upload are rebuilt for the current invocation.

The small ownership tokens are unique objects held by the existing graph packing
and snapshot summary. Cache keys retain the tokens, not full graphs or dual maps,
and cannot confuse recycled Python object IDs. Raw dual mappings continue to be
normalized anew by the public pricing adapter. Reuse requires an existing
immutable `DualView`, consistent with the existing shared-price-cache contract.

Endpoint device trees and future rows remain invocation-local. The endpoint
mapping in a cached root-input key includes selected rows, the exact two-word
root masks and ordered mask-entry identities, rather than just an entry count.
Root-input builds with an installed future table bypass persistent caching;
there is no alias between endpoint and future generations or between absent and
available future data. Future retries still clear layer buffers, retained-root
offsets and the old certificate accounting before restarting the complete domain.

## Ownership, publication and limits

The default limits are 128 MiB charged host retention, 128 MiB retained device
buffer requests and 8,192 entries per runtime. Host charging includes numeric
arrays, keys and container allowances; shared key/array storage is conservatively
charged again across entries. Empty device allocations count at least four
bytes. These limits supplement the existing 64 MiB live numeric and 128 MiB
resident-search budgets in the measured version (the current resident budget
is 256 MiB following PR145). They are not a total process-RSS or backend-allocator
memory guarantee, and per-flight peaks must not be summed to size the cache.

A build becomes reusable only after all its arrays and requested uploads finish
and the deadline is checked again. Exceptions leave no entry. Oversized complete
blocks can serve the current invocation without being admitted. Unrepresentable
numeric values retain the existing explicit invalid-interval semantics; a fully
built unavailable interval is not a partial build or a pruning certificate.

Eviction drops only cache ownership. Active invocations and command buffers hold
their own immutable handles until use completes. A different runtime cannot
borrow them. `clear_batch()` preserves cached operands, while `clear_cache()`
releases all runtime cache references. Existing live borrowers remain valid.
No extra workers, asynchronous GPU work or CPU/GPU overlap were introduced.

Hop operands are still allocated only for roots that survived closure and label
allocation. Continuation root constants also retain their original lazy upload
point, so all-closed domains cannot acquire a new storage requirement.

## Functional validation

| Requirement | Coverage |
| --- | --- |
| Warm parity and actual reuse | Extended exhaustive integer-search test across signed prices, audits and optional approach bounds; guards root/price builders, cost construction and repeated cached-array uploads; exact RC/column/upper equality and fresh label counts |
| Invalidation and restoration | Parameterized assignment, benefit, objective, exclusions, clocks, hop limit, exponent, endpoint selection, root reordering, actual incumbent-filtered root domains and batch regrouping; cached versus uncached actual-Metal results and independent canonical checks |
| Full price support | New origin rows, zero-to-positive and sign changes, complete negative credits, followed by restoration of the original owned snapshot |
| Failure and lifetime | Deadline at each retained block's publication boundary; overflow/deadline after upload in runtime tests; host/device/entry bounds, active borrower eviction, explicit clear and owner isolation |
| Existing proof/search contracts | Focused integer, bounds, endpoint, future, runtime, solver, options, validation and pricing-platform suites, including both paid mask words, role/hop legality, origin overlap, canonical scoring, unavailable grids and explicit incomplete caps |
| Lazy allocation | Existing retained-root budget regression and strengthened all-closed root tests; continuation allocation is forbidden on both cold and warm closure-only calls |

Production timing runs guard all original full CPU pricing entry points and the
per-prune FP64 verifier. Independent CPU canonical completion validation remains
enabled. CUDA validation is generated-source/interface and adapter coverage;
there is no NVIDIA hardware claim.

The final focused nine-file suite passed **423 tests in 78.94 seconds**. The
separate host/device byte-budget and active-borrower checks passed for both
runtime adapters. Ruff and `git diff --check` also passed.

## Measurement protocol

`execution_replay.py` uses only the original saved early/late captures and the
ordered cohorts in `arrival_union_results.json`. There is no reference-route
oracle seeding. All sustained tests and replays acquire the shared advisory lock
and release it between jobs. The original base was measured before production
edits. Two fresh-process final repeats compare enabled caching with
`--disable-integer-cache`, alternating order, on otherwise identical final code.

Each hard88 process runs cold pricing and two same-price warm passes. Each late18
process additionally runs changed prices, the same changed snapshot warm, and
the restored early snapshot. Batch size is one, the label cap is 250,000, chunk
size is 2,048 and each batch has a 180-second budget. Approach bounds are off in
these matched cache measurements; focused tests also exercise them enabled.

`integer_cache_report.py` checks capture hashes, exact ordered cohorts, limits,
final source hashes, complete certificates and **exact** returned columns,
reduced costs and certificate uppers against both the paired control and the
original stacked base. It preserves preparation/search/verification timers,
records/candidates, allocations/copies/waits, current cache storage, per-invocation
peaks and process peak RSS separately. Inclusive timers overlap.

Cache counters record hits, misses, skipped admissions, evictions, measured host
build/encoding time, previously measured build time avoided and encoded/upload
bytes reused. Avoided-time counters sum the original build durations; they are
not an independent measurement of elapsed time saved. Deliberately uncached
future-root builds appear as misses/skips.

## Final measurements and retention decision

[Checked evidence](integer_cache_results.json) contains two independent process
pairs per cohort. Every phase is complete and certified (88/88 or 18/18), with
exact column, reduced-cost and upper-certificate equality against the original
stacked base and the paired disabled-cache control. The following are medians
of the two process repeats, in seconds; a negative reduction is a regression.
Graph construction, snapshot indexing and runtime initialization are reported
separately in the JSON and excluded from these pricing-phase times.

| Cohort / phase | Cache disabled | Cache enabled | Wall reduction |
| --- | ---: | ---: | ---: |
| hard88 cold | 65.470 | 65.293 | 0.3% |
| hard88 same-price warm 1 | 26.030 | 22.594 | 13.2% |
| hard88 same-price warm 2 | 26.108 | 22.654 | 13.2% |
| late18 early cold | 16.841 | 16.804 | 0.2% |
| late18 early warm 1 | 4.729 | 3.972 | 16.0% |
| late18 early warm 2 | 4.727 | 3.974 | 15.9% |
| late18 changed snapshot, first call | 3.563 | 3.578 | -0.4% |
| late18 changed snapshot, warm | 2.614 | 1.334 | 49.0% |
| late18 restored early snapshot | 4.811 | 3.992 | 17.0% |

Retain this narrower cache for the repeatable late18 benefit. Hard88 improves
consistently but does **not** reach the prior 15% target. Compared with the
original pre-implementation baseline (rather than the same-code disabled
control), hard88 warm reductions are 12.5–12.7%, late18 early warm reductions
15.1–15.5%, late warm 48.9% and restored early 15.3%. Cold pricing has no material
gain. The first changed-price call regresses by 0.015 seconds (0.4%); its small
geometry reuse does not remove the new snapshot's required construction.

Representative warm-phase details below are medians of the two process repeats.
Timers are inclusive observations and must not be summed into a wall-time total.

| Observation | hard88 warm 2 disabled → cached | late18 late warm disabled → cached |
| --- | ---: | ---: |
| Base preparation | 0.792 → 0.792 s | 0.167 → 0.165 s |
| Integer preparation | 3.550 → 0.228 s | 1.298 → 0.029 s |
| GPU kernel execution | 1.601 → 1.573 s | 0.065 → 0.063 s |
| CPU canonical completions | 4.261 → 4.256 s | 0.025 → 0.025 s |
| Independent canonical validation | 0.281 → 0.280 s | 0.006 → 0.006 s |
| Labels, including preliminary searches | 511,172 → 511,172 | 19,668 → 19,668 |
| Candidate records | 2,467,999 → 2,467,988 | 84,152 → 84,152 |
| Canonical completion checks | 1,674 → 1,674 | 7 → 7 |
| Per-prune CPU checks | 0 → 0 | 0 → 0 |
| Native allocations | 25,478 → 25,057 | 1,550 → 1,501 |
| Allocated bytes | 1,172,722,544 → 1,150,802,932 | 64,164,992 → 60,347,868 |
| Uploaded bytes | 1,559,670,428 → 1,537,750,816 | 71,832,608 → 68,015,484 |
| Downloaded bytes | 352,869,304 → 352,869,304 | 8,392,056 → 8,392,056 |
| Wait count | 16,176 → 16,176 | 670 → 670 |
| Host wait time | 6.222 → 6.113 s | 0.268 → 0.254 s |

The small hard88 candidate-count variation occurs only on flight 492, in both
disabled and cached repeats and the original baseline. Across those runs its
candidate count ranges from 42,355 to 42,403, while retained labels, canonical
checks, returned columns and certificates remain equal. The report retains each
paired difference explicitly; candidate counts are not claimed to be identical.

Every cached warm hard88 pass has 2,319 hits, five deliberately uncached future
builds and zero evictions. It avoids 28,255,660 encoded bytes and 21,919,612 upload
bytes per pass; the prior-build-time counter reports about 3.215 seconds avoided.
Late18 early/restored passes have 452 hits and no misses; late warm has 33 hits and
one uncached future build, avoiding 4,814,260 encoded bytes and 3,817,124 upload
bytes. All final cache-enabled phases have zero evictions. Transfers and waits
remain substantial; this change primarily removes repeated host encoding.

Measured retained storage is 63.70 MiB charged host / 20.66 MiB device for hard88,
and 22.56 MiB host / 8.58 MiB device after both late18 snapshots. These are actual
cache gauges, not sums of flight peaks. Process peak RSS after hard88 warm 2 is
6.717–6.719 GB cached versus 6.773–6.781 GB disabled; late18 after restoration is
3.520–3.527 GB versus 3.547–3.558 GB. RSS is a process high-water mark, not live GPU
memory or a measurement of cache retention.

## Rejected retention policies

Retaining every cumulative cost and endpoint upload caused complete sequential
LRU thrashing on hard88: zero warm hits, 1,081 evictions per pass and 26.36–26.41
seconds warm versus 25.82–25.96 seconds on the original base. Retaining individual
cost encodings alone did not fix this while cumulative endpoint uploads remained:
endpoint entries consumed 87.8 MB of charged host storage, and the next policy
still evicted 2,426 entries per sweep with warm times of 26.03–26.10 seconds.
Neither timing enters the retained-policy comparison.

The retained implementation removes those overlapping cumulative generations.
Only individual retained-root cost encodings persist; aggregate cost/metadata
uploads and changing endpoint trees retain invocation lifetimes. This preserves
the byte bounds without enlarging them to hide the observed thrashing.

## Combined parallel-preparation diagnostic after rebase

One locked, lightly instrumented hard88 production solve on main `c62e8bbe`
plus this cache used 12 preparation workers, preparation groups of 12, GPU batch
size 1, 250,000 labels, integer and approach bounds, and one Gurobi thread.
Original capture and cohort selection were unchanged; all CPU canonical checks
remained enabled. Local artifacts are
`.agent/runs/rebased-stage-profile-1/{manifest,stages,result,planner_stats,rounds}.json`;
the wrapper is `.agent/profile_current_stages.py`. Nested owner timers charge
each interval only to the innermost measured category. This is a diagnostic
sample, not a repeated speedup experiment.

| Exclusive pricing wall category | Seconds |
| --- | ---: |
| Worker preparation barrier, including IPC | 9.306 |
| Owner integer operands and cutoffs | 8.580 |
| GPU calls: submission, internal buffer setup and waits | 4.789 |
| Completed-route construction, validation and scoring | 2.676 |
| Other owner base/price/approach preparation | 1.387 |
| Explicit buffer allocation/copy calls | 0.347 |
| Remaining batch CPU work | 3.942 |
| Pricing outside measured batch/pool methods | 0.251 |
| Total pricing wall | 31.280 |

Total solve wall was 35.346 s. All 88 flights were accepted and verified; all
352 flight-pricing calls completed/certified in four rounds, with cost
8708.244488722878 and the same 0.1343% global integer gap as earlier full-solver
runs. There were 754 canonical checks, zero rejections and zero CPU per-prune
checks. These results do not establish an exact global integer optimum.

The 10,776 GPU dispatches each waited; device execution was 0.872 s and owner
wait time was 4.066 s. These overlap the GPU-call category and must not be added
to the table. All 270,641 labels were from the initial beam search for an
incumbent (`bootstrap_labels`); `full_search_labels` was zero. Thus this run
never entered the resident full-search expansion loop: integer root bounds
closed the complete domain after the initial routes were found. The initial
search still reads expansion records back, processes flags/completions, selects
the next frontier and submits another GPU call. Removing per-prune CPU
verification did not eliminate these handoffs.

Cache telemetry recorded 73 hits, 897 misses and zero evictions; it avoided
324,160 encoding bytes and 546,000 upload bytes. Fresh LP prices/domain changes
limit reuse here. The two largest preparation categories account for 57% of
pricing wall; integer operand work remains on the owner after PR145's worker
barrier. These data support prioritizing owner numeric preparation and the
initial-search handoffs over removing the remaining route checks.

Earlier uninstrumented, pinned-source paired solves (PR145 `b605aba3` versus
that version plus cache `e2831fea`) averaged 34.482 versus 35.232 s total and
30.332 versus 31.168 s pricing: the cache regressed this changed-price workload
by 2.2% total/2.8% pricing. They are distinct from the same-price replay wins.
A separate single diagnostic disabling only completed-route feasibility gates,
while retaining construction/scoring, took 33.381 s total/29.154 s pricing;
that unchecked run cannot carry a valid pricing certificate. Its observed
5.3% total/6.5% pricing reduction is not a paired repeated estimate.

## Reproduction

Use the shared `.venv/bin/python` with `PYTHONPATH` set to this checkout. Wrap
each command in `fcntl.flock(..., LOCK_EX)` on
`/Users/victor/Documents/Research/congestion-utm-freespace/.agent/gpu-followups-20261106/metal-benchmark.lock`.
Pass saved cohort IDs in their original order.

```sh
python -m analysis.colgen.execution_replay \
  --capture /Users/victor/Documents/Research/congestion-utm-freespace/.agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl \
  --flight-ids SAVED_ORDERED_IDS --batch-size 1 --warm-repeats 2 \
  --integer-bounds --out .agent/cache-replays/hard88-cached-1.json
```

For the paired control add `--disable-integer-cache`. For late18 use its saved
cohort and add the original
`/Users/victor/Documents/Research/congestion-utm-freespace/.agent/runs/colgen-capture-20261105-211249/pricing_captures/round_0011.pkl`
as the second `--capture`. Repeat each mode in a new process. Local exact commands
are retained in `.agent/run_cache_jobs.py` and `.agent/run_final_cache_jobs.py`.

```sh
python -m analysis.colgen.integer_cache_report --replays .agent/cache-replays \
  --out analysis/colgen/integer_cache_results.json
```
