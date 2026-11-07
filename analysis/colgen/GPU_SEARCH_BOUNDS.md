# Endpoint-guided search and remaining-hop bounds

This follow-up preserves GPU route expansion and the full certified route domain.
Preliminary search may discard routes, but only canonically validated complete
routes improve its incumbent. The subsequent certified pass restarts every legal
root and retains distinct histories. No original CPU pricing search is called.

## Mandatory arrival claim union

The arrival bound now includes the final cell's visit window when every legal
completion ends at that same cell. This closes a specific gap in the nominal
endpoint claim set: a final visit can require earlier occupancy rows than the
nominal endpoint window. For example, flight 480's arrival at step 461 requires
the final-cell row at step 459, priced at 0.476609739. Its earlier endpoint bound
omitted that row and stayed above the known route by precisely that amount.

Let \(E(t)\) be the guaranteed nominal destination claims at arrival \(t\),
\(D\) every distinct destination cell in the prepared topology, and \(V(c,t)\)
the visit claims for final cell \(c\). The new mandatory set is

\[
K(t)=E(t)\cup\bigcap_{c\in D}V(c,t).
\]

Every canonical completion arriving at \(t\) contains \(K(t)\). Distinct cells
have disjoint visit-row identities, so the intersection adds the final visit
only when \(D\) has one distinct cell. With multiple cells, the bound retains
the nominal endpoint claims; it never forces one destination lane. Empty
destination sets likewise add no visit information, rather than proving
unreachability through this optional bound.

For already claimed union \(P\), price the selected positive rows in
\(K(t)\setminus P\), then minimize over the same allowed arrival interval as
before. The prefix and residual sets are disjoint by construction. The set
union deduplicates endpoint/final-visit overlap before price accumulation, and
the existing paid flags remove any origin or earlier-visit overlap. Omitting
rows through the existing 64-row/cache budgets remains conservative. Negative
prices retain the existing global credit and original FP64 outward arithmetic.
This does not add the independent endpoint and future-visit bounds together.

The unpriced membership cache additionally keys the destination-cell set and
visit-window offsets. Weighted prices and paid-mask trees still refresh per
snapshot. GPU records, buffers, kernels and arguments are unchanged. Exhaustive
interval tests now require a final-visit-only row in the high flag word;
canonical complete-path tests cover every prefix, and the cache tests cover
alternate destinations, changed windows, repricing and already-paid overlap.

## Preliminary route discovery

The 64-state-per-flight preliminary beam now accounts for positive endpoint rows
not yet paid by its origin/visit union. It uses the existing two-word paid flags
and bounded arrival minima. Independent endpoint and future-visit estimates are
combined by maximum as heuristic remaining costs, never added. This ranking is
not a certificate. Half the slots can preserve distinct root/earliest-arrival
choices; remaining slots use the best scores. Flights with neither endpoint
prices nor future tables keep the previous inexpensive ranking.

The original flight-2 capture supplies no oracle seed. The changed preliminary
search finds a canonically validated RC 14 route, after which the existing bounds
close the complete domain. The saved CPU-reference route is not an input.

## Remaining-hop relaxation

Let \(H\) be the graph's actual air-hop ceiling, \(h\) the moves already taken,
\(d(c)\) the shortest any-role reverse distance to a destination, and \(f(c)\) the
shortest any-role distance from any current root cell. A rooted continuation has
remaining allowance \(b=H-h\), represented by \(k=b-d(c)\). Every allowed edge
\(c\to u\) satisfies \(d(c)\leq 1+d(u)\), so its successor allowance is

\[
k'=k+d(c)-1-d(u)\leq k.
\]

Consequently, keeping only a budgeted set of the lowest \(k\) values is closed
under continuation. A rooted state also satisfies
\(0\leq k\leq H-f(c)-d(c)\), allowing a separate count of stored states at each
cell. The forward-distance inequality \(f(u)\leq f(c)+1\) guarantees that a
successor with nonnegative \(k'\) fits this reachable-state ceiling. Counts are
derived from the actual topology and roots; they are not the configured overrun.

For every stored state/time pair, a backward recurrence minimizes future air
cost and positive visit charges, optionally stopping at a destination with the
existing disjoint arrival lower bound. It uses the existing regular
internal/last-role relaxation; roots use first/first-last roles. Allowing a stop
or an onward move more freely than the exact role/revisit rules can only lower
the minimum. Every canonical continuation is still represented. Time strictly
increases, and each move consumes one unit of the actual allowance. Current
forbidden visit rows and guaranteed arrival rows make the corresponding moves
or stops unavailable.

All additions use original FP64 prices with downward rounding. For any canonical
completion, the suffix table is at most its future air/visit/disjoint-arrival
cost. The verified visit-only prefix uses disjoint visit windows, enforced by
the main search's revisit rule. Origin-only charges are omitted from this bound.
The existing canonical geometry and arithmetic allowance convert the sum into a
reduced-cost upper bound. Negative dual credit is preserved. The resulting upper
bound is combined with the existing geometry and endpoint bounds by minimum.
Endpoint and visit bounds must never be directly added: their claims can overlap.

## Budgets, snapshots and device interface

`HopBounds` defaults to 64 MiB of extra numeric table/root-value/row-scratch
budget and 48 million conservative state/arc/time evaluations per flight and
snapshot. The existing unbounded table is separate. Weighted storage ends just after the
last positive or forbidden visit/arrival row. At that boundary, the recurrence
uses the existing unbounded lower bound for later continuations; it does not
assume they are unreachable or enforce the remaining allowance beyond the
boundary. This additional relaxation is safe and avoids storing a long unpriced
tail. Later times fall back to the old table. A binary search chooses the largest
complete set of low allowance bins fitting both budgets. Unpriced
transition arrays are cached on the flight graph, keyed by topology identity,
root-cell set and state limit; their bytes are reported separately. Prices,
root times, exclusions and weighted tables are refreshed per call.
The numeric budget is not a total process-memory limit: the existing future
table, cached transition arrays and Python container overhead are additional.

A missing bin, insufficient storage or an unsupported state returns no refinement.
The old future bound remains available. No partial weighted table is published;
a deadline aborts the pricing call without certification. A fully computed
infinite cost, in contrast, proves that particular relaxed continuation empty.

The remaining-hop change adds no GPU record or buffer layout changes. Before a frontier expands,
the host gathers the largest remaining allowance among all parents of each
candidate target. A lower bound for that larger allowance covers every child
reading the shared per-cell visit buffer. The GPU proposes deletions through
its existing future-cost field. CPU verification uses each child's exact
remaining allowance and independently accumulated prefix prices. A proved
infinite cost is represented by a finite FP32 maximum for proposal purposes
because infinity already means unavailable in that device interface. Only FP64
verification may accept the deletion.

Unchecked FP32 mode does not build the additional table and remains explicitly
uncertified. The Metal/CUDA shared source and argument layout are unchanged;
NVIDIA hardware execution has not been validated.

The imported endpoint prerequisites do change the layout relative to PR142.
These are the integration points for the separate GPU execution branch:

| Buffer | Layout |
| --- | --- |
| Integer label | Cell, root, hops, predecessor ID, first-hop source/target, status, paid low/high words, then recent cells. History starts at index 9. |
| Root endpoint data | Three int32 fields: paid low/high words viewed as signed integers, absolute root start step. |
| Visit masks | Two uint32 words per packed cell; refreshed for each absolute visit time. |
| Endpoint metadata | Five int32 fields per flight: first time, last time, tree leaf count, first entry, end entry. |
| Endpoint entries | Four 32-bit words: paid low/high, tree offset, validity marker. |
| Endpoint trees | Two FP32 range-minimum trees per mask; FP64 originals remain on the host. |

The shared `expand` interface appends visit masks, endpoint metadata, endpoint
entries and endpoint trees at argument indices 12–15. Both generated Metal and
CUDA wrappers use that same interface. The hop gather continues to populate the
third column of the existing signed/positive/future visit-price buffer; it needs
the full parent frontier's cell, root and hop fields. An execution refactor that
keeps frontiers on the device must preserve its maximum-allowance relaxation or
replace it with a separately proved per-child proposal.

## Validation and remaining work

The tests compare original-price Decimal arithmetic on small complete relaxed
paths and canonical reduced costs on every prefix of independently enumerated
valid routes. They include partial/zero budgets, actual allowances exceeding six,
first-hop roles, exclusions, signed prices, endpoint/origin/visit overlap,
repricing, changed root sets and interrupted table builds. Existing GPU tests
cover both flag words, recent revisits, canonical rejection, near ties and limits.

The final focused suite passed **183 tests** in 29.68 seconds. Changed Python
files passed Ruff, and `git diff --check` passed. Original full CPU pricing was
blocked during the actual Metal replays and solver check. The canonical full
solver, including the licensed Gurobi master, accepted and verified all ten
flights in the longest-nine-plus-1752 capture subset. This is a full solver
integration check on ten flights, not a full 1,537-flight run. The pre-existing
full-suite import failure for `analysis.profile_native_dp` was not part of these
focused checks. No separate independent review has been performed.

### Replay with the mandatory final-visit union

[The arrival-union report](arrival_union_results.json) compares the unchanged
`7c23d1e6` implementation with `d6ac84dc`. Capture prices, cap 250,000 per pass,
batch size one and chunk size 2,048 are unchanged. All measured production-source
hashes match the corresponding checkout or archived baseline. Full CPU pricing
is blocked, canonical validation remains enabled, and no saved reference route
is supplied as an input. Sustained Metal work uses the shared advisory lock.

The union certifies **88/88 early-price hard cases**, up from 84/88, and **18/18
late-price cases**, up from 16/18. All reduced costs agree within \(10^{-6}\),
and every new upper covers the corresponding previous feasible witness. Thus
the four formerly capped early-price incumbents were already optimal within the
certificate tolerance. Flight 1646 has no improving route.

| Metric, same 84 early-price certified flights | Before union | With union |
| --- | ---: | ---: |
| Pricing wall | 202.490 s | 130.566 s |
| Allocated records, all passes | 3,458,459 | 1,247,920 |
| Candidate extensions, all passes | 14,738,172 | 5,817,585 |
| Initial preparation | 56.164 s | 55.823 s |
| Optional future preparation | 7.606 s | 1.516 s |
| Preliminary search | 34.735 s | 34.318 s |
| Certified search passes | 103.786 s | 38.714 s |

Matched-certificate wall is 35.5% lower, records 63.9% lower and candidate
extensions 60.5% lower. These are single serialized replay aggregates, not a
statistical speedup estimate. All-88 wall is 135.342 s with the union versus
257.703 s before, but those totals compare different certificate counts.
For the same 16 already-certified late-price flights, wall is 23.731 s versus
41.136 s, with 166,719 versus 678,777 records. All 18 now finish in 25.306 s.
The remaining-hop code is unchanged; no separate benefit is attributed to it.

A fresh baseline archive replay and a first/repeated union replay confirm the
four closures and the improvement on flight 278. First use means the first call
for that graph in the process, with the runtime shared across flights; repeat
reuses the graph and price view. Capture loading and graph construction are
outside pricing wall. Each pair below is **first use / repeat**, in seconds.

| Flight | Before wall | Union wall | Before initial prep | Union initial prep | Records before / union |
| --- | ---: | ---: | ---: | ---: | ---: |
| 480 | 12.637 / 11.848 | 1.813 / 1.112 | 0.840 / 0.193 | 0.842 / 0.320 | 375,223 / 7,937 |
| 806 | 16.065 / 15.752 | 1.285 / 0.897 | 0.568 / 0.230 | 0.570 / 0.186 | 486,105 / 7,419 |
| 1646 | 11.817 / 11.543 | 0.679 / 0.385 | 0.613 / 0.345 | 0.544 / 0.248 | 339,030 / 0 |
| 2526 | 14.822 / 14.574 | 1.094 / 0.737 | 0.518 / 0.187 | 0.590 / 0.226 | 348,973 / 3,392 |
| 278 | 7.235 / 6.720 | 1.483 / 0.969 | 0.707 / 0.184 | 0.692 / 0.182 | 201,446 / 8,059 |

The first four baseline cases stop at the cap, so their wall reductions are not
matched-certificate speedups. Every union call certifies; flight 278 certifies
before and after. Flight 480's union upper is 8.631455483172406 against feasible
RC 8.631455480197474. Its full pass allocates zero continuation records after
the preliminary route is found. Flight 1646 closes all roots without allocating
route records. The other seven selected controls retain their prior work counts.

For repeat flight 480, device allocations fall from 388 to 181, allocated device
bytes from 17.58 to 7.92 MB, host-to-device copy volume from 306.21 to 96.18 MB,
and device-to-host copy volume from 97.23 to 3.07 MB. Allocation time is
7.50 / 2.92 ms, command-wait time 348.86 / 128.97 ms and GPU execution
26.97 / 10.20 ms, before / after. These timers overlap and must not be added.
Transfers are copy volumes, not separately timed transfers. Per-flight telemetry,
preparation and search stages are recorded independently in the report.

#### Flight 1638 regression

Flight 1638 retains the same certified RC 15.42779899989182 and upper
15.42779901153304, but total records increase from 171,147 to 208,396 and
candidate extensions from 811,269 to 845,097. Wall is 7.662 / 7.870 s in the
hard-case replay. Observational before/after Metal traces reproduce those work
counts without changing search inputs or supplying an incumbent.

The union's frontier is no larger at **every shared time layer** of the initial
full pass. At step 374 it has 48,183 live states and 125,406 allocated records,
versus 65,524 live states and 162,036 allocated records before. The old pass
stops while generating the next layer: 162,036 allocated plus 90,470 pending
exceeds the 250,000 cap. The new pass fits that complete layer, reaching step
375 with 73,955 live states and 199,361 allocated records; 51,613 pending records
then exceed the same cap. Pending records are not counted as allocated because
the entire next layer must fit. This explains the 21.8% record increase alongside
the smaller 4.2% increase in candidate extensions.

Both initial preliminary passes find RC approximately 8. Both initial full
passes leave it unchanged. The retry's future-informed preliminary pass finds
the same RC 15.427799 optimum, and its final full pass allocates zero records.
The regression is extra work before that useful retry, rather than a weaker
incumbent or larger frontier at the same search time. A general policy for
starting future preparation sooner could address it, but is not changed in this
bound-only update. Flight 2054 has a separate four-record increase (9,438 to
9,442); both calls still certify. These regressions remain in the aggregate.

### Replay before the mandatory final-visit union

[The machine-readable report](endpoint_search_results.json) records capture and
production-source hashes, per-flight costs/uppers, stage timings, work counts,
first-use/repeated-call telemetry and the remaining-bound diagnostics.
The prerequisite comparison uses `16f860c8`, heuristic-only uses `4486448c`, and
the final implementation uses `ed286dba`. All use the original capture prices,
Metal on Apple M1 Ultra, batch size one, chunk size 2,048 and cap 250,000 per
search pass. The shared advisory lock serialized sustained Metal work.

| Version | Certified hard cases | Pricing wall, all 88 | Wall, same 69 certified | Records, same 69 | Candidates, same 69 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Endpoint prerequisites | 69/88 | 462.008 s | 219.759 s | 4,883,180 | 19,340,081 |
| Endpoint-aware preliminary search | 84/88 | 258.132 s | 169.239 s | 2,949,922 | 12,286,950 |
| Plus remaining-hop bounds | 84/88 | 257.703 s | 168.783 s | 2,949,922 | 12,286,950 |

There were no lost certificates or reduced-cost/upper-bound disagreements on
the 69 matched certified flights. Their heuristic-only pricing wall fell by
about 23%, allocated records by 40%, and candidates by 36%. This is a single
serialized replay per version, not a statistical speedup estimate. The all-88
wall totals compare different certificate counts and must not be described as
matched-certificate speedup. Counts include all preliminary, full and retry
passes, so total records may exceed the per-pass cap.

Flight 2 now discovers RC 14.000000007265044 with 3,908 allocated records and
certifies an upper of 14.000000008441557. The prerequisite run found RC 6 and
capped after 412,235 total records. Its reference root was already selected;
neither more roots nor a saved reference route was needed. Widening the beam
from 64 to 256 increased flight-2 work to 15,106 records and failed to resolve
480 or 806, so the default remains 64 with 32 selected preliminary roots.

Before the union, capped early-price cases were **480, 806, 1646 and 2526**. Their
incumbents were respectively 8.6314554802, 5.1044821774, approximately zero, and
7.1785379972, with no upper certificate. On the 18 representative late-price
cases, both heuristic-only and pre-union versions certified 16; **806 and
2526** were capped. The union results above supersede these cap outcomes.
No claimed certificate depends on an exact route identity.

The hop refinement changed no candidate/record counts or certificate outcomes
on either the 88 early-price or 18 late-price cases. It added 5.009 seconds of
table construction across the early-price replay. Its tiny wall-time difference
is measurement variation, not evidence of speedup. For the four early-price
caps, the extra weighted tables occupy 47.85/39.88/17.15/12.53 MB with 5/9/10/10
stored allowance bins; cached structure occupies another 2.64/1.89/1.36/1.63 MB.
These costs are reported separately from the original future table.

### First-use and repeated preparation

All eight selected hard/control flights certify on both calls and match the
saved reference reduced costs within \(10^{-6}\); every reported upper covers
its reference. Saved reference routes were used only for result comparison.
The first-use column means the first call for that graph in the replay process;
the runtime is shared across flights. The repeat reuses the graph and price
view. Pricing wall excludes capture loading and flight-graph construction.

| Flight | First-use wall | Repeat wall | First-use initial prep | Repeat initial prep | Records on either call |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.187 s | 0.631 s | 0.501 s | 0.182 s | 3,908 |
| 32 | 1.207 s | 0.762 s | 0.582 s | 0.186 s | 5,238 |
| 278 | 7.314 s | 6.681 s | 0.719 s | 0.176 s | 201,446 |
| 322 | 1.329 s | 0.659 s | 0.817 s | 0.177 s | 6,695 |
| 0 | 1.198 s | 0.511 s | 0.868 s | 0.185 s | 7,770 |
| 34 | 1.124 s | 0.483 s | 0.824 s | 0.184 s | 7,010 |
| 276 | 1.138 s | 0.466 s | 0.852 s | 0.184 s | 6,495 |
| 320 | 0.673 s | 0.330 s | 0.517 s | 0.175 s | 1,903 |

The report separates initial preparation, optional future preparation,
preliminary search, certified passes and device telemetry. For example, repeat
flight 2 allocates 127 device buffers in 1.47 ms, uploads 43.27 MB and downloads
1.58 MB, and spends 52.83 ms in 128 command waits with 4.78 ms of GPU execution.
Repeat flight 278 allocates 275 buffers in 4.46 ms, uploads 147.94 MB and downloads
55.25 MB, and spends 148.01 ms in 353 waits with 13.28 ms of GPU execution.
Transfers are measured as copy volume, not standalone transfer time. Runtime,
allocation, waits and device timers overlap; they must not be added. Dispatch
reduction alone is not state reduction or end-to-end speedup.

### Joint-bound follow-up

Three bounded diagnostics partitioned positive endpoint prices out of the
visit suffix and sampled 128 final surviving prefixes per capped early-price
flight. The strongest tried a price-only suffix with actual hop allowance,
reserved every origin/root-visit row against duplicate charging, and combined
it with the full positive prefix, unpaid endpoint arrival minimum and clamped
geometry lower bound. Even with a 128 MiB / 150 million-work diagnostic budget,
it closed **zero of 128 prefixes on each flight**. No general production joint
recurrence is added on this evidence. The simpler mandatory arrival union above
now closes these four cases without a joint suffix recurrence.

A further joint-bound experiment could couple arrival time and unpaid mask *inside* the
suffix recurrence; independently minimized terms lose that relationship.
First identify cases still needing it and measure that coupled relaxation, with
an explicit disjoint price partition and bounded cache. Then independently
enumerate tiny canonical completions to verify every prefix upper before
adding device proposals. Preserve the current valid bounds by taking their
minimum with any new valid reduced-cost upper. Do not directly add endpoint
and visit bounds. Certified state merging remains research-only: current
heuristic keys do not prove complete future equivalence or dominance.

## Departure screening and earlier future retry

Before constructing a departure's endpoint claims, the GPU host can use ground
cost alone to bound every route starting there. Both multiplications round
downward; the existing reduced-cost helper rounds upward and includes all
negative prices. A departure is omitted only when its finite upper is no greater
than an already canonically validated incumbent. There is no closure tolerance
in this comparison. Consequently the final incumbent also covers every omitted
departure, including after a retry clears the separate closed-branch upper.
Missing witnesses, nonfinite bounds, negative prices, and zero objective weights
retain the conservative behavior. The callback checks the deadline even for
screened departures. CPU callers of `prepare_variants` retain their existing gate.

The first complete-domain GPU pass now stops after at most 8,192 allocated
records, or the caller's smaller cap. Hitting that limit triggers the existing
future preparation, future-informed preliminary search, and complete-domain
retry at the original cap. It never closes an unresolved branch. The same
canonical candidate checks, signed prices, root/hop domains, and deadline remain
in force. Exhaustion of the final pass still returns an incomplete result.
This specifically avoids spending most of the cap before discovering the useful
incumbent in cases such as flight 1638. It changes scheduling, not bound validity.

Neither change modifies the GPU record/buffer layout, kernels, or public pricing
options. Integration needs the per-batch `search_label_limit` in both allocation
and pending-layer checks; the allocated arena remains sized to the caller's cap.
The departure callback is optional and runs before origin-claim construction.

## CPU occupied-slot maintenance

Each of the two native CPU state tables records the slot index on its first
insertion in a layer. Replacing a dominated label does not append another slot.
Gathering reads the latest occupant of each recorded slot and retains the
existing full sort before expansion. Distinct live dominance keys cannot have
identical full path/departure/lane sort keys, so changing the unsorted gather
order cannot change the subsequent relaxation order. Every consumed layer clears
only its recorded slots and resets its count. Both lists and counts survive
sink, envelope, and output-buffer pauses; retries allocate fresh state.

The internal native call gains two int32 slot arrays of table capacity and a
two-element int64 count array. Additional storage is \(8C+16\) bytes for capacity
\(C\); no cross-flight arena or device residency policy is introduced. Public
pricing interfaces, dominance keys, numerical operations, validation, and search
work remain unchanged. Functional coverage includes forced table/label growth,
one-entry output buffers, signed prices, exclusions, and exact reference parity.

## Measured optimization trials (2026-11-07)

The experiment baseline is `e1257b51`, which already includes the mandatory
arrival union. The retention criterion was a repeatable reduction of at least
15% in matched-certificate pricing wall time, with no lost certificates. These
are serialized original-input replays on the M1 Metal device and native CPU
backend, with unchanged canonical validation. Metal uses cap 250,000, batch 1,
and chunk 2,048. CPU JIT warmup is separate; graph construction and capture
loading are outside the measured pricing calls. First use includes cold pricing
preparation; the warm pass reuses the same process and preparation caches.

| Isolated experiment | First-use hard-88 seconds | Warm hard-88 seconds | Decision |
| --- | ---: | ---: | --- |
| CPU baseline | 53.716 | 24.442 | Reference |
| CPU occupied slots | 49.293 | 20.279 | Keep; repeated warm gain exceeds 18% |
| CPU obstacle-aware reverse distance | 53.046 | 24.064 | Discard; 1.5% warm gain |
| CPU bounded buffer reuse | 53.096 | 24.547 | Discard |
| CPU completion-range minima | 53.570 | 24.564 | Discard |
| CPU paid-mask bound and range minima | 61.040 | 25.084 | Discard; slower |
| Metal baseline | 135.943 | 95.883 | Reference |
| Metal early departure screening | 103.389 | 64.988 | Keep; 32.2% warm gain |
| Metal earlier future preparation | 110.390 | 70.557 | Keep; 26.4% warm gain |
| Metal destination-choice unions | 135.313 | 97.177 | Discard; every case has one final cell |
| Metal combined prototype | 77.640 | 38.657 | Attribution check |
| Metal final source, with telemetry | 77.162 | 38.808 | 43.2% first / 59.5% warm gain |

Every row represents 88/88 certificates on both passes, with identical best
reduced costs. CPU occupied-slot warm timings repeat in both orders:
24.709 to 20.219 seconds, then 24.600 to 20.137 seconds (18.17% and 18.14%).
Exact routes and all reported search counts are unchanged. Late-price CPU gains
are smaller, 11.74% and 12.10%. The existing CPU `n_labels` counter is the last
native search and can retain bootstrap counts when the main search is empty;
it must not be interpreted as total all-pass allocations.

Final-source Metal checks retain 18/18 late-price certificates in both passes:
25.329 to 14.993 seconds first use, 13.136 to 3.034 seconds warm. The selected
17 hard/control flights retain 17/17 in both passes: 50.149 to 17.494 seconds
first use, 42.713 to 10.084 seconds warm. All returned columns are canonically
validated and every upper covers the reference witness. No saved route is used
as an oracle input; original full CPU pricing is blocked throughout Metal runs.

Earlier future preparation reduces all-pass hard-88 allocated route records
from 1,266,668 to 503,064 and candidates from 5,910,883 to 2,471,158. Screening
alone leaves those counts unchanged and omits 75,912 departures before claim
construction. Flight 1638 falls from 208,396 to 13,649 records and from 7.465 to
1.147 warm seconds. Its optimum and upper remain valid; this fixes the earlier
cap-timing regression without flight-specific policy.

Final-source hard-88 preparation takes 39.935 seconds first use and 1.625 warm.
Warm preliminary search takes 33.307 seconds, future preparation 1.417, and
complete-domain passes 2.417. These stages include numerical verification,
canonical validation, dispatch, and waits; they do not partition CPU versus GPU
execution time. In each pass the device telemetry records 13,688 allocations
(318,980,620 bytes), 1,149,530,432 uploaded bytes, 196,217,072 downloaded bytes,
and 16,159 waits. Warm allocation/wait/device/runtime times are respectively
0.196/6.388/0.610/7.705 seconds; these timers overlap. Copy volume is measured,
not independent transfer time. The final hard-88 baseline has no telemetry;
the independently measured combined prototype confirms the gain without it.
Selected-17 and late-18 comparisons have telemetry on both sides.

A final discarded experiment transports complete certificates between saved
price rounds 6 and 11. It uses the conservative upper
\(U_{old}+\pi_{old}-\pi_{new}+\sum_r\max(0,p_{old,r}-p_{new,r})\), with
outward rounding and every potentially claimable row in the prepared spatial
and padded time domain. Both passes close 0/88 cases. Warm querying costs
10.736 seconds; first-use domain preparation adds 27.437 seconds. These saved
rounds are not consecutive. This result does not rule out reuse across smaller
price changes, but provides no reason to add it to production here.

The retained implementation has 211 unique focused test passes and separate
10-flight canonical CPU/Gurobi and Metal/Gurobi solver checks. One new test's
coverage assertion initially failed because its ground weight did not create
screenable departures; increasing that fixture weight exercises the gate and
passes. An existing large pure-Python overrun-9 reference comparison was stopped
after the test run reached 324 seconds and is not counted as a pass. The full
suite was not run (the base has an unrelated missing `analysis.profile_native_dp`
fixture). CUDA source/interfaces are preserved, without NVIDIA hardware testing.

`pricing_followup_trials.json` contains aligned per-flight final-source results,
aggregate isolated trials, CPU startup times, source/capture digests, validation
evidence and separate preparation/device counters. The CPU optimization is an
independent commit and does not require the Metal changes. These results do not
establish a full 1,537-flight solver speedup or that Metal outperforms the CPU.
