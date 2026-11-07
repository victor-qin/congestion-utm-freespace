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

The final focused suite passed **182 tests** in 29.91 seconds. Changed Python
files passed Ruff, and `git diff --check` passed. Original full CPU pricing was
blocked during the actual Metal replays and solver check. The canonical full
solver, including the licensed Gurobi master, accepted and verified all ten
flights in the longest-nine-plus-1752 capture subset. This is a full solver
integration check on ten flights, not a full 1,537-flight run. The pre-existing
full-suite import failure for `analysis.profile_native_dp` was not part of these
focused checks. No separate independent review has been performed.

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

The remaining capped early-price cases are **480, 806, 1646 and 2526**. Their
incumbents are respectively 8.6314554802, 5.1044821774, approximately zero, and
7.1785379972. Their upper bounds remain unavailable. On the 18 representative
late-price cases, both heuristic-only and final versions certify 16; **806 and
2526** remain capped. No claimed certificate depends on an exact route identity.

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
it closed **zero of 128 prefixes on each flight**. No production joint bound
is added on this evidence.

A follow-up would need to couple the arrival time and unpaid mask *inside* the
suffix recurrence; independently minimized terms lose that relationship.
First measure that coupled relaxation on these same surviving prefixes, with
an explicit disjoint price partition and bounded cache. Then independently
enumerate tiny canonical completions to verify every prefix upper before
adding device proposals. Preserve the current valid bounds by taking their
minimum with any new valid reduced-cost upper. Do not directly add endpoint
and visit bounds. Certified state merging remains research-only: current
heuristic keys do not prove complete future equivalence or dominance.
