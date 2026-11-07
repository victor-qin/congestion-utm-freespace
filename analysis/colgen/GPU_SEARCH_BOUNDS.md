# Endpoint-guided search and remaining-hop bounds

This follow-up preserves GPU route expansion and the full certified route domain.
Preliminary search may discard routes, but only canonically validated complete
routes improve its incumbent. The subsequent certified pass restarts every legal
root and retains distinct histories. No original CPU pricing search is called.

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

A missing bin, insufficient storage or an unsupported state returns no refinement.
The old future bound remains available. No partial weighted table is published;
a deadline aborts the pricing call without certification. A fully computed
infinite cost, in contrast, proves that particular relaxed continuation empty.

No GPU record or buffer layout changes are required. Before a frontier expands,
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

## Validation and remaining work

The tests compare original-price Decimal arithmetic on small complete relaxed
paths and canonical reduced costs on every prefix of independently enumerated
valid routes. They include partial/zero budgets, actual allowances exceeding six,
first-hop roles, exclusions, signed prices, endpoint/origin/visit overlap,
repricing, changed root sets and interrupted table builds. Existing GPU tests
cover both flag words, recent revisits, canonical rejection, near ties and limits.

Recorded capture evidence and final performance conclusions will be added after
matched replay and full-solver validation. Certified state merging remains
research-only: current heuristic keys do not prove future equivalence.
