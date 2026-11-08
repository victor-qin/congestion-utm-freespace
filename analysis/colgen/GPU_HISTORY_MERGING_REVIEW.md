# Can certified GPU search merge equal-cost histories?

2026-11-07. Independent source audit of `a794c68e`, plus the saved flight-2224
round-two records. This report does not enable grouping or change production
decisions. Tests/diagnostics below use canonical translation only, never a
native full CPU pricing search.

## The user's principle is correct

If two histories have the same possible continuations and each common
continuation has the same final objective, either history can represent both.
A deterministic tie-break (lexicographic path, stable ID, etc.) is sufficient;
identical route identity and identical CPU pruning order are not requirements.
For already completed, canonically validated columns with equal reduced cost,
choosing any one is fine for the fixed-price maximum even if their resource
claims differ. The question for a partial history is what happens after it is
extended; an equal current scalar does not answer that.

More generally, history A dominates B when every legal complete continuation
of B can also complete A, with A's completed reduced cost at least B's. A
bounded difference can also be used, but must be carried into the objective-gap
certificate. Merely observing equal current scores does not prove either
condition.

"Not proved in the current certified GPU implementation" is the accurate
description. Merging is not impossible. The native CPU code already uses a
richer state and deterministic ties (`pricing.py:2583`, `_prefer` at line 349).
Its state keeps the cell, recent cells, origin-paid row identities, first hop
when needed, and consumed hops, inside one absolute-time layer. That existing
implementation is useful evidence/design input, not an independent proof that
its partial score safely represents the canonical FP64 objective everywhere.

## What downstream behavior can differ?

| Information omitted by cell/time/equal scalar score | Why it can change a completion | Relevant source |
| --- | --- | --- |
| Recent visited cells | Returning to a recent cell is forbidden for one prefix and allowed for another. | `pricing_gpu.h:118`; `pricing.py:2522` |
| Paid resource identities | A future endpoint/visit charges the union of rows, so a previously paid row is charged once. Equal sums can represent different paid rows. | `network.py:2114`; `pricing.py:214` |
| Root/departure/lane and hop budget | Same absolute time can leave different remaining hops and different ground/origin cost. | `pricing.py:2097`; `dp_prepare.py:1791` |
| First/last role and endpoint geometry | Terminal tags and folding can depend on the path's boundary arcs. | `network.py:1322`; `volumes.py:349`; `translate.py:214` |
| Original canonical cost, rather than FP32 ranking score | The final cost is rebuilt from resampled FP64 geometry, a norm reduction, clamping and weighted arithmetic. Equal FP32 or ideal hop costs need not be equal canonical costs. | `volumes.py:495`; `volumes.py:340`; `translate.py:430` |

The paid-row issue is an exact set issue, not a rounding issue. If A has paid
row X at price 5 and B row Y at price 5, both prefix costs are 5. If the same
continuation claims X, A's union costs 5 and B's costs 10. Signed prices make
the same identity requirement apply to credits. A complete paid-set equality
is sufficient, but often stronger than necessary: only identities that can
overlap a future charge must remain in the state; expired disjoint charges
can be summarized by a proved cost accumulator.

The repository already contains a canonical endpoint-overlap regression in
`tests/test_colgen_solver.py:427`: when the visit window narrows, retaining
only the current cell loses predecessor information needed to deduplicate
the destination union. `dp_prepare.py:132` therefore retains at least two
cells even when the revisit ban itself needs less history.

Rebuilt that fixture with both predecessor prices set to 10 and specified the
routes explicitly, without running a pricing search:

| Route | Time at (2,1) | Visit/origin price before arrival dwell | Canonical geometry cost | Complete union price |
| --- | ---: | ---: | ---: | ---: |
| (0,0) → (1,0) → (2,0) → (2,1) | 8 | 10 | 4.250984267224913 | 10 |
| (0,0) → (0,1) → (1,1) → (2,1) | 8 | 10 | 4.250984267224913 | 20 |

Both routes pass canonical validation. The arrival claims include row
`('cell',2,0,0,7)`: route A already paid it, while route B paid a different
10-price row. Completing the destination's dwell therefore changes the
ranking despite equal cell, time, hop count, geometry cost and stub price.
This is a terminal-completion example: once the complete union has already
been priced, the resulting candidates are not tied. Raw results and guarded
reproduction: `.agent/merge-audit/equal-stub-example.json` and
`.agent/merge-audit/equal_stub_example.py`.

## What is missing from today's GPU grouping?

The existing GPU grouping key is already richer than cell/time: root, cell,
hops, first arc, both selected-endpoint paid words, and recent cells
(`pricing_gpu.h:180`). It chooses the larger **FP32 ranking score**, then a
stable record ID (`pricing_gpu.h:206`). The certified path deliberately
bypasses it (`gpu_pricing.py:945`); it is used only to find incumbents or in
explicitly uncertified search.

The integer-pruning branch adds certified upper-bound arithmetic, not a
certified dominance score. A loose equal upper bound says that two routes
have the same ceiling; it does not say that their final rewards or available
continuations are equal. The current record's positive-prefix intervals and
global negative credit are bound operands, not a complete signed canonical
prefix-cost ordering.

Two exact partial scalar costs can also lose their equality after a changed
floating-point summation grouping. The canonical code constructs different
FP64 edge lengths at different coordinates, may split nominally equal hops
into different numbers of subsegments, and reduces their norms. That is a
proof obligation to bound, not a reason to require bitwise route equality or
abandon merging under the existing 1e-6 objective tolerance.

## Flight 2224 is a particularly favorable specialization

In the captured final arena, all 250000 prefixes have zero positive charges,
there are no negative prices in the snapshot, and all endpoint paid masks are
zero. Thus every active prefix claim has price zero. For a fixed continuation,
different old zero-price rows cannot change the completed resource price in
this immutable snapshot. This removes the price-identity obstacle here.

There are only 967 existing structural keys among the 250000 histories, with
3140 histories in the largest key. Within that key, the root, time, hops,
first arc and recent cells already match. The general revisit and root
counterexamples do not justify retaining every member of this particular
group. Remaining work is a canonical geometry/feasibility equivalence or a
proved bounded-difference argument and its certificate bookkeeping.

The 967 count is an observed grouping opportunity, not a prediction that a
merged full search would use exactly 967 labels or complete the solver.
The saved arrival probe also makes later arrivals noncompetitive in this
snapshot. Proving that all competitive completions must use the shortest
119-hop length would make them monotone; that could remove additional recent
history distinctions. Such a restriction must come from a valid objective
bound, not from assuming the preferred route shape is optimal.

An actual-record audit gives both sides of the distinction:

- Records 4 and 5 both reach (314,186) after two hops, at time 285, with zero
  prefix price. Their preceding paths are respectively
  `(315,187) -> (315,186) -> (314,186)` and
  `(315,187) -> (314,187) -> (314,186)`. Returning to (314,187) is allowed from
  the first and forbidden from the second; returning to (315,186) reverses
  those verdicts. These continuations meet the current role/hop constraints.
  A separate objective argument may show such detours are uncompetitive,
  but same cell/time/price alone has not done so.
- Selected 32 evenly spaced histories from the largest identical-key group,
  appended one common legal 102-hop continuation, and validated each complete
  119-hop route with `pricing.certify_column`. All 32 were valid and had
  **bit-identical** canonical cost 172.07795355923773, complete dual charge
  52.14694860440556 and reduced cost 3.9999998948933353. Their differing prefix
  claim sets all contained zero active priced rows.

The second result supports the user's intuition for the richer group. It is
a sample with one common suffix, not a proof over all 3140 histories and all
continuations. Raw data and guarded canonical-only reproduction are in
`.agent/merge-audit/results.json` and `.agent/merge-audit/audit.py`. The shared
benchmark lock covered the diagnostic; no native CPU pricer was invoked.

## A minimal certifiable route forward

1. Start with a sufficient state, initially finer than necessary: immutable
   graph/snapshot/domain, root, absolute time, consumed hops, recent-history
   state, endpoint-role/first-arc information, and exact active overlap state.
2. Keep a signed price accumulator and an interval describing the original
   canonical cost, rather than comparing the FP32 ranking scores. For a
   zero-priced equal-hop group like 2224, a proved bound around a common ideal
   geometry may be much cheaper than a general exact accumulator.
3. Prove the representative preserves feasible continuations, including
   canonical wall and detour-budget acceptance. A representative that later
   fails validation cannot stand in for another history without that proof.
4. For exact dominance, tie-break arbitrarily but deterministically. For
   approximate dominance, carry the worst represented objective difference
   with the survivor, propagate it to children and pruning/certificate
   uppers, and retain it across interruptions and caps. Repeated epsilon
   tie-breaking without this accounting is not transitive and can accumulate
   more than the allowed tolerance.
5. Validate every merged-prefix continuation exhaustively on small graphs,
   then audit real groups against canonical completions and compare matched
   certificates. Do not reuse the existing unchecked grouping switch as a
   certificate by changing only its flag.

No merging implementation or performance claim is made by this report.
