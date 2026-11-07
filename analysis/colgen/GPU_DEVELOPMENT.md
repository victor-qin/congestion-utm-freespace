# GPU pricing: implementation boundary and next work

Status: 2026-11-06. The endpoint reachability bound is implemented. Endpoint paid
flags, certified state merging, and parallel CPU preparation for GPU pricing are
plans, not implemented features. CUDA has an adapter but has not been tested on
an NVIDIA machine. This document separates those plans from the current merge.

## What the search cap means

Metal expands partial routes. The Python owner collects the surviving states,
checks proposed numerical deletions in FP64, validates completions, and retains
predecessors so routes can be reconstructed. The default `max_labels=250000`
limits cumulative predecessor records in one search stage, including the pending
next layer. It is shared by a batch, not allocated separately to each flight.
It is not a GPU memory-allocation failure or a cap in the original CPU pricing DP.

Certified GPU search currently retains histories unless a proved bound removes
them. Approximate grouping is used only for the initial feasible-route search
and explicitly uncertified FP32 mode. Equal cell/time keys alone do not establish
dominance for canonical trajectory feasibility and cost.

An initial cap triggers a stronger future-bound table and a fresh search from
all roots. A second cap returns a valid incumbent, if found, with no certified
upper bound. All flights in that stopped batch currently receive the incomplete
status. This explains why 170 failures in 85 two-flight batches became 88
individually capped flights and 82 complete flights when replayed separately.

The original CPU backend remains available and has its own worker pool. It is
not called to certify a GPU result. CPU preparation time and search-state growth
are different problems: parallel preparation addresses the former, not the cap.

## Implemented: reachability strengthens endpoint charges

A capacity row identifies a resource and absolute time. A cell row at time
\(\tau\) can be charged by a visit at time \(s\) only if
\(s\in[\tau-\mathrm{hi},\tau-\mathrm{lo}]\), where the offsets describe the
occupancy window around a visit.

Let \(e(x)\) be a lower bound on the earliest visit to cell \(x\), over **all**
allowed departure roots. Let \(d(x)\) be a lower bound on its remaining hops to
any destination. For a completion at time \(T\), overlap is possible only if

\[
\max(e(x),\tau-\mathrm{hi})
\leq \min(T-d(x),\tau-\mathrm{lo}).
\]

If the intervals do not intersect, a mandatory positive endpoint price can be
included separately without charging any visit twice. `gpu_future.py` now uses
this test, a multi-source shortest-arrival pass over all arc roles, and the
existing reverse distances. Ignoring role/hop/revisit restrictions makes the
reachability bounds optimistic and therefore safe. Every real rooted route
remains covered. Arbitrary unrooted suffixes are outside this table's contract.

The extra shortest-arrival pass runs when the stronger future table is built;
it is not a new LP-independent persistent cache. Root availability can change
with restrictions. Price-dependent tail sums still refresh with prices. This
change adds no new arrival-by-cell DP dimension and no GPU kernel.

For flight 322, the priced cell cannot be visited before 472, and needs one more
hop to the destination. To arrive at 472, it would need to be visited by 471.
The endpoint charge can therefore be included even though its time window
overlaps the route's overall clock range.

Measured validation: all 88 individually capped cases were replayed on Metal
at the same 250,000-record cap and original first-round prices. Now 26 complete
with certificates and 62 remain capped. This fixes part of the search growth;
the whole 88-case validation still reports failure. Flights 278 and 322 now
certify at their saved CPU reference optima. Flight 32 finds its reference
optimum but remains uncertified. Four previously certified controls stay
certified, and all six certified selected-case upper bounds cover their saved
CPU reference values. CPU full pricing was forbidden throughout GPU replays.
The 109 targeted tests pass; Ruff and diff checks pass. Local evidence is in
`.agent/gpu-endpoint-bound-20261106/summary.json`. This is not a new full-1537
replay, full solver run, or timing speedup comparison.

## Proposed: paid flags without another whole-graph DP

### State and price independence

Each bit represents one **complete row identity**, including absolute time and
level, not just a hex. The bit means that the origin-and-visit union of this
partial route already contains that row. It does not mean that the route has
already paid every future endpoint cost.

Start with two `uint32` words, up to 64 tracked rows per flight and LP snapshot.
Use unsigned shifts and preserve both words in the shared Metal/CUDA record ABI.
Keep them separate from the existing finish/continue/prune status flags; update
the recent-history offsets explicitly rather than extending an implicit copy loop.

Cache geometric endpoint membership across LP rounds: for every allowed arrival,
which nominal destination rows are mandatory (`timing_steps=0`). Select up to 64
positive-price rows from that support at each new snapshot, initially by price
with a stable row-identity tie break. A bounded sparse selection is much smaller
than rerunning a weighted DP over all cells and times. Keep its row-to-bit mapping
fixed throughout the search. New labels and weighted caches start fresh next LP.

All positive endpoint rows fit in one 32-bit word for four inspected hard cases:
flights 2, 32, 278 and 322 have respectively 14, 11, 27 and 9 such rows across
971, 1033, 1029 and 1014 arrival slots. Their geometric endpoint supports contain
3932, 4180, 4164 and 4104 rows respectively. These are preparation-only counts
from the saved first-round capture, not flag-search performance measurements.

Rows outside the selected 64 are omitted only from this additional bound, not
from actual route costs or canonical validation. Omission of positive costs
weakens a lower bound safely. All negative prices retain the existing global
negative-price credit. A later LP may add rows or change signs: refresh the
selection, weights and snapshot ID, and never reuse old weighted minima.

Initial version: retain bits for the entire search. Do not roll bits forward or
forget old visits. Forgetting needs a proof covering every remaining endpoint
window as well as visit windows, and is unnecessary for a fixed 64-bit subset.

### How flags strengthen pruning

Let \(P\) be the already implemented positive-price lower sum for the union of
origin claims and visits. Let \(M\) be its paid bits, \(S\) the tracked rows,
and \(E_T\) the guaranteed endpoint claims for arrival \(T\). Define

\[
Q_M(T)=\sum_{r\in(E_T\cap S)\setminus M}\max(0,\lambda_r).
\]

Every completion must claim those residual rows. It may claim them in a later
visit or at the endpoint; this bound counts them once either way because it
does **not** separately include future visit charges. A valid cost-plus-price
lower bound is

\[
P+\min_{T\in I}\left[c_{\mathrm{lower}}(\text{root},T-t_{\mathrm{start}})
+Q_M(T)\right]+N,
\]

where \(N\leq0\) is the global negative-price credit and \(I\) includes every
possible completion time. Use the interval from current time plus relaxed
remaining hops to the smaller of the horizon and root start plus hop limit.
Allowing additional infeasible times is safe but weakens this bound.

Example: one tracked row costs 5 and is mandatory at the chosen arrival. A
prefix that already visited it has `paid=1`, includes 5 in \(P\), and adds 0.
A prefix that has not visited it has `paid=0` and must reserve 5. If a later
visit pays it, that visit raises \(P\) and sets the bit, so the endpoint remainder
drops to 0. Both routes account for the row exactly once.

Convert the lower bound to a conservative reduced-cost upper bound, then take
the **minimum** with the existing upper bounds. Never add this bound to the
current future-price bound: their air and price terms overlap.

### Avoid a root-by-mask-by-arrival table

The exact expression above motivates the bound; its first implementation can
use two cheaper conservative versions. Let \(a\geq0\) be the existing affine
air-cost lower coefficient per hop and \(t_{ref}\) the earliest root time.
For each encountered paid mask, construct two one-dimensional arrays:

\[
Q[T]=Q_M(T),\qquad H[T]=Q_M(T)+a(T-t_{ref}).
\]

Use downward arithmetic and build a range-minimum index for each array. A range
minimum answers "what is the cheapest value in this allowed arrival interval?"
without scanning every arrival for every label. A segment tree uses linear
storage and logarithmic query work. Both arrays are shared by all roots of the
same flight, price snapshot and mask.

For a label at time \(t\), compute

\[
q=\min_{T\in I}Q[T],\qquad
d=\max(0,\operatorname{down}(\min_{T\in I}H[T]
-\operatorname{up}(a(t-t_{ref})))).
\]

The first bound combines \(q\) with the existing clamped geometry lower bound
at minimum total hops. The second combines \(d\) with the existing affine air
bound through the current hop. This retains a penalty for later arrivals, so
waiting for a zero-price endpoint is not incorrectly treated as free flight.
Both are conservative; neither requires a table indexed by root identity.

The numerical helper must preserve canonical rounding allowances. The current
`future_upper` caller takes **visit-only** prefix prices because its suffix can
overlap origin claims. Add a clearly named endpoint wrapper or generalize that
helper's contract; do not silently give the existing future table an origin
union prefix. That would double charge some routes.

```text
once per flight geometry/configuration/time domain:
    endpoint_rows[T] = guaranteed nominal endpoint claims
    cache endpoint membership and sparse visit-incidence metadata

each LP price snapshot:
    choose up to 64 positive endpoint rows; freeze row-to-bit mapping
    endpoint_mask[T] = selected rows belonging to endpoint_rows[T]
    reset weighted range-min caches and search labels

initialize root:
    paid = mask(origin claims UNION initial visit)
    P = existing verified positive sum for that union

GPU expansion to (target, next_time):
    paid_next = paid OR visit_mask(target, next_time)
    update existing prefix-price bookkeeping
    propose pruning with old bounds and available endpoint range minima

CPU verification of a proposed numeric deletion:
    use exact integer masks and original FP64 prices
    I = [time + remaining_hops, min(horizon, root_start + hop_limit)]
    if a proved time domain is empty: close as infeasible
    if weighted cache entry is missing: use an existing safe bound
    otherwise:
        q = query Q over I
        d = downward(query H over I - upward(a * (time - t_ref)))
        U1 = canonical_upper(clamped_geometry_lower, disjoint_prices=(P, q), negative_credit)
        U2 = affine_canonical_upper(prefix_air_lower, disjoint_prices=(P, max(0,d)), negative_credit)
        U = min(existing_upper, U1, U2)
        discard only if verified U closes against the validated incumbent
        retain U in the reported closed-branch upper bound
```

Generate current-layer visit masks alongside the sparse visit-price gather,
shared by every label entering the same cell at that time. Set root bits from
the exact root union already available in preparation. Keep parent-aligned
integer masks for verification; no route reconstruction is needed to update
the bits. Two words add 8 bytes per retained record (2 MB at 250,000 records),
plus transient candidate buffers and lookup storage.

### Bound the overhead and test the claim

A 64-bit mask does not imply a small number of distinct masks. Never allocate
all combinations. Build weighted arrays lazily, starting with masks actually
encountered, and impose a build-work and byte budget. On a miss beyond that
budget, use the existing safe bound; never drop a route or interpret a missing
entry as unreachable. Store conservative GPU arrays as well as FP64 verification
data so useful new bounds can actually trigger GPU pruning proposals.

Try this bound before the existing expensive retry, for flights whose ordinary
root bounds leave work. Root masks receive entries first. At a layer boundary,
collect distinct masks among surviving labels and consider adding entries for
frequently reused masks within the build budget. Upload new conservative trees
once; GPU expansion threads query them, and FP64 verification independently
uses the original-price trees. Missing entries have an explicit validity bit,
not an infinity that might be mistaken for proof of infeasibility. Clearing or
evicting a cache entry changes available pruning strength, not the route domain.
Retain the current future-table retry until measurements justify replacing it.

For scale only, with 1014 arrival slots and 64 cached masks, two segment trees
rounded up to 1024 leaves require about 2 MiB in FP64 and 1 MiB in FP32 per
flight, excluding construction arrays and metadata. This is a memory estimate,
not measured performance or a commitment to a 64-entry default. Build positive
sums in vectorized downward operations over the selected rows; do not introduce
a Python loop over every label and arrival. Cache counts and build limits must
be visible in profiling output.

If there are \(A\) arrival slots, \(K\leq64\) tracked rows and \(C\) built mask
entries, a simple build costs up to \(O(CAK)\) per LP with \(O(CA)\) storage;
range queries cost \(O(\log A)\). This is not free, but it removes the graph-cell
dimension and root-count multiplier. Numeric arrays must refresh after every
LP; only membership and geometry can persist. Measure mask counts, build time,
cache hit rates, bytes, extra pruning and total pricing time before claiming it
is cheaper. Start with a small configurable cache rather than guessing that all
masks will be useful.

Tests must cover full origin/visit/endpoint unions, repeated visits, newly priced
and sign-changing rows across LPs, multiple roots/clocks, bit 31 and bit 63,
different flights in one batch, mask truncation, cache exhaustion, unbuilt versus
empty arrival intervals, outward arithmetic, and exhaustive tiny-route optima.
Then replay the individually hard cases and previously certified controls on
Metal. Equal masks alone do not authorize certified state merging; geometry,
recent history and completed-cost dominance still require their own proof.

## Code intended for the GPU merge

| Area | Files or changes |
| --- | --- |
| New production modules | `gpu_bounds.py`, `gpu_future.py`, `gpu_prices.py`, `gpu_pricing.py`, `gpu_runtime.py`, `pricing_gpu.h`, `pricing_platform.py` under `freespace_sim/planner/colgen/` |
| Solver integration | GPU options in `params.py`, cache slots in `network.py`, GPU sweeps in `pricing_pool.py`, bounds/repair/status handling in `solver.py` |
| Shared preparation hooks | `DualView._gpu_visit_prices` in `pricing.py`; optional retained root unions in `dp_prepare.py` |
| Installation | Optional macOS Metal dependency in `pyproject.toml` and `uv.lock` |
| Repeatable analysis | `platform_replay.py`, `gpu_validation.py`, related `benchmark.py`/`common.py` CLI support and README; explicit flight IDs replace hardcoded replay scripts |
| Validation | GPU/platform tests, related pricing-pool and solver coverage; shared CPU preparation/kernel regression checks |
| Local scratch policy | `/.agent/` ignored; captures, profiles and historical exploratory scripts remain local and are not deleted |

Earlier CPU kernel/preparation changes also exist in the working tree:
`bootstrap_kernel.py`, `dp_kernel.py`, much of `dp_prepare.py`, removed unused
helpers in `pricing.py`, and their CPU tests. `pricing.py` and `dp_prepare.py`
contain both sets of edits: separate by hunks or include the reviewed dependency
explicitly, never stage/revert entire files assuming a clean GPU-only boundary.
The edited `AGENTS.md`, older context notes and historical figures are separate
from the production GPU change. This is a scope map, not an assertion that any
files have been staged, committed or merged.

## Parallel CPU preparation and setup can proceed separately

`n_pricing_workers` currently controls the existing complete CPU pricing pool.
It does not parallelize the CPU preparation inside GPU pricing. The planned
separate feature is persistent CPU processes preparing flights while one owner
process drives Metal/CUDA:

1. Extract a serial per-flight preparation result with explicit FP64 data and
   exact claim membership; preserve behavior before adding processes.
2. Give each persistent worker fixed flight ownership so topology/geometry stay
   warm. Graph pickling currently discards search caches, so resending whole
   graphs each round would defeat this.
3. Tag jobs/results with the LP snapshot and root/restriction configuration.
   Prefetch a bounded number of prepared batches; discard stale replies.
4. Keep the shared sparse price cache and GPU context in the owner. Do not use
   the CPU pool's complete pricing operation as a substitute for preparation.
5. Test serial/worker parity, changed prices, exclusions, deadlines, worker loss
   and cleanup. Measure 1/2/4 workers on warm captures, including transfers and
   peak memory; more workers are not a guaranteed speedup.

NVIDIA environment setup and CUDA parity tests form another independent task
once the supercomputer environment is available. No NVIDIA availability or CUDA
performance is assumed here. Before an actual merge, run relevant CPU/shared
tests, the GPU suite and full repository checks, actual Metal captured-price
and small solver checks, and clearly retain the remaining capped-search and
untested-CUDA limitations. Do not present an experimental backend as proven
full-workload convergence.
