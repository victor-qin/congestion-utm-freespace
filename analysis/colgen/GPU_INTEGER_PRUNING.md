# Integer containment for GPU pruning

Baseline: `fc30fca4c410602111e724688d3b924c428b6f03`. This is a staged
implementation/proof record. A primitive passing its tests does not establish
a complete search certificate. PR144 remains draft and unmerged.

## Reference and decision contract

The reference is `_Batch.bound` in `gpu_pricing.py`, not the native CPU pricing
search. For a state \(s\), it returns the minimum of available independent
reduced-cost uppers:

1. `completion_upper_bound`: canonical geometry cost plus the positive union
   of mandatory origin and prefix visits, with every negative input price.
2. `CompletionBounds.future_upper`: visit-only prefix plus a relaxed suffix;
   separate origin charges are omitted because they can overlap that suffix.
3. `CompletionBounds.endpoint_upper`: full positive prefix union plus unpaid
   mandatory arrival claims; its clamped and affine bounds are also independent.

The numerical target is \(U_I(s)\ge U_R(s)\). Omitting a bound or replacing it
with a weaker relaxation is allowed. Adding endpoint and future costs is not:
their claim sets can overlap. State merging has no certificate here.

The host's cutoff is the represented binary64 value
\(C=\operatorname{fl}(b+10^{-6})\), where \(b\) came from a canonically validated
route. Encode \(C\) downward. A finite integer upper can delete only when its
upper endpoint is at most that encoded lower cutoff. Preserve the maximum
discarded upper, rounded outward when returned as binary64. The certificate
is the maximum of that value and the final incumbent, not the incumbent alone.
Changing the tolerance or treating it as exact optimality is outside this work.

Missing data, nonfinite values, checked arithmetic overflow, unavailable scale,
or a comparison straddling the cutoff retain the alternative. Refinement may
increase precision, but cannot repair a weak mathematical bound. Queue, memory,
state and time limits remain incomplete exits, including pending refinements.

## Operation-level audit

| Reference operation | Required containment |
| --- | --- |
| `PriceSummary.negative` | Every negative row, original ordered `nextafter(sum + price, -inf)`; not a net signed total or FP32 table |
| `CompletionBounds.root_positive` | Complete exact origin/first-visit union; downward additions, then clamp at zero |
| `_Batch.positive_lower` | Visit claims minus exact origin union; no recent revisit overlap; separately rounded increment and parent addition |
| `_Batch.visit_lower` | Disjoint visit-only windows, with its own parent accumulation |
| `CompletionBounds.cost_lower` | Original binary64 geometry preparation and ordered multiply/add/subtract/divide/clamp operations, including geometry reconstruction allowances |
| Endpoint residuals | Exact two-word paid membership, allowed arrival range, downward trees; absent masks weaken rather than close |
| Future tables | Existing downward FP64 recurrence and actual remaining-hop lookup; a less restrictive suffix can only weaken the upper |
| `_roundoff_upper` | Preserve original gamma-32 canonical cost/`fsum`/final-subtraction allowance, not just the real-number sum |
| `completion_upper_bound` | Extra downward prefix accumulation and cost step, negative addition, then three ordered upward subtractions |

CPU preparation may encode already computed FP64 geometry/table/summary values
outward; their mathematical proof remains the existing reference proof. This
does not justify encoding GPU FP32 proposals as certified values. A new on-device
accumulation must separately enclose the reference's operational result.

`DualView.claim_cost` uses `math.fsum` for complete route scores. The initial
design retains original CPU canonical completion scoring and validation.
The reference bounds' existing canonical roundoff allowances remain necessary.
No claim that an exact integer sum reproduces `fsum` or canonical geometry is made.

## Primitive proof

Use an immutable scale \(q=2^e\), with \(-1074\le e\le1023\), and signed
endpoints in \([-(2^{63}-1),2^{63}-1]\), each stored as two unsigned 32-bit
words. Excluding the asymmetric signed minimum makes checked negation exact.
An invalid interval denotes unresolved information, not a large finite value.

For represented finite binary64 \(x=n/d\), compute
\(\lfloor n/(dq)\rfloor\) and \(\lceil n/(dq)\rceil\) using exact host integers.
No rounded multiplication by the scale precedes floor/ceiling. Endpoints that
do not fit invalidate the result. The scale selector uses exact input ratios
and a caller-supplied accumulation bound; runtime checks remain mandatory even
when that bound is wrong. Scale identity belongs to the immutable snapshot.

Addition uses low-word carry and high-word modular addition. Overflow occurs
when equal input signs produce the opposite output sign, or the excluded signed
minimum occurs. Subtraction is checked negation then addition. Multiplication
by a nonnegative uint32 uses checked binary doubling/addition; any failed
intermediate invalidates the result. All device word operations are unsigned,
so signed-language overflow cannot silently authorize a deletion.

For integer interval endpoints of magnitude at most \(m\), let \(k\) be the
bit length of \(m\), and set
\[
p=2^{\max(1,k-51)}.
\]
Before modeling a rounded binary64 operation, require \(e+k\le1023\), a
conservative finite-range guard. The real operation lies below magnitude
\(2^{e+k}\). Its binary64 rounding error followed by one `nextafter` is at
most \(1.5\) spacings, bounded by \(pq\); the minimum \(2q\) also covers
subnormals since \(q\ge2^{-1074}\). A downward operation is enclosed by
\([L-p,H]q\), an upward one by \([L,H+p]q\), and nearest rounding by
\([L-p,H+p]q\). Checked padding overflow invalidates. Multiplication by
uint32 uses its exact binary64 integer value before the same enclosure.

This is intentionally wider than exact binary64 emulation. It can lose a
near-tie deletion, never justify an additional unsafe one. Exact rational
oracles, original FP64 shadow calculations and actual Metal execution must
independently exercise carries, signs, cancellation, subnormals and overflow.

## Integration gates

1. Independently test primitive containment before using it to discard states.
2. Prove positive accumulation against the reference even if row iteration order
   differs. A real sum alone is insufficient. Preserve exact origin overlap.
3. Bind all numeric arrays to the current graph, snapshot, clocks, objective,
   restrictions, roots and complete memberships. Newly nonzero or sign-changing
   rows must rebuild weighted data. Both paid words must survive every copy.
4. Carry intervals through resident predecessors and interruption/retry paths;
   preserve unresolved and discarded-upper accounting. Canonical completions
   remain CPU-validated. No CPU full-search fallback is permitted.
5. Exhaustively compare every valid prefix against canonical completions and
   shadow bounds. Exercise failure paths and guard per-prune CPU verification.
6. Only then measure original hard88, late-price cases and canonical solver
   subsets, separating cold, warm and changed-price preparation/search/checking
   and matched completion/certification. Use the shared benchmark lock, including
   sustained CPU tests while the diagnosis branch profiles. CUDA source checks
   are compatibility evidence, not NVIDIA execution.

The parallel-preparation branch owns `prepare_flight` extraction. Numeric data
will use a separate immutable payload rather than changing that contract before
the arithmetic and membership gates pass. Production integration and speedup
are not yet established.

## Prototype validation

The first focused gate passed 20 tests on 2026-11-07 under the shared lock:
exact-rational conversion/arithmetic oracles, every ordering of small positive
price sets, cancellation/subnormal/extreme values, checked overflow and ambiguous
cutoffs. Actual Metal executed 6,000 multiword arithmetic cases and 500 complete
bound-expression cases. Original binary64 expressions were independently checked
against the latter uppers. Generated CUDA source passed a C++ syntax check with
CUDA annotations/intrinsics stubbed; no NVIDIA hardware was used.

These tests validate arithmetic given valid operands. Exact membership transport,
resident interruption/accounting, the production verifier guard and matched
performance are separate gates. The opt-in integration below is still being validated.

## Opt-in integration contract

`gpu_integer_bounds` selects a separate batch class after ordinary preparation.
The default path remains unchanged. `gpu_integer_audit` independently reconstructs
original-price prefixes and checks each proposed device deletion with `_Batch.bound`.
Without auditing, that verifier is guarded in validation runs and is not a fallback.
Original canonical completion validation and scoring remain enabled in both modes.

The numeric payload belongs to one invocation, graph domain and immutable price
view. It is rebuilt for changed prices, clocks, exclusions, objectives and roots;
there is no persistent weighted numeric cache. It carries exact positive row
identities and absolute times, original negative credit, original geometry/hop
cost values and the original canonical arithmetic allowance. Each flight's
price/objective payload has one explicit scale. Packed intervals use five uint32
words: low/high lower, low/high upper, then validity. All bounds used by a state
share its payload's scale; scales are never mixed in arithmetic.

The positive-row lookup includes every currently positive claimable cell row.
Membership in the original paid-cell table excludes exact origin overlap before
summation. Origin terminal rows remain in the precomputed root union; visits
cannot claim terminal rows. Zero/negative rows do not add a positive charge, and
every negative row remains in the original global credit. New support and sign
changes rebuild the lookup rather than reusing an earlier positive-only table.

For \(n\) positive row additions, the integer real-sum enclosure receives a
downward allowance of \(npq\), with \(p\) derived from that sum's upper magnitude
as in the primitive proof. Every intermediate reference positive sum is between
zero and the exact total, so this covers every original row iteration order.
The parent-plus-increment operation receives a separate directed FP64 enclosure.
The revisit rule makes different visit windows disjoint; exact origin exclusion
then proves the full union induction. Visit-only prefixes have their own interval.

The resident record appends both prefix intervals and an audit upper (15 words).
All load/copy/compaction operations preserve the complete stride. The existing
two endpoint-mask words retain their exact mapping and update rules. Numeric
words do not change preliminary-search grouping; complete search never groups.
Each expansion first applies the existing integer structural rules, then erases
all FP32 numerical flags. Only the new checked comparison can authorize deletion.
Missing trees, masks, nonfinite operands or failed checked operations weaken the
bound and retain the state. The optional remaining-hop refinement can be omitted
by using the existing unrestricted suffix, which is no stronger than the reference.

Every direct deletion satisfies \(U_I(s)\le C_d\) for its dispatch's original
binary64 cutoff. The host preserves \(\max_d C_d\), a conservative envelope of
the discarded upper maximum. This avoids FP32 loss and a nonportable 64-bit
atomic maximum. It can report a slightly looser upper than the previous verifier,
but it remains at most the final represented incumbent-plus-tolerance cutoff.
Near ties therefore retain a valid objective gap rather than becoming exact
optimality claims. Root uppers are read as integer words and converted outward.

Root checks precede lazy continuation-table and resident allocation. Additional
persistent numeric host/device arrays have a separate 64 MiB budget; the existing
resident arena has its original 128 MiB budget. These are not total process-memory
limits. At most two suffix rows are retained across layers. Command blocks own
their immutable buffers until completion. All caps, deadlines, queue failures
and unresolved decisions remain incomplete exits. An aborted pass may discard
its old envelope only because the retry restarts the complete root domain.

Hop-cost tables are allocated only after a root survives its checked comparison
and predecessor allocation. Each new generation appends exact root-specific
operands and keeps existing offsets stable; device buffers remain immutable while
referenced by a command block. Replacement checks include old/new device buffers
and retained host rows. A missing offset disables that bound. Future preparation
restarts the whole domain and releases the old cost generation. Preparing all
13,515 roots eagerly caused an early numeric-buffer cap on flight1920 in the
hard88 solver; retaining only needed tables removes that artificial early exit.

The integrated focused gate passed on 2026-11-07: a 50-test arithmetic/search
smoke and 196 platform/endpoint/runtime checks (these overlap). Coverage includes
exhaustive signed/overlapping routes and both objectives, forced future retry,
both mask words at large absolute clocks, missing endpoint caches, changed
snapshots, near ties, cancellation, invalid integer operands, rejected canonical
completions, and queue/state/memory/deadline incompletion. The ordinary integer
path runs with the original numerical verifier and full CPU pricing guarded in
dedicated tests. Saved-case audits, solver-level validation and performance gates
remain outstanding; these counts are not a claim that all seven milestones passed.
