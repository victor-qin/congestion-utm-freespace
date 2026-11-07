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
uses a separate invocation-owned payload after ordinary preparation. PR145's
optional final `prepared` argument can pass through the batch-class selection;
`IntegerBatch` inherits that constructor. This branch does not merge or validate
the combined PR145 implementation, add worker numeric payloads, or overlap CPU/GPU work.

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
performance are separate gates, recorded below.

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
13,515 roots eagerly caused an early numeric-buffer cap on flight 1920 in the
hard88 solver; retaining only needed tables removes that artificial early exit.

## Final functional and containment validation

On 2026-11-07, 378 distinct tests passed across the nine focused files below.
The first combined run passed 309 tests before two legacy-capture fixture
failures: filtering out only historical GPU fields accidentally included newly
appended integer fields. Archived positional states are prefixes. Correcting
that fixture and adding the pre-integer prefix produced 71 passing option and
validation tests (two passes overlap the first run); no production slot reorder
was needed.

| Requirement | Focused coverage |
| --- | --- |
| Outward conversion, checked overflow, ordered binary64 operations | `test_colgen_gpu_integer.py`: exact rational oracles, subnormals/extremes/cancellation, actual Metal arithmetic/expression probes, CUDA syntax/interface |
| Original geometry and canonical completion bounds | `test_colgen_gpu_bounds.py`, `test_colgen_gpu_endpoint.py`: every prefix of exhaustive small canonical completions, cost-vector/scalar parity, signed unions and endpoint choices |
| Exact membership and legal domain | `test_colgen_pricing_platform.py`, `test_colgen_gpu_endpoint.py`: both words including bits31/63, origin overlap, revisits, roles/hops/obstacles, large/negative clocks, exclusions, ties, both objectives |
| Ambiguity, changes and interrupted work | Platform/endpoint/future/runtime files: invalid scale retains states, new/sign-changing prices, missing endpoint data, future retry, rejection, cancellation, queue/state/memory/deadline limits |
| Solver and configuration contracts | `test_colgen_gpu_solver.py`, `test_colgen_gpu_options.py`, `test_colgen_gpu_validation.py`: actual Metal HiGHS/Gurobi paths, guarded CPU entry points, canonical completions, archived prefixes |

The final hard88 audit certified 88/88 and independently checked 1,130,566
integer deletion proposals against the original FP64 verifier. Late18 certified
18/18 through cold, same-price warm, changed-price, changed-price warm and restored
snapshots. The maximum reported gap was \(1.00000000103\times10^{-6}\), the
represented cutoff tolerance. Non-audit timing and solver runs guard `_Batch.bound`
and all original full-CPU pricing entry points. Independent canonical validation
remains enabled. There is no reference-route oracle seed.

The implemented ambiguity policy is retention, as permitted by milestone 5.
There is no wider-integer/software-binary64 refinement queue or ambiguity-rate
counter. Overflow can increase work or cause an explicitly incomplete cap; it
does not force deletion. No certified heuristic merging, removal of CPU canonical
validation, proof of exact route identity, or NVIDIA execution is claimed.

## Matched performance and retention decision

[Committed measurements](integer_pruning_results.json) record actual Apple M1 Ultra
Metal runs at numeric source `d8879f8f`. Hard88 and late18 have two fresh-process
integer repeats and two reference repeats; controls have one pair. The earlier
reference repeat differs only in the unused integer implementation files. The
ordinary reference path is unchanged. All compared fixed-price phases complete
with the same certificates and **exactly equal reduced costs**; no capped phase
is included in these percentages.

| Cohort / phase | Reference median, s | Integer median, s | Wall-time change |
| --- | ---: | ---: | ---: |
| hard88 cold | 78.323 | 65.257 | 16.68% lower |
| hard88 same-price warm | 38.909 | 25.861 | 33.53% lower |
| late18 early-price cold | 18.312 | 16.794 | 8.29% lower |
| late18 early-price warm | 6.161 | 4.675 | 24.11% lower |
| late18 changed prices | 2.677 | 3.547 | **32.50% higher** |
| late18 changed-price warm | 1.733 | 2.609 | **50.60% higher** |
| late18 restored early prices | 6.222 | 4.695 | 24.54% lower |
| controls4 cold / warm (one pair) | 3.014 / 0.840 | 2.946 / 0.746 | 2.25% / 11.21% lower |

Retain the implementation **opt-in**, with no default change. Hard88 cold and warm
benefits repeat and exceed the earlier 15% target. The changed late snapshot has
far fewer CPU checks to remove, so numeric preparation/transport outweighs that
saving. This is not a uniform acceleration of pricing or complete solver convergence.

The fresh hard88 warm pair illustrates the mechanism and cost (inclusive timers
overlap; MB below means decimal megabytes):

| Measurement | Reference | Integer |
| --- | ---: | ---: |
| Original CPU preparation | 0.798 s | 0.794 s |
| Added integer preparation / table gathering | 0 / 0 s | 3.420 / 0.418 s |
| Per-prune CPU verifier / its inclusive time | 1,132,147 / 12.600 s | 0 / 0 s |
| Canonical validations / completion time | 1,674 / 4.256 s | 1,674 / 4.254 s |
| Search labels / candidates | 511,170 / 2,467,958 | 511,172 / 2,467,993 |
| Metal execution / host waiting | 0.641 / 6.540 s | 1.598 / 6.177 s |
| Allocations / cumulative allocated volume | 14,894 / 340.95 MB | 25,468 / 1,171.27 MB |
| Upload / download volume | 667.89 / 187.08 MB | 1,559.67 / 352.87 MB |
| Wait count | 16,083 | 16,176 |
| Peak resident / numeric persistent storage | 23.18 / 0 MB | 42.97 / 2.25 MB |
| Process peak RSS after warm phase | 3.23 GB | **5.07 GB** |

Preparation remains on the CPU, and canonical completion validation costs remain.
The benefit comes from removing repeated numerical verification and prefix work,
not fewer alternatives, dispatches or transfers. Larger records and extra operand
buffers increase transfer/allocation volume and observed process RSS. Persistent
component budgets do not bound total process memory. There is no separate numeric
refinement queue; ambiguity is retained and its frequency is not instrumented.
Graph/snapshot/runtime setup for this pair was approximately 0.026/0.029/0.095 s
in both modes and is excluded from phase wall time. Cold pricing preparation
itself was 39.94 s integer versus 39.96 s reference, included in the cold totals.

Canonical solver checks used licensed Gurobi with one thread and original captures:

| Subset | Reference | Integer | Completion and quality |
| --- | ---: | ---: | --- |
| long10 | 8.474 s | 8.804 s | 10/10 certified; same cost 1895.591897429123; canonical filing validation passes |
| hard88 | 52.326 s | 46.052 s | Both stop at flight 2224, round 2 `label_limit`; 102/103 pricing results certified; same feasible cost 8716.244488722932 |

The hard88 run remains **incomplete**. Reference lower bound 8604.35016967624
versus integer8604.350157717476 gives approximately 1.2837% gap in both cases;
the slight lower-bound difference reflects conservative numerical closure.
This single matched capped-solve pair is not evidence of complete convergence
speedup. The long10 pair shows no benefit at this scale.

The evidence also retains the rejected all-root allocation trial: 42.504 s but
an earlier `integer_buffer_limit` at flight 1920, 91/92 certified and worse
cost 8720.244488722932. That timing is excluded from performance comparisons.
The retained-root correction restores the reference stopping point without
raising the numeric budget. Final-source verification, exhaustive checks and
matched measurements justify this bounded opt-in implementation; wider GPU
refinement, lower transport/RSS, PR145 integration and complete hard88 solver
convergence remain future work.

## Reproduction

Use the shared environment and original captures. The cohort order comes from
`arrival_union_results.json`, not a newly selected easy subset. A cold phase is
the first pricing call in a fresh process; graph construction, snapshot indexing
and backend compilation are separately reported. Every phase includes independent
canonical validation. Instrumentation is identical between modes. All sustained
jobs, including tests, use the same advisory lock and release it between jobs.

```sh
export PYTHONPATH="$PWD"
/Users/victor/Documents/Research/congestion-utm-freespace/.venv/bin/python - <<'PY'
import fcntl, json, subprocess, sys
from pathlib import Path
root = Path('/Users/victor/Documents/Research/congestion-utm-freespace')
cohort, mode = 'hard88', 'integer'  # also late18; reference or audit
saved = json.loads(Path('analysis/colgen/arrival_union_results.json').read_text())
ids = [row['flight_id'] for row in saved[cohort]['after']['rows']]
early = root / '.agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl'
late = root / '.agent/runs/colgen-capture-20261105-211249/pricing_captures/round_0011.pkl'
cmd = [sys.executable, '-m', 'analysis.colgen.execution_replay', '--capture', str(early),
       '--flight-ids', *map(str, ids), '--batch-size', '1', '--warm-repeats', '1',
       '--out', f'.agent/replay-{cohort}-{mode}.json']
if cohort == 'late18':
    cmd += ['--capture', str(late)]
if mode != 'reference':
    cmd += ['--integer-bounds']
if mode == 'audit':
    cmd += ['--audit-integer']
with (root / '.agent/gpu-followups-20261106/metal-benchmark.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    subprocess.run(cmd, check=True)
PY
```

For the canonical solver, run `analysis.colgen.gpu_validation solve` under the
same lock with the early `--capture`, a fresh `--out`, `--budget 300`, and optional
`--integer-bounds`; set `COLGEN_GUROBI_THREADS=1`. Its default subset is the saved
long10 selection; supply the exact hard88 IDs above for that solver comparison.
An incomplete cap intentionally returns a failing validation exit code and must
be inspected, not relabeled as a completed solve. Controls use IDs 0,34,276,320.

Run the eight `test_colgen_gpu_{integer,bounds,endpoint,future,runtime,solver,options,validation}.py`
files plus `test_colgen_pricing_platform.py` under the lock for focused coverage.
The summary command is `python -m analysis.colgen.integer_pruning_report --replays DIR
--solver SOLVER_DIR --out FILE`; repeat `--solver` for each paired result directory.
It checks input hashes, cohort order and completed objectives before aggregating
timings. Raw output hashes, per-phase result hashes and source manifests remain
in the committed evidence. Timers overlap and are not additive.
