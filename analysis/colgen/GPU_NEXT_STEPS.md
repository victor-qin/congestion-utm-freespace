# Three follow-up branches on draft PR144

## Authorization and order

- 2026-11-07 [USER] Current authorization supersedes the earlier landing-first
  sequence: PR143 has landed; PR144 stays draft and unmerged while all three
  follow-up branches proceed. No automatic merge after their completion.
- Branch 1 runs at High effort for diagnosis only: measured function locations,
  figures, supporting data and a proposed improvement plan. Implementation
  requires the user's review and approval of that diagnosis.
- Branches 2 and 3 run at Extra High effort. Branch 3's first PR is synchronous
  parallel CPU preparation only. CPU/GPU overlap belongs on a separate branch/PR.
- The endpoint conversation coordinates PR143 and the handoff. The GPU execution
  conversation owns PR144 review, integration and landing. Each edits only its
  bound checkout. Independent reviews, when requested, use at least High effort and identify the exact
  reviewed commit and base.
- PR143: <https://github.com/victor-qin/congestion-utm-freespace/pull/143>.
  PR144: <https://github.com/victor-qin/congestion-utm-freespace/pull/144>.

For each landing, resolve review findings, test the final combined source,
verify that the remote head still matches the reviewed head, and merge only
after required checks pass. PR144 needs a fresh review after integrating landed
PR143, even if its earlier review is clean. Record both merge commits and the
resulting baseline in the handoff if a future landing is authorized. The three
follow-ups start from PR144's pushed `fc30fca4c410602111e724688d3b924c428b6f03`,
which includes landed PR143; they do not need to wait for PR144 to merge.

## Common requirements

Keep genuine Metal search, complete signed claim unions, current exclusions,
all legal starts and hops, canonical route validation and original FP64 scoring
semantics. No original full CPU pricing fallback or certifier. Caps, deadlines,
overflow and unresolved alternatives must not become successful certificates.
Unchecked FP32 and unaudited epsilon modes remain explicitly uncertified.
CUDA source/interface compatibility is required; NVIDIA hardware is untested.

Use independent worktree-bound conversations and PRs from the common pinned
baseline. Read local `AGENTS.md` and maintain local `.agent/CONTINUITY.md`. Read
analysis scripts against current production sources before reusing them.
Serialize sustained Metal workloads, CPU-intensive tests and timing comparisons with the shared
advisory lock at
`/Users/victor/Documents/Research/congestion-utm-freespace/.agent/gpu-followups-20261106/metal-benchmark.lock`.
Source inspection and editing may proceed independently. Release the lock between jobs.

Benchmark first-use, same-price warm and changed-price calls separately. Compare
matched certificates and solver completion, not dispatch counts or differently
capped runs. Report preparation, search records/candidates, allocations, copy
volume, waits and refinement/verification costs separately; overlapping timers
are not additive. Retain performance changes only after a repeatable material
wall-time benefit; the previous trials used a 15% retention threshold.

## Branch A: faster first construction and CPU preparation

Diagnosis only until user approval: return ranked function/file/line locations,
exclusive and cumulative timing, call counts, allocation/cache evidence, figures
and reproducible raw data. Separate unprofiled timing from profiler overhead.
Propose improvements with expected benefit, risks and functional validation;
do not implement those optimizations before the user approves the plan.

Start with a function-level profile of first-use preparation on the combined
pinned implementation. Topology already has a per-graph cache; PR144 additionally
caches GPU structures and memberships. Measure construction and invalidation
costs before proposing more caches.

Candidates are Python neighbor/role packing, exact root claim construction,
price gathering and packed-array assembly. After approval, compile hot loops or
replace object-heavy work with array operations while preserving exact resource
membership, role restrictions and conservative arithmetic. Keep zero-priced
memberships available for later new/sign-changing prices. Bound cache storage.
Extend lazy preparation behind proved bounds where useful; no legal start may
disappear through a heuristic gate.

Functional tests should compare prepared domains, memberships and original-price
bounds under changed prices, clocks, restrictions and objectives. Validate
resulting pricing certificates on small exhaustive graphs, the original hard-88
capture, late-price cases and representative complete solves. Keep profiling
overhead out of final wall-time comparisons. This branch does not own worker
pools, GPU arithmetic certification or resident-runtime redesign.

Historical context only: PR143's final first-use pricing preparation was
39.935 seconds of 77.162 seconds on the serial hard-88 replay; same-price warm
preparation was 1.625 of 38.808 seconds. These are phase wall times, not exclusive
CPU timers. Changed-price preparation must be measured on the new common base.

## Branch B: GPU-certified numerical search bounds

The initial goal is certified GPU pruning without a CPU check for every proposed
deletion. CPU preparation, the LP/integer solver and canonical completion
validation may remain. This is not initially a fully GPU-only solver.

Audit the reference proof and target
\(U_{GPU}(s)\ge U_{reference}(s)\) for every search state, using a conservative
incumbent cutoff. Preserve original FP64 operational semantics, including
`fsum`, `nextafter`, operation order, geometry allowances and the explicit
closure tolerance. Retain discarded upper maxima in the final certificate.
An exact real-number sum alone does not prove containment of the FP64 result.

Begin with a bounded prototype of outward integer intervals at a power-of-two
scale selected for each price snapshot. Prove accumulation limits and checked
multiword arithmetic before integrating production deletion rules. FP32 may
rank preliminary candidates but must not independently delete or merge states
in certified search. Exact integer memberships, paid masks, origin overlap,
complete signed unions and snapshot invalidation remain mandatory.

Ambiguous decisions refine on the GPU using wider arithmetic or remain live.
Missing tables, overflow, exhausted queues, state limits and deadlines cannot
authorize deletion or certification. Keep independent valid bounds by taking
the minimum of their reduced-cost uppers; never add overlapping endpoint and
visit charges. Certified state merging needs its own complete proof.

Validate primitive containment with exact arithmetic oracles and the complete
bound against FP64, including cancellation, extreme/tiny values, both paid-flag
words, revisits, role/hop restrictions and exhaustive canonical completions.
Use CPU shadow audits during development, then guard the per-prune CPU verifier
to establish that the intended production work is removed. Measure added
preparation/storage and refinement frequency as well as matched runtime.
Removing canonical completion validation is a distinct later scope.

Branch2 now has an opt-in checked-integer resident implementation and independent
FP64 shadow audits. Ambiguous/overflowed comparisons retain states; original
canonical scoring stays on the CPU. See [the proof and validation report](GPU_INTEGER_PRUNING.md)
for the measured retention decision, limitations and reproduction commands.
PR145 preparation remains a separate integration: forward its optional final
`prepared` argument through the batch-class selection. No overlap is implemented.

## Branch C: parallel CPU preparation; overlap in a separate branch

Prepare independent flights on persistent CPU workers and feed compact prepared
data to one GPU owner through a bounded ready queue. Keep graph/preparation caches
alive across rounds; repeatedly transporting whole graphs can lose those caches.
Reuse the landed preparation contracts and inspect the earlier unmerged pool
prototype before reusing any of it. Do not import it without source review.

First establish synchronous parallel preparation with exact result parity.
Fully drain each preparation group before GPU search in this first PR. A later,
separate branch/PR may prepare upcoming flights while the GPU handles ready work.
Do not implement or enable that overlap in this branch. Existing
full-CPU pricing workers are not a GPU preparation pool. Keep current price
snapshots immutable and tag prepared work so stale results cannot reach a new
round. Preserve restrictions, cancellation, deadlines and explicit worker-loss
handling. Bound outstanding work and memory; measure transport/queue overhead
and competition with CPU verification before increasing worker counts.

Test serial/parallel equivalence across signed/new prices and restrictions,
snapshot changes, cancellation, worker failure and backpressure. Benchmark one
worker and several worker counts with equal completion requirements, first use
and changed-price rounds. Reserve overlap comparisons for its separate branch. Do not infer
linear scaling or a speedup from CPU utilization alone.

## Handoff and coordination

Launch the three conversations from the common pinned PR144 base with this plan,
their scoped ownership, current continuity, source/capture
fingerprints and verified replay commands. Preserve separate commits and PRs.
Branches A and C must coordinate preparation interfaces; branch B should expose
its numeric preparation data through narrow interfaces that C can transport.
Freeze those contracts early, then perform a separate combined integration
check. PR144 remains draft throughout this work; any later landing requires new
user direction and review of the exact integrated head.
