# PR146 on main, without PR144

This is an isolated experiment, not a retargeted PR. PR146 and PR144 stay
unchanged, draft and unmerged. The requested integer-cache conversation is
deferred until the experiment informs the choice of base.

The runner creates ordinary source exports beneath this worktree's `.agent`.
It does not create or switch Git checkouts, rewrite branches, or change PRs.

## What is compared

| Arm | Production source | Enabled changes |
| --- | --- | --- |
| `main` | Main at `0cbd284a4d46b42416b5ab1ef09ebf7f903ac13c` (PR143) | Original CPU-verified GPU pricing |
| `main-pr146` | That main plus the explicit numerical port | Integer GPU pruning and the reachable approach/arrival union bound |
| `stacked-pr146` | PR146 at `78f1d4d2504bb8b67a535865049012bd16583918` | Same two improvements, including inherited PR144 execution changes |

The port keeps main's host-managed frontier, per-layer GPU dispatches,
preparation, price lookup and batch-local array cache. It has no PR144 resident
search, owned-buffer interface, shared device-price cache or preparation caches.
The checked integer arithmetic and bound mathematics come from pinned PR146.
Integer arrays use main's existing immutable-array interface and cache, which
is cleared after each batch. Cutoffs and certificate accounting remain fresh.
The kernel signature is adapted to main's 16 ordinary arguments plus eight
integer arguments. CUDA source remains present; no NVIDIA hardware was tested.

`main_pruning_port.patch` contains the reviewable adaptation. The build rejects
unexpected production modifications: only `gpu_runtime.py`, `gpu_pricing.py`,
`pricing_gpu.h`, `params.py`, `pricing_platform.py` and `pricing_pool.py` may
change relative to main, besides the eight explicitly listed PR146-only files.
Every other main production file stays byte-identical. Each build records
per-file hashes in `build.json` and writes `production-port.diff`.

## Run the hard88 warm comparison

Run from this bound worktree, using the existing environment:

```bash
cd /Users/victor/.t3/worktrees/congestion-utm-freespace/codex-gpu-certified-pruning
/Users/victor/Documents/Research/congestion-utm-freespace/.venv/bin/python \
  analysis/colgen/main_pruning_experiment.py \
  --mode replay --cohort hard88 --repeats 2 --warm-repeats 2 \
  --out ".agent/main-pr146-hard88-$(date +%Y%m%d-%H%M%S)"
```

This runs all three arms. Each process performs one first-call phase and two
same-price warm calls. The order reverses for the second process repeat. The
runner acquires the shared CPU/Metal advisory lock separately for every child
and releases it before the next job. Set aside time for six complete processes;
the runner prints the active log path while each child runs.

The exact ordered hard88 IDs come from the saved PR143 selection; they match
the earlier repeat-6 comparison. Inputs are the original round-1 full-scenario
capture, with original prices and captured seeds, no oracle routes. Limits are
250,000 labels, 2,048 labels per dispatch chunk and one flight per batch.
The 900-second deadline is per pricing batch for replay, not a target duration.

`comparison.json` contains warm medians and per-phase completion/certificate
checks. It reports a relative warm reduction only when all compared phases
complete with certificates, gaps within the existing tolerance, matching
ordered flight IDs and matching reduced costs within that tolerance. Cold
phases are excluded from warm medians. Inspect `completion.json` for child
failures and the individual raw JSON/logs for unresolved cases.

Raw results include source/capture hashes, canonical validation, CPU numerical
checks, preparation/search components, labels/candidates, allocation/copy/wait
estimates and process peak RSS. Timers overlap. Peak RSS is the process lifetime
high-water mark, not current live GPU memory. Metal copy counts describe explicit
shared-buffer copies, not PCIe traffic. First-call timing can benefit from disk
JIT caches left by earlier process repeats; it is not a cache-flushed system
measurement. Warm calls retain the same graphs, snapshots and runtime within
one process, but integer payloads still rebuild for each invocation.

## Other comparisons

- To isolate integer pruning without the newer approach bound, add
  `--variants main main-integer stacked-reference stacked-integer`.
- To check changed prices, use `--cohort late18`. This runs the original early
  snapshot, its warm repeats, changed late prices, a late warm call, and restored
  early prices, using the saved 18-flight selection.
- To compare fresh hard88 subset solver runs, use `--mode solve --cohort hard88
  --repeats 2`. These generate their own evolving subset LP prices. They are
  **not** the fixed-price warm-cache replay. A capped solve exits nonzero and
  preserves its report; it must not be described as an equal-completion speedup.
- `--smoke` selects only flights 940 and 2224 from the original early capture.
  That does not recreate flight 2224's difficult subset-solver round-2 prices.
- `--audit-integer --variants main-pr146` enables independent FP64 shadow
  checks. Audit timings are not production performance measurements.
- `--mode test` builds the same sources and runs the focused numerical,
  endpoint and platform tests against the port under the shared lock.

Every invocation requires a fresh output directory. For example, the solver
command is:

```bash
/Users/victor/Documents/Research/congestion-utm-freespace/.venv/bin/python \
  analysis/colgen/main_pruning_experiment.py \
  --mode solve --cohort hard88 --repeats 2 \
  --out ".agent/main-pr146-solver-$(date +%Y%m%d-%H%M%S)"
```

## Validation before handoff

The main-based port passed 149 focused cases covering arithmetic, overflow,
caps, changed snapshots, exact endpoint membership, every-prefix completion
coverage and actual Metal pricing. Two-flight cold/warm replay and fresh subset
solver smoke runs completed and certified on all three arms. Original full CPU
pricing was guarded in every measurement; production integer runs also forbade
the per-prune CPU numerical verifier. CPU canonical completion validation stayed
enabled. These are setup checks, not full-hard88 performance evidence.

The same two-flight port replay also passed 66,963 independent FP64 deletion
audits in each cold/warm phase. The reporting tests passed 45 cases, including
rejection of capped, mismatched-price, mismatched-flight and oversized-gap
comparisons. Local smoke artifacts are `.agent/main-port-smoke`,
`.agent/main-port-solver-smoke` and `.agent/main-port-audit-smoke`.

Earlier PR146 wall times all included PR144. This experiment provides new
measurements; it does not retroactively attribute those times to main alone.

## Completed hard88 experiment and decision

The two-process-repeat run completed all 18 phases with 88/88 certificates,
identical reduced costs and returned columns. Warm medians over four calls were
39.743073 s for main, 30.355693 s for main plus PR146, and 25.939761 s for the
PR144+PR146 stack. First-call medians were 78.275586, 68.889249 and 65.109863 s.
The standalone port saves 23.62% versus main; the stack saves a further 4.415932 s
(14.55% versus the port). These are fixed-price replay measurements.

The raw bundle is `.agent/main-pr146-hard88-20261107-215305`. On 2026-11-08 the
user chose to keep PR146 stacked on PR144 and prepare for merge, then launch
integer caching from PR146. This supersedes the earlier deferred-cache decision.
