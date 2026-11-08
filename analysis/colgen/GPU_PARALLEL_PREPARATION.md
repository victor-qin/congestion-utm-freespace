# Synchronous parallel CPU preparation on main

PR145 is rebased on main `ebc60608`, which now includes PR144 and PR146.
Its diff adds synchronous parallel CPU preparation; resident GPU execution,
shared prices, checked integer bounds and the tightened approach bound come
from main. Serial preparation remains the default. No CPU/GPU overlap is added.

The worker stage reuses main's GPUPreparation and fixed graph/root caches.
The owner can use either original FP64 pruning verification or checked GPU
integer pruning; both retain original canonical completion validation.

## Stage and ownership contract

`gpu_prepared.prepare_flight` is the shared serial/worker implementation. It
prepares the complete topology and row schema, canonical shortest seed, current
legal roots, exact origin/first-visit unions, original FP64 geometry/positive
bounds, and initial endpoint membership/masks/trees. Departure screening uses the
same canonical incumbent and original FP64 ground bound. No full CPU pricing is
called. All unpriced memberships remain available for later new/sign-changing
prices.

Each bounded preparation group finishes before any of its GPU search batches
starts. Workers are idle during owner search; there is no prefetch or background
refill. Search batch size stays independent of preparation group size, preserving
batch-one label caps and solver result order. The GPU owner performs root-bound
dispatch, all search, canonical completion checks, deletion verification and lazy
future-table preparation/retry. No Metal/CUDA context or buffer is created by a
worker. The shared device kernel/interface is unchanged.

Worker registration sends an immutable `FlightGraph` once per cache lifetime.
Its existing serialization excludes mutable search caches. Later jobs carry a
graph version, current objective, restrictions, assignment dual and incumbent.
A graph replacement gets a new version; mutation of a logically immutable graph
is unsupported by the existing graph contract. Workers retain topology, fixed packing, geometry and complete unpriced membership
caches from main; weighted data is refreshed for each invocation. Objective and root-domain
changes use the existing cache keys; no weighted payload survives into a new job.

`DualView` is the immutable price snapshot. Raw mappings are copied at public
iteration entry (or during a direct `price_batch` call), before any job starts.
Workers hold one snapshot, updated from original normalized
rows when its version changes. The owner encodes one complete row mapping and
reuses those bytes across workers. Only the latest encoded snapshot is retained,
bounded by the frame limit; workers still reconstruct their own indexed view.
This preserves row order, signed values and all currently zero-priced rows.
Each result tag contains invocation, position,
graph version and snapshot version. Mismatched replies fail before consumption.
The GPU boundary additionally checks the original problem, snapshot and options.

Replies contain fixed topology/local adjacency only when the owner's cache lacks
it, `PreparedVariants`, explicit `CompletionBoundData` with complete root claim
unions and FP64 coefficients, selected endpoint membership/masks/trees, and an
incumbent. They contain no graph, global price mapping, GPU owner or callback.
The owner reattaches the original problem and view, revalidates the incumbent
canonically, and applies batch offsets. Selected endpoint membership is complete
for every selected row; unselected rows remain in the worker cache for future
price snapshots. Both paid-mask words travel unchanged.

## Resource limits and cancellation

The options, also exposed with `pricing_gpu_` prefixes in `ColGenParams`, are:

| Platform option | Default | Scope |
| --- | ---: | --- |
| `prepare_workers` | 0 | Spawned persistent CPU processes; zero preserves serial execution |
| `prepare_batch_size` | 8 | Maximum flights in one synchronous preparation group |
| `prepare_max_bytes` | 128 MiB | Maximum serialized request or reply frame |
| `prepare_cache_bytes` | 128 MiB | Accounted graph/cache bytes per worker and fixed-data bytes in owner |

One request/reply can be outstanding per worker. Nonblocking framed owner I/O
checks deadlines during partial transfers and watches process exit. Graph cache
admission is also bounded to 128 graphs per worker. Graph identity stays assigned
to the same worker until its registration is released. Byte accounting counts
Python containers and NumPy backing storage. Sizes of unchanged immutable
components and append-only graph caches are reused; a shallow strong-reference snapshot
detects replacements in the canonical endpoint LRU even at unchanged length.
The small accounting registry is itself charged and released on eviction. Shared
objects across independently measured components/graphs can be counted more than
once, so the byte policy remains conservative. These
are retained-cache/transport limits, not an RSS cap: interpreter/native state,
price indexing, current preparation and decoding allocations are additional.
Peak process RSS and cache/transport costs are recorded separately.

A deadline, consumer cancellation or exception closes sockets and reaps workers.
A later invocation can start fresh workers. Worker loss and payload exhaustion
produce explicit incomplete platform results without a certificate; invalid or
stale protocol data raises an error. There is no original CPU pricing fallback.
Completed sweeps keep workers; the solver releases them before the final IP.
Completed GPU batches detach endpoint callbacks so the remaining prepared group
cannot retain their predecessor/device storage. Finished groups are released
before the next group is prepared.
Sequential repair uses the existing serial preparation path.

For a manual platform, consume `price_batches(problems, batch_size=1)` and close
its iterator when stopping early. Call `platform.close()` in a `finally` block.
`price_batch` keeps its original single-GPU-batch meaning; with workers enabled,
that batch must fit the configured preparation group.


## Validation and measurements

Measured on Apple M1 Ultra with licensed Gurobi (one thread), production source
`82e22724` above main `ebc60608`. All sustained runs held the shared benchmark lock;
lock waiting is excluded. Input/source hashes, parameters, round timings and
profile summaries are in [gpu_parallel_rebased_validation.json](gpu_parallel_rebased_validation.json).

### Hard88 production wall time

Two fresh-process runs per backend, original ordered hard88 inputs, fresh subset
LP prices, the same legal route domain and solver limits. GPU uses twelve
preparation workers, group twelve, 178956970 accounted cache bytes per worker,
GPU batch one, cap 250000, chunk 2048, and checked integer plus approach bounds.
CPU uses twelve native pricing workers. The CPU implementation is current main's
descendant of PR133, not an exact PR133 checkout. Production canonical validation
and final independent feasibility checking remain enabled. GPU full-CPU pricing,
CPU per-prune numeric verification and worker GPU initialization are forbidden by
runtime guards. No CPU/GPU overlap occurs.

| Measurement | GPU + 12 preparation workers | Native CPU, 12 pricing workers |
| --- | ---: | ---: |
| Total wall, repeats (s) | 34.377 / 34.292 | 14.641 / 14.566 |
| Mean total wall (s) | **34.335** | **14.604** |
| Mean pricing wall (s) | **30.292** | **9.581** |
| Pricing rounds | 4 | 6 |
| Completed/certified flight results | 352/352 | 528/528 |
| Accepted and independently verified flights | 88 | 88 |
| Final cost | 8708.244489 | 8716.244489 |
| Cost lower bound | 8696.551502 | 8700.244489 |
| Final LP gap | 0.042447% | 0% |
| Remaining global integer gap | 0.134275% | 0.183565% |

CPU remains faster on this cohort. Both stop at the configured LP-gap criterion,
with no label cap or incomplete pricing. They generate different columns and
follow different price trajectories, producing different final integer costs.
This is a same-input/backend comparison, not an equal-final-result speedup claim;
neither run proves global integer optimality.

Mean pricing time per round across the two runs:

| Round | GPU (s) | CPU (s) |
| --- | ---: | ---: |
| 1 | 10.785 | 3.326 |
| 2 | 10.763 | 2.297 |
| 3 | 4.749 | 1.127 |
| 4 | 3.994 | 0.982 |
| 5 | stopped | 0.856 |
| 6 | stopped | 0.994 |

These are full solver sweep timers, including synchronous preparation, transport
and search. After the initial round, corresponding round numbers do not imply
identical dual prices or incumbents. Cold work is included in round one; later
rounds use warm caches with changed prices, not same-price warm replays.

### Preparation benefit and measurement overhead

The separately audited harness, which performs extra per-batch canonical audits
and rewrites a growing pricing JSON file, measured 67.183 s with serial preparation
versus 43.120 / 43.340 s with twelve preparation workers: 35.7% less elapsed time
using the repeated parallel mean against one serial control. All three have the
same cost, bound, four rounds, 352 certificates, 1,340,700 candidates and 10,776
dispatches. This supports a benefit from parallel preparation within the GPU path,
not a speed advantage over native CPU pricing. The serial control was not repeated.

Do not mix those audited timings with production timings. Profiling exposed about
nine seconds of extra unprofiled harness overhead; it repeatedly serializes all
accumulated results after every flight. `parallel_profile --direct` avoids that
extra observation while retaining production checks and fallback guards. Its small
per-round callback is present in both CPU and GPU runs. The audited and direct GPU
runs preserve identical final cost/bound, rounds and dispatch counts.

### Million-label batch experiment

The resident search budget is now 256 MiB, allowing the tested integer-mode
searches with a 1048576-label cap. This budget is separate from worker caches,
host mirrors and numeric-table limits; it is not a total-process memory limit.
The label cap remains shared across the GPU batch. The default label cap is unchanged.

User-run, single-trial production hard88 experiments with twelve preparation
workers measured batch1/2/4 at 34.329 / 34.095 / 34.429 s. All completed four rounds,
352 certified results and 88 verified flights at cost8708.244489, with no resource
failure. Dispatches fell10776/9735/8751, but wall time did not materially improve.
Evidence: [gpu_million_label_batches.json](gpu_million_label_batches.json).
Use `parallel_profile --direct --batch-size 1 --max-labels 1048576`, replacing
batch size with2 or4 for the other configurations. The harness preserves the
same twelve-flight preparation groups and shared-lock wrapper described below.

### Profile and remaining CPU work

Owner-only cProfile increases production-path wall from about 34.3 s to 79.059 s
(81.491 s including outer input/manifest work). These instrumented times identify
hot functions; they are not normal-runtime measurements. Worker processes are not
cProfile-instrumented. Selected **nested, nonadditive** cumulative times:

| Owner function | Instrumented cumulative time (s) |
| --- | ---: |
| `gpu_integer_search.py:323 IntegerBatch.prepare` | 25.272 |
| `gpu_integer_search.py:89 root_operands` (inside preparation) | 23.964 |
| `gpu_integer.py:114 encode` (inside preparation) | 14.004 |
| `gpu_pricing.py:950 search` | 25.423 |
| `gpu_pricing.py:810 complete` (inside search) | 10.580 |
| `gpu_prepare_pool.py:516 prepare` (includes waiting for workers) | 10.801 |
| `gpu_runtime.py:458 run` | 6.558 |

The unprofiled audited repeat supplies lower-overhead stage telemetry:

- Synchronous preparation groups: 9.139 s wall total; by round
  5.997 / 1.903 / 0.666 / 0.574 s. Startup 0.752 s, owner serialization 0.138 s,
  restoration 1.613 s, and price encoding 0.006 s are included/overlapping.
- Integer-table preparation in the GPU owner: 8.018 s; approach preparation
  0.811 s; bootstrap search 10.337 s. These overlap other search/runtime timers.
- Measured GPU execution 0.877 s and owner waits 4.094 s over 10,776 dispatches.
  GPU time is device-reported command execution, not total search time or a
  measured utilization percentage; GPU idle time is not directly instrumented.
- Zero CPU per-prune numeric checks; 754 canonical candidate validations, zero
  rejections. Canonical checks execute in the owner, not in the preparation pool.
  There is no queue waiting for a preparation worker to approve GPU deletions.
- 88 graph registrations across all four rounds, four encoded price snapshots,
  zero cache evictions; 10.63 MB requests and 79.89 MB replies. Stable assignments
  and retained caches avoid re-registering flights at each price change.
- Worker preparation sums to 38.090 CPU-stage seconds and cache accounting to
  12.048 seconds across workers; these are not wall time. Reported per-worker RSS
  highwaters range 265–336 MB in this repeat. Their 3.57 GB sum is not a simultaneous
  memory measurement. The roughly 2 GiB aggregate accounted-cache budget is not an
  RSS limit. Earlier audited repeat peaked at 248 MB for its largest worker.
- Device allocation counters: 18,481 allocations / 824.42 MB cumulative allocated;
  1117.59 MB uploaded and 205.80 MB downloaded. These are traffic/cumulative
  allocation counts, not peak retained device memory.

Owner integer operand generation/encoding and canonical candidate construction
remain prominent CPU costs after moving independent flight preparation to workers.
The profile does not establish that more preparation workers or CPU/GPU overlap
would remove those costs. Overlap remains a separate follow-up, not this PR.

### Relation to PR133 CPU numbers

The historical optimized results associated with PR133 in
[results.json](results.json) used **1537 / 1545 flights**, twelve CPU workers and
four Gurobi threads: 472.031 / 644.577 s pricing over 11 / 14 rounds, or
**42.912 / 46.041 s per round on average**. Total walls were 1065.784 / 1345.489 s.
The PR explicitly labels these as measurements before final integration; they
are not fresh post-merge timings. Comparing their 43–46 s rounds with 88-flight
GPU rounds would falsely imply a GPU speedup. The fresh 88-flight native CPU
comparison above is the relevant scale-matched evidence after PR133.

### Functional checks and earlier removal of PR144

379 distinct focused tests pass across preparation pool, pricing platform,
endpoint, options, validation, solver and integer paths, including actual Metal
and licensed Gurobi. Two analysis ablation tests initially patched the module
from before CPU extraction; fixing their target and rechecking resolved both.
Changed Python passes Ruff; whitespace checks pass. NVIDIA execution is untested.

The earlier port onto main `0cbd284a`, before PR144 landed, preserved the hard88
cost and all 352 pricing certificates/four rounds. It did not establish equal
runtime. The current rebase inherits PR144 and PR146 because both are now in main;
their changes are not part of PR145's diff.

## Tightened bound and reproduction

The optional `--approach-bounds` uses the proof in [GPU_APPROACH_BOUND.md](GPU_APPROACH_BOUND.md).
It runs in the owner after an incumbent search, with a bounded work budget and
no certificate from partial enumeration. Prepared payloads remain CPU-only.

Run a fresh canonical hard88 solve from this worktree with twelve workers and
the tightened bound (the command waits for the shared measurement lock):

```bash
cd /Users/victor/.t3/worktrees/congestion-utm-freespace/codex-parallel-cpu-preparation
export PYTHONPATH="$PWD"
export COLGEN_GUROBI_THREADS=1
/Users/victor/Documents/Research/congestion-utm-freespace/.venv/bin/python - <<'PY'
import fcntl, subprocess, sys
from datetime import datetime
from pathlib import Path

root = Path('/Users/victor/Documents/Research/congestion-utm-freespace')
out = '.agent/runs/parallel-approach-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
cmd = [sys.executable, '-m', 'analysis.colgen.parallel_profile',
       '--capture', str(root / '.agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl'),
       '--direct', '--out', out]
# Add '--profile' for owner cProfile; add '--device', 'cpu' for native CPU comparison.
with (root / '.agent/gpu-followups-20261106/metal-benchmark.lock').open('a') as lock:
    print('Waiting for benchmark lock; results:', out, flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    subprocess.run(cmd, check=True)
PY
```

This uses new LP prices in a full solver run, original inputs, canonical
validation and the full CPU-pricer guard. It preserves a 900-second solver
budget and a 250000-label pricing cap. The output directory contains the source
and input manifest, per-round times, final verification and planner statistics.
Omit `--direct` for the extra audited GPU run, which also writes individual pricing
records and the final schedule; those audited wall times include substantial
measurement overhead.
The configured LP-gap stop does not prove exact global integer optimality.

CPU/GPU overlap remains a separate future branch: any future pipeline must keep
snapshot identity, bounded ready groups, one device owner and cancellation guarantees.
