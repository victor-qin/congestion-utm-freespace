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

Rebased validation and fresh hard88 measurements are recorded below. Previous
measurements on the earlier main revision are superseded.

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
import fcntl, json, subprocess, sys
from datetime import datetime
from pathlib import Path

root = Path('/Users/victor/Documents/Research/congestion-utm-freespace')
ids = [r['flight_id'] for r in json.loads(Path(
    'analysis/colgen/arrival_union_results.json').read_text())['hard88']['after']['rows']]
out = '.agent/runs/parallel-approach-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
cmd = [sys.executable, '-m', 'analysis.colgen.gpu_validation', 'solve',
       '--capture', str(root / '.agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl'),
       '--flight-ids', *map(str, ids), '--approach-bounds', '--integer-bounds',
       '--prepare-workers', '12', '--prepare-batch-size', '12',
       '--prepare-cache-bytes', '178956970', '--batch-size', '1',
       '--max-labels', '250000', '--chunk-size', '2048', '--budget', '900', '--out', out]
with (root / '.agent/gpu-followups-20261106/metal-benchmark.lock').open('a') as lock:
    print('Waiting for benchmark lock; results:', out, flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    subprocess.run(cmd, check=True)
PY
```

This uses new LP prices in a full solver run, original inputs, canonical
validation and the full CPU-pricer guard. It preserves a 900-second solver
budget and a 250000-label pricing cap. The output directory contains the source
and input manifest, pricing records, final schedule and planner statistics.
The configured LP-gap stop does not prove exact global integer optimality.

CPU/GPU overlap remains a separate future branch: any future pipeline must keep
snapshot identity, bounded ready groups, one device owner and cancellation guarantees.
