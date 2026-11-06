# CPU–GPU exchanges during one pricing search

Status: 2026-11-06. This describes the current implementation and proposed
follow-up optimizations. It does not claim those optimizations are implemented.

## Current execution

The GPU is an expansion engine inside a CPU-driven time-layer loop. It does
not yet own the whole pricing search. A label is a partial route together with
the state needed to continue it. A frontier is the current collection of labels.

The sequence below is written for one flight. The implementation can pack several
flights into the same dispatch, using their absolute times, but the same exchanges
remain at each occupied time layer and each chunk of up to 2048 labels.

```text
CPU: prepare topology, departure roots, prices and a validated starting route
GPU: compute initial root-bound proposals
CPU: wait; copy root bounds back

run bounded GPU search to find a better validated starting route
run complete GPU search from every legal root

SEARCH:
    for time in the flight's allowed time horizon:
        CPU: initialize roots starting now
        CPU: assemble visit prices, paid-row masks and blocked-cell flags
             lazily build budgeted Q/H trees for surviving endpoint masks

        for chunk of current labels:
            CPU: copy labels, scores, root values and time-specific inputs
                 into GPU buffers; reuse cached graph/root/paid-row buffers
            GPU: expand each label toward up to six neighbours
                 apply integer feasibility filters
                 compute FP32 scores and proposed pruning bounds
            CPU: wait until the kernel finishes
            CPU: copy candidate records and scores into NumPy arrays

            CPU: inspect valid candidates
                 recompute every proposed numerical deletion in FP64
                 reconstruct and canonically validate completed routes
                 retain continuing candidates and enforce the state cap

        if heuristic search:
            GPU: group candidates with the heuristic state comparison
            CPU: wait; copy grouping table; select the bounded frontier

        CPU: assign predecessor IDs and update verified prefix-price sums
        CPU: next frontier = retained candidates
        # That frontier is uploaded again at the next time step.

if the first full search reaches its state cap:
    construct a stronger backward future-cost table
    rerun the bounded search, then restart the complete search

return validated route and certified upper bound only if search completes
```

The integer feasibility checks already run on the GPU. CPU numerical checking
does not rerun the original constrained CPU pricing DP. CPU trajectory validation
still checks reconstructed geometry, claims and objective values. The current
certificate mode does not merge continuing histories using an unproved score
comparison.

### What each runtime call does

`ComputeRuntime.run` currently combines buffer preparation, submission, waiting
and output copying into one synchronous operation. Metal uses shared buffers,
but the adapter explicitly copies NumPy input bytes into them and copies outputs
back to separate NumPy arrays. Shared physical memory on the M1 therefore does
not make the current implementation free of copying or synchronization.

Static graph, root-index, and paid-row buffers are reused within a batch. Dynamic
Metal allocations are also reused, and write-only outputs skip their initial
upload. However, every call still waits, and outputs are returned as independent
copies. The batch finally clears those GPU caches. CUDA follows the same
synchronous interface, with host-to-device inputs, stream synchronization and
device-to-host outputs.

In the expansion call, `root_values` is uploaded repeatedly even though only its
cutoff column changes frequently. The same time-layer visit and blocked-cell
arrays are also uploaded again for every chunk in that layer.

### The stronger future table has distinct numerical modes

```text
CPU: prepare relaxed graph, positive visit prices and mandatory endpoint costs

for time in reverse:
    if certified:
        CPU: compute the regular relaxed row directly in FP64
    else:
        GPU: compute the regular future-cost row from the next row
        CPU: wait and copy the FP32 row back

    if departures start at this time:
        CPU or GPU: compute first-hop bounds in the same numerical mode

    check the shared deadline before publishing further rows
```

Certified mode uses exactly the original FP64 `_relax` dataflow and omits the
unused FP32 recurrence. Unchecked mode still uses the GPU recurrence and never
certifies a result. Later constrained searches and root-bound proposals remain
on the GPU in both modes. No original CPU constrained pricing is invoked.

## A measured example

The saved post-reachability-change replay of flight 322 used 2200 GPU dispatches.
Each was a synchronous runtime call. Of these, 1916 belonged to future-bound
preparation including refreshed root bounds, 211 to the heuristic searches, and
73 to initial root bounds and the remaining full search. The future stage thus
accounted for about 87% of its dispatches, not necessarily 87% of runtime.

Its 394144 FP64 pruning checks were arithmetic on returned CPU data; they were
not 394144 additional GPU round trips. It completed with a certified upper bound.
Flight 0 in the same report needed 248 dispatches without the stronger future
table; flight 32 needed 2686 and still hit the state cap. These are different
search workloads, not a controlled timing comparison.

Evidence: `.agent/gpu-endpoint-bound-20261106/replay-selected/pricing.json`.
No new benchmark was run for this explanation.

## Reducing exchanges in stages

### 1. Completed: remove the unused GPU future recurrence in certified mode

When CPU recomputation is enabled, construct the verified relaxed table directly
with the existing FP64 recurrence and omit its duplicate FP32 GPU calculations.
The CPU already does this arithmetic and provides all stored values today. The
constrained route expansion, heuristic search and root-bound proposals remain
on the GPU; this is not a fallback to the original CPU pricing search.

For the saved flight-322 execution, future preparation's 1916 dispatches comprise
1915 future-table kernels and one refreshed root-bound kernel. Removing the
unused recurrence would predict 285 total dispatches instead of 2200 if the
rest of the search work stays unchanged. This is a source-based count estimate,
not a measured runtime speedup. FP32-only mode still needs its GPU recurrence.

Stored arrays and root costs were checked bit-for-bit against the original
source, including FP32 overflow/underflow inputs. Functional tests require no
GPU recurrence in certified mode, retained device computation in FP32 mode,
and deadline propagation from either recurrence.

### Also completed: endpoint paid flags and arrival bounds

Each retained record now has two independent paid-row words before its recent
cell history. The arrival-only Q/H trees are shared by all roots of a flight
and built lazily within configurable byte/work budgets. GPU queries produce
prune proposals; FP64 verification uses original prices and independently
maintained exact masks. Layer masks are uploaded once and immutable tree buffers
are replaced only after cache additions. Geometry alone persists across LP
snapshots. See `GPU_DEVELOPMENT.md` for the accounting proof and API controls.

### 2. Avoid repeatedly copying unchanged inputs

Separate the changing per-flight cutoff from the mostly fixed per-root
coefficient matrix. The kernel can use the root's existing flight index to read
a small cutoff array. Cache the visit and blocked-cell arrays for an entire
time layer rather than copying them for every chunk. Use explicit update/lifetime
rules: marking a mutable array as static by identity would leave stale prices
or cutoffs on the GPU.

These changes reduce bytes copied without changing numerical decisions. They
are easier to review than moving the search state machine at the same time.
Measure bytes and time separately; small dispatches may remain launch-bound.

### 3. Batch the retained FP32 future recurrence

Split the runtime into buffer ownership, command submission, and waiting. Keep
the previous and next FP32 rows in GPU buffers, alternating their roles. Encode
several dependent time-layer kernels in order, with appropriate resource
dependencies, and wait once per bounded block. Prepare first-hop root-role
outputs in the same stream without copying the regular row through NumPy.

This applies to FP32-only mode, where the recurrence is still needed after the
first cleanup, or a later independently proved GPU numerical implementation.
Check device errors, nonfinite values, deadlines and incomplete-block status
before publication. The bounded blocks allow deadline responsiveness; one huge
uninterruptible command is not the goal. This retains the mathematical
relaxation. FP32-only mode remains explicitly uncertified.

### 4. Keep frontiers and predecessors on the GPU

This is the larger redesign. Let the GPU compact valid continuing labels and
assign stable predecessor IDs; leave those arrays resident for the next layer.
Read back compact lists of proposed numerical deletions and completed-route
candidates rather than every candidate record, including unused slots.

For the first certified version, keep a layer barrier while the CPU verifies
those lists. Return keep/drop decisions, validated incumbent updates, and any
needed verified prefix values. This reduces copying and Python iteration while
retaining the existing verification requirement. Stable IDs, prefix accounting,
queue overflow, and batch/flight ownership all need explicit tests.

### 5. Run several search layers between CPU checkpoints

Only after resident state works, let the GPU continue ordinary branches across
several layers. Numerical deletion proposals must remain unresolved work until
CPU verification; a failed proposal must be resumable at its original time.
Completions also remain pending until canonical validation. A pending queue
must never silently drop alternatives, and the search cannot certify while any
pending branch or completion can exceed its reported bound.

```text
CPU: prepare input and verified initial bound
GPU: retain topology, prices, roots, frontier and predecessors

while live states or pending decisions remain:
    GPU: run a bounded block of time layers
         keep ordinary states on the device
         queue unresolved numeric decisions and completed-route candidates
         stop before any storage/queue capacity is exceeded

    CPU: wait once for the block
    CPU: verify queued decisions and validate queued completions
         account for every rejected candidate and unresolved alternative
    GPU: apply decisions, resume retained branches, update validated incumbent

CPU: emit a certificate only after all remaining alternatives are accounted for
```

A stale validated incumbent is safe: it weakens pruning and can create extra
states. An unvalidated GPU completion is not a safe pruning threshold. Deferred
decisions can therefore trade fewer waits for more expansion and memory. The
block size must be measured, not selected solely to minimize dispatch count.

To reduce even the decision-list traffic on M1, a later kernel could carry proved
FP32 error intervals and discard only when its entire upper-bound interval is
below a validated threshold. Ambiguous cases still go to the CPU. Ordinary
FP32 comparisons alone are not such a proof; disabling verification currently
selects the explicitly uncertified mode.

## Functional checks and code map

Tests should preserve canonical route feasibility and upper-bound coverage,
not exact route identity among ties. Cover changed LP snapshots, layer/chunk
boundaries, stale cutoffs, shared-buffer ownership, rejected completions,
resume-after-rejected-prune, queue overflow, deadlines and device errors. Compare
full pricing time and peak states as well as dispatches, waits and bytes copied.
Parallel CPU preparation is independent: it can overlap preparation with GPU
work, but it does not eliminate these per-layer exchanges.

- `gpu_pricing.py:166`: preparation and root packing.
- `gpu_pricing.py:318`: initial/refreshed root-bound dispatch.
- `gpu_pricing.py:328`: stronger bound preparation.
- `gpu_pricing.py:419`: FP64 numerical pruning verification.
- `gpu_pricing.py:473`: CPU predecessor and verified-prefix storage.
- `gpu_pricing.py:645`: heuristic/unchecked grouping round trip.
- `gpu_pricing.py:678`: time-layer/chunk loop.
- `gpu_pricing.py:799`: bounded incumbent search.
- `gpu_pricing.py:825`: full search and capped-search retry.
- `gpu_runtime.py:197`: synchronous dispatch interface.
- `gpu_runtime.py:253`: explicit shared-buffer input copy.
- `gpu_runtime.py:275`: Metal wait; outputs copied at 278.
- `gpu_future.py:138`: per-layer future-table construction.
