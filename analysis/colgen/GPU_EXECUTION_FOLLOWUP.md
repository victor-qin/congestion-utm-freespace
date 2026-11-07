# GPU execution and data reuse follow-up

Subsequent endpoint integration and the 88-flight comparison are recorded in
[GPU_HARD88_COMPARISON.md](GPU_HARD88_COMPARISON.md). The measurements below
remain evidence for the original execution milestone.

Measured 2026-11-07 UTC on Apple M1 Ultra. Base: PR142 at `997dfd11`;
implementation measured: `863146c7`. This branch targets
`codex/gpu-pricing-platform`, above PR133. Endpoint-bound/heuristic changes and
the independent persistent CPU preparation pool are not included.

Stages 1–3 and the bounded resident-search milestone are implemented. The
fixed-capture results preserve certification, but do **not** establish an
overall speedup: warm first-round pricing is slightly slower, warm later-round
pricing slightly faster, and the selected hard/control set is slower. CUDA
source and adapter contracts were checked; no NVIDIA hardware was available.

## Reviewable milestones

| Commit | Scope |
| --- | --- |
| `9d0a40b6` | Fixed graph packing, stable physical resources, legal unpriced starts and complete origin/first-visit membership caches |
| `eab11fca` | Original FP64 price validation and outward negative/absolute-price sums shared per immutable snapshot |
| `4504ec4a` | Sparse device prices shared across batches, mixed-snapshot partitioning, bounded lifetime and transfer counters |
| `93322acc` | Preserve invocation-local root unions when persistent cache admission fills |
| `863146c7` | Resident state/predecessor/frontier storage, device compaction and bounded verification checkpoints |

## Ownership and numerical contracts

`gpu_prepare.py` reuses the existing graph topology cache. A packing owns local
adjacency and physical `(cell, level)` identities. Absolute unpriced start
descriptions retain at most 4096 records. Exact origin/first-visit sets retain
at most 16384 row references, including zero-priced and terminal rows. These
are bounded exact sets, not translated-time approximations. Oversized domains
rebuild uncached memberships; call-local verification still reuses its own
complete unions. Current root filtering, exclusions, objectives and incumbents
are applied after the fixed data is retrieved. Configuration/topology changes
replace the packing.

`DualView` remains the immutable price snapshot. Raw mappings are copied anew
on every public call, so new rows, sign changes and in-place caller mutations
are observed. Repricing uses the existing `math.fsum` claim/visit semantics and
exact overlap membership. Snapshot-wide summaries preserve the original
ordered FP64 `nextafter` arithmetic. Failed builds publish no partial entry.

`gpu_prices.py` shares one sparse resource/time device table per live snapshot.
Each flight maps its local vertices into this store, including resources
outside the nominal region. Missing resources use an empty range. Mixed
snapshots are partitioned without changing result order. Topology, exclusions,
future bounds and search state remain flight-specific. The runtime retains at
most two snapshots and 128 MiB of device price tables; an oversized live table
is used without retaining it. Batch cleanup releases batch buffers but keeps
these bounded stores. Explicit full cleanup releases cache ownership; active
owners keep their buffers alive. CPU sparse tables remain owned by their
`DualView`, as before.

`gpu_buffers.py` and `gpu_runtime.py` provide owned typed allocations, narrow
readback, explicit mutable writes, and ordered dispatch blocks. Foreign-runtime
handles are rejected. Metal uses separate dependent encoders in one command
buffer and waits once; CUDA uses one ordered stream and synchronizes once.
Mutable inputs are refreshed only after the preceding block has completed.

`gpu_resident.py` owns full device records and stable predecessor IDs for the
certified search. Expansion runs over flights, labels and six neighbor slots;
atomic compaction retains continuing histories without merging them. Full
records of numerical deletion proposals survive in a pending queue. CPU
checkpoints read compact metadata and recover only ancestors needed by the
original FP64 verifier or canonical route validation. A rejected deletion
returns to its original absolute time. A rejected completion does not remove
its legal continuation. Only validated incumbents set pruning cutoffs.

Blocks contain at most eight chunks, or two time layers when worst-case
growth fits all queues and no intervening root arrival is skipped. CPU work
between blocks may change a cutoff; retaining a stale weaker cutoff within a
block can only retain extra work. The implementation keeps every unresolved
alternative until verification or an explicit incomplete return.

Resident arenas, scratch and ready buffers have a 128 MiB byte budget. Decision
queues have at most 65536 records, reduced with the caller's label cap. This
budget excludes immutable graph/price buffers, small command uploads and the
CPU verification mirror. `buffer_limit`, `queue_limit`, `label_limit`, deadline
and unresolved-state exits never certify. The existing label-cap retry still
restarts all roots with the existing stronger future table. Unchecked FP32 and
the initial incumbent search keep their existing host-managed execution path;
unchecked FP32 remains explicitly uncertified. No original full CPU pricer is
called as fallback or certificate producer.

## Functional validation

The final focused run passed **292 tests**, with zero skips, failures or errors
in 38.830 seconds, under the shared Metal lock. It includes actual Metal and
licensed Gurobi/HiGHS small full-solver integrations. Local JUnit evidence:
`.agent/execution-focused.xml`. The focused files were `test_colgen_dp_prepare`,
`gpu_bounds`, `gpu_future`, `gpu_options`, `gpu_runtime`, `gpu_solver`,
`gpu_validation`, `pricing_platform`, `pricing_pool` and `performance`.

The captured ten-flight full solver also passed with licensed Gurobi and Metal:
all ten accepted and independently verified, one complete certified pricing
sweep, no CPU fallback, termination `lp_gap`. Its 10.502-second wall time is an
integration result, not a baseline speed comparison. The subset's fresh LP
prices closed every pricing root (five root-bound GPU dispatches, no expanded
labels); resident expansion is exercised by the fixed-price replays and tests.
Local evidence: `.agent/final-captured-solve/{result,planner_stats,pricing}.json`.

| Requirement | Coverage |
| --- | --- |
| Complete zero-priced/terminal memberships, bounded caches, interrupted builds | Parametrized preparation tests, exact canonical union comparisons and forced small-cache admission |
| Fresh signed/new prices, raw mapping mutation, clock/support changes | Preparation/platform snapshot cases, exact outward-sum and interrupted-publication cases |
| Root reordering/filtering, topology/objective changes, origin overlap | Existing preparation/bounds cases extended to GPU preparation; exhaustive prefix-union checks |
| Mixed snapshots, flight restrictions, resources outside region | Platform shared-price tests and sparse-table lifetime tests |
| Buffer ownership, eviction and partial construction failure | Runtime Metal/CUDA doubles and weak-reference lifetime assertions |
| Restored numerical proposals and rejected completions | Actual Metal, both one- and two-layer checkpoint configurations, exhaustive-route oracle |
| Queue overflow, memory budget, deadlines/caps and runtime reuse | Actual Metal fault injection plus partial-allocation lifetime tests |
| Extensible records | Actual Metal copy/load test preserves opaque mask words and invalidates padded inputs |
| Certificate/column meaning and CPU-pricer guard | Original FP64/canonical checks, exhaustive tiny routes, real full solver, captured replays |

Generated CUDA source passed a C++ syntax/interface check with CUDA annotations
and atomic intrinsics stubbed, and both runtime adapters passed ownership/block
tests. This is not a CUDA compiler or hardware execution result. Ruff and
`git diff --check` passed. The unrelated missing `analysis.profile_native_dp`
fixture in the base was avoided by running focused tests, not the full suite.

## Matched captures and timings

Compact source/capture hashes, counters, result status, reduced costs and bounds
are saved in [gpu_execution_measurements.json](gpu_execution_measurements.json).
The same graphs, original signed prices, known columns, 250000-label cap,
2048-label chunks and 180-second per-batch deadline were used on both versions.
No oracle incumbent was injected. Full CPU pricing entry points were patched
to fail. Every returned column was independently rebuilt and checked.

The ten-flight selection is `1420,160,872,2442,2084,850,274,1850,764,1752`, in
two-flight batches. Each version certified all ten in all five phases, including
returning to the first snapshot after the second. The eight-flight selection
is `0,2,32,34,276,278,320,322`, priced individually at round 1. Both versions
certified six and left flights 2 and 32 at `label_limit`, without a bound.
Reduced costs and completion statuses match across versions; certified bounds
cover the other version's incumbent. Equal route identity is not required.

| Workload/phase | PR142 elapsed s | Follow-up elapsed s | PR142 preparation s | Follow-up preparation s |
| --- | ---: | ---: | ---: | ---: |
| Ten, round 1 cold | 12.756 | 12.665 | 9.485 | 9.437 |
| Ten, round 1 warm | 4.873 | 4.965 | 1.649 | 1.788 |
| Ten, round 11 changed | 9.147 | 8.916 | 2.556 | 1.965 |
| Ten, round 11 warm | 7.866 | 7.554 | 2.094 | 1.614 |
| Ten, round 1 restored | 5.095 | 4.955 | 1.865 | 1.748 |
| Eight hard/controls, cold | 23.256 | 26.703 | — | — |
| Eight hard/controls, warm | 19.947 | 23.026 | 1.434 | 1.353 |

These are single ordered measurements, not repeated statistical estimates.
Cold means new process, graphs and snapshots, and includes existing preparation
and JIT cold costs. Elapsed includes observer overhead and independent result
checking (about 0.050 s for round 1 ten, 0.012 s for round 11 ten and 0.027 s for
the eight). It excludes graph construction (~0.015 s for ten), snapshot indexing
(0.896/0.913 s across both snapshots) and runtime initialization (0.094/0.153 s),
all recorded separately. `prepare_s` is inclusive; other method timings are not
a disjoint wall-time decomposition.

| Warm workload | Metric | PR142 | Follow-up |
| --- | --- | ---: | ---: |
| Ten, round 1 | Uploaded / downloaded MB | 1629.238 / 30.734 | 171.647 / 30.734 |
| Ten, round 1 | Allocations / allocated MB | 262 / 21.742 | 362 / 119.527 |
| Ten, round 1 | Waits / host wait s / GPU s | 2249 / 0.687 / 0.062 | 2249 / 0.687 / 0.065 |
| Ten, round 11 | Uploaded / downloaded MB | 834.302 / 110.092 | 336.863 / 86.580 |
| Ten, round 11 | Allocations / allocated MB | 215 / 17.295 | 551 / 142.317 |
| Ten, round 11 | Waits / host wait s / GPU s | 4630 / 1.256 / 0.069 | 4592 / 1.234 / 0.074 |
| Eight hard/controls | Uploaded / downloaded MB | 1257.923 / 307.878 | 262.104 / 140.912 |
| Eight hard/controls | Allocations / allocated MB | 453 / 20.942 | 2201 / 249.876 |
| Eight hard/controls | Waits / host wait s / GPU s | 10358 / 2.876 / 0.165 | 10093 / 2.684 / 0.172 |

MB is decimal bytes. Metal transfer counts mean explicit host/shared-buffer
copies, not PCIe traffic. Allocated bytes are cumulative, not peak residency.
Legacy transfer observation covers its host-array interface; current runtime
counters additionally include owned-buffer operations. Host wait times use the
same outer command observer for both versions.

Price tables upload once on each new snapshot and zero times on warm/restored
calls. Warm visit preparation/gather falls from 0.021 to 0.004 s in round 1 and
0.090 to 0.004 s in round 11. That reuse does not remove all preparation work.

The first-round ten-flight certificate pass closes entirely at roots after
bootstrap, so it cannot demonstrate a resident-expansion speedup. In warm
round 11, the follow-up performs 32 resident checkpoints; total kernel
dispatches increase from 4630 to 4789 because load/compaction are now kernels.
There are fewer waits, not fewer kernels. In the warm eight-flight set, 205
resident checkpoints replace larger host frontier transfers. None of these
captured blocks meets the conservative two-layer bound; the two-layer path is
covered by actual Metal functional tests.

CPU verification dominates the difficult cases. On the warm eight-flight set,
numeric deletion verification is 7.970/8.002 s and canonical completion work
1.350/1.350 s. Host-managed prefix allocation/accounting is 1.597 s in PR142;
the follow-up spends 0.274 s there plus 4.616 s lazily recovering 682134
predecessors. Extra device allocations and these narrow predecessor reads are
measured costs. Resident compaction can fill the cap where the old whole-layer
allocator stopped earlier, and delayed validated cutoffs can retain extra
states. Label counts therefore need the status and search-stage context saved
in the artifact. No speedup is inferred from capped searches.

The initial stage-3 profile also exposed duplicate union construction when the
persistent membership cache filled. `93322acc` restored the existing
invocation-local retention, and stage 5 reuses mutable layer inputs rather than
allocating each replacement array. Those corrections were included in the
final measurements. Further work should reduce prefix-read overhead and avoid
allocating resident arenas for batches closed at roots before claiming an
end-to-end performance improvement. The serial preparation pool was not used.

A final restored-snapshot cProfile run through the committed replay script
again certified all ten. Inclusive preparation time under instrumentation was
6.552 s versus 6.731 s in the original baseline profile; these are not
unprofiled wall timings. `packing_for` calls fell from 426193 in the initial
stage-3 profile to 9040 after correction; full membership lookup calls fell
from 417163 to 145061. `prepare_variants` (4.519 s inclusive) and exact root
union lookup/construction (2.376 s inclusive, overlapping that total) remain
substantial serial work. Evidence: `.agent/final-profile-ten.json.prof`.

## Reproducing the measurement

Use [execution_replay.py](execution_replay.py) against each frozen source via
`PYTHONPATH`; it imports the existing capture parser and canonical validator.
Acquire the shared advisory `fcntl` lock at
`.agent/gpu-followups-20261106/metal-benchmark.lock` in the shared reference root
before sustained Metal runs. The captures are trusted local artifacts:

- `.agent/runs/colgen-metal-20261105-220722/pricing_captures/round_0001.pkl`
- `.agent/runs/colgen-capture-20261105-211249/pricing_captures/round_0011.pkl`

Pass both with repeated `--capture`, the ten IDs with `--flight-ids`, and
`--batch-size 2 --out <report.json>`. For the eight, pass only round 1 and
`--batch-size 1`. Optional `--profile` instruments only the restored phase;
that phase's elapsed time must not be compared with unprofiled timings.

## Endpoint-branch integration touchpoints

The sibling branch is separate and has not been imported or validated in
combination. Its record stride is `9 + history`, with two paid-mask words at
offsets 7/8. Resident load/copy/compaction preserve the full caller stride and
all flags, but expansion still uses PR142's `7 + history` layout until combined.

- Merge its record-layout/root changes into `gpu_pricing.py`/`pricing_gpu.h`.
  Move its CPU paid-mask bookkeeping from `allocate` into `store_prefix`, so
  lazily recovered predecessors receive the same exact accounting.
- Append its `visit_masks`, `endpoint_meta`, `endpoint_entries` and
  `endpoint_trees` arguments consistently to runtime wrappers and both host
  and resident dispatch construction. This branch appends shared price inputs
  and separate per-flight cutoffs to the original expansion signature.
- Supply distinct visit-mask buffers for each time layer in a block. Retain a
  weighted-array generation until the commands using it complete, then refresh
  at a checkpoint; a missing bound must remain unavailable rather than prune.
- The sibling currently considers masks from the host frontier. Resident
  frontiers need a bounded mask-discovery interface or an explicitly safe
  missing-bound path. No endpoint-bound selection policy is changed here.
- Preserve endpoint proposal flag/counter bookkeeping at the compact CPU
  checkpoint, and rerun combined FP64 proposal-restoration/union tests. The
  opaque-record copy test is an interface check, not an integrated proof.
