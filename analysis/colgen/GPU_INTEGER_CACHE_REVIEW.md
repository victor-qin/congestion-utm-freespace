# Integer payload reuse and remaining host/device work

2026-11-07. Source and measurements audited at `a794c68e`, before the separate
approach-and-arrival bound experiment. This is a design/evidence report; no
production cache or new timing experiment was added. Measurements below are
from existing serialized actual-Metal runs on the M1 Ultra. No CUDA hardware
claim is made.

The integer payload **can be cached**. Its current one-invocation lifetime is
an implementation choice that makes invalidation straightforward, not a
requirement of integer interval arithmetic. Reusing an unchanged immutable
encoding preserves exactly the same enclosure. The cache must own immutable
data, validate its full input identity, and keep mutable search state separate.

## What the current measurements actually show

The claim that integer pruning expands substantially more labels is not
supported by these matched runs. The label explosion in flight 2224 also
occurs with the original FP64 verifier.

| Hard88, first warm pass in repeat 6 | Reference GPU + CPU verifier | Integer GPU |
| --- | ---: | ---: |
| Pricing wall time | 38.8001 s | 25.8464 s |
| Complete/certified flights | 88/88 | 88/88 |
| Total stored labels | 511,170 | 511,172 |
| Final full-search labels | 3,105 | 3,105 |
| Preliminary-search labels | 467,105 | 467,107 |
| Candidate records counted | 2,467,942 | 2,467,987 |
| Per-prune CPU numerical checks | 1,130,556 | 0 |
| Original canonical completion validations | 1,674 | 1,674 |
| Accounted upload copies | 667.891 MB | 1,559.670 MB |
| Accounted download copies | 187.076 MB | 352.869 MB |
| Host waits | 16,083 | 16,176 |
| Measured device command time | 0.633 s | 1.586 s |
| CPU numerical-retention timer | 12.605 s | absent |
| Integer preparation / per-layer gathering | — | 3.428 / 0.424 s |

The two additional labels belong to flight 1920's preliminary search; its
final full-search label count is zero in both modes. Candidate counts differ
slightly between repeated runs, but all returned RCs and columns match.
Across both warm repetitions the wall medians are 38.7924 and 25.8866 seconds,
a 33.27% reduction. This is historical fixed-price pricing replay, not
complete solver convergence.

The copy-volume increase is real. Removing CPU numerical decisions does not
currently remove all record readback. The preliminary search still uploads
frontiers and downloads all candidate records; integer mode adds 15 int32
words, or 60 bytes, to every record. It also encodes/uploads integer operands.
Thus the main measured gain is removal of roughly 1.13 million CPU numerical
checks, despite more copying and arithmetic. It is inaccurate to describe this
implementation as having eliminated CPU/GPU traffic.

The same distinction explains a regression:

| Late18, repeat 4 | Reference | Integer |
| --- | ---: | ---: |
| Changed-price call | 2.6744 s | 3.5369 s |
| Same changed prices, warm call | 1.7266 s | 2.6118 s |
| Labels on either call | 19,668 | 19,668 |
| Candidates on either call | 84,152 | 84,152 |
| Per-prune CPU checks on either call | 59,742 | 0 |
| Integer preparation, changed / warm | — | 1.296 / 1.300 s |

All 18 flights certify with exact matching RCs. The reference retention timer
is only about 0.378 seconds on the changed-price call. Paying roughly 1.3
seconds to rebuild the integer inputs is expensive when there are relatively
few CPU checks to remove. The same overhead persists on the next warm call;
snapshot change alone does not explain the regression. Timers overlap and
must not be added into an exact wall-time attribution.

For the actual flight-2224 subset-solver cap, both modes store 272,210 labels
across all attempts, including the same final 250,000-label pass, and count
731,569 candidates. The integer bound's extra margin in sampled final states
is only 3.813e-11. That case is a shared search/bound weakness, not evidence
that integer rounding caused substantial extra search.

## Where parallelism and traffic remain

GPU expansion is parallel in both modes. `ResidentSearch.expand` dispatches
one invocation per parent/successor slot, up to six slots per parent. The
default 2,048-parent chunk therefore provides 12,288 logical expansion
invocations. Metal groups up to 128 invocations per threadgroup; the GPU
schedules many groups. This is parallel work within a frontier, not a claim
that every time layer or every flight is independent.

The next frontier depends on the previous one. The resident implementation
chains load/expand/compact kernels and can group chunks or two suitable
layers before a host checkpoint. CPU preparation, layer-input uploads,
completion scoring, explicit limits and command submission still take time.
The preliminary search has a 64-label beam per flight and uses the older
host-facing path. In this hard88 run it issued 15,706 preliminary-search
dispatches, versus 284 resident command blocks; batch size was one flight.
These small frontiers and many waits limit how much GPU throughput can help.

The device-time counter is not an occupancy measurement. Still, roughly 1.59
seconds of device command time within 25.85 seconds of wall time is evidence
that speeding up the expansion kernel alone cannot remove most remaining
latency. Reusing prepared inputs and reducing preliminary-search readback
target different costs from improving the bound or proving safe dominance.
Parallel hardware reduces the cost per expansion; it does not eliminate the
exponential number of distinct prefixes observed in flight 2224.

On this Metal backend, upload/download counters count explicit copies into
and out of shared Metal buffers. They are not measurements of PCIe traffic or
standalone transfer latency. CUDA uses device transfers behind the same
interface, but was not measured on hardware.

## What can be reused safely

| Data | Same snapshot/domain | Changed prices |
| --- | --- | --- |
| Topology, exact unpriced memberships, geometry | Already partly cached; retain immutable forms | Reuse when graph/config/objective dependencies match |
| Original FP64 geometry/hop cost operands | Cache by geometry, canonical root and hop domain | Reuse independently of prices |
| Encoded geometry/hop cost arrays | Reuse with the same integer exponent | Reuse only if exponent and original operands still match |
| Exact positive resource/time lookup and encoded prices | Reuse by snapshot, local layout, clocks and exponent | Rebuild; new rows and sign changes matter |
| Root constants and positive origin/visit unions | Reuse with exact root identity, objective, assignment dual and snapshot | Rebuild price-dependent pieces; keep unpriced membership |
| Endpoint weighted trees and their integer encodings | Reuse only with identical selected rows, masks, domain and generation | Rebuild weights, selection and scale-dependent encoding |
| Future tables and encoded rows | Reuse only with matching topology, root domain, restrictions, objective, snapshot and table mode | Rebuild price-dependent tables |
| Cutoffs, labels, predecessor IDs, queues, discarded-upper accounting | Fresh for every search | Fresh for every search |

The existing `DevicePriceCache` already retains up to two original `DualView`
snapshots under a byte budget. That explains the zero **shared price-table**
uploads on warm runs, but it does not cache the separate integer payload.
`IntegerBatch.prepare` constructs a new `IntegerData`; batch cleanup clears
the runtime's per-flight uploads and its `integer_data` reference.

Scale must be part of an encoded-array key. It depends on benefit, assignment
dual, signed-price summaries, geometry cost scale and maximum ground cost in
the prepared root domain. Even identical resource prices do not alone imply
identical scale or root layout. A better incumbent can screen departures and
change compact root indices. Reusing arrays by packed root number would be
unsafe; use canonical departure/lane/start-cell identity and remap to the
current compact layout, or require an exact ordered-domain match.

## Smallest useful implementation

1. Add an explicit runtime-owned, byte-bounded cache of immutable integer
   operands. Do not retain the whole `IntegerData` object: it refers to the
   invocation, incumbent, offsets and dynamic generations.
2. Start with exact same-snapshot hits: key by the owned snapshot object,
   topology/geometry token, config/objective, assignment dual and benefit,
   ordered root descriptors, restrictions, relevant numeric options and scale.
   Cache root operand blocks and exact positive-row lookup encodings. Keep
   endpoint/future-dependent operands as distinct generations.
3. Borrow immutable device buffers only from the same runtime owner. A hit
   skips encoding and upload, but creates fresh mutable search metadata,
   cutoffs and certificate accounting. Publish only after a complete build;
   deadline or overflow never publishes a partial entry.
4. Add price-independent FP64/encoded geometry caching as a separate tier.
   That tier can help changed-price calls. Do not eagerly materialize tables
   for roots that close: the previous eager-allocation failure must stay fixed.
5. Bound both host and device retention, count cache bytes separately from
   live invocation budgets, and record hits, misses, evictions, encoding time
   and bytes avoided. Eviction releases cache ownership only; in-flight
   commands keep their immutable generation alive. Clearing the runtime must
   release every cached generation.

The 88 flights' recorded peak integer-storage counters sum to about 82.9 MB,
but these include per-call scratch/generation peaks and are not a measurement
of a proposed cache's size. A budget smaller than a repeated sequential cohort
can cause an LRU to miss on every sweep; use measured admission/hit rates
rather than assuming that a bounded cache will automatically be warm. Keeping
two whole snapshots can also worsen the already higher process RSS. Cache
only measured reusable pieces before expanding retention.

Caching could remove part of the measured 3.43-second preparation cost on
hard88 and 1.30 seconds on late18; those figures are **not promised speedups**.
Lookup, identity validation, uncached generations and memory pressure remain.
It also cannot remove the preliminary-search record traffic by itself.

Required functional checks are warm-hit parity with encoding/upload guarded,
misses for new/zero-to-positive/sign-changing rows and changed assignment
duals, objective/clocks/hop/restriction changes, root filtering/reordering and
batch regrouping, integer-exponent changes, deadline/overflow during build,
bounded eviction with active borrowers, and runtime-owner isolation. Preserve
canonical completion validation and compare complete certificates on the
same-price, changed-price and restored-snapshot sequences.

## Source and evidence index

Line references are to `a794c68e`:

- `gpu_integer_search.py:34–69`: fresh numeric owner, scale inputs and roots.
- `gpu_integer_search.py:89–158`: per-root original operands and GPU root pass.
- `gpu_integer_search.py:160–243`: exact price packing and lazy retained-root costs.
- `gpu_integer_search.py:245–309`: fresh cutoffs, endpoint generations and two-layer cache.
- `gpu_pricing.py:1032–1180`: resident versus host-facing preliminary search.
- `gpu_pricing.py:1182–1206`: preliminary beam and distinct predecessor lifetime.
- `gpu_pricing.py:1252–1267`: batch cleanup and count scope.
- `gpu_resident.py:168–228,232–289`: checkpoint events and parallel expansion.
- `gpu_runtime.py:270–350,392–448,536–565`: cache lifetime, shared-buffer copies,
  command submission, threadgroups and counters.
- `gpu_prices.py:153–199`: existing bounded same-snapshot price cache.
- `pricing.py:140–179`: normalized snapshot identity; raw mappings are copied.

Raw repeat-6 evidence:
`.agent/hard88-warm-comparison-6.json` and
`.agent/integer-replays/hard88-{reference,integer}-6.json`.
Late-price evidence: `analysis/colgen/integer_pruning_results.json` and its
`.agent/integer-final-replays/late18-{reference,integer}-4.json` inputs.
Compact extracted evidence, including input hashes and per-flight label
differences: `.agent/cache-audit/evidence.json`.
