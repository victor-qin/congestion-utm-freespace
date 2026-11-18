# Shared native Zeta state index

Implementation now lives in the shared [Zeta package](ZETA_PACKAGE.md), usable
by column generation and standalone FCFS planning.

`ZetaOptions(native_state_index=True)` (CLI `--native-state-index`) enables an
experimental representation change in the shared compact pricing loop. It requires
`compiled_loop=True`; it works with either Python or compiled history comparisons.
The default stays off. No production planner, geometry, price or work-cap setting changes.

Every compact Zeta domain uses the same implementation: full-radius best-first,
240 m or 480 m connections, twelve-link and twelve-plus-long-link graphs, hybrid
discovery/full certification, and forward12 DP. The index assumes no spatial order,
fixed link duration, constant hop length or bounded recent history. Only the existing
forward DP traversal requires a forward graph; best-first can retain cyclic graphs.

## Identity and dominance

The native open-addressed table stores the exact waypoint, context, arrival timestamp,
both accumulated lengths, endpoint/first-segment roles and segment-sequence ID.
It returns stable integer group IDs for the existing three key projections:

- Exact comparison retains all fields.
- Guarded broad comparison omits only the segment-sequence ID.
- Guarded shorter-history comparison also omits both lengths from its grouping key;
  the unchanged dominance rule still compares both lengths separately.

All hash collisions check every retained field. Floating values are not rounded;
positive and negative zero receive the same hash and compare equal. Resizing preserves
IDs. IDs are local to one pricing call and never recycled when labels are replaced.
The history helper still retains multiple live paid histories in each group and applies
the same reject-before-replace order, strict numerical margin and certificate allowance.

The compact loop already interns sorted paid-row arrays by exact byte equality. With
the index enabled, it does so before dominance and supplies that stable paid-set ID to
the history helper. Each live set and its positive/negative partitions are then prepared
once per pricing call. Retired charges remain arbitrary-size Python integers. This
cache is owned by the current frozen price vector: changed duals construct a fresh
history helper, including fresh signed partitions. Geometry and seed caches retain
their existing worker lifetimes.

This implements native continuation lookup and reuses paid-set records. The remaining
history containers and exact arithmetic remain shared with the reference implementation;
it does not replace the full dominance algorithm with production's different state model.

## Verification and measurements

Existing functional suites cover exact key identity through growth, signed zero and
nearby floats; signed cancellation/tiny prices and changed-price replay; both length
budgets; loops and variable durations; cap/deadline bounds; duplicates and improving
columns; full/hybrid/forward domains; persistent-worker parity; and physical/capacity
audits. Indexed searches must preserve routes, certificates and operation counts against
the same traversal with indexing disabled.

Reproducible snapshots, raw test logs, guarded fixed-price probes and actual100 results
are maintained in the canonical workspace under `.agent/zeta-native-index-20261118/`.
The initial index-only probes were mixed; caching the existing paid-set partitions
addressed the measured history-preparation cost. Consult that task's report for measured
benefits and limits; no timing claim follows from native compilation alone.

## Recorded actual100 comparison

| Domain | Index off pricing (s) | Index on pricing (s) | Index off total (s) | Index on total (s) |
|---|---:|---:|---:|---:|
| Forward12 DP, two runs per option | 50.752 | 46.195 | 82.926 | 78.515 |
| Full480, first 8 identical complete rounds, one pair | 245.367 | 236.904 | — | — |

Forward DP preserves all 500 pricing results, routes, bounds and operation counts
per run: 100 served, cost 1026.096762, valid lower bound 836.906819, gap 18.438%,
88 complete / 12 unresolved. Pricing decreased 8.98% and total wall 5.32% in this
cohort. The unresolved calls remain 10 caps and 2 numerical-merge allowances.

Full480's matching eight rounds used 3.45% less pricing wall in one pair. Both
whole runs reached the pricing deadline at approximately 284.19 s total; their
different partial ninth rounds prevent a whole-run speed comparison. Both selected
the same 100 routes at cost 1007.828388, lower bound 596.351271 and gap 40.828%.
Neither domain converged. Certificate scopes are the configured graph plus the
fixed seed pool: 3,495 forward-compatible full480 seeds for forward12; 3,500 for full480.

The comparison uses 12 persistent workers, 50k / 250k work caps, a 300 s solver
budget, 1 s snap / 5% allowance, and no future-price/completion bounds. Total wall
includes seed preparation, graph, pool, master, IP and physical audit. Cold native
compilation took 7.514 s separately. Peak RSS off/on was 7.679/7.638 GiB for DP and
11.768/11.787 GiB for full480. Interned rejected keys/paid sets remain until a call
ends, so memory reduction is not guaranteed on other inputs.

Validation before integration: 200 focused tests, 24 matched fixed-price calls and
six actual100 solves. All six physical collision/capacity/cost audits passed.
These measurements belong to the frozen pre-integration snapshot, subsequently
committed as `ae0734a33f994777c3de0c261d7d99facf5c01ba`; current-main integration
checks are recorded separately in the PR. Raw logs, input/source hashes, negative
probes and reproduction harnesses remain in the task directory named above.
