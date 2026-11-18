# Forward-only twelve-direction dynamic programming experiment

This is an opt-in Zeta experiment. Production DP and the default Zeta best-first
traversal are unchanged. Select `ZetaOptions(backend="compiled",
connection_mode="hex12", forward_cruise=True, traversal="forward_dp", ...)`, or
use the corresponding `experiments.zeta_prototype` flags. The native Cython loop
and compiled history helpers remain optional; the measured arm enables them.

## Domain and state

The existing graph filter is authoritative: retain an allowed hex offset only
when its projection onto the request's fixed origin-to-destination direction is
strictly greater than `1e-8` metres. At 120 m pitch the original twelve offsets
have lengths 120 m and sqrt(3) × 120 m, six each. A generic heading retains six;
alignment can remove perpendicular links, and a zero direction retains none.
The graph is narrower than unrestricted twelve-direction or full-radius Zeta.

`ForwardTopology` prepares immutable variable-degree CSR adjacency and validates
an order with Kahn's algorithm against the actual neighbors. A cached reverse-DAG
mask excludes waypoints with no directed path to the chosen destination; physical
constraints can remove further paths, but cannot make an excluded waypoint reachable. Each waypoint bucket
holds all distinct departure, endpoint, arrival-time and reservation-history
states. Every state at a waypoint is processed before its successors. Different
edge durations need no synchronized time layers. There is no cost-priority queue
in the primary traversal. Auxiliary shortest-route seed construction still uses
its established queue and has a separate `incumbent_setup_s` counter.

The shared compact kernel retains packed integer/float parent arrays, exact
arrival times, both accumulated lengths, first/terminal roles, interned contexts,
paid sets and segment sequences, arbitrary-size exact retired price charge,
expiry and guarded dominance. Neither hop count nor destination-cell occupancy
replaces the simulator's geometry. All intersected cells/altitudes and touched
4 s reservation periods, including the 4 s buffer, remain authoritative. Optional
1 s arrival snapping with 5% speed allowance still permits nominal fallback.
Signed dual prices apply to the union of reservation rows once each.

## Bounds and stopping

A bucket stores a conservative minimum of its labels' completion bounds. The
minimum across every pending bucket covers all unprocessed states. Within the
current waypoint, the floor of a consumed or dominated label may remain until
that waypoint finishes; this can loosen the certificate but cannot strengthen it
incorrectly. Existing history-merge numerical uncertainty is still included.

The expansion cap counts surviving labels actually expanded, including terminal
labels. The generation cap counts retained labels after state merging. DP has no
lazy successor-pop count: it evaluates the outgoing edges of each expanded label.
It stops exactly at its generation cap. During a successor-batch interruption,
the active parent's floor also covers successors not yet generated. Best-first
retains its existing one-successor-batch cap convention, possibly completing
both endpoint roles at the boundary. There is one primary pricing stage; no
cheap-stage budget or future congestion-price bound is enabled in the comparison.

Early discovery returns an improving new canonical column with incomplete status.
A no-improvement proof uses the bound over all pending states. A cap, deadline or
unresolved numerical allowance never becomes convergence. Exhaustion is exact
subject to the existing numerical-merge allowance.

The fleet experiment uses the same 3,495 filtered full480 seeds for both arms.
Those fixed seeds may use longer links. Certificates cover the forward12 graph
plus that declared fixed seed pool: LP constraints cover the already-present seed
columns, and pricing bounds cover missing forward12 routes. A standalone pricing
call revalidates its incumbent against its search graph; it does not independently
certify arbitrary long seeds outside that graph.

## Infrastructure inventory

The newer production reference was checked against commit
`17793ec2e8fa2b204b1dbce36e60f3f7269098e7`, including `pricing.py`, `dp_kernel.py`,
`dp_prepare.py`, `pricing_pool.py` and `solver.py`.

| Mechanism | Reuse in this experiment |
|---|---|
| Prepared variable-degree CSR | Adapted to authoritative Zeta neighbors and validated spatial order; cached on each worker-owned graph. |
| Packed parent arena, native writes | Existing Zeta implementation reused, including write-before-read allocation and lazy route reconstruction. |
| Exact summation and state buffers | Existing native union/expiry kernels and generation-mark scratch reused; immutable retained arrays remain separate. |
| Interned state and dominance indexes | Existing exact contexts, paid sets, sequence IDs, both lengths and guarded equal/shorter-history rules reused. |
| Geometry, footprint, timing and dual mapping caches | Existing Zeta persistent geometry/leg/time-window and route-coverage caches reused; price-table-local row IDs rebuilt for changed duals. |
| Completion, canonical and duplicate checks | Shared Zeta implementation; immediate useful-sink return added for forward DP. |
| Persistent pricing workers | Existing Zeta spawn/pipe pool reused, with 12 owners, parent-built native artifact, streamed results, deadline and partial-sweep handling. It already follows the production protocol but owns Zeta graphs/seeds/results. |
| Production time-layer kernel | Incompatible without redesign: fixed next-step arithmetic, constant link cost, hop-derived length/time, visit-window footprints and recent-hop state do not represent this model. |
| Production worker class | Cannot substitute directly because it dispatches a different graph/kernel and result protocol. The compatible Zeta pool preserves cached graph/seed ownership. |

Targeted tests extend the existing objective, network, pricing-pool and end-to-end
suites. They compare small exhaustive and best-first solutions in the same forward
domain; cover signed/changed prices, full claims, timing, both length limits,
forward alignments/degeneracy, new columns, duplicate exclusion, caps/deadlines,
serial/parallel parity, and physical collision/capacity/forward audits.

Frozen sources, guarded fixed-dual profiles, alternating actual100 runs, resource
measurements and the results report are in the canonical checkout's
`.agent/zeta-forward-dp-20261118/`. Every optimizer, test and compilation run uses
its absolute shared `metal-benchmark.lock`. No PR, merge or default switch is part
of this experiment.
