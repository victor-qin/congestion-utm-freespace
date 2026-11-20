# Shared Zeta search and standalone planning

`freespace_sim.planner.zeta` owns the route search shared by column generation
and the simulator's per-request planner. Full-radius, twelve-direction,
forward-only DP, Python, compact and native execution all use this package.
The production planner selection remains unchanged.

## Interfaces and ownership

| Component | Responsibility |
| --- | --- |
| `planner/zeta` | Connections, exact arrival clocks, union-priced histories, dominance, bounds, native indexing, geometry/timing caches and route search. |
| `planner/route_domain.py` | Shared physical corridor, endpoint lanes, walls and reservation-row identities/capacities. |
| `planner/route_geometry.py`, `route_cost.py`, `reservation_windows.py`, `route_numeric.py` | Single implementations of route translation, physical objective, reservation periods and exact summation. Both planners import them. |
| `planner/colgen/zeta.py` | Converts master columns to routes and route cost bounds to reduced-cost bounds, retaining the master's numerical improvement guard. |
| `planner/colgen/zeta_pool.py`, `zeta_solver.py` | Persistent pricing ownership, seed scheduling, LP/IP master and column-generation iterations. |
| `planner/zeta/planner.py` | One flight against the current ledger; normal simulator FCFS filing commits the result. |
| `planner/zeta/fcfs.py` | `run_fcfs` files one-way requests in FCFS order and audits row loads and claims. |

The core does not import `colgen`, construct `ColGenParams`, or start an LP solver.
`Route` carries waypoint/endpoint/departure choices, physical cost and immutable
reservation claims. `SearchSettings` supplies physical weights, tolerance and an
optional target cost. `search` returns a `SearchResult` with a feasible route,
physical-plus-reservation cost, a conservative lower bound and completion/work
information. Signed reservation prices are optional; every row is paid once.
Positive finite physical weights are required by the distance bounds.

Without a target, search seeks the best route. With a target and early return
enabled, finding a qualifying route can return incomplete; a no-improvement
proof refers only to that target. Caps and deadlines retain conservative bounds.
Numerical history-merge uncertainty remains explicit. The unadjusted floor is
retained separately so the CG adapter preserves its previous floating-point
bound arithmetic exactly.

Ordinary price bounds use one shared summation-error allowance for accumulated
positive charges, global negative credit and the canonical signed union. Prices
cancel before physical cost is added, with downward rounding at each operation.
If a finite allowance cannot be established, that bound becomes negative infinity;
caps and deadlines then remain incomplete with a conservative certificate.

`colgen` retains only `zeta.py` (CG objective/column adapter), `zeta_pool.py`
(persistent pricing workers), and `zeta_solver.py` (LP/IP and column-generation
scheduling). Search, geometry, bounds, caches and native compilation live in
`planner.zeta`; current callers import those modules directly.

CG callers use `colgen.zeta.price_zeta`; direct search callers use `zeta.search`
with `SearchSettings`, `Route` and `SearchResult`. The public CG adapter retains
its CG arguments and result type.

Current code reads only captures made with current code. Graph pickles from the
prototype branch are not migrated; scripts that select frozen source keep running
against that source, and anything reused against current code is recaptured.

## Traversal for column generation

Use `traversal="best_first"` for column generation. It expands partial routes in
order of their completion-cost lower bound; this is **best-first**, not breadth-first
search. It is already the default in `ZetaOptions` and
`experiments.colgen_full_cohort`. To apply it to an existing sweep configuration,
change only `traversal`: connection geometry, `forward_cruise`, turn rules and work
caps remain independent choices. An options-file override is:

```json
{"zeta_options": {"traversal": "best_first"}}
```

The opt-in `forward_dp` traversal processes waypoints in forward topological order.
It can spend its expansion budget on partial routes before reaching a destination.
Keep it available for explicit comparisons; the CG measurements below do not
establish which traversal is faster for standalone planning.

### 100-flight forward-graph comparison (2026-11-20)

The comparison used main `c28a30aa` plus the then-unmerged price-rounding and compiled
arrival-bound patches. All arms used price snapping, arrival bounds, the same
forward hex12 graph and seed pool, turns strictly below 90 degrees, 12 workers,
50,000 expansions / 250,000 labels per call and a 300-second solve budget.
`target_cutoff` is the dependency patch's option to prune at the pricing target.

| Traversal | Target cutoff | Pricing seconds | Rounds | Expanded partial routes | Final capped calls | Reported gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Forward sweep | Off | 192.1 | 17 | 7,070,749 | 7 | 11.23% |
| Forward sweep | On | 144.3–164.4 | 16 | 3,294,881 | 3 | 9.70% |
| Best-first | On | 57.9–59.6 | 6 | 894,852 | 2 | 4.77% |

The cutoff and best-first arms each ran twice in mirrored order and reproduced
their work counts and outcomes exactly. Unrelated VM CPU load varied, so the
timings are indicative; expansion counts provide the more reproducible comparison.
All five solves served all 100 flights with no detected collision, capacity or turn
violations. Gaps cover the restricted graph and supplied seed pool.

On identical final-round inputs, best-first found an improving route for flight
2278 in 5,398 expansions; both traversals still capped on flights 2386 and 3098.
The sweep capped on all three. Thus best-first reduces capping but does not remove
it. These capped calls had feasible routes available, but neither found an improving
route nor proved that no improvement existed before the cap. A cap is not a denial
of service for that flight.

## Standalone FCFS

Use `get_planner("zeta", ZetaPlannerParams(...))`, or `sim.run` with
`cfg.planner="zeta"` and the same parameter object. `get_planner` applies the
normal itinerary wrapper; a directly constructed `ZetaPlanner` accepts one leg
and explicitly refuses round trips.

```python
from freespace_sim.planner import get_planner
from freespace_sim.planner.zeta import ZetaOptions, ZetaPlannerParams

planner = get_planner("zeta", ZetaPlannerParams(
    options=ZetaOptions(
        connection_mode="hex12", forward_cruise=True, traversal="best_first",
        backend="compiled", compiled_loop=True, native_state_index=True,
        expire_history=True, history_dominance=True, strong_history_dominance=True,
        compiled_history=True, arrival_snap_s=1.0, speed_fraction=0.05,
    ),
    max_air_overrun_hops=6,
    time_limit_s=10.0,  # per leg, including graph preparation; not the one-time native build
))
intent = planner.plan(request, ledger, cfg)
# The simulator/USS owns validation and commit; plan itself never mutates the ledger.
```

The default standalone options are `tuned_options()`: full pruning on the Cython native
loop, which needs the `zeta-native` extra (`uv sync --extra zeta-native`) and a C compiler.
`tuned_options(native=False)` keeps the same pruning on the pure-Python reference backend.
Native options are identical to those used by CG and require the compiled and
`zeta-native` extras plus a C compiler. Current domain support remains one cruise
level and, for hub endpoints, fixed terminal exit lanes with registered permanent
terminal walls. `ZetaPlanner.plan` refuses anything else before any work, naming the
zeta planner and the setting (`flight_levels_m`, `terminal_airspace_always_active`,
`fixed_exit_lanes`); `experiments.run` applies the same check once per run.

Before each search, the planner unions the live volumes **per flight ID**, then
counts how many flights use each row. It reads only volumes that can share a row
with the leg: the ledger's time/space buckets around the corridor waypoints and
endpoint footprints, over the departure-to-latest-landing window. On every row the
leg can claim, the result equals a whole-ledger count, so preparation follows local
traffic rather than ledger size. Cell capacity remains one; pad capacity
comes from the terminal registry. Only saturated rows are forbidden. Permanent
walls are geometry constraints, not dynamic occupants. Zero prices never stand
in for these occupancy constraints. Exact endpoint and ground-hold checks cover
itinerary parking/turnaround times; those checks depend only on the continuation
state, preserving history dominance. The ledger stays fixed throughout one call.

`last_result` exposes incomplete/deadline status even if a feasible route is
returned. A preparation timeout does not search against a partial reservation
map. Preparation checks the deadline between stages and volume conversions;
individual geometry operations are cooperative rather than preemptible.

## Optional cruise turn limit

`ZetaOptions(turn_below_90=True)` and production DP
`ColGenParams(turn_below_90=True)` (CLI `--colgen-turn-below-90`) restrict the
route domain. At every interior cruise waypoint, the incoming and outgoing
waypoint links must have a strictly positive dot product. The test uses exact
integer axial arithmetic (`planner/turn_rule.py`), so exactly 90 degrees and
reversals are rejected and straight continuations are kept. The first cruise
link, climb/descent and terminal lanes or customer stubs are not checked.
Forward-only cruise remains a separate option. Inside one `solve_zeta`, both
flags must agree.

Zeta labels carry an exact heading class. Classes group incoming links with
identical legal next links, and every history and dominance key includes the
class. Successor caches are keyed by heading, while geometry caches stay
shared. Supplied, imported and seeded routes are validated against the same
domain. A free shortest seed that already obeys the rule is reused before the
heading-aware seed search. Production DP needs no new state because its
dominance key already contains the predecessor cell. Certificates cover only
the restricted domain. GPU pricing refuses the option.

## Optional price rounding and target pruning

`ZetaOptions(price_snap=True)` searches on rounded prices: every positive price
below the column-generation epsilon (1e-6) and every negative price is withheld
from the search vector. Labels, floors, histories and the pruning cutoff all
live in that rounded space, so solver noise no longer splits paid-row sets.
Candidate routes are re-priced on the full prices; the reported column and any
early return use full-price reduced costs. The certified bound adds back the
withheld negative mass, rounded down; withheld positives need no correction. A
complete search that cannot decide the target under the full prices reports
`snap_uncertainty` (incomplete), as `merge_uncertainty` does. With nothing
withheld the search is unchanged.

`ZetaOptions(target_cutoff=True)` applies to early-return calls only. It prunes
at `min(incumbent, cut)` with `cut = target - withheld negative mass + largest
endpoint merge margin`, and exhausting below the cut finishes `no_improvement`.
Soundness never depends on the cut, because the bound is re-derived when the
search finishes; the two allowances only let that exhaustion certify. Without
the merge-margin term, merge-guarded proofs became `merge_uncertainty`. Only
`traversal="forward_dp"` changes; best-first already expands in floor order.

Measured on actual100 with the turn limit on (2026-11-20): best-first radius240
with rounding narrowed the certified gap from 11.6% to 8.5% with 44% fewer
expansions; the forward sweep ran 17 rounds instead of 5, because Zeta stops
only when a sweep finds nothing, and the cut halved its expansions. Production
DP does not use price rounding (`context/HISTORY.md`).

## Comparing fleet policies

```sh
uv run python -m analysis.colgen.zeta_prototype --arm zeta_fcfs \
  --case crossing --flights 2 --distance-cells 2 --workers 0 \
  --radius 480 --fcfs-leg-seconds 10 --out results/zeta-fcfs.json
```

Use the existing `--arm zeta` for column generation. Every connection, history,
cache and native option also applies to `zeta_fcfs`, which calls
`freespace_sim.planner.zeta.fcfs.run_fcfs`. FCFS processes requests in
`FlightRequest.sort_key()` order. Requests cannot be priced in parallel against
stale copies of the ledger; the CG arm retains its persistent pricing pool.

FCFS certificates are **per leg with earlier filings fixed**, not fleet LP lower
bounds. `--fcfs-leg-seconds` is a per-leg budget; the CG `--seconds` budget covers
its whole solve. The harness reports this distinction and independently audits
physical conflicts and full-volume row capacities. Compare matching domains:
full480+FCFS with full480+CG, or forward12+FCFS with forward12+CG. If fixed seeds
contain longer links, disclose them in the certificate scope for each arm.

### Captured 100-request radius240 FCFS trial (2026-11-18)

The original captured first 100 requests were filed with all connections within
240 m (18 offsets at 120 m pitch), no forward restriction, and no CG seed pool.
The process refused all CG imports. Native state/history/coverage helpers and
both guarded history merges were enabled; 50k expansions / 250k labels and a
5 s per-leg deadline, with the unchanged 1 s snap/5% speed allowance and 4 s
buffered reservation rows. Captured physical cost weights remained ground 1 / air 3.

- 98 accepted; flights 330 and 1454 timed out without a route. All collision,
  capacities, reconstructed claims and radius audits passed. No selected route
  revisited a waypoint; the search domain itself permits backward links.
- Search 277.932 s; FCFS with claim audit 282.882 s; complete guarded process 294.891 s,
  including 8.297 s cold native compilation and 2.025 s separate kernel warmup.
  Peak process-tree RSS 0.708 GiB.
- Accepted-route cost 2725.748287; including two 10,000 denial penalties,
  objective 22725.748287. Certificates: 2 optimal, 59 numerical allowances,
  39 deadlines (37 retain feasible routes). Maximum merge allowance 0.000102 cost
  units; no small-gap guarantee for deadline results and no fleet lower bound.
- 1,771,720 expanded / 2,446,913 generated histories / 7,072,008 edge evaluations.

The earlier full480 trial also accepted 98, at cost 1002.364407. It is a historical
reference from before the signed-price fix, not a fresh matched timing control.
This radius240 trial does not establish a runtime or cost benefit.

Frozen source, input checks, progress and audits are in the local, untracked
`.agent/zeta-fcfs240-20261118/` folder. The figure can be regenerated with
`fig_zeta_fcfs240("context/figures/zeta_fcfs240_20261118.json")` from
`context/figures/make_figures.py`.
