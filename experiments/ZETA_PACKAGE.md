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

The twenty old helper modules, including the compact/hybrid CG entry points,
have been removed. For example, replace `colgen.zeta_geometry` with `zeta.geometry`.
CG callers use `colgen.zeta.price_zeta`; direct search callers use `zeta.search`
with `SearchSettings`, `Route` and `SearchResult`. The public CG adapter retains
its CG arguments and result type. Independent workspace/price-preparation
experiments need the same import migration before integration.

Archived scripts that select frozen source keep running against that source.
For trusted graph pickles reused against current code, explicitly load through
`experiments.zeta_migrate.load_archive`, or write a new migrated copy:

```sh
python -m experiments.zeta_migrate archived-graph.pkl migrated-graph.pkl
```

The command refuses to overwrite an existing destination and leaves the archive
unchanged. Old column-valued graph caches are rebuilt on loading; the new copy
uses shared helper class paths and can subsequently use ordinary `pickle.load`.

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
        connection_mode="hex12", forward_cruise=True, traversal="forward_dp",
        backend="compiled", compiled_loop=True, native_state_index=True,
        expire_history=True, history_dominance=True, strong_history_dominance=True,
        compiled_history=True, arrival_snap_s=1.0, speed_fraction=0.05,
    ),
    max_air_overrun_hops=6,
    time_limit_s=10.0,  # per leg, including preparation
))
intent = planner.plan(request, ledger, cfg)
# The simulator/USS owns validation and commit; plan itself never mutates the ledger.
```

The default standalone options use the reference backend, so Cython is optional.
Native options are identical to those used by CG and require the compiled and
`zeta-native` extras plus a C compiler. Current domain support remains one cruise
level and fixed terminal exit lanes with registered permanent terminal walls.

Before each search, the planner unions the live volumes **per flight ID**, then
counts how many flights use each row. Cell capacity remains one; pad capacity
comes from the terminal registry. Only saturated rows are forbidden. Permanent
walls are geometry constraints, not dynamic occupants. Zero prices never stand
in for these occupancy constraints. Exact endpoint and ground-hold checks cover
itinerary parking/turnaround times; those checks depend only on the continuation
state, preserving history dominance. The ledger stays fixed throughout one call.

`last_result` exposes incomplete/deadline status even if a feasible route is
returned. A preparation timeout does not search against a partial reservation
map. Preparation checks the deadline between stages and volume conversions;
individual geometry operations are cooperative rather than preemptible.

## Comparing fleet policies

```sh
uv run python -m experiments.zeta_prototype --arm zeta_fcfs \
  --case crossing --flights 2 --distance-cells 2 --workers 0 \
  --radius 480 --fcfs-leg-seconds 10 --out results/zeta-fcfs.json
```

Use the existing `--arm zeta` for column generation. Every connection, history,
cache and native option also applies to `zeta_fcfs`. FCFS processes requests in
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

Frozen source, input checks, progress and audits are in canonical
`.agent/zeta-fcfs240-20261118/`. The figure can be regenerated with
`fig_zeta_fcfs240("context/figures/zeta_fcfs240_20261118.json")` from
`context/figures/make_figures.py`.
