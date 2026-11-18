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

Old `colgen.zeta_*` import paths remain thin compatibility modules for archived
experiments and pickles. There is one search implementation. Old graph caches
containing master columns are rebuilt on loading. New work should import the
shared package; the independent workspace/price-preparation experiments must be
adapted to these moved modules before integration.
The legacy `zeta_compact.price_zeta_compact` and `zeta_hybrid.price_hybrid`
entry points retain `pi_f`, `params`, `known_columns` and `ZetaPricingResult` through
CG adapters. Their shared-engine equivalents accept `SearchSettings` and routes.

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
