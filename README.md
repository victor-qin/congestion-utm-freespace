# freespace-sim

A **strategic-layer UTM (drone traffic management) simulator** that demonstrates the
[ASTM F3548-21](context/F3548-21.pdf) strategic-deconfliction policy under **first-come-first-serve
(FCFS)** allocation, in **continuous free space** — no grid. It is the free-space sibling of the
hex-grid `congestion-demo-real` project: same research question (how does FCFS airspace reservation
congest as demand grows?), but trajectories are planned in continuous 3D + time and previously
committed reservations are treated as space-time obstacles.

Scope is **strategic pre-flight coordination only**. Tactical simulation (wind, position
uncertainty, conformance) is deferred to a future BlueSky integration.

## How ASTM F3548-21 maps onto the code

| ASTM concept | Code artifact |
|---|---|
| 4D volume (3D shape + `[t_start, t_end)`) | `Volume4D` ([volumes.py](freespace_sim/volumes.py)) |
| Trajectory-based operational intent (overlapping corridor boxes) | `build_corridor` → `list[Volume4D]` |
| Area-based operational intent (hover cylinder) | `hover_reservation` |
| Conflict = spatial ∩ temporal (§3.2.8) | `volumes_conflict` ([conflict.py](freespace_sim/conflict.py)) |
| Strategic conflict detection (method not prescribed) | pluggable `Planner` ([planner/](freespace_sim/planner)) |
| FCFS within a priority level (§4.2.5) | `FCFSMechanism` ([mechanism.py](freespace_sim/mechanism.py)) |
| Operational-intent states (§4.4) | `IntentStatus` ([types.py](freespace_sim/types.py)) |

The core invariant: **committed volumes of different flights never overlap in 4D**. `verify.py`
re-checks it after every run, and `sim.run()` asserts it, so every experiment self-validates.

## Quickstart

```bash
uv sync
uv run pytest -q -m "not slow"          # full test suite, ASTM invariant included

# Run a named scenario and capture a complete, replayable run folder:
uv run python -m experiments.metro_scenario --region 5000 5000 --lam 600 --planner lazy
#   → results/<timestamp>_metro_lam600_<hash>/  (config, scenario, trajectories, reservations,
#     metrics, replay.html, figures). Open replay.html to scrub the timeline.

# The headline congestion curve (denial / delay / detour / throughput vs demand λ):
uv run python -m experiments.lambda_sweep --quick --planner lazy

# (Re)generate the replay viewer for any saved run folder:
uv run python -m experiments.replay results/<folder> --open
```

> **Planner speed:** the default planner is `astar_milp` (A\* homotopy + MILP geometry refine) —
> high fidelity but ~3–4 s/flight at high density. Pass `--planner lazy` for fast statistical sweeps.

## Architecture

```
freespace_sim/
  types.py        FlightRequest, OperationalIntent, IntentStatus, DenialReason
  config.py       SimConfig — every knob (geometry, kinematics, cost weights, budgets) as defaults
  geometry.py     FCL-backed BoxSpec / CylinderSpec (oriented 3D box, vertical cylinder)
  volumes.py      Volume4D + the corridor / hover builders (the build-then-check contract)
  conflict.py     volumes_conflict() — temporal prune then exact FCL 3D collision
  ledger.py       ReservationLedger — time-bucketed FCL broadphase (commit / query)
  planner/        pluggable planners (see below), sharing one cost model
  mechanism.py    FCFSMechanism (commit the first conflict-free plan; later flights yield)
  sim.py          run() — the FCFS event loop (+ optional live progress reporter)
  metrics.py      per-flight rows + aggregate rollups (delay / detour / utilization / solve time)
  runs.py         save_run / load_run — full self-contained, replayable run folders
  verify.py       the post-run no-inter-flight-conflict invariant
  viz.py          top-down snapshot, congestion heatmap, 3D trimesh scene, delay histograms
  viz_html.py     standalone HTML replay (scrub / step / hex-grid toggle / dashed origin→dest)
```

Configuration is **override, not edit**: `SimConfig` is a frozen dataclass of defaults; an
experiment customizes a run by constructing `SimConfig(region_size_m=..., lam_per_hour=..., ...)`.
You never edit `config.py` to run a different scenario.

## Planners

All implement one `Planner` protocol and minimize the same cost model (distinct weights for ground
delay vs air detour vs air hold vs altitude change), so they are directly comparable.

| name | strategy |
|---|---|
| `straight` | direct path + departure time-shift into a free slot (deny if space is blocked) |
| `rrt` | space-time RRT\* — reroute / delay / hover / altitude in one search |
| `lazy` | straight first, escalate to RRT\* only for blocked flights |
| `astar` | A\* on a fixed hex lattice (pitch = speed·dt); ground delay + reroute + hover |
| `milp` | MILP trajectory optimization (Richards & How big-M) |
| `opt` / `opt_astar` | NLP (CasADi/IPOPT) continuous polish; `opt_astar` warm-starts from A\* |
| **`astar_milp`** (default) | A\* picks the homotopy + delay; a homotopy-locked MILP refines the geometry as a fast LP |

## Experiments

| command | what it does |
|---|---|
| `metro_scenario` | named stress scenario — you choose region + λ list; each λ → a full captured run folder + replay |
| `lambda_sweep` | the FCFS congestion curve (2×2 panel + total-delay histograms) across demand λ |
| `make_replay` | a single ad-hoc run → full capture + replay + snapshot/heatmap/3D-GLB |
| `replay` | (re)generate/open the HTML replay of any saved run folder, entirely from disk |
| `compare_planners` / `compare_optimizers` | acceptance / cost / runtime across planners on one scenario |

Every run folder is self-contained (`config.json`, `experiment.json`, `scenario.parquet`,
`trajectories.parquet`, `reservations.parquet`, `flights.parquet`, `replay.html`) and a row is
appended to `results/index.parquet` for cross-run queries.

## The replay viewer (`replay.html`)

A standalone webpage (no server) that plays the reservations back like a video:

- **Play / pause** and a **scrub slider**; **⏮ / ⏭** step one timestep (`dt`); **← / →** keys too.
- **Hex-grid toggle** — overlays the exact lattice A\* searched on (only shown when an A\*-based
  planner ran).
- **Dashed origin→dest** reference line per active flight — the gap to its solid corridor *is* the
  detour the FCFS newcomer paid.

## Metrics

Per flight: ground delay, air hold, air detour, altitude change, cost, **stretch** (flown ÷
straight), **total delay** (hold + loiter + detour-time, excluding the mandatory climb), reserved
**volume-seconds**, and **planner solve time**. Aggregates roll these up plus denial rate (with
budget-vs-search-artifact split), throughput, and **airspace utilization** (reserved volume-seconds
÷ region × horizon — the free-space analog of the hex repo's occupancy).

## Status

- **Done:** 3D geometry + FCL conflict engine, all planners, FCFS sim, metrics, run capture,
  visualization, congestion experiments. Tests green (ASTM invariant enforced).
- **Not yet:** real-geography (lat/lon) projection; per-operator async clocks; the BlueSky tactical
  layer (designed-for behind an execution seam, not built).
