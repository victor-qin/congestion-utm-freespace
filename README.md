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

Experiments are a three-stage pipeline joined through saved run folders on disk — **define** a scenario,
**execute** it, **read out** artifacts — composed with plain shell (see [Experiments](#experiments)).

```bash
uv sync
uv run pytest -q -m "not slow"          # full test suite, ASTM invariant included

# EXECUTE one named scenario → a complete, reloadable run folder (the folder path is the last stdout line):
FOLDER=$(uv run python -m experiments.run --scenario dallas_hub_2uss --planner astar_shortcut | tail -1)

# READ OUT artifacts from that saved run (no re-simulation):
uv run python -m experiments.readouts.replay        "$FOLDER" --open   # scrub the timeline, colored by USS
uv run python -m experiments.readouts.figures       "$FOLDER"          # snapshot / heatmap / 3D-GLB
uv run python -m experiments.readouts.uss_breakdown "$FOLDER"          # per-operator table + bars

# Compose sweeps / comparisons as pure shell (loop the run box, then a cross-run readout):
bash experiments/batch/lambda_sweep.sh demo            # λ×seed sweep → congestion curve
bash experiments/batch/compare_planners.sh demo        # several planners → comparison table
```

> **Planner speed:** the default planner is `astar` (A\* on the hex lattice) — fast and 0-denial on the
> metro scenarios. Pass `--planner astar_shortcut` for tighter berths (solver-free), or
> `astar_milp_shortcut` for headline-quality MILP refinement (~1–5 s/flight).

## Architecture

```
freespace_sim/
  types.py        FlightRequest, OperationalIntent, IntentStatus, DenialReason
  config.py       SimConfig — every knob (geometry, kinematics, cost weights, budgets) as defaults
  scenarios.py    ScenarioSpec + DemandSpec + SCENARIOS registry — named worlds → (SimConfig, demand)
  demand.py       DemandModel: UniformPoissonDemand (1+ USS) · HubVoronoiDemand (geographic hubs)
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
| **`astar`** (default) | A\* on a fixed hex lattice (pitch = speed·dt); ground delay + reroute + hover |
| `milp` | MILP trajectory optimization (Richards & How big-M); continuous multi-altitude band, shared-terminal + pad-capacity aware |
| `astar_milp` | A\* picks the homotopy + delay; a homotopy-locked MILP refines the geometry as a fast LP |
| `astar_shortcut` | A\* + a deterministic greedy shortcut pass — solver-free berth tightening |
| `astar_milp_shortcut` | the sandwich: A\* → shortcut → MILP → shortcut. Pre-shortcut speeds MILP gap-certification; post-shortcut crosses residual lock slack + halves the knots |

## Experiments

Three composable stages, joined through saved run folders on disk — so analysis never re-runs the sim,
and the demand pattern / USS count is a property of the **scenario** (reused by every stage for free):

**1. DEFINE** — a `ScenarioSpec` is a named *world* (region, horizon, λ, planner, demand pattern). The
registry in [`scenarios.py`](freespace_sim/scenarios.py) ships `metro_uniform` (1 USS), `metro_2uss`
(2 USS, uniform), and `dallas_hub_2uss` (2 USS, geographic hub-and-spoke). Any field is overridable.

**2. EXECUTE** — `experiments.run` runs **one** scenario and persists it (no plots). Sweeps and
comparisons are pure-shell loops over it, joined by a shared `--tag`:

```bash
uv run python -m experiments.run --scenario dallas_hub_2uss --planner astar_shortcut --tag demo
uv run python -m experiments.run --scenario metro_2uss --demand hub --uss a b --hubs 5 15 --lam 240
```

**3. READ OUT** — standalone consumers that load saved data (never re-simulate):

| readout | scope | from | produces |
|---|---|---|---|
| `readouts.replay` | per-run | a run folder | `replay.html` (scrub, colored by USS) |
| `readouts.figures` | per-run | a run folder | snapshot / heatmap / 3D-GLB (`--uss` slices) |
| `readouts.uss_breakdown` | per-run | `per_uss.parquet` | per-operator table + bar chart |
| `readouts.histograms` | per-run | `flights.parquet` | delay / delay-% / delay-source distributions |
| `readouts.curve` | cross-run | `index.parquet` | congestion curve vs λ (filter by `--tag`/`--scenario`) |
| `readouts.compare` | cross-run | `index.parquet` | comparison table (group by `--by`, default planner) |

Distributions are a *single-run* property, so `histograms` is per-run; the **shell** owns multiplicity
— `lambda_sweep.sh` loops `run`, feeds each folder to `histograms`, and collects them under the sweep
folder. The only genuinely cross-run readout is `curve` (a *trend* needs many points), which reads the
index the loop populated.

**Orchestration** lives in [`experiments/batch/`](experiments/batch) (`lambda_sweep.sh`,
`compare_planners.sh`, `replay_demo.sh`) — plain shell composing the run box + readouts.
(`compare_optimizers.py` stays standalone: it's a planner micro-benchmark on hand-built obstacles, not
the demand pipeline.)

Every run folder is self-contained (`config.json`, `experiment.json`, `scenario.parquet`,
`trajectories.parquet`, `reservations.parquet`, `flights.parquet`, `per_uss.parquet`) and a row is
appended to `results/index.parquet` (with `scenario`/`tag`/`demand`/`n_uss` columns) for cross-run
readouts. **Per-run** readouts (`replay`/`figures`/`uss_breakdown`/`histograms`) write *into* the run
folder (or a collecting `--out-dir`); the **cross-run** `curve`/`compare` describe a run *set*, so they
write into `results/sweeps/<tag-or-scenario>/` (stable per label — re-running refreshes in place).

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

**Steady-state window.** A run's airborne density is a trapezoid — it ramps up from an empty sky,
plateaus, then ramps down as the last flights (and post-horizon returns) land. Metrics over the whole
run are diluted by the low-density ramps, so `metrics.steady_state_window(result)` finds the plateau
(the widest interval where density ≥ `frac`×peak, `frac=0.9` default) and every surface reports both
the whole-run number **and** its steady-state twin measured over that window: `summary.json` carries a
nested `steady_state` block, `index.parquet` gains `steady_*` / `window_*` columns, and the `curve` /
`compare` / `histograms` readouts overlay the two. `--window-frac` tunes the plateau threshold; the
replay clips to the horizon by default (`--no-clip` keeps the return tail). This **supersedes** the
removed `clip_returns_to_horizon` demand hack (issue #25): run the natural demand, but *measure* only
the representative window instead of mutilating the flight set.

## Status

- **Done:** 3D geometry + FCL conflict engine, all planners, FCFS sim, metrics, run capture,
  visualization, congestion experiments. Tests green (ASTM invariant enforced).
- **Not yet:** real-geography (lat/lon) projection; per-operator async clocks; the BlueSky tactical
  layer (designed-for behind an execution seam, not built).
