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
uv sync                                 # installs numba too (default-group), so A* gets the compiled kernel
uv run --extra dev pytest -q -m "not slow"   # full suite; pytest lives in the `dev` EXTRA, which uv sync does not install

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
> metro scenarios. Pass `--planner astar_shortcut` for tighter berths (solver-free),
> `astar_heading_shortcut` for its exact-heading fast-path A/B arm (byte-equivalent
> `OperationalIntent` values after excluding runtime `solve_time_s`), or
> `astar_batched_shortcut` for the route-changing turn-batched A/B arm. Use
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
delay vs air detour vs air hold vs altitude change), so they are directly comparable. All but the
last plan one flight at a time, in FCFS order, against whatever the earlier flights already
reserved; `colgen` instead solves the whole schedule at once.

| name | strategy |
|---|---|
| `straight` | direct path + departure time-shift into a free slot (deny if space is blocked) |
| **`astar`** (default) | A\* on a fixed hex lattice (pitch = speed·dt); ground delay + reroute + hover |
| `milp` | MILP trajectory optimization (Richards & How big-M); continuous multi-altitude band, shared-terminal + pad-capacity aware |
| `astar_milp` | A\* picks the homotopy + delay; a homotopy-locked MILP refines the geometry as a fast LP |
| `astar_shortcut` | A\* + a deterministic greedy shortcut pass — solver-free berth tightening |
| `astar_heading_shortcut` | `OperationalIntent`-equivalent legacy ordering; skips same-heading probes only when reservation sampling is exactly unchanged (the run config retains the distinct public planner name) |
| `astar_batched_shortcut` | experimental A\* shortcut: seed at 3D turns, then batch maximal straight runs for A/B evaluation |
| `astar_milp_shortcut` | the sandwich: A\* → shortcut → MILP → shortcut. Pre-shortcut speeds MILP gap-certification; post-shortcut crosses residual lock slack + halves the knots |
| `colgen` | **whole-schedule** column generation (Balakrishnan–Chandran): a route is a column, the master is a set-partitioning LP over them, pricing is an exact label DP per flight. Single flight level; needs `terminal_airspace_always_active` for hub endpoints |

`colgen` is not FCFS — it optimizes every flight jointly, so it answers a different question from the
rest of the table ("what is the best schedule" rather than "what can this flight get, given the
others"). Its solver knobs are exposed as `--colgen-time-limit`, `--colgen-max-iterations`,
`--colgen-objective`, `--colgen-solver` and `--colgen-gap-metric`, and the one that sizes the pricing
search itself as `--colgen-max-air-overrun` (the hop budget over the lattice geodesic, which is also
the half-width of the O-D ellipse a flight is priced over — the budget implies the ellipse, since a
route within it cannot reach a cell outside); pricing dominates its
cost, and one sweep at 100 flights is already ~147 s, so the 1200 s default budget buys roughly three
iterations there rather than a converged solve. A run that stops on that
budget still files a complete, feasible schedule, so read the WARNING and `planner_stats.json` in the
run folder rather than assuming the result converged. One reporting caveat: a whole-schedule planner
has no per-flight solve, so `colgen` stamps every intent with the same amortized share (solve wall ÷
flights) — its `*_solve_time_s` columns in `index.parquet` are not comparable with an FCFS run's.

## Experiments

Three composable stages, joined through saved run folders on disk — so analysis never re-runs the sim,
and the demand pattern / USS count is a property of the **scenario** (reused by every stage for free):

**1. DEFINE** — a `ScenarioSpec` is a named *world* (region, horizon, λ, planner, demand pattern). The
registry in [`scenarios/`](freespace_sim/scenarios) ships `metro_uniform` (1 USS), `metro_2uss`
(2 USS, uniform), `dallas_hub_2uss` (2 USS, geographic hub-and-spoke), `colgen_test` (a small
congested eight-hub world sized for whole-schedule optimization), and the four explicit density
worlds below. Any field is overridable.

**2. EXECUTE** — `experiments.run` runs **one** scenario and persists it (no plots). Sweeps and
comparisons are pure-shell loops over it, joined by a shared `--tag`:

```bash
uv run python -m experiments.run --scenario dallas_hub_2uss --planner astar_shortcut --tag demo
uv run python -m experiments.run --scenario metro_2uss --demand hub --uss a b --hubs 5 15 --lam 240
```

The density matrix has four canonical recipes:

| scenario | operator mix |
|---|---|
| `density_faa_wing_zipline` | FAA-filing Wing/Zipline-type traffic |
| `density_future_wing_zipline` | far-future Wing/Zipline-type traffic |
| `density_faa_wing_zipline_amazon` | FAA-filing Wing/Zipline plus Amazon traffic |
| `density_future_wing_zipline_amazon` | far-future Wing/Zipline plus Amazon traffic |

In these scenarios λ counts outbound delivery missions. Each outbound is paired with a return, so the
expected number of flight legs is twice the outbound count. Mixed scenarios use two distinct USS
instances (`wing_zipline_uss` and `amazon_uss`), with independent demand streams and operator-specific
service radii and scheduling leads. The canonical density runs generate outbound demand over 30 minutes
inside a two-hour planner envelope used to size the scheduling machinery. That envelope is not a cutoff:
every generated request is processed, and the realized simulation runs from the first flight activity
through the final landing without diluting the reported hourly demand rate. Run the complete matrix with:

```bash
bash experiments/batch/density_matrix.sh paper 0 1 2
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
`compare_planners.sh`, `density_matrix.sh`, `replay_demo.sh`) — plain shell composing the run box + readouts.
(`compare_optimizers.py` stays standalone: it's a planner micro-benchmark on hand-built obstacles, not
the demand pipeline.)

Every run folder is self-contained (`config.json`, `experiment.json`, `scenario.parquet`,
`trajectories.parquet`, `reservations.parquet`, `flights.parquet`, `per_uss.parquet`); runs launched
through `experiments.run` also carry `scenario_spec.json`, the resolved post-override recipe, so the
world is reproducible from the folder alone. A row is
appended to `results/index.parquet` (with `scenario`/`tag`/`demand`/`n_uss` columns) for cross-run
readouts. **Per-run** readouts (`replay`/`figures`/`uss_breakdown`/`histograms`) write *into* the run
folder (or a collecting `--out-dir`); the **cross-run** `curve`/`compare` describe a run *set*, so they
write into `results/sweeps/<tag-or-scenario>/` (stable per label — re-running refreshes in place).

## The replay viewer (`replay.html`)

A standalone webpage (no server) that plays the reservations back like a video:

- **Play / pause** and a **scrub slider**; **⏮ / ⏭** step one timestep (`dt`); **← / →** keys too.
- **Altitude readout** (multi-level runs only) — a fixed 13 px screen-space label, never scaled by the
  view (`LABEL_PX` in `viz_html` is the knob; the declutter grid derives from it, so the two cannot
  drift apart). A drone is labelled only when its label footprint does not overlap another drone's,
  including across neighboring grid cells; dense regions therefore stay clean and zooming in annotates
  everything (1% of drones labelled at fit on a 28k-flight run, 93% by 8×, 100% by 16×).
- **Terminal columns and their exit lanes** — a permanent (always-active) terminal airspace draws as
  an amber no-fly disc at its column radius, with a fine ring at `volumes.exit_radius`, where the hub's
  *reserved* lanes begin. The ring may sit inside or outside the column according to the terminal's
  configured corridor overlap. With the default flush geometry its separation is `corridor_width/2`
  (0.4 px at fit on a 60 km region), so it is drawn only once zoom makes it distinct from the column.
- **Zoom / pan** — scroll to zoom at the cursor, drag to pan, double-click to zoom in, `0` to fit
  (also `+` / `−` and the on-screen buttons). 1–64×, clamped so the region can't slide off-screen.
- **Hex-grid toggle** — overlays the exact lattice A\* searched on (only shown when an A\*-based
  planner ran). The lattice is culled to the viewport and hidden below ~6 px per cell (it reads
  `— zoom in`): on a 60 km region the full grid is ~292k hexagons, which is both illegible at fit
  and far too slow to draw.
- **Dashed origin→dest** reference line per active flight — the gap to its solid corridor *is* the
  detour the FCFS newcomer paid.
- **The clock is the realized run**, not `[0, horizon_s]`: it opens on the first reservation and ends
  on the last to clear, so the slider is all traffic and no empty sky.

**It stays small.** A dense run reserves hundreds of thousands of corridor boxes, and dumping them
verbatim reached 78 MB at 4.7k flights (~400 MB at 26k) — too big to archive or open comfortably.
Three encodings stack to **~140x smaller** (78 MB → 0.55 MB) with no loss of visual fidelity:
corridor boxes are **not stored at all** but rebuilt in the browser from the path (they are exactly
the swept centerline, which `viz_html._rebuildable` verifies per flight — a planner that reserves
anything else keeps explicit polygons); coordinates are quantised to decimetres or finer and delta-encoded;
and the result is gzipped into one base64 string, inflated on load via `DecompressionStream`
(Chrome 80+ / Safari 16.4+ / Firefox 113+). Loading got ~14x faster as a side effect, since the
browser no longer parses tens of MB of JavaScript source.

## Metrics

Per flight: ground delay, air hold, air detour, altitude change, cost, **stretch** (flown ÷
straight), **total delay** (hold + loiter + detour-time, excluding the mandatory climb), reserved
**volume-seconds**, and **planner solve time**. Aggregates roll these up plus denial rate (with
budget-vs-search-artifact split), throughput, and **airspace utilization** (reserved volume-seconds
÷ region × realized simulation duration — the free-space analog of the hex repo's occupancy).

**Steady-state window.** A run's airborne density is a trapezoid — it ramps up from an empty sky,
plateaus, then ramps down as the last flights (and post-horizon returns) land. Metrics over the whole
run are diluted by the low-density ramps, so `metrics.steady_state_window(result)` finds the plateau
(the widest interval where density ≥ `frac`×peak, `frac=0.9` default) and every surface reports both
the whole-run number **and** its steady-state twin measured over that window: `summary.json` carries a
nested `steady_state` block, `index.parquet` gains `steady_*` / `window_*` columns, and the `curve` /
`compare` / `histograms` readouts overlay the two. `--window-frac` tunes the plateau threshold; the
replay spans the **realized operation** — first reservation through last to clear — so post-horizon
return traffic stays visible *and* an early-finishing run no longer scrubs out to `horizon_s`. This
matters because `horizon_s` is a planner envelope, not a schedule: on the density scenarios flights
are filed from t=0 but the first departs ~768 s in and the last lands ~3300 s before the envelope
closes, so anchoring on it left **57% of the slider empty**. This **supersedes** the removed
`clip_returns_to_horizon` demand hack (issue #25): run and preserve the natural demand and its full tail,
while the separate steady-state view measures only the representative plateau.

## Status

- **Done:** 3D geometry + FCL conflict engine, all planners, FCFS sim, metrics, run capture,
  visualization, congestion experiments. Tests green (ASTM invariant enforced).
- **Not yet:** real-geography (lat/lon) projection; per-operator async clocks; the BlueSky tactical
  layer (designed-for behind an execution seam, not built).
