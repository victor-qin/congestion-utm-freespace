---
pretty_name: Free-space UTM strategic deconfliction — simulation runs
tags:
  - uas-traffic-management
  - utm
  - strategic-deconfliction
  - astm-f3548-21
  - drone-delivery
  - air-traffic
  - path-planning
  - simulation
  - congestion
size_categories:
  - n<1K
---

# Free-space UTM simulation runs

Archived output of the `freespace-sim` strategic-deconfliction simulator: first-come-first-served
flight-plan submission into a shared free-space volume, ASTM F3548-21 style, with a 3D hex-lattice
airspace and a reservation ledger. Each run is one full simulation — every flight's plan, the
resulting trajectories, the reservation ledger, and an interactive 3D replay.

This repo is an **artifact archive, not a benchmark**. Runs come from different points in the
simulator's development, and the sections below say exactly which ones are comparable to which.

## Layout

    results/<UTC-timestamp>_<tag>_<config-hash>/

The `config-hash` is a digest of the effective `SimConfig`, so two folders sharing a hash were run
under identical configuration. The timestamp is the run's start, in UTC.

### What each run folder contains

| file | what it is |
|---|---|
| `summary.json` | headline metrics — delay, detour, denial rate, solve time |
| `config.json` | the full effective `SimConfig` (the thing the hash digests) |
| `scenario_spec.json` / `scenario.parquet` | the demand: hubs, lanes, per-flight requests |
| `experiment.json` | invocation record — CLI args, scenario description, tag |
| `git.json` | commit the simulator was at, and whether the tree was dirty |
| `env.json` | Python / platform / library versions |
| `flights.parquet` | one row per flight: accepted or denied, delay decomposition, cost terms |
| `trajectories.parquet` | sampled 4D positions for every accepted flight |
| `reservations.parquet` | the volumes each flight reserved in the ledger |
| `ledger_end.parquet` | final ledger state |
| `per_uss.parquet` | metrics grouped by USS operator |
| `replay.html` | self-contained interactive 3D replay (no server, no network) |
| `*_hist.png`, `delay_sources.png` | delay / trip-ratio histograms and the delay-source breakdown |

The 2026-07-21 runs predate the histogram and replay outputs and carry 11 files rather than 17.

## Reading the numbers: two cost regimes

The planner minimises a weighted sum of ground delay, lateral flight, hover, and altitude change.
**How those weights are denominated changed on 2026-07-28**, and it changes what the planner
chooses — so headline delay is only comparable *within* a campaign, never across.

| regime | weights stored as | effect |
|---|---|---|
| **per-metre** (2026-07-21 runs) | two weights per second, two per metre | one hex of lateral detour cost ≈ 360 s of holding; detour and climb were never rational, so ground delay absorbed essentially all congestion |
| **per-second** (2026-07-28 runs) | all four per second — 1 : 3 : 3 : 4 per step | detour and climb become affordable; conflicts resolve as sidesteps instead of long ground holds |

A run's regime is visible in its `config.json`: the per-metre regime has
`cost_air_lateral_per_m`, the per-second regime has `cost_air_lateral_per_s`.


## Campaign 1 — altitude layering under saturation (2026-07-21)

**Question: does adding flight levels relieve congestion, and does the shortcut refiner change the
answer?** A 3x2 matrix over `dallas_full` — one 40 m level, one 100 m level, and a three-level
40/70/100 m ladder, each planned with bare A* and with the shortcut refiner. Metro-scale demand at
saturation: 33,888 requests at lambda = 34,500/h, all accepted.

Simulator at commit `508a442`, **clean tree** — these are reproducible. Per-metre cost regime.

| run | planner | levels | mean delay | ground | air detour | solve time |
|---|---|---|---|---|---|---|
| [`dallas_full_ch25_L40_astar_16554bc0`](results/2026-07-21T20-56-08Z_dallas_full_ch25_L40_astar_16554bc0) | `astar` | 40 m | 168.3 s | 158.3 s | 174 m | 42 min |
| [`dallas_full_ch25_L100_astar_858e32f9`](results/2026-07-21T20-56-17Z_dallas_full_ch25_L100_astar_858e32f9) | `astar` | 100 m | 182.7 s | 172.2 s | 175 m | 42 min |
| [`dallas_full_ch25_L40_shortcut_eff5caae`](results/2026-07-21T21-00-45Z_dallas_full_ch25_L40_shortcut_eff5caae) | `astar_shortcut` | 40 m | 120.0 s | 115.4 s | 41 m | 51 min |
| [`dallas_full_ch25_L100_shortcut_e025b9de`](results/2026-07-21T21-00-45Z_dallas_full_ch25_L100_shortcut_e025b9de) | `astar_shortcut` | 100 m | 135.9 s | 130.8 s | 44 m | 51 min |
| [`dallas_full_ch25_3L_astar_c7c15fef`](results/2026-07-21T21-23-13Z_dallas_full_ch25_3L_astar_c7c15fef) | `astar` | 40/70/100 m | 71.1 s | 62.0 s | 166 m | 71 min |
| [`dallas_full_ch25_3L_shortcut_e26ea45d`](results/2026-07-21T21-34-44Z_dallas_full_ch25_3L_shortcut_e26ea45d) | `astar_shortcut` | 40/70/100 m | 54.2 s | 50.5 s | 26 m | 84 min |

## Campaign 2 — speculative parallel FCFS, exact vs relaxed (2026-07-28)

**Question: what does relaxing strict FCFS ordering cost in solution quality, and does the answer
depend on the planner?** A 2x2 over `density_faa_wing_zipline` — a 182-hub FAA-filing-density
delivery scenario with paired returns — crossing planner (bare A* / shortcut refiner) with the
parallel simulator's two modes:

- **`exact`** — speculative parallel execution that reproduces sequential FCFS exactly. Workers plan
  ahead optimistically; any flight whose read-envelope was invalidated is replanned. Byte-identical
  to the sequential result.
- **`relaxed`** — speculative results are kept even when the envelope moved. Faster, but the accepted
  set can differ from sequential FCFS.

4,680 requests at lambda ≈ 4,854/h, all accepted, single 100 m level, 8 workers, `window_frac=0.9`.
Per-second cost regime.

Since there is one flight level, every flight climbs and descends exactly 200 m: excess altitude is
structurally zero across this campaign and carries no signal.

> [!WARNING]
> All four runs record `git.json: {"commit": "e7fe872", "dirty": true}`. The working tree that
> produced them was not committed, so **they cannot be reproduced from that SHA.** The artifacts are
> internally consistent and mutually comparable; their provenance is not pinned.

| run | planner | mode | mean delay | ground | detour (lattice / traffic) | solve time |
|---|---|---|---|---|---|---|
| [`as_exact`](results/2026-07-28T03-45-33Z_as_exact_da3af64c) | `astar` | `exact` | 68.42 s | 32.21 s | 1070 / 15.7 m | 9 min |
| [`as_relaxed`](results/2026-07-28T03-40-46Z_as_relaxed_da3af64c) | `astar` | `relaxed` | 67.91 s | 31.74 s | 1070 / 14.9 m | 12 min |
| [`sc_exact`](results/2026-07-28T03-35-15Z_sc_exact_8954a3d2) | `astar_shortcut` | `exact` | 29.27 s | 19.52 s | 282 / 9.6 m | 29 min |
| [`sc_relaxed`](results/2026-07-28T03-21-47Z_sc_relaxed_8954a3d2) | `astar_shortcut` | `relaxed` | 27.27 s | 17.53 s | 282 / 9.8 m | 34 min |

## Cross-campaign gotcha

Campaign 2's `air_detour_m` is **not** the same measurement as Campaign 1's. Since the lattice/
deconfliction split, `air_detour_m` decomposes into:

- `lattice_overhead_m` — non-traffic overhead: hex quantisation plus the terminal-airspace fold.
  Roughly constant for a given scenario and bearing distribution; flat in congestion.
- `deconfliction_detour_m` — the part actually caused by other traffic.

In Campaign 2 the lattice term is 97–98% of the raw figure, which is why `as_*` shows ~1,085 m of
"detour" while its traffic-induced component is under 16 m. Campaign 1 has no such split and its
`air_detour_m` should be read as the undecomposed total.

A useful diagnostic when comparing any two runs: **ground delay is a search decision, `air_detour_m`
is a post-hoc measurement.** If two runs agree on ground delay but disagree on detour, what changed
is the metric definition, not the planner.

## Using it

    pip install huggingface_hub

    # one run
    hf download vicqin/congestion-utm-freespace-runs --repo-type dataset \
      --include "results/2026-07-28T03-45-33Z_as_exact_da3af64c/**" --local-dir .

    # everything
    hf download vicqin/congestion-utm-freespace-runs --repo-type dataset --local-dir .

Or, from a checkout of the simulator:

    uv run python -m experiments.push_pull_results pull all \
      --remote vicqin/congestion-utm-freespace-runs

`replay.html` opens directly in a browser — it is fully self-contained.

## Adding a run

    uv run python -m experiments.push_pull_results push results/<run-folder> \
      --remote vicqin/congestion-utm-freespace-runs

then add a row to the campaign table above (or open a new campaign section). The tables are
maintained by hand — nothing regenerates them, so a pushed-but-unlisted run is invisible here.
