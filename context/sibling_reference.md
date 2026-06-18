# Sibling project reference — congestion-demo-real

Path: `/Users/victorqin/Documents/Research/congestion-demo-real`

This free-space project is a standalone sibling of `congestion-demo-real` (the H3-grid +
space-time A* version). We **mirror its conventions** but share no code. Useful references:

## Architecture conventions to mirror
- Four-layer split: airspace/state · demand · reservation (planner/USS/DSS/mechanism/ledger) ·
  execution (tactical placeholder).
- Protocol-based extensibility: `DemandModel`, `Mechanism`, `ExecutionBackend` are swappable.
  We add a `Planner` protocol on top.
- Determinism: seeded RNG, frozen `SimConfig`, run-folder capture (config/env/git + parquet).
- One pytest module per layer.

## Files worth reading for patterns
- `congestion_sim/types.py` — FlightRequest / OperationalIntent / IntentStatus / FlightLog shapes.
- `congestion_sim/config.py` — frozen `SimConfig` with cost knobs (`cost_lateral`, `cost_air_hold`,
  `cost_level_change`) — our cost model generalizes these to continuous metres/seconds.
- `congestion_sim/mechanism.py` — `FCFSMechanism` (we reuse the policy verbatim, new geometry).
- `congestion_sim/ledger.py` — occupancy + head-on bans (our ledger replaces this with FCL
  broadphase over 3D volumes).
- `congestion_sim/viz_html.py` — self-contained HTML scrubber replay UX (play/pause/step/slider +
  event log) — visual reference for our `viz_html.py`.
- `congestion_sim/viz_html_3d.py` — Cesium.js 3D replay — reference for our true-3D replay.
- `experiments/lambda_sweep.py` — the headline congestion-vs-demand experiment we replicate.

## Recordings / outputs
- `congestion-demo-real/results/` — prior run folders and HTML replays; use as the visual target
  for layout, colour scheme (golden-ratio hue stepping), and scrubber controls.
