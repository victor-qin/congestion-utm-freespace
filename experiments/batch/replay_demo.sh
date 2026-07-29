#!/usr/bin/env bash
# Single-run demo = pure-shell composition: EXECUTE one scenario (capture the folder path off stdout),
# then READ OUT the per-run artifacts for it (replay + figures + per-USS breakdown). No re-simulation
# in the readout steps — they all load the one saved folder.
#
# Usage:   bash experiments/batch/replay_demo.sh [scenario] [planner] [lambda] [horizon]
# Example: bash experiments/batch/replay_demo.sh dallas_hub_2uss astar_shortcut
#          bash experiments/batch/replay_demo.sh dallas_hub_2uss astar 34500 1800   # the dense demo
#
# lambda/horizon are OPT-IN: unset, the scenario runs at its own declared rate and envelope.
# They used to default to 34500/1800 — a 57x rate override even on dallas_hub_2uss (600/h, 3600 s) —
# which was harmless demo tuning until --lam became load-bearing: it now rescales lam_per_uss on any
# scenario that declares per-USS rates, so those defaults silently 7x'd density_faa_wing_zipline and
# clamped its 7200 s envelope to 1800 s (== its demand window, so the horizon guard let it pass, and
# every late departure then fell through to the slow box-guard path).
set -euo pipefail

SCENARIO="${1:-dallas_hub_2uss}"
PLANNER="${2:-astar}"     # astar (not astar_shortcut): refiners are incompatible with always-active walls
LAMBDA="${3:-}"
HORIZON="${4:-}"

OVERRIDES=()
[[ -n "$LAMBDA" ]] && OVERRIDES+=(--lam "$LAMBDA")
[[ -n "$HORIZON" ]] && OVERRIDES+=(--horizon "$HORIZON")

echo "EXECUTE demo scenario=$SCENARIO planner=$PLANNER overrides=[${OVERRIDES[*]-none}] tag=demo"
FOLDER=$(uv run python -m experiments.run --scenario "$SCENARIO" --planner "$PLANNER" \
  ${OVERRIDES[@]+"${OVERRIDES[@]}"} --tag demo --no-progress | tail -1)
echo "EXECUTE → $FOLDER"

uv run python -m experiments.readouts.replay        "$FOLDER"  # return-flight tail is shown by default
uv run python -m experiments.readouts.figures       "$FOLDER" --no-3d
uv run python -m experiments.readouts.uss_breakdown "$FOLDER"
uv run python -m experiments.readouts.histograms    "$FOLDER"
echo "artifacts in $FOLDER (open replay.html to scrub)"
