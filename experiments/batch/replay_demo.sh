#!/usr/bin/env bash
# Single-run demo = pure-shell composition: EXECUTE one scenario (capture the folder path off stdout),
# then READ OUT the per-run artifacts for it (replay + figures + per-USS breakdown). No re-simulation
# in the readout steps — they all load the one saved folder.
#
# Usage:   bash experiments/batch/replay_demo.sh [scenario] [planner]
# Example: bash experiments/batch/replay_demo.sh dallas_hub_2uss astar_shortcut
set -euo pipefail

SCENARIO="${1:-dallas_hub_2uss}"
PLANNER="${2:-astar}"     # astar (not astar_shortcut): refiners are incompatible with always-active walls
LAMBDA="${3:-34500}"
HORIZON="${4:-1800}"

echo "EXECUTE demo scenario=$SCENARIO planner=$PLANNER lam=$LAMBDA horizon=${HORIZON}s tag=demo"
FOLDER=$(uv run python -m experiments.run --scenario "$SCENARIO" --planner "$PLANNER" \
  --lam "$LAMBDA" --horizon "$HORIZON" --tag demo --no-progress | tail -1)
echo "EXECUTE → $FOLDER"

uv run python -m experiments.readouts.replay        "$FOLDER" --no-clip  # demo: show the return-flight tail
uv run python -m experiments.readouts.figures       "$FOLDER" --no-3d
uv run python -m experiments.readouts.uss_breakdown "$FOLDER"
uv run python -m experiments.readouts.histograms    "$FOLDER"
echo "artifacts in $FOLDER (open replay.html to scrub)"
