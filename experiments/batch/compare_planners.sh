#!/usr/bin/env bash
# Planner comparison = pure-shell composition: EXECUTE one scenario across several planners (loop the
# run box), then READ OUT a comparison table from the shared index, filtered by --tag.
#
# Usage:   bash experiments/batch/compare_planners.sh [suffix] [scenario]
# Example: bash experiments/batch/compare_planners.sh hub dallas_hub_2uss
set -euo pipefail

SUFFIX="${1:-run}"
SCENARIO="${2:-metro_2uss}"
TAG="cmp_${SUFFIX}"
PLANNERS=(straight astar astar_shortcut)
LAM=240
HORIZON=900
SEED=0

echo "EXECUTE planner comparison tag=$TAG scenario=$SCENARIO planners=[${PLANNERS[*]}] lam=$LAM horizon=${HORIZON}s seed=$SEED"
for P in "${PLANNERS[@]}"; do
  uv run python -m experiments.run --scenario "$SCENARIO" --planner "$P" \
    --lam "$LAM" --horizon "$HORIZON" --seed "$SEED" --tag "$TAG" --no-progress >/dev/null
done

echo "READ OUT comparison table (no re-sim, from the index)"
uv run python -m experiments.readouts.compare --tag "$TAG"
echo "done: tag=$TAG"
