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
PLANNERS=(straight lazy astar astar_shortcut)

echo "EXECUTE planner comparison tag=$TAG scenario=$SCENARIO"
for P in "${PLANNERS[@]}"; do
  uv run python -m experiments.run --scenario "$SCENARIO" --planner "$P" \
    --lam 240 --horizon 900 --seed 0 --tag "$TAG" --no-progress >/dev/null
done

echo "READ OUT comparison table (no re-sim, from the index)"
uv run python -m experiments.readouts.compare --tag "$TAG"
echo "done: tag=$TAG"
