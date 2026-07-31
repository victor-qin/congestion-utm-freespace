#!/usr/bin/env bash
# Run the four canonical density scenarios at their declared rates and durations, then compare them.
#
# Usage:   bash experiments/batch/density_matrix.sh [suffix] [seed ...]
# Example: bash experiments/batch/density_matrix.sh paper 0 1 2
set -euo pipefail

SUFFIX="${1:-run}"
if (($# > 0)); then
  shift
fi
SEEDS=("$@")
if ((${#SEEDS[@]} == 0)); then
  SEEDS=(0)
fi

TAG="density_${SUFFIX}"
SCENARIOS=(
  density_faa_wing_zipline
  density_future_wing_zipline
  density_faa_wing_zipline_amazon
  density_future_wing_zipline_amazon
)

echo "EXECUTE density matrix tag=$TAG scenarios=[${SCENARIOS[*]}] seeds=[${SEEDS[*]}]"
for SCENARIO in "${SCENARIOS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    FOLDER=$(uv run python -m experiments.run \
      --scenario "$SCENARIO" --seed "$SEED" --tag "$TAG" --mode exact --no-progress | tail -1)
    uv run python -m experiments.readouts.uss_breakdown "$FOLDER"
  done
done

echo "READ OUT density comparison (no re-simulation, grouped by scenario)"
uv run python -m experiments.readouts.compare --tag "$TAG" --by scenario
echo "done: tag=$TAG"
