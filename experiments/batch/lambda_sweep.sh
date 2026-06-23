#!/usr/bin/env bash
# Congestion curve = pure-shell composition: EXECUTE a λ×seed sweep over a scenario by looping the
# run box (one process per point), then READ OUT the curve from the shared index. The --tag joins
# this batch's runs so the readout selects exactly them.
#
# Usage:   bash experiments/batch/lambda_sweep.sh [suffix] [scenario] [planner] [horizon]
# Example: bash experiments/batch/lambda_sweep.sh fullrun dallas_hub_2uss astar_shortcut 3600
#
# Defaults are modest (fast smoke); scale LAMS/SEEDS/HORIZON up for a real curve.
set -euo pipefail

SUFFIX="${1:-run}"
SCENARIO="${2:-metro_2uss}"
PLANNER="${3:-astar}"
HORIZON="${4:-600}"
TAG="lamsweep_${SUFFIX}"
LAMS=(60 240)
SEEDS=(0 1)

SWEEP="results/sweeps/${TAG}"   # shell owns multiplicity; per-run histograms collect here

echo "EXECUTE sweep tag=$TAG scenario=$SCENARIO planner=$PLANNER horizon=${HORIZON}s"
for L in "${LAMS[@]}"; do
  for S in "${SEEDS[@]}"; do
    # capture each run's folder off stdout, then feed it to the per-run histograms readout
    FOLDER=$(uv run python -m experiments.run --scenario "$SCENARIO" --planner "$PLANNER" \
      --lam "$L" --seed "$S" --horizon "$HORIZON" --tag "$TAG" --no-progress | tail -1)
    uv run python -m experiments.readouts.histograms "$FOLDER" --out-dir "$SWEEP"   # per-run, collected
  done
done

echo "READ OUT curve (cross-run trend, from the index the loop populated)"
uv run python -m experiments.readouts.curve --tag "$TAG"          # 2x2 trend panel (index grain)
echo "done: tag=$TAG  · per-run histograms in $SWEEP/ · curve in results/sweeps/$TAG/"
