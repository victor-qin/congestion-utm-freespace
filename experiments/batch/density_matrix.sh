#!/usr/bin/env bash
# Run the four canonical density scenarios at their declared rates and durations, then compare them.
#
# Usage:   bash experiments/batch/density_matrix.sh [suffix] [seed ...]
# Example: bash experiments/batch/density_matrix.sh paper 0 1 2
#          WORKERS=8 PY=python bash experiments/batch/density_matrix.sh paper 0
#
# WORKERS is pinned rather than left to ParallelConfig's min(8, cores-2) default, because that
# default is chosen from core count alone and the binding constraint here is MEMORY: run_parallel
# spawns, so peak RSS is ~(workers+1) x per-process — each worker holds a full ledger + hex-occupancy
# replica. density_future_wing_zipline (25,902 flights) measures ~55 GB at 4 workers and ~99 GB at 8,
# so on a 48-core node the default silently selects the 99 GB configuration. 4 is also the exact-mode
# speedup sweet spot, not a memory compromise. See experiments/batch/supercloud_run.sh.
#
# PY overrides the interpreter for machines without uv (e.g. a conda env on an HPC login node).
set -euo pipefail

WORKERS="${WORKERS:-4}"
PY="${PY:-}"
if [[ -n "$PY" ]]; then
  RUN=("$PY")
else
  RUN=(uv run python)
fi

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

echo "EXECUTE density matrix tag=$TAG scenarios=[${SCENARIOS[*]}] seeds=[${SEEDS[*]}] workers=$WORKERS"
for SCENARIO in "${SCENARIOS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    FOLDER=$("${RUN[@]}" -m experiments.run \
      --scenario "$SCENARIO" --seed "$SEED" --tag "$TAG" --mode exact \
      --workers "$WORKERS" --no-progress | tail -1)
    "${RUN[@]}" -m experiments.readouts.uss_breakdown "$FOLDER"
  done
done

echo "READ OUT density comparison (no re-simulation, grouped by scenario)"
"${RUN[@]}" -m experiments.readouts.compare --tag "$TAG" --by scenario
echo "done: tag=$TAG"
