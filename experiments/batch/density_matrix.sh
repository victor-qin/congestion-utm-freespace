#!/usr/bin/env bash
# Run the four canonical density scenarios at their declared rates and durations, then compare them.
#
# Usage:   bash experiments/batch/density_matrix.sh [suffix] [seed ...]
# Example: bash experiments/batch/density_matrix.sh paper 0 1 2
#          MODE=exact WORKERS=4 PY=python bash experiments/batch/density_matrix.sh paper 0
#
# Sequential is the default. For an explicit parallel run, WORKERS defaults to 4 rather than using
# ParallelConfig's min(8, cores-2) default, because that
# default is chosen from core count alone and the binding constraint here is MEMORY: run_parallel
# spawns, so peak RSS is ~(workers+1) x per-process — each worker holds a full ledger + hex-occupancy
# replica. density_future_wing_zipline (25,902 flights) measures ~55 GB at 4 workers and ~99 GB at 8,
# so on a 48-core node ParallelConfig's core-count default silently selects the 99 GB configuration.
# Four is also the exact-mode speedup sweet spot, not a memory compromise. See
# experiments/batch/supercloud_run.sh.
#
# PY overrides the interpreter for machines without uv (e.g. a conda env on an HPC login node).
set -euo pipefail

MODE="${MODE:-sequential}"
WORKERS="${WORKERS:-}"
case "$MODE" in
  sequential)
    if [[ -n "$WORKERS" ]]; then
      echo "FATAL: WORKERS requires MODE=exact or MODE=relaxed" >&2
      exit 2
    fi
    ;;
  exact|relaxed)
    WORKERS="${WORKERS:-4}"
    ;;
  *)
    echo "FATAL: MODE must be sequential, exact, or relaxed (got $MODE)" >&2
    exit 2
    ;;
esac

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

echo "EXECUTE density matrix tag=$TAG scenarios=[${SCENARIOS[*]}] seeds=[${SEEDS[*]}] mode=$MODE workers=${WORKERS:-n/a}"
for SCENARIO in "${SCENARIOS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    ARGS=(--scenario "$SCENARIO" --seed "$SEED" --tag "$TAG" --mode "$MODE" --no-progress)
    if [[ "$MODE" != "sequential" ]]; then ARGS+=(--workers "$WORKERS"); fi
    FOLDER=$("${RUN[@]}" -m experiments.run "${ARGS[@]}" | tail -1)
    "${RUN[@]}" -m experiments.readouts.uss_breakdown "$FOLDER"
  done
done

echo "READ OUT density comparison (no re-simulation, grouped by scenario)"
"${RUN[@]}" -m experiments.readouts.compare --tag "$TAG" --by scenario
echo "done: tag=$TAG"
