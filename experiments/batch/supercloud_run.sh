#!/usr/bin/env bash
# One density run on MIT SuperCloud (xeon-p8: 48 cores / 192 GB, 4 GB per core requested).
#
# One-time setup on the LOGIN node (compute nodes have no internet).
#
# SuperCloud's default advice is `pip install --user` on top of a module, NOT a conda env
# (conda puts everything in $HOME and slows imports on the shared FS). That doesn't apply
# here: pyproject requires Python >=3.12 and the newest module (Python-ML-2024b) is 3.11,
# so no module's Python can install this project — this is their documented "when you need
# complex dependencies" conda case. Do NOT fall back to --user: ~/.local is on sys.path for
# EVERY Python of that version, so a numpy 2.x there shadows the module's numpy 1.x and
# breaks its prebuilt numba (SystemError in numba/np/ufunc/_internal).
#
#   rm -rf $HOME/.local/lib/python3.9/site-packages     # only if already contaminated
#   module load anaconda/Python-ML-2024b
#   mamba create -y -n py312 python=3.12
#   source activate py312
#   pip install -e "$HOME/congestion-utm-freespace[compiled]"
#   python -c "import numba, numpy, pulp, fcl; print(numba.__version__, numpy.__version__, pulp.__version__)"
#   # expect: 0.66.0 2.4.6 3.3.2
#   # `mamba create` may print "Could not set lock" warnings — the env is still created.
#
# Submit:  LLsub experiments/batch/supercloud_run.sh -s 24
# Check:   sacct -j $JOBID -o JobID,MaxRSS,Elapsed --units=G
#
# -s picks BOTH cores and memory (4 GB/core). Peak RSS is ~(workers+1) x per-process,
# and per-process is ~2.7 GB at 4680 flights / ~11 GB at 25902 (ledger + hex occupancy,
# neither shared: run_parallel spawns, so every worker holds a full replica).
#   density_faa_wing_zipline     4680 flights  --workers 4  ~13 GB  -> -s 24 is ample
#   density_future_wing_zipline 25902 flights  --workers 4  ~55 GB  -> -s 24 (96 GB)
#                                              --workers 8  ~99 GB  -> LLsub -i full
set -euo pipefail

SCENARIO="${SCENARIO:-density_faa_wing_zipline}"
SEED="${SEED:-0}"
TAG="${TAG:-calib}"
MODE="${MODE:-exact}"
# 4 is the exact-mode sweet spot, not a memory compromise: speedup peaks at ~4 and
# regresses past it (see ParallelConfig, freespace_sim/parallel.py). Use 8 for relaxed.
WORKERS="${WORKERS:-4}"

REPO="${REPO:-$HOME/congestion-utm-freespace}"
CONDA_ENV="${CONDA_ENV:-congestion-utm}"

source /etc/profile
module load anaconda/Python-ML-2024b

# Absolute path, NOT `source activate`. Measured on this cluster: activation changes the
# prompt to (congestion-utm) but does NOT prepend the env's bin to PATH, so `python` still
# resolves to the module's 3.9 and the run dies on `from datetime import UTC` (3.11+). The
# interpreter path is unambiguous and needs no shell state.
PY="${PY:-$HOME/.conda/envs/$CONDA_ENV/bin/python}"

# Each of the WORKERS+1 processes would otherwise start 48 BLAS/OpenMP threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# The parent warms the JIT so spawned workers load from disk instead of racing to
# compile; that needs a writable cache, and node-local scratch is the fast one.
export NUMBA_CACHE_DIR="/state/partition1/user/$USER/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

cd "$REPO"

# Gate on the interpreter BEFORE burning hours: a 3.9 python dies on `from datetime import
# UTC`, and a missing numba makes astar.py log a WARNING and silently fall back to the
# pure-Python A* (~5-7x slower) — a failure you would otherwise discover 6 hours in.
"$PY" -c "import sys, numba, fcl; assert sys.version_info >= (3, 12), sys.version" || {
  echo "FATAL: $PY is not a working 3.12 env — see the setup block at the top" >&2
  exit 1
}
echo "python: $PY $("$PY" -V 2>&1)"

FOLDER=$("$PY" -m experiments.run \
  --scenario "$SCENARIO" --seed "$SEED" --tag "$TAG" \
  --mode "$MODE" --workers "$WORKERS" --no-progress | tail -1)

echo "run folder: $FOLDER"
