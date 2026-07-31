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
# Submit:  LLsub experiments/batch/supercloud_run.sh -s 10
# Check:   sacct -j $JOBID -o JobID,AllocCPUS,MaxRSS,Elapsed --units=G
#
# Deliberately NO #SBATCH directives: LLsub ignores every command-line argument if the script
# contains any, so `-s N` would silently do nothing and the core/memory request would be
# frozen in the file. Keep `-s` on the command line so one script serves every scenario.
# (Slurm also does no ~ expansion in `#SBATCH -o` — it would create a literal `~` directory.)
#
# MEASURED (sacct MaxRSS on the batch step = whole-tree total, cross-checked against `ps` on
# an M1): density_faa_wing_zipline, 8 workers, 9 processes -> 28.19 GB, i.e. 3.13 GB/process
#   fixed    ~0.93 GB/process  (g_pack+heap 896 MiB at max_expansions=6e6, pools ~50 MiB)
#   variable ~0.47 GB/process per 1000 flights
#
# -s picks BOTH cores and memory (4 GB/core on xeon-p8). You need workers+1 cores for the
# processes, but memory usually binds first on the big scenarios. Nothing is shared between
# processes -- run_parallel uses mp spawn, so every worker holds a full ledger replica.
#
#   scenario / workers                 processes   peak RAM        -s
#   density_faa_wing_zipline      w=8      9        28 GB (meas.)   10   <- cores bind
#   density_future_wing_zipline   w=4      5       ~66 GB (extrap)  17   <- memory binds
#   density_future_wing_zipline   w=8      9      ~118 GB (extrap)  30   <- memory binds
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

# NO `module load` here: the module system is not initialised in a Slurm batch shell
# (`module: command not found`), and a conda env needs no module at runtime anyway. Loading
# one also leaks PYTHONPATH/PYTHONHOME at a 3.10 stdlib into the 3.12 interpreter.
unset PYTHONHOME PYTHONPATH

# Absolute path, NOT `source activate`. Measured on this cluster: activation changes the
# prompt to (congestion-utm) but does NOT prepend the env's bin to PATH, so `python` still
# resolves to the module's 3.9/3.10 and the run dies on `from datetime import UTC` (3.11+).
# It only appeared to work under sbatch because --export=ALL inherited a login shell that
# already had conda on PATH — not something a batch job should depend on.
PY="${PY:-$HOME/.conda/envs/$CONDA_ENV/bin/python}"

# Each of the WORKERS+1 processes would otherwise start 48 BLAS/OpenMP threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# The parent warms the JIT so spawned workers load from disk instead of racing to
# compile; that needs a writable cache, and node-local scratch is the fast one.
export NUMBA_CACHE_DIR="/state/partition1/user/$USER/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

cd "$REPO"

# Fail loudly here rather than 6 hours in on the pure-Python fallback: without numba
# astar.py logs a WARNING and silently runs ~5-7x slower.
# Gate on the interpreter BEFORE burning hours: a 3.9 python dies on `from datetime import
# UTC`, and a missing numba silently falls back to the pure-Python A* (~5-7x slower).
"$PY" -c "import sys, numba, fcl; assert sys.version_info >= (3, 12), sys.version" || {
  echo "FATAL: $PY is not a working 3.12 env — see the setup block at the top" >&2
  exit 1
}
echo "python: $PY $("$PY" -V 2>&1)"

FOLDER=$("$PY" -m experiments.run \
  --scenario "$SCENARIO" --seed "$SEED" --tag "$TAG" \
  --mode "$MODE" --workers "$WORKERS" --no-progress | tail -1)

echo "run folder: $FOLDER"
