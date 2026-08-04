#!/usr/bin/env bash
# One density run per Slurm array task — one seed each. Companion to supercloud_run.sh
# (single run); see that file's header for the one-time conda/env setup on the login node.
#
# SUBMIT WITH sbatch, NOT LLsub. LLsub has no job-array mode (`-s N` is slots on ONE job;
# triples sets LLSUB_RANK, which is not an array), and sbatch is the only one that honours
# BOTH --array and command-line core/memory requests.
#
#   mkdir -p $HOME/congestion-utm-freespace/results/supercloud      # -o dir must pre-exist
#   cd $HOME/congestion-utm-freespace
#   sbatch --array=0-4 -c 10 \
#          -o results/supercloud/run-%A_%a.log \
#          experiments/batch/supercloud_array.sh \
#            --scenario density_faa_wing_zipline --tag w8_win12_relaxed \
#            --mode relaxed --workers 8 --window 12
#
# Array task N runs seed N (or --seeds' Nth entry). Everything after the script path is
# parsed by THIS script; everything before it is Slurm's. Every knob also reads an env var
# of the same name in caps, so `SCENARIO=x sbatch ...` works too (and so does LLsub, which
# cannot forward script arguments).
#
# There are deliberately NO #SBATCH directives in this file. Two reasons:
#   * cores/memory/array/-o then live on the submit line, which is the whole point of (2);
#   * LLsub ignores every command-line argument if a script contains any #SBATCH, so a
#     baked-in `-c 10` silently overrides the `-s 24` you typed. That is what happened to
#     the previous version of this script.
#
# CORES *ARE* MEMORY on xeon-p8: 48 cores / 192 GB = 4 GB per core, and `-c N` grants both.
# Nothing is shared between processes (run_parallel uses mp spawn — every worker holds a
# full ledger replica), so peak RAM is (workers+1) x per-process. MEASURED:
# density_faa_wing_zipline, 8 workers, 9 procs -> 28.19 GB = 3.13 GB/proc
#   fixed ~0.93 GB/proc (g_pack+heap 896 MiB, pools ~50 MiB) + ~0.47 GB/proc per 1k flights
#
#   scenario                      workers  procs  peak RAM       -c
#   density_faa_wing_zipline         4       5     ~16 GB         5    <- procs bind
#   density_faa_wing_zipline         8       9     ~28 GB (meas)  10   <- procs bind
#   density_future_wing_zipline      4       5     ~66 GB         17   <- MEMORY binds
#   density_future_wing_zipline      8       9    ~118 GB         30   <- MEMORY binds
#
# The script warns if -c cannot cover the estimate; it does not guess for you.
set -euo pipefail

# ---------------------------------------------------------------- knobs (flag > env > default)
SCENARIO="${SCENARIO:-density_faa_wing_zipline}"
TAG="${TAG:-}"                      # default derived below, once MODE/WORKERS/WINDOW are known
MODE="${MODE:-sequential}"
WORKERS="${WORKERS:-}"              # parallel only; empty sequential, defaults to 4 after validation
WINDOW="${WINDOW:-}"                # --parallel-window; empty = ParallelConfig default (4 x workers)
SEEDS="${SEEDS:-}"                  # comma list; empty = seed IS the array task id
STAGGER_S="${STAGGER_S:-45}"        # per-task start offset — see "why stagger" below
REPO="${REPO:-$HOME/congestion-utm-freespace}"
CONDA_ENV="${CONDA_ENV:-congestion-utm}"
PASSTHRU=()                         # anything unrecognised goes straight to experiments.run

while (($#)); do
  case "$1" in
    --scenario) SCENARIO="$2"; shift 2 ;;
    --tag)      TAG="$2";      shift 2 ;;
    --mode)     MODE="$2";     shift 2 ;;
    --workers)  WORKERS="$2";  shift 2 ;;
    --window|--parallel-window) WINDOW="$2"; shift 2 ;;
    --seeds)    SEEDS="$2";    shift 2 ;;
    --stagger)  STAGGER_S="$2"; shift 2 ;;
    -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
    *)          PASSTHRU+=("$1"); shift ;;      # e.g. --lam 8000 --telemetry --window-frac 0.8
  esac
done

# Parallelism is explicit: worker/window knobs alongside sequential mode are almost certainly a
# forgotten `--mode exact|relaxed`, so fail before deriving tags or cluster resources from them.
case "$MODE" in
  sequential)
    if [[ -n "$WORKERS" || -n "$WINDOW" ]]; then
      echo "FATAL: --workers/--window require --mode exact or --mode relaxed" >&2
      exit 2
    fi
    ;;
  exact|relaxed)
    WORKERS="${WORKERS:-4}"          # exact sweet spot; pass 8 explicitly for relaxed
    ;;
  *)
    echo "FATAL: --mode must be sequential, exact, or relaxed (got $MODE)" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------- seed <- array task
# SLURM_ARRAY_TASK_ID under sbatch --array; LLSUB_RANK under LLsub triples; 0 when run bare
# (so `bash experiments/batch/supercloud_array.sh --seeds 3` still works off-cluster).
TASK="${SLURM_ARRAY_TASK_ID:-${LLSUB_RANK:-0}}"
if [[ -n "$SEEDS" ]]; then
  IFS=',' read -ra SEED_LIST <<< "$SEEDS"
  # Duplicate seeds are duplicate WORK: two tasks with an identical config compute the same run twice.
  # save_run no longer merges them (the folder is {stamp}_{tag}_s{seed}_{hash} and a collision is
  # suffixed __2 rather than written into), but the second row still lands in the index as a bogus
  # replicate of the first, so a cross-run mean silently double-counts it. Refuse up front.
  if [[ $(printf '%s\n' "${SEED_LIST[@]}" | sort | uniq -d | wc -l) -gt 0 ]]; then
    echo "FATAL: --seeds has duplicates ($SEEDS) — identical configs collide in one run folder" >&2
    exit 1
  fi
  if ((TASK >= ${#SEED_LIST[@]})); then
    echo "FATAL: array task $TASK but only ${#SEED_LIST[@]} seeds in --seeds $SEEDS." >&2
    echo "       Use --array=0-$((${#SEED_LIST[@]} - 1))" >&2
    exit 1                          # loudly, NOT a silent fallback to seed 0
  fi
  # The reverse mismatch is the SILENT one: --array=0-2 with 5 seeds just never runs seeds 3-4
  # and nothing complains. SLURM_ARRAY_TASK_COUNT (documented alongside SLURM_ARRAY_TASK_ID)
  # lets us say so. A warning, not fatal — running a deliberate subset is legitimate.
  if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]] && ((SLURM_ARRAY_TASK_COUNT < ${#SEED_LIST[@]})); then
    echo "WARN: --array has only $SLURM_ARRAY_TASK_COUNT tasks for ${#SEED_LIST[@]} seeds" \
         "— seeds ${SEED_LIST[*]:$SLURM_ARRAY_TASK_COUNT} will NOT run." \
         "Use --array=0-$((${#SEED_LIST[@]} - 1))" >&2
  fi
  # (the opposite mismatch — more tasks than seeds — is already FATAL in the range guard above)
  SEED="${SEED_LIST[$TASK]}"
else
  SEED="$TASK"                       # --array=0-4 -> seeds 0,1,2,3,4
fi

# Default tag records the execution strategy in the folder name. `window` is RESULT-AFFECTING in
# relaxed mode, so parallel tags retain the complete mode/worker/window tuple.
if [[ -z "$TAG" ]]; then
  if [[ "$MODE" == "sequential" ]]; then
    TAG="sequential"
  else
    TAG="${MODE}_w${WORKERS}${WINDOW:+_win${WINDOW}}"
  fi
fi

# ---------------------------------------------------------------- allocation sanity
# getconf, not nproc: nproc is GNU coreutils and absent on macOS, so the off-cluster path died.
CORES="${SLURM_CPUS_PER_TASK:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
if [[ "$MODE" == "sequential" ]]; then
  PROCS=1
  WORKERS_LABEL="n/a"
  WINDOW_LABEL="n/a"
else
  PROCS=$((WORKERS + 1))             # coordinator + workers, all independent (mp spawn)
  WORKERS_LABEL="$WORKERS"
  WINDOW_LABEL="${WINDOW:-default}"
fi
GB=$((CORES * 4))                    # xeon-p8: -c grants 4 GB per core
case "$SCENARIO" in
  *future*) PER_PROC_GB=14 ;;        # ~25.9k flights: 0.93 + 0.47*25.9 extrapolated
  *)        PER_PROC_GB=4  ;;        # ~4.7k flights: 3.13 measured, rounded up
esac
NEED_GB=$((PROCS * PER_PROC_GB))
echo "task=$TASK seed=$SEED scenario=$SCENARIO tag=$TAG mode=$MODE workers=$WORKERS_LABEL window=$WINDOW_LABEL"
echo "alloc: -c $CORES (~${GB} GB) | need: $PROCS procs x ~${PER_PROC_GB} GB = ~${NEED_GB} GB"
# `if`, not `cond && echo`: under `set -e` an AND-list whose condition is FALSE exits non-zero
# and kills the script. A passing sanity check must not abort the run.
if ((CORES < PROCS)); then
  echo "WARN: $PROCS processes on $CORES cores — they will timeshare" >&2
fi
if ((GB < NEED_GB)); then
  echo "WARN: ~${NEED_GB} GB estimated vs ~${GB} GB allocated — OOM risk," \
       "raise to -c $(( (NEED_GB + 3) / 4 ))" >&2
fi

# ---------------------------------------------------------------- environment
# NO `module load`: the module system is not initialised in a batch shell ("module: command
# not found"), a conda env needs no module at runtime, and loading one leaks a 3.10
# PYTHONPATH/PYTHONHOME into the 3.12 interpreter.
unset PYTHONHOME PYTHONPATH
# Absolute path, NOT `source activate`: activation changes the prompt but does not reliably
# prepend the env's bin to PATH in a batch shell, so `python` resolves to the module's 3.9/3.10
# and the run dies on `from datetime import UTC`.
PY="${PY:-$HOME/.conda/envs/$CONDA_ENV/bin/python}"
# Each of the PROCS processes would otherwise spawn 48 BLAS/OpenMP threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# Node-local scratch, deliberately NOT $TMPDIR. SuperCloud sets and exports $TMPDIR to a
# per-job dir under /state/partition1/user/$USER and DELETES IT WHEN THE JOB ENDS — which is
# the opposite of what a JIT cache wants. Pointing NUMBA_CACHE_DIR at $TMPDIR would make every
# job recompile from cold; a sibling path under the same node-local disk survives job exit (if
# the cluster reaps the whole user dir, we simply recompile — the fallback below covers it).
# Overridable so the script also runs off-cluster.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/state/partition1/user/$USER/numba_cache}"
mkdir -p "$NUMBA_CACHE_DIR" 2>/dev/null || {
  echo "WARN: cannot create NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR — JIT recompiles every run" >&2
  unset NUMBA_CACHE_DIR; }

cd "$REPO"

ARGS=(--scenario "$SCENARIO" --seed "$SEED" --tag "$TAG" --mode "$MODE" --no-progress)
if [[ "$MODE" != "sequential" ]]; then
  ARGS+=(--workers "$WORKERS")
  if [[ -n "$WINDOW" ]]; then ARGS+=(--parallel-window "$WINDOW"); fi
fi
ARGS+=("${PASSTHRU[@]+"${PASSTHRU[@]}"}")

echo "exec: $PY -m experiments.run ${ARGS[*]}"
# DRY_RUN=1 prints the resolved command and exits — check the seed<->task mapping for every
# task (`for i in 0 1 2 3 4; do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 bash ... ; done`) before
# submitting an array that would otherwise queue for hours to run the wrong seeds.
if [[ -n "${DRY_RUN:-}" ]]; then exit 0; fi

# Gate BEFORE the stagger sleep, not after: a broken env should fail in seconds, not after
# minutes of sleeping. A 3.9 python dies on `datetime.UTC`, and a missing numba only logs a
# WARNING and runs the pure-Python A* ~5-7x slower.
"$PY" -c "import sys, numba, fcl; assert sys.version_info >= (3, 12), sys.version" || {
  echo "FATAL: $PY is not a working 3.12 env — see supercloud_run.sh's setup block" >&2; exit 1; }

# Why stagger: array tasks start simultaneously and take near-identical time, so they collide
# twice. (1) NUMBA_CACHE_DIR is node-local and SHARED by every task on the node — run_parallel
# warms the JIT in the parent before spawning its own workers, but nothing coordinates ACROSS
# tasks, so cold tasks stampede the compile. (2) save_run's index append is a read-modify-write
# of results/index.parquet (runs.py:293-296) with no lock: concurrent finishers silently LOSE
# rows, and `readouts.compare --tag` reads only that index. Offsetting the starts desynchronises
# both. It REDUCES the index race, it does not eliminate it — verify after the array (see below).
if ((STAGGER_S > 0 && TASK > 0)); then
  echo "stagger: sleeping $((TASK * STAGGER_S))s (JIT cache + index-append desync)"
  sleep $((TASK * STAGGER_S))
fi

FOLDER=$("$PY" -m experiments.run "${ARGS[@]}" | tail -1)
echo "run folder: $FOLDER"

# After the whole array finishes, confirm no index rows were lost to the race, then compare:
#   python -c "from freespace_sim import runs; d=runs.load_index(); \
#              print(d[d.tag=='$TAG'][['seed','n_accepted','mean_total_delay_s','wall_seconds']])"
#   python -m experiments.readouts.compare --tag "$TAG" --by seed
# Expect one row per array task. Missing rows = the index race; the run FOLDERS are intact
# either way (each is self-contained), so nothing is unrecoverable.
