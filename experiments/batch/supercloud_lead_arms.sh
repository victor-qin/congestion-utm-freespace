#!/usr/bin/env bash
# The scheduling-lead sweep: one (arm x seed) run per Slurm array task, all in parallel.
#
# Answers "does filing further ahead buy FCFS priority, and does it push delay onto the other
# operator?" by re-cutting ONE world with one operator's lead replaced. Within a world+seed every
# arm has a byte-identical flight set and byte-identical desired departures, so arms can be
# differenced flight-by-flight — see the README's "Scheduling-lead arms" section.
#
# SUBMIT WITH sbatch, NOT LLsub (LLsub has no array mode, and ignores CLI args if a script contains
# any #SBATCH — which is why there are none in this file; cores/memory/array live on the submit
# line). Companion to supercloud_array.sh, which sweeps SEEDS of a single scenario; this one sweeps
# the ARM MATRIX. See supercloud_run.sh for the one-time conda/env setup on the login node.
#
#   mkdir -p $HOME/congestion-utm-freespace/results/supercloud      # -o dir must pre-exist
#   cd $HOME/congestion-utm-freespace
#
#   # FAA world  — 3 Amazon-lead arms x 1 seed = 3 tasks
#   sbatch --array=0-2 -c 5 -t 12:00:00 -o results/supercloud/lead-%A_%a.log \
#          experiments/batch/supercloud_lead_arms.sh --world faa
#
#   # far-future world — same 3 tasks, much longer
#   sbatch --array=0-2 -c 5 -t 48:00:00 -o results/supercloud/leadf-%A_%a.log \
#          experiments/batch/supercloud_lead_arms.sh --world future
#
# The matrix is ARMS x SEEDS and the array must be sized to match (the script refuses a task past the
# end and warns if the array is short of it). To extend later:
#   --seeds 0,1,2                                    -> 9 tasks, --array=0-8   (error bars)
#   --arms azlead08m,azlead15m,azlead30m,wzlead15m,wzlead30m
#                                                    -> 5 tasks, --array=0-4   (both sweeps)
#
# -c 5 (= 20 GB, since xeon-p8 grants 4 GB per core) for BOTH worlds, deliberately uniform: it clears
# the ~3.3 GB the FAA world needs many times over and still leaves ~6 GB of headroom on the ~13.6 GB
# far-future one, so the same submit line is safe everywhere and there is no per-world footgun. The
# extra cores are bought for MEMORY, not compute — each task is a single sequential process.
#
# Add %N to the array (e.g. --array=0-14%5) to cap how many run at once. Every knob also reads an
# env var of the same name in caps. DRY_RUN=1 prints the resolved command and exits — check the
# task->(arm, seed) mapping before committing a multi-hour array:
#
#   for i in $(seq 0 14); do SLURM_ARRAY_TASK_ID=$i DRY_RUN=1 \
#     bash experiments/batch/supercloud_lead_arms.sh --world faa; done
#
# WHY SEQUENTIAL: exact parallel LOSES to sequential on both density scenarios (the compiled kernel
# makes the serial commit floor dominant), and relaxed mode is result-affecting — it would confound
# the arms. One process per task, so cores are requested for MEMORY (xeon-p8 grants 4 GB per core).
set -euo pipefail

# ---------------------------------------------------------------- the matrix
# DEFAULT: hold Wing/Zipline at its own 8-minute lead and vary only Amazon's. That is the question as
# posed — does Amazon's advance filing buy it priority, and at whose expense — and it keeps the
# competitor fixed, so each arm is a clean contrast rather than a shift of the whole field.
#   azlead08m  wing 8 / amazon  8   Amazon's advantage removed
#   azlead15m  wing 8 / amazon 15
#   azlead30m  wing 8 / amazon 30   the status quo               <- reference arm
#
# The mirrored sweep exists in the registry and is one flag away (--arms), for when the reverse
# question comes up — what if Wing/Zipline filed further ahead instead:
#   wzlead15m  wing 15 / amazon 30
#   wzlead30m  wing 30 / amazon 30  Wing/Zipline caught up
# There is no wzlead08m here: it is both operators at their defaults, i.e. the SAME recipe as
# azlead30m under a second name (the pivot the two sweeps rotate about), so running it as well would
# pay twice for one world.
KNOWN_ARMS=(azlead08m azlead15m azlead30m wzlead08m wzlead15m wzlead30m)
ARMS_CSV="${ARMS:-azlead08m,azlead15m,azlead30m}"

WORLD="${WORLD:-faa}"               # faa | future  (or a full scenario prefix)
# One seed by default. The arms are PAIRED per flight — within a seed every arm has a byte-identical
# flight set and byte-identical desired departures, so the contrast is already clean and seeds are not
# carrying the comparison the way they would in an unpaired design. They guard only against one demand
# draw happening to favour an operator, so add them for error bars once the effect is confirmed.
SEEDS="${SEEDS:-0}"
TAG="${TAG:-leadarms}"              # index join key for the readouts
STAGGER_S="${STAGGER_S:-45}"
REPO="${REPO:-$HOME/congestion-utm-freespace}"
CONDA_ENV="${CONDA_ENV:-congestion-utm}"
PASSTHRU=()                         # anything unrecognised goes straight to experiments.run

while (($#)); do
  case "$1" in
    --world)   WORLD="$2";     shift 2 ;;
    --arms)    ARMS_CSV="$2";  shift 2 ;;
    --seeds)   SEEDS="$2";     shift 2 ;;
    --tag)     TAG="$2";       shift 2 ;;
    --stagger) STAGGER_S="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *)         PASSTHRU+=("$1"); shift ;;       # e.g. --telemetry --window-frac 0.8
  esac
done

case "$WORLD" in
  faa|future) BASE="density_${WORLD}_wing_zipline_amazon" ;;
  *)          BASE="$WORLD" ;;                  # allow a full scenario prefix (e.g. a 3lvl variant)
esac

IFS=',' read -ra ARMS <<< "$ARMS_CSV"
# Validate here, not on the compute node: a typo would otherwise surface as experiments.run's
# argparse rejecting an unknown --scenario after the array has already queued.
for arm in "${ARMS[@]}"; do
  if ! printf '%s\n' "${KNOWN_ARMS[@]}" | grep -qx "$arm"; then
    echo "FATAL: unknown arm '$arm' — pick from: ${KNOWN_ARMS[*]}" >&2
    exit 2
  fi
done
if [[ $(printf '%s\n' "${ARMS[@]}" | sort | uniq -d | wc -l) -gt 0 ]]; then
  echo "FATAL: --arms has duplicates ($ARMS_CSV)" >&2
  exit 2
fi
# azlead30m and wzlead08m are the same recipe; running both burns a slot for a duplicate world.
if printf '%s\n' "${ARMS[@]}" | grep -qx azlead30m && \
   printf '%s\n' "${ARMS[@]}" | grep -qx wzlead08m; then
  echo "WARN: azlead30m and wzlead08m are both operators at their defaults — the SAME recipe under" \
       "two names. Drop one." >&2
fi

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
# Duplicate seeds give two tasks an IDENTICAL SimConfig -> the same _config_hash, and save_run's
# folder is {stamp}_{tag}_{hash} with mkdir(exist_ok=True): same-second finishers would interleave
# parquet into ONE folder. Refuse rather than corrupt.
if [[ $(printf '%s\n' "${SEED_LIST[@]}" | sort | uniq -d | wc -l) -gt 0 ]]; then
  echo "FATAL: --seeds has duplicates ($SEEDS) — identical configs collide in one run folder" >&2
  exit 1
fi

NSEEDS=${#SEED_LIST[@]}
NTASKS=$((${#ARMS[@]} * NSEEDS))

# ---------------------------------------------------------------- task -> (arm, seed)
# SLURM_ARRAY_TASK_ID under sbatch --array; LLSUB_RANK under LLsub; 0 when run bare.
TASK="${SLURM_ARRAY_TASK_ID:-${LLSUB_RANK:-0}}"
if ((TASK >= NTASKS)); then
  echo "FATAL: array task $TASK but the matrix is ${#ARMS[@]} arms x $NSEEDS seeds = $NTASKS." >&2
  echo "       Use --array=0-$((NTASKS - 1))" >&2
  exit 1                            # loudly, NOT a silent fallback to task 0
fi
# The reverse mismatch is the SILENT one: too few tasks just never runs the tail of the matrix.
if [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]] && ((SLURM_ARRAY_TASK_COUNT < NTASKS)); then
  echo "WARN: --array has only $SLURM_ARRAY_TASK_COUNT tasks for $NTASKS (arm, seed) pairs" \
       "— the last $((NTASKS - SLURM_ARRAY_TASK_COUNT)) will NOT run. Use --array=0-$((NTASKS - 1))" >&2
fi
ARM="${ARMS[$((TASK / NSEEDS))]}"
SEED="${SEED_LIST[$((TASK % NSEEDS))]}"
SCENARIO="${BASE}_${ARM}"

# ---------------------------------------------------------------- allocation sanity
CORES="${SLURM_CPUS_PER_TASK:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
GB=$((CORES * 4))                   # xeon-p8: -c grants 4 GB per core
case "$SCENARIO" in                 # sequential ⇒ ONE process; ~0.93 GB + ~0.47 GB per 1k flights
  *future*) NEED_GB=14 ;;           # ~26.9k flights
  *)        NEED_GB=4  ;;           # ~5.1k flights
esac
echo "task=$TASK/$((NTASKS - 1)) arm=$ARM seed=$SEED scenario=$SCENARIO tag=$TAG mode=sequential"
echo "alloc: -c $CORES (~${GB} GB) | need: 1 proc x ~${NEED_GB} GB"
# `if`, not `cond && echo`: under `set -e` a false AND-list exits non-zero and kills the script.
if ((GB < NEED_GB)); then
  echo "WARN: ~${NEED_GB} GB estimated vs ~${GB} GB allocated — OOM risk," \
       "raise to -c $(( (NEED_GB + 3) / 4 ))" >&2
fi

# ---------------------------------------------------------------- environment
# NO `module load`: it is not initialised in a batch shell, a conda env needs no module at runtime,
# and loading one leaks a 3.10 PYTHONPATH/PYTHONHOME into the 3.12 interpreter.
unset PYTHONHOME PYTHONPATH
# Absolute path, NOT `source activate`: activation does not reliably prepend the env's bin to PATH in
# a batch shell, so `python` resolves to the module's 3.9/3.10 and dies on `from datetime import UTC`.
PY="${PY:-$HOME/.conda/envs/$CONDA_ENV/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# Node-local scratch, deliberately NOT $TMPDIR — SuperCloud deletes $TMPDIR when the job ends, which
# is the opposite of what a JIT cache wants. Overridable so this also runs off-cluster.
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/state/partition1/user/$USER/numba_cache}"
mkdir -p "$NUMBA_CACHE_DIR" 2>/dev/null || {
  echo "WARN: cannot create NUMBA_CACHE_DIR=$NUMBA_CACHE_DIR — JIT recompiles every run" >&2
  unset NUMBA_CACHE_DIR; }

cd "$REPO"

# --return-anchor nominal is the default, but it is passed EXPLICITLY because it is load-bearing
# here: the realized anchor ties a return's departure to its outbound's actual arrival, which is
# exactly what differs between arms — returns would stop being paired and only outbound legs would
# difference cleanly.
ARGS=(--scenario "$SCENARIO" --seed "$SEED" --tag "$TAG"
      --mode sequential --return-anchor nominal --no-progress)
ARGS+=("${PASSTHRU[@]+"${PASSTHRU[@]}"}")

echo "exec: $PY -m experiments.run ${ARGS[*]}"
if [[ -n "${DRY_RUN:-}" ]]; then exit 0; fi

# Gate BEFORE the stagger sleep: a broken env should fail in seconds, not after minutes of sleeping.
# A missing numba only logs a WARNING and then runs the pure-Python A* ~5-7x slower.
"$PY" -c "import sys, numba, fcl; assert sys.version_info >= (3, 12), sys.version" || {
  echo "FATAL: $PY is not a working 3.12 env — see supercloud_run.sh's setup block" >&2; exit 1; }

# Why stagger: array tasks start together and take near-identical time, so they collide twice.
# (1) NUMBA_CACHE_DIR is node-local and shared by every task on the node, so cold tasks stampede the
# JIT compile. (2) save_run's index append is an unlocked read-modify-write of results/index.parquet,
# so concurrent finishers silently LOSE rows. Offsetting the starts desynchronises both. It REDUCES
# the index race, it does not eliminate it — verify after the array (see below).
if ((STAGGER_S > 0 && TASK > 0)); then
  echo "stagger: sleeping $((TASK * STAGGER_S))s (JIT cache + index-append desync)"
  sleep $((TASK * STAGGER_S))
fi

FOLDER=$("$PY" -m experiments.run "${ARGS[@]}" | tail -1)
echo "run folder: $FOLDER"

# After the array finishes, confirm no index rows were lost to the race, then read out per operator:
#
#   python -c "from freespace_sim import runs; d=runs.load_index(); d=d[d.tag=='$TAG']; \
#              print(len(d)); print(d[['scenario','seed','mean_total_delay_s','denial_rate']])"
#   python -m experiments.readouts.compare --tag "$TAG" --by scenario
#
# Expect ${#ARMS[@]} x $NSEEDS rows. Missing rows = the index race; the run FOLDERS are intact either
# way (each is self-contained), so nothing is unrecoverable — re-derive the index from them if needed.
#
# NOTE: compare.py reports RUN-WIDE metrics only. The per-operator split this experiment is about
# lives in each folder's per_uss.parquet (uss_id x mean_total_delay_s, denial_rate, ...); there is no
# cross-run per-USS readout yet, so concatenate them:
#
#   python -c "
#   import pandas as pd; from pathlib import Path; from freespace_sim import runs
#   idx = runs.load_index(); idx = idx[idx.tag=='$TAG']
#   df = pd.concat([pd.read_parquet(Path(p)/'per_uss.parquet').assign(scenario=s, seed=d)
#                   for p, s, d in zip(idx.path, idx.scenario, idx.seed)])
#   print(df.groupby(['scenario','uss_id'])[['mean_total_delay_s','denial_rate']].mean().round(2))"
