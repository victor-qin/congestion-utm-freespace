#!/bin/bash

#SBATCH -o /home/gridsan/vqin/congestion-utm-freespace/results/supercloud/lead_arms.log-%A-%a
#SBATCH -c 5
#SBATCH --array=0-2

## The scheduling-lead sweep: one arm per array task, all running in parallel.
##
## Wing/Zipline holds at its own 8-minute filing lead; only Amazon's moves. Filing earlier puts a
## flight earlier in the FCFS queue at the same desired departure, so the arms measure what that
## queue position is worth — to Amazon, and to the Wing/Zipline flights it displaces.
##
## TO ADJUST, edit the four lines below (and --array above to match the length of ARMS):
##
##   ARMS   azlead08m / 15m / 30m = Amazon at 8 / 15 / 30 min, Wing/Zipline held at 8; wzlead15m /
##          wzlead30m mirror it (Wing/Zipline files ahead instead). azlead30m and wzlead08m are the
##          same world under two names — run one, not both.
##   WORLD  faa (~5k flights) or future (~27k flights, several times longer).
##   SEED   arms are paired per flight — within a seed every arm gets an identical flight set and
##          identical desired departures — so one seed already gives a clean contrast. Resubmit with
##          another SEED and the same TAG for error bars.
##   TAG    the join key the cross-run readouts filter on.

ARMS=(azlead08m azlead15m azlead30m)
WORLD=faa
SEED=0
TAG=leadarms

module load anaconda/Python-ML-2024b

source activate congestion-utm

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/state/partition1/user/$USER/numba_cache
mkdir -p "$NUMBA_CACHE_DIR"

cd $HOME/congestion-utm-freespace

TASK=${SLURM_ARRAY_TASK_ID:-0}
## --array and ARMS drift because they are edited separately. Checking the array's SIZE — not just this
## task's index — is what catches a too-SHORT array: it leaves no task to notice the missing tail, so it
## would quietly run a partial sweep. Unset outside Slurm, where there is no array to disagree with.
COUNT=${SLURM_ARRAY_TASK_COUNT:-${#ARMS[@]}}
if [ "$COUNT" -ne "${#ARMS[@]}" ] || [ "$TASK" -ge "${#ARMS[@]}" ]; then
  echo "FATAL: --array covers $COUNT task(s) and ARMS has ${#ARMS[@]} entries (this is task $TASK) —" \
       "use --array=0-$((${#ARMS[@]} - 1))" >&2
  exit 1
fi
ARM=${ARMS[$TASK]}

## Tasks start together and take about the same time, so they finish together and all append to the
## shared index at once. That append is locked now, but staggering keeps them off the lock entirely
## (and spreads the numba cache warm-up).
sleep $((TASK * 30))

## --mode sequential: exact parallel LOSES to sequential on the density scenarios, and relaxed mode
##   is result-affecting, which would confound the arms.
## --return-anchor nominal: the realized anchor ties a return's departure to its outbound's realized
##   delay — exactly what differs between arms — so returns would stop being comparable.
python -m experiments.run \
  --scenario density_${WORLD}_wing_zipline_amazon_${ARM} \
  --seed $SEED --mode sequential --return-anchor nominal \
  --tag $TAG --no-progress

## A lost index row is silent — the readout below would just report one arm fewer. Rebuilding from each
## run's own copy means whichever task finishes LAST leaves a complete index, whatever the shared
## filesystem did with the lock. Idempotent; never removes anyone else's runs.
python -c "from freespace_sim import runs; print(len(runs.rebuild_index()), 'runs in index.parquet')"

## Read out per operator once the array finishes. compare.py is run-wide only; the per-USS split this
## experiment is about lives in each run folder's per_uss.parquet:
##
##   python -c "
##   import pandas as pd; from pathlib import Path; from freespace_sim import runs
##   idx = runs.load_index(); idx = idx[idx.tag=='leadarms']
##   df = pd.concat([pd.read_parquet(Path(p)/'per_uss.parquet').assign(scenario=s)
##                   for p, s in zip(idx.path, idx.scenario)])
##   print(df.groupby(['scenario','uss_id'])[['mean_total_delay_s','denial_rate']].mean().round(2))"
