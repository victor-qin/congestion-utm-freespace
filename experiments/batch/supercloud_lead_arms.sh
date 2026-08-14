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
##   ARMS   azlead08m / azlead15m / azlead30m = Amazon at 8 / 15 / 30 min, Wing/Zipline at 8.
##          The mirrored sweep is wzlead15m / wzlead30m (Wing/Zipline files further ahead instead).
##          Note azlead30m and wzlead08m are both operators at their defaults — the SAME world under
##          two names — so run one of them, not both.
##   WORLD  faa (~5k flights) or future (~27k flights, several times longer).
##   SEED   the arms are paired per flight — within a seed every arm has an identical flight set and
##          identical desired departures — so one seed already gives a clean contrast. Add seeds for
##          error bars by submitting again with a different SEED and the same TAG.
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
## A short --array silently drops the tail of ARMS, so say so rather than run a partial sweep.
if [ "$TASK" -ge "${#ARMS[@]}" ]; then
  echo "FATAL: array task $TASK but ARMS has ${#ARMS[@]} entries — use --array=0-$((${#ARMS[@]} - 1))" >&2
  exit 1
fi
ARM=${ARMS[$TASK]}

## Array tasks start together and take about the same time, so they finish together too — and
## save_run's index append is an unlocked read-modify-write that can lose rows when they do.
sleep $((TASK * 30))

## --mode sequential: exact parallel LOSES to sequential on the density scenarios, and relaxed mode
##   is result-affecting, which would confound the arms.
## --return-anchor nominal: the realized anchor ties a return's departure to its outbound's realized
##   delay — exactly what differs between arms — so returns would stop being comparable.
python -m experiments.run \
  --scenario density_${WORLD}_wing_zipline_amazon_${ARM} \
  --seed $SEED --mode sequential --return-anchor nominal \
  --tag $TAG --no-progress

## Read out per operator once the array finishes. compare.py is run-wide only; the per-USS split this
## experiment is about lives in each run folder's per_uss.parquet:
##
##   python -c "
##   import pandas as pd; from pathlib import Path; from freespace_sim import runs
##   idx = runs.load_index(); idx = idx[idx.tag=='leadarms']
##   df = pd.concat([pd.read_parquet(Path(p)/'per_uss.parquet').assign(scenario=s)
##                   for p, s in zip(idx.path, idx.scenario)])
##   print(df.groupby(['scenario','uss_id'])[['mean_total_delay_s','denial_rate']].mean().round(2))"
