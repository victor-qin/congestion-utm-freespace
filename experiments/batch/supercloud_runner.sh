#!/bin/bash

#SBATCH -o /home/gridsan/vqin/congestion-utm-freespace/results/supercloud/supercloud_runner.log-%j
#SBATCH -c 20

module load anaconda/Python-ML-2024b

source activate congestion-utm

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/state/partition1/user/$USER/numba_cache
mkdir -p "$NUMBA_CACHE_DIR"

cd $HOME/congestion-utm-freespace
## Sequential
python -m experiments.run \
  --scenario density_future_wing_zipline --planner astar\
  --seed 0  --mode sequential --tag density_future_wing_zipline_1to1-5cost

## Parllel
# python -m experiments.run \
#   --scenario density_future_wing_zipline_amazon --planner astar_shortcut\
#   --seed 0  --mode exact --workers 4 --parallel-window 16 --tag density_future_wing_zipline_amazon
