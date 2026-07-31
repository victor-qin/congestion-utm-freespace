#!/usr/bin/env bash
# Characterize SuperCloud for this simulator: how many cores, and how much memory.
#
# Emits one CSV row per run: scenario,lam,workers,flights,wall_s,peak_rss_mb,mean_solve_s
#
#   Sweep A (workers) — the speedup knee and the per-worker memory slope. The <=8 cap in
#     ParallelConfig was measured on an M1 Ultra (4-core clusters, shared 12 MB L2); a Xeon
#     8260 is 24 cores/socket with 35.75 MB shared L3, so that curve does NOT transfer and
#     the knee here is unknown. Small scenario, so this is minutes per point.
#   Sweep B (flights) — memory vs committed flights at the FUTURE 476-hub geometry, via
#     --lam (which scales per-USS rates proportionally). This is the term that decides
#     whether the full future run fits: extrapolating it from the small scenario alone
#     has already been wrong by ~2x once.
#
# Run under `LLsub -i -s 48` (or `LLsub -i full`) so worker counts up to 24 are not
# core-starved and the memory ceiling is the node's, not a slot's share.
#
#   Sweep Q (quick, ~10 min) — both memory slopes only, at ~2k flights on the future
#     geometry. Memory per worker and memory per flight are measurable at ANY flight count
#     (they are slopes, not levels), so they do not need full runs. What Q canNOT give you
#     is the speedup knee: at 2k flights the JIT warm and process spawn are a large share
#     of wall time, so the timings are not a fair scaling signal. Run A for that.
#
#   bash experiments/batch/supercloud_scaling.sh q      # ~10 min, memory only  <- START HERE
#   bash experiments/batch/supercloud_scaling.sh a      # workers sweep, ~1 h
#   bash experiments/batch/supercloud_scaling.sh b 8    # flights sweep at 8 workers
set -euo pipefail

SWEEP="${1:-a}"
FIXED_W="${2:-8}"
REPO="${REPO:-$HOME/congestion-utm-freespace}"
CONDA_ENV="${CONDA_ENV:-congestion-utm}"
PY="${PY:-$HOME/.conda/envs/$CONDA_ENV/bin/python}"
OUT="${OUT:-$REPO/supercloud_scaling_${SWEEP}.csv}"

# The conda env is self-contained; loading an anaconda module alongside it leaks
# PYTHONPATH/PYTHONHOME at a 3.10 stdlib into the 3.12 interpreter.
unset PYTHONHOME PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="/state/partition1/user/$USER/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

"$PY" -c "import sys, numba, fcl; assert sys.version_info >= (3,12), sys.version" || {
  echo "FATAL: $PY is not a working 3.12 env" >&2; exit 1; }

cd "$REPO"

# Sum RSS over the whole process tree. `/usr/bin/time -v` and `sacct MaxRSS` both report the
# max of any SINGLE process, which is useless here: run_parallel uses mp spawn, so the N+1
# processes are independent (no COW sharing) and the total is what fills the node.
peak_tree_rss_kb() {
  local root=$1 peak=0 pids total
  while kill -0 "$root" 2>/dev/null; do
    pids=$(pgrep -P "$root" 2>/dev/null | tr '\n' ' ')
    total=$(ps -o rss= -p "$root" $pids 2>/dev/null | awk '{s+=$1} END{print s+0}')
    ((total > peak)) && peak=$total
    sleep 2
  done
  echo "$peak"
}

measure() {                       # measure <scenario> <lam|-> <workers>
  local scenario=$1 lam=$2 workers=$3 lam_args=() t0 t1 folder rss_kb
  [[ "$lam" != "-" ]] && lam_args=(--lam "$lam")

  t0=$SECONDS
  "$PY" -m experiments.run --scenario "$scenario" "${lam_args[@]}" \
    --seed 0 --tag "scale_${SWEEP}" --mode exact --workers "$workers" --no-progress \
    > /tmp/scale_folder.$$ 2>/tmp/scale_log.$$ &
  local pid=$!
  rss_kb=$(peak_tree_rss_kb "$pid")
  wait "$pid" || { echo "RUN FAILED (workers=$workers lam=$lam); see /tmp/scale_log.$$" >&2; return 1; }
  t1=$SECONDS

  folder=$(tail -1 /tmp/scale_folder.$$)
  "$PY" - "$folder" "$scenario" "$lam" "$workers" "$((t1 - t0))" "$rss_kb" <<'EOF' >> "$OUT"
import json, sys
folder, scenario, lam, workers, wall, rss_kb = sys.argv[1:7]
s = json.load(open(f"{folder}/summary.json"))
print(",".join(str(x) for x in (
    scenario, lam, workers, s["n_requests"], wall,
    round(int(rss_kb) / 1024, 1), round(s["mean_solve_time_s"], 4))))
EOF
  tail -1 "$OUT"
}

echo "scenario,lam,workers,flights,wall_s,peak_rss_mb,mean_solve_s" > "$OUT"

if [[ "$SWEEP" == "q" ]]; then
  # One throwaway run first: a cold numba cache costs minutes of JIT that would otherwise
  # land entirely on the first measured point and look like memory/time noise.
  echo "== warming numba cache (discarded) =="
  "$PY" -m experiments.run --scenario density_future_wing_zipline --lam 400 \
    --seed 0 --tag scale_warm --mode exact --workers 2 --no-progress >/dev/null 2>&1 || true

  # Memory vs workers, at the future 476-hub geometry (the layout that actually matters).
  for W in 2 4 8 16; do
    echo "== quick: lam=2000 workers=$W =="
    measure density_future_wing_zipline 2000 "$W" || true
  done
  # One more flight count at fixed W: with the lam=2000/W=8 row above, this gives the
  # per-flight slope -- the term that decides whether the full 25,902-flight run fits.
  echo "== quick: lam=6000 workers=8 =="
  measure density_future_wing_zipline 6000 8 || true

elif [[ "$SWEEP" == "a" ]]; then
  # Fixed small scenario; only the worker count moves. exact mode => byte-identical results
  # at every W, so any delta is pure performance and correctness is independently checkable.
  for W in 1 2 4 8 16 24; do
    echo "== sweep A: workers=$W =="
    measure density_faa_wing_zipline - "$W" || true
  done
else
  # Fixed workers; flight count moves at the FUTURE 476-hub layout. 27322 is the real rate.
  for LAM in 5000 10000 20000 27322; do
    echo "== sweep B: lam=$LAM workers=$FIXED_W =="
    measure density_future_wing_zipline "$LAM" "$FIXED_W" || true
  done
fi

echo
echo "wrote $OUT"
column -s, -t < "$OUT"
