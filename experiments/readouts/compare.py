"""READ OUT (cross-run) — a comparison table over a filtered run set, from the shared index.

Reads ``results/index.parquet``, filters to a run set (``--tag`` is the batch join key), and prints a
table grouped by a chosen key (``--by``, default ``planner``) averaged across seeds: acceptance,
denial, delay, detour, utilization, solve time, wall time, verified. The free-space analog of the old
``compare_planners`` print loop — but it reads persisted runs, so it never re-simulates.

    bash experiments/batch/compare_planners.sh mycmp     # executes the runs (tag=cmp_mycmp)
    uv run python -m experiments.readouts.compare --tag cmp_mycmp
    uv run python -m experiments.readouts.compare --tag cmp_mycmp --csv results/cmp.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from freespace_sim import runs

_AGG = {
    "n_accepted": "mean", "n_denied": "mean", "denial_rate": "mean",
    "mean_total_delay_s": "mean", "mean_air_detour_m": "mean", "mean_cost": "mean",
    "airspace_utilization": "mean", "mean_solve_time_s": "mean", "wall_seconds": "mean",
    "verified": "all",
    # steady-state twin (issue #25) — present on runs saved after the window feature landed; the
    # `if k in df.columns` filter below silently drops them for older index rows.
    "steady_mean_total_delay_s": "mean", "steady_p95_total_delay_s": "mean",
    "steady_throughput_per_h": "mean", "steady_denial_rate": "mean",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Comparison table over a filtered run set (cross-run).")
    p.add_argument("--tag", default=None, help="filter to a batch's runs (the join key)")
    p.add_argument("--scenario", default=None)
    p.add_argument("--by", default="planner", help="grouping key (default: planner)")
    p.add_argument("--csv", default=None, help="also write the table to this CSV path")
    p.add_argument("--root", default="results")
    args = p.parse_args()

    idx = runs.load_index(args.root)
    if idx.empty:
        raise SystemExit(f"no index at {args.root}/index.parquet — run some scenarios first")
    df = idx
    if args.tag is not None:
        df = df[df["tag"] == args.tag]
    if args.scenario is not None:
        df = df[df["scenario"] == args.scenario]
    if df.empty:
        raise SystemExit("no runs match the given --tag/--scenario filter")

    cols = {k: v for k, v in _AGG.items() if k in df.columns}
    table = df.groupby(args.by).agg(cols)
    table.insert(0, "n_runs", df.groupby(args.by).size())
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(table.to_string())
    # persist alongside the set's other cross-run artifacts (sweeps/<label>/), plus optional --csv
    label = args.tag or args.scenario or "all"
    csv_path = Path(args.csv) if args.csv else runs.sweep_dir(label, args.root) / "compare.csv"
    table.to_csv(csv_path)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
