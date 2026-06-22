"""READ OUT (per-run) — delay distributions for ONE saved run.

A delay distribution is a property of a *single* run (the spread of its flights' delays), so this is a
per-run readout: it takes one run folder, reads that folder's ``flights.parquet``, and writes the
distribution figures next to it (or into ``--out-dir``):

  delay_hist.png       — histogram of total delay (seconds)
  delay_pct_hist.png   — histogram of delay as % of trip time
  delay_sources.png    — mean delay decomposed into ground-hold / air-hold / detour-time

Multiplicity is the shell's job: to compare across λ or scenarios, the batch script loops `experiments.run`
and feeds each resulting folder here one at a time (see experiments/batch/lambda_sweep.sh). The
cross-run *trend* (the congestion curve over λ) stays in `readouts.curve`, which reads the index the
loop populated. No re-simulation — this plots flights already persisted.

    uv run python -m experiments.readouts.histograms results/<folder>
    uv run python -m experiments.readouts.histograms results/<folder> --out-dir results/sweeps/mysweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from freespace_sim import viz


def main() -> None:
    p = argparse.ArgumentParser(description="Delay-distribution plots for one saved run (per-run).")
    p.add_argument("folder", help="a results/ run folder written by experiments.run")
    p.add_argument("--out-dir", default=None,
                   help="where to write the PNGs (default: the run folder itself); when collecting "
                        "many runs into one folder, files are prefixed with the run name")
    args = p.parse_args()

    folder = Path(args.folder)
    df = pd.read_parquet(folder / "flights.parquet")
    acc = df[df["accepted"]] if "accepted" in df.columns else df
    if acc["total_delay_s"].notna().sum() == 0:
        raise SystemExit(f"{folder}: no accepted flights — nothing to plot")

    out = Path(args.out_dir) if args.out_dir else folder
    out.mkdir(parents=True, exist_ok=True)
    # prefix filenames with the run name only when collecting into a shared --out-dir (avoid clobber)
    prefix = f"{folder.name}_" if args.out_dir else ""

    lam = json.loads((folder / "config.json").read_text()).get("lam_per_hour")
    lam_title = f" — λ={lam:g}/h" if lam is not None else ""

    viz.delay_histogram(acc["total_delay_s"].dropna(), out=out / f"{prefix}delay_hist.png",
                        title=f"Total delay{lam_title}")
    viz.delay_pct_histogram(acc["delay_pct"].dropna(), out=out / f"{prefix}delay_pct_hist.png",
                            title=f"Delay % of trip{lam_title}")
    viz.delay_sources(acc, out=out / f"{prefix}delay_sources.png", by=None)
    print(f"wrote {prefix}delay_hist.png + {prefix}delay_pct_hist.png + {prefix}delay_sources.png "
          f"to {out}/  ({len(acc)} accepted flights)")


if __name__ == "__main__":
    main()
