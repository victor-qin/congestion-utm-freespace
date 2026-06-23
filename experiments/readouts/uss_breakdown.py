"""READ OUT (per-run) — per-USS breakdown of a saved run: a printed table + a bar chart.

Reads the ``per_uss.parquet`` slice the execute step wrote (one row per operator) and shows how the
operators fared against each other under FCFS — counts, denial, delay, flight length, airspace share.
Bars are colored with the same per-USS palette the replay/snapshot use. No re-simulation.

    uv run python -m experiments.readouts.uss_breakdown results/<folder>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from freespace_sim import viz  # noqa: E402

_COLS = ["uss_id", "n_requests", "n_accepted", "denial_rate", "mean_total_delay_s",
         "mean_air_detour_m", "mean_straight_line_m", "airspace_utilization", "share_of_accepted"]


def main() -> None:
    p = argparse.ArgumentParser(description="Per-USS breakdown table + bar chart from a saved run.")
    p.add_argument("folder", help="a results/ run folder written by experiments.run")
    args = p.parse_args()

    folder = Path(args.folder)
    pu = pd.read_parquet(folder / "per_uss.parquet")
    print(pu[[c for c in _COLS if c in pu.columns]].to_string(index=False))

    hues = viz.uss_hues(list(pu["uss_id"]))
    colors = [viz.uss_swatch_hex(u, hues) for u in pu["uss_id"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax1.bar(pu["uss_id"], pu["n_accepted"], color=colors, edgecolor="white")
    ax1.set_title("accepted flights by USS"); ax1.set_ylabel("accepted")
    ax2.bar(pu["uss_id"], pu["mean_total_delay_s"], color=colors, edgecolor="white")
    ax2.set_title("mean total delay by USS"); ax2.set_ylabel("seconds")
    out = folder / "uss_breakdown.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
