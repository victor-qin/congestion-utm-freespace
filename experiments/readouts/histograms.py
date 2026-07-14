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


def _steady_subset(folder, acc, window_frac):
    """The steady-state-windowed slice of ``acc`` (accepted flights filed in the density plateau), with
    the window stashed in ``.attrs["window"]``. ``None`` when there is no window, no ``t_request``
    column, or the window trims nothing (no plateau ⇒ identical to the whole run, so an overlay would be
    redundant). Default reads the window ``save_run`` stored in ``summary.json``; ``window_frac``
    recomputes it from the reloaded run's reservations (issue #25)."""
    if "t_request" not in acc.columns:
        return None
    win = None
    if window_frac is not None:
        from freespace_sim import metrics, runs
        win = metrics.steady_state_window(runs.load_run(folder), frac=window_frac)
    else:
        sp = folder / "summary.json"
        if sp.exists():
            st = json.loads(sp.read_text()).get("steady_state") or {}
            if "window_lo" in st and "window_hi" in st:
                win = (st["window_lo"], st["window_hi"])
    if win is None:
        return None
    lo, hi = win
    sub = acc[(acc["t_request"] >= lo) & (acc["t_request"] < hi)]
    if not 0 < len(sub) < len(acc):   # no plateau trimmed anything → skip the (redundant) overlay
        return None
    sub = sub.copy()
    sub.attrs["window"] = (lo, hi)
    return sub


def main() -> None:
    p = argparse.ArgumentParser(description="Delay-distribution plots for one saved run (per-run).")
    p.add_argument("folder", help="a results/ run folder written by experiments.run")
    p.add_argument("--out-dir", default=None,
                   help="where to write the PNGs (default: the run folder itself); when collecting "
                        "many runs into one folder, files are prefixed with the run name")
    p.add_argument("--window-frac", type=float, default=None,
                   help="recompute the steady-state window at this plateau threshold (needs the run's "
                        "reservations); default reads the window save_run already stored in summary.json")
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

    # steady-state window (issue #25): overlay the density-plateau distribution on the whole-run one so
    # the leftward ramp-tail bias reads off directly. Default reads the window save_run stored in
    # summary.json; --window-frac recomputes it from the reloaded run's reservations.
    acc_steady = _steady_subset(folder, acc, args.window_frac)
    win_note = ""
    if acc_steady is not None:
        lo, hi = acc_steady.attrs["window"]
        win_note = f"  ·  steady [{lo:.0f},{hi:.0f}]s"

    def ov(col):   # the windowed twin of `col`, or None when no plateau trimmed anything
        return acc_steady[col].dropna() if acc_steady is not None else None

    viz.delay_histogram(acc["total_delay_s"].dropna(), out=out / f"{prefix}delay_hist.png",
                        title=f"Total delay{lam_title}{win_note}", overlay=ov("total_delay_s"))
    viz.delay_pct_histogram(acc["delay_pct"].dropna(), out=out / f"{prefix}delay_pct_hist.png",
                            title=f"Delay % of trip{lam_title}{win_note}", overlay=ov("delay_pct"))
    # trip-time inflation = (straight-line time + delay) / straight-line time (≥ 1, unbounded). Newer
    # runs store it directly; for older flights.parquet derive it from delay_pct (== 100/(100-pct)).
    def _ratio(frame):
        return (frame["trip_time_ratio"] if "trip_time_ratio" in frame.columns
                else 100.0 / (100.0 - frame["delay_pct"])).dropna()
    viz.trip_ratio_histogram(_ratio(acc), out=out / f"{prefix}trip_ratio_hist.png",
                             title=f"Trip-time inflation{lam_title}{win_note}",
                             overlay=_ratio(acc_steady) if acc_steady is not None else None)
    viz.delay_sources(acc, out=out / f"{prefix}delay_sources.png", by=None)
    print(f"wrote {prefix}delay_hist.png + {prefix}delay_pct_hist.png + {prefix}trip_ratio_hist.png "
          f"+ {prefix}delay_sources.png to {out}/  ({len(acc)} accepted flights"
          f"{f', {len(acc_steady)} in steady window' if acc_steady is not None else ''})")


if __name__ == "__main__":
    main()
