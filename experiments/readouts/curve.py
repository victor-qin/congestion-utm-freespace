"""READ OUT (cross-run) — the FCFS congestion curve vs demand λ, from the shared index.

Reads ``results/index.parquet`` (every run ever executed), filters to a run set (``--tag`` is the
batch join key; ``--scenario`` / ``--planner`` also work), averages across seeds, and writes the 2×2
congestion panel: denial, total delay, air detour, throughput vs offered load. No re-simulation — it
plots rows a sweep already persisted.

    bash experiments/batch/lambda_sweep.sh mysweep      # executes the runs (tag=lamsweep_mysweep)
    uv run python -m experiments.readouts.curve --tag lamsweep_mysweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from freespace_sim import runs  # noqa: E402


def _filter(idx, args):
    df = idx
    for col, val in (("tag", args.tag), ("scenario", args.scenario), ("planner", args.planner)):
        if val is not None:
            df = df[df[col] == val]
    return df


def _mean_by_lambda(df, key):
    g = df.groupby("lam_per_hour")[key].mean().sort_index()
    return list(g.index), list(g.values)


def plot_curve(df, out_png, title) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    lam, _ = _mean_by_lambda(df, "denial_rate")
    # steady-state twins (issue #25) — plotted dotted alongside the whole-run curves when the index
    # carries them (runs saved after the window feature landed). Each guarded by column presence.
    has_steady = "steady_mean_total_delay_s" in df.columns

    ax = axes[0, 0]
    ax.plot(*_mean_by_lambda(df, "denial_rate"), "o-", label="all denials")
    if "congestion_denial_rate" in df.columns:
        ax.plot(*_mean_by_lambda(df, "congestion_denial_rate"), "s--", label="budget-exceeded")
    if "steady_denial_rate" in df.columns:
        ax.plot(*_mean_by_lambda(df, "steady_denial_rate"), "o:", color="gray",
                label="all denials (steady)")
    ax.set_title("Denial rate vs demand"); ax.set_xlabel("offered load λ (req/h)")
    ax.set_ylabel("fraction denied"); ax.legend()

    ax = axes[0, 1]
    ax.plot(*_mean_by_lambda(df, "mean_total_delay_s"), "o-", label="mean")
    ax.plot(*_mean_by_lambda(df, "p95_total_delay_s"), "s--", label="p95")
    if has_steady:
        ax.plot(*_mean_by_lambda(df, "steady_mean_total_delay_s"), "o:", color="tab:red",
                label="mean (steady-state)")
        if "steady_p95_total_delay_s" in df.columns:
            ax.plot(*_mean_by_lambda(df, "steady_p95_total_delay_s"), "s:", color="tab:red",
                    alpha=0.6, label="p95 (steady-state)")
    ax.set_title("Total delay vs demand"); ax.set_xlabel("offered load λ (req/h)")
    ax.set_ylabel("total delay (s)"); ax.legend()

    ax = axes[1, 0]
    ax.plot(*_mean_by_lambda(df, "mean_air_detour_m"), "o-", color="tab:green")
    ax.set_title("Air detour vs demand"); ax.set_xlabel("offered load λ (req/h)")
    ax.set_ylabel("mean detour (m)")

    ax = axes[1, 1]
    ax.plot(*_mean_by_lambda(df, "throughput_per_h"), "o-", label="throughput (acc/h)")
    if "steady_throughput_per_h" in df.columns:
        ax.plot(*_mean_by_lambda(df, "steady_throughput_per_h"), "o:", color="tab:purple",
                label="throughput (steady)")
    ax.plot(lam, lam, ":", color="gray", label="offered = accepted")
    ax.set_title("Throughput vs demand (saturation)"); ax.set_xlabel("offered load λ (req/h)")
    ax.set_ylabel("accepted/h"); ax.legend()

    if has_steady:
        fig.text(0.5, 0.005, "dotted = steady-state window (density plateau; ramp-up/-down tails dropped)",
                 ha="center", fontsize=9, color="dimgray")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="Congestion curve vs λ from the cross-run index.")
    p.add_argument("--tag", default=None, help="filter to a batch's runs (the join key)")
    p.add_argument("--scenario", default=None)
    p.add_argument("--planner", default=None)
    p.add_argument("--out", default=None,
                   help="output PNG (default: <root>/sweeps/<label>/curve.png)")
    p.add_argument("--root", default="results")
    args = p.parse_args()

    idx = runs.load_index(args.root)
    if idx.empty:
        raise SystemExit(f"no index at {args.root}/index.parquet — run some scenarios first")
    df = _filter(idx, args)
    if df.empty:
        raise SystemExit("no runs match the given --tag/--scenario/--planner filter")

    label = args.tag or args.scenario or "all"
    n_lam = df["lam_per_hour"].nunique()
    if n_lam < 2:
        print(f"note: only {n_lam} distinct λ in this set — the curve is a single point, not a trend "
              f"(run a λ-sweep, e.g. experiments/batch/lambda_sweep.sh, for a real curve)")
    out = Path(args.out) if args.out else runs.sweep_dir(label, args.root) / "curve.png"
    plot_curve(df, out, title=f"FCFS congestion curve — {label}  ({len(df)} runs)")


if __name__ == "__main__":
    main()
