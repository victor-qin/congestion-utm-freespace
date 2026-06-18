"""Headline experiment — the FCFS congestion curve in continuous free space.

Sweep the demand rate λ (requests/hour); for each λ (× seeds) run a full strategic-deconfliction
simulation and record the aggregate outcomes. As λ rises, the FCFS newcomer is pushed into ever
costlier delays and detours; once no plan fits the operator's budget the request is *denied*. The
output is the curve that motivates the whole project: **how congestion degrades a free-space UTM as
demand grows**, and how that degradation splits across the levers (delay, detour, denial).

    uv run python -m experiments.lambda_sweep --quick           # fast smoke (few λ, 1 seed)
    uv run python -m experiments.lambda_sweep                   # the full curve
    uv run python -m experiments.lambda_sweep --planner lazy    # swap planner (faster at scale)

Writes ``results/<stamp>-lambda_sweep/sweep.parquet`` + ``lambda_sweep.png`` (the 2×2 panel).
"""

from __future__ import annotations

import argparse
import time

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: render to file, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from freespace_sim import metrics, runs, viz  # noqa: E402
from freespace_sim.config import SimConfig  # noqa: E402
from freespace_sim.sim import run  # noqa: E402


def sweep(lambdas, seeds, *, planner: str | None, horizon_s: float, region_m: float):
    """Run every (λ, seed) cell; return (aggregate rows, per-flight DataFrame tagged with λ/seed)."""
    rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    for lam in lambdas:
        for seed in seeds:
            cfg = SimConfig(
                lam_per_hour=lam,
                seed=seed,
                horizon_s=horizon_s,
                region_size_m=(region_m, region_m),
                **({"planner": planner} if planner else {}),
            )
            t0 = time.time()
            res = run(cfg, progress=True)
            agg = metrics.aggregate(res)
            agg["wall_s"] = time.time() - t0
            rows.append(agg)
            fdf = metrics.flight_frame(res)
            fdf["lam_per_hour"], fdf["seed"] = lam, seed
            frames.append(fdf)
            print(
                f"  λ={lam:6.0f}/h seed={seed}  n={agg['n_requests']:4d} "
                f"den={agg['denial_rate']:.2f} totDly={agg['mean_total_delay_s']:6.1f}s "
                f"(p95 {agg['p95_total_delay_s']:5.0f}s) det={agg['mean_air_detour_m']:5.0f}m "
                f"util={agg['airspace_utilization']:.4f} verif={agg['verified']}  ({agg['wall_s']:.1f}s)"
            )
    return rows, pd.concat(frames, ignore_index=True)


def _mean_by_lambda(rows, key):
    """Average ``key`` across seeds for each λ → (sorted λ list, mean list)."""
    lams = sorted({r["lam_per_hour"] for r in rows})
    means = [
        sum(r[key] for r in rows if r["lam_per_hour"] == lam)
        / max(1, sum(1 for r in rows if r["lam_per_hour"] == lam))
        for lam in lams
    ]
    return lams, means


def plot_sweep(rows: list[dict], out_png) -> None:
    """The 2×2 congestion panel: denial, delay, detour/stretch, throughput vs offered load."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    lam, _ = _mean_by_lambda(rows, "denial_rate")

    ax = axes[0, 0]
    ax.plot(lam, _mean_by_lambda(rows, "denial_rate")[1], "o-", label="all denials")
    ax.plot(*_mean_by_lambda(rows, "congestion_denial_rate"), "s--", label="budget-exceeded")
    ax.set_title("Denial rate vs demand")
    ax.set_xlabel("offered load λ (req/h)")
    ax.set_ylabel("fraction denied")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(*_mean_by_lambda(rows, "mean_total_delay_s"), "o-", label="mean")
    ax.plot(*_mean_by_lambda(rows, "p95_total_delay_s"), "s--", label="p95")
    ax.set_title("Total delay vs demand (hold + loiter + detour-time)")
    ax.set_xlabel("offered load λ (req/h)")
    ax.set_ylabel("total delay (s)")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(*_mean_by_lambda(rows, "mean_air_detour_m"), "o-", color="tab:green")
    ax.set_title("Air detour vs demand")
    ax.set_xlabel("offered load λ (req/h)")
    ax.set_ylabel("mean detour (m)")

    ax = axes[1, 1]
    ax.plot(*_mean_by_lambda(rows, "throughput_per_h"), "o-", label="throughput (acc/h)")
    ax.plot(lam, lam, ":", color="gray", label="offered = accepted")
    ax.set_title("Throughput vs demand (saturation)")
    ax.set_xlabel("offered load λ (req/h)")
    ax.set_ylabel("accepted/h")
    ax.legend()

    fig.suptitle("FCFS strategic deconfliction — free-space congestion curve", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="fast smoke: few λ, 1 seed, short horizon")
    p.add_argument("--planner", default=None, help="override planner (default: SimConfig default)")
    p.add_argument("--horizon", type=float, default=1800.0, help="sim horizon (s)")
    p.add_argument("--region", type=float, default=4000.0, help="square region side (m)")
    args = p.parse_args()

    if args.quick:
        lambdas, seeds, horizon, region = [60.0, 180.0, 360.0], [0], 900.0, 3000.0
    else:
        lambdas = [60.0, 120.0, 240.0, 360.0, 480.0, 600.0]
        seeds, horizon, region = [0, 1, 2], args.horizon, args.region

    print(f"sweep: λ={lambdas} seeds={seeds} horizon={horizon}s region={region}m "
          f"planner={args.planner or SimConfig().planner}")
    rows, per_flight = sweep(lambdas, seeds, planner=args.planner, horizon_s=horizon, region_m=region)
    folder = runs.save_sweep(rows, label="lambda_sweep", experiment_args={
        "lambdas": lambdas, "seeds": seeds, "horizon_s": horizon, "region_m": region,
        "planner": args.planner or SimConfig().planner, "quick": args.quick})
    per_flight.to_parquet(folder / "per_flight.parquet", index=False)
    plot_sweep(rows, folder / "lambda_sweep.png")
    viz.delay_histograms_by_lambda(per_flight, out=folder / "delay_histograms.png")
    viz.delay_pct_histograms_by_lambda(per_flight, out=folder / "delay_pct_histograms.png")
    viz.delay_sources(per_flight, out=folder / "delay_sources.png")   # where delay comes from, vs λ
    print(f"sweep saved → {folder}  (summary + delay_histograms + delay_pct_histograms + per_flight)")


if __name__ == "__main__":
    main()
