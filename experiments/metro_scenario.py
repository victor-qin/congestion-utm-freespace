"""Metro stress scenario — the free-space analog of the companion repo's ``dfw_scenario.py``.

A named, reproducible FCFS stress test over a continuous free-space region you choose. Unlike the
λ-sweep (which samples a curve), this runs one or more *headline* operating points to the full
capture: every λ you pass produces its own complete, replayable run folder (config + experiment
metadata + scenario + flown trajectories + reserved 4D volumes + metrics + replay.html), plus an
outcomes-by-hour figure and a total-delay histogram. All runs append to ``results/index.parquet``.

You parameterise the *scenario*, not the source: region, λ list, horizon, seed, and planner are all
CLI inputs that **override the SimConfig defaults** — config.py is never edited (see module note
below). Examples:

    uv run python -m experiments.metro_scenario --region 8000 8000 --lam 600 1200 2400
    uv run python -m experiments.metro_scenario --region 5000 5000 --lam 900 --horizon 3600 --planner lazy
    uv run python -m experiments.metro_scenario --smoke            # fast: small region, short horizon

Config note: ``SimConfig`` is a *frozen dataclass of defaults*. An experiment customises a run by
constructing ``SimConfig(region_size_m=..., lam_per_hour=..., ...)`` — passing overrides to the
constructor, never mutating config.py. That file holds the baseline constants; the scenario is the
override layer on top. ``build_config`` below is exactly that override layer.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from freespace_sim import metrics, runs, viz  # noqa: E402
from freespace_sim.config import SimConfig  # noqa: E402
from freespace_sim.sim import run  # noqa: E402

HORIZON_FULL = 3600.0   # 1 simulated hour
HORIZON_SMOKE = 600.0   # ~10 simulated minutes — fast wall time


def build_config(region, lam, horizon, seed, planner) -> SimConfig:
    """The override layer: a SimConfig with this scenario's region/λ/horizon on top of the defaults."""
    return SimConfig(
        region_size_m=(float(region[0]), float(region[1])),
        lam_per_hour=lam,
        horizon_s=horizon,
        seed=seed,
        **({"planner": planner} if planner else {}),
    )


def outcomes_by_hour(result) -> dict[int, dict[str, int]]:
    """Bucket intents by simulated hour of filing → {hour: {requested, accepted, denied}}."""
    by_hour: dict[int, dict[str, int]] = defaultdict(
        lambda: {"requested": 0, "accepted": 0, "denied": 0})
    for i in result.intents:
        b = by_hour[int(i.request.t_request // 3600)]
        b["requested"] += 1
        b["accepted" if i.accepted else "denied"] += 1
    return dict(sorted(by_hour.items()))


def plot_outcomes_by_hour(per_hour, out) -> None:
    """Stacked accepted/denied bars per simulated hour."""
    hours = list(per_hour)
    acc = np.array([per_hour[h]["accepted"] for h in hours])
    den = np.array([per_hour[h]["denied"] for h in hours])
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x = np.arange(len(hours))
    ax.bar(x, acc, label="accepted", color="#22c55e", edgecolor="white", linewidth=0.5)
    ax.bar(x, den, bottom=acc, label="denied", color="#ef4444", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h{h}" for h in hours])
    ax.set_xlabel("simulated hour of request")
    ax.set_ylabel("flights")
    ax.set_title("Outcomes by simulated hour")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(out, dpi=120)
    plt.close(fig)


def run_one(region, lam, horizon, seed, planner, tag) -> None:
    """Run one operating point to the full capture and print its headline line."""
    cfg = build_config(region, lam, horizon, seed, planner)
    sim_h = cfg.horizon_s / 3600.0
    print(f"\nλ={lam:g}/h  region={cfg.region_size_m[0]:g}×{cfg.region_size_m[1]:g} m  "
          f"horizon={sim_h:.2f} h  planner={cfg.planner}  → ~{lam * sim_h:.0f} flights expected")

    t0 = time.monotonic()
    res = run(cfg, progress=True)
    wall = time.monotonic() - t0

    folder = runs.save_run(
        res, label=f"{tag}_lam{int(lam)}", experiment="metro_scenario",
        experiment_args={"region_m": list(cfg.region_size_m), "lam_per_hour": lam,
                         "horizon_s": horizon, "seed": seed, "planner": cfg.planner},
        wall_seconds=wall)

    s = res.summary()
    agg = metrics.aggregate(res)
    print(f"  n={s['n_requests']:4d}  accepted={s['n_accepted']:4d}  denied={s['n_denied']:3d}  "
          f"denial={agg['denial_rate']:.1%}  meanDelay={agg['mean_total_delay_s']:.0f}s "
          f"(p95 {agg['p95_total_delay_s']:.0f}s)  verified={res.verified}  ({wall:.1f}s)")
    print(f"  planner solve/flight: mean={agg['mean_solve_time_s'] * 1000:.0f}ms  "
          f"p95={agg['p95_solve_time_s'] * 1000:.0f}ms  max={agg['max_solve_time_s']:.2f}s  "
          f"total={agg['total_solve_time_s']:.1f}s")

    # per-run figures alongside the replay.html save_run already wrote
    plot_outcomes_by_hour(outcomes_by_hour(res), folder / "outcomes_by_hour.png")
    df = metrics.flight_frame(res)
    if df["accepted"].any():
        acc_df = df.loc[df["accepted"]]
        viz.delay_histogram(acc_df["total_delay_s"],
                            out=folder / "delay_hist.png", title=f"Total delay — λ={lam:g}/h")
        viz.delay_pct_histogram(acc_df["delay_pct"],
                                out=folder / "delay_pct_hist.png", title=f"Delay % — λ={lam:g}/h")
        viz.delay_sources(acc_df, out=folder / "delay_sources.png", by=None)   # single-run breakdown
    viz.congestion_heatmap(res, out=folder / "heatmap.png")
    print(f"  captured → {folder}")


def main() -> None:
    p = argparse.ArgumentParser(description="Free-space metro FCFS stress scenario.")
    p.add_argument("--region", type=float, nargs=2, metavar=("W", "H"), default=[8000.0, 8000.0],
                   help="region size in metres (default 8000 8000)")
    p.add_argument("--lam", type=float, nargs="+", default=[600.0, 1200.0],
                   help="one or more arrival rates (req/h) — each becomes its own captured run")
    p.add_argument("--horizon", type=float, default=None, help="sim horizon (s); default 1 h")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--planner", default=None, help="override planner (default: astar_milp)")
    p.add_argument("--tag", default="metro", help="goes into each run-folder name")
    p.add_argument("--smoke", action="store_true", help="short horizon for ~fast wall time")
    args = p.parse_args()

    horizon = args.horizon if args.horizon is not None else (HORIZON_SMOKE if args.smoke else HORIZON_FULL)
    print(f"metro scenario · region={args.region} · λ={args.lam} · horizon={horizon}s · "
          f"seed={args.seed} · planner={args.planner or SimConfig().planner}")
    for lam in args.lam:
        run_one(args.region, lam, horizon, args.seed, args.planner, args.tag)
    print(f"\ndone · {len(args.lam)} run(s) captured under results/ (indexed in results/index.parquet)")


if __name__ == "__main__":
    main()
