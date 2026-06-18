"""Run a scenario and capture a complete, replayable run folder — the "watch it deconflict" demo.

Writes a timestamped ``results/`` folder with the full capture (config, experiment metadata,
scenario, flown trajectories, reserved 4D volumes, metrics) plus the standalone HTML replay and
static figures (snapshot, congestion heatmap, 3D GLB). Defaults to the high-fidelity ``astar_milp``
planner at a small scale where its delay/detour trade-offs are visible.

    uv run python -m experiments.make_replay                       # default small demo
    uv run python -m experiments.make_replay --lam 240 --planner lazy

To re-open / regenerate the replay of a saved run without re-running, see experiments/replay.py.
"""

from __future__ import annotations

import argparse
import time

from freespace_sim import runs, viz
from freespace_sim.config import SimConfig
from freespace_sim.sim import run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lam", type=float, default=120.0, help="demand rate (req/h)")
    p.add_argument("--horizon", type=float, default=600.0, help="sim horizon (s)")
    p.add_argument("--region", type=float, default=2500.0, help="square region side (m)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--planner", default=None, help="override planner (default: astar_milp)")
    args = p.parse_args()

    cfg = SimConfig(
        lam_per_hour=args.lam, horizon_s=args.horizon, seed=args.seed,
        region_size_m=(args.region, args.region),
        **({"planner": args.planner} if args.planner else {}),
    )
    t0 = time.time()
    res = run(cfg, progress=True)
    wall = time.time() - t0
    s = res.summary()
    print(f"ran {s['n_requests']} flights · {s['n_accepted']} accepted · "
          f"{s['n_denied']} denied · verified={res.verified}  ({wall:.1f}s)")

    folder = runs.save_run(res, label="replay", experiment="make_replay",
                           experiment_args=vars(args), wall_seconds=wall)
    # extra static figures alongside the replay.html that save_run already wrote
    viz.snapshot(res, t=cfg.horizon_s * 0.4, out=folder / "snapshot.png")
    viz.congestion_heatmap(res, out=folder / "heatmap.png")
    viz.scene_3d(res).export(folder / "scene.glb")
    print(f"run captured → {folder}")
    print(f"  open {folder / 'replay.html'} in a browser to play / pause / scrub the timeline")


if __name__ == "__main__":
    main()
