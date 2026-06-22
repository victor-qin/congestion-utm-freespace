"""READ OUT (per-run) — static figures from a saved run: snapshot, congestion heatmap, 3D scene.

Loads a run folder from disk and writes PNG/GLB figures next to it. ``--uss`` slices the snapshot to a
single operator; ``--t`` sets the snapshot time. No re-simulation.

    uv run python -m experiments.readouts.figures results/<folder>
    uv run python -m experiments.readouts.figures results/<folder> --uss walmart_uss --t 720
"""

from __future__ import annotations

import argparse
from pathlib import Path

from freespace_sim import runs, viz


def main() -> None:
    p = argparse.ArgumentParser(description="Write snapshot/heatmap/3D figures from a saved run folder.")
    p.add_argument("folder", help="a results/ run folder written by experiments.run")
    p.add_argument("--t", type=float, default=None, help="snapshot time (s); default 0.4·horizon")
    p.add_argument("--uss", default=None, help="restrict the snapshot to one USS")
    p.add_argument("--no-3d", action="store_true", help="skip the (slower) 3D scene.glb export")
    args = p.parse_args()

    folder = Path(args.folder)
    loaded = runs.load_run(folder)
    t = args.t if args.t is not None else loaded.config.horizon_s * 0.4

    snap_name = f"snapshot_{args.uss}.png" if args.uss else "snapshot.png"
    viz.snapshot(loaded, t=t, out=folder / snap_name, uss=args.uss)
    viz.congestion_heatmap(loaded, out=folder / "heatmap.png")
    print(f"wrote {snap_name} + heatmap.png")
    if not args.no_3d:
        viz.scene_3d(loaded).export(folder / "scene.glb")
        print("wrote scene.glb")


if __name__ == "__main__":
    main()
