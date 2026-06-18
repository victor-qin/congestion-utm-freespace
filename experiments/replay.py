"""Open / regenerate the replay of a *saved* run — the live free-space visualization viewer.

Loads a run folder produced by `runs.save_run` (config + scenario + flown trajectories + reserved
4D volumes) entirely from disk and (re)writes ``replay.html`` — a standalone webpage that plays the
reservations back like a video: press play, pause, or drag the slider to scrub to any instant and
see exactly which corridors and hover cylinders were reserved at that moment.

    uv run python -m experiments.replay results/2026-...-replay_ab12cd34
    uv run python -m experiments.replay results/<folder> --open      # also open in the browser

No re-simulation happens — it reads what was stored, so the replay always matches the saved run.
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from freespace_sim import runs, viz, viz_html


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="a results/ run folder written by save_run")
    p.add_argument("--open", action="store_true", help="open the replay in the default browser")
    p.add_argument("--figures", action="store_true", help="also (re)write snapshot + heatmap PNGs")
    args = p.parse_args()

    folder = Path(args.folder)
    loaded = runs.load_run(folder)
    s = loaded.summary()
    print(f"loaded {s['n_requests']} flights · {s['n_accepted']} accepted · "
          f"{s['n_denied']} denied · planner={loaded.config.planner}")

    out = viz_html.write_html(loaded, folder / "replay.html")
    print(f"replay → {out}")
    if args.figures:
        viz.snapshot(loaded, t=loaded.config.horizon_s * 0.4, out=folder / "snapshot.png")
        viz.congestion_heatmap(loaded, out=folder / "heatmap.png")
        print("wrote snapshot.png + heatmap.png")
    if args.open:
        webbrowser.open(f"file://{Path(out).resolve()}")


if __name__ == "__main__":
    main()
