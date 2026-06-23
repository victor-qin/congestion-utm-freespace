"""READ OUT (per-run) — (re)generate the standalone HTML replay of a saved run.

Loads a run folder produced by ``experiments.run`` entirely from disk and writes ``replay.html`` — the
scrub/play/pause viewer of the FCFS deconfliction, colored by USS. No re-simulation: it reads what was
stored, so the replay always matches the saved run.

    uv run python -m experiments.readouts.replay results/<folder>
    uv run python -m experiments.readouts.replay results/<folder> --open
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from freespace_sim import runs, viz_html


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate replay.html from a saved run folder.")
    p.add_argument("folder", help="a results/ run folder written by experiments.run")
    p.add_argument("--open", action="store_true", help="open the replay in the default browser")
    args = p.parse_args()

    folder = Path(args.folder)
    loaded = runs.load_run(folder)
    s = loaded.summary()
    print(f"loaded {s['n_requests']} flights · {s['n_accepted']} accepted · "
          f"{s['n_denied']} denied · planner={loaded.config.planner}")

    out = viz_html.write_html(loaded, folder / "replay.html")
    print(f"replay → {out}")
    if args.open:
        webbrowser.open(f"file://{Path(out).resolve()}")


if __name__ == "__main__":
    main()
