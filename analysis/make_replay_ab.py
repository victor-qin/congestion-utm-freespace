"""Build two replay HTMLs from ONE run to isolate what the filing change does on screen.

A* plans are bit-identical under both filings (same volume windows, same centerline -- see
``analysis/probe_physics_500.py``), so anything that differs on screen is the *rendering* rule
alone.  This writes:

* ``replay_today.html``  -- shipped: a transit box is live ``[t[i], t[i+1] + buf)``
* ``replay_legacy.html`` -- the two pre-2026-08-14 lines restored, so a box is live
  ``[t[i] - buf, t[i+1] + buf)``

Same scene bytes in both, so a pixel diff is purely the live-window rule.

    uv run python analysis/make_replay_ab.py --flights 24
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import freespace_sim

REPO_ROOT = Path(__file__).resolve().parent.parent
if REPO_ROOT not in Path(freespace_sim.__file__).resolve().parents:
    raise SystemExit("loaded the wrong tree")

from freespace_sim import viz_html  # noqa: E402
from freespace_sim.scenario import scenario_from_requests  # noqa: E402
from freespace_sim.scenarios import get_scenario  # noqa: E402
from freespace_sim.sim import run as sim_run  # noqa: E402

# The exact shipped lines, and their pre-change form.  Patching the GENERATED html (not the
# template) keeps the scene identical -- only the live-window arithmetic moves.
TODAY_TQ = "const tq = (t - START)*DATA.qt, lo = tq - TBQ;"
LEGACY_TQ = "const tq = (t - START)*DATA.qt, lo = tq - TBQ, hi = tq + TBQ;"
TODAY_Z = "const z = Math.min(fl.t.length - 2, upperBound(fl.t, tq) - 1);"
LEGACY_Z = "const z = Math.min(fl.t.length - 2, upperBound(fl.t, hi) - 1);"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="density_faa_wing_zipline")
    ap.add_argument("--flights", type=int, default=24)
    ap.add_argument("--planner", default="astar")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "analysis")
    args = ap.parse_args()

    spec = get_scenario(args.scenario)
    cfg = spec.config()
    demand = spec.demand_model()
    requests = sorted(
        demand.generate(cfg, np.random.default_rng(cfg.seed)), key=lambda r: r.flight_id
    )[: args.flights]
    result = sim_run(cfg, scenario=scenario_from_requests(requests), demand=demand,
                     planner_name=args.planner)
    accepted = [i for i in result.intents if i.accepted]
    print(f"{args.scenario} x{len(requests)} {args.planner}: {len(accepted)} accepted")

    today_path = args.outdir / "replay_today.html"
    viz_html.write_html(result, today_path)          # returns the PATH, not the markup
    html = today_path.read_text(encoding="utf-8")

    if html.count(TODAY_TQ) != 1 or html.count(TODAY_Z) != 1:
        raise SystemExit("shipped replay lines not found exactly once — the JS moved; update this script")
    legacy = html.replace(TODAY_TQ, LEGACY_TQ).replace(TODAY_Z, LEGACY_Z)
    legacy_path = args.outdir / "replay_legacy.html"
    legacy_path.write_text(legacy)

    print(f"wrote {today_path}")
    print(f"wrote {legacy_path}")
    # A frame with plenty airborne, and one flight to centre on.
    mid = [i for i in accepted if len(i.centerline) > 6]
    mid.sort(key=lambda i: -len(i.centerline))
    focus = mid[0]
    t0, t1 = focus.centerline[0][1], focus.centerline[-1][1]
    tq = t0 + 0.55 * (t1 - t0)
    p = focus.centerline[len(focus.centerline) // 2][0]
    print(f"focus flight {focus.request.flight_id}  hops={len(focus.centerline)-1}  "
          f"airborne {t0:.0f}..{t1:.0f}s  suggest t={tq:.0f}  centre=({float(p[0]):.0f},{float(p[1]):.0f})")


if __name__ == "__main__":
    main()
