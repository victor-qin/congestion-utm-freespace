"""Full-scenario parity check for the optional compiled hex rasteriser.

The unit suite uses a short real cut. This script runs the two LNS benchmark cuts and compares all
three public raster forms as ordered rows for every committed volume. It also reports the closest
reference slack to either inflation threshold as a diagnostic; correctness no longer depends on
that margin because numerically ambiguous box cells are resolved by the numpy oracle.

    uv run python analysis/verify_rasteriser.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import freespace_sim
from freespace_sim import sim
from freespace_sim.planner import hexgrid as hg
from freespace_sim.scenarios import get_scenario
from freespace_sim.scenarios.spec import with_overrides

ROOT = Path(__file__).resolve().parents[1]
assert Path(freespace_sim.__file__).resolve().is_relative_to(ROOT), (
    f"freespace_sim resolved to {freespace_sim.__file__}, not {ROOT}")

CUTS = ((120.0, 1500.0), (600.0, 2400.0))


def _sweep(vol, cfg, radius, infl_blocked, infl_pad, *, compiled):
    with hg.rasterizer_backend(compiled):
        rows = (
            list(hg.rasterize_volume_ranges(vol, cfg, radius, infl_blocked, infl_pad)),
            list(hg.rasterize_volume_dual(vol, cfg, radius, infl_blocked, infl_pad)),
            list(hg.rasterize_volume(vol, cfg, radius)),
        )
    if compiled and not hg._COMPILED:
        raise RuntimeError("compiled rasteriser failed during verification; see warning above")
    return rows


def main() -> int:
    if not hg._COMPILED:
        print("compiled rasteriser unavailable — nothing to verify", file=sys.stderr)
        return 2
    worst = 0
    for demand_duration, horizon in CUTS:
        spec = with_overrides(
            get_scenario("density_faa_wing_zipline"),
            demand_duration_s=demand_duration,
            horizon_s=horizon,
        )
        cfg = spec.config()
        with hg.rasterizer_backend(True):
            result = sim.run(cfg, demand=spec.demand_model(), planner_name="astar", progress=False)
        radius = hg.circumradius(cfg)
        infl_blocked = cfg.corridor_width_m / 2.0 + radius
        infl_pad = cfg.effective_hover_radius_m + radius
        volumes = [volume for _fid, volume in result.ledger.iter_committed()]

        started = time.monotonic()
        different = order_only = 0
        for volume in volumes:
            reference = _sweep(volume, cfg, radius, infl_blocked, infl_pad, compiled=False)
            compiled = _sweep(volume, cfg, radius, infl_blocked, infl_pad, compiled=True)
            if reference != compiled:
                different += 1
                order_only += all(
                    set(map(tuple, left)) == set(map(tuple, right))
                    for left, right in zip(reference, compiled)
                )

        margin_pad = margin_blocked = np.inf
        for volume in volumes:
            for level in hg._levels_overlapped(volume, cfg):
                _q, _r, slack = hg._candidate_slack(
                    volume, cfg, radius, infl_pad, z=cfg.flight_levels_m[level]
                )
                if slack.size:
                    margin_pad = min(margin_pad, float(np.abs(slack - infl_pad).min()))
                    margin_blocked = min(
                        margin_blocked, float(np.abs(slack - infl_blocked).min())
                    )

        print(
            f"{demand_duration:.0f}s cut: {len(result.intents)} legs, {len(volumes)} volumes "
            f"({time.monotonic() - started:.0f}s verification)"
        )
        suffix = f" (order-only: {order_only})" if different else ""
        print(f"  ordered-row differences: {different}{suffix}")
        print(
            f"  diagnostic boundary margin: pad={margin_pad:.3e} m, "
            f"blocked={margin_blocked:.3e} m"
        )
        worst = max(worst, different)

    verdict = "PASS — ordered rows match for every volume" if worst == 0 else "FAIL"
    print(f"\nVERDICT: {verdict}")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
