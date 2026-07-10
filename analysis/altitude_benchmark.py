"""Altitude benchmark: rerun the dallas_full world pinned to a SINGLE flight level at `alt` to isolate
what altitude buys. climb_time = alt / climb_rate, and the pad/column DWELL = hover + climb, so a lower
plane shortens the climb -> shorter dwell -> pads and exit lanes cycle faster -> less same-hub
serialisation (the constraint that dominates ground delay). The shipped dallas_full routes across three
levels (30/70/110 m); this pins one plane so the measured delta is altitude alone. Prints the run folder.

Overrides flight_levels_m=(alt,) / cruise_level_m / z bounds on the built config via dataclasses.replace
rather than adding more ScenarioSpec fields, deliberately: SimConfig defaults to the multi-level ladder
(30/70/110 m, cruise 75 m, ceiling 125 m) and the test suite assumes it, so the pin stays scoped here.

Finding (single 50 m plane vs a single 150 m plane, the pre-multi-altitude comparison, lam=12k, 1800 s):
ground delay ~-10%, air delay flat (the lateral hex geometry is invariant to which single cruise plane
you fly), ~-27% wall clock -> altitude is a weak, ground-only lever; the binding constraint is horizontal.

Usage: uv run python analysis/altitude_benchmark.py [alt_m] [lam]
       uv run python analysis/altitude_benchmark.py 50 12000
"""
from __future__ import annotations

import dataclasses as dc
import sys
import time

from freespace_sim import runs
from freespace_sim.scenarios import get_scenario, with_overrides
from freespace_sim.sim import run

alt = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
lam = float(sys.argv[2]) if len(sys.argv) > 2 else 12000.0

spec = with_overrides(get_scenario("dallas_full"), lam_per_hour=lam, horizon_s=1800.0)
cfg = spec.config()
# Pin the whole world to a SINGLE flight level at `alt` (dallas_full ships 3-level (30,70,110); this benchmark
# sweeps that plane). Raise the ceiling if `alt`'s corridor box would poke through the default 125 m.
ceiling = max(cfg.airspace_ceiling_m, alt + cfg.corridor_height_m)
cfg = dc.replace(cfg, flight_levels_m=(alt,), cruise_level_m=alt, z_min_m=alt, z_max_m=alt,
                 airspace_ceiling_m=ceiling, terminal_airspace_always_active=True)
demand = spec.demand_model()
tag = f"dallas_full_{int(lam/1000)}k_alt{int(alt)}_taa"
print(f"cruise={alt:.0f}m  climb_time={cfg.climb_time_s:.1f}s  dwell(hover+climb)={cfg.hover_time_s + cfg.climb_time_s:.1f}s "
      f"lam={lam}/h horizon={cfg.horizon_s}s taa=True", file=sys.stderr)

t0 = time.time()
res = run(cfg, demand=demand, progress=True)
wall = time.time() - t0
folder = runs.save_run(res, label=tag, experiment="run", scenario=spec.name, demand=spec.demand.pattern,
                       experiment_args={"tag": tag, "cruise_level_m": alt, "lam": lam}, wall_seconds=wall,
                       write_replay=False)
s = res.summary()
print(f"  n={s['n_requests']} acc={s['n_accepted']} den={s['n_denied']} verified={res.verified} "
      f"({wall:.1f}s) → {folder}", file=sys.stderr)
print(folder)
