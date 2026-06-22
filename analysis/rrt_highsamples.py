"""Rerun the 5000x5000 / 1800s / lam=600 RRT scenario at a higher RRT* sample cap.

The metro CLI doesn't expose max_samples, so we patch the factory the sim resolves planners through
(sim.get_planner) to hand back an RRT* with a bigger cap. FCFS order matters, so the whole run is
replayed. Prints acceptance / denial-by-reason / verified, and whether every path stayed planar.
"""
import sys, time
import numpy as np
import freespace_sim.sim as sim
from freespace_sim.config import SimConfig
from freespace_sim.planner.rrt import SpaceTimeRRTStar
from freespace_sim import verify

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
cfg = SimConfig(region_size_m=(5000.0, 5000.0), lam_per_hour=600.0, horizon_s=1800.0, seed=0,
                planner="rrt")

_orig = sim.get_planner
sim.get_planner = lambda name: SpaceTimeRRTStar(max_samples=N) if name == "rrt" else _orig(name)

t0 = time.monotonic()
res = sim.run(cfg, progress=True)
wall = time.monotonic() - t0
s = res.summary()
bad = verify.find_interflight_conflict(res.intents, cfg)
zs = np.concatenate([[p[2] for p, _ in (i.centerline or [])] for i in res.accepted if i.centerline])
print(f"\nmax_samples={N}  n={s['n_requests']} acc={s['n_accepted']} den={s['n_denied']} "
      f"reasons={s['denials_by_reason']}")
print(f"verified={res.verified} conflict={bad}  planar(z min/max/std)="
      f"{zs.min():.1f}/{zs.max():.1f}/{zs.std():.3f}  wall={wall:.0f}s")
