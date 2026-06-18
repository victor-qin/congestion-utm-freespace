"""Compare planners on one scenario: acceptance, cost, denial reasons, wall-time.

Shows the trade-off — straight/decoupled are fast but deny where only a spatial detour works; RRT*
recovers those at the cost of detour/compute; lazy gets the best of both. Run:

    uv run python -m experiments.compare_planners
"""

from __future__ import annotations

import time

from freespace_sim.config import SimConfig
from freespace_sim.sim import run


def main() -> None:
    base = dict(lam_per_hour=300.0, horizon_s=1800.0, seed=2, region_size_m=(5000.0, 5000.0))
    print(
        f'{"planner":>10} {"acc":>4} {"den":>4} {"meanCost":>9} {"meanDly":>8} '
        f'{"meanDet":>8} {"sec":>6} {"verif":>6}  denial_reasons'
    )
    for planner in ("straight", "decoupled", "rrt", "lazy"):
        cfg = SimConfig(planner=planner, **base)
        t0 = time.time()
        res = run(cfg)
        dt = time.time() - t0
        s = res.summary()
        costs = [i.cost for i in res.accepted]
        mean_cost = sum(costs) / len(costs) if costs else 0.0
        print(
            f'{planner:>10} {s["n_accepted"]:>4} {s["n_denied"]:>4} {mean_cost:>9.1f} '
            f'{s["mean_ground_delay_s"]:>7.1f}s {s["mean_air_detour_m"]:>7.0f}m {dt:>5.1f}s '
            f'{str(s["verified"]):>6}  {s["denials_by_reason"]}'
        )


if __name__ == "__main__":
    main()
