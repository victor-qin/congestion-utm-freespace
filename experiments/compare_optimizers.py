"""Compare the trajectory optimizers (A*, MILP) on scenarios that exercise each lever.

Reports per-flight cost (lower = better), the levers used (delay / detour), conflict-free, and
wall-time. The MILP is the global optimum, so its cost is the yardstick the others are measured
against. Run:

    uv run python -m experiments.compare_optimizers
"""

from __future__ import annotations

import time

from freespace_sim.config import SimConfig
from freespace_sim.geometry import CylinderSpec, box_from_segment
from freespace_sim.ledger import ReservationLedger
from freespace_sim.planner import get_planner
from freespace_sim.types import FlightRequest, vec
from freespace_sim.volumes import Volume4D

CFG = SimConfig()
REQ = FlightRequest(1, vec(0, 0, 0), vec(2000, 0, 0), 0.0)


def _thin_wall(clear=1e6):
    return Volume4D(box_from_segment(vec(1000, -200, 150), vec(1000, 200, 150), 40, 400), 0.0, clear)


def _dest_pad_block():
    return Volume4D(CylinderSpec(2000, 0, CFG.effective_hover_radius_m, 0, 150), 0.0, 200.0)


SCENARIOS = {
    "empty (no obstacle)": [],
    "thin wall (permanent)": [_thin_wall()],
    "thin wall (clears@70s)": [_thin_wall(70.0)],
    "busy dest pad (clears@200s)": [_dest_pad_block()],
}


def main() -> None:
    planners = ["straight", "astar", "milp", "astar_milp"]
    print(f'{"scenario":>28} | {"planner":>8} {"cost":>8} {"delay":>6} {"detour":>7} {"cf":>3} {"sec":>6}')
    print("-" * 80)
    for name, obs in SCENARIOS.items():
        baseline = None
        for p in planners:
            led = ReservationLedger(CFG)
            if obs:
                led.commit(99, obs)
            t0 = time.time()
            intent = get_planner(p).plan(REQ, led, CFG)
            dt = time.time() - t0
            if not intent.accepted:
                print(f'{name:>28} | {p:>8} {"DENIED":>8} {"":>6} {"":>7} {"":>3} {dt:6.2f}'
                      f'  ({intent.denial_reason.value})')
                continue
            cf = not led.any_conflict(intent.volumes)
            if p == "milp":
                baseline = intent.cost
            gap = "" if baseline is None else f" (+{intent.cost - baseline:.0f} vs milp)"
            print(f'{name:>28} | {p:>8} {intent.cost:8.1f} {intent.ground_delay_s:6.1f} '
                  f'{intent.air_detour_m:7.1f} {str(cf):>3} {dt:6.2f}{gap}')
        print("-" * 80)


if __name__ == "__main__":
    main()
