"""Shared trajectory cost — the single function every planner minimizes.

Keeping this in one place is what makes planners swappable *and* tunable: the same intent, scored
the same way, no matter which planner produced it. The weights live in `SimConfig` and express the
real FCFS trade-off — wait on the pad vs. fly a detour vs. hover vs. change altitude.
"""

from __future__ import annotations

from .config import SimConfig
from .types import OperationalIntent


def trajectory_cost(intent: OperationalIntent, cfg: SimConfig) -> float:
    """Weighted sum of the four deconfliction levers (metres / seconds → cost)."""
    return (
        cfg.cost_ground_delay_per_s * intent.ground_delay_s
        + cfg.cost_air_hold_per_s * intent.air_hold_s
        + cfg.cost_air_lateral_per_m * intent.air_detour_m
        + cfg.cost_altitude_change_per_m * intent.altitude_change_m
    )
