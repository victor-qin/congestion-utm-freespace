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


def endpoint_altitude_change_m(z0: float, z1: float, interior_dz: float, cfg: SimConfig) -> float:
    """Total vertical travel booked for a flight's ``altitude_change_m``: the mandatory climb from ground
    to its first cruise altitude ``z0``, the interior climb/descent ``interior_dz`` summed along the
    cruise, and the descent from its last cruise altitude ``z1`` back to ground. Shared by every planner
    that reports ``altitude_change_m`` so the endpoint-climb booking lives in one place."""
    return (z0 - cfg.ground_level_m) + (z1 - cfg.ground_level_m) + interior_dz
