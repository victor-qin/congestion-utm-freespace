"""Shared trajectory cost — the single function every planner minimizes.

Keeping this in one place is what makes planners swappable *and* tunable: the same intent, scored
the same way, no matter which planner produced it. The weights live in `SimConfig` and express the
real FCFS trade-off — wait on the pad vs. fly a detour vs. hover vs. change altitude.
"""

from __future__ import annotations

from .config import SimConfig
from .types import OperationalIntent


def trajectory_cost(intent: OperationalIntent, cfg: SimConfig) -> float:
    """Weighted sum of the four deconfliction levers — the single cost every planner minimizes.

    Ground delay and air hold are charged per second; lateral detour and altitude change per metre
    (the per-metre weights derive from the per-second dials in ``cfg``), so one intent scores the
    same regardless of which planner produced it.

    Parameters
    ------------
    - intent (OperationalIntent): the trajectory whose ground delay, air hold, lateral detour, and
      altitude change are being priced.
    - cfg (SimConfig): supplies the four cost weights.

    Return
    --------
    - output (float): the weighted trajectory cost.
    """
    return (
        cfg.cost_ground_delay_per_s * intent.ground_delay_s
        + cfg.cost_air_hold_per_s * intent.air_hold_s
        + cfg.cost_air_lateral_per_m * intent.air_detour_m
        + cfg.cost_altitude_change_per_m * intent.altitude_change_m
    )


def endpoint_altitude_change_m(z0: float, z1: float, interior_dz: float, cfg: SimConfig) -> float:
    """Total vertical travel booked for a flight's ``altitude_change_m``.

    Sums the mandatory climb from ground to the first cruise altitude, the interior climb/descent
    along the cruise, and the descent from the last cruise altitude back to ground. Shared by every
    planner that reports ``altitude_change_m`` so the endpoint-climb booking lives in one place.

    Parameters
    ------------
    - z0 (float): first cruise altitude (m); the ground→z0 climb is booked.
    - z1 (float): last cruise altitude (m); the z1→ground descent is booked.
    - interior_dz (float): total interior climb/descent (m) summed along the cruise.
    - cfg (SimConfig): supplies ``ground_level_m``.

    Return
    --------
    - output (float): total vertical travel (m) to book as ``altitude_change_m``.
    """
    return (z0 - cfg.ground_level_m) + (z1 - cfg.ground_level_m) + interior_dz
