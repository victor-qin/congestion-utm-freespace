"""SimConfig — every modelling knob in one frozen dataclass (mirrors `congestion_sim/config.py`).

Physical/geometry parameters live here directly; *derived* quantities are exposed as ``@property``
so nothing leaks into separate classes. The cost-model weights are the FCFS trade-off dials:
is it cheaper to wait on the pad, fly a detour, hover, or change altitude?
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # --- dimensionality & altitude (full continuous 3D) ---
    dims: int = 3
    ground_level_m: float = 0.0
    cruise_level_m: float = 150.0      # PREFERRED cruise height (not a hard level)
    # Altitude deconfliction band. Currently COLLAPSED to the single cruise level: the airspace has
    # one flight level, so altitude is not yet a deconfliction lever (planners that sample z — RRT*,
    # MILP, NLP — stay on the cruise plane, matching the cruise-plane A* search and keeping the
    # top-down replay faithful). Widen this band (e.g. 0–200) to re-enable multi-altitude routing.
    z_min_m: float = 150.0
    z_max_m: float = 150.0

    # --- region (continuous horizontal free space), local ENU metres ---
    region_size_m: tuple[float, float] = (10_000.0, 10_000.0)
    region_center_latlon: tuple[float, float] = (32.90, -97.04)  # for 3D/Cesium projection later

    # --- global discrete clock (everyone shares it for now) ---
    dt_s: float = 4.0                  # optimization timestep; ONE corridor volume per step

    # --- kinematics ---
    nominal_speed_mps: float = 30.0    # horizontal cruise
    climb_rate_mps: float = 6.0        # vertical climb/descent (0↔150 m ⇒ 25 s)

    # --- corridor geometry (WIDTH & HEIGHT are knobs; LENGTH is derived from speed×dt) ---
    corridor_width_m: float = 60.0     # full lateral width of each corridor box
    corridor_height_m: float = 30.0    # full vertical extent, centered on the segment
    time_buffer_s: float = 4.0         # ASTM time buffer (§4.3.11); ≈ one dt

    # --- hover cylinder (own radius knob; defaults to corridor width) ---
    hover_radius_m: float | None = None   # None ⇒ effective_hover_radius_m = corridor_width_m
    hover_time_s: float = 30.0         # dwell at takeoff/landing (climb time added on top)

    # --- COST MODEL (shared by every planner; the FCFS trade-off knobs) ---
    cost_ground_delay_per_s: float = 1.0      # wait on the pad
    cost_air_lateral_per_m: float = 1.0       # extra detour length flown
    cost_air_hold_per_s: float = 3.0          # loiter/hover mid-route (expensive)
    cost_altitude_change_per_m: float = 2.0   # climb/descend

    # --- denial budgets ---
    max_ground_delay_s: float = 100000.0
    max_detour_factor: float = 100.0     # deny if flown/straight-line exceeds this

    # --- demand / horizon ---
    horizon_s: float = 14_400.0        # 4 h
    lam_per_hour: float = 200.0
    seed: int = 0

    # --- planner selection (pluggable; DEFAULT = A* → shortcut → MILP → shortcut sandwich) ---
    planner: str = "astar"  # "straight"|"rrt"|"lazy"|"astar"|"milp"|"astar_milp"|...

    # ----- DERIVED (kept inside SimConfig) -----
    @property
    def corridor_segment_len_m(self) -> float:
        """Box length per timestep = cruise speed × timestep."""
        return self.nominal_speed_mps * self.dt_s

    @property
    def effective_hover_radius_m(self) -> float:
        """Hover-cylinder radius; defaults to the corridor width."""
        return self.hover_radius_m if self.hover_radius_m is not None else self.corridor_width_m

    @property
    def climb_time_s(self) -> float:
        """Seconds to climb ground → cruise (150 / 6 = 25 s)."""
        return (self.cruise_level_m - self.ground_level_m) / self.climb_rate_mps

    @property
    def n_steps(self) -> int:
        """Number of discrete timesteps in the horizon."""
        return int(self.horizon_s / self.dt_s)
