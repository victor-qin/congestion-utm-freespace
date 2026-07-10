"""SimConfig — every modelling knob in one frozen dataclass (mirrors `congestion_sim/config.py`).

Physical/geometry parameters live here directly; *derived* quantities are exposed as ``@property``
so nothing leaks into separate classes. The cost-model weights are the FCFS trade-off dials:
is it cheaper to wait on the pad, fly a detour, hover, or change altitude?
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    # --- dimensionality & altitude (full 3D, regulated band [ground_level_m, airspace_ceiling_m]) ---
    dims: int = 3
    ground_level_m: float = 0.0
    cruise_level_m: float = 75.0       # single-plane planners' cruise altitude (RRT*/MILP/NLP/straight)
    # Continuous-sampler band (RRT*/MILP/NLP). Collapsed to the single cruise level so those planners stay
    # on one plane. A* instead deconflicts by altitude on the DISCRETE ``flight_levels_m`` ladder below;
    # widening this band (z_min_m < z_max_m) to give the samplers multi-altitude too is a follow-up.
    z_min_m: float = 75.0
    z_max_m: float = 75.0
    # Regulated airspace ceiling: every hover/terminal column spans [ground_level_m, airspace_ceiling_m].
    airspace_ceiling_m: float = 125.0
    # A*'s discrete cruise levels. Strictly ascending; adjacent gaps must EXCEED corridor_height_m (so
    # neighbouring level boxes don't touch in z and stay FCL-disjoint), and the top/bottom boxes
    # (level ± corridor_height_m/2) must fit within [ground_level_m, airspace_ceiling_m]. Set
    # ``flight_levels_m=(cruise_level_m,)`` (and matching ceiling) for legacy single-level behaviour.
    flight_levels_m: tuple[float, ...] = (30.0, 70.0, 110.0)

    # --- region (continuous horizontal free space), local ENU metres ---
    region_size_m: tuple[float, float] = (10_000.0, 10_000.0)
    region_center_latlon: tuple[float, float] = (32.90, -97.04)  # for 3D/Cesium projection later

    # --- global discrete clock (everyone shares it for now) ---
    dt_s: float = 4.0                  # optimization timestep; ONE corridor volume per step

    # --- kinematics ---
    nominal_speed_mps: float = 30.0    # horizontal cruise
    climb_rate_mps: float = 6.0        # vertical climb/descent (0↔75 m ⇒ 12.5 s)

    # --- corridor geometry (WIDTH & HEIGHT are knobs; LENGTH is derived from speed×dt) ---
    corridor_width_m: float = 60.0     # full lateral width of each corridor box
    corridor_height_m: float = 30.0    # full vertical extent, centered on the segment
    time_buffer_s: float = 4.0         # ASTM time buffer (§4.3.11); ≈ one dt

    # --- hover cylinder (own radius knob; defaults to corridor width) ---
    hover_radius_m: float | None = None   # None ⇒ effective_hover_radius_m = corridor_width_m
    hover_time_s: float = 30.0         # dwell at takeoff/landing (climb time added on top)
    # default shared-terminal COLUMN radius when a Terminal doesn't set its own (per-hub Terminal.radius
    # overrides). 90 m (> corridor_width) gives divergent same-hub exit lanes enough angular spread to
    # start flush with the column edge (corridor_overlap=0) and still launch concurrently. See volumes.exit_radius.
    terminal_radius_m: float = 90.0

    # --- COST MODEL (shared by every planner; the FCFS trade-off knobs) ---
    cost_ground_delay_per_s: float = 1.0      # wait on the pad
    cost_air_lateral_per_m: float = 1.0       # extra detour length flown
    cost_air_hold_per_s: float = 3.0          # loiter/hover mid-route (expensive)
    cost_altitude_change_per_m: float = 2.0   # climb/descend

    # --- search (A*) ---
    # A* multi-altitude only: generate mid-route climb/descend edges at every air node so a flight can
    # change flight level en route. These dominate the multi-level search cost (an all-levels column
    # check per air node) — set False on large multi-level runs to recover most of the single-plane
    # speed while keeping the per-level TAKEOFF capacity gain (which is independent). No effect when
    # n_levels == 1 (a single plane has no rungs to climb).
    vertical_edges: bool = False

    # --- denial budgets ---
    max_ground_delay_s: float = 3600.0
    max_detour_factor: float = 100.0     # deny if flown/straight-line exceeds this

    # --- demand / horizon ---
    horizon_s: float = 14_400.0        # 4 h
    lam_per_hour: float = 200.0
    seed: int = 0

    # --- planner selection (pluggable; DEFAULT = A* → shortcut → MILP → shortcut sandwich) ---
    planner: str = "astar"  # "straight"|"rrt"|"lazy"|"astar"|"milp"|"astar_milp"|...

    # --- fixed terminal exit lanes (issue #18); A* only ---
    # When True, A* (and astar_shortcut) routes shared-terminal takeoff/landing through the hub's
    # boundary-hex lanes and deconflicts same-hub launches by exact cell occupancy (is_blocked), killing
    # same-hub exit-lane CONFLICT_FILED. False ⇒ the legacy A* fold/exit_clear path. Other planners
    # (milp/opt/rrt) don't route through lanes — the flag only tags their hub boxes. Default on (#18).
    fixed_exit_lanes: bool = True

    # --- always-active terminal airspace (foreign-transit isolation); A* only ---
    # When True, every hub's column + exit lanes are permanently reserved as a FOREIGN-no-fly zone for
    # the whole horizon (not just during dwell windows): foreign cruise traffic routes AROUND the
    # terminal (extra air detour) instead of crossing it and ground-blocking same-hub takeoffs. Converts
    # foreign-transit GROUND delay into airspace-density AIR delay. The static column spans every flight
    # level (the [ground, ceiling] tube). The demand generator drops deliveries whose customer falls
    # inside a foreign column (unreachable).
    terminal_airspace_always_active: bool = False

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
        """Seconds to climb ground → the preferred cruise level (75 / 6 = 12.5 s).

        This is the single-plane planners' climb time; A* uses :meth:`climb_time_to` per flight level.
        """
        return (self.cruise_level_m - self.ground_level_m) / self.climb_rate_mps

    @property
    def n_steps(self) -> int:
        """Number of discrete timesteps in the horizon."""
        return int(self.horizon_s / self.dt_s)

    # ----- discrete flight levels (A*'s altitude ladder) -----
    @property
    def n_levels(self) -> int:
        """Number of discrete cruise levels A* can route on."""
        return len(self.flight_levels_m)

    def level_z(self, L: int) -> float:
        """Altitude (m) of flight-level index ``L``."""
        return self.flight_levels_m[L]

    def nearest_level(self, z: float) -> int:
        """Index of the flight level closest to altitude ``z``."""
        return min(range(self.n_levels), key=lambda i: abs(self.flight_levels_m[i] - z))

    def climb_time_to(self, z: float) -> float:
        """Seconds to climb ground → ``z`` (or descend ``z`` → ground) at the climb rate."""
        return (z - self.ground_level_m) / self.climb_rate_mps

    def climb_steps_to(self, z: float, dt: float | None = None) -> int:
        """Discrete timesteps to climb ground → ``z`` (≥ 1)."""
        dt = self.dt_s if dt is None else dt
        return max(1, int(math.ceil(self.climb_time_to(z) / dt)))

    @staticmethod
    def equidistant_levels(z_lo: float, z_hi: float, n: int) -> tuple[float, ...]:
        """``n`` evenly spaced levels in [``z_lo``, ``z_hi``] inclusive (n ≥ 1)."""
        if n <= 1:
            return (z_lo,)
        step = (z_hi - z_lo) / (n - 1)
        return tuple(z_lo + step * i for i in range(n))

    def __post_init__(self) -> None:
        """Validate the flight-level ladder (frozen dataclass — raise only, never mutate)."""
        lv = self.flight_levels_m
        if not lv:
            raise ValueError("flight_levels_m must be non-empty")
        if list(lv) != sorted(lv) or len(set(lv)) != len(lv):
            raise ValueError(f"flight_levels_m must be strictly ascending: {lv}")
        half = self.corridor_height_m / 2.0
        if lv[0] - half < self.ground_level_m - 1e-9:
            raise ValueError(
                f"lowest level {lv[0]} box dips below ground_level_m {self.ground_level_m}")
        if lv[-1] + half > self.airspace_ceiling_m + 1e-9:
            raise ValueError(
                f"top level {lv[-1]} box exceeds airspace_ceiling_m {self.airspace_ceiling_m}")
        for a, b in zip(lv, lv[1:]):
            if (b - a) <= self.corridor_height_m + 1e-9:
                raise ValueError(
                    f"levels {a},{b} gap {b - a} <= corridor_height_m {self.corridor_height_m}; "
                    "adjacent level boxes would overlap in z")
        if not (self.ground_level_m <= self.cruise_level_m <= self.airspace_ceiling_m):
            raise ValueError(
                f"cruise_level_m {self.cruise_level_m} outside "
                f"[{self.ground_level_m}, {self.airspace_ceiling_m}]")
