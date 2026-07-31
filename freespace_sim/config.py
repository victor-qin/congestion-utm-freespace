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
    # Altitude is defined by ONE knob: ``flight_levels_m`` (below). ``cruise_level_m`` / ``z_min_m`` /
    # ``z_max_m`` are NOT stored; they are DERIVED @properties (see the DERIVED section): cruise = the
    # ladder's middle level (straight/decoupled), and the MILP continuous band [z_min_m, z_max_m] = the
    # ladder's floor→top. A single-level ladder collapses the band to that one plane.
    # Regulated airspace ceiling: every hover/terminal column spans [ground_level_m, airspace_ceiling_m].
    airspace_ceiling_m: float = 125.0
    # A*'s discrete cruise levels — the SINGLE altitude knob (cruise / z-band derive from it). Strictly
    # ascending; adjacent gaps must EXCEED corridor_height_m (so neighbouring level boxes don't touch in z
    # and stay FCL-disjoint), and the top/bottom boxes (level ± corridor_height_m/2) must fit within
    # [ground_level_m, airspace_ceiling_m]. Set ``flight_levels_m=(z,)`` (+ matching ceiling) for one plane.
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
    # ONE currency: cost per SECOND. Every A* edge advances the clock by an integer number of dt
    # steps, so seconds are the only basis on which the four levers are comparable. The per-METRE
    # weights the planners actually multiply by are DERIVED (see the DERIVED section), which keeps
    # the ratios below invariant under any dt_s / nominal_speed_mps / climb_rate_mps.
    # Storing lateral/altitude per-metre (as this did before) silently scaled them by pitch=120 m
    # and climb_rate*dt=24 m while ground/hold were scaled by dt=4 s, so the advertised 1:3:3:4
    # was really 1:90:3:24 — one hex step cost as much as 360 s of ground delay and no detour or
    # climb was ever rational. Per step, these now read exactly 1 : 3 : 3 : 4.
    cost_ground_delay_per_s: float = 1.0        # wait on the pad          (1x, the numeraire)
    cost_air_lateral_per_s: float = 3.0         # cruise flight            (3x)
    cost_air_hold_per_s: float = 3.0            # loiter/hover mid-route   (3x)
    cost_altitude_change_per_s: float = 4.0     # climb/descend            (4x)

    # --- denial budgets ---
    max_ground_delay_s: float = 3600.0
    max_detour_factor: float = 100.0     # deny if flown/straight-line exceeds this

    # --- demand / horizon ---
    horizon_s: float = 14_400.0        # 4 h
    # Active demand-generation duration. None preserves the legacy contract: demand is generated over
    # the whole configured horizon. Density studies use a shorter offered-load window inside a longer
    # planner envelope; the realized run still continues through the final landing without clipping.
    demand_duration_s: float | None = None
    lam_per_hour: float = 200.0
    seed: int = 0

    # --- planner selection (pluggable; DEFAULT = A* → shortcut → MILP → shortcut sandwich) ---
    planner: str = "astar"  # "straight"|"astar"|"astar_shortcut"|"milp"|"astar_milp"|...

    # --- fixed terminal exit lanes (issue #18); A* only ---
    # When True, A* (and astar_shortcut) routes shared-terminal takeoff/landing through the hub's
    # boundary-hex lanes and deconflicts same-hub launches by exact cell occupancy (is_blocked), killing
    # same-hub exit-lane CONFLICT_FILED. False ⇒ the legacy A* fold/exit_clear path. Other planners
    # (milp/straight) don't route through lanes — the flag only tags their hub boxes. Default on (#18).
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

    # ----- cost weights per METRE, DERIVED from the per-second knobs (not stored) -----
    # The planners charge lateral/vertical travel by LENGTH (``c_lat * pitch``, ``c_alt * dz``),
    # but the levers are only commensurable in TIME. Converting here — rather than storing a
    # per-metre number — is what makes the 1:3:3:4 step ratio hold for any speed/climb-rate/dt.
    @property
    def cost_air_lateral_per_m(self) -> float:
        """Cost of one flown metre = per-second weight ÷ cruise speed (one metre takes 1/v s).

        Keeps every planner's ``cost_air_lateral_per_m * pitch`` expression correct untouched:
        pitch is ``nominal_speed_mps * dt_s``, so the product collapses to ``c_lat_per_s * dt``.
        """
        return self.cost_air_lateral_per_s / self.nominal_speed_mps

    @property
    def cost_altitude_change_per_m(self) -> float:
        """Cost of one climbed/descended metre = per-second weight ÷ climb rate (1/climb_rate s)."""
        return self.cost_altitude_change_per_s / self.climb_rate_mps

    # ----- altitude, DERIVED from flight_levels_m (not stored) -----
    @property
    def cruise_level_m(self) -> float:
        """Single-plane planners' cruise altitude (straight/decoupled) — the ladder's middle level.

        Derived, never stored: ``flight_levels_m`` is the single source of truth. A* deconflicts on the
        discrete ladder and MILP in the ``[z_min_m, z_max_m]`` band; only straight/decoupled pin here.
        """
        return self.flight_levels_m[len(self.flight_levels_m) // 2]

    @property
    def z_min_m(self) -> float:
        """MILP continuous cruise-band floor = the ladder's lowest level. A single-level ladder ⇒ z_min==z_max."""
        return self.flight_levels_m[0]

    @property
    def z_max_m(self) -> float:
        """MILP continuous cruise-band ceiling = the ladder's highest level. A single-level ladder ⇒ z_min==z_max."""
        return self.flight_levels_m[-1]

    @property
    def climb_time_s(self) -> float:
        """Seconds to climb ground → the single-plane cruise level at climb_rate (e.g. 70/6 ≈ 11.7 s).

        This is the single-plane planners' climb time; A* uses :meth:`climb_time_to` per flight level.
        """
        return (self.cruise_level_m - self.ground_level_m) / self.climb_rate_mps

    @property
    def n_steps(self) -> int:
        """Number of discrete timesteps in the horizon."""
        return int(self.horizon_s / self.dt_s)

    @property
    def effective_demand_duration_s(self) -> float:
        """Active demand duration, defaulting to the full simulation horizon."""
        return self.horizon_s if self.demand_duration_s is None else self.demand_duration_s

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
        """Validate the flight-level ladder (frozen dataclass — raise only, never mutate).

        ``flight_levels_m`` is the single source of truth for altitude; the single-plane planners' cruise +
        sampling band are DERIVED from it (the ``cruise_level_m`` / ``z_min_m`` / ``z_max_m`` properties).
        """
        if self.demand_duration_s is not None:
            if self.demand_duration_s <= 0.0:
                raise ValueError("demand_duration_s must be positive")
            if self.demand_duration_s > self.horizon_s:
                raise ValueError(
                    f"demand_duration_s {self.demand_duration_s} exceeds horizon_s {self.horizon_s}")
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
