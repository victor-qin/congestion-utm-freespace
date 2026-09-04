"""Translate column-generation paths into ledger-ready operational intents.

The pricing network uses an integer clock, while reservations and metrics use wall-clock
seconds.  Keeping that conversion here gives a column one unambiguous interpretation and,
in particular, preserves A*'s distinction between the delay used to anchor a reservation and
the delay reported to users (which excludes departure-time rounding).
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from ...config import SimConfig
from ...cost import endpoint_altitude_change_m, trajectory_cost
from ...types import (
    DenialReason,
    FlightRequest,
    IntentStatus,
    OperationalIntent,
    Terminal,
    TimedPoint,
    Vec,
    as_terminal,
)
from ...volumes import (
    Volume4D,
    build_reservation_from_corners,
    column_dwell_s,
    enroute_detour_m,
    enroute_flown_m,
    enroute_reference_m,
)
from .. import hexgrid as hg

if TYPE_CHECKING:
    from .network import RowKey

Cell = tuple[int, int]


def _index_field(value, name: str) -> int:
    """Normalize one discrete-network field without truncating fractions."""

    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"column {name} must be an integer") from exc


@dataclass(frozen=True)
class Column:
    """One timed route offered to the restricted master problem.

    ``departure_step`` is the takeoff step, before the quantised climb and optional terminal-lane
    traverse.  ``cell_path`` begins at the origin lane/customer cell and ends at the destination
    lane/customer cell.  Claims are populated only by ``network.column_claims``, the single owner
    of capacity-row membership.
    """

    flight_id: int
    departure_step: int
    level: int
    origin_lane_idx: int | None
    dest_lane_idx: int | None
    cell_path: tuple[Cell, ...]
    delay_s: float
    claims: frozenset[RowKey] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Pricing naturally reconstructs paths as lists.  Canonical immutable containers make
        # columns safe dictionary values and ensure claim coefficients cannot contain duplicates.
        object.__setattr__(self, "flight_id", _index_field(self.flight_id, "flight_id"))
        object.__setattr__(
            self,
            "departure_step",
            _index_field(self.departure_step, "departure_step"),
        )
        object.__setattr__(self, "level", _index_field(self.level, "level"))
        if self.origin_lane_idx is not None:
            object.__setattr__(
                self,
                "origin_lane_idx",
                _index_field(self.origin_lane_idx, "origin_lane_idx"),
            )
        if self.dest_lane_idx is not None:
            object.__setattr__(
                self,
                "dest_lane_idx",
                _index_field(self.dest_lane_idx, "dest_lane_idx"),
            )
        try:
            path = tuple(
                (_index_field(q, "cell q coordinate"), _index_field(r, "cell r coordinate"))
                for q, r in self.cell_path
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("column cells must be integer (q, r) pairs") from exc
        object.__setattr__(self, "cell_path", path)
        object.__setattr__(self, "claims", frozenset(self.claims))
        if len(self.cell_path) < 2:
            raise ValueError("a column path must contain at least two cells")

def _selected_lane(
    cell: Cell, lane_idx: int | None, lanes: list[hg.Lane], endpoint: str
) -> hg.Lane:
    """Validate and return the terminal lane encoded by a column endpoint."""
    if lane_idx is None:
        raise ValueError(f"{endpoint}_lane_idx is required for a terminal endpoint")
    lane_idx = _index_field(lane_idx, f"{endpoint}_lane_idx")
    if not 0 <= lane_idx < len(lanes):
        raise ValueError(
            f"{endpoint}_lane_idx {lane_idx} is outside the available lane range [0, {len(lanes)})"
        )
    lane = lanes[lane_idx]
    if cell != lane.cell:
        raise ValueError(
            f"column {endpoint} cell {cell} does not match selected terminal lane {lane.cell}"
        )
    return lane


def column_to_corners(
    col: Column,
    req: FlightRequest,
    cfg: SimConfig,
) -> tuple[list[Vec], float, float, float, Terminal | None, Terminal | None]:
    """Convert a discrete column to hex-centre corners and A*-equivalent timing metadata.

    The returned delays deliberately differ when ``req.t_departure`` is not on the integer clock:
    ``build_g_delay`` anchors the takeoff cylinder at ``departure_step * dt``; the reported delay
    counts only ground-hold steps after ``ceil(t_departure / dt)``.  ``corridor_t0`` includes the
    quantised climb and, for a hub origin, the chosen lane's quantised traverse.
    """
    flight_id = _index_field(col.flight_id, "flight_id")
    departure_step = _index_field(col.departure_step, "departure_step")
    level = _index_field(col.level, "level")
    if flight_id != req.flight_id:
        raise ValueError(f"column flight_id {col.flight_id} does not match request {req.flight_id}")
    if not 0 <= level < len(cfg.flight_levels_m):
        raise ValueError(
            f"column level {col.level} is outside the flight-level range "
            f"[0, {len(cfg.flight_levels_m)})"
        )
    if len(col.cell_path) < 2:
        # Defensive backstop for callers that bypassed the frozen dataclass constructor.
        raise ValueError("a column path must contain at least two cells")

    dt = cfg.dt_s
    base = int(math.ceil(req.t_departure / dt))
    if departure_step < base:
        raise ValueError(
            f"column departs at step {col.departure_step}, before request base step {base}"
        )

    origin_term = as_terminal(req.origin_terminal)
    dest_term = as_terminal(req.dest_terminal)
    if not cfg.terminal_airspace_always_active and (
        origin_term is not None or dest_term is not None
    ):
        raise NotImplementedError(
            "colgen v1 requires terminal_airspace_always_active=True for terminal endpoints"
        )
    if not cfg.fixed_exit_lanes and (origin_term is not None or dest_term is not None):
        raise NotImplementedError("colgen v1 requires fixed_exit_lanes=True for terminal endpoints")
    lane_steps = 0
    if origin_term is not None:
        origin_lanes = hg.terminal_lanes(req.origin, origin_term, cfg)
        lane_steps = _selected_lane(
            col.cell_path[0], col.origin_lane_idx, origin_lanes, "origin"
        ).steps
    elif col.origin_lane_idx is not None:
        raise ValueError("origin_lane_idx must be None for a non-terminal endpoint")

    if dest_term is not None:
        dest_lanes = hg.terminal_lanes(req.dest, dest_term, cfg)
        _selected_lane(col.cell_path[-1], col.dest_lane_idx, dest_lanes, "dest")
    elif col.dest_lane_idx is not None:
        raise ValueError("dest_lane_idx must be None for a non-terminal endpoint")

    z = cfg.flight_levels_m[level]
    radius = hg.circumradius(cfg)
    if origin_term is None:
        expected = hg.enu_to_axial(float(req.origin[0]), float(req.origin[1]), radius)
        if col.cell_path[0] != expected:
            raise ValueError(
                f"column origin cell {col.cell_path[0]} does not match customer cell {expected}"
            )
    if dest_term is None:
        expected = hg.enu_to_axial(float(req.dest[0]), float(req.dest[1]), radius)
        if col.cell_path[-1] != expected:
            raise ValueError(
                f"column destination cell {col.cell_path[-1]} does not match customer cell {expected}"
            )
    for first, second in zip(col.cell_path, col.cell_path[1:]):
        if hg.hex_distance(first, second) != 1:
            raise ValueError(
                f"column paths must use adjacent lateral hex hops; found {first} -> {second}"
            )
    corners = [np.array([*hg.hex_center(q, r, radius), z], dtype=float) for q, r in col.cell_path]

    build_g_delay = departure_step * dt - req.t_departure
    report_g_delay = (departure_step - base) * dt
    takeoff_steps = cfg.climb_steps_to(z)
    corridor_t0 = (departure_step + takeoff_steps + lane_steps) * dt
    return (
        corners,
        build_g_delay,
        report_g_delay,
        corridor_t0,
        origin_term,
        dest_term,
    )


def _retime_lattice_reservation(
    volumes: list[Volume4D],
    centerline: list[TimedPoint],
    corners: list[Vec],
    corridor_t0: float,
    origin_t0: float,
    origin_dwell_s: float,
    destination_dwell_s: float,
    cfg: SimConfig,
) -> tuple[list[Volume4D], list[TimedPoint]]:
    """Stamp resampled geometry onto the pricing network's exact lattice clock.

    ``build_reservation_from_corners`` deliberately times arbitrary polylines from their
    floating-point lengths.  A colgen corner pair, however, is one discrete lateral edge and its
    row claims therefore represent exactly one ``dt``.  A rare round-off-induced resample can make
    the builder accumulate a few femtoseconds across that edge.  Reusing its subdivision arithmetic
    here preserves every shape and terminal tag while distributing the fixed edge duration across
    its sub-boxes.  In particular, every original corner boundary and the landing cylinder start on
    the exact discrete arrival clock rather than on an accumulated distance clock.
    """
    nsubs: list[int] = []
    segment_len = cfg.corridor_segment_len_m
    for first, second in zip(corners, corners[1:]):
        dx = float(second[0]) - float(first[0])
        dy = float(second[1]) - float(first[1])
        dz = float(second[2]) - float(first[2])
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        nsubs.append(max(1, math.ceil(length / segment_len)))

    edge_count = sum(nsubs)
    if len(volumes) != edge_count + 2 or len(centerline) != edge_count + 1:
        raise RuntimeError(
            "reservation builder output does not match the column's lattice subdivisions"
        )

    corridor_start_step = int(round(corridor_t0 / cfg.dt_s))
    exact_corridor_t0 = corridor_start_step * cfg.dt_s
    retimed_volumes = [
        replace(
            volumes[0],
            t_start=origin_t0,
            t_end=origin_t0 + cfg.hover_time_s + origin_dwell_s,
        )
    ]
    retimed_centerline: list[TimedPoint] = [(centerline[0][0], exact_corridor_t0)]
    edge_index = 0
    for hop_index, nsub in enumerate(nsubs):
        # Multiplying the absolute integer step mirrors ``column_claims`` exactly;
        # repeated float addition would reintroduce drift for non-binary ``dt_s``.
        hop_t0 = (corridor_start_step + hop_index) * cfg.dt_s
        hop_t1 = (corridor_start_step + hop_index + 1) * cfg.dt_s
        for sub_index in range(nsub):
            # Pin both original-hop boundaries explicitly; interpolate only interior cuts.
            raw_t0 = hop_t0 if sub_index == 0 else hop_t0 + cfg.dt_s * sub_index / nsub
            raw_t1 = hop_t1 if sub_index + 1 == nsub else hop_t0 + cfg.dt_s * (sub_index + 1) / nsub
            edge = volumes[edge_index + 1]
            retimed_volumes.append(
                replace(
                    edge,
                    # Leading-only pad, matching ``corridor_segment_volume`` exactly: this
                    # ``replace`` overwrites the builder's windows wholesale, so any drift here
                    # silently re-files boxes wider than the capacity rows measured from the builder.
                    t_start=raw_t0,
                    t_end=raw_t1 + cfg.time_buffer_s,
                )
            )
            point, _old_time = centerline[edge_index + 1]
            retimed_centerline.append((point, raw_t1))
            edge_index += 1

    arrival_t = (corridor_start_step + len(nsubs)) * cfg.dt_s
    destination = volumes[-1]
    retimed_volumes.append(
        replace(
            destination,
            t_start=arrival_t,
            # Match ``hover_reservation``'s operation order exactly so parity
            # holds even when the dwell duration is not binary-exact.
            t_end=arrival_t + cfg.hover_time_s + destination_dwell_s,
        )
    )
    # The final sub-box timestamp is the same expression in normal operation; pin it to the one
    # arrival value used by the destination cylinder to keep their shared boundary bit-identical.
    final_point, _old_time = retimed_centerline[-1]
    retimed_centerline[-1] = (final_point, arrival_t)
    return retimed_volumes, retimed_centerline


def column_to_intent(
    col: Column,
    req: FlightRequest,
    cfg: SimConfig,
    solve_share_s: float = 0.0,
) -> OperationalIntent:
    """Build the exact reservation and reported metrics represented by ``col``."""
    (
        corners,
        build_g_delay,
        report_g_delay,
        corridor_t0,
        origin_term,
        dest_term,
    ) = column_to_corners(col, req, cfg)
    level = _index_field(col.level, "level")
    departure_step = _index_field(col.departure_step, "departure_step")
    z = cfg.flight_levels_m[level]

    base = int(math.ceil(req.t_departure / cfg.dt_s))
    max_ground_steps = int(math.floor(cfg.max_ground_delay_s / cfg.dt_s + 1e-12))
    if departure_step > base + max_ground_steps:
        return OperationalIntent(
            request=req,
            status=IntentStatus.REJECTED,
            denial_reason=DenialReason.BUDGET_EXCEEDED,
            planner="colgen",
            solve_time_s=solve_share_s,
        )

    volumes, centerline, _cum_horiz, cum_dz = build_reservation_from_corners(
        corners,
        req.origin,
        req.dest,
        req.t_departure,
        build_g_delay,
        cfg,
        origin_term=origin_term,
        dest_term=dest_term,
        corridor_t0=corridor_t0,
    )
    volumes, centerline = _retime_lattice_reservation(
        volumes,
        centerline,
        corners,
        corridor_t0,
        departure_step * cfg.dt_s,
        column_dwell_s(req.origin, origin_term, cfg, z),
        column_dwell_s(req.dest, dest_term, cfg, z),
        cfg,
    )
    reference = enroute_reference_m(req.origin, req.dest, origin_term, dest_term, cfg)
    flown = enroute_flown_m(
        [point for point, _time in centerline],
        req.origin,
        req.dest,
        origin_term,
        dest_term,
        cfg,
    )
    if reference > 1e-9 and flown / reference > cfg.max_detour_factor:
        return OperationalIntent(
            request=req,
            status=IntentStatus.REJECTED,
            denial_reason=DenialReason.BUDGET_EXCEEDED,
            planner="colgen",
            solve_time_s=solve_share_s,
        )
    detour = enroute_detour_m(flown, reference)
    intent = OperationalIntent(
        request=req,
        status=IntentStatus.ACCEPTED,
        volumes=volumes,
        centerline=centerline,
        ground_delay_s=report_g_delay,
        air_hold_s=0.0,
        air_detour_m=detour,
        lattice_overhead_m=hg.lattice_overhead_m(col.cell_path, cfg.corridor_segment_len_m, detour),
        altitude_change_m=endpoint_altitude_change_m(z, z, cum_dz, cfg),
        planner="colgen",
        solve_time_s=solve_share_s,
    )
    intent.cost = trajectory_cost(intent, cfg)
    return intent
