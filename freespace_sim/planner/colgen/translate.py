"""Column-generation route type and adapters to shared simulator geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...config import SimConfig
from ...types import FlightRequest, OperationalIntent, Terminal, Vec
from ..route_geometry import (
    _index_field as _index_field,
    _selected_lane as _selected_lane,
    _retime_lattice_reservation as _retime_lattice_reservation,
    lattice_centerline as lattice_centerline,
    route_to_corners,
    route_to_intent,
)

if TYPE_CHECKING:
    from .network import RowKey

Cell = tuple[int, int]


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
        """Coerce every field to its canonical immutable form, raising on bad input.

        Pricing reconstructs paths as lists; freezing the cells to a tuple and the claims to a
        frozenset makes a column a safe dict key and stops a claim coefficient from carrying a
        duplicate.

        Parameters
        ------------
        - none: reads and rewrites the frozen dataclass fields in place.

        Return
        --------
        - output (None): rewrites fields via ``object.__setattr__``; raises ``TypeError`` on a
          non-integer field or cell, ``ValueError`` on a path shorter than two cells.
        """
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



def column_to_corners(
    col: Column,
    req: FlightRequest,
    cfg: SimConfig,
    *,
    _validate: bool = True,
    _any_angle: bool = False,
) -> tuple[list[Vec], float, float, float, Terminal | None, Terminal | None]:
    """Convert a discrete column to hex-centre corners and A*-equivalent timing metadata.

    The returned delays deliberately differ when ``req.t_departure`` is not on the integer clock:
    ``build_g_delay`` anchors the takeoff cylinder at ``departure_step * dt``; the reported delay
    counts only ground-hold steps after ``ceil(t_departure / dt)``.  ``corridor_t0`` includes the
    quantised climb and, for a hub origin, the chosen lane's quantised traverse.

    Parameters
    ------------
    - col (Column): the discrete route to expand; its cells, level, and departure step are read.
    - req (FlightRequest): the flight this column serves, checked to match ``col.flight_id``.
    - cfg (SimConfig): supplies the clock, flight levels, and lattice/terminal geometry.
    - _validate (bool): enforce route feasibility; False is internal diagnostic construction.
      Required types, indices and supported terminal configuration are still checked.
    - _any_angle (bool): internal prototype mode permitting nonadjacent candidate waypoints;
      endpoint and lane validation are unchanged.

    Return
    --------
    - output (tuple[list[Vec], float, float, float, Terminal | None, Terminal | None]): the
      hex-centre corner polyline, build-anchor ground delay (s), reported ground delay (s),
      corridor start stamp (s), and the origin/destination terminals (``None`` for a customer
      endpoint).  Raises ``ValueError`` when the column does not match the request or lattice,
      and ``NotImplementedError`` for terminal endpoints without the required colgen-v1 flags.
    """

    return route_to_corners(col, req, cfg, _validate=_validate, _any_angle=_any_angle)


def column_to_intent(
    col: Column,
    req: FlightRequest,
    cfg: SimConfig,
    solve_share_s: float = 0.0,
    *,
    _validate: bool = True,
    _any_angle: bool = False,
) -> OperationalIntent:
    """Build the exact reservation and reported metrics represented by ``col``.

    Parameters
    ------------
    - col (Column): the route to realise.
    - req (FlightRequest): the flight this column serves.
    - cfg (SimConfig): supplies geometry, clock, and cost weights.
    - solve_share_s (float): this column's share of solve wall time, recorded on the intent.
    - _validate (bool): enforce route and budget checks; False is internal diagnostic
      construction, with identical geometry, timing and cost calculations.
    - _any_angle (bool): internal prototype mode using nominal-speed arbitrary segments,
      retaining the existing departure, climb, terminal-lane and objective rules.

    Return
    --------
    - output (OperationalIntent): an ACCEPTED intent carrying the reservation volumes, timed
      centreline, delays, and cost; or a REJECTED intent (``BUDGET_EXCEEDED``) when the ground
      hold exceeds ``max_ground_delay_s`` or the detour exceeds ``max_detour_factor``.
    """

    return route_to_intent(
        col, req, cfg, solve_share_s, _validate=_validate, _any_angle=_any_angle,
        planner="colgen",
    )
