"""Optimizer-independent route and stopping contracts for Zeta search."""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
import math
import operator
from ..route_domain import RowKey
from ..route_cost import CostModel

Cell = tuple[int, int]

def _index_field(value: object, name: str) -> int:
    """Require integral route coordinates and choices."""
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc

@dataclass(frozen=True)
class Route:
    """One timed route with physical cost and coefficient-one reservation claims.

    Attributes
    ------------
    - flight_id (int): request identity.
    - departure_step (int): takeoff step before climb and terminal-lane traversal.
    - level (int): configured cruise-level index.
    - origin_lane_idx (int | None): origin terminal lane, or None for a customer.
    - dest_lane_idx (int | None): destination terminal lane, or None for a customer.
    - cell_path (tuple[Cell, ...]): origin-to-destination candidate waypoints.
    - physical_cost (float): canonical cost of ground delay and excess flight.
    - claims (frozenset[RowKey]): coefficient-one union of filed reservation rows.
    """

    flight_id: int
    departure_step: int
    level: int
    origin_lane_idx: int | None
    dest_lane_idx: int | None
    cell_path: tuple[Cell, ...]
    physical_cost: float
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


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """Objective and optional threshold for bounded route discovery.

    Attributes
    ------------
    - model (CostModel): physical ground/air objective weights.
    - epsilon (float): finite positive comparison tolerance.
    - target_cost (float | None): optional cost threshold; None seeks the best route.
    - improvement_guard (Callable | None): optional extra numerical check on physical
      and reservation costs before an early return, supplied by an optimizer adapter.
    """

    model: CostModel = CostModel()
    epsilon: float = 1e-6
    target_cost: float | None = None
    improvement_guard: Callable[[float, float], bool] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Reject invalid numerical stopping controls before entering native code."""
        if any(not math.isfinite(weight) or weight <= 0.0
               for weight in (self.model.ground_weight, self.model.air_weight)):
            raise ValueError("physical cost weights must be finite and positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if self.target_cost is not None and not math.isfinite(self.target_cost):
            raise ValueError("target_cost must be finite or None")

    def accepts(self, physical_cost: float, reservation_cost: float) -> bool:
        """Apply an adapter's optional extra check without changing route ranking."""
        return self.improvement_guard is None or self.improvement_guard(physical_cost, reservation_cost)
