"""Optional independent completion relaxations, combined without double charging."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import time
from typing import Any, TYPE_CHECKING

from ..route_domain import RowKey
from ..route_cost import CostModel

if TYPE_CHECKING:
    from .search import Cell, ZetaGraph


@dataclass(slots=True)
class CompletionBounds:
    """Complete safe components; a declined component leaves the ordinary bound intact."""

    components: tuple[Any, ...]
    names: tuple[str, ...]
    build_s: float
    table_states: int
    _rows: tuple[RowKey, ...] = ()

    @classmethod
    def build(
        cls, graph: ZetaGraph, prices: Mapping[RowKey, float], model: CostModel,
        deadline: float | None = None,
    ) -> CompletionBounds:
        """Prepare requested relaxations once for the current frozen price vector.

        Parameters
        ------------
        - graph (ZetaGraph): unchanged route and timing universe with selected bound names.
        - prices (Mapping[RowKey, float]): current finite signed dual prices.
        - model (CostModel): original weighted physical objective.
        - deadline (float | None): absolute monotonic cutoff for preparation and search.

        Return
        --------
        - output (CompletionBounds): complete components and inclusive preparation cost.
        """
        from .arrival_bound import ArrivalBound
        from .crossing_bound import CrossingBound
        from .resource_bound import ResourceBound
        from .selected_bound import SelectedPriceBound

        builders = {"arrival": ArrivalBound, "resource": ResourceBound,
                    "selected": SelectedPriceBound, "crossing": CrossingBound}
        started = time.perf_counter()
        components, names = [], []
        for name in graph.options.completion_bounds:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            bound = builders[name].build(graph, prices, model, deadline)
            if bound is not None:
                components.append(bound)
                names.append(name)
        return cls(tuple(components), tuple(names), time.perf_counter() - started,
                   sum(bound.table_states for bound in components))

    def bind(self, rows: tuple[RowKey, ...]) -> None:
        """Let components with a compiled id path pack per-row data for one row table."""
        for bound in self.components:
            if hasattr(bound, "floor_ids"):
                bound.bind(rows)
        self._rows = rows

    def floor_ids(
        self, cell: Cell, goal: Cell, stamp: float, length: float,
        positive_paid: float, ground: float, negative: float, fold_m: float | None,
        paid: Any, terminal: bool = False,
    ) -> float:
        """Same maximum as floor() for unique table-row ids from the row tuple given to bind().

        Parameters
        ------------
        - cell, goal, stamp, length, positive_paid, ground, negative, fold_m, terminal:
          as floor().
        - paid (np.ndarray): unique live paid ids; components without an id path receive
          the matching row keys, converted once.

        Return
        --------
        - output (float): safe full completion floor; negative infinity when none applies.
        """
        best = -math.inf
        keys = None
        for bound in self.components:
            if hasattr(bound, "floor_ids"):
                value = bound.floor_ids(cell, goal, stamp, length, positive_paid, ground,
                                        negative, fold_m, paid, terminal)
            else:
                if keys is None:
                    keys = tuple(self._rows[int(row)] for row in paid)
                value = bound.floor(cell, goal, stamp, length, positive_paid, ground,
                                    negative, fold_m, keys, terminal)
            best = max(best, value)
        return best

    def floor(
        self, cell: Cell, goal: Cell, stamp: float, length: float,
        positive_paid: float, ground: float, negative: float, fold_m: float | None,
        paid: Iterable[RowKey], terminal: bool = False,
    ) -> float:
        """Take the largest complete floor; component reservation charges may overlap.

        Parameters
        ------------
        - cell (Cell): current candidate waypoint.
        - goal (Cell): this prefix's destination attachment.
        - stamp (float): actual arrival timestamp at cell.
        - length (float): sequentially accumulated physical length.
        - positive_paid (float): prefix's positive reservation charge.
        - ground (float): chosen departure's ground cost.
        - negative (float): all possible negative reservation credit.
        - fold_m (float | None): proven additive endpoint distance, or unsupported.
        - paid (Iterable[RowKey]): unique live paid reservations.
        - terminal (bool): whether the destination endpoint role was already selected.

        Return
        --------
        - output (float): safe full completion floor; negative infinity when none applies.
        """
        if not self.components:
            return -math.inf
        paid = tuple(paid)
        return max((bound.floor(cell, goal, stamp, length, positive_paid, ground,
                                negative, fold_m, paid, terminal)
                    for bound in self.components), default=-math.inf)
