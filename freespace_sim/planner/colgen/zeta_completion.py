"""Optional independent completion relaxations, combined without double charging."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import time
from typing import Any, TYPE_CHECKING

from .network import RowKey
from .objective import CostModel

if TYPE_CHECKING:
    from .zeta import Cell, ZetaGraph


@dataclass(slots=True)
class CompletionBounds:
    """Complete safe components; a declined component leaves the ordinary bound intact."""

    components: tuple[Any, ...]
    names: tuple[str, ...]
    build_s: float
    table_states: int

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
        from .zeta_arrival_bound import ArrivalBound
        from .zeta_crossing_bound import CrossingBound
        from .zeta_resource_bound import ResourceBound
        from .zeta_selected_bound import SelectedPriceBound

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
