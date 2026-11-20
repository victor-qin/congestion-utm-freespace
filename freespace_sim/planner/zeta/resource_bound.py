"""Remaining-distance corrections for relaxed future reservation prices."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import time
from typing import TYPE_CHECKING

from ..route_domain import RowKey
from ..route_cost import CostModel
from .bound import FuturePriceBound

if TYPE_CHECKING:
    from .search import Cell, ZetaGraph


@dataclass(slots=True)
class ResourceBound:
    """Several distance-penalized relaxations sharing the real route's length allowance."""

    graph: ZetaGraph
    model: CostModel
    tables: tuple[FuturePriceBound, ...]
    build_s: float
    table_states: int

    @classmethod
    def build(
        cls, graph: ZetaGraph, prices: Mapping[RowKey, float], model: CostModel,
        deadline: float | None = None,
    ) -> ResourceBound | None:
        """Build two valid price-distance relaxations without changing permitted routes.

        Parameters
        ------------
        - graph (ZetaGraph): exact geometry, clocks and route length allowance.
        - prices (Mapping[RowKey, float]): immutable current dual prices, including negatives.
        - model (CostModel): unchanged physical objective weights.
        - deadline (float | None): monotonic preparation cutoff.

        Return
        --------
        - output (ResourceBound | None): available complete tables, or a safe decline.
        """
        started = time.perf_counter()
        positive = math.fsum(p for p in prices.values() if p > 0.0)
        if not math.isfinite(positive) or positive <= 0.0 or graph.max_length <= 0.0:
            return None
        # Scale penalties to one physical second of detour; a second scale probes a
        # different supporting line of the price-versus-length tradeoff.
        unit = model.air_weight / graph.cfg.nominal_speed_mps
        if unit <= 0.0:
            unit = positive / graph.max_length
        if not math.isfinite(unit) or unit <= 0.0:
            return None
        tables = []
        for penalty in (unit * .25, unit):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            table = FuturePriceBound.build(graph, prices, deadline, distance_penalty=penalty)
            if table is not None:
                tables.append(table)
        if not tables:
            return None
        return cls(graph, model, tuple(tables), time.perf_counter() - started,
                   sum(t.table_states for t in tables))

    def floor(
        self, cell: Cell, goal: Cell, stamp: float, length: float,
        positive_paid: float, ground: float, negative: float, fold_m: float | None,
        paid: Iterable[RowKey], terminal: bool = False,
    ) -> float:
        """Correct each penalized table by an upper bound on remaining flown distance.

        Parameters
        ------------
        - cell (Cell): current waypoint.
        - goal (Cell): destination attachment for this prefix.
        - stamp (float): actual prefix arrival time.
        - length (float): sequentially accumulated prefix length.
        - positive_paid (float): accumulated positive reservation charges.
        - ground (float): fixed departure cost.
        - negative (float): original all-negative-price credit.
        - fold_m (float | None): retained for the common bound interface.
        - paid (Iterable[RowKey]): live previously paid reservations.
        - terminal (bool): whether this prefix already uses its terminal endpoint role.

        Return
        --------
        - output (float): conservative full price-only completion floor, or negative infinity.
        """
        del goal, fold_m, terminal
        paid = tuple(paid)
        best = -math.inf
        for table in self.tables:
            penalty = table._resource_penalty
            assert penalty is not None
            # The error includes sequential-length rounding throughout the entire walk.
            budget = math.nextafter(self.graph.max_length + 1e-7 - length, math.inf)
            budget = math.nextafter(budget + table.distance_error_m, math.inf)
            if budget < 0.0:
                continue
            discount = sum(table.row_units.get(row, 0) for row in paid
                           if table.row_samples.get(row, -math.inf) > stamp)
            units = table.units(cell, stamp) - discount
            sampled = math.nextafter(units * table.quantum, -math.inf)
            correction = math.nextafter(penalty * budget, math.inf)
            extra = max(0.0, math.nextafter(sampled - correction, -math.inf))
            if not math.isfinite(extra):
                continue
            positive_error, negative_error, union_error = table._price_errors
            value = math.nextafter(positive_paid - positive_error, -math.inf)
            value = math.nextafter(value + negative, -math.inf)
            value = math.nextafter(value - negative_error, -math.inf)
            value = math.nextafter(value + extra, -math.inf)
            value = math.nextafter(value - union_error, -math.inf)
            value = math.nextafter(ground + value, -math.inf)
            if math.isfinite(value):
                best = max(best, value)
        return best
