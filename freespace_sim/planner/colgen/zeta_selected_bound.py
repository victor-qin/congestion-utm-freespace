"""Conservative continuation prices retaining the union of a few expensive rows."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from .network import RowKey
from .zeta_completion_geometry import distance_error as _distance_error, geometry as _geometry
from .zeta_bound import (
    _bin_index,
    _completion_delta,
    _customer_attachments,
    _distance_boundary,
    _fraction_down,
    _fraction_up,
    _price_units,
    _time_enclosures,
)

if TYPE_CHECKING:
    from .objective import CostModel
    from .zeta import Cell, ZetaGraph


@njit(cache=True, nogil=True)
def _selected_chunk(
    table: np.ndarray, edge_offsets: np.ndarray, targets: np.ndarray,
    part_offsets: np.ndarray, first_targets: np.ndarray, last_targets: np.ndarray,
    latest_starts: np.ndarray, earliest_ends: np.ndarray, covered: np.ndarray,
    period_starts: np.ndarray, period_ends: np.ndarray, buffer: float,
    mask_costs: np.ndarray, distances: np.ndarray, boundary: np.ndarray,
    goals: np.ndarray, ceiling: int, first: int, stop: int,
) -> None:
    """Fill descending time bins while paying each selected row at most once.

    Parameters
    ------------
    - table (np.ndarray): mutable integer costs indexed by time, node and paid mask.
    - edge_offsets (np.ndarray): outgoing-edge ranges for each node.
    - targets (np.ndarray): target node of each relaxed connection.
    - part_offsets (np.ndarray): subpart ranges for each connection.
    - first_targets (np.ndarray): earliest possible target bin of each connection.
    - last_targets (np.ndarray): latest possible target bin of each connection.
    - latest_starts (np.ndarray): upper bounds on actual subpart start timestamps.
    - earliest_ends (np.ndarray): lower bounds on actual subpart end timestamps.
    - covered (np.ndarray): selected-row spatial membership for each subpart.
    - period_starts (np.ndarray): exact multiplied starts of selected row periods.
    - period_ends (np.ndarray): exact multiplied ends of selected row periods.
    - buffer (float): unchanged nonnegative simulator reservation buffer.
    - mask_costs (np.ndarray): exact integer sum of prices selected by each mask.
    - distances (np.ndarray): downward-quantized weighted connection lengths.
    - boundary (np.ndarray): static relaxed distance cost after the table horizon.
    - goals (np.ndarray): nodes at which stopping without additional charges is allowed.
    - ceiling (int): finite cap; lowering an entry to this cap remains admissible.
    - first (int): latest bin to fill, inclusive.
    - stop (int): earliest bin to fill, exclusive.

    Return
    --------
    - output (None): updates only the requested block of already ordered states.
    """
    all_bits = table.shape[2] - 1
    for b in range(first, stop, -1):
        for node in range(table.shape[1]):
            if goals[node] or edge_offsets[node] == edge_offsets[node + 1]:
                table[b, node, :] = 0
                continue
            table[b, node, :] = ceiling
            for edge in range(edge_offsets[node], edge_offsets[node + 1]):
                guaranteed = 0
                for part in range(part_offsets[edge], part_offsets[edge + 1]):
                    for row in range(len(period_starts)):
                        if (covered[part, row]
                                and latest_starts[b, part] - buffer < period_ends[row]
                                and earliest_ends[b, part] + buffer > period_starts[row]):
                            guaranteed |= 1 << row
                for paid in range(table.shape[2]):
                    next_paid = paid | guaranteed
                    charge = distances[edge] + mask_costs[guaranteed & (all_bits ^ paid)]
                    best = table[b, node, paid]
                    for end_bin in range(first_targets[b, edge], last_targets[b, edge] + 1):
                        remaining = (table[end_bin, targets[edge], next_paid]
                                     if end_bin < table.shape[0] else boundary[targets[edge]])
                        best = min(best, charge + remaining)
                    table[b, node, paid] = best


@dataclass(slots=True)
class SelectedPriceBound:
    """A relaxed completion table with explicit paid memory for up to six cell rows.

    Attributes
    ------------
    - row_bits (dict[RowKey, int]): selected row to its single-bit paid-memory value.
    - row_units (dict[RowKey, int]): downward-quantized selected prices.
    - table (np.ndarray): immutable costs indexed by time, waypoint and paid mask.
    - quantum (float): power-of-two value of one exact integer cost unit.
    - build_s (float): elapsed geometry, price and table preparation time.
    - table_states (int): number of entries including the paid-memory dimension.
    """

    cell_ids: dict[Cell, int]
    first_bin: int
    delta: float
    table: np.ndarray
    row_bits: dict[RowKey, int]
    row_units: dict[RowKey, int]
    quantum: float
    build_s: float
    table_states: int
    edge_count: int
    boundary: np.ndarray
    distance_weight: float
    distance_error_m: float
    attachment_m: float
    _reference: float
    _coefficient_lo: float
    _coefficient_hi: float
    _price_errors: tuple[float, float, float]
    _air_error: float
    _air_ceiling: float
    _twice_abs_prices: float

    @classmethod
    def build(
        cls, graph: ZetaGraph, prices: Mapping[RowKey, float], model: CostModel,
        deadline: float | None = None,
    ) -> SelectedPriceBound | None:
        """Build a complete optional bound, declining unsupported or oversized tables.

        Parameters
        ------------
        - graph (ZetaGraph): unchanged waypoint, endpoint-role and timing authority.
        - prices (Mapping[RowKey, float]): finite frozen row prices of this pricing call.
        - model (CostModel): canonical nonnegative ground and air objective weights.
        - deadline (float | None): absolute monotonic preparation cutoff.

        Return
        --------
        - output (SelectedPriceBound | None): reusable lower bound, or None for an empty
          selection, unsupported arithmetic, nonadvancing relaxed clocks or resource limit.
          Immutable geometry is reused; selected prices and masks belong to this LP call.
        """
        started = time.perf_counter()
        cfg = graph.cfg
        if (deadline is not None and (not math.isfinite(deadline) or started >= deadline)
                or not math.isfinite(cfg.dt_s) or cfg.dt_s <= 0.0
                or not math.isfinite(cfg.time_buffer_s) or cfg.time_buffer_s < 0.0
                or not math.isfinite(cfg.nominal_speed_mps) or cfg.nominal_speed_mps <= 0.0
                or any(not math.isfinite(weight) or weight < 0.0
                       for weight in (model.air_weight, model.ground_weight))
                or any(not math.isfinite(price) for price in prices.values())):
            return None
        delta = min(1.0, cfg.dt_s / 4.0)
        if delta <= 0.0:
            return None
        earliest = (graph.fg.base_step + graph.fg.takeoff_steps[0]
                    + min(steps for _, _, steps in graph.origins)) * cfg.dt_s
        if not math.isfinite(earliest) or abs(earliest / delta) >= 2**48:
            return None
        candidates = [RowKey(row) for row, value in prices.items()
                      if value > 0.0 and RowKey(row).kind == "cell"
                      and (RowKey(row).step + 1) * cfg.dt_s > earliest - cfg.time_buffer_s]
        if not candidates:
            return None
        static = _geometry(graph, prices, deadline)
        if static is None:
            return None
        cells, outgoing = static
        edges = [edge for group in outgoing for edge in group]
        if not edges or deadline is not None and time.perf_counter() >= deadline:
            return None
        delta = _completion_delta(graph, outgoing)
        spaces = {space for edge in edges for covered in edge.covered for space in covered}
        selected = sorted((row for row in candidates if row[1:4] in spaces),
                          key=lambda row: (-prices[row], tuple(row)))[:6]
        first_bin = _bin_index(earliest, delta)
        bins = 0
        # Reducing selected memory changes only this relaxation, never candidate routes.
        while selected:
            horizon = max((row.step + 1) * cfg.dt_s for row in selected) + cfg.time_buffer_s
            if not math.isfinite(horizon) or abs(horizon / delta) >= 2**48:
                return None
            bins = _bin_index(math.nextafter(horizon, math.inf), delta) - first_bin + 2
            if 0 < bins * len(cells) * (1 << len(selected)) <= 2_000_000:
                break
            selected.pop()
        if not selected:
            return None
        parts_count = sum(len(edge.durations) for edge in edges)
        if bins * parts_count > 24_000_000:
            return None
        geometry_error = _distance_error(graph, cells, outgoing)
        attachments = _customer_attachments(graph)
        if geometry_error is None or attachments is None:
            return None
        distance_error, edge_limit, length_ceiling = geometry_error
        coefficient = Fraction(model.air_weight) / Fraction(cfg.nominal_speed_mps)
        try:
            distance_weight = float(coefficient) if (
                graph.fg.origin_terminal is None and graph.fg.dest_terminal is None
                and graph.reference > 1e-9
            ) else 0.0
        except OverflowError:
            return None
        exact_costs = ([coefficient * sum(map(Fraction, edge.lengths)) for edge in edges]
                       if distance_weight else [Fraction(0)] * len(edges))
        try:
            extra_budget = _fraction_up(max(exact_costs) * (len(cells) + 1))
        except OverflowError:
            return None
        quantized = _price_units({row: prices[row] for row in selected}, extra_budget)
        if quantized is None:
            return None
        row_units, quantum = quantized
        selected = [row for row in selected if row in row_units]
        distance_costs = [int(cost // Fraction(quantum)) for cost in exact_costs]
        ceiling = sum(row_units.values()) + (len(cells) + 1) * max(distance_costs)
        if ceiling >= 2**50 or not math.isfinite(ceiling * quantum):
            return None
        positive_sum = sum((Fraction(float(value)) for value in prices.values() if value > 0.0),
                           Fraction(0))
        negative_sum = sum((-Fraction(float(value)) for value in prices.values() if value < 0.0),
                           Fraction(0))
        if (edge_limit + 2) * 2.0**-52 >= 0.5:
            return None
        try:
            twice_positive = math.nextafter(2 * _fraction_up(positive_sum), math.inf)
            twice_negative = math.nextafter(2 * _fraction_up(negative_sum), math.inf)
            twice_absolute = math.nextafter(2 * _fraction_up(positive_sum + negative_sum), math.inf)
            positive_error = math.nextafter((2 * edge_limit + 2) * math.ulp(twice_positive), math.inf)
            price_errors = positive_error, math.ulp(twice_negative), math.ulp(twice_absolute)
            detour_ceiling = math.nextafter(length_ceiling + abs(graph.reference), math.inf)
            seconds_ceiling = math.nextafter(detour_ceiling / cfg.nominal_speed_mps, math.inf)
            air_ceiling = math.nextafter(model.air_weight * seconds_ceiling, math.inf)
            air_error = _fraction_up(coefficient * Fraction(math.ulp(detour_ceiling))
                                     + Fraction(model.air_weight) * Fraction(math.ulp(seconds_ceiling))
                                     + Fraction(math.ulp(air_ceiling)))
        except (OverflowError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (*price_errors, air_error, air_ceiling,
                                                      twice_absolute)):
            return None
        nominal = [math.fsum(edge.durations) for edge in edges]
        part_offsets, durations, fractions, covered_rows = [0], [], [], []
        for edge in edges:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            for index, covered in enumerate(edge.covered):
                durations.append(edge.durations[index])
                fractions.append(math.fsum(edge.durations[:index + 1]) / math.fsum(edge.durations))
                covered_rows.append([row[1:4] in covered for row in selected])
            part_offsets.append(len(durations))
        part_offsets_array = np.asarray(part_offsets, dtype=np.int64)
        fraction = graph.options.speed_fraction if graph.options.arrival_snap_s > 0.0 else 0.0
        (first_targets, last_targets, _earliest_starts, _latest_ends,
         latest_starts, earliest_ends, valid) = _time_enclosures(
            first_bin, bins, delta, graph.options.arrival_snap_s, fraction,
            part_offsets_array, np.asarray(durations), np.asarray(nominal), np.asarray(fractions),
        )
        if not valid or deadline is not None and time.perf_counter() >= deadline:
            return None
        masks = 1 << len(selected)
        if int(np.sum(last_targets - first_targets + 1)) * masks > 200_000_000:
            return None
        cell_ids = {cell: index for index, cell in enumerate(cells)}
        goals = np.zeros(len(cells), dtype=np.bool_)
        for _, goal in graph.destinations:
            goals[cell_ids[goal]] = True
        boundary = _distance_boundary(outgoing, distance_costs, np.flatnonzero(goals))
        row_bits = {row: 1 << index for index, row in enumerate(selected)}
        mask_costs = np.asarray([
            sum(row_units[row] for row, bit in row_bits.items() if mask & bit)
            for mask in range(masks)
        ], dtype=np.int64)
        table = np.zeros((bins, len(cells), masks), dtype=np.int64)
        arrays = (
            np.asarray([0, *np.cumsum([len(group) for group in outgoing])], dtype=np.int64),
            np.asarray([edge.target for edge in edges], dtype=np.int64), part_offsets_array,
            first_targets, last_targets, latest_starts, earliest_ends,
            np.asarray(covered_rows, dtype=np.bool_),
            np.asarray([row.step * cfg.dt_s for row in selected]),
            np.asarray([(row.step + 1) * cfg.dt_s for row in selected]), cfg.time_buffer_s,
            mask_costs, np.asarray(distance_costs, dtype=np.int64), boundary, goals,
        )
        for first in range(bins - 1, -1, -8):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            _selected_chunk(table, *arrays, ceiling, first, max(-1, first - 8))
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        table.flags.writeable = False
        boundary.flags.writeable = False
        return cls(cell_ids, first_bin, delta, table, row_bits, row_units, quantum,
                   time.perf_counter() - started, int(table.size), len(edges), boundary,
                   distance_weight, distance_error, attachments, graph.reference,
                   _fraction_down(coefficient), _fraction_up(coefficient), price_errors,
                   air_error, air_ceiling, twice_absolute)

    def units(
        self, cell: Cell, stamp: float, paid: Iterable[RowKey] | int = (),
        terminal: bool = False,
    ) -> int:
        """Look up quantized continuation cost for exact prefix paid-row membership.

        Parameters
        ------------
        - cell (Cell): current exact waypoint.
        - stamp (float): current physical arrival time.
        - paid (Iterable[RowKey] | int): unique live paid rows, or their selected-row bitmask.
        - terminal (bool): whether this prefix already ends the route.

        Return
        --------
        - output (int): downward distance-plus-selected-price continuation units, with zero
          for terminal prefixes or states outside the table's lower domain.
        """
        identifier = self.cell_ids.get(cell)
        if terminal or identifier is None or not math.isfinite(stamp):
            return 0
        index = _bin_index(stamp, self.delta) - self.first_bin
        if index < 0:
            return 0
        if index >= self.table.shape[0]:
            return int(self.boundary[identifier])
        mask = paid if isinstance(paid, int) else 0
        if not isinstance(paid, int):
            for row in paid:
                mask |= self.row_bits.get(row, 0)
        mask &= self.table.shape[2] - 1
        return int(self.table[index, identifier, mask])

    def floor(
        self, cell: Cell, goal: Cell, stamp: float, length: float,
        positive_paid: float, ground: float, negative: float, fold_m: float | None,
        paid: Iterable[RowKey] | int, terminal: bool = False,
    ) -> float:
        """Return a guarded floor on the canonical cost of every exact completion.

        Parameters
        ------------
        - cell (Cell): current exact waypoint.
        - goal (Cell): exact chosen destination; the table allows every destination.
        - stamp (float): exact prefix arrival timestamp.
        - length (float): sequentially accumulated graph-part length.
        - positive_paid (float): accumulated positive prefix charges, including retired rows.
        - ground (float): canonical ground cost of this departure.
        - negative (float): unchanged global sum of all negative row prices.
        - fold_m (float | None): exact-prefix attachment hint; the bound uses global minima.
        - paid (Iterable[RowKey] | int): live paid rows, or their selected-row bitmask.
        - terminal (bool): whether the prefix already ends the route.

        Return
        --------
        - output (float): admissible full minimization floor, or minus infinity when finite
          arithmetic cannot be guaranteed. Untracked future positive rows cost zero;
          terminal endpoints retain the price-only fallback of the sampled-price bound.
        """
        def down(value: float) -> float:
            """Round one arithmetic operation downward."""
            return math.nextafter(value, -math.inf)

        sampled = self.units(cell, stamp, paid, terminal) * self.quantum
        if not all(math.isfinite(value) for value in (sampled, stamp, length, positive_paid,
                                                      ground, negative)):
            return -math.inf
        positive_error, negative_error, union_error = self._price_errors
        charged = down(down(down(positive_paid - positive_error) + negative) - negative_error)
        if not self.distance_weight:
            adjustment = down(down(charged + max(0.0, sampled)) - union_error)
            result = down(ground + adjustment)
            return result if math.isfinite(result) else -math.inf
        distance = down(down(down(length + self.attachment_m) - self._reference)
                        - self.distance_error_m)
        coefficient = self._coefficient_lo if distance >= 0.0 else self._coefficient_hi
        joint = max(0.0, down(down(down(coefficient * distance) + sampled) - self._air_error))
        cost_ceiling = math.nextafter(ground + self._air_ceiling, math.inf)
        final_scale = math.nextafter(cost_ceiling + self._twice_abs_prices, math.inf)
        if not all(math.isfinite(value) for value in (charged, distance, joint, cost_ceiling,
                                                      final_scale)):
            return -math.inf
        result = down(down(ground + charged) + joint)
        result = down(result - math.ulp(cost_ceiling))
        result = down(result - union_error)
        result = down(result - math.ulp(final_scale))
        return result if math.isfinite(result) else -math.inf
