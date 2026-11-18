"""Unavoidable priced crossings in a relaxation of every permitted timed connection.

Each relaxed connection receives only row charges guaranteed throughout its source
time interval.  The continuation table minimizes the largest connection charge,
not their sum.  For any positive threshold below that value, deleting connections
with at least that charge disconnects the state from every relaxed destination.
This certifies a crossing without assuming the current best route is mandatory.

The proof also covers long connections and revisits: every actual completion maps
to a walk in the role-relaxed, time-advancing graph, and one of its connections has
at least the table's charge.  Rows within that connection are unique; charges on
different connections are never added.  Subtracting all live paid row units then
bounds newly charged positive rows.  Expired rows cannot occur on an actual future
connection and therefore need no discount.  Endpoint reservations are omitted and
every destination is a free stopping state, both conservative relaxations.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from .network import RowKey
from .zeta_completion_geometry import distance_error as _distance_error, geometry as _geometry
from .zeta_bound import (
    _bin_index, _completion_delta, _fraction_up, _price_units,
    _time_enclosures,
)

if TYPE_CHECKING:
    from .objective import CostModel
    from .zeta import Cell, ZetaGraph


@njit(cache=True, nogil=True)
def _crossing_chunk(
    table: np.ndarray, edge_offsets: np.ndarray, targets: np.ndarray,
    first_targets: np.ndarray, last_targets: np.ndarray,
    latest_starts: np.ndarray, earliest_ends: np.ndarray,
    row_offsets: np.ndarray, member_offsets: np.ndarray, members: np.ndarray,
    row_starts: np.ndarray, row_ends: np.ndarray, row_weights: np.ndarray,
    goals: np.ndarray, buffer: float, total_units: int, first: int, stop: int,
) -> None:
    """Solve a descending block of the minimum-largest-crossing recurrence.

    Parameters
    ------------
    - table (np.ndarray): integer bounds indexed by source time interval and waypoint.
    - edge_offsets (np.ndarray): outgoing connection ranges for each waypoint.
    - targets (np.ndarray): destination waypoint of each connection.
    - first_targets (np.ndarray): earliest possible target interval for each source/edge.
    - last_targets (np.ndarray): latest possible target interval for each source/edge.
    - latest_starts (np.ndarray): upper subpart-start bounds for each source interval.
    - earliest_ends (np.ndarray): lower subpart-end bounds for each source interval.
    - row_offsets (np.ndarray): unique positive cell-row ranges for each connection.
    - member_offsets (np.ndarray): subpart-membership ranges for each connection-row pair.
    - members (np.ndarray): flattened global subpart indices whose footprints claim a row.
    - row_starts (np.ndarray): exact reservation-period starts for connection-row pairs.
    - row_ends (np.ndarray): exact reservation-period ends for connection-row pairs.
    - row_weights (np.ndarray): downward integer prices for connection-row pairs.
    - goals (np.ndarray): destinations at which the relaxed walk may stop freely.
    - buffer (float): unchanged nonnegative simulator reservation time buffer.
    - total_units (int): upper bound on all distinct positive row units.
    - first (int): latest source interval in this block, inclusive.
    - stop (int): earliest source interval in this block, exclusive.

    Return
    --------
    - output (None): fills the block using already completed later intervals.
    """
    for b in range(first, stop, -1):
        for source in range(table.shape[1]):
            if goals[source] or edge_offsets[source] == edge_offsets[source + 1]:
                table[b, source] = 0
                continue
            best = total_units
            for edge in range(edge_offsets[source], edge_offsets[source + 1]):
                charge = 0
                for row in range(row_offsets[edge], row_offsets[edge + 1]):
                    for member in range(member_offsets[row], member_offsets[row + 1]):
                        part = members[member]
                        if (latest_starts[b, part] - buffer < row_ends[row]
                                and earliest_ends[b, part] + buffer > row_starts[row]):
                            charge += row_weights[row]
                            break
                if charge >= best:
                    continue
                continuation = total_units
                for target_bin in range(first_targets[b, edge], last_targets[b, edge] + 1):
                    if target_bin >= table.shape[0]:
                        continuation = 0
                        break
                    continuation = min(continuation, table[target_bin, targets[edge]])
                    if continuation <= charge:
                        break
                best = min(best, max(charge, continuation))
                if best == 0:
                    break
            table[b, source] = best


@dataclass(slots=True)
class CrossingBound:
    """A conservative price floor certified by unavoidable connection crossings.

    Attributes
    ------------
    - cell_ids (dict[Cell, int]): stable waypoint-to-column lookup.
    - first_bin (int): absolute index of the first prepared time interval.
    - delta (float): width of the time intervals used only by the relaxation.
    - table (np.ndarray): minimum largest guaranteed connection charge in integer units.
    - row_units (dict[RowKey, int]): downward positive cell prices for paid-row discounts.
    - quantum (float): exact power-of-two value of one integer unit.
    - build_s (float): elapsed preparation time, including shared geometry preparation.
    - table_states (int): number of prepared waypoint/time states.
    """

    cell_ids: dict[Cell, int]
    first_bin: int
    delta: float
    table: np.ndarray
    row_units: dict[RowKey, int]
    quantum: float
    build_s: float
    table_states: int
    _price_errors: tuple[float, float, float]

    @classmethod
    def build(
        cls, graph: ZetaGraph, prices: Mapping[RowKey, float], model: CostModel,
        deadline: float | None = None,
    ) -> CrossingBound | None:
        """Prepare a complete crossing table or decline unsupported or excessive work.

        Parameters
        ------------
        - graph (ZetaGraph): finite waypoints, unchanged geometry, endpoint roles and timing.
        - prices (Mapping[RowKey, float]): finite frozen prices; only positive cell rows enter
          the crossing table, while all signed prices enter the rounding guards.
        - model (CostModel): canonical objective; nonnegative air cost may safely be omitted.
        - deadline (float | None): absolute monotonic cutoff; partial tables are discarded.

        Return
        --------
        - output (CrossingBound | None): complete lookup, or None for expired preparation,
          empty positive cell prices, unsupported clocks or bounded preparation limits.
        """
        started = time.perf_counter()
        cfg = graph.cfg
        if (not all(math.isfinite(value) and value >= 0.0
                    for value in (model.ground_weight, model.air_weight, cfg.time_buffer_s))
                or not math.isfinite(cfg.dt_s) or cfg.dt_s <= 0.0
                or any(not math.isfinite(value) for value in prices.values())
                or (deadline is not None and started >= deadline)):
            return None
        quantized = _price_units(prices)
        if quantized is None:
            return None
        all_units, quantum = quantized
        row_units = {row: units for row, units in all_units.items() if row.kind == "cell"}
        if not row_units:
            return None
        periods = {row.step: (row.step * cfg.dt_s, (row.step + 1) * cfg.dt_s)
                   for row in row_units}
        if any(not math.isfinite(start) or not math.isfinite(end) or start >= end
               for start, end in periods.values()):
            return None
        delta = min(1.0, cfg.dt_s / 4.0)
        earliest = (graph.fg.base_step + graph.fg.takeoff_steps[0]
                    + min(steps for _, _, steps in graph.origins)) * cfg.dt_s
        horizon = math.nextafter(max(end for _, end in periods.values()) + cfg.time_buffer_s,
                                 math.inf)
        if (delta <= 0.0 or not math.isfinite(earliest) or not math.isfinite(horizon)
                or max(abs(earliest / delta), abs(horizon / delta)) >= 2**48):
            return None
        first_bin = _bin_index(earliest, delta)
        bins = _bin_index(horizon, delta) - first_bin + 2
        if bins <= 0:
            return None
        static = _geometry(graph, prices, deadline)
        if static is None:
            return None
        cells, outgoing = static
        edges = [edge for group in outgoing for edge in group]
        if not edges:
            return None
        delta = _completion_delta(graph, outgoing)
        first_bin = _bin_index(earliest, delta)
        bins = _bin_index(horizon, delta) - first_bin + 2
        parts_count = sum(len(edge.durations) for edge in edges)
        if (not edges or bins * len(cells) > 2_000_000
                or bins * parts_count > 24_000_000
                or (deadline is not None and time.perf_counter() >= deadline)):
            return None
        geometry_error = _distance_error(graph, cells, outgoing)
        if geometry_error is None:
            return None
        _, edge_limit, _ = geometry_error
        if (edge_limit + 2) * 2.0**-52 >= 0.5:
            return None
        positive = sum((Fraction(float(value)) for value in prices.values() if value > 0.0),
                       Fraction(0))
        negative = sum((-Fraction(float(value)) for value in prices.values() if value < 0.0),
                       Fraction(0))
        try:
            twice_positive = math.nextafter(2 * _fraction_up(positive), math.inf)
            twice_negative = math.nextafter(2 * _fraction_up(negative), math.inf)
            twice_absolute = math.nextafter(2 * _fraction_up(positive + negative), math.inf)
            price_errors = (
                math.nextafter((2 * edge_limit + 2) * math.ulp(twice_positive), math.inf),
                math.ulp(twice_negative), math.ulp(twice_absolute),
            )
        except (OverflowError, ValueError):
            return None
        if not all(math.isfinite(value) for value in price_errors):
            return None
        by_space: dict[tuple[int, int, int], list[RowKey]] = defaultdict(list)
        for row in row_units:
            by_space[row[1:4]].append(row)
        part_offsets, durations, fractions, nominal = [0], [], [], []
        row_offsets, member_offsets, members = [0], [0], []
        row_starts, row_ends, row_weights = [], [], []
        for edge in edges:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            total = math.fsum(edge.durations)
            nominal.append(total)
            covered_rows: dict[RowKey, list[int]] = defaultdict(list)
            for index, covered in enumerate(edge.covered):
                durations.append(edge.durations[index])
                fractions.append(math.fsum(edge.durations[:index + 1]) / total)
                for space in covered:
                    for row in by_space.get(space, ()):
                        covered_rows[row].append(part_offsets[-1] + index)
            part_offsets.append(len(durations))
            for row, parts in covered_rows.items():
                start, end = periods[row.step]
                row_starts.append(start)
                row_ends.append(end)
                row_weights.append(row_units[row])
                members.extend(parts)
                member_offsets.append(len(members))
            row_offsets.append(len(row_weights))
            if bins * (len(members) + parts_count) > 64_000_000:
                return None
        if not members or (deadline is not None and time.perf_counter() >= deadline):
            return None
        fraction = graph.options.speed_fraction if graph.options.arrival_snap_s > 0.0 else 0.0
        first_targets, last_targets, _, _, latest_starts, earliest_ends, valid = _time_enclosures(
            first_bin, bins, delta, graph.options.arrival_snap_s, fraction,
            np.asarray(part_offsets, dtype=np.int64), np.asarray(durations),
            np.asarray(nominal), np.asarray(fractions),
        )
        if (not valid or int(np.sum(last_targets - first_targets + 1)) > 200_000_000
                or (deadline is not None and time.perf_counter() >= deadline)):
            return None
        cell_ids = {cell: index for index, cell in enumerate(cells)}
        goals = np.zeros(len(cells), dtype=np.bool_)
        for _, goal in graph.destinations:
            goals[cell_ids[goal]] = True
        table = np.zeros((bins, len(cells)), dtype=np.int64)
        arguments = (
            np.asarray([0, *np.cumsum([len(group) for group in outgoing])], dtype=np.int64),
            np.asarray([edge.target for edge in edges], dtype=np.int64),
            first_targets, last_targets, latest_starts, earliest_ends,
            np.asarray(row_offsets, dtype=np.int64), np.asarray(member_offsets, dtype=np.int64),
            np.asarray(members, dtype=np.int64), np.asarray(row_starts), np.asarray(row_ends),
            np.asarray(row_weights, dtype=np.int64), goals, cfg.time_buffer_s,
            sum(row_units.values()),
        )
        for first in range(bins - 1, -1, -32):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            _crossing_chunk(table, *arguments, first, max(-1, first - 32))
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        return cls(cell_ids, first_bin, delta, table, row_units, quantum,
                   time.perf_counter() - started, int(table.size), price_errors)

    def floor(
        self, cell: Cell, goal: Cell, stamp: float, length: float,
        positive_paid: float, ground: float, negative: float, fold_m: float | None,
        paid: Iterable[RowKey], terminal: bool = False,
    ) -> float:
        """Lower-bound the full completion objective after discounting already-paid rows.

        Parameters
        ------------
        - cell (Cell): current exact waypoint.
        - goal (Cell): chosen destination; all graph destinations remain free in the relaxation.
        - stamp (float): exact physical arrival time at the current waypoint.
        - length (float): prefix graph-part length; this bound omits nonnegative air cost.
        - positive_paid (float): accumulated positive price of every prefix reservation.
        - ground (float): canonical ground cost of the chosen departure.
        - negative (float): sum of all negative row prices, whether reachable or not.
        - fold_m (float | None): endpoint attachment length; unused by this price-only bound.
        - paid (Iterable[RowKey]): unique live paid rows; expired rows cannot recur.
        - terminal (bool): whether the route may already stop with its terminal endpoint role.

        Return
        --------
        - output (float): conservative minimization floor including signed-price rounding,
          or minus infinity for unsupported nonfinite arithmetic.
        """
        if not all(math.isfinite(value) for value in (stamp, length, positive_paid, ground, negative)):
            return -math.inf
        index = _bin_index(stamp, self.delta) - self.first_bin
        identifier = self.cell_ids.get(cell)
        units = 0
        if not terminal and identifier is not None and 0 <= index < self.table.shape[0]:
            units = max(0, int(self.table[index, identifier])
                        - sum(self.row_units.get(row, 0) for row in paid))
        crossing = units * self.quantum
        if not math.isfinite(crossing):
            return -math.inf

        def down(value: float) -> float:
            """Round each floating arithmetic operation outward toward a lower floor."""
            return math.nextafter(value, -math.inf)

        positive_error, negative_error, union_error = self._price_errors
        prefix = down(down(down(positive_paid - positive_error) + negative) - negative_error)
        total = down(down(prefix + crossing) - union_error)
        result = down(ground + total)
        return result if math.isfinite(result) else -math.inf
