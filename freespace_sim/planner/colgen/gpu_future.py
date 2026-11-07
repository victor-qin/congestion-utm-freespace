"""Relaxed future visit-price and air-cost bounds for GPU pricing."""

from __future__ import annotations

import math
import time
from heapq import heappop, heappush

import numpy as np

from . import pricing
from .dp_prepare import (
    ARC_FIRST,
    ARC_FIRST_LAST,
    ARC_INTERNAL,
    ARC_LAST,
    PreparedTopology,
    PreparedVariants,
)
from .gpu_bounds import CompletionBounds
from .gpu_runtime import ComputeRuntime
from .pricing_platform import PricingProblem


def _float32_lower(values: np.ndarray) -> np.ndarray:
    """Convert nonnegative binary64 bounds to binary32 without rounding upward."""
    with np.errstate(over="ignore", under="ignore"):
        result = values.astype(np.float32)
        above = result.astype(np.float64) > values
        result[above] = np.nextafter(result[above], np.float32(0.0))
    return result


def _add_lower(first: np.ndarray, second: np.ndarray | float) -> np.ndarray:
    """Add nonnegative bounds downward, retaining infinity for unreachable states."""
    with np.errstate(over="ignore", under="ignore"):
        result = np.maximum(0.0, np.nextafter(first + second, -np.inf))
    result[np.isposinf(first) | np.isposinf(second)] = np.inf
    return result


class FutureBounds:
    """Lower-bound future air and positive visit costs on a relaxed time graph.

    The regular table allows cruise/finish roles, while requested roots use a
    separate first-hop relaxation into the next regular row. Destination stops
    add only mandatory endpoint rows disjoint from all preceding visit windows.
    The relaxation ignores hop, revisit and forbidden-row constraints. Endpoint
    disjointness uses reachability from every supplied root, so the bounds apply
    to continuations of those roots, not arbitrary unrooted suffixes. Combining
    these costs with canonical geometry and origin claims remains the caller's responsibility.
    Negative-price credits are also handled by callers.
    """

    def __init__(
        self,
        problem: PricingProblem,
        topology: PreparedTopology,
        variants: PreparedVariants,
        geometry: CompletionBounds,
        view: pricing.DualView,
        runtime: ComputeRuntime,
        deadline: float | None,
        *,
        cpu_recompute: bool = True,
    ) -> None:
        """Build the backward table on the GPU and optionally refine its arithmetic.

        Parameters
        ------------
        - problem (PricingProblem): original flight and objective snapshot.
        - topology (PreparedTopology): role-specific outgoing arcs and destination cells.
        - variants (PreparedVariants): establishes the earliest root start time.
        - geometry (CompletionBounds): conservative nonnegative air cost per hop.
        - view (pricing.DualView): original binary64 prices and visit-window offsets.
        - runtime (ComputeRuntime): selected GPU with the ``future_bounds`` kernel.
        - deadline (float | None): shared monotonic deadline, checked each layer.
        - cpu_recompute (bool): independently recompute relaxed layers in binary64;
          this does not invoke the constrained CPU pricing search.

        Return
        --------
        - output (None): initializes inclusive-time lower bounds; timeout raises
          ``PricingTimeout`` rather than exposing an unfinished table.
        """
        started = time.monotonic()
        self.problem, self.topology = problem, topology
        self.cpu_recompute, self.deadline = cpu_recompute, deadline
        self.last_step = int(topology.max_step)
        self.first_step = (
            min(int(np.min(variants.start_step)), self.last_step)
            if len(variants.start_step)
            else self.last_step
        )
        self.stats: dict[str, int | float | bool] = {
            "future_dispatches": 0,
            "future_layers": 0,
            "future_root_layers": 0,
            "future_cells": topology.n_cells,
            "future_cpu_recompute": cpu_recompute,
            "future_prepare_s": 0.0,
            "future_s": 0.0,
        }
        self._check_deadline()
        air = float(geometry.air_per_hop)
        if not math.isfinite(air) or air < 0.0:
            raise ValueError("future air cost must be finite and nonnegative")
        graph = self._graph(ARC_INTERNAL | ARC_LAST)
        first_graph = self._graph(ARC_FIRST | ARC_FIRST_LAST)
        roots_by_step: dict[int, set[int]] = {}
        self._root_costs: dict[tuple[int, int], float] = {}
        for cell, step in zip(variants.cell, variants.start_step, strict=True):
            cell, step = int(cell), int(step)
            if not 0 <= cell < topology.n_cells:
                raise ValueError("future-bound root cell is outside the prepared topology")
            roots_by_step.setdefault(step, set()).add(cell)
            self._root_costs[cell, step] = math.inf
        self._earliest = self._earliest_visits(variants)
        visits = self._visits(view)
        tails = self._tails(view)
        self._check_deadline()
        self.stats["future_prepare_s"] = time.monotonic() - started
        n_cells = topology.n_cells
        dtype = np.float64 if cpu_recompute else np.float32
        self._costs = np.full(visits.shape, np.inf, dtype=dtype)
        destination = np.asarray(topology.dest_mask, dtype=bool)
        self._costs[-1, destination] = tails[-1] if cpu_recompute else _float32_lower(tails[-1:])[0]
        for cell in roots_by_step.get(self.last_step, ()):
            self._root_costs[cell, self.last_step] = float(self._costs[-1, cell])
        next32 = _float32_lower(self._costs[-1].astype(np.float64, copy=False))
        control = np.array([n_cells], np.int32)
        static_prefix = f"future:{id(self)}"
        static = {
            0: f"{static_prefix}:control",
            1: f"{static_prefix}:graph",
        }
        dispatches = runtime.dispatches
        for offset in range(len(self._costs) - 2, -1, -1):
            self._check_deadline()
            visit = visits[offset + 1]
            visit32 = _float32_lower(visit)
            stop = float(tails[offset])
            air32 = _float32_lower(np.array([air, stop], np.float64))
            (current32,) = runtime.run(
                "future_bounds",
                n_cells,
                (
                    control,
                    graph,
                    next32,
                    visit32,
                    air32,
                    np.empty(n_cells, np.float32),
                ),
                (5,),
                static=static,
                write_only=(5,),
            )
            self._check_deadline()
            if np.any(np.isnan(current32)) or np.any(current32 < 0.0):
                raise ValueError("GPU future bounds must be nonnegative and not NaN")
            if cpu_recompute:
                self._costs[offset] = self._relax(
                    graph, destination, visit, self._costs[offset + 1], air, stop
                )
            else:
                self._costs[offset] = current32
            step = self.first_step + offset
            root_cells = roots_by_step.get(step)
            if root_cells:
                cells = np.asarray(sorted(root_cells), np.int32)
                root_graph = first_graph[cells]
                (root32,) = runtime.run(
                    "future_bounds",
                    len(cells),
                    (
                        np.array([len(cells)], np.int32),
                        root_graph,
                        next32,
                        visit32,
                        air32,
                        np.empty(len(cells), np.float32),
                    ),
                    (5,),
                    static={},
                    write_only=(5,),
                )
                self._check_deadline()
                if np.any(np.isnan(root32)) or np.any(root32 < 0.0):
                    raise ValueError("GPU future root bounds must be nonnegative and not NaN")
                root_costs = (
                    self._relax(
                        root_graph, destination[cells], visit, self._costs[offset + 1], air, stop
                    )
                    if cpu_recompute
                    else root32
                )
                for cell, cost in zip(cells, root_costs, strict=True):
                    self._root_costs[int(cell), step] = float(cost)
                self.stats["future_root_layers"] += 1
            next32 = current32
            self.stats["future_layers"] += 1
        self._check_deadline()
        self._costs.flags.writeable = False
        self.stats["future_dispatches"] = runtime.dispatches - dispatches
        self.stats["future_table_bytes"] = self._costs.nbytes
        self.stats["future_roots"] = len(self._root_costs)
        self.stats["future_s"] = time.monotonic() - started

    def _check_deadline(self) -> None:
        """Never publish partial future bounds as a completed relaxation."""
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise pricing.PricingTimeout("GPU future-bound preparation exceeded the deadline")

    def _graph(self, allowed_roles: int) -> np.ndarray:
        """Pack selected outgoing roles into six targets plus a destination marker.

        Parameters
        ------------
        - self (FutureBounds): prepared topology for one flight.
        - allowed_roles (int): arc-role union for regular suffixes or first hops.

        Return
        --------
        - output (np.ndarray): contiguous 32-bit integers, with absent targets -1.
        """
        topology = self.topology
        graph = np.full((topology.n_cells, 7), -1, np.int32)
        graph[:, 6] = topology.dest_mask
        for cell in range(topology.n_cells):
            if cell % 1024 == 0:
                self._check_deadline()
            start, end = int(topology.arc_start[cell]), int(topology.arc_start[cell + 1])
            roles = topology.arc_roles[start:end]
            targets = topology.arc_target[start:end][(roles & allowed_roles) != 0]
            if len(targets) > 6:
                raise ValueError("future-bound graph requires at most six hex neighbors")
            graph[cell, : len(targets)] = targets
        return graph

    def _visits(self, view: pricing.DualView) -> np.ndarray:
        """Accumulate positive sparse rows into their inclusive visit-time windows.

        Parameters
        ------------
        - view (pricing.DualView): normalized prices; only level-zero cell rows matter.

        Return
        --------
        - output (np.ndarray): downward binary64 sums indexed by relative step and cell.
        """
        topology = self.topology
        visits = np.zeros((self.last_step - self.first_step + 1, topology.n_cells), np.float64)
        index = {
            (int(q), int(r)): cell
            for cell, (q, r) in enumerate(zip(topology.cell_q, topology.cell_r, strict=True))
        }
        lo, hi = view.offsets
        for number, (row, price) in enumerate(view._duals.items()):
            if number % 1024 == 0:
                self._check_deadline()
            if row.kind != "cell" or row.level != 0 or price <= 0.0:
                continue
            cell = index.get(row.cell_coord)
            if cell is None:
                continue
            if not math.isfinite(price):
                raise ValueError("future visit prices must be finite")
            first = max(self.first_step, int(row.step) - hi)
            last = min(self.last_step, int(row.step) - lo)
            if first <= last:
                target = visits[first - self.first_step : last - self.first_step + 1, cell]
                target[:] = _add_lower(target, price)
        return visits

    def _earliest_visits(self, variants: PreparedVariants) -> np.ndarray:
        """Lower-bound each cell's first possible visit over all supplied roots.

        Parameters
        ------------
        - variants (PreparedVariants): every allowed root's cell and absolute start time.

        Return
        --------
        - output (np.ndarray): earliest relaxed arrival per cell, or the int64 maximum
          for unreachable cells. All arc roles are allowed and hop/revisit restrictions
          are ignored, so no legal rooted visit can occur earlier than this bound.
        """
        topology = self.topology
        earliest = np.full(topology.n_cells, np.iinfo(np.int64).max, np.int64)
        queue: list[tuple[int, int]] = []
        for cell, step in zip(variants.cell, variants.start_step, strict=True):
            cell, step = int(cell), int(step)
            if step < earliest[cell]:
                earliest[cell] = step
                heappush(queue, (step, cell))
        while queue:
            self._check_deadline()
            step, cell = heappop(queue)
            if step != earliest[cell]:
                continue
            start, end = int(topology.arc_start[cell]), int(topology.arc_start[cell + 1])
            for target in topology.arc_target[start:end]:
                target = int(target)
                if step + 1 < earliest[target]:
                    earliest[target] = step + 1
                    heappush(queue, (step + 1, target))
        earliest.flags.writeable = False
        return earliest

    def _tails(self, view: pricing.DualView) -> np.ndarray:
        """Price mandatory arrival claims that cannot overlap any earlier lattice visit.

        Parameters
        ------------
        - view (pricing.DualView): original prices and inclusive visit-window offsets.

        Return
        --------
        - output (np.ndarray): downward binary64 positive-price sums per arrival time.
          Zero accumulated hops selects a subset of every completion's endpoint rows.
          A row is included only if no visit from any supplied root could claim it
          and still reach a destination by this arrival. Terminal rows and cell rows
          outside the level-zero topology are disjoint by resource identity.
        """
        topology = self.topology
        index = {
            (int(q), int(r)): cell
            for cell, (q, r) in enumerate(zip(topology.cell_q, topology.cell_r, strict=True))
        }
        lo, hi = view.offsets
        tails = np.zeros(self.last_step - self.first_step + 1, np.float64)
        for offset in range(len(tails)):
            self._check_deadline()
            step = self.first_step + offset
            claims = pricing._endpoint_claims(
                self.problem.graph, self.problem.cfg, origin=False, step=step, timing_steps=0
            )
            total = 0.0
            for row in claims:
                price = view.row_cost(row)
                if price <= 0.0:
                    continue
                if not math.isfinite(price):
                    raise ValueError("future arrival prices must be finite")
                if row.kind == "cell" and row.level == 0:
                    cell = index.get(row.cell_coord)
                    if cell is not None:
                        first_visit = max(int(self._earliest[cell]), int(row.step) - hi)
                        last_visit = min(
                            step - int(topology.rev_remaining[cell]), int(row.step) - lo
                        )
                        if first_visit <= last_visit:
                            continue
                total = max(0.0, math.nextafter(total + price, -math.inf))
            tails[offset] = total
        return tails

    @staticmethod
    def _relax(
        graph: np.ndarray,
        destination: np.ndarray,
        visit: np.ndarray,
        following: np.ndarray,
        air: float,
        stop: float,
    ) -> np.ndarray:
        """Verify one relaxed layer using only original-price binary64 lower bounds.

        Parameters
        ------------
        - graph (np.ndarray): six permitted targets and a destination marker per source.
        - destination (np.ndarray): cells permitted to pay the current stop cost.
        - visit (np.ndarray): next-time positive visit-price lower sums.
        - following (np.ndarray): next-time independently recomputed future costs.
        - air (float): geometric air-cost lower bound per hop.
        - stop (float): disjoint positive endpoint-price lower bound when stopping now.

        Return
        --------
        - output (np.ndarray): current-time lower bounds; unreachable cells retain infinity.
        """
        costs = _add_lower(_add_lower(visit, air), following)
        current = np.full(len(graph), np.inf, np.float64)
        for neighbor in range(6):
            targets = graph[:, neighbor]
            valid = targets >= 0
            current[valid] = np.minimum(current[valid], costs[targets[valid]])
        current[destination] = np.minimum(current[destination], stop)
        return current

    def row(self, step: int) -> np.ndarray:
        """Return the read-only regular-suffix cost row at an absolute lattice step.

        Parameters
        ------------
        - step (int): inclusive absolute time in ``[first_step, last_step]``.

        Return
        --------
        - output (np.ndarray): binary64 cell-indexed bounds; out-of-range times raise
          ``IndexError``. Unchecked binary32 mode converts values without refinement.
        """
        if not self.first_step <= step <= self.last_step:
            raise IndexError("future-bound step is outside the prepared time horizon")
        result = self._costs[step - self.first_step].astype(np.float64, copy=False)
        result.flags.writeable = False
        return result

    def lookup(self, cell: int, step: int) -> float:
        """Return a regular-suffix bound after at least one hop has been taken.

        Parameters
        ------------
        - cell (int): local topology cell index.
        - step (int): inclusive absolute time in ``[first_step, last_step]``.

        Return
        --------
        - output (float): future-cost lower bound, possibly positive infinity.
        """
        if not 0 <= cell < self.topology.n_cells:
            raise IndexError("future-bound cell is outside the prepared topology")
        if not self.first_step <= step <= self.last_step:
            raise IndexError("future-bound step is outside the prepared time horizon")
        return float(self._costs[step - self.first_step, cell])

    def root_lookup(self, cell: int, step: int) -> float:
        """Return a requested root's first-hop cost followed by a regular suffix.

        Parameters
        ------------
        - cell (int): local cell index from one of the supplied root variants.
        - step (int): absolute start step paired with that root cell.

        Return
        --------
        - output (float): first-hop relaxed bound, possibly infinity. Unrequested
          cell/time pairs raise ``KeyError`` instead of using an invalid regular bound.
        """
        return self._root_costs[cell, step]
