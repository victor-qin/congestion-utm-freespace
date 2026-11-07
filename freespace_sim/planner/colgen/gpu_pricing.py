"""Lossless GPU label expansion with CPU-verified numerical pruning.

Certified mode deliberately retains every continuation not excluded by a proven
bound. The production partial score is not yet a proven dominance measure for
canonical geometry costs. FP32 mode additionally uses GPU state grouping.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from . import pricing
from .dp_prepare import prepare_variants, prepared_for
from .gpu_bounds import CompletionBounds
from .gpu_endpoint import EndpointBounds, Mask
from .gpu_future import FutureBounds
from .gpu_prices import SparseVisitPrices, summary_for
from .gpu_prepare import GPUPreparation, packing_for
from .gpu_runtime import ComputeRuntime
from .network import RowKey
from .objective import cost_model
from .pricing_platform import PricingOptions, PricingProblem, PricingResult
from .windows import derive_cell_window

# Near-tie branches retain their proved upper bound in the result certificate.
# This tolerance never turns a candidate's score into the optimum's exact value.
_BOUND_CLOSURE_TOL = 1e-6
_INITIAL_FULL_LABEL_LIMIT = 8192


def _down(value: float) -> float:
    """Round one finite arithmetic result conservatively toward negative infinity."""
    return math.nextafter(value, -math.inf)


def _up(value: float) -> float:
    """Round one finite arithmetic result conservatively toward positive infinity."""
    return math.nextafter(value, math.inf)


def completion_upper_bound(
    benefit: float,
    pi_f: float,
    cost_lower: float,
    positive_prices: Sequence[float],
    negative_lower: float,
) -> float:
    """Bound canonical completion cost using only irrevocable prefix charges.

    Every canonical route claims its discrete visit rows, including folded
    corners. Positive prices must be deduplicated by row before this call.
    Including every negative input dual is conservative for every completion.
    Outward rounding covers the final FP64 reduced-cost subtraction order.

    Parameters
    ------------
    - benefit (float): flight benefit.
    - pi_f (float): assignment dual.
    - cost_lower (float): conservative weighted cost of every completion of this prefix.
    - positive_prices (Sequence[float]): distinct positive prefix row prices.
    - negative_lower (float): downward-rounded sum of all negative input prices.

    Return
    --------
    - output (float): upper bound on every completion's canonical reduced cost.
    """
    positive = 0.0
    for value in positive_prices:
        positive = _down(positive + value)
    dual_lower = _down(positive + negative_lower)
    return _up(_up(_up(benefit - _down(cost_lower)) - dual_lower) - pi_f)


class _Stopped(Exception):
    """A resource limit ended the search before every alternative was resolved."""


class _Batch:
    """Pack flights sharing visit offsets into one GPU time-layer search."""

    def __init__(
        self,
        problems: Sequence[PricingProblem],
        options: PricingOptions,
        runtime: ComputeRuntime,
        deadline: float | None,
    ) -> None:
        """Allocate bounded predecessor storage and initialize per-flight incumbents.

        Parameters
        ------------
        - problems (Sequence[PricingProblem]): flights with equal visit offsets.
        - options (PricingOptions): precision mode and allocation limits.
        - runtime (ComputeRuntime): selected physical GPU runtime.
        - deadline (float | None): shared monotonic deadline.

        Return
        --------
        - output (None): initializes batch state without searching on the CPU.
        """
        self.problems, self.options, self.runtime, self.deadline = (
            problems,
            options,
            runtime,
            deadline,
        )
        self.started = time.monotonic()
        self.first_dispatch = runtime.dispatches
        self.first_counters = dict(getattr(runtime, "counters", {}))
        self.best: list[tuple[float, Any]] = [(-math.inf, None) for _ in problems]
        self.closed_upper = [-math.inf for _ in problems]
        normalized: dict[int, pricing.DualView] = {}
        self.views: list[pricing.DualView] = []
        for p in problems:
            if isinstance(p.duals, pricing.DualView):
                view = p.duals
            else:
                # Raw mappings may change between calls; reuse them only within this batch.
                key = id(p.duals)
                if key not in normalized:
                    normalized[key] = pricing.DualView(p.duals, p.cfg)
                view = normalized[key]
            self.views.append(view)
        if any(
            view.offsets != derive_cell_window(p.cfg)
            for p, view in zip(problems, self.views, strict=True)
        ):
            raise ValueError("DualView visit offsets must match the problem configuration")
        self.models = [cost_model(p.cfg, p.params) for p in problems]
        self.benefits = [pricing._benefit(p.params) for p in problems]
        self.negative: list[float] = []
        self.arena = np.empty((options.max_labels, 3), np.int32)
        self.search_label_limit = options.max_labels
        self.prefix_positive = np.empty(options.max_labels, np.float64)
        self.visit_prefix_positive = np.empty(options.max_labels, np.float64)
        self.paid_masks = np.empty((options.max_labels, 2), np.uint32)
        self._visit_positive: dict[tuple[int, int, int], float] = {}
        self.future: list[FutureBounds | None] = [None for _ in problems]
        self.endpoints: list[EndpointBounds | None] = [None for _ in problems]
        self._endpoint_generation: tuple[int, ...] | None = None
        self._endpoint_visit_step: int | None = None
        self._endpoint_visits = np.empty((0, 2), np.uint32)
        self.used = 0
        self.stats = dict(
            candidates=0,
            cpu_recomputations=0,
            verified_prunes=0,
            gpu_prune_proposals=0,
            canonical_validations=0,
            max_frontier=0,
            grouped_labels=0,
            root_prunes=0,
            gpu_endpoint_prune_proposals=0,
            endpoint_verified_prunes=0,
            screened_departures=0,
        )
        self.offsets = self.views[0].offsets
        self.destinations = [pricing._destination_options(p.graph) for p in problems]
        self._visit_layers: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        self._forbidden_layers: dict[int, np.ndarray] = {}
        self.device_prices = None

    def check(self) -> None:
        """Stop cooperatively without converting an interrupted search into a certificate."""
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise _Stopped("deadline")

    def keep_departure(self, flight: int, departure: int) -> bool:
        """Screen a whole departure using only unavoidable ground cost and a valid witness.

        Parameters
        ------------
        - flight (int): index of the flight whose canonical incumbent is available.
        - departure (int): absolute departure step, at or after the flight's base step.

        Return
        --------
        - output (bool): False only when every completion is bounded by the incumbent.
        """
        self.check()
        p = self.problems[flight]
        ground = max(0.0, _down((departure - p.graph.base_step) * p.cfg.dt_s))
        cost = max(0.0, _down(ground * self.models[flight].ground_weight))
        upper = completion_upper_bound(
            self.benefits[flight], p.pi_f, cost, (), self.negative[flight],
        )
        # Use the witness itself, without a closure tolerance: it still covers these
        # roots after the future retry resets the separately accumulated closed upper.
        keep = not math.isfinite(upper) or upper > self.best[flight][0]
        self.stats["screened_departures"] += int(not keep)
        return keep

    def prepare(self) -> None:
        """Pack topology and roots that can improve an already validated incumbent.

        Parameters
        ------------
        - self (_Batch): batch carrying graphs, prices and resource limits.

        Return
        --------
        - output (None): initializes shared graph/root arrays and feasible incumbents.
        """
        topologies, variants = [], []
        self.negative = [summary_for(view, self.check).negative for view in self.views]
        self.bounds: list[CompletionBounds] = []
        self.cells: list[tuple[int, int]] = []
        self.cell_flights: list[int] = []
        self.cell_offsets = []
        self.root_flights, self.root_variants = [], []
        self.start_buckets: dict[int, list[int]] = defaultdict(list)
        graphs, roots, root_values, paid_rows, paid_values = [], [], [], [], []
        for i, (p, view, model, benefit) in enumerate(
            zip(self.problems, self.views, self.models, self.benefits, strict=True)
        ):
            self.check()
            if p.cfg != p.graph._cfg or not math.isfinite(p.pi_f):
                raise ValueError("Pricing needs the graph's configuration and a finite flight dual")
            if not all(
                math.isfinite(v) and v >= 0 for v in (model.ground_weight, model.air_weight)
            ):
                raise ValueError("Certified bounds require finite nonnegative objective weights")
            topology, rows = prepared_for(p.graph, p.cfg)
            if not topology.ok or not rows.ok:
                raise _Stopped("unsupported")
            # The reverse graph includes every legal role, so its distances remain
            # lower bounds while accounting for walls that plain hex distance ignores.
            remaining_hops = np.minimum(topology.rev_remaining, topology.air_hop_limit + 1)
            remaining = int(remaining_hops.max(initial=0))
            if (
                min(topology.min_step, topology.base_step) + min(0, self.offsets[0]) < -(2**31)
                or max(topology.max_step, topology.air_hop_limit)
                + remaining
                + max(1, self.offsets[1])
                >= 2**31
            ):
                raise ValueError("GPU time/hop arithmetic exceeds int32 range")
            self.check()
            try:
                # This seed helper never falls back to a full CPU pricing DP.
                seed = pricing._shortest_seed(p.graph, p.cfg, deadline=self.deadline, model=model)
            except ValueError:
                seed = None
            for column in (seed, p.known_column):
                if column is None:
                    continue
                column = pricing.certify_column(column, p.graph, p.cfg, model)
                if column.claims.isdisjoint(p.forbidden_rows):
                    rc = model.reduced_cost(
                        benefit=benefit,
                        cost=column.delay_s,
                        dual_cost=view.claim_cost(column.claims),
                        pi_f=p.pi_f,
                    )
                    if rc > self.best[i][0]:
                        self.best[i] = rc, column
            self.check()
            preparation = GPUPreparation(
                p.graph, p.cfg, view, benefit=benefit, pi_f=p.pi_f, model=model,
                forbidden_rows=p.forbidden_rows, retain_root_claims=True,
            )
            v = prepare_variants(
                p.graph,
                p.cfg,
                view,
                topology,
                rows,
                benefit=benefit,
                pi_f=p.pi_f,
                model=model,
                forbidden_rows=p.forbidden_rows,
                preparation=preparation,
                screen_departure=lambda departure: self.keep_departure(i, departure),
            )
            if not v.ok:
                raise _Stopped("unsupported")
            topologies.append(topology)
            variants.append(v)
            bounds = CompletionBounds(p, topology, v, view, preparation=preparation)
            self.bounds.append(bounds)
            if (self.options.endpoint_bounds and self.options.endpoint_max_rows and any(
                completion_upper_bound(
                    benefit, p.pi_f,
                    bounds.cost_lower(variant, int(remaining_hops[v.cell[variant]])),
                    (bounds.root_positive(variant),), self.negative[i],
                ) > self.best[i][0] + _BOUND_CLOSURE_TOL
                for variant in range(v.n_variants)
            )):
                self.endpoints[i] = EndpointBounds(bounds, self.options, self.check)
            offset = len(self.cells)
            self.cell_offsets.append(offset)
            packing = packing_for(p.graph, p.cfg, topology)
            self.cells.extend(packing.cells)
            self.cell_flights.extend([i] * topology.n_cells)
            graph = packing.graph.copy()
            neighbors = graph[:, :6]
            neighbors[neighbors >= 0] += offset
            graphs.append(graph)
            paid_offset = len(paid_values)
            paid_rows.extend(zip(map(int, v.paid_cell + offset), map(int, v.paid_step)))
            paid_values.extend(v.paid_value)
            for variant in range(v.n_variants):
                root = len(roots)
                paid_class = int(v.paid_class[variant])
                roots.append(
                    (
                        i,
                        paid_offset + v.paid_start[paid_class],
                        paid_offset + v.paid_start[paid_class + 1],
                        topology.max_step,
                        topology.air_hop_limit,
                        topology.revisit_depth,
                    )
                )
                root_values.append(
                    (
                        benefit,
                        p.pi_f,
                        v.ground_delay_s[variant],
                        -self.negative[i],
                        model.air_weight * p.cfg.dt_s,
                        self.best[i][0],
                        bounds.air_offset[variant],
                        bounds.air_per_hop,
                        bounds.root_positive(variant),
                        remaining_hops[v.cell[variant]],
                        math.inf,
                    )
                )
                self.root_flights.append(i)
                self.root_variants.append(variant)
                self.start_buckets[int(v.start_step[variant])].append(root)
            self.check()
        if len(self.cells) >= 2**27 or len(roots) >= 2**27 or len(paid_rows) >= 2**27:
            raise _Stopped("label_limit")
        self.topologies, self.variants = topologies, variants
        self.stride = 9 + max(t.state_history_depth for t in topologies)
        if (
            self.options.max_labels * self.stride >= 2**31
            or 6 * min(self.options.chunk_size, self.options.max_labels) * self.stride >= 2**31
        ):
            raise ValueError("GPU label indexing exceeds int32 range")
        self.graph = np.concatenate(graphs)
        self.roots = np.asarray(roots, np.int32).reshape(-1, 6)
        self.root_values = np.asarray(root_values, np.float32).reshape(-1, 11)
        self.paid_rows = np.asarray(paid_rows, np.int32).reshape(-1, 2)
        self.paid_values = np.asarray(paid_values, np.float32)
        self.root_endpoint = np.zeros((len(roots), 3), np.int32)
        for root, i in enumerate(self.root_flights):
            variant = self.root_variants[root]
            endpoint = self.endpoints[i]
            if endpoint is not None:
                self.root_endpoint[root, :2] = endpoint.root_masks[variant].view(np.int32)
            self.root_endpoint[root, 2] = self.variants[i].start_step[variant]
        self.pack_endpoints()
        self.evaluate_roots()
        self.stats["prepare_s"] = time.monotonic() - self.started

    def prepare_device_prices(self) -> None:
        """Bind one shared snapshot table and map stable physical IDs to its storage slots.

        Parameters
        ------------
        - self (_Batch): flights partitioned by immutable snapshot before execution.

        Return
        --------
        - output (None): live shared table and an independent flight-local vertex map.
        """
        if any(view is not self.views[0] for view in self.views):
            raise ValueError("GPU batch must share one immutable price snapshot")
        started = time.monotonic()
        self.prepare_visit_prices(gather=False)
        self.device_prices = self.runtime.prices.get(self.views[0], self.runtime, self.check)
        self.price_map = np.asarray([
            self.device_prices.slots.get((cell, 0), 0) for cell in self.cells
        ], np.int32)
        self.stats["price_prepare_s"] = time.monotonic() - started

    def evaluate_roots(self) -> None:
        """Dispatch all root completion proposals after initial packing or stronger preparation."""
        count = len(self.roots)
        (self.root_upper,) = self.runtime.run(
            "bound_roots", count,
            (np.asarray((count,), np.int32), self.root_values,
             np.empty(count, np.float32), self.roots, self.root_endpoint,
             self.endpoint_meta, self.endpoint_entries, self.endpoint_trees), (2,),
            static={3: "roots", 4: "root_endpoint", 5: "endpoint_meta",
                    6: "endpoint_entries", 7: "endpoint_trees"},
            write_only=(2,),
        )

    def pack_endpoints(self) -> None:
        """Refresh immutable GPU cache buffers only after new mask entries have been built.

        Parameters
        ------------
        - self (_Batch): snapshot-local per-flight caches; no weighted data persists across batches.

        Return
        --------
        - output (None): prepares shared FP32 proposal trees and explicit entry-validity metadata.
        """
        generation = tuple(len(e.entries) if e is not None else 0 for e in self.endpoints)
        if generation == self._endpoint_generation:
            return
        metadata, entries, arrays = [], [], []
        offset = 0
        for endpoint in self.endpoints:
            if endpoint is None:
                metadata.append((0, 0, 1, 0, 0))
                continue
            start = len(entries)
            for low, high in endpoint.entries:
                entries.append((low, high, offset, 1))
                offset += 4 * endpoint.leaves
            metadata.append((endpoint.first, endpoint.last, endpoint.leaves, start, len(entries)))
            arrays.append(endpoint.gpu_trees())
        if offset >= 2**31:
            raise ValueError("GPU endpoint-tree indexing exceeds int32 range")
        self.endpoint_meta = np.asarray(metadata, np.int32).reshape(-1, 5)
        self.endpoint_entries = np.asarray(entries, np.uint32).reshape(-1, 4).view(np.int32)
        self.endpoint_trees = np.concatenate(arrays) if arrays else np.empty(0, np.float32)
        self._endpoint_generation = generation

    def endpoint_visits(self, step: int) -> np.ndarray:
        """Gather exact visit masks once per layer, sharing them across expansion chunks.

        Parameters
        ------------
        - step (int): absolute time of the candidate visits.

        Return
        --------
        - output (np.ndarray): cell-aligned uint32 pairs, immutable until replaced for a new step.
        """
        if step != self._endpoint_visit_step:
            masks = np.zeros((len(self.cells), 2), np.uint32)
            for i, endpoint in enumerate(self.endpoints):
                if endpoint is not None and step in endpoint.visit_masks:
                    cells, words = endpoint.visit_masks[step]
                    masks[cells + self.cell_offsets[i]] = words
            self._endpoint_visits = masks
            self._endpoint_visit_step = step
        return self._endpoint_visits

    def paid_mask(self, root: int, parent: int, cell: int, hops: int) -> Mask:
        """Update parent-aligned exact masks independently of the GPU's proposed bits.

        Parameters
        ------------
        - root (int): packed departure/lane root.
        - parent (int): stable predecessor ID, or -1 for an initial root.
        - cell (int): packed current cell.
        - hops (int): completed moves, used to recover the exact absolute visit time.

        Return
        --------
        - output (Mask): original origin-and-visit union in the snapshot's fixed row mapping.
        """
        i, variant = self.root_flights[root], self.root_variants[root]
        endpoint = self.endpoints[i]
        if endpoint is None:
            return 0, 0
        if parent < 0:
            return tuple(map(int, endpoint.root_masks[variant]))
        step = int(self.variants[i].start_step[variant]) + hops
        visit = self.endpoint_visits(step)[cell]
        return tuple(map(int, self.paid_masks[parent] | visit))

    def prepare_future(self) -> None:
        """Build relaxed GPU lookahead only when the inexpensive full pass exhausts storage.

        Parameters
        ------------
        - self (_Batch): completed topology preparation and current validated incumbents.

        Return
        --------
        - output (None): installs future bounds and refreshes root proposals for a fresh pass.
        """
        started = time.monotonic()
        before = self.runtime.dispatches
        for i, problem in enumerate(self.problems):
            self.check()
            self.future[i] = FutureBounds(
                problem, self.topologies[i], self.variants[i], self.bounds[i],
                self.views[i], self.runtime, self.deadline,
                cpu_recompute=self.options.cpu_recompute,
            )
        for root, i in enumerate(self.root_flights):
            variant = self.root_variants[root]
            future = self.future[i]
            step = int(self.variants[i].start_step[variant])
            if future.first_step <= step <= future.last_step:
                lower = future.root_lookup(int(self.variants[i].cell[variant]), step)
                self.root_values[root, 10] = min(lower, float(np.finfo(np.float32).max))
        self.evaluate_roots()
        self.stats["future_s"] = time.monotonic() - started
        self.stats["future_gpu_dispatches"] = self.runtime.dispatches - before

    def path(self, parent: int, cell: int) -> tuple[tuple[int, int], ...]:
        """Reconstruct an immutable GPU-generated prefix from stable predecessor IDs."""
        cells = [self.cells[cell]]
        while parent >= 0:
            parent, previous, _ = map(int, self.arena[parent])
            cells.append(self.cells[previous])
        return tuple(reversed(cells))

    def positive_lower(self, root: int, parent: int, cell: int, hops: int) -> float:
        """Accumulate original positive prices without reconstructing a candidate path.

        Parameters
        ------------
        - root (int): departure/lane root.
        - parent (int): immutable predecessor ID, or -1 for a root.
        - cell (int): packed destination cell of this prefix.
        - hops (int): number of moves from the root.

        Return
        --------
        - output (float): downward bound on mandatory-origin and prefix-visit prices.

        The GPU forbids cell revisits within hi-lo moves, so different visits'
        time windows cannot overlap. Only the origin occupancy can overlap a
        later visit; excluding its exact claim set makes the additions disjoint.
        """
        i, variant = self.root_flights[root], self.root_variants[root]
        if parent < 0:
            return self.bounds[i].root_positive(variant)
        step = int(self.variants[i].start_step[variant]) + hops
        key = root, cell, step
        increment = self._visit_positive.get(key)
        if increment is None:
            claims = pricing._visit_claims(self.cells[cell], 0, step, self.offsets)
            origin = self.bounds[i].origin_claims(variant)
            increment = 0.0
            for row in claims - origin:
                price = self.views[i].row_cost(row)
                if price > 0.0:
                    increment = _down(increment + price)
            increment = max(0.0, increment)
            self._visit_positive[key] = increment
        return max(0.0, _down(float(self.prefix_positive[parent]) + increment))

    def visit_lower(self, root: int, parent: int, cell: int, hops: int) -> float:
        """Downward-sum disjoint visit windows without adding the separate origin endpoint."""
        i, variant = self.root_flights[root], self.root_variants[root]
        step = int(self.variants[i].start_step[variant]) + hops
        key = -1, cell, step
        increment = self._visit_positive.get(key)
        if increment is None:
            increment = 0.0
            for row in pricing._visit_claims(self.cells[cell], 0, step, self.offsets):
                price = self.views[i].row_cost(row)
                if price > 0.0:
                    increment = _down(increment + price)
            increment = max(0.0, increment)
            self._visit_positive[key] = increment
        return (increment if parent < 0 else
                max(0.0, _down(float(self.visit_prefix_positive[parent]) + increment)))

    def bound(self, root: int, parent: int, cell: int, hops: int) -> float:
        """Verify a deletion with original prices, disjoint prefix charges and geometry.

        Parameters
        ------------
        - root (int): packed departure/lane identifier.
        - parent (int): immutable predecessor ID, or -1 for a root.
        - cell (int): packed cell of this GPU-generated prefix.
        - hops (int): number of moves from the root.

        Return
        --------
        - output (float): bound on all valid canonical completions of this prefix.
        """
        i, variant = self.root_flights[root], self.root_variants[root]
        remaining = int(self.graph[cell, 12])
        self.stats["cpu_recomputations"] += 1
        upper = completion_upper_bound(
            self.benefits[i],
            self.problems[i].pi_f,
            self.bounds[i].cost_lower(variant, hops + remaining),
            (self.positive_lower(root, parent, cell, hops),),
            self.negative[i],
        )
        if self.future[i] is not None:
            step = int(self.variants[i].start_step[variant]) + hops
            local = cell - self.cell_offsets[i]
            suffix = (self.future[i].root_lookup(local, step) if hops == 0 else
                      self.future[i].lookup(local, step, self.topologies[i].air_hop_limit - hops))
            upper = min(upper, self.bounds[i].future_upper(
                variant, hops, self.visit_lower(root, parent, cell, hops), suffix,
                self.benefits[i], self.problems[i].pi_f, self.negative[i],
            ))
        self._endpoint_closed = False
        endpoint = self.endpoints[i]
        if endpoint is not None:
            start = int(self.variants[i].start_step[variant])
            step = start + hops
            last = min(self.topologies[i].max_step, start + self.topologies[i].air_hop_limit)
            if step + remaining > last:
                return -math.inf
            residual = endpoint.residual(
                self.paid_mask(root, parent, cell, hops), step, step + remaining, last,
            )
            if residual is not None:
                endpoint_upper = self.bounds[i].endpoint_upper(
                    variant, hops, remaining, self.positive_lower(root, parent, cell, hops),
                    residual, self.benefits[i], self.problems[i].pi_f, self.negative[i],
                )
                cutoff = self.best[i][0] + _BOUND_CLOSURE_TOL
                self._endpoint_closed = endpoint_upper <= cutoff < upper
                upper = min(upper, endpoint_upper)
        return upper

    def retain(self, root: int, parent: int, cell: int, hops: int) -> bool:
        """Require CPU verification for every proposed numeric deletion in certified mode."""
        if not self.options.cpu_recompute:
            return False
        i = self.root_flights[root]
        upper = self.bound(root, parent, cell, hops)
        if upper <= self.best[i][0] + _BOUND_CLOSURE_TOL:
            self.closed_upper[i] = max(self.closed_upper[i], upper)
            self.stats["verified_prunes"] += 1
            self.stats["endpoint_verified_prunes"] += int(self._endpoint_closed)
            return False
        return True

    def proposal_cutoff(self, flight: int) -> float:
        """Ask FP32 to flag near ties generously; only the FP64 verifier can discard them."""
        best = self.best[flight][0]
        if not self.options.cpu_recompute:
            return best
        scale = max(1.0, abs(self.benefits[flight]), abs(self.problems[flight].pi_f), abs(best))
        return best + max(_BOUND_CLOSURE_TOL, 1e-6 * scale) if math.isfinite(best) else best

    def allocate(self, records: np.ndarray) -> np.ndarray:
        """Assign predecessor IDs without ever truncating a full frontier on overflow."""
        end = self.used + len(records)
        if end > self.search_label_limit:
            raise _Stopped("label_limit")
        if self.options.cpu_recompute:
            for offset, record in enumerate(records):
                self.paid_masks[self.used + offset] = self.paid_mask(
                    int(record[1]), int(record[3]), int(record[0]), int(record[2]),
                )
                self.prefix_positive[self.used + offset] = self.positive_lower(
                    int(record[1]), int(record[3]), int(record[0]), int(record[2]),
                )
                self.visit_prefix_positive[self.used + offset] = self.visit_lower(
                    int(record[1]), int(record[3]), int(record[0]), int(record[2]),
                )
        self.arena[self.used : end] = records[:, [3, 0, 1]]
        records[:, 3] = np.arange(self.used, end, dtype=np.int32)
        self.used = end
        return records

    def roots_at(
        self, step: int, allowed_roots: frozenset[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize this time's roots, including those after an empty earlier frontier.

        Parameters
        ------------
        - step (int): absolute lattice time.
        - allowed_roots (frozenset[int] | None): heuristic-only root restriction.

        Return
        --------
        - output (tuple[np.ndarray, np.ndarray]): root records and FP32 score/price pairs.
        """
        records, values = [], []
        for root in self.start_buckets.get(step, ()):
            if allowed_roots is not None and root not in allowed_roots:
                continue
            self.check()
            i, variant = self.root_flights[root], self.root_variants[root]
            v = self.variants[i]
            cell = self.cell_offsets[i] + int(v.cell[variant])
            upper = self.root_upper[root]
            if upper <= np.float32(self.proposal_cutoff(i)):
                self.stats["gpu_prune_proposals"] += 1
                if not self.retain(root, -1, cell, 0):
                    self.stats["root_prunes"] += 1
                    continue
            record = np.full(self.stride, -1, np.int32)
            record[:7] = cell, root, 0, -1, -1, -1, 2
            record[7:9] = self.root_endpoint[root, :2]
            record[9] = cell
            records.append(record)
            positive = self.bounds[i].root_positive(variant)
            values.append((v.score[variant], positive))
        packed = np.asarray(records, np.int32).reshape(-1, self.stride)
        return self.allocate(packed), np.asarray(values, np.float32).reshape(-1, 2)

    def prepare_visit_prices(self, *, gather: bool = True) -> None:
        """Gather shared sparse prices and flight-specific exclusions for this batch.

        Parameters
        ------------
        - self (_Batch): prepared cell order and fixed price snapshots.
        - gather (bool): materialize host visit layers only for diagnostic callers.

        Return
        --------
        - output (None): stores sparse indices/values for each active visit time.
        """
        indices: dict[int, list[int]] = defaultdict(list)
        prices: dict[int, list[tuple[float, float]]] = defaultdict(list)
        forbidden: dict[int, set[int]] = defaultdict(set)
        lo, hi = self.offsets
        for i, (view, problem, topology) in enumerate(
            zip(self.views, self.problems, self.topologies, strict=True)
        ):
            self.check()
            if view._gpu_visit_prices is None:
                # Install only a completed cache, so an interrupted build cannot be reused.
                view._gpu_visit_prices = SparseVisitPrices(view, self.check)
            cache = view._gpu_visit_prices
            offset = self.cell_offsets[i]
            cells = self.cells[offset:offset + topology.n_cells]
            lookup = {coord: offset + local for local, coord in enumerate(cells)}
            if gather:
                for local, coord in enumerate(cells):
                    if local % 256 == 0:
                        self.check()
                    for step, value in cache.cells.get((coord, 0), {}).items():
                        indices[step].append(offset + local)
                        prices[step].append(value)
            for raw_row in problem.forbidden_rows:
                row = raw_row if isinstance(raw_row, RowKey) else RowKey(raw_row)
                if row.kind != "cell" or row.level != 0:
                    continue
                cell = lookup.get(row.cell_coord)
                if cell is not None:
                    for step in range(row.step - hi, row.step - lo + 1):
                        forbidden[step].add(cell)
        layers = {}
        for step, cells in indices.items():
            self.check()
            layers[step] = np.asarray(cells, np.intp), np.asarray(prices[step], np.float32)
        self._forbidden_layers = {
            step: np.fromiter(cells, dtype=np.intp, count=len(cells))
            for step, cells in forbidden.items()
        }
        self.check()
        self._visit_layers = layers

    def visits(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Prepare per-cell signed/positive visit sums and exact forbidden-row flags.

        Parameters
        ------------
        - step (int): absolute time of the next GPU-generated visit.

        Return
        --------
        - output (tuple[np.ndarray, np.ndarray]): FP32 signed/positive/future costs and exclusions.
        """
        if self._visit_layers is None:
            self.prepare_visit_prices()
        values = np.zeros((len(self.cells), 3), np.float32)
        values[:, 2] = math.inf
        priced = self._visit_layers.get(step)
        if priced is not None:
            indices, prices = priced
            values[indices, :2] = prices
        # A retry can install future bounds after price gathering; always read the live table.
        for i, future in enumerate(self.future):
            if future is not None and future.first_step <= step <= future.last_step:
                offset = self.cell_offsets[i]
                values[offset:offset + self.topologies[i].n_cells, 2] = future.row(step)
        blocked = np.zeros(len(self.cells), np.int32)
        forbidden = self._forbidden_layers.get(step)
        if forbidden is not None:
            blocked[forbidden] = 1
        return values, blocked

    def future_proposals(
        self, records: np.ndarray, visits: np.ndarray, step: int,
    ) -> np.ndarray:
        """Gather hop-aware proposals into the existing per-cell GPU visit buffer.

        Parameters
        ------------
        - records (np.ndarray): all parents that will read the returned buffer.
        - visits (np.ndarray): current signed/positive prices and old future proposals.
        - step (int): absolute time of their children.

        Return
        --------
        - output (np.ndarray): same buffer shape; each target uses the largest remaining
          allowance among all its candidate parents. This relaxes every actual child;
          FP64 verification still uses that child's exact allowance before deletion.
        """
        if not any(f is not None and f.hops is not None and f.hops.costs is not None
                   for f in self.future):
            return visits
        started = time.monotonic()
        available = np.full(len(self.cells), -1, np.int32)
        targets = self.graph[records[:, 0], :6].ravel()
        budgets = np.repeat(self.roots[records[:, 1], 4] - records[:, 2] - 1, 6)
        valid = targets >= 0
        np.maximum.at(available, targets[valid], budgets[valid])
        for i, future in enumerate(self.future):
            if (future is None or future.hops is None or future.hops.costs is None
                    or not future.first_step <= step < future.first_step + len(future.hops.costs)):
                continue
            hop = future.hops
            first, count = self.cell_offsets[i], self.topologies[i].n_cells
            allowance = available[first:first + count]
            slack = allowance.astype(np.int64) - self.topologies[i].rev_remaining
            index = hop.structure.offsets[:-1] + slack
            cells = np.flatnonzero((slack >= 0) & (index < hop.structure.offsets[1:]))
            values = hop.costs[step - future.first_step, index[cells]]
            # Infinity in the device ABI also means "unavailable". A finite maximum
            # asks for verification of a proved empty table state without losing the proposal.
            lower = np.maximum(visits[first + cells, 2], values)
            visits[first + cells, 2] = np.minimum(lower, float(np.finfo(np.float32).max))
        self.stats["hop_gather_s"] = self.stats.get("hop_gather_s", 0.0) + time.monotonic() - started
        return visits

    def complete(self, record: np.ndarray, path: tuple[tuple[int, int], ...]) -> None:
        """Validate every destination lane before any continuation grouping can discard it.

        Parameters
        ------------
        - record (np.ndarray): one GPU-generated candidate's integer state.
        - path (tuple[tuple[int, int], ...]): reconstructed candidate prefix.

        Return
        --------
        - output (None): updates the incumbent only with canonical feasible trajectories.
        """
        root = int(record[1])
        i, variant = self.root_flights[root], self.root_variants[root]
        p, v = self.problems[i], self.variants[i]
        lane = int(v.lane_idx[variant])
        label = pricing._Label(
            0.0, int(v.departure_step[variant]), None if lane < 0 else lane, path, frozenset()
        )
        for dest_lane in self.destinations[i].get(path[-1], ()):
            self.check()
            self.stats["canonical_validations"] += 1
            candidate = pricing._Candidate(0.0, 0.0, label, dest_lane)
            result = pricing._canonical_candidate(
                candidate,
                p.graph,
                self.views[i],
                p.pi_f,
                p.cfg,
                self.benefits[i],
                p.forbidden_rows,
                self.models[i],
            )
            if result is not None and result[0] > self.best[i][0]:
                self.best[i] = result

    def group(
        self, records: np.ndarray, scores: np.ndarray, *, heuristic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select FP32 state winners on the GPU only in explicitly uncertified mode.

        Parameters
        ------------
        - records (np.ndarray): continuing labels from every chunk of one time layer.
        - scores (np.ndarray): FP32 score, positive-prefix sum and bound per label.
        - heuristic (bool): permits approximate grouping only in the incumbent search.

        Return
        --------
        - output (tuple[np.ndarray, np.ndarray]): retained records and aligned scores.
        """
        if (self.options.cpu_recompute and not heuristic) or not len(records):
            return records, scores
        capacity = 1 << (2 * len(records) - 1).bit_length()
        (table,) = self.runtime.run(
            "group_best",
            len(records),
            (
                np.asarray((len(records), self.stride, capacity), np.int32),
                records,
                scores,
                np.full(capacity, -1, np.int32),
            ),
            (3,),
        )
        indices = np.sort(table[table >= 0])
        self.stats["grouped_labels"] += len(records) - len(indices)
        return records[indices].copy(), scores[indices].copy()

    def select_beam(
        self, records: np.ndarray, scores: np.ndarray, step: int, width: int,
    ) -> np.ndarray:
        """Rank preliminary routes by unpaid endpoints and reserve root/arrival diversity.

        Parameters
        ------------
        - records (np.ndarray): GPU-generated continuing states, before allocation.
        - scores (np.ndarray): approximate signed prefix scores and pruning proposals.
        - step (int): absolute time of these states.
        - width (int): maximum retained states per flight in this heuristic pass.

        Return
        --------
        - output (np.ndarray): selected indices; these omissions never enter a certificate.
        """
        flights = np.asarray(self.root_flights)[records[:, 1]]
        remaining = self.graph[records[:, 0], 12]
        cost = remaining * self.root_values[records[:, 1], 4]
        selected = []
        for i in range(len(self.problems)):
            choices = np.flatnonzero(flights == i)
            endpoint = self.endpoints[i]
            if endpoint is not None and not endpoint.rows:
                endpoint = None
            if endpoint is None and self.future[i] is None:
                priority = scores[choices, 0] - cost[choices]
                order = np.lexsort((choices, remaining[choices], -priority))
                selected.extend(map(int, choices[order[:width]]))
                continue
            if endpoint is not None:
                endpoint.consider(records[choices, 7:9].copy().view(np.uint32))
            for index in choices:
                self.check()
                record = records[index]
                root, hops = int(record[1]), int(record[2])
                if self.future[i] is not None:
                    cost[index] = max(cost[index], self.future[i].lookup(
                        int(record[0]) - self.cell_offsets[i], step,
                        self.topologies[i].air_hop_limit - hops,
                    ))
                if endpoint is not None:
                    last = min(self.topologies[i].max_step,
                               step + self.topologies[i].air_hop_limit - hops)
                    residual = endpoint.residual(
                        tuple(map(int, record[7:9].view(np.uint32))),
                        step, step + int(remaining[index]), last,
                    )
                    if residual is not None:
                        cost[index] = max(cost[index], residual[1],
                                          residual[0] + remaining[index]
                                          * self.root_values[root, 4])
            priority = scores[choices, 0] - cost[choices]
            order = choices[np.lexsort((choices, remaining[choices], -priority))]
            # A busy departure must not consume every slot before other promising
            # roots or earliest-arrival regimes can produce a feasible completion.
            reserved, groups = [], set()
            for index in order:
                key = int(records[index, 1]), int(remaining[index])
                if key not in groups:
                    reserved.append(int(index))
                    groups.add(key)
                    if len(reserved) >= width // 2:
                        break
            chosen = set(reserved)
            reserved.extend(int(index) for index in order if index not in chosen)
            selected.extend(reserved[:width])
        return np.asarray(selected, np.intp)

    def search(self, beam_width: int | None = None) -> None:
        """Expand immutable time frontiers on the GPU and account for every completion.

        Parameters
        ------------
        - self (_Batch): fully prepared batch with bounded predecessor storage.
        - beam_width (int | None): per-flight heuristic frontier cap; None searches all roots.

        Return
        --------
        - output (None): exhaustion completes the search; limits raise ``_Stopped``.
        """
        if not self.start_buckets:
            return
        allowed_roots = None
        first_step = min(self.start_buckets)
        last_step = max(t.max_step for t in self.topologies)
        root_flights = np.asarray(self.root_flights, np.int32)
        if beam_width is not None:
            selected = []
            for i in range(len(self.problems)):
                roots = np.flatnonzero(root_flights == i)
                order = np.argsort(-self.root_upper[roots], kind="stable")
                selected.extend(map(int, roots[order[:32]]))
            allowed_roots = frozenset(selected)
            steps = [step for step, roots in self.start_buckets.items()
                     if any(root in allowed_roots for root in roots)]
            if not steps:
                return
            first_step = min(steps)
            last_step = min(last_step, max(steps) + max(t.air_hop_limit for t in self.topologies))
        frontier = np.empty((0, self.stride), np.int32)
        values = np.empty((0, 2), np.float32)
        for step in range(first_step, last_step + 1):
            self.check()
            self._visit_positive.clear()
            roots, root_scores = self.roots_at(step, allowed_roots)
            frontier = np.concatenate((frontier, roots))
            values = np.concatenate((values, root_scores))
            self.stats["max_frontier"] = max(self.stats["max_frontier"], len(frontier))
            if not len(frontier):
                continue
            for i, endpoint in enumerate(self.endpoints):
                if endpoint is not None:
                    selected = root_flights[frontier[:, 1]] == i
                    endpoint.consider(frontier[selected, 7:9].copy().view(np.uint32))
            self.pack_endpoints()
            visits, blocked = self.visits(step + 1)
            visits = self.future_proposals(frontier, visits, step + 1)
            visit_masks = self.endpoint_visits(step + 1)
            future_values = np.ascontiguousarray(visits[:, 2])
            next_records, next_scores = [], []
            next_count = 0
            for start in range(0, len(frontier), self.options.chunk_size):
                self.check()
                current = frontier[start : start + self.options.chunk_size].copy()
                cutoffs = np.asarray([
                    self.proposal_cutoff(i) for i in range(len(self.problems))
                ], np.float32)
                self.root_values[:, 5] = cutoffs[root_flights]
                count = len(current) * 6
                records, scores = self.runtime.run(
                    "expand",
                    count,
                    (
                        np.asarray((len(current), self.stride, step, *self.offsets), np.int32),
                        current,
                        values[start : start + len(current)].copy(),
                        self.graph,
                        self.roots,
                        self.root_values,
                        future_values,
                        blocked,
                        self.paid_rows,
                        self.paid_values,
                        np.empty((count, self.stride), np.int32),
                        np.empty((count, 3), np.float32),
                        self.price_map, self.device_prices.ranges,
                        self.device_prices.steps, self.device_prices.values,
                        visit_masks, self.endpoint_meta, self.endpoint_entries, self.endpoint_trees,
                    ),
                    (10, 11),
                    static={3: "graph", 4: "roots", 6: "future_values", 7: "blocked",
                            8: "paid_rows", 9: "paid_values", 12: "price_map",
                            16: "endpoint_visits", 17: "endpoint_meta", 18: "endpoint_entries",
                            19: "endpoint_trees"},
                    write_only=(10, 11),
                )
                valid = np.flatnonzero(records[:, 6])
                self.stats["candidates"] += len(valid)
                keep = []
                for index in valid:
                    self.check()
                    record = records[index]
                    root, flags = int(record[1]), int(record[6])
                    path = None
                    if flags & 4:
                        self.stats["gpu_prune_proposals"] += 1
                        self.stats["gpu_endpoint_prune_proposals"] += int(bool(flags & 8))
                        if not self.retain(root, int(record[3]), int(record[0]), int(record[2])):
                            continue
                    if flags & 1:
                        if path is None:
                            path = self.path(int(record[3]), int(record[0]))
                        self.complete(record, path)
                    if flags & 2:
                        keep.append(index)
                if keep:
                    next_records.append(records[keep])
                    next_scores.append(scores[keep])
                    next_count += len(keep)
                    if beam_width is None and self.used + next_count > self.search_label_limit:
                        raise _Stopped("label_limit")
            if next_records:
                frontier, scores = self.group(
                    np.concatenate(next_records), np.concatenate(next_scores),
                    heuristic=beam_width is not None,
                )
                if beam_width is not None:
                    selected = self.select_beam(frontier, scores, step + 1, beam_width)
                    frontier, scores = frontier[selected], scores[selected]
                frontier = self.allocate(frontier)
                values = scores[:, :2].copy()
            else:
                frontier = np.empty((0, self.stride), np.int32)
                values = np.empty((0, 2), np.float32)

    def bootstrap(self) -> None:
        """Find validated incumbents with bounded GPU search before the full certificate pass.

        Parameters
        ------------
        - self (_Batch): prepared roots and topology, with shared canonical incumbents.

        Return
        --------
        - output (None): preserves only validated routes; the full pass restarts all roots.
        """
        started = time.monotonic()
        before = self.runtime.dispatches
        try:
            self.search(beam_width=64)
        except _Stopped as stop:
            if str(stop) != "label_limit":
                raise
        finally:
            self.stats["bootstrap_labels"] = self.stats.get("bootstrap_labels", 0) + self.used
            self.stats["bootstrap_gpu_dispatches"] = (self.stats.get("bootstrap_gpu_dispatches", 0)
                                                     + self.runtime.dispatches - before)
            self.stats["bootstrap_s"] = self.stats.get("bootstrap_s", 0.0) + time.monotonic() - started
            # No predecessor IDs from the heuristic stage enter the certificate pass.
            self.used = 0

    def run(self) -> tuple[PricingResult, ...]:
        """Return validated incumbents and certify only complete, CPU-refined searches.

        Parameters
        ------------
        - self (_Batch): batch with a shared numeric mode and resource limits.

        Return
        --------
        - output (tuple[PricingResult, ...]): ordered incumbents, statuses and batch counters.
        """
        status = "complete" if self.options.cpu_recompute else "approximate"
        complete = False
        try:
            self.prepare()
            self.prepare_device_prices()
            self.bootstrap()
            try:
                # Stop an unproductive first pass early enough to benefit from the
                # future-informed beam. The retry still searches every root at the
                # caller's original cap; this limit never authorizes a certificate.
                self.search_label_limit = min(self.options.max_labels, _INITIAL_FULL_LABEL_LIMIT)
                self.stats["initial_full_label_limit"] = self.search_label_limit
                try:
                    self.search()
                finally:
                    self.search_label_limit = self.options.max_labels
            except _Stopped as stop:
                if str(stop) != "label_limit":
                    raise
                self.stats["initial_full_search_labels"] = self.used
                self.used = 0
                self.closed_upper = [-math.inf for _ in self.problems]
                self.prepare_future()
                self.bootstrap()
                self.search()
            self.check()
            complete = True
        except _Stopped as stop:
            status = str(stop)
        except pricing.PricingTimeout:
            status = "deadline"
        finally:
            self.runtime.clear_batch()
        certified = complete and self.options.cpu_recompute
        stats = {
            **self.stats,
            **{key: value - self.first_counters[key]
               for key, value in self.runtime.counters.items()},
            "labels": (self.used + self.stats.get("bootstrap_labels", 0)
                       + self.stats.get("initial_full_search_labels", 0)),
            "full_search_labels": self.used,
            "device_name": self.runtime.device_name,
            "gpu_dispatches": self.runtime.dispatches - self.first_dispatch,
            "elapsed_s": time.monotonic() - self.started,
            "batch_size": len(self.problems),
            "counts_scope": "batch",
            "dominance": "none" if self.options.cpu_recompute else "fp32",
            "bound_closure_tolerance": _BOUND_CLOSURE_TOL,
        }
        results = []
        for i, (p, (rc, column)) in enumerate(zip(self.problems, self.best, strict=True)):
            if (p.require_improving and rc <= pricing._IMPROVING_RC_TOL) or column == p.known_column:
                column = None
            results.append(
                PricingResult(
                    p.graph.request.flight_id,
                    rc,
                    column,
                    max(rc, self.closed_upper[i]) if certified else None,
                    complete,
                    certified,
                    self.runtime.backend,
                    self.options.cpu_recompute,
                    status,
                    dict(stats, endpoint=(dict(self.endpoints[i].stats)
                                          if self.endpoints[i] is not None else {}),
                         future=(dict(self.future[i].stats) if self.future[i] is not None else {}),
                         bound_gap=(max(rc, self.closed_upper[i]) - rc
                                          if certified and math.isfinite(rc) else None)),
                )
            )
        return tuple(results)


def price_gpu_batch(
    problems: Sequence[PricingProblem],
    options: PricingOptions,
    runtime: ComputeRuntime,
    deadline: float | None,
) -> tuple[PricingResult, ...]:
    """Batch compatible flights and preserve caller order without running CPU pricing.

    Parameters
    ------------
    - problems (Sequence[PricingProblem]): flights with fixed prices and graphs.
    - options (PricingOptions): numeric mode, chunk size and label cap.
    - runtime (ComputeRuntime): reusable physical GPU runtime.
    - deadline (float | None): shared monotonic deadline.

    Return
    --------
    - output (tuple[PricingResult, ...]): ordered validated incumbents and certificate statuses.
    """
    groups: dict[pricing.DualView, list[int]] = defaultdict(list)
    normalized: dict[tuple[int, tuple[int, int]], pricing.DualView] = {}
    current = []
    for index, problem in enumerate(problems):
        view = problem.duals
        if not isinstance(view, pricing.DualView):
            key = id(view), derive_cell_window(problem.cfg)
            if key not in normalized:
                normalized[key] = pricing.DualView(view, problem.cfg)
            view = normalized[key]
        current.append(replace(problem, duals=view))
        groups[view].append(index)
    results: list[PricingResult | None] = [None] * len(problems)
    for indices in groups.values():
        batch = _Batch([current[i] for i in indices], options, runtime, deadline)
        for index, result in zip(indices, batch.run(), strict=True):
            results[index] = result
    return tuple(results)
