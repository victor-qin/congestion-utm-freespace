"""Paid endpoint rows and bounded, root-independent arrival-price minima."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
import time

import numpy as np

from . import pricing
from .gpu_bounds import CompletionBounds
from .gpu_future import _add_lower, _float32_lower
from .network import RowKey
from .pricing_platform import PricingOptions

Mask = tuple[int, int]


@dataclass(frozen=True, slots=True)
class _Membership:
    """Price-independent nominal endpoint incidence for one configuration/time domain."""

    cfg: object
    first: int
    last: int
    arrivals: dict[RowKey, np.ndarray]


def _range_min(tree: np.ndarray, first: int, last: int) -> float:
    """Query an inclusive nonempty leaf interval without allocating an arrival slice."""
    first += len(tree) // 2
    last += len(tree) // 2 + 1
    result = math.inf
    while first < last:
        if first & 1:
            result = min(result, float(tree[first]))
            first += 1
        if last & 1:
            last -= 1
            result = min(result, float(tree[last]))
        first //= 2
        last //= 2
    return result


class EndpointBounds:
    """Track at most 64 positive endpoint rows within one immutable price snapshot.

    Only geometric membership persists across snapshots. Each paid mask owns two
    FP64 range-minimum trees, Q and Q plus affine arrival cost; roots share both.
    Unbuilt masks never imply infeasibility. Budgets bound weighted build work and
    resident numeric cache bytes (FP64 verification, FP32 upload and device copy).
    """

    def __init__(
        self, bounds: CompletionBounds, options: PricingOptions, check: Callable[[], None],
    ) -> None:
        """Select a fresh row mapping and initialize lazy original-price caches.

        Parameters
        ------------
        - bounds (CompletionBounds): original prices, root unions and proved geometry.
        - options (PricingOptions): row limit and per-flight weighted-cache budgets.
        - check (Callable): cooperative deadline check; partial builds are not published.

        Return
        --------
        - output (None): initializes a snapshot-local mask mapping and empty weighted cache.
        """
        started = time.monotonic()
        self.bounds, self.options, self.check = bounds, options, check
        p, t, v = bounds.problem, bounds.topology, bounds.variants
        self.first = min(int(v.start_step.min()), int(t.max_step)) if v.n_variants else t.max_step
        self.last = int(t.max_step)
        self.size = self.last - self.first + 1
        self.leaves = 1 << (self.size - 1).bit_length()
        self.entries: dict[Mask, np.ndarray] = {}
        self.stats: dict[str, int | float] = dict(
            endpoint_rows=0, endpoint_cache_entries=0, endpoint_cache_bytes=0,
            endpoint_build_work=0, endpoint_build_s=0.0, endpoint_cache_hits=0,
            endpoint_cache_misses=0, endpoint_cache_skips=0,
            endpoint_max_layer_masks=0,
        )
        self.check()
        cache = p.graph._search_cache
        with cache.lock:
            membership = cache.gpu_endpoints
        if (membership is None or membership.cfg != p.cfg
                or membership.first != self.first or membership.last != self.last):
            arrivals: dict[RowKey, list[int]] = defaultdict(list)
            for offset in range(self.size):
                self.check()
                for row in pricing._endpoint_claims(
                    p.graph, p.cfg, origin=False, step=self.first + offset, timing_steps=0,
                ):
                    arrivals[row].append(offset)
            packed = {row: np.asarray(indices, np.intp) for row, indices in arrivals.items()}
            for indices in packed.values():
                indices.flags.writeable = False
            membership = _Membership(p.cfg, self.first, self.last, packed)
            self.check()
            with cache.lock:
                cache.gpu_endpoints = membership
        self.membership = membership
        selected = [row for row in membership.arrivals if bounds.view.row_cost(row) > 0.0]
        # RowKey is a complete tuple identity, including resource, level and absolute time.
        selected.sort(key=lambda row: (-bounds.view.row_cost(row), tuple(row)))
        self.rows = tuple(selected[:options.endpoint_max_rows])
        self.bits = {row: (bit // 32, 1 << (bit % 32)) for bit, row in enumerate(self.rows)}
        self.weights = np.asarray([bounds.view.row_cost(row) for row in self.rows], np.float64)
        self.stats["endpoint_rows"] = len(self.rows)
        self.root_masks = np.asarray([
            self.mask(bounds.origin_claims(root)) for root in range(v.n_variants)
        ], np.uint32).reshape(-1, 2)
        cell_index = {
            (int(q), int(r)): cell
            for cell, (q, r) in enumerate(zip(t.cell_q, t.cell_r, strict=True))
        }
        visits: dict[int, dict[int, list[int]]] = defaultdict(dict)
        lo, hi = bounds.view.offsets
        for row, (word, bit) in self.bits.items():
            self.check()
            if row.kind != "cell" or row.level != 0:
                continue
            cell = cell_index.get(row.cell_coord)
            if cell is None:
                continue
            for step in range(max(self.first, row.step - hi), min(self.last, row.step - lo) + 1):
                visits[step].setdefault(cell, [0, 0])[word] |= bit
        self.visit_masks = {
            step: (np.asarray(list(cells), np.intp), np.asarray(list(cells.values()), np.uint32))
            for step, cells in visits.items()
        }
        self.check()
        self.stats["endpoint_prepare_s"] = time.monotonic() - started
        self.consider(self.root_masks)

    def mask(self, claims: Iterable[RowKey]) -> Mask:
        """Map an exact claim union into the two unsigned words fixed for this snapshot."""
        words = [0, 0]
        for row in claims:
            selected = self.bits.get(row)
            if selected is not None:
                word, bit = selected
                words[word] |= bit
        return words[0], words[1]

    def consider(self, masks: np.ndarray) -> None:
        """Build frequent encountered masks within hard byte and arithmetic-work budgets.

        Parameters
        ------------
        - masks (np.ndarray): aligned uint32 pairs from roots or a surviving frontier.

        Return
        --------
        - output (None): adds complete trees only; skipped entries retain safe fallback behavior.
        """
        if not self.rows or not len(masks):
            return
        unique, counts = np.unique(masks, axis=0, return_counts=True)
        self.stats["endpoint_max_layer_masks"] = max(
            self.stats["endpoint_max_layer_masks"], len(unique),
        )
        for index in np.argsort(-counts, kind="stable"):
            self.check()
            key = tuple(map(int, unique[index]))
            if key in self.entries:
                continue
            # Two trees, each with 2*leaves nodes; include host/device metadata copies.
            byte_cost = 4 * self.leaves * (8 + 4 + 4) + 2 * 4 * 4
            work = self.size * (len(self.rows) + 2) + 2 * self.leaves
            if (self.stats["endpoint_cache_bytes"] + byte_cost > self.options.endpoint_cache_bytes
                    or self.stats["endpoint_build_work"] + work > self.options.endpoint_cache_work):
                self.stats["endpoint_cache_skips"] += 1
                continue
            started = time.monotonic()
            q = np.zeros(self.size, np.float64)
            for bit, row in enumerate(self.rows):
                self.check()
                if not (key[bit // 32] & (1 << (bit % 32))):
                    indices = self.membership.arrivals[row]
                    q[indices] = _add_lower(q[indices], float(self.weights[bit]))
            with np.errstate(over="ignore", under="ignore"):
                air = np.maximum(0.0, np.nextafter(
                    np.arange(self.size, dtype=np.float64) * self.bounds.air_per_hop, -np.inf,
                ))
            trees = np.full((2, 2 * self.leaves), np.inf, np.float64)
            trees[0, self.leaves:self.leaves + self.size] = q
            trees[1, self.leaves:self.leaves + self.size] = _add_lower(q, air)
            width = self.leaves
            while width > 1:
                self.check()
                trees[:, width // 2:width] = np.minimum(
                    trees[:, width:2 * width:2], trees[:, width + 1:2 * width:2],
                )
                width //= 2
            self.check()
            trees.flags.writeable = False
            self.entries[key] = trees
            self.stats["endpoint_cache_entries"] += 1
            self.stats["endpoint_cache_bytes"] += byte_cost
            self.stats["endpoint_build_work"] += work
            self.stats["endpoint_build_s"] += time.monotonic() - started

    def residual(self, mask: Mask, step: int, first: int, last: int) -> tuple[float, float] | None:
        """Query Q and affine-delay minima from original FP64 prices for a nonempty interval.

        Parameters
        ------------
        - mask (Mask): independently retained origin-and-visit paid bits.
        - step (int): current absolute label time, at or after the earliest root.
        - first (int): earliest relaxed arrival, inclusive.
        - last (int): latest arrival allowed by horizon and root hop limit, inclusive.

        Return
        --------
        - output (tuple[float, float] | None): downward endpoint/air residuals, or None on
          an unbuilt mask/domain miss. Empty proved domains are handled by the caller.
        """
        tree = self.entries.get(mask)
        if tree is None or not self.first <= step <= first <= last <= self.last:
            self.stats["endpoint_cache_misses"] += 1
            return None
        self.stats["endpoint_cache_hits"] += 1
        q = _range_min(tree[0], first - self.first, last - self.first)
        h = _range_min(tree[1], first - self.first, last - self.first)
        elapsed = math.nextafter(self.bounds.air_per_hop * (step - self.first), math.inf)
        d = max(0.0, math.nextafter(h - elapsed, -math.inf))
        return q, d

    def gpu_trees(self) -> np.ndarray:
        """Return conservative FP32 trees for proposals; verification keeps the FP64 originals."""
        if not self.entries:
            return np.empty(0, np.float32)
        return np.concatenate([_float32_lower(tree).ravel() for tree in self.entries.values()])
