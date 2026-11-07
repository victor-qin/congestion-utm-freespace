"""Bounded, price-independent flight data for GPU preparation."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import numpy as np

from ...config import SimConfig
from . import pricing
from .dp_prepare import PricingPreparation, PreparedTopology, _prepare_origin_data
from .network import FlightGraph, RowKey

# Count retained row references, including empty sets, rather than just root keys.
_MAX_CLAIM_ROWS = 16_384
_MAX_STARTS = 4096


class FlightPacking:
    """One graph's immutable local GPU layout and bounded exact unpriced root data."""

    def __init__(self, graph: FlightGraph, cfg: SimConfig, topology: PreparedTopology) -> None:
        """Build only fixed geometry; prices, objectives and exclusions remain call-local.

        Parameters
        ------------
        - graph (FlightGraph): immutable flight domain owning this cache.
        - cfg (SimConfig): graph configuration, including occupancy clocks.
        - topology (PreparedTopology): existing immutable topology packing.

        Return
        --------
        - output (None): initializes local adjacency and bounded membership caches.
        """
        self.topology, self.cfg = topology, cfg
        self.cells = tuple(zip(map(int, topology.cell_q), map(int, topology.cell_r)))
        self.origin_data = _prepare_origin_data(graph, cfg, topology)
        self.graph = np.full((topology.n_cells, 14), -1, np.int32)
        self.graph[:, 6:12] = 0
        self.graph[:, 12] = np.minimum(topology.rev_remaining, topology.air_hop_limit + 1)
        self.graph[:, 13] = topology.dest_mask
        for cell in range(topology.n_cells):
            start, stop = map(int, topology.arc_start[cell:cell + 2])
            self.graph[cell, :stop - start] = topology.arc_target[start:stop]
            self.graph[cell, 6:6 + stop - start] = topology.arc_roles[start:stop]
        self.graph.flags.writeable = False
        self.claims: OrderedDict[tuple, frozenset[RowKey]] = OrderedDict()
        self.starts: OrderedDict[int, tuple] = OrderedDict()
        self.claim_rows = 0
        self.start_count = 0
        self.claim_hits = self.claim_builds = 0

    def membership(self, key: tuple, build: Callable[[], frozenset[RowKey]]) -> frozenset[RowKey]:
        """Cache complete exact membership, publishing only successfully built sets.

        Parameters
        ------------
        - key (tuple): departure and lane/first-visit description, never a price key.
        - build (Callable): canonical unpriced construction on a cache miss.

        Return
        --------
        - output (frozenset[RowKey]): all rows, including currently unpriced rows.
        """
        if key in self.claims:
            self.claim_hits += 1
            self.claims.move_to_end(key)
            return self.claims[key]
        value = build()
        self.claim_builds += 1
        size = max(1, len(value))
        # Repeated sweeps visit the same roots in order. LRU eviction would cycle
        # through an oversized flight with no warm hits; retain bounded admissions.
        if self.claim_rows + size <= _MAX_CLAIM_ROWS:
            self.claims[key] = value
            self.claim_rows += size
        return value


def packing_for(graph: FlightGraph, cfg: SimConfig, topology: PreparedTopology) -> FlightPacking:
    """Reuse one fixed packing alongside the graph's existing topology cache.

    Parameters
    ------------
    - graph (FlightGraph): fixed graph owning its search caches.
    - cfg (SimConfig): configuration used to build the graph.
    - topology (PreparedTopology): topology identity also invalidates the packing.

    Return
    --------
    - output (FlightPacking): price-independent cached graph and unpriced memberships.
    """
    cache = graph._search_cache
    with cache.lock:
        value = cache.gpu_packing
        if value is None or value.topology is not topology or value.cfg != cfg:
            value = FlightPacking(graph, cfg, topology)
            cache.gpu_packing = value
        return value


class GPUPreparation(PricingPreparation):
    """Current prices and restrictions over graph-owned unpriced root descriptions."""

    @property
    def packing(self) -> FlightPacking:
        """Resolve packing using the same topology as the shared CPU preparation."""
        if not hasattr(self, "_packing"):
            self._packing = packing_for(self.fg, self.cfg, self.topology_rows[0])
        return self._packing

    def origin_data(self, topology: PreparedTopology) -> tuple:
        """Reuse geometry without caching weighted costs or current root filtering."""
        return packing_for(self.fg, self.cfg, topology).origin_data

    def legal_starts(self, departure: int, topology: PreparedTopology, build: Callable) -> tuple:
        """Reuse legal unpriced starts with a bounded cache, before current root filters.

        Parameters
        ------------
        - departure (int): absolute departure time.
        - topology (PreparedTopology): current immutable search domain.
        - build (Callable): computes starts using the shared preparation's domain guards.

        Return
        --------
        - output (tuple): lane, cell, local index, start time and remaining distance.
        """
        packing = packing_for(self.fg, self.cfg, topology)
        with self.fg._search_cache.lock:
            if departure in packing.starts:
                packing.starts.move_to_end(departure)
                return packing.starts[departure]
            value = tuple(build())
            size = max(1, len(value))
            if size <= _MAX_STARTS:
                while packing.starts and packing.start_count + size > _MAX_STARTS:
                    _, old = packing.starts.popitem(last=False)
                    packing.start_count -= max(1, len(old))
                packing.starts[departure] = value
                packing.start_count += size
            return value

    def origin_claims(self, departure: int) -> frozenset[RowKey]:
        """Reuse exact endpoint sets including terminals and zero-priced rows."""
        if departure in self._origin_claims:
            return self._origin_claims[departure]
        with self.fg._search_cache.lock:
            claims = self.packing.membership((departure,), lambda: super(
                GPUPreparation, self).origin_claims(departure))
        self._origin_claims[departure] = claims
        return claims

    def root_claims(
        self, departure: int, lane: int | None, origin_claims: frozenset[RowKey],
        cell: tuple[int, int], start_step: int,
    ) -> frozenset[RowKey]:
        """Reuse the full origin/first-visit union independently of active price support.

        Parameters
        ------------
        - departure (int): absolute departure time.
        - lane (int | None): current origin lane identity.
        - origin_claims (frozenset[RowKey]): complete canonical endpoint membership.
        - cell (tuple[int, int]): first visited physical cell.
        - start_step (int): absolute first-visit time.

        Return
        --------
        - output (frozenset[RowKey]): exact union used for repricing and overlap verification.
        """
        local_key = departure, lane
        if self._root_claims is not None and local_key in self._root_claims:
            return self._root_claims[local_key]
        key = departure, lane, cell, start_step, self.view.offsets
        with self.fg._search_cache.lock:
            claims = self.packing.membership(key, lambda: origin_claims | pricing._visit_claims(
                cell, 0, start_step, self.view.offsets))
        if self._root_claims is not None:
            self._root_claims[local_key] = claims
        return claims
