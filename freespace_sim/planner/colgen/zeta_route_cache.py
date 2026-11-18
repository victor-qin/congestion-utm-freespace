"""Reuse complete reservation coverage without retaining stale dual prices."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import math

from .network import RowKey


class RouteCoverageCache:
    """Bounded row-to-route coverage shared across changes in the priced-row set.

    Complete coverage stays in the graph's canonical columns. This cache holds only
    references to those immutable unions and their row positions, never prices.
    """

    def __init__(self, max_routes: int = 256) -> None:
        """Index at most max_routes complete reservation unions in access order."""
        if max_routes < 0:
            raise ValueError("max_routes must be nonnegative")
        self.max_routes = max_routes
        self._coverage: OrderedDict[frozenset[RowKey], tuple[int, frozenset[RowKey], tuple[RowKey, ...]]] = OrderedDict()
        self._rows: dict[RowKey, dict[int, int]] = {}
        self._next_id = 0

    def bind(self, prices: Mapping[RowKey, float]) -> RoutePrices:
        """Bind current prices without invalidating price-independent route coverage.

        Parameters
        ------------
        - prices (Mapping[RowKey, float]): finite frozen prices owned by this pricing call.

        Return
        --------
        - output (RoutePrices): per-call price evaluator and hit/miss counters. Current
          prices are projected lazily, including reservations unpriced in earlier calls.
        """
        return RoutePrices(prices, self)

    def _register(self, claims: frozenset[RowKey]) -> tuple[int, frozenset[RowKey], tuple[RowKey, ...]]:
        """Index a new complete union and remove all postings of an evicted route.

        Parameters
        ------------
        - claims (frozenset[RowKey]): immutable canonical union absent from the index.

        Return
        --------
        - output (tuple): never-reused route ID, canonical union and its original claim
          order. The caller must bypass registration when max_routes is zero.
        """
        if len(self._coverage) >= self.max_routes:
            _claims, (expired_id, _original, expired_rows) = self._coverage.popitem(last=False)
            for row in expired_rows:
                postings = self._rows[row]
                del postings[expired_id]
                if not postings:
                    del self._rows[row]
        identifier = self._next_id
        self._next_id += 1
        ordered = tuple(claims)
        for position, row in enumerate(ordered):
            self._rows.setdefault(row, {})[identifier] = position
        entry = identifier, claims, ordered
        self._coverage[claims] = entry
        return entry


class RoutePrices:
    """One frozen dual vector with a lazy snapshot of indexed covered-row prices."""

    def __init__(
        self, prices: Mapping[RowKey, float], coverage: RouteCoverageCache,
    ) -> None:
        """Retain this call's frozen prices without storing them in the shared cache."""
        self._prices = prices
        self._coverage = coverage
        self._terms: dict[int, list[tuple[int, float]]] | None = None
        self.hits = self.misses = 0

    def _project(self) -> dict[int, list[tuple[int, float]]]:
        """Collect current nonzero prices once for the routes indexed at this instant.

        Parameters
        ------------
        - None: uses this call's frozen prices and the shared complete-coverage index.

        Return
        --------
        - output (dict[int, list[tuple[int, float]]]): original position and price terms
          for each indexed route. No route is summed until explicitly requested.
        """
        terms: dict[int, list[tuple[int, float]]] = {
            identifier: [] for identifier, _claims, _rows in self._coverage._coverage.values()
        }
        rows = self._coverage._rows
        if len(self._prices) < len(rows):
            for row, value in self._prices.items():
                if value != 0.0:
                    for identifier, position in rows.get(row, {}).items():
                        terms[identifier].append((position, value))
        else:
            for row, postings in rows.items():
                value = self._prices.get(row, 0.0)
                if value != 0.0:
                    for identifier, position in postings.items():
                        terms[identifier].append((position, value))
        return terms

    def price(self, claims: frozenset[RowKey]) -> float:
        """Sum current duals once per covered reservation, including signed prices.

        Parameters
        ------------
        - claims (frozenset[RowKey]): complete canonical union, including unpriced rows.

        Return
        --------
        - output (float): accurately summed price of the union under this call's duals.
          Route cost, forbidden-row checks and pricing certification remain independent.
        """
        cache = self._coverage
        if not cache.max_routes:
            self.misses += 1
            return math.fsum(self._prices.get(row, 0.0) for row in claims)
        if self._terms is None:
            self._terms = self._project()
        entry = cache._coverage.get(claims)
        if entry is None:
            self.misses += 1
            _identifier, _original, ordered = cache._register(claims)
            return math.fsum(self._prices.get(row, 0.0) for row in ordered)
        self.hits += 1
        cache._coverage.move_to_end(claims)
        identifier, original, ordered = entry
        if claims is not original:
            # Equal frozensets can have different iteration orders. Preserve the
            # requested canonical set's fsum behavior even for extreme prices.
            return math.fsum(self._prices.get(row, 0.0) for row in claims)
        covered = self._terms.get(identifier)
        if covered is None:
            # Another view may register a route after this call's projection. IDs
            # are never reused, so an old projection cannot alias a replacement.
            return math.fsum(self._prices.get(row, 0.0) for row in ordered)
        # Preserve fsum's term order, including intermediate-overflow behavior.
        covered.sort()
        return math.fsum(value for _position, value in covered)
