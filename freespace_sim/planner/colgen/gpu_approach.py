"""Bound complete roots by exact final-approach/arrival unions and relaxed reachability."""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np

from ..hexgrid import hex_distance
from . import pricing
from .gpu_bounds import CompletionBounds, _down
from .network import RowKey

_MAX_WORK = 1_000_000


class ApproachBounds:
    """Prepare optional whole-root bounds without enumerating route histories.

    Attributes
    ------------
    - bounds (CompletionBounds): original snapshot, exact root claims and canonical geometry.
    - work (int): charged adjacency, membership and price operations in this invocation.
    - max_work (int): hard preparation budget; exhaustion yields an unavailable bound.
    - pairs (list[tuple[int, int]]): all prepared incoming edges to any destination.
    """

    def __init__(
        self, bounds: CompletionBounds, check: Callable[[], None], max_work: int = _MAX_WORK,
    ) -> None:
        """Collect a superset of legal final edges under a bounded preparation budget.

        Parameters
        ------------
        - bounds (CompletionBounds): immutable objective, topology and price snapshot.
        - check (Callable): cooperative deadline check; interruptions propagate to pricing.
        - max_work (int): maximum adjacency/membership/price work for this flight.

        Return
        --------
        - output (None): complete incoming-edge set, or an unavailable optional helper.
        """
        self.bounds, self.check, self.max_work = bounds, check, max_work
        self.work, self.available = 0, False
        self.pairs: list[tuple[int, int]] = []
        t = bounds.topology
        self.cells = tuple(zip(map(int, t.cell_q), map(int, t.cell_r)))
        self.destinations = frozenset(map(int, np.flatnonzero(t.dest_mask)))
        for source in range(t.n_cells):
            targets = t.arc_target[int(t.arc_start[source]):int(t.arc_start[source + 1])]
            if not self.spend(1 + len(targets)):
                return
            # All roles are included: extra final-edge choices only weaken the bound.
            self.pairs.extend((source, int(target)) for target in targets
                              if int(target) in self.destinations)
        self.available = True

    def spend(self, amount: int) -> bool:
        """Check interruption and refuse optional work before exceeding its budget."""
        self.check()
        if amount > self.max_work - self.work:
            return False
        self.work += amount
        return True

    def upper(self, variant: int, benefit: float, negative: float, cutoff: float) -> float:
        """Enclose every canonical completion of one root, or return infinity on uncertainty.

        Parameters
        ------------
        - variant (int): exact prepared departure/lane root.
        - benefit (float): original flight benefit.
        - negative (float): original downward credit for every negative snapshot price.
        - cutoff (float): validated incumbent plus closure tolerance; permits safe early deferral.

        Return
        --------
        - output (float): maximum upper over all arrival/final-edge alternatives, only after
          their complete enumeration; infinity means the optional bound is unavailable.
        """
        if not self.available or not all(map(math.isfinite, (benefit, negative, cutoff))):
            return math.inf
        b = self.bounds
        p, t, v = b.problem, b.topology, b.variants
        root, start = int(v.cell[variant]), int(v.start_step[variant])
        first = start + int(t.rev_remaining[root])
        last = min(t.max_step, start + t.air_hop_limit)
        origin = b.origin_claims(variant)
        if not self.spend(len(origin) + len(self.pairs)):
            return math.inf
        distances = [hex_distance(self.cells[root], self.cells[source]) + 1
                     for source, _ in self.pairs]
        upper = -math.inf
        for arrival in range(first, last + 1):
            if not self.spend(1 + len(self.pairs)):
                return math.inf
            hops = arrival - start
            cost = b.cost_lower(variant, hops)
            endpoint = pricing._endpoint_claims(
                p.graph, p.cfg, origin=False, step=arrival, timing_steps=0,
            )
            choices = [(source, target) for (source, target), distance in
                       zip(self.pairs, distances, strict=True) if distance <= hops]
            if hops == 0 and root in self.destinations:
                choices.append((-1, root))
            for source, target in choices:
                final = pricing._visit_claims(self.cells[target], 0, arrival, b.view.offsets)
                previous = (pricing._visit_claims(self.cells[source], 0, arrival - 1, b.view.offsets)
                            if source >= 0 else frozenset())
                if not self.spend(len(origin) + len(endpoint) + len(final) + len(previous)):
                    return math.inf
                union: frozenset[RowKey] = origin | endpoint | final | previous
                positive = 0.0
                for row in sorted(union):
                    price = b.view.row_cost(row)
                    if price > 0.0:
                        positive = _down(positive + price)
                lower = _down(_down(cost + positive) + negative)
                candidate = b._roundoff_upper(lower, benefit, p.pi_f)
                if not math.isfinite(candidate) or candidate > cutoff:
                    return math.inf
                upper = max(upper, candidate)
        # An empty or unfinished relaxation never asserts root infeasibility.
        return upper if math.isfinite(upper) else math.inf
