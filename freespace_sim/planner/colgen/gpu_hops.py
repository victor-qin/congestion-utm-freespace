"""Budgeted remaining-hop refinements of the rooted future-price relaxation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np

from .dp_prepare import PreparedTopology, PreparedVariants
from .gpu_future import _add_lower
from .pricing_platform import PricingProblem


@dataclass(frozen=True, slots=True)
class _Structure:
    """Unpriced reachable slack states and transitions for one topology/root domain."""

    topology: PreparedTopology
    roots: tuple[int, ...]
    limit: int
    bins: int
    offsets: np.ndarray
    cells: np.ndarray
    extra: np.ndarray
    following: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]


def _structure(
    problem: PricingProblem, topology: PreparedTopology, variants: PreparedVariants,
    graph: np.ndarray, limit: int, check: Callable[[], None],
) -> _Structure:
    """Cache ragged slack states reachable from some supplied root within the hop ceiling.

    Parameters
    ------------
    - problem (PricingProblem): owns an answer-neutral unpriced cache.
    - topology (PreparedTopology): any-role reverse distances and permitted arcs.
    - variants (PreparedVariants): supplies every current root cell.
    - graph (np.ndarray): six regular-role targets per cell, with absent targets -1.
    - limit (int): maximum number of states permitted by the numeric budget.
    - check (Callable): cooperative deadline check.

    Return
    --------
    - output (_Structure): immutable state layout; no prices or root times are cached.
    """
    roots = tuple(sorted(set(map(int, variants.cell))))
    cache = problem.graph._search_cache
    with cache.lock:
        cached = cache.gpu_hops
    if (cached is not None and cached.topology is topology
            and cached.roots == roots and cached.limit == limit):
        return cached
    distance = np.full(topology.n_cells, topology.air_hop_limit + 1, np.int64)
    queue = deque(roots)
    distance[list(roots)] = 0
    while queue:
        check()
        cell = queue.popleft()
        start, end = map(int, topology.arc_start[cell:cell + 2])
        for target in topology.arc_target[start:end]:
            target = int(target)
            if distance[cell] + 1 < distance[target]:
                distance[target] = distance[cell] + 1
                queue.append(target)
    maximum = np.maximum(
        0, topology.air_hop_limit - distance - topology.rev_remaining.astype(np.int64) + 1,
    )
    low, high = 0, int(maximum.max(initial=0))
    while low < high:
        check()
        middle = (low + high + 1) // 2
        if int(np.minimum(maximum, middle).sum()) <= limit:
            low = middle
        else:
            high = middle - 1
    bins = low
    counts = np.minimum(maximum, bins)
    offsets = np.empty(topology.n_cells + 1, np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    cells = np.repeat(np.arange(topology.n_cells, dtype=np.intp), counts)
    extra = np.arange(len(cells)) - np.repeat(offsets[:-1], counts)
    following = []
    for arc in range(6):
        check()
        target = graph[cells, arc]
        valid = np.flatnonzero(target >= 0)
        target = target[valid]
        slack = extra[valid] + topology.rev_remaining[cells[valid]] - 1 - topology.rev_remaining[target]
        keep = (slack >= 0) & (slack < counts[target])
        source, target = valid[keep], target[keep]
        next_index = offsets[target] + slack[keep]
        following.append((source, target, next_index))
    for array in (offsets, cells, extra, *(a for transition in following for a in transition)):
        array.flags.writeable = False
    result = _Structure(topology, roots, limit, bins, offsets, cells, extra, tuple(following))
    check()
    with cache.lock:
        cache.gpu_hops = result
    return result


class HopBounds:
    """Original-price FP64 future bounds indexed by time and remaining extra hops.

    With reverse distance d, remaining allowance b becomes k=b-d(cell). Each
    edge changes k to k+d(cell)-1-d(target), which cannot increase because d is
    a shortest distance on a superset of the allowed arcs. Thus low-k tables
    are closed under continuation. Missing higher-k tables provide no proof.
    Forward distances discard only states unreachable from every supplied root
    within the total hop ceiling. The regular and first-hop role relaxations,
    disjoint arrival charges and visit sums are inherited from FutureBounds.
    """

    def __init__(
        self, problem: PricingProblem, topology: PreparedTopology, variants: PreparedVariants,
        graph: np.ndarray, first_graph: np.ndarray, visits: np.ndarray, tails: np.ndarray,
        air: float, first_step: int, check: Callable[[], None], *,
        max_bytes: int = 64 * 1024 * 1024, max_work: int = 48_000_000,
        boundary: np.ndarray | None = None,
    ) -> None:
        """Publish only a complete budgeted refinement; absent states use the old bound.

        Parameters
        ------------
        - problem (PricingProblem): current flight and exclusions.
        - topology (PreparedTopology): hop ceiling and reverse distances.
        - variants (PreparedVariants): all root cells and start times.
        - graph (np.ndarray): regular-role relaxed transitions.
        - first_graph (np.ndarray): first-role relaxed transitions.
        - visits (np.ndarray): original positive visit-price lower sums by time/cell.
        - tails (np.ndarray): positive arrival charges proved disjoint from rooted visits.
        - air (float): nonnegative air-cost lower bound per move.
        - first_step (int): absolute time of row zero.
        - check (Callable): aborts preparation on deadline, before publication.
        - max_bytes (int): extra FP64 table/root values and row scratch budget.
        - max_work (int): conservative count of state/arc/time evaluations.
        - boundary (np.ndarray | None): old regular-cost row at a truncated final time;
          None means the actual horizon. Missing later hop states use this lower bound.

        Return
        --------
        - output (None): complete optional costs and counters; exhausted budgets mean a miss.
        """
        started = time.monotonic()
        self.topology, self.first_step = topology, first_step
        self.costs: np.ndarray | None = None
        self.structure: _Structure | None = None
        self.roots: dict[tuple[int, int], float] = {}
        self.stats: dict[str, int | float] = dict(
            hop_bins=0, hop_states=0, hop_table_bytes=0, hop_work=0, hop_build_s=0.0,
            hop_layers=len(visits), hop_boundary_relaxed=int(boundary is not None),
        )
        if not len(variants.cell) or not max_bytes or not max_work:
            return
        check()
        times, cells = visits.shape
        root_count = len(set(zip(map(int, variants.cell), map(int, variants.start_step))))
        fixed_bytes, fixed_work = root_count * 8 + cells * 32, root_count * 7
        limit = min(max(0, (max_bytes - fixed_bytes) // (8 * times + 8 * 8)),
                    max(0, (max_work - fixed_work) // (times * 7)))
        if limit <= 0:
            return
        structure = _structure(problem, topology, variants, graph, limit, check)
        n = len(structure.cells)
        if not n:
            return
        costs = np.full((times, n), np.inf, np.float64)
        destination = topology.dest_mask[structure.cells].astype(bool)
        if boundary is not None:
            costs[-1] = boundary[structure.cells]
        roots_by_time: dict[int, list[int]] = {}
        for cell, step in set(zip(map(int, variants.cell), map(int, variants.start_step))):
            roots_by_time.setdefault(step - first_step, []).append(cell)
        root_costs = {}
        for offset in range(times - 1, -1, -1):
            check()
            current = costs[offset]
            current[destination] = tails[offset]
            if offset + 1 < times:
                following = costs[offset + 1]
                move = _add_lower(visits[offset + 1], air)
                for source, target, next_index in structure.following:
                    current[source] = np.minimum(
                        current[source], _add_lower(move[target], following[next_index]),
                    )
            for cell in roots_by_time.get(offset, ()):
                if boundary is not None and offset == times - 1:
                    continue
                available = topology.air_hop_limit
                best = float(tails[offset]) if topology.dest_mask[cell] else np.inf
                complete = True
                if offset + 1 < times and available:
                    for target in first_graph[cell, :6]:
                        if target < 0:
                            continue
                        slack = available - 1 - int(topology.rev_remaining[target])
                        if slack < 0:
                            continue
                        index = int(structure.offsets[target]) + slack
                        if index >= structure.offsets[target + 1]:
                            complete = False
                            break
                        best = min(best, float(_add_lower(
                            move[target:target + 1], following[index:index + 1],
                        )[0]))
                if complete:
                    root_costs[cell, first_step + offset] = best
        check()
        costs.flags.writeable = False
        self.costs, self.structure, self.roots = costs, structure, root_costs
        self.stats.update(hop_bins=structure.bins, hop_states=n, hop_table_bytes=costs.nbytes,
                          hop_numeric_budget_bytes=fixed_bytes + n * (times * 8 + 64),
                          hop_structure_bytes=sum(a.nbytes for a in (
                              structure.offsets, structure.cells, structure.extra,
                              *(a for transition in structure.following for a in transition),
                          )),
                          hop_work=fixed_work + n * times * 7,
                          hop_build_s=time.monotonic() - started)

    def lookup(self, cell: int, step: int, available: int) -> float | None:
        """Return a complete remaining-hop lower bound, or None for an unbuilt state.

        Parameters
        ------------
        - cell (int): local topology cell.
        - step (int): absolute current visit time within the parent future table.
        - available (int): remaining legal moves, including the final hop.

        Return
        --------
        - output (float | None): valid cost lower bound; only a proved empty domain is infinity.
        """
        if not 0 <= cell < self.topology.n_cells:
            raise IndexError("hop-bound cell is outside the prepared topology")
        if not self.first_step <= step <= self.topology.max_step:
            raise IndexError("hop-bound step is outside the prepared time horizon")
        slack = available - int(self.topology.rev_remaining[cell])
        if slack < 0:
            return float('inf')
        if self.structure is None:
            return None
        if step - self.first_step >= len(self.costs):
            return None
        index = int(self.structure.offsets[cell]) + slack
        if index >= self.structure.offsets[cell + 1]:
            return None
        return float(self.costs[step - self.first_step, index])
