"""Prepared forward topology and waypoint buckets for exact asynchronous pricing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time

import numpy as np

from .zeta import Cell, ZetaGraph


class ForwardLimit(Exception):
    """Interrupt a batch while its parent's floor still covers omitted work.

    Attributes
    ------------
    - args (tuple[str]): the cap or deadline termination reason.
    """


@dataclass(frozen=True, slots=True)
class ForwardTopology:
    """Immutable variable-degree CSR topology, validated against the graph authority.

    Attributes
    ------------
    - cells (tuple[Cell, ...]): stable waypoint IDs shared with the label arena.
    - offsets (np.ndarray): CSR row offsets into targets.
    - targets (np.ndarray): successor waypoint IDs in authoritative neighbor order.
    - order (tuple[int, ...]): topological waypoint order.
    - ranks (tuple[int, ...]): inverse permutation used for bucket placement.
    - reaching (dict[int, np.ndarray]): immutable goal reachability masks, cached by goal ID.
    """

    cells: tuple[Cell, ...]
    offsets: np.ndarray
    targets: np.ndarray
    order: tuple[int, ...]
    ranks: tuple[int, ...]
    reaching: dict[int, np.ndarray] = field(default_factory=dict, compare=False)

    def reaches(self, goal: int) -> np.ndarray:
        """Keep precisely the waypoints with a directed topological path to a goal.

        Parameters
        ------------
        - goal (int): destination waypoint ID, independent of timing or endpoint role.

        Return
        --------
        - output (np.ndarray): immutable necessary-connectivity mask; physical constraints
          can remove additional paths but cannot add a path absent from this mask.
        """
        if goal not in self.reaching:
            mask = np.zeros(len(self.cells), dtype=np.bool_)
            mask[goal] = True
            for node in reversed(self.order):
                if not mask[node]:
                    mask[node] = any(mask[target] for target in self.targets[
                        self.offsets[node]:self.offsets[node+1]])
            mask.flags.writeable = False
            self.reaching[goal] = mask
        return self.reaching[goal]

    @classmethod
    def build(cls, graph: ZetaGraph, deadline: float | None = None) -> ForwardTopology:
        """Prepare and cache a DAG without recomputing the forward geometry filter.

        Parameters
        ------------
        - graph (ZetaGraph): authoritative strictly forward twelve-link graph.
        - deadline (float | None): monotonic preparation cutoff.

        Return
        --------
        - output (ForwardTopology): validated immutable adjacency and ordering.
        """
        if not graph.options.forward_cruise or graph.options.connection_mode != "hex12":
            raise ValueError("forward topology requires forward_cruise and hex12")
        cached = getattr(graph, "_forward_topology", None)
        if cached is not None:
            return cached
        cells = tuple(sorted(graph.cells | {c for _i, c, _s in graph.origins}
                             | {c for _i, c in graph.destinations}))
        ids = {cell: i for i, cell in enumerate(cells)}
        offsets, targets = [0], []
        degree = [0] * len(cells)
        for cell in cells:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("forward topology preparation deadline")
            for target in graph.neighbors(cell):
                index = ids[target]
                targets.append(index)
                degree[index] += 1
            offsets.append(len(targets))
        ready = deque(i for i, count in enumerate(degree) if count == 0)
        order = []
        while ready:
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("forward topology preparation deadline")
            node = ready.popleft()
            order.append(node)
            for target in targets[offsets[node]:offsets[node+1]]:
                degree[target] -= 1
                if degree[target] == 0:
                    ready.append(target)
        if len(order) != len(cells):
            raise ValueError("forward cruise graph contains a cycle")
        ranks = [0] * len(cells)
        for rank, node in enumerate(order):
            ranks[node] = rank
        if any(ranks[node] >= ranks[target] for node in range(len(cells))
               for target in targets[offsets[node]:offsets[node+1]]):
            raise ValueError("invalid forward topological order")
        arrays = [np.asarray(values, dtype=np.int64) for values in (offsets, targets)]
        for array in arrays:
            array.flags.writeable = False
        result = cls(cells, *arrays, tuple(order), tuple(ranks))
        graph._forward_topology = result
        return result


class ForwardFrontier:
    """Process every history at one waypoint before visiting any successor waypoint.

    Attributes
    ------------
    - buckets (list): insertion-ordered label/floor pairs indexed by topological rank.
    - minima (list[float]): conservative pending floors, refreshed between waypoints.
    - cursor (int): current waypoint rank; completed buckets never accept new states.
    - floor (float): lower bound covering every unprocessed state in every bucket.
    """

    def __init__(self, topology: ForwardTopology) -> None:
        """Allocate empty per-waypoint state lists independently of arrival clocks."""
        self.ranks = topology.ranks
        self.buckets: list[list[tuple[float, int]]] = [[] for _ in topology.order]
        self.minima = [math.inf] * len(topology.order)
        self.cursor = self.index = 0
        self.floor = math.inf
        self.pending = self.peak_pending = self.visited = 0

    def add(self, node: int, label: int, floor: float) -> None:
        """Retain a label and its certificate floor without any cost-priority ordering."""
        rank = self.ranks[node]
        if rank < self.cursor:
            raise AssertionError("a successor reached an already processed waypoint")
        self.buckets[rank].append((floor, label))
        self.minima[rank] = min(self.minima[rank], floor)
        self.floor = min(self.floor, floor)
        self.pending += 1
        self.peak_pending = max(self.peak_pending, self.pending)

    def peek(self) -> tuple[float, int] | None:
        """Advance past completed waypoints and return the next insertion-ordered state."""
        moved = False
        while self.cursor < len(self.buckets):
            bucket = self.buckets[self.cursor]
            if self.index < len(bucket):
                if moved:
                    self.floor = min(self.minima[self.cursor:], default=math.inf)
                return bucket[self.index]
            self.visited += bool(bucket)
            bucket.clear()
            self.minima[self.cursor] = math.inf
            self.cursor += 1
            self.index = 0
            moved = True
        self.floor = math.inf
        return None

    def pop(self) -> tuple[float, int]:
        """Consume a state; keep its floor conservatively until this waypoint is complete."""
        entry = self.buckets[self.cursor][self.index]
        self.index += 1
        self.pending -= 1
        return entry
