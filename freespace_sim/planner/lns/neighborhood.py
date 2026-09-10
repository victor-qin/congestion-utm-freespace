"""Destroy heuristics for MAPF-LNS (Li et al., IJCAI-21, Algorithms 1-2) on the hex lattice.

Three neighborhood generators over a committed schedule, plus the ALNS roulette
selector. All of them are pure reads against a :class:`DestroyContext`; the solver
owns the mutable state and implements the protocol.

World adaptations (rationale in context/lns_plan.md):

* ``delay(p_i)`` is the flight's weighted premium over its own unimpeded plan
  (the 1 s ground : 3 s air currency), not a hop count.
* The paper samples the walk start anywhere on the path, including pre-wait
  timesteps. Our ground hold is the dominant delay mode but is not part of the
  airborne visit sequence, so the walk start is sampled from the *virtual*
  timeline [unimpeded launch step, arrival step): starts during the hold sit at
  the launch cell and immediately explore "what if I had launched on time",
  collecting the owners that forced the hold.
* "Collides with" is resolved against committed claim rows (already W-expanded
  by the rasterizer), so no extra k-robust window is applied here.
* Map-based "intersection vertices" (grid degree >= 3) carry no signal on a hex
  lattice (uniform degree 6); the analog is a contention cell - one claimed by
  >= 2 distinct flights - which concentrates at hub lane mouths where the
  measured conflicts live.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

import numpy as np

from freespace_sim.planner.hexgrid import AXIAL_NEIGHBORS, hex_distance

Cell = tuple[int, int, int]  # (q, r, level)


class DestroyContext(Protocol):
    """Read view of the incumbent schedule that destroy heuristics operate on."""

    rng: np.random.Generator
    n_levels: int

    def movable_ids(self) -> Sequence[int]:
        """Ids of the flights a destroy heuristic is allowed to remove."""
        ...

    def is_movable(self, fid: int) -> bool:
        """Whether flight ``fid`` is one the destroy heuristics may remove."""
        ...

    def delay(self, fid: int) -> float:
        """Weighted premium of the incumbent plan over the unimpeded plan (>= 0)."""
        ...

    def visits(self, fid: int) -> Sequence[tuple[int, Cell]]:
        """Ordered airborne (step, cell) sequence of the incumbent plan."""
        ...

    def unimpeded_launch_step(self, fid: int) -> int:
        """Step at which the flight would launch with zero ground hold."""
        ...

    def owners_over(self, cell: Cell, s_lo: int, s_hi: int) -> Iterable[int]:
        """Flight ids whose committed claim rows on ``cell`` overlap [s_lo, s_hi]."""
        ...

    def claim_span(self, cell: Cell) -> tuple[int, int]:
        """(min, max) claimed step on ``cell``; only called for contended cells."""
        ...

    def contention_cells(self) -> Sequence[Cell]:
        """Cells claimed by >= 2 distinct movable flights."""
        ...


def _neighbor_cells(cell: Cell, n_levels: int, include_stay: bool) -> list[Cell]:
    """The cells one step from ``cell``: its six in-plane hex neighbours plus one flight level
    down and/or up (clamped to ``[0, n_levels)``), and ``cell`` itself when ``include_stay``.
    This is the per-step move set the destroy walk and the map BFS both expand over.

    Parameters
    ------------
    - cell (Cell): the ``(q, r, level)`` cell to expand from.
    - n_levels (int): number of flight levels; caps the upward level move at ``n_levels - 1``.
    - include_stay (bool): when True, append ``cell`` itself so the walk may hold in place.

    Return
    --------
    - output (list[Cell]): the reachable one-step cells (6 to 9 entries depending on level
      clamping and ``include_stay``).
    """
    q, r, level = cell
    out = [(q + dq, r + dr, level) for dq, dr in AXIAL_NEIGHBORS]
    if level > 0:
        out.append((q, r, level - 1))
    if level < n_levels - 1:
        out.append((q, r, level + 1))
    if include_stay:
        out.append(cell)
    return out


def _steps_to(cell: Cell, goal: Cell) -> int:
    """Minimum steps from ``cell`` to ``goal``: in-plane hex distance plus level difference."""
    return hex_distance((cell[0], cell[1]), (goal[0], goal[1])) + abs(cell[2] - goal[2])


def _select_most_delayed(ctx: DestroyContext, tabu: set[int]) -> int | None:
    """Algorithm 1 lines 1-4 with one deviation: after a tabu reset we re-select, so a delay-0
    pick is only returned when every movable flight has zero delay (the schedule is
    unimpeded-optimal and the caller can stop).

    Parameters
    ------------
    - ctx (DestroyContext): read view of the incumbent schedule; queried for movable ids and
      delays.
    - tabu (set[int]): seeds already tried this round; mutated in place — the pick is added, and
      the set is cleared when every movable flight is tabu or the best non-tabu pick has zero
      delay.

    Return
    --------
    - output (int | None): the chosen most-delayed movable flight id, or ``None`` when no flight
      is movable.
    """
    movable = list(ctx.movable_ids())
    if not movable:
        return None
    candidates = [f for f in movable if f not in tabu]
    if not candidates:
        tabu.clear()
        candidates = movable
    best = max(candidates, key=lambda f: (ctx.delay(f), -f))
    if ctx.delay(best) <= 0.0 and tabu:
        tabu.clear()
        best = max(movable, key=lambda f: (ctx.delay(f), -f))
    tabu.add(best)
    return best


def _random_walk(ctx: DestroyContext, fid: int, collected: set[int], n_target: int) -> None:
    """Algorithm 1 RANDOMWALK: from a step sampled on flight ``fid``'s virtual timeline, take
    moves that could still beat its incumbent arrival, adding to ``collected`` the movable owners
    of the claims each move runs into.

    Parameters
    ------------
    - ctx (DestroyContext): read view of the incumbent schedule; supplies visits, launch step,
      the RNG, claim owners, and the movable test.
    - fid (int): flight whose plan is walked.
    - collected (set[int]): destroy set accumulated so far; mutated in place with the owners hit.
    - n_target (int): stop once ``collected`` reaches this size.

    Return
    --------
    - output (None): mutates ``collected``; returns early without change when ``fid`` has fewer
      than two visits or no move can still beat the arrival.
    """
    vis = ctx.visits(fid)
    if len(vis) < 2:
        return
    arrival_step, goal = vis[-1]
    launch_step = vis[0][0]
    t0 = int(ctx.rng.integers(min(ctx.unimpeded_launch_step(fid), launch_step), arrival_step))
    if t0 < launch_step:
        t, x = t0, vis[0][1]
    else:
        steps = [s for s, _ in vis]
        t, x = vis[min(int(np.searchsorted(steps, t0)), len(vis) - 2)]
    while len(collected) < n_target:
        candidates = [
            v
            for v in _neighbor_cells(x, ctx.n_levels, include_stay=True)
            if t + 1 + _steps_to(v, goal) < arrival_step
        ]
        if not candidates:
            return
        y = candidates[int(ctx.rng.integers(len(candidates)))]
        for owner in ctx.owners_over(y, t + 1, t + 1):
            if owner != fid and ctx.is_movable(owner):
                collected.add(owner)
        x, t = y, t + 1


def agent_based_neighborhood(
    ctx: DestroyContext, n: int, tabu: set[int], max_walks: int = 10,
    seed_fid: int | None = None,
) -> set[int]:
    """Algorithm 1: seed with the most-delayed non-tabu flight, then random-walk from members of
    the growing set until ``n`` flights are collected (or ``max_walks`` walks come up dry).
    See context/figures/lns_destroy_operators.png (left).

    ``tabu`` is a serial recurrence — ``_select_most_delayed`` mutates it — so m workers each
    holding their own copy would every one pick the same most-delayed flight, and the pool would
    run m copies of a single neighborhood while looking healthy. A parallel coordinator instead
    advances one tabu across its slots and passes the choice down via ``seed_fid``, leaving
    ``_select_most_delayed`` the sole owner of the selection and its reset semantics.

    Parameters
    ------------
    - ctx (DestroyContext): read view of the incumbent schedule.
    - n (int): target neighborhood size to collect.
    - tabu (set[int]): most-delayed seeds already tried; mutated in place (reset when exhausted).
    - max_walks (int): random-walk restarts before giving up below size ``n``.
    - seed_fid (int | None): explicit seed flight (parallel coordinator); ``None`` picks the
      most-delayed non-tabu flight locally.

    Return
    --------
    - output (set[int]): flight ids to destroy (just the seed when every walk comes up dry).
    """
    seed = _select_most_delayed(ctx, tabu) if seed_fid is None else seed_fid
    if seed is None:
        return set()
    neighborhood = {seed}
    walker = seed
    for _ in range(max_walks):
        if len(neighborhood) >= n:
            break
        _random_walk(ctx, walker, neighborhood, n)
        members = sorted(neighborhood)
        walker = members[int(ctx.rng.integers(len(members)))]
    return neighborhood


def _collect_intersection_agents(ctx: DestroyContext, cell: Cell, n: int, out: set[int]) -> None:
    """Algorithm 2 GETINTERSECTIONAGENTS: from a random claimed step on ``cell``, spread outward
    in time, adding to ``out`` the movable flights that claim the cell at each step reached.

    Parameters
    ------------
    - ctx (DestroyContext): read view of the incumbent schedule; supplies the claim span, owners,
      the movable test, and the RNG.
    - cell (Cell): the contended cell whose claimants are collected.
    - n (int): stop once ``out`` reaches this size.
    - out (set[int]): destroy set accumulated so far; mutated in place with the claimants found.

    Return
    --------
    - output (None): mutates ``out`` in place.
    """
    s_lo, s_hi = ctx.claim_span(cell)
    t = int(ctx.rng.integers(s_lo, s_hi + 1))
    delta = 0
    while len(out) < n and delta <= max(t - s_lo, s_hi - t):
        for s in {t + delta, t - delta}:
            for owner in ctx.owners_over(cell, s, s):
                if ctx.is_movable(owner):
                    out.add(owner)
        delta += 1


def map_based_neighborhood(ctx: DestroyContext, n: int, max_cells: int = 4096) -> set[int]:
    """Algorithm 2: BFS outward over the lattice from a random contention cell, collecting the
    claimants of every contention cell it reaches.
    See context/figures/lns_destroy_operators.png (right).

    ``max_cells`` bounds the BFS: the paper explores the whole map, but ours is ~144k cells per
    level, so the walk is capped rather than run to exhaustion.

    Parameters
    ------------
    - ctx (DestroyContext): read view of the incumbent schedule.
    - n (int): target neighborhood size; the BFS stops once this many flights are collected.
    - max_cells (int): BFS exploration bound (cells popped) before giving up.

    Return
    --------
    - output (set[int]): flight ids to destroy (empty when no cell is contended).
    """
    contended = ctx.contention_cells()
    if not contended:
        return set()
    contended_set = set(contended)
    start = contended[int(ctx.rng.integers(len(contended)))]
    neighborhood: set[int] = set()
    queue: deque[Cell] = deque([start])
    seen = {start}
    popped = 0
    while queue and len(neighborhood) < n and popped < max_cells:
        cell = queue.popleft()
        popped += 1
        if cell in contended_set:
            _collect_intersection_agents(ctx, cell, n, neighborhood)
        for nxt in _neighbor_cells(cell, ctx.n_levels, include_stay=False):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return neighborhood


def random_neighborhood(ctx: DestroyContext, n: int) -> set[int]:
    """Uniform random subset of the movable flights (Section 5.3).

    Parameters
    ------------
    - ctx (DestroyContext): read view of the incumbent schedule.
    - n (int): target neighborhood size; capped at the number of movable flights.

    Return
    --------
    - output (set[int]): flight ids to destroy (empty when no flight is movable).
    """
    ids = np.asarray(list(ctx.movable_ids()), dtype=np.int64)
    if ids.size == 0:
        return set()
    picked = ctx.rng.choice(ids, size=min(n, ids.size), replace=False)
    return {int(f) for f in picked}


@dataclass
class AdaptiveSelector:
    """ALNS roulette wheel over destroy heuristics (Section 6).

    ``update`` applies w_i <- gamma * max(improvement, 0) + (1 - gamma) * w_i to
    the heuristic used this iteration only; a failed or non-improving iteration
    decays it. Improvements are in weighted seconds, so the first success can
    dwarf the initial weight of 1 - that matches the paper and self-balances as
    every heuristic accumulates rewards.
    """

    names: tuple[str, ...]
    gamma: float = 0.01
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Seed each heuristic's weight to the paper's uniform prior of 1.0.

        Parameters
        ------------
        - none: reads ``names`` and back-fills ``weights``.

        Return
        --------
        - output (None): mutates ``weights`` in place, adding any missing name at weight 1.0.
        """
        for name in self.names:
            self.weights.setdefault(name, 1.0)

    def pick(self, rng: np.random.Generator) -> str:
        """Sample one heuristic name with probability proportional to its weight.

        Falls back to a uniform draw when every weight has decayed to zero, so a pick is always
        returned.

        Parameters
        ------------
        - rng (np.random.Generator): source of the weighted (or uniform-fallback) draw.

        Return
        --------
        - output (str): the chosen heuristic name from ``names``.
        """
        w = np.asarray([self.weights[name] for name in self.names], dtype=float)
        total = float(w.sum())
        if total <= 0.0:
            return self.names[int(rng.integers(len(self.names)))]
        return self.names[int(rng.choice(len(self.names), p=w / total))]

    def update(self, name: str, improvement: float) -> None:
        """Apply this iteration's ALNS reward (formula in the class docstring) to ``name``'s weight
        from ``improvement`` (weighted seconds); a miss or non-improving move decays it."""
        self.weights[name] = self.gamma * max(improvement, 0.0) + (1.0 - self.gamma) * self.weights[name]
