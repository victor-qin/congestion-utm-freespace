"""Cost-aware Safe Interval Path Planning (SIPP) on the hex lattice.

A drop-in for the space-time A* planner (:mod:`astar`): same cost model, terminal gating, and output
contract — it returns the **identical optimal weighted cost** — but collapses the per-timestep ``step``
axis into per-cell **safe intervals**, so the air search expands O(cell × interval) nodes instead of
O(cell × step). (Phillips & Likhachev, "SIPP: Safe Interval Path Planning for Dynamic Environments,"
ICRA 2011.)

Because our objective is **weighted cost** (``c_hold ≠ c_gd`` ⇒ earliest-arrival is not cheapest), the
classic single-best-per-state SIPP is unsound here; we keep a **Pareto frontier** of
``(arrival_time, cost)`` per ``(cell, interval)`` with dominance pruning, which recovers exact A*
optimality.

Design: subclass :class:`AStarPlanner` to inherit ``_occupancy`` (occupancy + ``TerminalCapacity``
sync) and ``_build`` (corner→volumes). States are keyed on the A*-shaped tuple
``("g"/"a", q, r, step)`` so ``_committed_arrival``/``_build``/reconstruction run verbatim. The air
reroute is the only lever collapsed into intervals; the ground-wait ray and a goal-cell hover stay
per-step because the terminal capacity gates are per-step (not interval-captured). See the plan file.
"""
from __future__ import annotations

import heapq
import itertools
import math

import numpy as np

from ..cost import trajectory_cost
from ..types import (
    DenialReason,
    IntentStatus,
    OperationalIntent,
    TimedPoint,
    as_terminal,
)
from ..volumes import terminal_radius
from . import hexgrid as hg
from .astar import AStarPlanner, _committed_arrival

_EPS = 1e-6


def _deny(req, reason):
    return OperationalIntent(
        request=req, status=IntentStatus.REJECTED, denial_reason=reason, planner="sipp"
    )


class _SafeIntervals:
    """Lazy, per-plan, per-cell safe-interval index over the shared occupancy service.

    A cell's safe intervals are the maximal runs of steps in ``[base, max_step]`` where
    ``svc.is_blocked(q, r, s, own)`` is False. ``is_blocked`` already composes universal-blocked +
    foreign-column + own-exemption + (issue #18) same-hub sibling corridors, so no terminal
    special-casing is needed here. Built on first touch and memoised (v1 = forward scan; a maintained
    cell→interval index on the commit hook is the noted follow-up)."""

    def __init__(self, svc, own, base, max_step):
        self.svc = svc
        self.own = own
        self.base = base
        self.max_step = max_step
        self._cache: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def intervals(self, q, r):
        iv = self._cache.get((q, r))
        if iv is None:
            iv = self._scan(q, r)
            self._cache[(q, r)] = iv
        return iv

    def _scan(self, q, r):
        out: list[tuple[int, int]] = []
        blk, own, mx = self.svc.is_blocked, self.own, self.max_step
        s = self.base
        while s <= mx:
            if blk(q, r, s, own):
                s += 1
                continue
            lo = s
            while s + 1 <= mx and not blk(q, r, s + 1, own):
                s += 1
            out.append((lo, s))
            s += 1
        return out

    def index_of(self, q, r, step):
        for i, (lo, hi) in enumerate(self.intervals(q, r)):
            if lo <= step <= hi:
                return i
        return -1   # step is blocked (no interval) — the search never targets such a step


def _nondominated(frontier, key, t, g, w):
    """Weighted-SIPP Pareto insert at ``key=(q, r, interval)`` on ``(arrival_time, cost)``.

    The *only* in-air wait is hover at rate ``w = c_air_hold_per_s``, so an EARLIER, cheaper label can
    reproduce a LATER one by hovering forward — but it pays for it. Hence the dominance is **not** plain
    ``(t2<=t and g2<=g)`` (which wrongly prunes a later arrival that was reached via cheap upfront ground
    delay, forcing expensive goal-hover instead — observed as ``(c_hold-c_gd)·dt`` cost gaps vs A*).
    Stored ``(t2,g2)`` dominates new ``(t,g)`` iff it is no later AND can hover to ``t`` for ``<= g``:
    ``t2 <= t and g2 + (t - t2)*w <= g``. Symmetric for eviction. Returns False ⇒ caller skips."""
    F = frontier.get(key)
    if F is None:
        frontier[key] = [(t, g)]
        return True
    for (t2, g2) in F:
        if t2 <= t and g2 + (t - t2) * w <= g + 1e-9:
            return False
    frontier[key] = [(t2, g2) for (t2, g2) in F if not (t <= t2 and g + (t2 - t) * w <= g2 + 1e-9)]
    frontier[key].append((t, g))
    return True


class SIPPPlanner(AStarPlanner):
    """Safe-interval cost-aware planner; inherits occupancy/terminal sync and corridor build from A*."""

    def plan(self, req, ledger, cfg):
        # ---- setup: identical to AStarPlanner.plan (so cost/terminals/output match exactly) ----
        dt = cfg.dt_s
        pitch = cfg.nominal_speed_mps * dt
        R = hg.circumradius(cfg)
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        t_depart = req.t_departure
        base = int(math.ceil(t_depart / dt))
        climb_steps = max(1, int(math.ceil(cfg.climb_time_s / dt)))
        climb_cost = cfg.cost_altitude_change_per_m * (cfg.cruise_level_m - cfg.ground_level_m)

        oq, orr = hg.enu_to_axial(origin[0], origin[1], R)
        gq, grr = hg.enu_to_axial(dest[0], dest[1], R)
        gx, gy = R * hg.SQRT3 * (gq + grr / 2.0), R * 1.5 * grr
        straight = float(np.linalg.norm(dest[:2] - origin[:2]))

        svc = self._occupancy(req, ledger, cfg)
        tcap = self._tcap
        dwell_steps = max(1, int(math.ceil((cfg.hover_time_s + cfg.climb_time_s) / dt)))

        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        own = frozenset(t.id for t in (o_term, d_term) if t is not None)
        o_cap = o_term.capacity if o_term else 1
        d_cap = d_term.capacity if d_term else 1

        fixed_lanes = cfg.fixed_exit_lanes
        o_lanes = hg.terminal_lanes(origin, o_term, cfg) if fixed_lanes and o_term is not None else []
        d_lanes = hg.terminal_lanes(dest, d_term, cfg) if fixed_lanes and d_term is not None else []
        d_lane_by_cell = {L.cell: L for L in d_lanes}
        h_off = max((L.dist for L in d_lanes), default=0.0)
        o_r = terminal_radius(o_term, cfg) if o_term is not None else 0.0

        sqrt3, c_lat = hg.SQRT3, cfg.cost_air_lateral_per_m

        def h_air(q, r):
            dx, dy = R * sqrt3 * (q + r / 2.0) - gx, R * 1.5 * r - gy
            return c_lat * max(0.0, math.sqrt(dx * dx + dy * dy) - h_off) + climb_cost

        dx0, dy0 = R * sqrt3 * (oq + orr / 2.0) - gx, R * 1.5 * orr - gy
        h_ground = c_lat * max(0.0, math.sqrt(dx0 * dx0 + dy0 * dy0) - h_off) + 2.0 * climb_cost

        n_hops = int(math.ceil(max(straight, pitch) / pitch))
        max_step = base + climb_steps + int(math.ceil(cfg.max_ground_delay_s / dt)) + 3 * n_hops + 6

        SI = _SafeIntervals(svc, own, base, max_step)
        came: dict = {}

        def is_goal_cell(q, r):
            if d_term is not None and fixed_lanes:
                return (q, r) in d_lane_by_cell
            return q == gq and r == grr

        def goal_ok(st):
            q, r, s = st[1], st[2], st[3]
            if d_term is not None and fixed_lanes:
                return (q, r) in d_lane_by_cell and tcap.dwell_ok(d_term, dest, s * dt, d_cap)
            if not (q == gq and r == grr):
                return False
            if d_term is not None:
                arr = _committed_arrival(st, came, R, dt, cfg, origin, dest, o_term, d_term)
                return tcap.dwell_ok(d_term, dest, arr, d_cap, origin)
            return svc.pad_clear(gq, grr, s, dwell_steps)

        # ---- cost-aware safe-interval search; AS = ("g"/"a", q, r, step), A*-shaped ----
        start = ("g", oq, orr, base)
        g = {start: 0.0}
        wait_steps: dict = {}
        frontier: dict = {}
        counter = itertools.count()
        pq = [(h_ground, next(counter), start, 0.0)]
        goal_state = None
        expansions = 0

        while pq:
            _, _, st, gst = heapq.heappop(pq)
            if gst > g.get(st, math.inf):
                continue                                   # stale (a cheaper label for this AS won)
            if st[0] == "a" and is_goal_cell(st[1], st[2]) and goal_ok(st):
                goal_state = st
                break                                      # first gate-passing pop (f-order) = optimal
            expansions += 1
            if expansions > self.max_expansions:
                break
            for nst, cost, w in self._succ(
                st, SI, cfg, pitch, climb_steps, climb_cost, dwell_steps, own, o_cap, o_term,
                origin, tcap, dest, o_lanes, o_r, fixed_lanes, max_step, is_goal_cell,
            ):
                ng = gst + cost
                # Pareto frontier applies only to ordinary AIR cruise cells. The ground state is a
                # separate per-step ray, and a GOAL cell carries a per-step landing gate
                # (dwell_ok/pad_clear) the frontier can't see — cross-step dominance there would prune a
                # later gate-passing arrival under an earlier gate-failing one. Both are exempt (A*'s
                # per-step states never dominate across steps), gated only by ``g``/lazy-skip.
                if nst[0] == "a" and not is_goal_cell(nst[1], nst[2]):
                    niv = SI.index_of(nst[1], nst[2], nst[3])
                    if niv >= 0 and not _nondominated(frontier, (nst[1], nst[2], niv),
                                                      nst[3] * dt, ng, cfg.cost_air_hold_per_s):
                        continue                           # dominated at its (cell, interval) → prune
                if ng < g.get(nst, math.inf):
                    g[nst] = ng
                    came[nst] = st
                    wait_steps[nst] = w
                    hh = h_air(nst[1], nst[2]) if nst[0] == "a" else h_ground
                    heapq.heappush(pq, (ng + hh, next(counter), nst, ng))

        if goal_state is None:
            return _deny(req, DenialReason.SEARCH_EXHAUSTED)

        # ---- reconstruct, re-expanding folded reroute waits so cruise_wps matches A*'s per-step list ----
        path = [goal_state]
        while path[-1] != start:
            path.append(came[path[-1]])
        path.reverse()
        expanded = []
        for i, cur in enumerate(path):
            expanded.append(cur)
            if i + 1 < len(path):
                w = wait_steps.get(path[i + 1], 0)         # hover steps spent IN cur before the move
                for k in range(1, w + 1):
                    expanded.append((cur[0], cur[1], cur[2], cur[3] + k))
        air = [s for s in expanded if s[0] == "a"]
        ground_steps = air[0][3] - climb_steps - base
        delay = ground_steps * dt

        cruise_wps: list[TimedPoint] = [
            (np.array([*hg.hex_center(q, r, R), cfg.cruise_level_m]), s * dt)
            for (_, q, r, s) in air
        ]
        volumes, centerline, cum_horiz, n_hover = self._build(
            cruise_wps, origin, dest, base, ground_steps, cfg,
            origin_term=req.origin_terminal, dest_term=req.dest_terminal,
        )
        if straight > _EPS and cum_horiz / straight > cfg.max_detour_factor:
            return _deny(req, DenialReason.BUDGET_EXCEEDED)
        if ledger.any_conflict(volumes):
            return _deny(req, DenialReason.CONFLICT_FILED)

        intent = OperationalIntent(
            request=req,
            status=IntentStatus.ACCEPTED,
            volumes=volumes,
            centerline=centerline,
            ground_delay_s=delay,
            air_hold_s=n_hover * dt,
            air_detour_m=max(0.0, cum_horiz - straight),
            altitude_change_m=2.0 * (cfg.cruise_level_m - cfg.ground_level_m),
            planner="sipp",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent

    def _succ(self, st, SI, cfg, pitch, climb_steps, climb_cost, dwell_steps, own, o_cap, o_term,
              origin, tcap, dest, o_lanes, o_r, fixed_lanes, max_step, is_goal_cell):
        """Successors as ``(AS, edge_cost, wait_steps)``. Ground-wait is a per-step ray and takeoff is
        emitted at the current ground step (so the per-step pad gates match A*); the air reroute is the
        SIPP collapse (one successor per reachable neighbour safe-interval, folding the pre-move hover
        into the cost at the air rate); a standalone hover is emitted only at a goal cell (to retry the
        per-step landing gate). Mirrors :meth:`AStarPlanner._edges`."""
        dt = cfg.dt_s
        c_gd, c_hold, c_lat = (cfg.cost_ground_delay_per_s, cfg.cost_air_hold_per_s,
                               cfg.cost_air_lateral_per_m)
        svc = self._svc
        tag, q, r, s = st
        out = []
        if tag == "g":
            if s + 1 <= max_step:
                out.append((("g", q, r, s + 1), c_gd * dt, 0))          # ground-wait ray (== A* g→g)
            ts = s + climb_steps
            if ts > max_step:
                return out
            if fixed_lanes and o_term is not None:
                if tcap.dwell_ok(o_term, origin, s * dt, o_cap):        # capacity + column at takeoff
                    for L in o_lanes:
                        lq, lr = L.cell
                        if not svc.is_blocked(lq, lr, ts, own):
                            out.append((("a", lq, lr, ts),
                                        climb_cost + c_lat * (L.dist - o_r), 0))   # one edge per lane
            else:
                pad_ok = (tcap.dwell_ok(o_term, origin, s * dt, o_cap, dest) if o_term is not None
                          else svc.pad_clear(q, r, s, dwell_steps))
                if not svc.is_blocked(q, r, ts, own) and pad_ok:
                    out.append((("a", q, r, ts), climb_cost, 0))        # takeoff (legacy/no-terminal)
            return out

        ivs = SI.intervals(q, r)
        iv = SI.index_of(q, r, s)
        hi_c = ivs[iv][1] if iv >= 0 else s                            # last step this cell stays free
        for dq, dr in hg.AXIAL_NEIGHBORS:                              # reroute (collapsed)
            nq, nr = q + dq, r + dr
            for (lo, hi) in SI.intervals(nq, nr):
                arr = max(s + 1, lo)
                if arr > hi or arr > max_step:
                    continue
                if arr - 1 > hi_c:                                     # can't wait here that long
                    break                                             # later intervals need even more
                wait = arr - (s + 1)                                   # folded pre-move hover
                out.append((("a", nq, nr, arr), c_hold * dt * wait + c_lat * pitch, wait))
        if is_goal_cell(q, r) and s + 1 <= hi_c and s + 1 <= max_step:
            out.append((("a", q, r, s + 1), c_hold * dt, 0))           # hover to retry the landing gate
        return out
