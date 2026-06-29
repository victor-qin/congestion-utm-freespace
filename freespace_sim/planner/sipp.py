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
from ..geometry import CylinderSpec
from ..volumes import terminal_radius
from . import hexgrid as hg
from .astar import AStarPlanner, _absorb, _committed_arrival
from .compiled_occupancy import CompiledOccupancy

_EPS = 1e-6


def _deny(req, reason):
    return OperationalIntent(
        request=req, status=IntentStatus.REJECTED, denial_reason=reason, planner="sipp"
    )


def _iv_index_global(cocc, cell, step):
    """Pool slot id of ``cell``'s interval containing ``step`` (the kernel's frontier node id); ``-1`` if
    the step is blocked (no interval contains it). Walks the cell's interval chain from slot ``cell``."""
    lo, hi, nxt = cocc.iv_lo, cocc.iv_hi, cocc.iv_nxt
    slot = cell
    while slot != -1:
        if lo[slot] <= step <= hi[slot]:
            return slot
        slot = int(nxt[slot])
    return -1


class SafeIntervalIndex:
    """Cell-keyed inverse of the committed occupancy — the v2 engine behind SIPP's speedup.

    ``HexOccupancyService`` maps ``step -> {cells}``; to build a cell's safe intervals SIPP needs the
    OPPOSITE (``cell -> occupied steps``). v1 recovered it by scanning ``is_blocked`` over the full
    ``[base, max_step]`` horizon PER CELL (dominated by the empty ground-delay tail) — which made SIPP
    slower than A*. This index instead records, per hex cell, the corridor-blocked steps and the
    per-step column hub-coverage, fed incrementally by the ledger commit hook (the same dual-sweep
    rasterization ``HexOccupancyService`` uses). A cell's safe intervals are then built in
    O(#occupied steps of that cell) — O(1) for the common never-touched cell — and :meth:`cell_blocked`
    exactly replicates ``HexOccupancyService.is_blocked`` (pinned by a test).

    NOTE: storage is not reclaimed on eviction yet — the search only ever reads steps >= the request
    clock (so this is correct), but memory reclaim for very long runs is a follow-up."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.R = hg.circumradius(cfg)
        self.infl_blocked = cfg.corridor_width_m / 2.0 + self.R
        self.infl_pad = cfg.effective_hover_radius_m + self.R
        self.corr: dict[tuple[int, int], set[int]] = {}          # cell -> corridor-blocked steps
        self.cols: dict[tuple[int, int], dict[int, set]] = {}    # cell -> step -> {hub_id}
        self.static_cols: dict[tuple[int, int], set] = {}        # always-active: cell -> {hub_id} (step-indep)
        self.n_added = 0
        self.evicted_before: int | None = None

    def on_commit(self, _flight_id, volumes) -> None:
        own_cols = tuple((v.shape.cx, v.shape.cy, v.shape.radius) for v in volumes
                         if v.terminal_id is not None and isinstance(v.shape, CylinderSpec))
        for v in volumes:
            self._add(v, own_cols)
        self.n_added += len(volumes)

    def _inside_a_column(self, q, r, cols) -> bool:
        c = hg.hex_center(q, r, self.R)
        return any((c[0] - cx) ** 2 + (c[1] - cy) ** 2 <= rad * rad for cx, cy, rad in cols)

    def _add(self, vol, own_cols) -> None:
        tid = vol.terminal_id
        is_column = tid is not None and isinstance(vol.shape, CylinderSpec)
        for q, r, s, in_blk in hg.rasterize_volume_dual(
            vol, self.cfg, self.R, self.infl_blocked, self.infl_pad
        ):
            if not in_blk:
                continue                                        # is_blocked only consults in_blk cells
            if is_column:
                self.cols.setdefault((q, r), {}).setdefault(s, set()).add(tid)
            elif not (own_cols and self._inside_a_column(q, r, own_cols)):
                self.corr.setdefault((q, r), set()).add(s)      # (skip own terminal interior, as occupancy)

    def evict_before(self, step) -> None:
        if self.evicted_before is None or step > self.evicted_before:
            self.evicted_before = step   # queries read steps >= request clock; storage reclaim is TODO

    def reset(self) -> None:
        self.corr.clear(); self.cols.clear(); self.n_added = 0; self.evicted_before = None
        # static_cols intentionally preserved: always-active walls are infrastructure, not commit-derived

    def register_static_terminal(self, center, term) -> None:
        """Permanently wall a hub's terminal airspace (column + exit lanes) off from FOREIGN traffic
        (``cfg.terminal_airspace_always_active``) — the SafeIntervalIndex twin of
        ``HexOccupancyService.register_static_terminal``. Step-independent; idempotent per hub."""
        tid = as_terminal(term).id
        for cell in hg.terminal_cells(center, term, self.cfg):
            self.static_cols.setdefault(cell, set()).add(tid)

    def cell_blocked(self, q, r, s, own, fixed_lanes) -> bool:
        """Exact replica of ``HexOccupancyService.is_blocked(q, r, s, own)`` — including the always-active
        ``static_cols`` walls (foreign in EITHER the per-step columns OR the static set ⇒ blocked)."""
        cc = self.cols.get((q, r))
        hubs = cc.get(s) if cc else None
        stat = self.static_cols.get((q, r)) if self.static_cols else None
        if hubs is not None or stat is not None:
            if (hubs is not None and any(t not in own for t in hubs)) or \
                    (stat is not None and any(t not in own for t in stat)):
                return True                                     # foreign column (transient or static) → wall
            return fixed_lanes and s in self.corr.get((q, r), ())   # own-only column + sibling corridor
        return s in self.corr.get((q, r), ())

    def free_intervals(self, q, r, own, base, max_step, fixed_lanes):
        """Maximal free ``[lo,hi]`` step-runs in ``[base,max_step]`` — complement of the cell's blocked
        steps. O(#occupied steps of the cell); O(1) for a never-occupied cell (the common case)."""
        stat = self.static_cols.get((q, r)) if self.static_cols else None
        if stat is not None and any(t not in own for t in stat):
            return []                            # always-active FOREIGN wall ⇒ blocked at EVERY step
        corr = self.corr.get((q, r))
        cols = self.cols.get((q, r))
        if not corr and not cols:
            return [(base, max_step)]
        cand = set()
        if corr:
            cand.update(s for s in corr if base <= s <= max_step)
        if cols:
            cand.update(s for s in cols if base <= s <= max_step)
        blk = sorted(s for s in cand if self.cell_blocked(q, r, s, own, fixed_lanes))
        out, lo = [], base
        for s in blk:
            if s > lo:
                out.append((lo, s - 1))
            lo = s + 1
        if lo <= max_step:
            out.append((lo, max_step))
        return out


class _SafeIntervals:
    """Per-plan memoised view over :class:`SafeIntervalIndex`: a cell's free intervals for THIS flight
    (its ``own`` terminals, step domain, and the ``fixed_lanes`` flag)."""

    def __init__(self, sidx, own, base, max_step, fixed_lanes):
        self.sidx = sidx
        self.own = own
        self.base = base
        self.max_step = max_step
        self.fixed_lanes = fixed_lanes
        self._cache: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def intervals(self, q, r):
        iv = self._cache.get((q, r))
        if iv is None:
            iv = self.sidx.free_intervals(q, r, self.own, self.base, self.max_step, self.fixed_lanes)
            self._cache[(q, r)] = iv
        return iv

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
    evict = False
    for (t2, g2) in F:
        if t2 <= t and g2 + (t - t2) * w <= g + 1e-9:
            return False                                   # dominated by a stored label → skip
        if t <= t2 and g + (t2 - t) * w <= g2 + 1e-9:
            evict = True                                   # new label dominates this stored one
    if evict:                                              # rare: rebuild dropping now-dominated labels
        frontier[key] = [(t2, g2) for (t2, g2) in F if not (t <= t2 and g + (t2 - t) * w <= g2 + 1e-9)]
        frontier[key].append((t, g))
    else:
        F.append((t, g))                                   # common case: append in place (no realloc)
    return True


class SIPPPlanner(AStarPlanner):
    """Safe-interval cost-aware planner; inherits occupancy/terminal sync and corridor build from A*."""

    def __init__(self, max_expansions: int = 600_000, compiled: bool = True):
        super().__init__(max_expansions)
        self._sidx: SafeIntervalIndex | None = None    # cell-keyed inverse index (per ledger)
        self._sidx_ledger = None
        # --- compiled (numba) air-cruise kernel; falls back to the pure-Python reference ---
        self.compiled = compiled
        self._kernel = None
        if compiled:
            try:
                from .sipp_kernel import _search
                self._kernel = _search
            except ImportError:
                self.compiled = False                  # numba absent → pure-Python everywhere
        self._cocc: CompiledOccupancy | None = None    # interval pool (per ledger)
        self._cocc_ledger = None
        self._gen = 0                                  # version stamp for reused kernel state
        self._k_cap = -1                               # frontier size the kernel arrays are sized to
        self._k_lab_cell = None                        # kernel work arrays (allocated lazily)
        self._k_out_q = None
        self._fb = 0                                   # kernel→reference fallbacks (diagnostics/tests)
        self._n_expansions = 0                         # kernel expansions on the last compiled plan

    def _sipp_index(self, req, ledger, cfg) -> "SafeIntervalIndex":
        """Maintain the SafeIntervalIndex in lockstep with the ledger (mirrors ``_occupancy``): first use
        subscribes the commit hook + absorbs existing volumes; a ledger shrink rebuilds; then evict to
        the request clock."""
        sidx = self._sidx
        if sidx is None or self._sidx_ledger is not ledger:
            sidx = self._sidx = SafeIntervalIndex(cfg)
            self._sidx_ledger = ledger
            ledger.subscribe(sidx.on_commit)
            _absorb(sidx, ledger)
            self._register_static(sidx, cfg)
        elif ledger.n_volumes < sidx.n_added:
            sidx.reset()
            _absorb(sidx, ledger)
            self._register_static(sidx, cfg)
        sidx.evict_before(int(req.t_request // cfg.dt_s))
        return sidx

    def plan(self, req, ledger, cfg):
        """Dispatch: the compiled air-cruise kernel for non-terminal flights (Phase 1); the pure-Python
        reference for terminal flights (Phase 2, not yet compiled) and as the universal fallback when
        numba is absent, a flight strays out of the kernel box, or a capacity valve trips."""
        if not self.compiled:
            return self._plan_reference(req, ledger, cfg)
        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        if (o_term is not None or d_term is not None) and not cfg.fixed_exit_lanes:
            return self._plan_reference(req, ledger, cfg)   # legacy-terminal landing needs _committed_arrival
        return self._plan_compiled(req, ledger, cfg)

    def _plan_reference(self, req, ledger, cfg):
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
        sidx = self._sipp_index(req, ledger, cfg)
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

        SI = _SafeIntervals(sidx, own, base, max_step, fixed_lanes)
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
        c_hold = cfg.cost_air_hold_per_s
        pq = [(h_ground, next(counter), start, 0.0, -1)]   # heap: (f, tie, AS, g, interval-index)
        goal_state = None
        expansions = 0

        while pq:
            _, _, st, gst, iv = heapq.heappop(pq)
            if gst > g.get(st, math.inf):
                continue                                   # stale (a cheaper label for this AS won)
            if st[0] == "a" and is_goal_cell(st[1], st[2]) and goal_ok(st):
                goal_state = st
                break                                      # first gate-passing pop (f-order) = optimal
            expansions += 1
            if expansions > self.max_expansions:
                break
            for nst, cost, w, niv in self._succ(
                st, iv, SI, cfg, pitch, climb_steps, climb_cost, dwell_steps, own, o_cap, o_term,
                origin, tcap, dest, o_lanes, o_r, fixed_lanes, max_step, is_goal_cell,
            ):
                ng = gst + cost
                if ng >= g.get(nst, math.inf):
                    continue                               # a same cell-step label is already ≤ this cost
                # Pareto frontier applies only to ordinary AIR cruise cells; the ground ray (niv=-1) and
                # goal cells (per-step landing gate the frontier can't see) are exempt. `niv` comes from
                # _succ (no index_of). The g early-out above means this fires only for cost-improving
                # successors — the dominant cost saver (the frontier check is otherwise ~half the run).
                if niv >= 0 and not is_goal_cell(nst[1], nst[2]) and \
                        not _nondominated(frontier, (nst[1], nst[2], niv), nst[3] * dt, ng, c_hold):
                    continue                               # dominated at its (cell, interval) → prune
                g[nst] = ng
                came[nst] = st
                wait_steps[nst] = w
                hh = h_air(nst[1], nst[2]) if nst[0] == "a" else h_ground
                heapq.heappush(pq, (ng + hh, next(counter), nst, ng, niv))

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

    # ================= compiled (numba) air-cruise path (Phase 1: non-terminal) =================
    def _compiled_occ(self, req, ledger, cfg) -> CompiledOccupancy:
        """Maintain the dense interval table in lockstep with the ledger (mirrors ``_sipp_index``)."""
        cocc = self._cocc
        if cocc is None or self._cocc_ledger is not ledger:
            cocc = self._cocc = CompiledOccupancy(cfg)
            self._cocc_ledger = ledger
            ledger.subscribe(cocc.on_commit)
            _absorb(cocc, ledger)
            self._register_static(cocc, cfg)
        elif ledger.n_volumes < cocc.n_added:
            cocc.reset()
            _absorb(cocc, ledger)
            self._register_static(cocc, cfg)
        cocc.evict_before(int(req.t_request // cfg.dt_s))
        return cocc

    def _register_static(self, occ, cfg) -> None:
        """Register the always-active terminal walls (set by ``run()`` on ``self.static_terminals``) into a
        freshly (re)built occupancy structure (``SafeIntervalIndex`` / ``CompiledOccupancy``). No-op unless
        ``cfg.terminal_airspace_always_active``. Mirrors ``AStarPlanner._occupancy``'s static registration."""
        if getattr(cfg, "terminal_airspace_always_active", False):
            for center, term in self.static_terminals:
                occ.register_static_terminal(center, term)

    def share_occupancy_from(self, master) -> None:
        """Plan against MASTER's committed occupancy (``cocc``/``svc``/``tcap``/``sidx``) without
        subscribing the ledger hook or re-absorbing — for optimistic-batch worker threads (#8 Track A).
        The caller must keep the ledger FROZEN (no commits) while workers plan in parallel; each worker
        keeps its OWN kernel state (``_k_*``), so the only shared mutation is the benign
        ``evict_before`` watermark. With the ``nogil`` kernel, N workers search on N real threads."""
        self._svc = master._svc; self._svc_ledger = master._svc_ledger; self._tcap = master._tcap
        self._cocc = master._cocc; self._cocc_ledger = master._cocc_ledger
        self._sidx = master._sidx; self._sidx_ledger = master._sidx_ledger

    def _kernel_state(self, cocc) -> None:
        """(Re)allocate version-stamped kernel work arrays. The frontier is sized to the interval pool
        (grows with the ledger); labels/heap are fixed-cap (overflow → fallback); both are reused across
        plans — reset is a ``self._gen`` bump, not an O(N) clear."""
        if self._k_cap < cocc.cap:                       # frontier: one slot per pool interval
            self._k_cap = cocc.cap
            ovcap = 1 << 14                              # overlay interval pool (own terminal lane cells)
            self._k_ovcap = ovcap
            tot = cocc.cap + ovcap                       # frontier covers pool slots [0,cap) + overlay [cap,..)
            self._k_front_head = np.full(tot, -1, np.int64)
            self._k_front_tail = np.full(tot, -1, np.int64)   # sorted-by-arr staircase per slot
            self._k_front_gen = np.zeros(tot, np.int64)
            self._k_goal_gen = np.zeros(cocc.cap, np.int64)        # per-cell goal flag (version-stamped)
            self._k_ov_head = np.full(cocc.cap, -1, np.int64)      # per-cell overlay redirect (slot >= cap)
            self._k_ov_gen = np.zeros(cocc.cap, np.int64)          # version-stamped (own-cell transparency)
            self._k_ov_lo = np.empty(ovcap, np.int64)
            self._k_ov_hi = np.empty(ovcap, np.int64)
            self._k_ov_nxt = np.empty(ovcap, np.int64)
        if self._k_lab_cell is None:                     # labels + heap: allocate once
            ml = 1 << 21
            self._k_max = ml
            self._k_lab_cell = np.empty(ml, np.int64)
            self._k_lab_slot = np.empty(ml, np.int64)
            self._k_lab_arr = np.empty(ml, np.int64)
            self._k_lab_g = np.empty(ml, np.float64)
            self._k_lab_par = np.empty(ml, np.int64)
            self._k_lab_next = np.empty(ml, np.int64)
            self._k_lab_prev = np.empty(ml, np.int64)      # doubly-linked sorted frontier
            self._k_lab_dead = np.full(ml, -1, np.int64)   # version-stamped: == gen ⇒ evicted, skip at pop
            self._k_heap_f = np.empty(ml, np.float64)
            self._k_heap_c = np.empty(ml, np.int64)
            self._k_heap_n = np.empty(ml, np.int64)
        if self._k_out_q is None or self._k_out_q.shape[0] < cocc.MAXS + 8:
            self._k_out_q = np.empty(cocc.MAXS + 8, np.int64)
            self._k_out_r = np.empty(cocc.MAXS + 8, np.int64)
            self._k_out_s = np.empty(cocc.MAXS + 8, np.int64)

    def _build_overlay(self, cocc, sidx, cells, own, base, max_step, fixed) -> bool:
        """Fill the overlay interval pool with the OWN-exempt (transparent) free-intervals of ``cells`` —
        the flight's own terminal lane cells, which the global pool blocks as foreign columns. Overlay
        slots live at ``cap + idx``; ``ov_head[cell]``/``ov_gen[cell]`` (version-stamped) redirect the
        kernel there. Returns False on overlay overflow (caller falls back)."""
        cap = cocc.cap
        gen = self._gen
        n = 0
        for (cq, cr) in cells:
            c = cocc.cell_id(cq, cr)
            if c < 0 or self._k_ov_gen[c] == gen:        # out of box, or already built this plan
                continue
            head = prev = -1
            for lo, hi in sidx.free_intervals(cq, cr, own, base, max_step, fixed):
                if n >= self._k_ovcap:
                    return False
                self._k_ov_lo[n] = lo; self._k_ov_hi[n] = hi; self._k_ov_nxt[n] = -1
                slot = cap + n
                if prev < 0:
                    head = slot
                else:
                    self._k_ov_nxt[prev - cap] = slot
                prev = slot; n += 1
            self._k_ov_head[c] = head; self._k_ov_gen[c] = gen
        return True

    def _overlay_slot(self, cocc, cell, step):
        """Overlay slot (``>= cap``) of ``cell``'s transparent interval containing ``step``; -1 if none."""
        cap = cocc.cap
        slot = self._k_ov_head[cell]
        while slot >= 0:
            j = slot - cap
            if self._k_ov_lo[j] <= step <= self._k_ov_hi[j]:
                return slot
            slot = self._k_ov_nxt[j]
        return -1

    def _plan_compiled(self, req, ledger, cfg):
        from .sipp_kernel import FALLBACK, NO_PATH
        dt = cfg.dt_s
        pitch = cfg.nominal_speed_mps * dt
        R = hg.circumradius(cfg)
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        base = int(math.ceil(req.t_departure / dt))
        climb_steps = max(1, int(math.ceil(cfg.climb_time_s / dt)))
        climb_cost = cfg.cost_altitude_change_per_m * (cfg.cruise_level_m - cfg.ground_level_m)
        oq, orr = hg.enu_to_axial(origin[0], origin[1], R)
        gq, grr = hg.enu_to_axial(dest[0], dest[1], R)
        gx, gy = R * hg.SQRT3 * (gq + grr / 2.0), R * 1.5 * grr
        straight = float(np.linalg.norm(dest[:2] - origin[:2]))

        svc = self._occupancy(req, ledger, cfg)
        cocc = self._compiled_occ(req, ledger, cfg)
        dwell_steps = max(1, int(math.ceil((cfg.hover_time_s + cfg.climb_time_s) / dt)))
        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        own = frozenset(t.id for t in (o_term, d_term) if t is not None)
        self._own = own            # last plan's own terminal-id set (diagnostics + occupancy tests)
        o_cap = o_term.capacity if o_term is not None else 1
        d_cap = d_term.capacity if d_term is not None else 1
        fixed = cfg.fixed_exit_lanes
        o_lanes = hg.terminal_lanes(origin, o_term, cfg) if fixed and o_term is not None else []
        d_lanes = hg.terminal_lanes(dest, d_term, cfg) if fixed and d_term is not None else []
        h_off = max((L.dist for L in d_lanes), default=0.0)
        o_r = terminal_radius(o_term, cfg) if o_term is not None else 0.0
        tcap = self._tcap
        c_gd, c_hold, c_lat = (cfg.cost_ground_delay_per_s, cfg.cost_air_hold_per_s,
                               cfg.cost_air_lateral_per_m)
        n_hops = int(math.ceil(max(straight, pitch) / pitch))
        max_step = base + climb_steps + int(math.ceil(cfg.max_ground_delay_s / dt)) + 3 * n_hops + 6

        ocell, gcell = cocc.cell_id(oq, orr), cocc.cell_id(gq, grr)
        if ocell < 0 or gcell < 0:
            return self._plan_reference(req, ledger, cfg)          # out of kernel box → reference

        # ---- per-plan kernel state + own-lane transparency overlay (pool blocks columns foreign-to-all) ----
        self._kernel_state(cocc)
        self._gen += 1
        if fixed and (o_term is not None or d_term is not None):
            sidx = self._sipp_index(req, ledger, cfg)
            lanes = [L.cell for L in o_lanes] + [L.cell for L in d_lanes]
            if not self._build_overlay(cocc, sidx, lanes, own, base, max_step, fixed):
                return self._plan_reference(req, ledger, cfg)      # overlay overflow → reference

        # ---- start labels: ground-delayed takeoffs (lane exits if fixed+terminal, else climb-in-place) ----
        smax = base + int(math.ceil(cfg.max_ground_delay_s / dt))
        s_cell, s_slot, s_arr, s_g = [], [], [], []
        for s in range(base, smax + 1):
            ts = s + climb_steps
            if ts > max_step:
                break
            if fixed and o_term is not None:                  # exit-lane takeoffs (column gated by dwell_ok)
                if not tcap.dwell_ok(o_term, origin, s * dt, o_cap):
                    continue
                for L in o_lanes:
                    lcell = cocc.cell_id(L.cell[0], L.cell[1])
                    if lcell < 0:
                        continue
                    slot = self._overlay_slot(cocc, lcell, ts)    # own-transparent interval at ts (>= cap)
                    if slot < 0:
                        continue                                  # lane blocked at ts (own-exempt view)
                    s_cell.append(lcell); s_slot.append(slot); s_arr.append(ts)
                    s_g.append((s - base) * c_gd * dt + climb_cost + c_lat * (L.dist - o_r))
            else:                                             # climb in place (non-terminal pad)
                if svc.is_blocked(oq, orr, ts, own) or not svc.pad_clear(oq, orr, s, dwell_steps):
                    continue
                slot = _iv_index_global(cocc, ocell, ts)
                if slot < 0:
                    continue
                s_cell.append(ocell); s_slot.append(slot); s_arr.append(ts)
                s_g.append((s - base) * c_gd * dt + climb_cost)
        if not s_cell:
            return _deny(req, DenialReason.SEARCH_EXHAUSTED)

        # ---- goal cell(s) + landing-feasible step intervals (folds the per-step landing gate) ----
        if fixed and d_term is not None:                      # dest exit-lane cells; column-capacity landing
            goal_cells = [c for c in (cocc.cell_id(L.cell[0], L.cell[1]) for L in d_lanes) if c >= 0]
            if not goal_cells:
                return self._plan_reference(req, ledger, cfg)
            for c in goal_cells:
                self._k_goal_gen[c] = self._gen
            landing = [tcap.dwell_ok(d_term, dest, s * dt, d_cap) for s in range(base, max_step + 1)]
        else:                                                 # single dest hex; pad-clear landing
            self._k_goal_gen[gcell] = self._gen
            landing = [svc.pad_clear(gq, grr, s, dwell_steps) for s in range(base, max_step + 1)]
        lf_lo, lf_hi, lo = [], [], -1
        for i, ok in enumerate(landing):
            if ok:
                if lo < 0:
                    lo = base + i
            elif lo >= 0:
                lf_lo.append(lo); lf_hi.append(base + i - 1); lo = -1
        if lo >= 0:
            lf_lo.append(lo); lf_hi.append(max_step)
        if not lf_lo:
            return _deny(req, DenialReason.SEARCH_EXHAUSTED)

        # ---- call the kernel ----
        n, _cost, _n_exp, flag = self._kernel(
            cocc.iv_lo, cocc.iv_hi, cocc.iv_nxt,
            self._k_ov_lo, self._k_ov_hi, self._k_ov_nxt, self._k_ov_head, self._k_ov_gen, cocc.cap,
            cocc.qmin, cocc.rmin, cocc.rspan, cocc.qspan, base, max_step,
            np.asarray(s_cell, np.int64), np.asarray(s_slot, np.int64), np.asarray(s_arr, np.int64),
            np.asarray(s_g, np.float64), len(s_cell),
            self._k_goal_gen, np.asarray(lf_lo, np.int64), np.asarray(lf_hi, np.int64), len(lf_lo),
            c_hold, c_lat, pitch, dt, gx, gy, R, h_off, climb_cost,
            self._gen, self._k_front_head, self._k_front_tail, self._k_front_gen,
            self._k_lab_cell, self._k_lab_slot, self._k_lab_arr, self._k_lab_g, self._k_lab_par,
            self._k_lab_next, self._k_lab_prev, self._k_lab_dead, self._k_max,
            self._k_heap_f, self._k_heap_c, self._k_heap_n, self._k_max,
            self._k_out_q, self._k_out_r, self._k_out_s,
        )
        self._n_expansions = int(_n_exp)
        if flag == FALLBACK:
            self._fb += 1
            return self._plan_reference(req, ledger, cfg)
        if flag == NO_PATH:
            return _deny(req, DenialReason.SEARCH_EXHAUSTED)

        # ---- reconstruct: out_* is goal→start; reverse + re-expand folded hover waits ----
        labels = [(int(self._k_out_q[i]), int(self._k_out_r[i]), int(self._k_out_s[i]))
                  for i in range(n - 1, -1, -1)]
        air = []
        for i, (q, r, a) in enumerate(labels):
            air.append((q, r, a))
            if i + 1 < len(labels):
                for k in range(a + 1, labels[i + 1][2]):
                    air.append((q, r, k))
        self._air = air            # last compiled per-step search path [(q,r,step)] (diagnostics + tests)
        ground_steps = air[0][2] - climb_steps - base
        delay = ground_steps * dt
        cruise_wps: list[TimedPoint] = [
            (np.array([*hg.hex_center(q, r, R), cfg.cruise_level_m]), a * dt) for (q, r, a) in air]
        volumes, centerline, cum_horiz, n_hover = self._build(
            cruise_wps, origin, dest, base, ground_steps, cfg,
            origin_term=req.origin_terminal, dest_term=req.dest_terminal,
        )
        if straight > _EPS and cum_horiz / straight > cfg.max_detour_factor:
            return _deny(req, DenialReason.BUDGET_EXCEEDED)
        if ledger.any_conflict(volumes):
            return _deny(req, DenialReason.CONFLICT_FILED)
        intent = OperationalIntent(
            request=req, status=IntentStatus.ACCEPTED, volumes=volumes, centerline=centerline,
            ground_delay_s=delay, air_hold_s=n_hover * dt, air_detour_m=max(0.0, cum_horiz - straight),
            altitude_change_m=2.0 * (cfg.cruise_level_m - cfg.ground_level_m), planner="sipp",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent

    def _succ(self, st, iv, SI, cfg, pitch, climb_steps, climb_cost, dwell_steps, own, o_cap, o_term,
              origin, tcap, dest, o_lanes, o_r, fixed_lanes, max_step, is_goal_cell):
        """Successors as ``(AS, edge_cost, wait_steps, interval_index)``. ``iv`` is the popped state's
        interval index (carried in the heap → no ``index_of`` scan in the hot loop), and each emitted air
        successor carries its OWN interval index (``-1`` for ground). Ground-wait is a per-step ray and
        takeoff is emitted at the current ground step (so the per-step pad gates match A*); the air
        reroute is the SIPP collapse (one successor per reachable neighbour interval, folding pre-move
        hover at the air rate); a standalone hover is emitted only at a goal cell (to retry the per-step
        landing gate). Mirrors :meth:`AStarPlanner._edges`."""
        dt = cfg.dt_s
        c_gd, c_hold, c_lat = (cfg.cost_ground_delay_per_s, cfg.cost_air_hold_per_s,
                               cfg.cost_air_lateral_per_m)
        svc = self._svc
        tag, q, r, s = st
        out = []
        if tag == "g":
            if s + 1 <= max_step:
                out.append((("g", q, r, s + 1), c_gd * dt, 0, -1))      # ground-wait ray (== A* g→g)
            ts = s + climb_steps
            if ts > max_step:
                return out
            if fixed_lanes and o_term is not None:
                if tcap.dwell_ok(o_term, origin, s * dt, o_cap):        # capacity + column at takeoff
                    for L in o_lanes:
                        lq, lr = L.cell
                        if not svc.is_blocked(lq, lr, ts, own):
                            out.append((("a", lq, lr, ts), climb_cost + c_lat * (L.dist - o_r), 0,
                                        SI.index_of(lq, lr, ts)))        # one edge per lane
            else:
                pad_ok = (tcap.dwell_ok(o_term, origin, s * dt, o_cap, dest) if o_term is not None
                          else svc.pad_clear(q, r, s, dwell_steps))
                if not svc.is_blocked(q, r, ts, own) and pad_ok:
                    out.append((("a", q, r, ts), climb_cost, 0, SI.index_of(q, r, ts)))   # takeoff
            return out

        hi_c = SI.intervals(q, r)[iv][1] if iv >= 0 else s             # last step this cell stays free
        for dq, dr in hg.AXIAL_NEIGHBORS:                              # reroute (collapsed)
            nq, nr = q + dq, r + dr
            for j, (lo, hi) in enumerate(SI.intervals(nq, nr)):
                arr = max(s + 1, lo)
                if arr > hi or arr > max_step:
                    continue
                if arr - 1 > hi_c:                                     # can't wait here that long
                    break                                             # later intervals need even more
                wait = arr - (s + 1)                                   # folded pre-move hover
                out.append((("a", nq, nr, arr), c_hold * dt * wait + c_lat * pitch, wait, j))
        if is_goal_cell(q, r) and s + 1 <= hi_c and s + 1 <= max_step:
            out.append((("a", q, r, s + 1), c_hold * dt, 0, iv))       # hover to retry the landing gate
        return out
