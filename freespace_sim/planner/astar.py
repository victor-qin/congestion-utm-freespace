"""Space-time A* on the hex lattice — deterministic global homotopy + lever search.

Searches a graph whose states are ``("a", q, r, step)`` air cells plus a distinguished ``("g", step)``
ground state on the origin pad. The four cost levers are literally four edge types:

  reroute  → move to a neighbour hex          cost c_lat·pitch,  +1 step
  hover    → stay in an air hex               cost c_hold·dt,    +1 step   (reserves the cell)
  ground   → stay on the origin pad           cost c_gd·dt,      +1 step   (reserves nothing)
  altitude → takeoff/landing climb-descend    cost c_alt·Δz                 (ground ↔ cruise)

So a single shortest-path search finds the globally-optimal mix of wait/detour/hover — resolution-
optimal, deterministically, in polynomial time, with time-windowed obstacles handled natively
(the step is in the state). The admissible straight-dash heuristic keeps the time axis from
exploding. Output hex-centre corners are smoothed and rebuilt into the usual continuous corridor;
`verify` is the backstop. Pairs with the NLP as `opt ← astar` for continuous polish.
"""

from __future__ import annotations

import heapq
import itertools
import math
import sys
import warnings
from collections import Counter
from dataclasses import replace

import numpy as np

from ..config import SimConfig
from ..cost import endpoint_altitude_change_m, trajectory_cost
from ..ledger import ReservationLedger
from ..types import (
    DenialReason,
    FlightRequest,
    IntentStatus,
    OperationalIntent,
    TimedPoint,
    as_terminal,
)
from ..volumes import (
    corridor_segment_volume,
    exit_radius,
    hover_reservation,
    segment_overlaps_column,
    terminal_radius,
)
from . import hexgrid as hg
from .compiled_hex_occupancy import search_horizon
from .occupancy import HexOccupancyService
from .terminal_capacity import TerminalCapacity

_EPS = 1e-6


def _deny(req, reason):
    return OperationalIntent(
        request=req, status=IntentStatus.REJECTED, denial_reason=reason, planner="astar"
    )


def _absorb(svc, ledger):
    """Feed already-committed reservations into the occupancy service grouped BY FLIGHT (volumes of one
    flight are committed contiguously), so the per-flight own-column drop in ``on_commit`` applies to
    pre-existing flights exactly as it does to live commits — not volume-by-volume."""
    for _fid, grp in itertools.groupby(ledger.iter_committed(), key=lambda fv: fv[0]):
        svc.on_commit(_fid, [v for _, v in grp])


def _perimeter(center_xy, toward, radius, z):
    """A point ``radius`` m from ``center_xy`` toward ``toward`` (xy), at altitude ``z`` — where a
    hub's corridor starts/ends so same-hub flights diverge from the shared terminal edge."""
    d = np.asarray(toward, float)[:2] - center_xy
    n = float(np.linalg.norm(d))
    p = center_xy + (radius * d / n if n > 1e-9 else np.array([radius, 0.0]))
    return np.array([float(p[0]), float(p[1]), float(z)], float)


def _fold_head_into_column(wps, center, exit_r, speed):
    """Drop the leading waypoints that lie inside ``exit_r`` of ``center`` and re-root the corridor at
    the column edge (the flight's "exit lane"). The folded centre→edge leg is flown but left
    UNRESERVED — inside the terminal the vertiport deconflicts its own traffic tactically, so same-hub
    flights may share that space; only the exit lane reaches the ledger. ``wps`` is a list of
    ``[xyz, t]`` (mutable). Returns the trimmed list; a no-op if the whole cruise stays inside."""
    k = next((i for i in range(1, len(wps))
              if float(np.linalg.norm(wps[i][0][:2] - center)) >= exit_r), None)
    if k is None:
        return wps
    edge = _perimeter(center, wps[k][0], exit_r, wps[0][0][2])
    leg = float(np.linalg.norm(wps[k][0][:2] - edge[:2])) / speed   # unreserved edge→first-cell leg
    return [[edge, wps[k][1] - leg], *wps[k:]]


def _fold_tail_into_column(wps, center, exit_r, speed):
    """Landing-end mirror of :func:`_fold_head_into_column`: drop trailing waypoints inside the
    destination column and end the corridor at that column's edge (descent inside is unreserved)."""
    k = next((i for i in range(len(wps) - 2, -1, -1)
              if float(np.linalg.norm(wps[i][0][:2] - center)) >= exit_r), None)
    if k is None:
        return wps
    edge = _perimeter(center, wps[k][0], exit_r, wps[-1][0][2])
    leg = float(np.linalg.norm(wps[k][0][:2] - edge[:2])) / speed
    return [*wps[:k + 1], [edge, wps[k][1] + leg]]


def _fold_path(wps, origin, dest, origin_term, dest_term, cfg):
    """Apply the head + tail column folds exactly as :meth:`AStarPlanner._build` commits them, returning
    the folded ``[[xyz, t], ...]`` list. Extracted so the landing gate computes the SAME arrival time the
    commit stamps — gate and commit fold through one function and cannot drift. The fold edge is
    :func:`volumes.exit_radius` (the one radius ``_build``, this gate, and ``TerminalCapacity.exit_clear``
    all share). ``origin_term``/``dest_term`` must be normalized (:class:`Terminal` or ``None``)."""
    speed = cfg.nominal_speed_mps
    if origin_term is not None and len(wps) >= 2:
        wps = _fold_head_into_column(wps, np.asarray(origin, float)[:2], exit_radius(origin_term, cfg), speed)
    if dest_term is not None and len(wps) >= 2:
        wps = _fold_tail_into_column(wps, np.asarray(dest, float)[:2], exit_radius(dest_term, cfg), speed)
    return wps


def _committed_arrival(goal_st, came, R, dt, cfg, origin, dest, origin_term, dest_term):
    """The exact time ``_build`` will stamp on the destination hover column for a goal candidate.

    The landing capacity gate must count siblings in the coordinate they were COMMITTED in. ``_build``
    re-times the dest column to the tail-folded column EDGE (:func:`_fold_tail_into_column`) — ~2–7 s before
    the goal-hex (centre) step time the search arrives at — so gating at ``goal_st[3]*dt`` counts the wrong
    window and can silently over-subscribe a pad (capacity has no commit-time backstop: same-hub columns are
    conflict-exempt). Instead, reconstruct this candidate's air path from ``came``, rebuild the same cruise
    waypoints, run the SAME folds (:func:`_fold_path`), and return the folded edge-arrival time. Gate window
    ≡ commit window → the FCFS capacity count is exact, no margin needed. The goal hex is gated only when
    popped (≤ once per distinct arrival step — A* closes each state), each an O(path) reconstruction:
    negligible against the search even at a saturated hub where it fires once per candidate arrival time."""
    air = []
    s = goal_st
    while s is not None and s[0] == "a":
        air.append(s)
        s = came.get(s)
    air.reverse()
    wps = [[np.array([*hg.hex_center(q, r, R), cfg.flight_levels_m[L]]), step * dt]
           for (_, q, r, L, step) in air]
    return _fold_path(wps, origin, dest, origin_term, dest_term, cfg)[-1][1]


_kernel_fallback_warned = False


def _warn_kernel_fallback() -> None:
    """One stderr line, once per process, when the compiled kernel was REQUESTED but numba won't
    import. The fallback is byte-exact so nothing downstream ever notices — which is exactly how a
    ~5-7× slowdown stayed invisible across whole sweeps (issue #30). Explicit ``compiled=False``
    (the ``astar_ref`` oracle) is a request for the reference and does not warn."""
    global _kernel_fallback_warned
    if _kernel_fallback_warned:
        return
    _kernel_fallback_warned = True
    print("WARNING: compiled A* kernel unavailable (numba import failed) — using the pure-Python "
          "reference search, ~5-7x slower. Results are identical. Fix: run via plain `uv run` "
          "(numba is in tool.uv default-groups) or `uv sync`.", file=sys.stderr)


class AStarPlanner:
    def __init__(self, max_expansions: int = 900_000, vertical_edges: bool = True,
                 compiled: bool = True):
        self.max_expansions = max_expansions
        # mid-route layer-change edges (climb/descend en route). Generated at EVERY air state with an
        # all-levels column-clearance check, so they dominate the multi-altitude search cost; the
        # capacity gain comes from per-level TAKEOFF, which is independent. Disable on huge scenarios to
        # recover most of the single-plane speed and keep the gain.
        self.vertical_edges = vertical_edges
        self.last_expansions = 0                        # nodes expanded by the most recent plan (telemetry)
        self._tele = None                               # TelemetryCollector | None (observer-only; sim.run sets it)
        self._svc: HexOccupancyService | None = None   # incremental hex-occupancy (per ledger)
        self._svc_ledger: ReservationLedger | None = None
        self._tcap: TerminalCapacity | None = None     # temporal pad-capacity authority (per ledger)
        # Always-active terminal walls (cfg.terminal_airspace_always_active) are PERMANENT ledger volumes now
        # (filed by sim.run via ledger.register_static_terminal); the occupancy services derive their routing
        # walls from the ledger via subscribe_static in _occupancy/_compiled_occ. No planner-held list.
        # ---- compiled (numba) air-search kernel: reproduces the pure-Python search EXACTLY, ~multiple-x
        # faster; auto-falls back to `_plan_reference` if numba is absent or a safety valve trips. ----
        self.compiled = compiled
        self._kernel = None
        if compiled:
            try:
                from .astar_kernel import _search
                self._kernel = _search
            except ImportError:
                self.compiled = False                   # numba absent → pure-Python everywhere
                _warn_kernel_fallback()
        self._cocc = None                               # CompiledHexOccupancy (per ledger)
        self._cocc_ledger: ReservationLedger | None = None
        self._gen = 0                                   # version stamp for the reused kernel state
        self._ks = None                                 # lazily-allocated kernel work arrays (hash/heap/…)
        self._fb = 0                                    # IN-KERNEL fallback count (FB_OOB/HASH/HEAP + overlap)
        self._fb_reasons: Counter = Counter()           # fallback reason histogram (bench summary)
        # PRE-kernel reference dispatches (legacy-terminal / box-guard) — counted separately from in-kernel
        # FB_* so a bench's "kernel coverage" isn't overstated (these plans run pure Python end to end).
        self._ref_dispatch: Counter = Counter()
        self._remask = 0                                # bounded-mask → full-range widen count (diagnostics)
        if self.compiled:
            self._warm_jit()

    def _file_deny(self, req, reason, volumes, ledger):
        """Deny with the built (rejected) corridor recorded for forensics — observer-only. With telemetry
        on, capture the filed volumes and, for a conflict, the blocker(s) via ``ledger.conflicts`` (a second,
        non-short-circuiting ledger scan — but only on a denial and only when telemetry is enabled); then
        deny exactly as ``_deny`` (the returned intent is UNCHANGED ⇒ verify/reservations stay byte-exact)."""
        if self._tele is not None:
            hits = ledger.conflicts(volumes) if reason is DenialReason.CONFLICT_FILED else None
            self._tele.on_deny(req.flight_id, reason.value, volumes, hits)
        return _deny(req, reason)

    def _occupancy(self, req, ledger, cfg) -> HexOccupancyService:
        """Return the incremental occupancy service, kept in sync with the ledger via the commit
        publish hook (ASTM-subscription style). First use subscribes and absorbs any pre-existing
        volumes; a ledger shrink (release) trips a from-scratch rebuild + warning (add-only)."""
        svc = self._svc
        if svc is None or self._svc_ledger is not ledger:
            svc = self._svc = HexOccupancyService(cfg)
            self._tcap = TerminalCapacity(cfg, ledger)       # temporal pad capacity, same ledger
            self._svc_ledger = ledger
            ledger.subscribe(svc.on_commit)                 # publish hook: future commits auto-feed
            ledger.subscribe(self._tcap.on_commit)
            _absorb(svc, ledger)                             # absorb anything already committed
            _absorb(self._tcap, ledger)
            ledger.subscribe_static(svc._on_static)          # derive always-active routing walls from the
            #                                                  ledger's permanent terminal volumes (replays
            #                                                  all already-registered hubs; no-op if none)
        elif ledger.n_volumes < svc.n_added:
            warnings.warn(
                "ReservationLedger shrank (release?) — rebuilding A* hex-occupancy from scratch; "
                "the incremental occupancy service assumes add-only commits.",
                stacklevel=2,
            )
            svc.reset()
            self._tcap.reset()
            _absorb(svc, ledger)
            _absorb(self._tcap, ledger)
        # Evict state older than the request clock. Requests arrive in non-decreasing time, and with
        # ``t_departure >= t_request`` enforced (types) + ``base = ceil(t_depart/dt)``, the earliest
        # step/time any plan reads is ``base >= floor(t_request/dt)`` — so the bare request-clock
        # watermark (no buffer) drops only un-readable state. EXACTLY TIGHT: it relies on that
        # ``base >= floor(t_request/dt)`` invariant, so don't loosen base/t_departure without re-checking.
        svc.evict_before(int(req.t_request // cfg.dt_s))
        self._tcap.evict_before(req.t_request)
        return svc

    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent:
        """Dispatch to the compiled kernel when it can reproduce the reference cost exactly; else the
        pure-Python reference. The compiled path handles the default ``fixed_exit_lanes=True`` terminals,
        all non-terminal flights, and **always-active terminals** (``cfg.terminal_airspace_always_active``,
        #24): their permanent foreign-column walls are carried in ``CompiledHexOccupancy.static_col`` (the
        same ``terminal_cells`` the reference walls), so the kernel deconflicts against them exactly and the
        own hub's flights fly through their own terminal (the ``_build_overlay`` own-cell mark). **Legacy
        terminals** (``fixed_exit_lanes=False`` with a terminal end) still route to the reference because
        their landing gate is path-dependent (``_committed_arrival`` needs the search's ``came`` mid-flight),
        which the flat-array kernel cannot serve."""
        if not self.compiled:
            return self._plan_reference(req, ledger, cfg)
        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        if (o_term is not None or d_term is not None) and not cfg.fixed_exit_lanes:
            self._ref_dispatch["legacy-terminal"] += 1        # pre-kernel: ran pure Python end to end
            return self._plan_reference(req, ledger, cfg)
        return self._plan_compiled(req, ledger, cfg)

    def _plan_reference(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent:
        dt = cfg.dt_s
        pitch = cfg.nominal_speed_mps * dt
        R = hg.circumradius(cfg)
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        t_depart = req.t_departure                       # types enforces it is set and >= t_request
        base = int(math.ceil(t_depart / dt))             # ceil ⇒ base*dt >= t_depart: never depart before filing
        levels = cfg.flight_levels_m
        ground_z = cfg.ground_level_m
        c_alt = cfg.cost_altitude_change_per_m
        # per-level takeoff: integer climb-steps (≥1) and the climb cost ground → each cruise level
        takeoff_steps = tuple(cfg.climb_steps_to(z) for z in levels)
        takeoff_cost = tuple(c_alt * (z - ground_z) for z in levels)
        # per-rung (L ↔ L+1) mid-route layer change: step count ceil(Δz/(climb_rate·dt)) and cost c_alt·Δz,
        # precomputed once (like takeoff_*) and indexed by min(L, L2) in _edges — not recomputed per node.
        rung_steps = tuple(max(1, int(math.ceil((levels[L + 1] - levels[L]) / (cfg.climb_rate_mps * dt))))
                           for L in range(len(levels) - 1))
        rung_cost = tuple(c_alt * (levels[L + 1] - levels[L]) for L in range(len(levels) - 1))

        oq, orr = hg.enu_to_axial(origin[0], origin[1], R)
        gq, grr = hg.enu_to_axial(dest[0], dest[1], R)
        gx, gy = R * hg.SQRT3 * (gq + grr / 2.0), R * 1.5 * grr   # goal hex centre, as scalars
        straight = float(np.linalg.norm(dest[:2] - origin[:2]))

        # Incremental hex-occupancy service: holds the blocked (corridor footprint) and pad (wider
        # hover-cylinder footprint) cell maps, maintained across plans via the ledger's commit
        # publish hook and time-evicted to the request clock. Replaces the per-plan from-scratch
        # rebuild (O(committed) every plan) with O(new-volumes) maintenance. The pad map backs the
        # takeoff/landing dwell check: the hover reservation occupies the pad for
        # hover_time_s + climb_time_s, so the search must not land where a later corridor sweeps
        # through mid-descent (else post-build CONFLICT_FILED). See occupancy.py.
        svc = self._occupancy(req, ledger, cfg)
        tcap = self._tcap
        # per-level dwell window: hover plus the ACTUAL climb to that level (takeoff or landing)
        dwell_steps = tuple(max(1, int(math.ceil((cfg.hover_time_s + cfg.climb_time_to(z)) / dt)))
                            for z in levels)

        # Shared-terminal context (Phase B). A flight owns its origin/dest vertiports: their columns are
        # transparent to its search (``own``). A shared-hub takeoff/landing dwell is gated by
        # ``TerminalCapacity`` (the temporal authority: pad capacity + column activation); an ordinary
        # (no-terminal) pad still uses the binary ``svc.pad_clear``. No terminal ⇒ own=∅, old behavior.
        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        own = frozenset(t.id for t in (o_term, d_term) if t is not None)
        o_cap = o_term.capacity if o_term else 1
        d_cap = d_term.capacity if d_term else 1

        # Fixed exit lanes (issue #18). A shared-terminal takeoff/landing routes through one of the hub's
        # canonical boundary-hex lanes; same-hub launches are deconflicted by exact cell occupancy
        # (``svc.is_blocked`` sees committed sibling exit corridors). ``o_lanes``/``d_lanes`` are the
        # memoised boundary cells; empty when the flag is off or the end isn't a terminal, in which case
        # the legacy fold/exit_clear path runs.
        fixed_lanes = cfg.fixed_exit_lanes
        o_lanes = hg.terminal_lanes(origin, o_term, cfg) if fixed_lanes and o_term is not None else []
        d_lanes = hg.terminal_lanes(dest, d_term, cfg) if fixed_lanes and d_term is not None else []
        d_lane_by_cell = {L.cell: L for L in d_lanes}
        # When the dest is a terminal the goal is a boundary cell, ~``d_max`` before the hub centre the
        # heuristic targets — subtract it to stay admissible.
        h_off = max((L.dist for L in d_lanes), default=0.0)

        # admissible heuristic: straight dash at c_lat + the mandatory descent still to come.
        # h_air is evaluated for EVERY generated neighbour (the search hot path), so the hex centre is
        # computed inline as scalars and the distance via math.sqrt(dx*dx+dy*dy) — bit-for-bit the same
        # value np.linalg.norm returns for a length-2 vector, but ~19x cheaper (no array alloc / ufunc).
        sqrt3, c_lat = hg.SQRT3, cfg.cost_air_lateral_per_m

        def h_air(q, r, L):
            dx, dy = R * sqrt3 * (q + r / 2.0) - gx, R * 1.5 * r - gy
            return (c_lat * max(0.0, math.sqrt(dx * dx + dy * dy) - h_off)
                    + takeoff_cost[L])                       # == c_alt*(levels[L]-ground_z), precomputed

        dx0, dy0 = R * sqrt3 * (oq + orr / 2.0) - gx, R * 1.5 * orr - gy
        h_ground = (c_lat * max(0.0, math.sqrt(dx0 * dx0 + dy0 * dy0) - h_off)
                    + 2.0 * takeoff_cost[0])                 # mandatory descent from the lowest level + back

        n_hops = int(math.ceil(max(straight, pitch) / pitch))
        climb_span = (int(math.ceil((levels[-1] - levels[0]) / (cfg.climb_rate_mps * dt)))
                      if cfg.n_levels > 1 else 0)
        max_step = search_horizon(base, max(takeoff_steps), n_hops, climb_span, cfg)

        start = ("g", oq, orr, base)
        g = {start: 0.0}
        came: dict = {}
        counter = itertools.count()
        pq = [(h_ground, next(counter), start)]
        closed: set = set()
        goal_state = None
        expansions = 0
        truncated = False                                # True ⇒ stopped at the expansion cap (compute)

        while pq:
            _, _, st = heapq.heappop(pq)
            if st in closed:
                continue
            closed.add(st)
            if st[0] == "a":
                goal_ok = False
                if fixed_lanes and d_term is not None:
                    # Fixed exit lanes: the goal is reaching one of the dest's boundary-hex lanes (at any
                    # flight level st[3]), gated by capacity + column; same-hub siblings at this cell+level
                    # were already seen by the approach corridor's is_blocked check (issue #18).
                    lane = d_lane_by_cell.get((st[1], st[2]))
                    if lane is not None:
                        # window uses THIS flight's descent-from-level time (st[3]), matching the commit
                        goal_ok = tcap.dwell_ok(d_term, dest, st[4] * dt, d_cap, z=levels[st[3]])
                elif st[1] == gq and st[2] == grr:
                    # Legacy: gate landing capacity at the COMMITTED (tail-folded edge) arrival, not the
                    # goal-hex step time — they differ by the centre→edge fold; see _committed_arrival.
                    goal_ok = (
                        tcap.dwell_ok(
                            d_term, dest,
                            _committed_arrival(st, came, R, dt, cfg, origin, dest, o_term, d_term),
                            d_cap, origin, levels[st[3]],
                        ) if d_term is not None
                        else svc.pad_clear(gq, grr, st[4], dwell_steps[st[3]])
                    )
                if goal_ok:
                    goal_state = st
                    break
            # reaching the dest hex whose landing dwell is blocked is NOT a goal — fall through and
            # keep expanding (the ground-wait/hover levers find an arrival whose dwell is clear).
            expansions += 1
            if expansions > self.max_expansions:
                truncated = True
                break
            base_g = g[st]
            for nst, cost in self._edges(
                st, cfg, pitch, levels, takeoff_steps, takeoff_cost, rung_steps, rung_cost, dwell_steps,
                c_alt, svc, max_step, own, o_cap, o_term, origin, tcap, dest, o_lanes,
            ):
                ng = base_g + cost
                if ng < g.get(nst, math.inf):
                    g[nst] = ng
                    came[nst] = st
                    hh = h_air(nst[1], nst[2], nst[3]) if nst[0] == "a" else h_ground
                    heapq.heappush(pq, (ng + hh, next(counter), nst))

        self.last_expansions = expansions               # search-effort telemetry (node-count parity gate)
        if goal_state is None:
            # Two ways to reach no-goal, opposite meanings (see DenialReason). The queue emptied ⇒ A*
            # (complete within the horizon) proved NO feasible plan exists inside max_ground_delay /
            # max_step — real congestion ⇒ BUDGET_EXCEEDED. We stopped at the expansion cap ⇒ the search
            # was truncated before exhausting — a compute artifact a higher cap might beat ⇒ SEARCH_EXHAUSTED.
            return _deny(req, DenialReason.SEARCH_EXHAUSTED if truncated
                         else DenialReason.BUDGET_EXCEEDED)

        # reconstruct the path
        path = [goal_state]
        while path[-1] != start:
            path.append(came[path[-1]])
        path.reverse()
        air = [s for s in path if s[0] == "a"]
        ground_steps = air[0][4] - takeoff_steps[air[0][3]] - base
        delay = ground_steps * dt

        cruise_wps: list[TimedPoint] = [
            (np.array([*hg.hex_center(q, r, R), levels[L]]), s * dt)
            for (_, q, r, L, s) in air
        ]
        volumes, centerline, cum_horiz, n_hover = self._build(
            cruise_wps, origin, dest, base, ground_steps, cfg,
            origin_term=req.origin_terminal, dest_term=req.dest_terminal,
        )
        if straight > _EPS and cum_horiz / straight > cfg.max_detour_factor:
            return self._file_deny(req, DenialReason.BUDGET_EXCEEDED, volumes, ledger)
        if ledger.any_conflict(volumes):
            return self._file_deny(req, DenialReason.CONFLICT_FILED, volumes, ledger)  # raster slack / hover

        # true vertical travel: takeoff climb + every cruise layer change + landing descent
        z_takeoff, z_land = levels[air[0][3]], levels[air[-1][3]]
        cruise_dz = sum(abs(levels[air[i + 1][3]] - levels[air[i][3]]) for i in range(len(air) - 1))
        intent = OperationalIntent(
            request=req,
            status=IntentStatus.ACCEPTED,
            volumes=volumes,
            centerline=centerline,
            ground_delay_s=delay,
            air_hold_s=n_hover * dt,
            air_detour_m=max(0.0, cum_horiz - straight),
            altitude_change_m=endpoint_altitude_change_m(z_takeoff, z_land, cruise_dz, cfg),
            planner="astar",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent

    def _edges(self, st, cfg, pitch, levels, takeoff_steps, takeoff_cost, rung_steps, rung_cost,
               dwell_steps, c_alt, svc, max_step, own=(), o_cap=1, o_term=None, origin=None, tcap=None,
               dest=None, o_lanes=()):
        dt = cfg.dt_s
        out = []
        if st[0] == "g":
            _, q, r, s = st
            if s + 1 <= max_step:
                out.append((("g", q, r, s + 1), cfg.cost_ground_delay_per_s * dt))   # ground wait
            if cfg.fixed_exit_lanes and o_term is not None:
                # Fixed exit lanes × multi-altitude: one takeoff edge per (boundary-hex lane, flight
                # level). The capacity/column dwell window is PER-LEVEL (the committed column lasts
                # hover + climb_time_to(level); a top-level climb outlasts the preferred plane), so gate
                # each level with its own window — computed once per level, reused across lanes. Same-hub
                # siblings are deconflicted by exact cell occupancy at (lane cell, level): is_blocked sees
                # committed sibling exit corridors, so divergent lanes / different levels stay concurrent
                # while two launches into the same cell+level serialise (ground-wait). The lateral
                # traverse out to the lane cell is folded into the edge cost.
                o_r = terminal_radius(o_term, cfg)
                level_ok = tcap.dwell_ok_levels(o_term, origin, s * dt, o_cap, levels)
                for lane in o_lanes:
                    lq, lr = lane.cell
                    for L in range(len(levels)):
                        ts = s + takeoff_steps[L]
                        if level_ok[L] and ts <= max_step and not svc.is_blocked(lq, lr, L, ts, own):
                            out.append((("a", lq, lr, L, ts),
                                        takeoff_cost[L] + cfg.cost_air_lateral_per_m * (lane.dist - o_r)))
                return out
            # legacy / non-terminal takeoff: ONE successor per flight level at the origin hex. The hub
            # gate's capacity+column are level-agnostic (computed once in dwell_ok_levels), the exit lane
            # per level; an ordinary pad uses svc.pad_clear (which scans all levels of the tube).
            hub_ok = (tcap.dwell_ok_levels(o_term, origin, s * dt, o_cap, levels, toward=dest)
                      if o_term is not None else None)
            for L in range(len(levels)):
                ts = s + takeoff_steps[L]
                pad_ok = hub_ok[L] if o_term is not None else svc.pad_clear(q, r, s, dwell_steps[L])
                if ts <= max_step and not svc.is_blocked(q, r, L, ts, own) and pad_ok:
                    out.append((("a", q, r, L, ts), takeoff_cost[L]))                # takeoff to level L
            return out
        _, q, r, L, s = st
        ns = s + 1
        if ns > max_step:
            return out
        for dq, dr in hg.AXIAL_NEIGHBORS:                                            # reroute (same level)
            nq, nr = q + dq, r + dr
            if not svc.is_blocked(nq, nr, L, ns, own):
                out.append((("a", nq, nr, L, ns), cfg.cost_air_lateral_per_m * pitch))
        if not svc.is_blocked(q, r, L, ns, own):                                     # hover (same level)
            out.append((("a", q, r, L, ns), cfg.cost_air_hold_per_s * dt))
        for dL in (-1, 1) if self.vertical_edges else ():                           # vertical layer change
            L2 = L + dL
            if 0 <= L2 < len(levels):
                rung = L if dL == 1 else L2                              # index of the L ↔ L+1 rung
                ts = s + rung_steps[rung]                                # ≥2 steps for a 40 m rung, precomputed
                # the rebuilt climb box occupies only the levels it traverses ({L, L2}): volumes.py sizes
                # its z-extent to [z_L, z_L2] ± corridor_height/2, matching _levels_overlapped, so require
                # clearance on exactly those two levels across the window (s, ts] — not every level.
                if ts <= max_step and all(
                    not svc.is_blocked(q, r, Lk, sk, own)
                    for Lk in (L, L2) for sk in range(s + 1, ts + 1)
                ):
                    out.append((("a", q, r, L2, ts), rung_cost[rung]))
        return out

    def _build(self, cruise_wps, origin, dest, base, ground_steps, cfg, origin_term=None, dest_term=None):
        # Shared-terminal hubs: the drone climbs in the tagged hover column, then its strategic corridor
        # (the "exit lane") begins at the column EDGE. Waypoints inside the column are folded away — the
        # centre→edge leg is flown but NOT reserved, because inside the terminal the vertiport handles
        # its own traffic tactically (same-hub flights may share that space concurrently). Only the exit
        # lane + cruise reach the ledger, where corridor boxes stay strict (untagged): two flights can't
        # occupy the same exit lane at once, while divergent same-hub launches go concurrently.
        origin_term, dest_term = as_terminal(origin_term), as_terminal(dest_term)
        wps = [[np.asarray(p, float).copy(), t] for p, t in cruise_wps]
        if not cfg.fixed_exit_lanes:
            # Legacy: fold the head/tail into the hub columns exactly as the landing gate predicted
            # (shared _fold_path → gate arrival == committed dest-column time, bit-for-bit). With
            # fixed_exit_lanes the air path already starts/ends on boundary cells — nothing to fold.
            wps = _fold_path(wps, origin, dest, origin_term, dest_term, cfg)
        o_xy, d_xy = np.asarray(origin, float)[:2], np.asarray(dest, float)[:2]

        edges = []
        centerline: list[TimedPoint] = [(wps[0][0].copy(), wps[0][1])]
        cum_horiz = 0.0
        n_hover = 0
        o_r = terminal_radius(origin_term, cfg) if origin_term is not None else 0.0
        d_r = terminal_radius(dest_term, cfg) if dest_term is not None else 0.0
        for (a, ta), (b, tb) in zip(wps, wps[1:]):
            # Tag EVERY box that reaches into its hub's OWN column (not just the first/last exit lane),
            # so the column-involved exemption covers the whole in-column reach. An untagged cruise box
            # grazing the shared column would otherwise conflict (different tid) at commit — the
            # cruise-box-clip bug. The number of such boxes is geometry-dependent (radius × exit angle),
            # so we test each box, not a fixed index. Far cruise boxes stay untagged; two same-hub boxes
            # still conflict (box↔box), so same-direction launches contend — serialised by is_blocked
            # cell occupancy under fixed lanes, or by exit_clear on the legacy path.
            tid = (origin_term.id if origin_term is not None and segment_overlaps_column(a, b, o_xy, o_r, cfg)
                   else dest_term.id if dest_term is not None and segment_overlaps_column(a, b, d_xy, d_r, cfg)
                   else None)
            edges.append(corridor_segment_volume(a, ta, b, tb, cfg, terminal_id=tid))
            centerline.append((b.copy(), tb))
            horiz = float(np.linalg.norm((b - a)[:2]))
            cum_horiz += horiz
            if horiz < _EPS and abs(float(b[2] - a[2])) < _EPS:   # genuine hover (a layer change is not)
                n_hover += 1
        if cfg.fixed_exit_lanes and edges:
            # Force the hub tag on the first/last (boundary-cell) box: it leaves from / arrives at the
            # column edge and can graze the shared column, and an untagged box grazing it would conflict
            # at commit (different tid) — the cruise-box-clip. segment_overlaps_column tags interior
            # boxes; this guarantees the boundary box too.
            if origin_term is not None:
                edges[0] = replace(edges[0], terminal_id=origin_term.id)
            # When a single-box corridor is both the origin exit and the dest approach (a degenerate
            # hub→hub short hop), edges[-1] IS edges[0]; tag dest only when it is a distinct box, so it
            # can't clobber the origin tag above (which would conflict the box against the origin column).
            if dest_term is not None and not (origin_term is not None and len(edges) == 1):
                edges[-1] = replace(edges[-1], terminal_id=dest_term.id)
        t_takeoff = (base + ground_steps) * cfg.dt_s
        t_arrive = wps[-1][1]
        # the takeoff/landing columns span the regulated tube (z_hi defaults to airspace_ceiling_m);
        # their dwell window covers the ACTUAL climb to the chosen cruise level (first/last waypoint z).
        z_takeoff, z_land = float(wps[0][0][2]), float(wps[-1][0][2])
        volumes = [
            hover_reservation(origin, t_takeoff, cfg,
                              terminal_id=origin_term.id if origin_term else None,
                              radius=terminal_radius(origin_term, cfg) if origin_term else None,
                              climb_time_s=cfg.climb_time_to(z_takeoff)),
            *edges,
            hover_reservation(dest, t_arrive, cfg,
                              terminal_id=dest_term.id if dest_term else None,
                              radius=terminal_radius(dest_term, cfg) if dest_term else None,
                              climb_time_s=cfg.climb_time_to(z_land)),
        ]
        return volumes, centerline, cum_horiz, n_hover

    # ==================================================================================================
    # Compiled (numba) path — reproduces `_plan_reference` EXACTLY, ~multiple-x faster.
    # ==================================================================================================
    def _compiled_occ(self, req, ledger, cfg):
        """The flat-array occupancy (corridor + column pools), kept in lockstep with the ledger exactly
        as ``_occupancy`` keeps ``HexOccupancyService`` (first-use subscribe+absorb, shrink→rebuild,
        evict to the request clock). Coexists with ``_occupancy`` — the host still needs the reference
        service for ``pad_clear`` (non-terminal takeoff/landing gate)."""
        cocc = self._cocc
        if cocc is None or self._cocc_ledger is not ledger:
            from .compiled_hex_occupancy import CompiledHexOccupancy
            cocc = self._cocc = CompiledHexOccupancy(cfg)
            self._cocc_ledger = ledger
            ledger.subscribe(cocc.on_commit)
            _absorb(cocc, ledger)
            ledger.subscribe_static(cocc._on_static)         # derive the compiled routing walls from the
            #                                                  ledger's permanent terminal volumes (replays
            #                                                  all already-registered hubs; no-op if none)
        elif ledger.n_volumes < cocc.n_added:
            cocc.reset()
            _absorb(cocc, ledger)
        cocc.evict_before(int(req.t_request // cfg.dt_s))
        return cocc

    def _kernel_state(self, cocc):
        """(Re)allocate the version-stamped kernel work arrays, reused across plans (gen bump → O(1)
        reset). Sized once; ``ov_own_gen`` (per-cell own-column mark) grows if a new ledger's box is bigger."""
        NC = cocc.NC
        if self._ks is None:
            # The g-hash and frontier heap MUST stay ahead of the search cap, or the kernel silently
            # falls back to pure Python on exactly the hardest flights (the ones that reach
            # max_expansions) — the catastrophic-tail failure mode (a few long fallbacks eat the run).
            # Derive both from max_expansions so bumping the cap can never desync them: >=2x headroom
            # keeps the open-addressing load factor low and the frontier from overflowing. (Overflow only
            # ever triggers a SAFE reference fallback, so this governs the fallback RATE, not correctness.)
            log2 = max(20, (self.max_expansions * 2 - 1).bit_length())
            cap = 1 << log2
            mh = cap
            self._ks = {
                "g_key": np.empty(cap, np.int64), "g_gen": np.zeros(cap, np.int64),
                "g_val": np.empty(cap, np.float64), "g_came": np.empty(cap, np.int64),
                "g_flag": np.empty(cap, np.int8), "cap": cap, "log2": log2,
                "heap_f": np.empty(mh, np.float64), "heap_c": np.empty(mh, np.int64),
                "heap_n": np.empty(mh, np.int64), "mh": mh,
                "ov_own_gen": np.zeros(NC, np.int32), "NC": NC,
                "out_q": np.empty(cocc.MAXS + 8, np.int64), "out_r": np.empty(cocc.MAXS + 8, np.int64),
                "out_L": np.empty(cocc.MAXS + 8, np.int64), "out_s": np.empty(cocc.MAXS + 8, np.int64),
            }
        ks = self._ks
        if ks["NC"] < NC or len(ks["out_q"]) < cocc.MAXS + 8:
            ks["ov_own_gen"] = np.zeros(NC, np.int32); ks["NC"] = NC
            for k in ("out_q", "out_r", "out_L", "out_s"):
                ks[k] = np.empty(cocc.MAXS + 8, np.int64)
        return ks

    def _build_overlay(self, cocc, o_term, d_term, origin, dest, gen) -> bool:
        """Mark this flight's OWN terminal footprint cells (``ov_own_gen[cell] = gen``) so the kernel's
        ``_blocked`` treats them as transparent (own column) instead of walls. Cheap: rasterize the 1–2
        own hub columns (same rasterizer that built the column pool) and mark their in_blk cells — no
        per-step scan. Under ``terminal_airspace_always_active`` (#24) ALSO mark the hub's full
        ``terminal_cells`` (the wider flood-fill geometry the reference walls), so the permanent static wall
        is transparent to the hub that owns it — matching ``is_blocked``'s tid-based own-hub exemption.

        Returns ``True`` if any own cell is ALSO covered by a FOREIGN hub's column (via ``col_owners``):
        the single-boolean overlay cannot distinguish "own here" from "own AND foreign here", so the
        caller falls back to the reference for exactness (issue #3). ``demand.py`` reject-samples hub
        spacing (#27, and #24 on the wider ``exit_radius`` extent for static walls), making this rare, but
        detecting it keeps the kernel exact regardless of spacing rather than *assuming* separation."""
        ov = self._ks["ov_own_gen"]
        cfg = cocc.cfg
        z_hi = cfg.flight_levels_m[-1]
        own_ids = frozenset(t.id for t in (o_term, d_term) if t is not None)
        overlap = False

        def mark(c):                                     # mark cell own; flag if a FOREIGN column shares it
            nonlocal overlap
            if c < 0:
                return
            ov[c] = gen
            owners = cocc.col_owners.get(c)
            if owners is not None and not owners <= own_ids:
                overlap = True

        for term, center in ((o_term, origin), (d_term, dest)):
            if term is None:
                continue
            col = hover_reservation(np.asarray(center, float), 0.0, cfg,
                                    terminal_id=term.id, radius=terminal_radius(term, cfg),
                                    climb_time_s=cfg.climb_time_to(z_hi))
            for q, r, L, _s, in_blk in hg.rasterize_volume_dual(
                col, cfg, cocc.R, cocc.infl_blocked, cocc.infl_pad
            ):
                if in_blk:
                    mark(cocc.cell_id(q, r, L))
            if cfg.terminal_airspace_always_active:      # the permanent static wall's wider geometry (#24)
                # COUPLING: this own-hub exemption is GEOMETRIC (terminal_cells at `center`), whereas the
                # reference is.blocked exempts by terminal ID (occupancy.is_blocked, geometry-independent).
                # They agree only because `center` (the flight's terminal endpoint) is bit-identical to the
                # hub center registered into the static wall (both from demand.place_hubs). A future demand
                # model that offset a terminal endpoint from its hub center would leave some own static cells
                # unmarked here → the kernel would wall its own terminal → divergence. Holds for all shipped
                # demand models. (This own-static path is only reachable when fixed_exit_lanes=True — a
                # terminal flight with fixed_exit_lanes=False dispatches to the reference — so the kernel's
                # non-gating of the fixed-lane sibling rule never bites here.)
                for q, r in hg.terminal_cells(center, term, cfg):
                    for L in range(cfg.n_levels):
                        mark(cocc.cell_id(q, r, L))
        return overlap

    def _warm_jit(self):
        """Compile the kernel once at construction with a tiny synthetic input (off the hot path)."""
        if self._kernel is None:
            return
        try:
            NC, MAXS = 9, 5
            iv_lo = np.zeros(NC, np.int32); iv_hi = np.full(NC, MAXS, np.int32); iv_nxt = np.full(NC, -1, np.int32)
            cv_lo = np.zeros(NC, np.int32); cv_hi = np.full(NC, MAXS, np.int32); cv_nxt = np.full(NC, -1, np.int32)
            ng = 6
            self._kernel(
                iv_lo, iv_hi, iv_nxt, cv_lo, cv_hi, cv_nxt, np.zeros(NC, np.bool_), np.zeros(NC, np.int32),
                0, 0, 3, 3, 1, 0, MAXS,
                1, 1, np.array([1], np.int64), np.array([1], np.int64), np.array([0.0]), 1,
                np.array([1], np.int64), np.array([0.0]), np.ones(ng, np.bool_), ng, 1.0,
                np.array([1], np.int64), np.array([0.0]), 1.0, 3.0, False,
                np.array([1], np.int64), np.array([1], np.int64), 1, np.ones(ng, np.bool_),
                0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                1, np.empty(64, np.int64), np.zeros(64, np.int64), np.empty(64, np.float64),
                np.empty(64, np.int64), np.empty(64, np.int8), 64, 6,
                np.empty(64, np.float64), np.empty(64, np.int64), np.empty(64, np.int64), 64,
                np.empty(16, np.int64), np.empty(16, np.int64), np.empty(16, np.int64), np.empty(16, np.int64),
                1000,
            )
        except Exception as e:                                # compile failure → degrade to pure Python
            warnings.warn(
                f"astar numba kernel failed to warm/compile ({e!r}); falling back to the pure-Python "
                f"reference planner for ALL plans. Install a compatible numba, or clear stale "
                f".nbi/.nbc caches after a kernel-signature change.",
                RuntimeWarning, stacklevel=2,
            )
            self.compiled = False                             # dispatch every plan to _plan_reference
            self._kernel = None

    def _plan_compiled(self, req, ledger, cfg):
        from . import astar_kernel as K

        # ---- setup: IDENTICAL to _plan_reference's head, so the kernel gets identical inputs ----
        dt = cfg.dt_s
        pitch = cfg.nominal_speed_mps * dt
        R = hg.circumradius(cfg)
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        base = int(math.ceil(req.t_departure / dt))
        levels = cfg.flight_levels_m
        ground_z = cfg.ground_level_m
        c_alt = cfg.cost_altitude_change_per_m
        takeoff_steps = tuple(cfg.climb_steps_to(z) for z in levels)
        takeoff_cost = tuple(c_alt * (z - ground_z) for z in levels)
        rung_steps = tuple(max(1, int(math.ceil((levels[L + 1] - levels[L]) / (cfg.climb_rate_mps * dt))))
                           for L in range(len(levels) - 1))
        rung_cost = tuple(c_alt * (levels[L + 1] - levels[L]) for L in range(len(levels) - 1))

        oq, orr = hg.enu_to_axial(origin[0], origin[1], R)
        gq, grr = hg.enu_to_axial(dest[0], dest[1], R)
        gx, gy = R * hg.SQRT3 * (gq + grr / 2.0), R * 1.5 * grr
        straight = float(np.linalg.norm(dest[:2] - origin[:2]))

        svc = self._occupancy(req, ledger, cfg)
        tcap = self._tcap
        cocc = self._compiled_occ(req, ledger, cfg)
        dwell_steps = tuple(max(1, int(math.ceil((cfg.hover_time_s + cfg.climb_time_to(z)) / dt)))
                            for z in levels)

        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        own = frozenset(t.id for t in (o_term, d_term) if t is not None)
        o_cap = o_term.capacity if o_term else 1
        d_cap = d_term.capacity if d_term else 1
        fixed_lanes = cfg.fixed_exit_lanes
        o_lanes = hg.terminal_lanes(origin, o_term, cfg) if fixed_lanes and o_term is not None else []
        d_lanes = hg.terminal_lanes(dest, d_term, cfg) if fixed_lanes and d_term is not None else []
        h_off = max((L.dist for L in d_lanes), default=0.0)

        sqrt3, c_lat = hg.SQRT3, cfg.cost_air_lateral_per_m
        dx0, dy0 = R * sqrt3 * (oq + orr / 2.0) - gx, R * 1.5 * orr - gy
        h_ground = c_lat * max(0.0, math.sqrt(dx0 * dx0 + dy0 * dy0) - h_off) + 2.0 * takeoff_cost[0]
        n_hops = int(math.ceil(max(straight, pitch) / pitch))
        climb_span = (int(math.ceil((levels[-1] - levels[0]) / (cfg.climb_rate_mps * dt)))
                      if cfg.n_levels > 1 else 0)
        max_step = search_horizon(base, max(takeoff_steps), n_hops, climb_span, cfg)

        # ---- box / window membership guard: else fall back to the reference ----
        if cocc.cell_id(oq, orr, 0) < 0 or max_step > cocc.MAXS:
            self._ref_dispatch["box-guard"] += 1
            return self._plan_reference(req, ledger, cfg)

        ks = self._kernel_state(cocc)
        n_levels = len(levels)
        full_ng = max_step - base + 1

        # ---- takeoff lanes (once; independent of the mask window) ----
        if fixed_lanes and o_term is not None:
            o_r = terminal_radius(o_term, cfg)
            lane_q = np.asarray([L.cell[0] for L in o_lanes], np.int64)
            lane_r = np.asarray([L.cell[1] for L in o_lanes], np.int64)
            lane_lat = np.asarray([c_lat * (L.dist - o_r) for L in o_lanes], np.float64)
            to_terminal = True
        else:                                            # non-terminal origin (legacy-terminal fell back)
            lane_q = np.asarray([oq], np.int64)
            lane_r = np.asarray([orr], np.int64)
            lane_lat = np.asarray([0.0], np.float64)
            to_terminal = False
        # ---- goal cells (once) ----
        if fixed_lanes and d_term is not None:
            goal_q = np.asarray([L.cell[0] for L in d_lanes], np.int64)
            goal_r = np.asarray([L.cell[1] for L in d_lanes], np.int64)
            land_terminal = True
        else:                                            # non-terminal destination
            goal_q = np.asarray([gq], np.int64)
            goal_r = np.asarray([grr], np.int64)
            land_terminal = False

        rs = np.asarray(rung_steps if rung_steps else (0,), np.int64)
        rc = np.asarray(rung_cost if rung_cost else (0.0,), np.float64)
        tks = np.asarray(takeoff_steps, np.int64)
        tkc = np.asarray(takeoff_cost, np.float64)
        c_gd_dt, c_hold_dt, c_lat_pitch = (cfg.cost_ground_delay_per_s * dt,
                                           cfg.cost_air_hold_per_s * dt, c_lat * pitch)

        # ---- two-phase BOUNDED mask. ``max_step`` (hence the full mask width) is blown up ~7x by the
        # ground-delay allowance, which flights almost never use. Build the per-(step, level) takeoff/
        # landing feasibility masks over a TIGHT window first (a full detour + a modest ground delay); if
        # the search reaches a ground/goal step beyond it the kernel returns FB_MASK, and we rebuild over
        # the FULL range and re-run. Exact — the widened run IS the full-mask search — and no reference
        # fallback, just the rare re-run. Most flights finish in one tight pass. ----
        n_gsteps = min(full_ng, 3 * n_hops + 2 * climb_span + 134)
        while True:
            to_ok = np.zeros(n_gsteps * n_levels, np.bool_)
            if to_terminal:
                for gi in range(n_gsteps):
                    lvl_ok = tcap.dwell_ok_levels(o_term, origin, (base + gi) * dt, o_cap, levels)
                    for Lv in range(n_levels):
                        to_ok[gi * n_levels + Lv] = lvl_ok[Lv]
            else:
                for gi in range(n_gsteps):
                    for Lv in range(n_levels):
                        to_ok[gi * n_levels + Lv] = svc.pad_clear(oq, orr, base + gi, dwell_steps[Lv])
            land_ok = np.zeros(n_gsteps * n_levels, np.bool_)
            if land_terminal:
                for gi in range(n_gsteps):
                    for Lv in range(n_levels):
                        land_ok[gi * n_levels + Lv] = tcap.dwell_ok(
                            d_term, dest, (base + gi) * dt, d_cap, z=levels[Lv])
            else:
                for gi in range(n_gsteps):
                    for Lv in range(n_levels):
                        land_ok[gi * n_levels + Lv] = svc.pad_clear(gq, grr, base + gi, dwell_steps[Lv])

            # `gen` version-stamps BOTH the own-column overlay AND the kernel's open-addressing hash (its
            # O(1) reset — `g_gen[i] != gen` marks a slot empty). The FB_MASK widen re-run therefore needs a
            # FRESH gen; DO NOT hoist this above the loop, or the re-run reuses the tight pass's closed nodes
            # and returns a spurious NO_PATH. The overlay is window-independent (re-stamped cheaply on the
            # rare widen); the overlap→reference check (issue #3) is identical each pass, so it aborts the
            # whole plan on the first iteration.
            self._gen += 1
            gen = self._gen
            if own and self._build_overlay(cocc, o_term, d_term, origin, dest, gen):
                self._fb += 1
                self._fb_reasons["own-foreign-overlap"] += 1
                warnings.warn(
                    f"astar own∩foreign column cell for flight {req.flight_id} "
                    f"O={tuple(req.origin)}→D={tuple(req.dest)}; running reference (boolean overlay).",
                    RuntimeWarning, stacklevel=2,
                )
                return self._plan_reference(req, ledger, cfg)
            n_out, _cost, n_exp, status, aux = self._kernel(   # kernel g-cost unused: intent.cost = trajectory_cost below
                cocc.corr.lo, cocc.corr.hi, cocc.corr.nxt, cocc.col.lo, cocc.col.hi, cocc.col.nxt,
                cocc.static_col, ks["ov_own_gen"],
                cocc.qmin, cocc.rmin, cocc.qspan, cocc.rspan, n_levels, base, max_step,
                oq, orr, lane_q, lane_r, lane_lat, len(lane_q),
                tks, tkc, to_ok, n_gsteps, c_gd_dt,
                rs, rc, c_lat_pitch, c_hold_dt, self.vertical_edges,
                goal_q, goal_r, len(goal_q), land_ok,
                gx, gy, R, h_off, c_lat, h_ground,
                gen, ks["g_key"], ks["g_gen"], ks["g_val"], ks["g_came"], ks["g_flag"], ks["cap"], ks["log2"],
                ks["heap_f"], ks["heap_c"], ks["heap_n"], ks["mh"],
                ks["out_q"], ks["out_r"], ks["out_L"], ks["out_s"], self.max_expansions,
            )
            if status == K.FB_MASK and n_gsteps < full_ng:
                self._remask += 1
                n_gsteps = full_ng                       # widen to the full range and re-run (exact)
                continue
            break
        self.last_expansions = n_exp

        # ---- status handling ----
        if status >= K.FB_OOB:                           # safety valve → pure-Python reference
            reason = {K.FB_OOB: "out-of-box", K.FB_HASH: "hash-full", K.FB_HEAP: "heap-full",
                      K.FB_MASK: "mask-exhausted-at-full"}[status]   # FB_MASK here ⇒ bug (full mask signaled)
            cell = ""
            if status == K.FB_OOB:
                cell = f", straddled q={aux // 65536 - 32768} r={aux % 65536 - 32768}"
            self._fb += 1
            self._fb_reasons[reason] += 1
            warnings.warn(
                f"astar compiled kernel FALLBACK ({reason}) for flight {req.flight_id} "
                f"O={tuple(req.origin)}→D={tuple(req.dest)}{cell}, n_exp={n_exp}; running reference",
                RuntimeWarning, stacklevel=2,
            )
            return self._plan_reference(req, ledger, cfg)
        if status == K.NO_PATH_TRUNC:
            return _deny(req, DenialReason.SEARCH_EXHAUSTED)
        if status == K.NO_PATH_EMPTY:
            return _deny(req, DenialReason.BUDGET_EXCEEDED)

        # ---- reconstruct (out_* is goal-first; keep air states, reverse to start-first) ----
        air = [(int(ks["out_q"][i]), int(ks["out_r"][i]), int(ks["out_L"][i]), int(ks["out_s"][i]))
               for i in range(n_out) if ks["out_L"][i] >= 0]
        air.reverse()
        if not air:                                      # an accepted kernel path ALWAYS ends at an air goal,
            self._fb += 1                                # so an empty air list is a kernel ANOMALY, not a
            self._fb_reasons["empty-air"] += 1           # routine dispatch — surface it as a fallback + warn
            warnings.warn(
                f"astar compiled kernel returned a goal with no air states for flight "
                f"{req.flight_id}; running reference (kernel anomaly)",
                RuntimeWarning, stacklevel=2,
            )
            return self._plan_reference(req, ledger, cfg)
        ground_steps = air[0][3] - takeoff_steps[air[0][2]] - base
        cruise_wps: list[TimedPoint] = [
            (np.array([*hg.hex_center(q, r, R), levels[L]]), s * dt) for (q, r, L, s) in air
        ]
        volumes, centerline, cum_horiz, n_hover = self._build(
            cruise_wps, origin, dest, base, ground_steps, cfg,
            origin_term=req.origin_terminal, dest_term=req.dest_terminal,
        )
        if straight > _EPS and cum_horiz / straight > cfg.max_detour_factor:
            return self._file_deny(req, DenialReason.BUDGET_EXCEEDED, volumes, ledger)
        if ledger.any_conflict(volumes):
            return self._file_deny(req, DenialReason.CONFLICT_FILED, volumes, ledger)

        z_takeoff, z_land = levels[air[0][2]], levels[air[-1][2]]
        cruise_dz = sum(abs(levels[air[i + 1][2]] - levels[air[i][2]]) for i in range(len(air) - 1))
        intent = OperationalIntent(
            request=req,
            status=IntentStatus.ACCEPTED,
            volumes=volumes,
            centerline=centerline,
            ground_delay_s=ground_steps * dt,
            air_hold_s=n_hover * dt,
            air_detour_m=max(0.0, cum_horiz - straight),
            altitude_change_m=endpoint_altitude_change_m(z_takeoff, z_land, cruise_dz, cfg),
            planner="astar",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent
