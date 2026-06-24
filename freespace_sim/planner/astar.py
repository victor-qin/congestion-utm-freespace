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
import warnings

import numpy as np

from ..config import SimConfig
from ..cost import trajectory_cost
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
    hover_reservation,
    segment_overlaps_column,
    terminal_radius,
)
from . import hexgrid as hg
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


class AStarPlanner:
    def __init__(self, max_expansions: int = 600_000):
        self.max_expansions = max_expansions
        self._svc: HexOccupancyService | None = None   # incremental hex-occupancy (per ledger)
        self._svc_ledger: ReservationLedger | None = None
        self._tcap: TerminalCapacity | None = None     # temporal pad-capacity authority (per ledger)

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
        dt = cfg.dt_s
        pitch = cfg.nominal_speed_mps * dt
        R = hg.circumradius(cfg)
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        t_depart = req.t_departure                       # types enforces it is set and >= t_request
        base = int(math.ceil(t_depart / dt))             # ceil ⇒ base*dt >= t_depart: never depart before filing
        climb_steps = max(1, int(math.ceil(cfg.climb_time_s / dt)))
        climb_cost = cfg.cost_altitude_change_per_m * (cfg.cruise_level_m - cfg.ground_level_m)

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
        dwell_steps = max(1, int(math.ceil((cfg.hover_time_s + cfg.climb_time_s) / dt)))

        # Shared-terminal context (Phase B). A flight owns its origin/dest vertiports: their columns are
        # transparent to its search (``own``). A shared-hub takeoff/landing dwell is gated by
        # ``TerminalCapacity`` (the temporal authority: pad capacity + column activation); an ordinary
        # (no-terminal) pad still uses the binary ``svc.pad_clear``. No terminal ⇒ own=∅, old behavior.
        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        own = frozenset(t.id for t in (o_term, d_term) if t is not None)
        o_cap = o_term.capacity if o_term else 1
        d_cap = d_term.capacity if d_term else 1

        # admissible heuristic: straight dash at c_lat + the mandatory descent still to come.
        # h_air is evaluated for EVERY generated neighbour (the search hot path), so the hex centre is
        # computed inline as scalars and the distance via math.sqrt(dx*dx+dy*dy) — bit-for-bit the same
        # value np.linalg.norm returns for a length-2 vector, but ~19x cheaper (no array alloc / ufunc).
        sqrt3, c_lat = hg.SQRT3, cfg.cost_air_lateral_per_m

        def h_air(q, r):
            dx, dy = R * sqrt3 * (q + r / 2.0) - gx, R * 1.5 * r - gy
            return c_lat * math.sqrt(dx * dx + dy * dy) + climb_cost

        dx0, dy0 = R * sqrt3 * (oq + orr / 2.0) - gx, R * 1.5 * orr - gy
        h_ground = c_lat * math.sqrt(dx0 * dx0 + dy0 * dy0) + 2.0 * climb_cost

        n_hops = int(math.ceil(max(straight, pitch) / pitch))
        max_step = base + climb_steps + int(math.ceil(cfg.max_ground_delay_s / dt)) + 3 * n_hops + 6

        start = ("g", oq, orr, base)
        g = {start: 0.0}
        came: dict = {}
        counter = itertools.count()
        pq = [(h_ground, next(counter), start)]
        closed: set = set()
        goal_state = None
        expansions = 0

        while pq:
            _, _, st = heapq.heappop(pq)
            if st in closed:
                continue
            closed.add(st)
            if st[0] == "a" and st[1] == gq and st[2] == grr and (
                tcap.dwell_ok(d_term, dest, st[3] * dt, d_cap, origin) if d_term is not None
                else svc.pad_clear(gq, grr, st[3], dwell_steps)
            ):
                goal_state = st
                break
            # reaching the dest hex whose landing dwell is blocked is NOT a goal — fall through and
            # keep expanding (the ground-wait/hover levers find an arrival whose dwell is clear).
            expansions += 1
            if expansions > self.max_expansions:
                break
            base_g = g[st]
            for nst, cost in self._edges(
                st, cfg, pitch, climb_steps, climb_cost, svc, max_step, dwell_steps,
                own, o_cap, o_term, origin, tcap, dest,
            ):
                ng = base_g + cost
                if ng < g.get(nst, math.inf):
                    g[nst] = ng
                    came[nst] = st
                    hh = h_air(nst[1], nst[2]) if nst[0] == "a" else h_ground
                    heapq.heappush(pq, (ng + hh, next(counter), nst))

        if goal_state is None:
            return _deny(req, DenialReason.SEARCH_EXHAUSTED)

        # reconstruct the path
        path = [goal_state]
        while path[-1] != start:
            path.append(came[path[-1]])
        path.reverse()
        air = [s for s in path if s[0] == "a"]
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
            return _deny(req, DenialReason.CONFLICT_FILED)   # raster slack / hover contention

        intent = OperationalIntent(
            request=req,
            status=IntentStatus.ACCEPTED,
            volumes=volumes,
            centerline=centerline,
            ground_delay_s=delay,
            air_hold_s=n_hover * dt,
            air_detour_m=max(0.0, cum_horiz - straight),
            altitude_change_m=2.0 * (cfg.cruise_level_m - cfg.ground_level_m),
            planner="astar",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent

    def _edges(self, st, cfg, pitch, climb_steps, climb_cost, svc, max_step, dwell_steps,
               own=(), o_cap=1, o_term=None, origin=None, tcap=None, dest=None):
        dt = cfg.dt_s
        out = []
        if st[0] == "g":
            _, q, r, s = st
            if s + 1 <= max_step:
                out.append((("g", q, r, s + 1), cfg.cost_ground_delay_per_s * dt))   # ground wait
            ts = s + climb_steps
            # takeoff: the climb-completion air cell must be clear AND the origin pad must admit the whole
            # takeoff dwell starting at ground step s — a shared hub via TerminalCapacity (capacity +
            # column activation + exit-lane toward dest, gated at the exact takeoff time s*dt), an
            # ordinary pad via svc.pad_clear.
            pad_ok = (tcap.dwell_ok(o_term, origin, s * dt, o_cap, dest) if o_term is not None
                      else svc.pad_clear(q, r, s, dwell_steps))
            if ts <= max_step and not svc.is_blocked(q, r, ts, own) and pad_ok:
                out.append((("a", q, r, ts), climb_cost))                            # takeoff
            return out
        _, q, r, s = st
        ns = s + 1
        if ns > max_step:
            return out
        for dq, dr in hg.AXIAL_NEIGHBORS:                                            # reroute
            nq, nr = q + dq, r + dr
            if not svc.is_blocked(nq, nr, ns, own):
                out.append((("a", nq, nr, ns), cfg.cost_air_lateral_per_m * pitch))
        if not svc.is_blocked(q, r, ns, own):                                       # hover
            out.append((("a", q, r, ns), cfg.cost_air_hold_per_s * dt))
        return out

    def _build(self, cruise_wps, origin, dest, base, ground_steps, cfg, origin_term=None, dest_term=None):
        # Shared-terminal hubs: the drone climbs in the tagged hover column, then its strategic corridor
        # (the "exit lane") begins at the column EDGE. Waypoints inside the column are folded away — the
        # centre→edge leg is flown but NOT reserved, because inside the terminal the vertiport handles
        # its own traffic tactically (same-hub flights may share that space concurrently). Only the exit
        # lane + cruise reach the ledger, where corridor boxes stay strict (untagged): two flights can't
        # occupy the same exit lane at once, while divergent same-hub launches go concurrently.
        origin_term, dest_term = as_terminal(origin_term), as_terminal(dest_term)
        half = cfg.corridor_width_m / 2.0
        wps = [[np.asarray(p, float).copy(), t] for p, t in cruise_wps]
        o_xy, d_xy = np.asarray(origin, float)[:2], np.asarray(dest, float)[:2]
        speed = cfg.nominal_speed_mps

        def _exit_radius(term):
            # Inner edge = R − overlap, so the exit lane starts FLUSH with the column edge by default
            # (corridor_overlap = 0). The exit-lane box is tagged with the hub, and the column-involved
            # exemption (conflict.volumes_conflict) makes it transparent to same-hub COLUMNS — so flush is
            # conflict-free vs columns. Two same-hub *corridor* boxes still contend (box↔box stays strict),
            # so divergent lanes need the column wide enough not to crowd: cfg.terminal_radius_m defaults
            # to 90 m for exactly this. overlap>0 penetrates the column; overlap<0 leaves a gap. (Issue #10.)
            ov = term.corridor_overlap if term.corridor_overlap is not None else 0.0
            return terminal_radius(term, cfg) + cfg.corridor_width_m / 2.0 - ov

        if origin_term is not None and len(wps) >= 2:
            wps = _fold_head_into_column(wps, o_xy, _exit_radius(origin_term), speed)
        if dest_term is not None and len(wps) >= 2:
            wps = _fold_tail_into_column(wps, d_xy, _exit_radius(dest_term), speed)

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
            # still conflict (box↔box), so same-direction launches contend → gated by exit_clear.
            tid = (origin_term.id if origin_term is not None and segment_overlaps_column(a, b, o_xy, o_r, cfg)
                   else dest_term.id if dest_term is not None and segment_overlaps_column(a, b, d_xy, d_r, cfg)
                   else None)
            edges.append(corridor_segment_volume(a, ta, b, tb, cfg, terminal_id=tid))
            centerline.append((b.copy(), tb))
            horiz = float(np.linalg.norm((b - a)[:2]))
            cum_horiz += horiz
            if horiz < _EPS:
                n_hover += 1
        t_takeoff = (base + ground_steps) * cfg.dt_s
        t_arrive = wps[-1][1]
        volumes = [
            hover_reservation(origin, t_takeoff, cfg,
                              terminal_id=origin_term.id if origin_term else None,
                              radius=terminal_radius(origin_term, cfg) if origin_term else None),
            *edges,
            hover_reservation(dest, t_arrive, cfg,
                              terminal_id=dest_term.id if dest_term else None,
                              radius=terminal_radius(dest_term, cfg) if dest_term else None),
        ]
        return volumes, centerline, cum_horiz, n_hover
