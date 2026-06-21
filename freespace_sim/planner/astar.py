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

import numpy as np

from ..config import SimConfig
from ..cost import trajectory_cost
from ..ledger import ReservationLedger
from ..types import DenialReason, FlightRequest, IntentStatus, OperationalIntent, TimedPoint
from ..volumes import corridor_segment_volume, hover_reservation
from . import hexgrid as hg

_EPS = 1e-6


def _deny(req, reason):
    return OperationalIntent(
        request=req, status=IntentStatus.REJECTED, denial_reason=reason, planner="astar"
    )


class AStarPlanner:
    def __init__(self, max_expansions: int = 300_000):
        self.max_expansions = max_expansions

    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent:
        dt = cfg.dt_s
        pitch = cfg.nominal_speed_mps * dt
        R = hg.circumradius(cfg)
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        t_depart = req.t_departure if req.t_departure is not None else req.t_request
        base = int(round(t_depart / dt))
        climb_steps = max(1, int(math.ceil(cfg.climb_time_s / dt)))
        climb_cost = cfg.cost_altitude_change_per_m * (cfg.cruise_level_m - cfg.ground_level_m)

        oq, orr = hg.enu_to_axial(origin[0], origin[1], R)
        gq, grr = hg.enu_to_axial(dest[0], dest[1], R)
        goal_c = hg.hex_center(gq, grr, R)
        straight = float(np.linalg.norm(dest[:2] - origin[:2]))

        # Build the blocked-set AND the wider pad-occupancy set in a single vectorized sweep over
        # committed volumes. `blocked` uses the corridor footprint (corridor half-width + one hex);
        # `pad_blocked` uses the wider hover-cylinder footprint so the takeoff/landing dwell check
        # matches the actual reservation. The hover reservation occupies the pad for
        # hover_time_s + climb_time_s; without `pad_blocked` the search lands on a pad that a
        # later-crossing corridor sweeps through mid-descent → post-build CONFLICT_FILED. One
        # rasterize_volume_dual call computes each volume's geometry once (was two passes).
        infl_blocked = cfg.corridor_width_m / 2.0 + R
        infl_pad = cfg.effective_hover_radius_m + R
        blocked: set[tuple[int, int, int]] = set()
        pad_blocked: set[tuple[int, int, int]] = set()
        for _fid, vol in ledger.iter_committed():
            for q, r, s, in_blocked in hg.rasterize_volume_dual(vol, cfg, R, infl_blocked, infl_pad):
                pad_blocked.add((q, r, s))
                if in_blocked:
                    blocked.add((q, r, s))
        dwell_steps = max(1, int(math.ceil((cfg.hover_time_s + cfg.climb_time_s) / dt)))

        def pad_clear(q, r, s0):
            """Is the pad at hex (q, r) free for the full dwell window starting at step s0?"""
            return all((q, r, k) not in pad_blocked for k in range(s0, s0 + dwell_steps + 1))

        # admissible heuristic: straight dash at c_lat + the mandatory descent still to come
        def h_air(q, r):
            d = float(np.linalg.norm(hg.hex_center(q, r, R) - goal_c))
            return cfg.cost_air_lateral_per_m * d + climb_cost

        h_ground = cfg.cost_air_lateral_per_m * float(
            np.linalg.norm(hg.hex_center(oq, orr, R) - goal_c)
        ) + 2.0 * climb_cost

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
            if st[0] == "a" and st[1] == gq and st[2] == grr and pad_clear(gq, grr, st[3]):
                goal_state = st
                break
            # reaching the dest hex whose landing dwell is blocked is NOT a goal — fall through and
            # keep expanding (the ground-wait/hover levers find an arrival whose dwell is clear).
            expansions += 1
            if expansions > self.max_expansions:
                break
            base_g = g[st]
            for nst, cost in self._edges(
                st, cfg, pitch, climb_steps, climb_cost, blocked, max_step, pad_blocked, dwell_steps
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
            cruise_wps, origin, dest, base, ground_steps, cfg
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

    def _edges(self, st, cfg, pitch, climb_steps, climb_cost, blocked, max_step, pad_blocked, dwell_steps):
        dt = cfg.dt_s
        out = []
        if st[0] == "g":
            _, q, r, s = st
            if s + 1 <= max_step:
                out.append((("g", q, r, s + 1), cfg.cost_ground_delay_per_s * dt))   # ground wait
            ts = s + climb_steps
            # takeoff: the climb-completion air cell must be clear AND the origin pad must stay clear
            # for the whole takeoff dwell (which starts at this ground step s) — symmetric to landing.
            pad_ok = all((q, r, k) not in pad_blocked for k in range(s, s + dwell_steps + 1))
            if ts <= max_step and (q, r, ts) not in blocked and pad_ok:
                out.append((("a", q, r, ts), climb_cost))                            # takeoff
            return out
        _, q, r, s = st
        ns = s + 1
        if ns > max_step:
            return out
        for dq, dr in hg.AXIAL_NEIGHBORS:                                            # reroute
            nq, nr = q + dq, r + dr
            if (nq, nr, ns) not in blocked:
                out.append((("a", nq, nr, ns), cfg.cost_air_lateral_per_m * pitch))
        if (q, r, ns) not in blocked:                                               # hover
            out.append((("a", q, r, ns), cfg.cost_air_hold_per_s * dt))
        return out

    def _build(self, cruise_wps, origin, dest, base, ground_steps, cfg):
        edges = []
        centerline: list[TimedPoint] = [(cruise_wps[0][0].copy(), cruise_wps[0][1])]
        cum_horiz = 0.0
        n_hover = 0
        for (a, ta), (b, tb) in zip(cruise_wps, cruise_wps[1:]):
            edges.append(corridor_segment_volume(a, ta, b, tb, cfg))
            centerline.append((b.copy(), tb))
            horiz = float(np.linalg.norm((b - a)[:2]))
            cum_horiz += horiz
            if horiz < _EPS:
                n_hover += 1
        t_takeoff = (base + ground_steps) * cfg.dt_s
        t_arrive = cruise_wps[-1][1]
        volumes = [
            hover_reservation(origin, t_takeoff, cfg),
            *edges,
            hover_reservation(dest, t_arrive, cfg),
        ]
        return volumes, centerline, cum_horiz, n_hover
