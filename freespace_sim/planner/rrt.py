"""Path-biased seeded space-time RRT* — the default workhorse planner.

Rather than search the whole airspace, this treats RRT* as a **local repair of the straight line**:

- **Seed**: the straight line's conflict-free prefix is added as the initial tree branch, so the
  search begins at the first obstacle instead of the origin (and a fully-free straight line is
  solved with zero sampling).
- **Tube sampling**: most samples are drawn in a tube around the straight line, so the tree hugs
  it and only bulges where an obstacle forces a detour — killing the wander that uniform sampling
  produced.
- **`step_factor = 1`**: edges are one corridor segment (≈120 m / one timestep), matching the
  straight planner's granularity, so the committed boxes are the right size for the DSS.
- **Shortcut smoothing**: the corner polyline is shortcut-spliced, then the reservation is rebuilt
  at 120 m and re-checked against the ledger — so the committed boxes are minimal *and* exactly
  what was verified. If the path is timing-sensitive (needs an in-air hold that geometric
  smoothing would drop), the smoother bails to the verified RRT* edges; correctness never depends
  on it.

Levers: reroute (move edges), altitude (sampled z), air hold (zero-displacement edges), and ground
delay (delayed-takeoff roots, reserving no airspace while waiting). Nearest-neighbour is in weighted
space-time so a late-arrival goal sample selects a delayed root instead of starving it. A sample cap
bounds compute: a denial is SEARCH_EXHAUSTED only if the goal was never reached, else
BUDGET_EXCEEDED.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..config import SimConfig
from ..cost import trajectory_cost
from ..ledger import ReservationLedger
from ..types import DenialReason, FlightRequest, IntentStatus, OperationalIntent, TimedPoint
from ..volumes import (
    Volume4D,
    build_reservation_from_corners,
    corridor_segment_volume,
    hover_reservation,
)

_EPS = 1e-6


@dataclass
class _Node:
    pos: np.ndarray
    t: float
    parent: int
    g_delay: float            # ground delay of this node's root (inherited down the tree)
    cum_horiz: float          # horizontal path length from root
    cum_dz: float             # total vertical travel from root
    cum_hold: float           # total air-hold seconds from root
    edge_vol: Volume4D | None  # the committed-identical box for the edge from parent


def _deny(req, reason):
    return OperationalIntent(
        request=req, status=IntentStatus.REJECTED, denial_reason=reason, planner="rrt"
    )


class SpaceTimeRRTStar:
    def __init__(
        self,
        max_samples: int = 6000,
        goal_bias: float = 0.15,
        tube_frac: float = 0.55,      # of non-goal samples, fraction drawn in the straight-line tube
        tube_radius_m: float = 600.0,
        hold_prob: float = 0.1,
        ground_delay_prob: float = 0.1,
        step_factor: float = 1.0,     # extension step = step_factor · corridor segment length
        goal_tol_m: float = 1.0,
        rebuild_every: int = 32,
        smooth_iters: int = 60,
    ):
        self.max_samples = max_samples
        self.goal_bias = goal_bias
        self.tube_frac = tube_frac
        self.tube_radius_m = tube_radius_m
        self.hold_prob = hold_prob
        self.ground_delay_prob = ground_delay_prob
        self.step_factor = step_factor
        self.goal_tol_m = goal_tol_m
        self.rebuild_every = rebuild_every
        self.smooth_iters = smooth_iters

    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent:
        rng = np.random.default_rng([cfg.seed, req.flight_id])
        step = self.step_factor * cfg.corridor_segment_len_m
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        t_depart = req.t_departure if req.t_departure is not None else req.t_request

        start = np.array([origin[0], origin[1], cfg.cruise_level_m])
        goal = np.array([dest[0], dest[1], cfg.cruise_level_m])
        straight_horiz = float(np.linalg.norm(goal[:2] - start[:2]))
        w, h = cfg.region_size_m

        # straight-line direction + perpendicular (for tube sampling)
        line = goal - start
        line_xy = line[:2]
        line_len = float(np.linalg.norm(line_xy))
        perp = (np.array([-line_xy[1], line_xy[0]]) / line_len) if line_len > _EPS else np.array(
            [0.0, 1.0]
        )

        # Space-time NN: time is a weighted 4th coordinate so a late-arrival goal sample selects a
        # delayed-takeoff root over an early frontier stuck at the goal.
        w_t = cfg.nominal_speed_mps
        min_travel = straight_horiz / cfg.nominal_speed_mps
        earliest_arrival = t_depart + cfg.climb_time_s + min_travel
        t_hi = earliest_arrival + cfg.max_ground_delay_s

        def key4(pos: np.ndarray, t: float) -> np.ndarray:
            return np.array([pos[0], pos[1], pos[2], w_t * t])

        nodes: list[_Node] = []
        coords: list[np.ndarray] = []
        pending: list[int] = []
        kdt: cKDTree | None = None

        def add_node(node: _Node) -> int:
            nonlocal kdt, pending
            idx = len(nodes)
            nodes.append(node)
            coords.append(key4(node.pos, node.t))
            pending.append(idx)
            if len(pending) >= self.rebuild_every:
                kdt = cKDTree(np.array(coords))
                pending = []
            return idx

        def add_root(delay: float) -> bool:
            if ledger.any_conflict([hover_reservation(origin, t_depart + delay, cfg)]):
                return False
            add_node(_Node(start.copy(), t_depart + delay + cfg.climb_time_s, -1,
                           delay, 0.0, 0.0, 0.0, None))
            return True

        # Initial root: earliest free pad time within budget (jump-to-gap on the takeoff pad).
        d = 0.0
        while d <= cfg.max_ground_delay_s and not add_root(d):
            hits = ledger.conflicts([hover_reservation(origin, t_depart + d, cfg)])
            if not hits:
                break
            d = max((min(cv.t_end for _, cv in hits) + _EPS) - t_depart, d + cfg.dt_s)
        if not nodes:
            return _deny(req, DenialReason.BUDGET_EXCEEDED)   # pad never free within budget

        # SEED: extend the straight line's conflict-free prefix from the first root.
        seed_prev = 0
        n_seed = max(1, int(np.ceil(straight_horiz / step))) if straight_horiz > _EPS else 1
        for k in range(1, n_seed + 1):
            frac = min(1.0, k * step / straight_horiz) if straight_horiz > _EPS else 1.0
            target = start + frac * (goal - start)
            node = self._move_edge(nodes[seed_prev], seed_prev, target, step, cfg)
            if node is None or ledger.any_conflict([node.edge_vol]):
                break
            seed_prev = add_node(node)
            if float(np.linalg.norm(node.pos[:2] - goal[:2])) <= self.goal_tol_m:
                done = self._finish(req, nodes, seed_prev, origin, dest, t_depart,
                                    straight_horiz, cfg, ledger, rng)
                if done is not None:
                    return done                       # straight line was fully free
                break

        reached_goal = False
        for _ in range(self.max_samples):
            # Occasionally open a delayed-takeoff root (the ground-delay lever).
            if rng.random() < self.ground_delay_prob:
                add_root(float(rng.uniform(0.0, cfg.max_ground_delay_s)))
                continue

            r = rng.random()
            if r < self.goal_bias:
                sample_pos = goal
                t_s = earliest_arrival + (rng.uniform(0, 1) ** 3) * cfg.max_ground_delay_s
            elif r < self.goal_bias + (1 - self.goal_bias) * self.tube_frac:
                f = rng.uniform(0, 1)                  # along-line fraction
                base_pt = start + f * (goal - start)
                lat = rng.uniform(-self.tube_radius_m, self.tube_radius_m)
                sample_pos = np.array([
                    base_pt[0] + perp[0] * lat,
                    base_pt[1] + perp[1] * lat,
                    rng.uniform(cfg.z_min_m, cfg.z_max_m),
                ])
                t_s = (t_depart + cfg.climb_time_s + f * min_travel
                       + (rng.uniform(0, 1) ** 3) * cfg.max_ground_delay_s)
            else:                                      # wide uniform fallback (big detours)
                sample_pos = np.array(
                    [rng.uniform(0, w), rng.uniform(0, h), rng.uniform(cfg.z_min_m, cfg.z_max_m)]
                )
                t_s = rng.uniform(t_depart, t_hi)

            ni = self._nearest(coords, kdt, pending, key4(sample_pos, t_s))
            parent = nodes[ni]
            node = (
                self._hold_edge(parent, ni, cfg, rng)
                if rng.random() < self.hold_prob
                else self._move_edge(parent, ni, sample_pos, step, cfg)
            )
            if node is None or ledger.any_conflict([node.edge_vol]):
                continue
            idx = add_node(node)

            if float(np.linalg.norm(node.pos[:2] - goal[:2])) <= self.goal_tol_m:
                done = self._finish(req, nodes, idx, origin, dest, t_depart,
                                    straight_horiz, cfg, ledger, rng)
                if done is not None:
                    return done
                reached_goal = True   # reachable, but pad/detour budget not met yet → keep trying

        return _deny(
            req,
            DenialReason.BUDGET_EXCEEDED if reached_goal else DenialReason.SEARCH_EXHAUSTED,
        )

    # ----- edge proposals -----
    def _move_edge(self, parent, pi, sample_pos, step, cfg) -> _Node | None:
        d = sample_pos - parent.pos
        dist = float(np.linalg.norm(d))
        if dist < _EPS:
            return None
        new_pos = parent.pos + d / dist * min(step, dist)
        new_pos[2] = float(np.clip(new_pos[2], cfg.z_min_m, cfg.z_max_m))
        horiz = float(np.linalg.norm(new_pos[:2] - parent.pos[:2]))
        dz = abs(float(new_pos[2] - parent.pos[2]))
        dt_edge = max(horiz / cfg.nominal_speed_mps, dz / cfg.climb_rate_mps, 1e-3)
        t_new = parent.t + dt_edge
        edge = corridor_segment_volume(parent.pos, parent.t, new_pos, t_new, cfg)
        return _Node(new_pos, t_new, pi, parent.g_delay, parent.cum_horiz + horiz,
                     parent.cum_dz + dz, parent.cum_hold, edge)

    def _hold_edge(self, parent, pi, cfg, rng) -> _Node:
        hold = float(rng.integers(1, 4)) * cfg.dt_s     # loiter 1–3 timesteps in the air
        t_new = parent.t + hold
        edge = corridor_segment_volume(parent.pos, parent.t, parent.pos, t_new, cfg)
        return _Node(parent.pos.copy(), t_new, pi, parent.g_delay, parent.cum_horiz,
                     parent.cum_dz, parent.cum_hold + hold, edge)

    # ----- nearest neighbour -----
    @staticmethod
    def _nearest(coords, kdt, pending, sample) -> int:
        best_i, best_d = -1, np.inf
        if kdt is not None:
            best_d, best_i = kdt.query(sample)
        for idx in pending:
            dist = float(np.linalg.norm(coords[idx] - sample))
            if dist < best_d:
                best_d, best_i = dist, idx
        return int(best_i)

    # ----- finish: verified fallback, then try to smooth it tighter -----
    def _finish(self, req, nodes, goal_idx, origin, dest, t_depart, straight_horiz, cfg, ledger,
                rng) -> OperationalIntent | None:
        base = self._assemble_from_edges(
            req, nodes, goal_idx, origin, dest, t_depart, straight_horiz, cfg, ledger
        )
        if base is None:
            return None                                   # this goal connection is infeasible
        smoothed = self._smooth(
            req, nodes, goal_idx, origin, dest, t_depart, straight_horiz, cfg, ledger, rng
        )
        return smoothed if smoothed is not None else base

    def _assemble_from_edges(self, req, nodes, goal_idx, origin, dest, t_depart, straight_horiz,
                             cfg, ledger) -> OperationalIntent | None:
        """The verified fallback: commit the exact per-edge boxes RRT* checked (holds included)."""
        edges: list[Volume4D] = []
        centerline: list[TimedPoint] = []
        i = goal_idx
        root = nodes[goal_idx]
        while i != -1:
            n = nodes[i]
            centerline.append((n.pos.copy(), n.t))
            if n.edge_vol is not None:
                edges.append(n.edge_vol)
            root = n
            i = n.parent
        edges.reverse()
        centerline.reverse()

        goal_node = nodes[goal_idx]
        origin_hover = hover_reservation(origin, t_depart + root.g_delay, cfg)
        dest_hover = hover_reservation(dest, goal_node.t, cfg)
        if ledger.any_conflict([dest_hover]):
            return None
        total_horiz = goal_node.cum_horiz
        if straight_horiz > _EPS and total_horiz / straight_horiz > cfg.max_detour_factor:
            return None
        return self._make_intent(
            req, [origin_hover, *edges, dest_hover], centerline, root.g_delay,
            goal_node.cum_hold, total_horiz, goal_node.cum_dz, straight_horiz, cfg
        )

    # ----- shortcut smoothing on the corner polyline -----
    def _smooth(self, req, nodes, goal_idx, origin, dest, t_depart, straight_horiz, cfg, ledger,
                rng) -> OperationalIntent | None:
        path = []
        i = goal_idx
        while i != -1:
            path.append(nodes[i])
            i = nodes[i].parent
        path.reverse()
        g_delay = path[0].g_delay
        corners = [path[0].pos.copy()]
        for n in path[1:]:                                # collapse holds (repeated positions)
            if not np.allclose(n.pos, corners[-1]):
                corners.append(n.pos.copy())
        if len(corners) < 2:
            return None

        base = self._reservation_from_corners(
            corners, origin, dest, g_delay, t_depart, straight_horiz, cfg, ledger
        )
        if base is None:
            return None                                   # timing-sensitive → keep verified fallback
        for _ in range(self.smooth_iters):
            if len(corners) <= 2:
                break
            i = int(rng.integers(0, len(corners) - 2))
            j = int(rng.integers(i + 2, len(corners)))
            cand = corners[: i + 1] + corners[j:]
            res = self._reservation_from_corners(
                cand, origin, dest, g_delay, t_depart, straight_horiz, cfg, ledger
            )
            if res is not None:
                corners, base = cand, res
        volumes, centerline, cum_horiz, cum_dz = base
        return self._make_intent(
            req, volumes, centerline, g_delay, 0.0, cum_horiz, cum_dz, straight_horiz, cfg
        )

    def _reservation_from_corners(self, corners, origin, dest, g_delay, t_depart, straight_horiz,
                                  cfg, ledger):
        """Resample a corner polyline to ≤120 m boxes (shared builder), then budget + conflict check.

        Returns (volumes, centerline, cum_horiz, cum_dz) or None if it busts budget or conflicts.
        """
        volumes, centerline, cum_horiz, cum_dz = build_reservation_from_corners(
            corners, origin, dest, t_depart, g_delay, cfg
        )
        if straight_horiz > _EPS and cum_horiz / straight_horiz > cfg.max_detour_factor:
            return None
        if ledger.any_conflict(volumes):
            return None
        return volumes, centerline, cum_horiz, cum_dz

    def _make_intent(self, req, volumes, centerline, g_delay, air_hold, cum_horiz, cum_dz,
                     straight_horiz, cfg) -> OperationalIntent:
        intent = OperationalIntent(
            request=req,
            status=IntentStatus.ACCEPTED,
            volumes=volumes,
            centerline=centerline,
            ground_delay_s=g_delay,
            air_hold_s=air_hold,
            air_detour_m=max(0.0, cum_horiz - straight_horiz),
            altitude_change_m=2.0 * (cfg.cruise_level_m - cfg.ground_level_m) + cum_dz,
            planner="rrt",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent
