"""NLP trajectory-optimization planner (CasADi + IPOPT), bootstrapped from RRT*.

A smooth *local* optimizer that polishes the RRT* path: it moves the interior cruise knots and the
takeoff delay to minimize the true cost (ground delay + detour + altitude), pushing the centerline
out of nearby committed volumes via smooth keep-out penalties. Takeoff time is a decision variable,
and each obstacle penalty is **time-gated** — sliding a knot's time out of an obstacle's window
(by delaying) removes the penalty, so the optimizer can resolve a conflict by waiting *or* by bending.

Correctness never depends on the NLP: the result is rebuilt into the exact committed boxes and
re-checked against the ledger. If it conflicts, busts budget, fails to converge, or isn't cheaper
than the RRT* warm-start, the planner returns the (already feasible) RRT* intent. So `opt` is never
worse than `rrt`.

Default solver is open-source IPOPT; an SCP/QP variant could use Gurobi later.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from ..config import SimConfig
from ..cost import trajectory_cost
from ..geometry import BoxSpec, CylinderSpec
from ..ledger import ReservationLedger
from ..types import FlightRequest, IntentStatus, OperationalIntent
from ..volumes import build_reservation_from_corners
from .rrt import SpaceTimeRRTStar

_EPS = 1e-6


class NLPOptPlanner:
    def __init__(
        self,
        warm_planner=None,                       # any planner; its path seeds the NLP (default RRT*)
        penalty_weight: float = 200.0,
        softmin_eps: float = 3.0,
        keepout_margin_m: float = 8.0,
        time_tau: float = 2.0,
        spatial_margin_m: float = 1000.0,
        max_obstacles: int = 60,
        max_iter: int = 300,
    ):
        self.warm_planner = warm_planner or SpaceTimeRRTStar()
        self.penalty_weight = penalty_weight
        self.softmin_eps = softmin_eps
        self.keepout_margin_m = keepout_margin_m
        self.time_tau = time_tau
        self.spatial_margin_m = spatial_margin_m
        self.max_obstacles = max_obstacles
        self.max_iter = max_iter

    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent:
        warm = self.warm_planner.plan(req, ledger, cfg)
        if warm.accepted and warm.centerline and len(warm.centerline) >= 3:
            seed = [np.asarray(p, float) for p, _ in warm.centerline]
            seed_delay = warm.ground_delay_s
        else:
            # warm planner denied (or trivial): seed from the straight cruise line so the NLP can
            # still TRY — but a head-on seed usually can't discover a lateral detour (see comparison).
            seed, seed_delay = self._straight_seed(req, cfg)
        if len(seed) < 3:
            return warm
        try:
            polished = self._optimize(req, ledger, cfg, seed, seed_delay)
        except Exception:
            polished = None                      # solver blew up → keep the warm start
        if polished is not None and (not warm.accepted or polished.cost < warm.cost - _EPS):
            return polished
        warm.planner = "opt"                     # fell back (possibly denied)
        return warm

    def _straight_seed(self, req, cfg):
        o = np.asarray(req.origin, float)
        d = np.asarray(req.dest, float)
        z = cfg.cruise_level_m
        start, goal = np.array([o[0], o[1], z]), np.array([d[0], d[1], z])
        dist = float(np.linalg.norm(goal[:2] - start[:2]))
        n = max(2, int(np.ceil(dist / cfg.corridor_segment_len_m)))
        return [start + (k / n) * (goal - start) for k in range(n + 1)], 0.0

    # ----- the NLP -----
    def _optimize(self, req, ledger, cfg, knots, seed_delay) -> OperationalIntent | None:
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        t_depart = req.t_departure if req.t_departure is not None else req.t_request
        n = len(knots)
        start, goal = knots[0], knots[-1]
        straight_horiz = float(np.linalg.norm(goal[:2] - start[:2]))
        climb, dt = cfg.climb_time_s, cfg.dt_s
        v_step = cfg.nominal_speed_mps * dt
        z_step = cfg.climb_rate_mps * dt

        obstacles = self._nearby_obstacles(ledger, knots, t_depart, cfg, n)

        nint = n - 2
        pint = ca.MX.sym("pint", nint * 3)
        d = ca.MX.sym("d")

        def Pk(k):
            if k == 0:
                return ca.DM(start)
            if k == n - 1:
                return ca.DM(goal)
            return pint[(k - 1) * 3 : (k - 1) * 3 + 3]

        # objective: true cost (delay + horizontal length + vertical travel)
        obj = cfg.cost_ground_delay_per_s * d
        for k in range(n - 1):
            a, b = Pk(k), Pk(k + 1)
            obj += cfg.cost_air_lateral_per_m * ca.sqrt(ca.sumsqr(b[0:2] - a[0:2]) + 1e-3)
            obj += cfg.cost_altitude_change_per_m * ca.sqrt((b[2] - a[2]) ** 2 + 1e-3)

        # time-gated smooth keep-out penalties
        pen = ca.MX(0)
        for obs in obstacles:
            for k in range(n):
                t_k = t_depart + d + climb + k * dt
                gate = self._gate(t_k, obs["t0"], obs["t1"])
                pen = pen + gate * self._penetration(Pk(k), obs)
        obj = obj + self.penalty_weight * pen

        # constraints: per-segment speed + climb limits
        g, lbg, ubg = [], [], []
        for k in range(n - 1):
            a, b = Pk(k), Pk(k + 1)
            g.append(ca.sumsqr(b[0:2] - a[0:2]))
            lbg.append(0.0)
            ubg.append((v_step * 1.02) ** 2)
            g.append((b[2] - a[2]) ** 2)
            lbg.append(0.0)
            ubg.append((z_step * 1.02) ** 2)

        x = ca.vertcat(d, pint)
        w, h = cfg.region_size_m
        # bound interior z to the continuous band, WIDENED to include the warm seed's altitudes — so a
        # multi-level A* seed (e.g. a low flight level) is polished in place instead of being forced to
        # the single cruise plane (which would violate the per-segment climb limit and be rejected).
        seed_zs = [float(k[2]) for k in knots]
        z_lo, z_hi = min(cfg.z_min_m, min(seed_zs)), max(cfg.z_max_m, max(seed_zs))
        lbx = [0.0] + [0.0, 0.0, z_lo] * nint
        ubx = [cfg.max_ground_delay_s] + [w, h, z_hi] * nint
        x0 = [seed_delay] + np.array(knots[1:-1]).flatten().tolist()

        solver = ca.nlpsol(
            "opt", "ipopt", {"x": x, "f": obj, "g": ca.vertcat(*g)},
            {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": self.max_iter,
             "ipopt.tol": 1e-3, "ipopt.acceptable_tol": 1e-2},
        )
        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        xo = np.asarray(sol["x"]).flatten()
        d_opt = float(xo[0])
        interior = xo[1:].reshape(nint, 3)
        corners = [start, *interior, goal]

        # rebuild the exact committed boxes and re-check — correctness lives here, not in the NLP
        volumes, centerline, cum_horiz, cum_dz = build_reservation_from_corners(
            corners, origin, dest, t_depart, d_opt, cfg
        )
        if straight_horiz > _EPS and cum_horiz / straight_horiz > cfg.max_detour_factor:
            return None
        if ledger.any_conflict(volumes):
            return None
        intent = OperationalIntent(
            request=req,
            status=IntentStatus.ACCEPTED,
            volumes=volumes,
            centerline=centerline,
            ground_delay_s=d_opt,
            air_detour_m=max(0.0, cum_horiz - straight_horiz),
            altitude_change_m=(float(corners[0][2]) - cfg.ground_level_m)
            + (float(corners[-1][2]) - cfg.ground_level_m) + cum_dz,
            planner="opt",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent

    # ----- smooth geometry helpers (CasADi) -----
    def _gate(self, t_k, t0, t1):
        tau = self.time_tau
        return (1.0 / (1.0 + ca.exp(-(t_k - t0) / tau))) * (
            1.0 / (1.0 + ca.exp(-(t1 - t_k) / tau))
        )

    def _penetration(self, p, obs):
        eps = self.softmin_eps
        if obs["kind"] == "box":
            local = ca.mtimes(ca.DM(obs["R"]).T, p - ca.DM(obs["c"]))
            margins = ca.DM(obs["half"]) - ca.fabs(local)
            softmin = -eps * ca.log(ca.sum1(ca.exp(-margins / eps)))
        else:  # cylinder
            dx, dy, dz = p[0] - obs["cx"], p[1] - obs["cy"], p[2] - obs["cz"]
            radial = ca.sqrt(dx * dx + dy * dy + 1e-6)
            rad_m = obs["radius"] - radial
            z_m = obs["hz"] - ca.fabs(dz)
            softmin = -eps * ca.log(ca.exp(-rad_m / eps) + ca.exp(-z_m / eps))
        # + margin so the penalty stays active until the centerline is `keepout_margin` *outside*
        # the inflated obstacle — guaranteeing the rebuilt corridor clears with room to spare.
        return ca.fmax(0.0, softmin + self.keepout_margin_m) ** 2

    # ----- obstacle selection -----
    def _nearby_obstacles(self, ledger, knots, t_depart, cfg, n) -> list[dict]:
        t_lo = t_depart + cfg.climb_time_s
        t_hi = t_depart + cfg.max_ground_delay_s + cfg.climb_time_s + n * cfg.dt_s
        kn = np.array(knots)
        lo = kn.min(0) - self.spatial_margin_m
        hi = kn.max(0) + self.spatial_margin_m
        infl = cfg.corridor_width_m / 2.0
        out: list[dict] = []
        for _fid, vol in ledger.iter_committed():
            if not (vol.t_start < t_hi and t_lo < vol.t_end):
                continue
            amin, amax = vol.aabb()
            if bool(np.any(amax < lo) or np.any(amin > hi)):
                continue
            shape = vol.shape
            if isinstance(shape, BoxSpec):
                out.append({
                    "kind": "box", "R": shape.rotation(), "c": np.array(shape.center, float),
                    "half": np.array(shape.extents, float) / 2.0 + infl,
                    "t0": vol.t_start, "t1": vol.t_end,
                })
            elif isinstance(shape, CylinderSpec):
                out.append({
                    "kind": "cyl", "cx": shape.cx, "cy": shape.cy,
                    "cz": (shape.z_lo + shape.z_hi) / 2.0, "radius": shape.radius + infl,
                    "hz": (shape.z_hi - shape.z_lo) / 2.0 + infl,
                    "t0": vol.t_start, "t1": vol.t_end,
                })
            if len(out) >= self.max_obstacles:
                break
        return out
