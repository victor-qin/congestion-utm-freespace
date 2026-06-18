"""MILP trajectory-optimization planner (Richards & How big-M), bootstrapped from RRT*.

The *global* counterpart to the NLP `opt` planner. On an absolute-time step grid, position per step
is continuous; a **binary per obstacle face per active step** encodes the non-convex "stay outside"
disjunction (pass left/right/over/under), so branch-and-bound enumerates *which side* of every
obstacle to pass and returns the globally cheapest route — the homotopy choice the local NLP can't
make. Boxes map to their (oriented) linear faces; cylinders to a circumscribed polygon; the L2
detour length to a polyhedral norm (linear). Departure delay is a decision variable too, carrying
two temporal faces per obstacle ("pass before / after its window"), so a single solve trades delay
against detour globally.

A *warm planner* (RRT* by default, or A*) supplies a candidate + fallback, and two flags reshape the
solve (see `MILPOptPlanner`): `optimize_delay=False` fixes the delay to the warm path's (a pure
spatial refiner); `lock_homotopy=True` additionally pins each obstacle's which-side binary to the
warm path — collapsing the search to a fast LP that only *tightens* the geometry within the chosen
homotopy (this is how `astar_milp` gets MILP quality at ~LP speed).

Correctness lives outside the solver: the MILP path is rebuilt into the exact committed boxes and
re-checked against the ledger; on any conflict / infeasibility / non-improvement it falls back to
the warm intent. So the planner is never worse than its warm planner.

Default solver is open-source CBC (bundled with PuLP); Gurobi/MISOCP (exact L2) is a future option.
"""

from __future__ import annotations

import numpy as np
import pulp

from ..config import SimConfig
from ..cost import trajectory_cost
from ..geometry import BoxSpec, CylinderSpec
from ..ledger import ReservationLedger
from ..types import FlightRequest, IntentStatus, OperationalIntent
from ..volumes import build_reservation_from_corners
from .rrt import SpaceTimeRRTStar

_EPS = 1e-6


def _fix(var, value):
    """Pin a PuLP binary to a value (collapses its branch — turns the MILP toward an LP)."""
    var.lowBound = var.upBound = value


class MILPOptPlanner:
    """Big-M MILP trajectory planner, with three modes selected by the constructor flags:

    - default (`optimize_delay=True`): full global solve — chooses which side of every obstacle AND
      the optimal delay-vs-detour mix. Most powerful, slowest.
    - `optimize_delay=False`: spatial-only refiner at the warm planner's delay (drops the delay
      variable — faster).
    - `lock_homotopy=True`: also pin the which-side binaries to the warm path, so CBC solves a near-LP
      that only tightens geometry within the warm homotopy. `astar_milp` = A* warm + both flags.
    """

    def __init__(
        self,
        warm_planner=None,           # candidate/fallback + (when fixed-delay) the delay source
        optimize_delay: bool = True,  # False → pure spatial refiner at the warm planner's delay
        lock_homotopy: bool = False,  # True → fix each obstacle's which-side binary to the warm path
        n_dirs: int = 16,            # polyhedral-norm directions for horizontal length
        cyl_faces: int = 8,          # polygon faces approximating a hover cylinder
        detour_allow: float = 1.7,   # step budget = this × straight distance
        keepout_margin_m: float = 4.0,
        lock_margin_m: float = 80.0,  # only pin a knot's side when the warm path is this clearly outside
        max_steps: int = 60,
        max_obstacles: int = 40,
        time_limit_s: float = 20.0,
    ):
        self.warm_planner = warm_planner or SpaceTimeRRTStar()
        self.optimize_delay = optimize_delay
        self.lock_homotopy = lock_homotopy
        self.n_dirs = n_dirs
        self.cyl_faces = cyl_faces
        self.detour_allow = detour_allow
        self.keepout_margin_m = keepout_margin_m
        self.lock_margin_m = lock_margin_m
        self.max_steps = max_steps
        self.max_obstacles = max_obstacles
        self.time_limit_s = time_limit_s

    def plan(
        self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig
    ) -> OperationalIntent:
        """Run the warm planner, then the MILP, and return the cheaper feasible intent.

        The warm planner gives a candidate, a fallback (if the MILP fails/denies), and — when the
        flags are set — the fixed delay and/or the reference path whose which-side choices pin the
        binaries. A locked LP that returns infeasible is retried once unlocked.
        """
        warm = self.warm_planner.plan(req, ledger, cfg)   # candidate + fallback (+ delay if fixed)
        fixed = None if self.optimize_delay else (warm.ground_delay_s if warm.accepted else 0.0)
        ref = warm.centerline if (self.lock_homotopy and warm.accepted and warm.centerline) else None
        try:
            milp = self._solve(req, ledger, cfg, fixed, ref)
            if milp is None and ref is not None:
                milp = self._solve(req, ledger, cfg, fixed, None)   # locked LP infeasible → unlock
        except Exception:
            milp = None
        cands = [i for i in (warm, milp) if i is not None and i.accepted]
        if not cands:
            warm.planner = "milp"
            return warm                              # both denied
        best = min(cands, key=lambda i: i.cost)      # global delay-vs-detour trade-off, then min
        best.planner = "milp"
        return best

    def _solve(self, req, ledger, cfg, fixed_delay=None, ref_path=None) -> OperationalIntent | None:
        """Build and solve the big-M MILP for one flight; return an ACCEPTED intent or None.

        Variables: cruise position per step (endpoints pinned by bounds); per-segment horizontal
        length ``L_k`` (a polyhedral-norm lower bound on ‖Δxy‖) and vertical travel ``V_k``; and the
        departure delay ``d`` (a variable, or the ``fixed_delay`` constant). Cruise time at knot k is
        ``t0 + d + ΣL/v`` — tied to path LENGTH (not a fixed dt/step), so the vehicle can't "slow down
        for free" to dodge a time window, and the timing matches the speed-based rebuild.

        Each nearby obstacle adds, per segment, a disjunction "outside it on one spatial face OR the
        whole segment passes before/after its window" (big-M binaries, ≥1 enforced). With ``ref_path``
        set, those binaries are pinned to the warm path's choices, turning the solve into a fast LP.
        Objective = c_lat·ΣL + c_alt·ΣV + c_gd·d. The solved corners are rebuilt into real committed
        boxes and re-checked (`_verify_with_delay`) — correctness never relies on the LP being exact.
        """
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        t_depart = req.t_departure if req.t_departure is not None else req.t_request
        z = cfg.cruise_level_m
        start = np.array([origin[0], origin[1], z])
        goal = np.array([dest[0], dest[1], z])
        straight_horiz = float(np.linalg.norm(goal[:2] - start[:2]))
        v_step = cfg.nominal_speed_mps * cfg.dt_s
        z_step = cfg.climb_rate_mps * cfg.dt_s
        if straight_horiz < _EPS:
            return None

        N = min(self.max_steps, int(np.ceil(self.detour_allow * straight_horiz / v_step)) + 2)
        # the warm path resampled to the MILP grid → tells us which side/when the homotopy goes
        ref_pos, ref_t = (None, None)
        if ref_path is not None and len(ref_path) >= 2:
            ref_pos, ref_t = self._resample(ref_path, N)
        t0 = t_depart + cfg.climb_time_s             # cruise start at ZERO delay
        # reachable absolute-time range across all delays in [0, max_ground_delay]
        t_lo = t0 - cfg.time_buffer_s
        t_hi = t0 + cfg.max_ground_delay_s + (N - 1) * cfg.dt_s + cfg.time_buffer_s
        obstacles = self._nearby_obstacles(ledger, start, goal, t_lo, t_hi, cfg)

        # TIGHT, problem-scaled big-Ms — a loose M wrecks the LP relaxation and CBC crawls.
        m_space = 3.0 * max(cfg.region_size_m)       # bounds |Rᵀ(p−c)| over the region
        m_time = (t_hi - t_lo) + cfg.max_ground_delay_s + 100.0   # bounds the time-face range

        prob = pulp.LpProblem("milp_traj", pulp.LpMinimize)
        # departure delay: a free variable (global delay-vs-detour trade-off) or a fixed constant
        # (pure spatial refiner — e.g. seeded by A*, which already chose the delay)
        d = float(fixed_delay) if fixed_delay is not None else pulp.LpVariable(
            "delay", 0, cfg.max_ground_delay_s
        )
        # position variables (endpoints fixed by tight bounds)
        px = [pulp.LpVariable(f"x{k}", 0, cfg.region_size_m[0]) for k in range(N)]
        py = [pulp.LpVariable(f"y{k}", 0, cfg.region_size_m[1]) for k in range(N)]
        pz = [pulp.LpVariable(f"z{k}", cfg.z_min_m, cfg.z_max_m) for k in range(N)]
        for arr, s, g in ((px, start[0], goal[0]), (py, start[1], goal[1]), (pz, start[2], goal[2])):
            arr[0].lowBound = arr[0].upBound = s
            arr[-1].lowBound = arr[-1].upBound = g

        # polyhedral horizontal length L_k and vertical travel V_k per segment
        dirs = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, self.n_dirs, endpoint=False)]
        L = [pulp.LpVariable(f"L{k}", 0) for k in range(N - 1)]
        V = [pulp.LpVariable(f"V{k}", 0) for k in range(N - 1)]
        for k in range(N - 1):
            dx, dy, dz = px[k + 1] - px[k], py[k + 1] - py[k], pz[k + 1] - pz[k]
            for cx, cy in dirs:
                prob += L[k] >= cx * dx + cy * dy          # L_k ≈ ||Δxy|| (lower bound)
                prob += cx * dx + cy * dy <= v_step        # speed limit (polygon of radius v_step)
            prob += V[k] >= dz
            prob += V[k] >= -dz
            prob += dz <= z_step
            prob += dz >= -z_step

        # cumulative cruise time per knot, tied to path LENGTH (not a fixed dt/step) so the vehicle
        # can't "slow down for free" to dodge a time window — this matches the speed-based rebuild,
        # so the only ways to arrive later are real ground delay or a longer path. cumL[k]=Σ_{j<k} Lⱼ
        v = cfg.nominal_speed_mps
        cumL = [pulp.LpAffineExpression()]
        for k in range(N - 1):
            cumL.append(cumL[-1] + L[k])

        # big-M obstacle avoidance. SPATIAL faces are per sampled point (so a thin obstacle can't be
        # hopped over in one step). TEMPORAL faces are per SEGMENT and use the segment's start/end
        # time — because the committed corridor box spans the whole segment (± time_buffer), so the
        # whole segment must clear the window, not just the sampled instant.
        samples = (0.0, 1.0 / 3.0, 2.0 / 3.0)
        for oi, obs in enumerate(obstacles):
            for k in range(N - 1):
                t_start = t0 + d + cumL[k] / v
                t_end = t0 + d + (cumL[k] + L[k]) / v
                # margin = box time-buffer + one segment-time, since the obstacle crossing can fall
                # mid-segment while we constrain the segment's start/end time (keeps the rebuild safe)
                tmarg = cfg.time_buffer_s + cfg.dt_s
                b_before = pulp.LpVariable(f"tb{oi}_{k}", cat="Binary")
                b_after = pulp.LpVariable(f"ta{oi}_{k}", cat="Binary")
                prob += t_end <= (obs["t0"] - tmarg) + m_time * b_before     # whole seg before window
                prob += t_start >= (obs["t1"] + tmarg) - m_time * b_after    # whole seg after window

                # HOMOTOPY LOCK: copy the warm path's temporal choice (pass before / after / neither)
                tlock = False
                if ref_pos is not None:
                    if ref_t[k + 1] + tmarg <= obs["t0"]:                    # warm passes before
                        _fix(b_before, 0)
                        _fix(b_after, 1)
                        tlock = True
                    elif ref_t[k] - tmarg >= obs["t1"]:                      # warm passes after
                        _fix(b_after, 0)
                        _fix(b_before, 1)
                        tlock = True
                    else:                                                    # warm avoids spatially
                        _fix(b_before, 1)
                        _fix(b_after, 1)

                for si, s in enumerate(samples):
                    xe = (1 - s) * px[k] + s * px[k + 1]
                    ye = (1 - s) * py[k] + s * py[k + 1]
                    ze = (1 - s) * pz[k] + s * pz[k + 1]
                    ref_pt = None if ref_pos is None else (1 - s) * ref_pos[k] + s * ref_pos[k + 1]
                    self._add_spatial_keepout(prob, f"{oi}_{k}_{si}", xe, ye, ze, obs, m_space,
                                              [b_before, b_after], ref_pt=ref_pt,
                                              temporal_locked=tlock)

        prob += (cfg.cost_air_lateral_per_m * pulp.lpSum(L)
                 + cfg.cost_altitude_change_per_m * pulp.lpSum(V)
                 + cfg.cost_ground_delay_per_s * d)             # trade detour against delay globally
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit_s))
        if pulp.LpStatus[prob.status] not in ("Optimal",) or px[1].value() is None:
            return None

        d_opt = float(fixed_delay) if fixed_delay is not None else float(d.value() or 0.0)
        corners = [np.array([px[k].value(), py[k].value(), pz[k].value()], float) for k in range(N)]
        # The MILP fixes the spatial homotopy + an approximate delay; polish the delay with a small
        # jump-to-gap on the *rebuilt* path so it verifies exactly (absorbs the LP↔rebuild timing gap).
        polished = self._verify_with_delay(corners, origin, dest, t_depart, d_opt, straight_horiz,
                                           cfg, ledger)
        if polished is None:
            return None
        volumes, centerline, cum_horiz, cum_dz, d_final = polished
        intent = OperationalIntent(
            request=req,
            status=IntentStatus.ACCEPTED,
            volumes=volumes,
            centerline=centerline,
            ground_delay_s=d_final,
            air_detour_m=max(0.0, cum_horiz - straight_horiz),
            altitude_change_m=2.0 * (cfg.cruise_level_m - cfg.ground_level_m) + cum_dz,
            planner="milp",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent

    def _verify_with_delay(self, corners, origin, dest, t_depart, d_start, straight_horiz, cfg,
                           ledger):
        """Step the delay up from the MILP's value until the rebuilt path is conflict-free.

        Returns (volumes, centerline, cum_horiz, cum_dz, delay) or None if the spatial path can't be
        made feasible by waiting within budget (then the caller falls back to RRT*).
        """
        d = max(0.0, d_start)
        while d <= cfg.max_ground_delay_s + _EPS:
            volumes, centerline, cum_horiz, cum_dz = build_reservation_from_corners(
                corners, origin, dest, t_depart, d, cfg
            )
            if straight_horiz > _EPS and cum_horiz / straight_horiz > cfg.max_detour_factor:
                return None
            if not ledger.any_conflict(volumes):
                return volumes, centerline, cum_horiz, cum_dz, d
            d += cfg.dt_s
        return None

    def _resample(self, centerline, n):
        """Resample a timed polyline to n points evenly by arc length → (positions, times).

        Maps the warm path onto the MILP's n knots so each knot can read off the warm path's side and
        timing. Alignment is by arc-length *fraction*, so a long warm path and a short MILP path still
        correspond knot-for-knot (which is why far-apart knots agree but transition knots may not).
        """
        pts = [np.asarray(p, float) for p, _ in centerline]
        ts = [float(t) for _, t in centerline]
        cum = [0.0]
        for i in range(len(pts) - 1):
            cum.append(cum[-1] + float(np.linalg.norm(pts[i + 1] - pts[i])))
        total = cum[-1] or 1.0
        rp, rt, j = [], [], 0
        for k in range(n):
            target = total * k / (n - 1)
            while j < len(cum) - 2 and cum[j + 1] < target:
                j += 1
            seg = cum[j + 1] - cum[j]
            f = (target - cum[j]) / seg if seg > _EPS else 0.0
            rp.append(pts[j] + f * (pts[j + 1] - pts[j]))
            rt.append(ts[j] + f * (ts[j + 1] - ts[j]))
        return rp, rt

    def _best_spatial_face(self, obs, ref_pt, m) -> tuple[int, float]:
        """(face index, signed margin) of the obstacle face the warm point is most clearly outside.

        Margin > 0 means the point is that many metres beyond that face; the caller only pins the
        binary when the margin is large (a clear side), leaving near-boundary knots free so the locked
        problem stays feasible. Box faces are ±x/±y/±z in the box frame; cylinder faces are the
        polygon edges + the two z-caps.
        """
        p = np.asarray(ref_pt, float)
        if obs["kind"] == "box":
            local = obs["R"].T @ (p - obs["c"])
            half = obs["half"] + m
            margins = []
            for a in range(3):
                margins += [local[a] - half[a], -half[a] - local[a]]   # +face a, then −face a
        else:
            r, hz = obs["radius"] + m, obs["hz"] + m
            margins = [
                np.cos(2 * np.pi * i / self.cyl_faces) * (p[0] - obs["cx"])
                + np.sin(2 * np.pi * i / self.cyl_faces) * (p[1] - obs["cy"]) - r
                for i in range(self.cyl_faces)
            ]
            margins += [(p[2] - obs["cz"]) - hz, -hz - (p[2] - obs["cz"])]
        idx = int(np.argmax(margins))
        return idx, float(margins[idx])

    def _add_spatial_keepout(self, prob, tag, x, y, zz, obs, m_space, temporal,
                             ref_pt=None, temporal_locked=False) -> int:
        """Spatial big-M faces for one sampled point, OR'd with the segment's shared temporal faces.

        The disjunction is: outside the obstacle in at least one spatial dimension OR the whole
        segment passes before/after the window (the two `temporal` binaries, shared across the
        segment's samples). When ``ref_pt`` is given, the binaries are pinned to the warm path's
        choice (which side / before-after) so CBC doesn't re-enumerate — the MILP becomes an LP that
        tightens *within* that homotopy. Returns the number of binaries in the disjunction.
        """
        m = self.keepout_margin_m
        faces = list(temporal)

        if obs["kind"] == "box":
            R, c, half = obs["R"], obs["c"], obs["half"] + m
            bs = [pulp.LpVariable(f"b{tag}_{a}_{s}", cat="Binary") for a in range(3) for s in (0, 1)]
            for a in range(3):
                # local_a = column a of R dotted with (p - c)
                la = R[0, a] * (x - c[0]) + R[1, a] * (y - c[1]) + R[2, a] * (zz - c[2])
                prob += la >= half[a] - m_space * bs[2 * a]           # beyond +face a
                prob += la <= -half[a] + m_space * bs[2 * a + 1]      # beyond -face a
        else:
            # cylinder → circumscribed polygon + two z-faces
            cx, cy, cz = obs["cx"], obs["cy"], obs["cz"]
            r, hz = obs["radius"] + m, obs["hz"] + m
            bs = [pulp.LpVariable(f"c{tag}_{i}", cat="Binary") for i in range(self.cyl_faces + 2)]
            for i in range(self.cyl_faces):
                a = 2 * np.pi * i / self.cyl_faces
                prob += np.cos(a) * (x - cx) + np.sin(a) * (y - cy) >= r - m_space * bs[i]
            prob += (zz - cz) >= hz - m_space * bs[self.cyl_faces]
            prob += (zz - cz) <= -hz + m_space * bs[self.cyl_faces + 1]

        faces += bs
        prob += pulp.lpSum(faces) <= len(faces) - 1              # ≥1 face (spatial OR temporal) enforced

        if ref_pt is not None:                                   # pin spatial binaries to warm choice
            if temporal_locked:
                for b in bs:                                     # a temporal face already carries it
                    _fix(b, 1)
            else:
                idx, margin = self._best_spatial_face(obs, ref_pt, m)
                if margin > self.lock_margin_m:                  # warm is CLEARLY on one side → pin it
                    for j, b in enumerate(bs):
                        _fix(b, 0 if j == idx else 1)
                # else: a transition knot near the obstacle — leave it free for CBC (keeps it feasible)
        return len(faces)

    def _nearby_obstacles(self, ledger, start, goal, t_lo, t_hi, cfg) -> list[dict]:
        """Committed volumes that could constrain this flight → obstacle dicts for the MILP.

        Keeps volumes whose time window overlaps [t_lo, t_hi] and whose footprint falls in an xy reach
        box around the route (capped at ``max_obstacles``). Each footprint is inflated by the new
        corridor's half-width (Minkowski) and its window clamped to the reachable horizon — so a
        permanent obstacle within the flight gets a finite ``t1`` the tight big-M can disable. Box →
        ``{R, c, half}``; cylinder → ``{cx, cy, cz, radius, hz}``; both carry ``t0``/``t1``.
        """
        reach = self.detour_allow * float(np.linalg.norm(goal[:2] - start[:2])) + cfg.corridor_width_m
        lo = np.minimum(start, goal) - reach
        hi = np.maximum(start, goal) + reach
        infl = cfg.corridor_width_m / 2.0
        out: list[dict] = []
        for _fid, vol in ledger.iter_committed():
            if not (vol.t_start < t_hi and t_lo < vol.t_end):
                continue
            amin, amax = vol.aabb()
            if bool(np.any(amax < lo) or np.any(amin > hi)):
                continue
            # clamp the window to the reachable horizon so the tight big-M can disable the temporal
            # faces — a permanent obstacle within the flight is equivalent to one spanning [t_lo, t_hi]
            t0c, t1c = max(vol.t_start, t_lo), min(vol.t_end, t_hi)
            s = vol.shape
            if isinstance(s, BoxSpec):
                out.append({"kind": "box", "R": s.rotation(), "c": np.array(s.center, float),
                            "half": np.array(s.extents, float) / 2.0 + infl,
                            "t0": t0c, "t1": t1c})
            elif isinstance(s, CylinderSpec):
                out.append({"kind": "cyl", "cx": s.cx, "cy": s.cy, "cz": (s.z_lo + s.z_hi) / 2.0,
                            "radius": s.radius + infl, "hz": (s.z_hi - s.z_lo) / 2.0 + infl,
                            "t0": t0c, "t1": t1c})
            if len(out) >= self.max_obstacles:
                break
        return out
