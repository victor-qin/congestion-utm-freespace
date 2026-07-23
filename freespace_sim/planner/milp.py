"""MILP trajectory-optimization planner (Richards & How big-M), warm-started.

The *global* trajectory optimizer. On an absolute-time step grid, position per step
is continuous; a **binary per obstacle face per active step** encodes the non-convex "stay outside"
disjunction (pass left/right/over/under), so branch-and-bound enumerates *which side* of every
obstacle to pass and returns the globally cheapest route — the homotopy choice a local smoother
can't make. Boxes map to their (oriented) linear faces; cylinders to a circumscribed polygon; the L2
detour length to a polyhedral norm (linear). Departure delay is a decision variable too, carrying
two temporal faces per obstacle ("pass before / after its window"), so a single solve trades delay
against detour globally.

A *warm planner* (straight-line by default; A* in `astar_milp`) supplies a candidate + fallback, and two flags reshape the
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

import warnings

import numpy as np
import pulp

from ..config import SimConfig
from ..cost import endpoint_altitude_change_m, trajectory_cost
from ..geometry import BoxSpec, CylinderSpec
from ..ledger import ReservationLedger
from ..types import FlightRequest, IntentStatus, OperationalIntent, as_terminal
from ..volumes import build_reservation_from_corners, fold_corners_to_columns
from .straight import StraightLineTimeShift
from .terminal_capacity import TerminalCapacity

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

    Terminal-aware: hub geometry is folded to the column edge, tagged, and pad-capacity gated via
    ``TerminalCapacity`` (see ``_verify_with_delay`` / ``_capacity_ok``), so the sim admits the MILP
    family under ``terminal_airspace_always_active``.
    """

    plans_terminal_airspace = True   # consumed by sim._wall_aware (always-active admission gate)

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
        max_steps: int = 60,         # knot cap; floored by kinematic feasibility in _solve
        max_obstacles: int = 40,
        time_limit_s: float = 20.0,    # hard cap (backstop for genuinely hard MILPs the gap can't close)
        gap_rel: float | None = 0.01,  # stop CBC once the incumbent is within 1% of optimal
    ):
        self.warm_planner = warm_planner or StraightLineTimeShift()
        self.gap_rel = gap_rel
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
        # per-ledger pad-capacity authority (bound lazily in _capacity, AStarPlanner._occupancy-style)
        self._tcap: TerminalCapacity | None = None
        self._tcap_ledger: ReservationLedger | None = None
        self._tcap_seen = 0

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
        o_term, d_term = as_terminal(req.origin_terminal), as_terminal(req.dest_terminal)
        tcap = self._capacity(ledger, cfg, req.t_request)
        try:
            milp = self._solve(req, ledger, cfg, fixed, ref, o_term, d_term, tcap)
            if milp is None and ref is not None:
                milp = self._solve(req, ledger, cfg, fixed, None, o_term, d_term, tcap)  # unlock
        except Exception as e:
            # Degrade to the warm candidate, but never SILENTLY: a swallowed bug in the solver /
            # fold / capacity path looks exactly like a hard flight (warm fallback or denial), which
            # is how two real defects stayed invisible. The warning is the tripwire.
            warnings.warn(
                f"milp _solve raised {type(e).__name__}: {e} — falling back to the warm candidate",
                RuntimeWarning, stacklevel=2)
            milp = None
        cands = [i for i in (warm, milp) if i is not None and i.accepted]
        if not cands:
            warm.planner = "milp"
            return warm                              # both denied
        best = min(cands, key=lambda i: i.cost)      # global delay-vs-detour trade-off, then min
        best.planner = "milp"
        return best

    def _capacity(self, ledger, cfg, t_request) -> TerminalCapacity:
        """Per-ledger ``TerminalCapacity`` binding — the pad-capacity authority, same lifecycle as
        ``AStarPlanner._occupancy``: construct + subscribe ONCE per ledger and absorb the committed
        backlog; on a ledger shrink (a release) reset + re-absorb WITHOUT re-subscribing (a second
        subscription would leak a dead observer); evict expired dwells every plan.

        Per-planner-instance by design (the established A* pattern): in ``astar_milp`` both the MILP
        and its warm A* keep an index on the shared ledger — duplicated commit work, consistent by
        construction; a shared per-ledger authority is a possible future dedup."""
        if self._tcap_ledger is not ledger:
            self._tcap = TerminalCapacity(cfg, ledger)
            ledger.subscribe(self._tcap.on_commit)
            self._absorb_committed(ledger)
            self._tcap_ledger = ledger
        elif ledger.n_volumes < self._tcap_seen:
            self._tcap.reset()
            self._absorb_committed(ledger)
        self._tcap_seen = ledger.n_volumes
        self._tcap.evict_before(t_request)
        return self._tcap

    def _absorb_committed(self, ledger) -> None:
        by_fid: dict = {}
        for fid, vol in ledger.iter_committed():
            by_fid.setdefault(fid, []).append(vol)
        for fid, vols in by_fid.items():
            self._tcap.on_commit(fid, vols)

    def _capacity_ok(self, volumes, o_term, d_term, tcap) -> bool:
        """Pad-capacity gate on the REBUILT tagged dwell columns (their windows carry the exact
        committed timing, so no window math is duplicated here). This is the check ``any_conflict``
        cannot make: same-hub columns are conflict-EXEMPT, so only ``TerminalCapacity.admits``'
        interval count stands between the plan and an over-subscribed pad."""
        if tcap is None or (o_term is None and d_term is None):
            return True
        for v in volumes:
            if v.terminal_id is None or not isinstance(v.shape, CylinderSpec):
                continue
            term = o_term if (o_term is not None and v.terminal_id == o_term.id) else d_term
            if term is not None and v.terminal_id == term.id and not tcap.admits(
                    v.terminal_id, v.t_start, v.t_end, term.capacity):
                return False
        return True

    def _solve(self, req, ledger, cfg, fixed_delay=None, ref_path=None,
               o_term=None, d_term=None, tcap=None) -> OperationalIntent | None:
        """Build and solve the big-M MILP for one flight; return an ACCEPTED intent or None.

        Variables: cruise position per step (xy endpoints pinned by bounds; the entry/exit cruise
        ALTITUDE ``pz[0]``/``pz[-1]`` is free in the band [z_min_m, z_max_m] — choosing the level is
        part of the solve); per-segment horizontal length ``L_k`` (a polyhedral-norm lower bound on
        ‖Δxy‖) and vertical travel ``V_k``; and the departure delay ``d`` (a variable, or the
        ``fixed_delay`` constant). Cruise time at knot k is ``t_depart + d + climb(pz₀) + ΣL/v`` —
        the entry climb is affine in the chosen level, and time is otherwise tied to path LENGTH
        (not a fixed dt/step), so the vehicle can't "slow down for free" to dodge a time window.
        Mid-route climb time is NOT in this clock (only the rebuild charges it); `_verify_with_delay`
        absorbs the residual by delay-bumping.

        Each nearby obstacle adds, per segment, a disjunction "outside it on one spatial face OR the
        whole segment passes before/after its window" (big-M binaries, ≥1 enforced). With ``ref_path``
        set, those binaries are pinned to the warm path's choices, turning the solve into a fast LP.
        Objective = c_lat·ΣL + c_alt·(ΣV + entry climb + exit descent) + c_gd·d. The solved corners
        are rebuilt into real committed boxes and re-checked (`_verify_with_delay`) — correctness
        never relies on the LP being exact.
        """
        origin = np.asarray(req.origin, float)
        dest = np.asarray(req.dest, float)
        t_depart = req.t_departure if req.t_departure is not None else req.t_request
        z_lo, z_hi = cfg.z_min_m, cfg.z_max_m
        start = np.array([origin[0], origin[1], z_lo])
        goal = np.array([dest[0], dest[1], z_lo])
        straight_horiz = float(np.linalg.norm(goal[:2] - start[:2]))
        v_step = cfg.nominal_speed_mps * cfg.dt_s
        z_step = cfg.climb_rate_mps * cfg.dt_s
        if straight_horiz < _EPS:
            return None

        N = min(self.max_steps, int(np.ceil(self.detour_allow * straight_horiz / v_step)) + 2)
        # the max_steps cap must never make the trip ITSELF infeasible: the speed polygon's inradius
        # is v_step, so (N−1)·v_step ≥ straight is the kinematic floor. A flight past the cap gets
        # less DETOUR headroom (delay/altitude stay free) — never a vacuously infeasible model
        # (pre-#36 this silently denied every >7 km flight once no spatial warm masked it).
        N = max(N, int(np.ceil(straight_horiz / v_step)) + 2)
        # the warm path resampled to the MILP grid → tells us which side/when the homotopy goes
        ref_pos, ref_t = (None, None)
        if ref_path is not None and len(ref_path) >= 2:
            ref_pos, ref_t = self._resample(ref_path, N)
        # reachable absolute-time range across all delays in [0, max_ground_delay] AND all entry
        # levels in the band (the actual cruise start is the affine t_depart + d + climb(pz[0]))
        t_lo = t_depart + cfg.climb_time_to(z_lo) - cfg.time_buffer_s
        t_hi = (t_depart + cfg.max_ground_delay_s + cfg.climb_time_to(z_hi)
                + (N - 1) * cfg.dt_s + cfg.time_buffer_s)
        exempt_tids = frozenset(t.id for t in (o_term, d_term) if t is not None)
        obstacles = self._nearby_obstacles(ledger, start, goal, t_lo, t_hi, cfg, exempt_tids)

        # Problem-scaled big-Ms — a loose M weakens the LP relaxation and slows CBC. m_space is a
        # safe over-bound (~2× the region diagonal); tightening it toward the diagonal is a possible
        # future speedup, but model SIZE (see the lens pruning below) is the dominant lever.
        m_space = 3.0 * max(cfg.region_size_m)       # bounds |Rᵀ(p−c)| over the region
        m_time = (t_hi - t_lo) + cfg.max_ground_delay_s + 100.0   # bounds the time-face range

        prob = pulp.LpProblem("milp_traj", pulp.LpMinimize)
        # departure delay: a free variable (global delay-vs-detour trade-off) or a fixed constant
        # (pure spatial refiner — e.g. seeded by A*, which already chose the delay)
        d = float(fixed_delay) if fixed_delay is not None else pulp.LpVariable(
            "delay", 0, cfg.max_ground_delay_s
        )
        # position variables: xy endpoints fixed by tight bounds; z endpoints FREE in the band —
        # the ground↔cruise climb is flown inside the hover column, so pz[0]/pz[-1] are the chosen
        # entry/exit cruise levels, priced in the objective and timed by the affine climb below
        px = [pulp.LpVariable(f"x{k}", 0, cfg.region_size_m[0]) for k in range(N)]
        py = [pulp.LpVariable(f"y{k}", 0, cfg.region_size_m[1]) for k in range(N)]
        pz = [pulp.LpVariable(f"z{k}", z_lo, z_hi) for k in range(N)]
        for arr, s, g in ((px, start[0], goal[0]), (py, start[1], goal[1])):
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
        # cruise-clock origin: the ENTRY climb is affine in the (free) entry altitude pz[0] — a
        # higher chosen level starts the cruise later, exactly as the rebuild's climb_time_to(z₀)
        climb0 = (pz[0] - cfg.ground_level_m) / cfg.climb_rate_mps

        # big-M obstacle avoidance. SPATIAL faces are per SEGMENT and shared by both endpoints: a
        # keepout face is a half-space, so both endpoints outside the SAME face ⇒ the whole straight
        # segment is outside it (convexity) — no corner-cutting between differently-faced samples,
        # and no hopping a thin obstacle within one step. TEMPORAL faces are per segment too and use
        # the segment's start/end time — because the committed corridor box spans the whole segment
        # (± time_buffer), so the whole segment must clear the window, not just one instant.
        #
        # (obs, segment) REACHABILITY PRUNING: knot k lies within k·cap of the pinned start and
        # (N-1-k)·cap of the pinned goal (each step's xy displacement is bounded by the circumscribed
        # speed polygon), so a pair whose obstacle lies outside either lens can never be violated —
        # its whole disjunction is omitted. Sound, and it kills the O(obstacles × segments) binary
        # blow-up that made CBC time out with no incumbent once committed traffic accumulated.
        cap = v_step / float(np.cos(np.pi / self.n_dirs)) + 1e-6
        sx, sy = float(start[0]), float(start[1])
        gx, gy = float(goal[0]), float(goal[1])
        for oi, obs in enumerate(obstacles):
            if obs["kind"] == "box":
                ocx, ocy = float(obs["c"][0]), float(obs["c"][1])
                orad = float(np.linalg.norm(obs["half"])) + self.keepout_margin_m
            else:
                ocx, ocy = float(obs["cx"]), float(obs["cy"])
                orad = float(obs["radius"]) + self.keepout_margin_m
            d_s = float(np.hypot(ocx - sx, ocy - sy)) - orad
            d_g = float(np.hypot(ocx - gx, ocy - gy)) - orad
            for k in range(N - 1):
                if d_s > (k + 1) * cap or d_g > (N - 1 - k) * cap:
                    continue                                     # segment k can never touch this obstacle
                # margin = box time-buffer + one segment-time, since the obstacle crossing can fall
                # mid-segment while we constrain the segment's start/end time (keeps the rebuild safe)
                tmarg = cfg.time_buffer_s + cfg.dt_s
                tlock = False
                if obs["t0"] <= t_lo + _EPS and obs["t1"] >= t_hi - _EPS:
                    # permanent within the reachable horizon (an always-active wall, or a committed
                    # volume clamped to it): no temporal escape exists — omit the two dead binaries
                    temporal = []
                else:
                    t_start = t_depart + d + climb0 + cumL[k] / v
                    t_end = t_depart + d + climb0 + (cumL[k] + L[k]) / v
                    b_before = pulp.LpVariable(f"tb{oi}_{k}", cat="Binary")
                    b_after = pulp.LpVariable(f"ta{oi}_{k}", cat="Binary")
                    prob += t_end <= (obs["t0"] - tmarg) + m_time * b_before   # whole seg before window
                    prob += t_start >= (obs["t1"] + tmarg) - m_time * b_after  # whole seg after window

                    # HOMOTOPY LOCK: copy the warm path's temporal choice (before / after / neither)
                    if ref_pos is not None:
                        if ref_t[k + 1] + tmarg <= obs["t0"]:                  # warm passes before
                            _fix(b_before, 0)
                            _fix(b_after, 1)
                            tlock = True
                        elif ref_t[k] - tmarg >= obs["t1"]:                    # warm passes after
                            _fix(b_after, 0)
                            _fix(b_before, 1)
                            tlock = True
                        else:                                                  # warm avoids spatially
                            _fix(b_before, 1)
                            _fix(b_after, 1)
                    temporal = [b_before, b_after]

                ends = ((px[k], py[k], pz[k]), (px[k + 1], py[k + 1], pz[k + 1]))
                ref_ends = None if ref_pos is None else (ref_pos[k], ref_pos[k + 1])
                self._add_spatial_keepout(prob, f"{oi}_{k}", ends, obs, m_space,
                                          temporal, ref_ends=ref_ends,
                                          temporal_locked=tlock)

        prob += (cfg.cost_air_lateral_per_m * pulp.lpSum(L)
                 + cfg.cost_altitude_change_per_m * (
                     pulp.lpSum(V)                              # interior climbs/descents
                     + (pz[0] - cfg.ground_level_m)             # entry climb to the chosen level
                     + (pz[-1] - cfg.ground_level_m))           # final descent from the exit level
                 + cfg.cost_ground_delay_per_s * d)             # detour vs altitude vs delay, globally
        solver_kw = {"msg": 0, "timeLimit": self.time_limit_s}
        if self.gap_rel is not None:
            solver_kw["gapRel"] = self.gap_rel   # stop proving within gap_rel of optimal (big-M tail)
        prob.solve(pulp.PULP_CBC_CMD(**solver_kw))
        # "Not Solved" = CBC stopped on its budget; when it holds an incumbent it LOADS the values
        # (else they stay None). A loaded incumbent is a perfectly usable candidate — correctness
        # comes from the rebuild + ledger recheck below, never from the optimality proof — so only
        # a truly valueless stop (infeasible / no incumbent) returns None.
        if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved") or px[1].value() is None:
            return None

        d_opt = float(fixed_delay) if fixed_delay is not None else float(d.value() or 0.0)
        corners = [np.array([px[k].value(), py[k].value(), pz[k].value()], float) for k in range(N)]
        # The MILP fixes the spatial homotopy + an approximate delay; polish the delay with a small
        # jump-to-gap on the *rebuilt* path so it verifies exactly (absorbs the LP↔rebuild timing gap).
        polished = self._verify_with_delay(corners, origin, dest, t_depart, d_opt, straight_horiz,
                                           cfg, ledger, o_term, d_term, tcap)
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
            altitude_change_m=endpoint_altitude_change_m(
                float(corners[0][2]), float(corners[-1][2]), cum_dz, cfg),
            planner="milp",
        )
        intent.cost = trajectory_cost(intent, cfg)
        return intent

    def _verify_with_delay(self, corners, origin, dest, t_depart, d_start, straight_horiz, cfg,
                           ledger, o_term=None, d_term=None, tcap=None):
        """Fold the corners to the terminal column edges, then step the delay up from the MILP's
        value until the rebuilt path is conflict-free AND pad-capacity-admitted, then splice out
        redundant knots at that delay.

        Returns (volumes, centerline, cum_horiz, cum_dz, delay) or None if the spatial path can't be
        made feasible by waiting within budget (then the caller falls back to the warm intent).
        """
        corners = fold_corners_to_columns(corners, origin, dest, o_term, d_term, cfg)
        d = max(0.0, d_start)
        while d <= cfg.max_ground_delay_s + _EPS:
            volumes, centerline, cum_horiz, cum_dz = build_reservation_from_corners(
                corners, origin, dest, t_depart, d, cfg, origin_term=o_term, dest_term=d_term
            )
            if straight_horiz > _EPS and cum_horiz / straight_horiz > cfg.max_detour_factor:
                return None
            if (not ledger.any_conflict(volumes)
                    and self._capacity_ok(volumes, o_term, d_term, tcap)):
                built = self._splice(corners, origin, dest, t_depart, d, cfg, ledger,
                                     (volumes, centerline, cum_horiz, cum_dz), o_term, d_term, tcap)
                return (*built, d)
            d += cfg.dt_s
        return None

    def _splice(self, corners, origin, dest, t_depart, d, cfg, ledger, best,
                o_term=None, d_term=None, tcap=None):
        """Greedy single-knot splice at the verified delay — the ShortcutRefiner fixpoint sweep on
        the raw solved corners. It removes the lateral wiggle CBC cannot see: the polyhedral ``L`` is
        a lower bound on true L2 length (up to ~2% short at 16 directions) and ``gapRel`` adds more
        slack, so CBC's vertex can wander metres off the chord "for free"; the rebuild would bill
        that as air detour. A removal is kept iff the rebuilt reservation stays conflict-free AND
        still pad-capacity-admitted — a shorter path lands EARLIER, shifting the dest dwell window,
        so capacity must be re-checked. Removals only shorten the path (triangle inequality), so the
        detour budget never re-trips.
        """
        corners = [np.asarray(c, float) for c in corners]
        changed = True
        while changed and len(corners) > 2:
            changed = False
            i = 1
            while i < len(corners) - 1:
                cand = corners[:i] + corners[i + 1:]
                rebuilt = build_reservation_from_corners(
                    cand, origin, dest, t_depart, d, cfg, origin_term=o_term, dest_term=d_term)
                if (not ledger.any_conflict(rebuilt[0])
                        and self._capacity_ok(rebuilt[0], o_term, d_term, tcap)):
                    corners, best, changed = cand, rebuilt, True
                else:
                    i += 1
        return best

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

    def _add_spatial_keepout(self, prob, tag, ends, obs, m_space, temporal,
                             ref_ends=None, temporal_locked=False) -> int:
        """Spatial big-M faces for one SEGMENT (both endpoints share the binaries), OR'd with the
        segment's temporal faces.

        The disjunction is: BOTH endpoints beyond the same spatial face — a face is a half-space, so
        by convexity the whole straight segment is then outside the obstacle (no corner-cutting, no
        hopping a thin obstacle mid-step) — OR the whole segment passes before/after the window (the
        two `temporal` binaries). When ``ref_ends`` is given, the binaries are pinned to the warm
        path's choice (which side / before-after) so CBC doesn't re-enumerate — the MILP becomes an
        LP that tightens *within* that homotopy. Returns the number of binaries in the disjunction.
        """
        m = self.keepout_margin_m
        faces = list(temporal)

        if obs["kind"] == "box":
            R, c, half = obs["R"], obs["c"], obs["half"] + m
            bs = [pulp.LpVariable(f"b{tag}_{a}_{s}", cat="Binary") for a in range(3) for s in (0, 1)]
            for x, y, zz in ends:
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
            for x, y, zz in ends:
                for i in range(self.cyl_faces):
                    a = 2 * np.pi * i / self.cyl_faces
                    prob += np.cos(a) * (x - cx) + np.sin(a) * (y - cy) >= r - m_space * bs[i]
                prob += (zz - cz) >= hz - m_space * bs[self.cyl_faces]
                prob += (zz - cz) <= -hz + m_space * bs[self.cyl_faces + 1]

        faces += bs
        prob += pulp.lpSum(faces) <= len(faces) - 1              # ≥1 face (spatial OR temporal) enforced

        if ref_ends is not None:                                 # pin spatial binaries to warm choice
            if temporal_locked:
                for b in bs:                                     # a temporal face already carries it
                    _fix(b, 1)
            else:
                (i0, m0), (i1, m1) = (self._best_spatial_face(obs, rp, m) for rp in ref_ends)
                if i0 == i1 and min(m0, m1) > self.lock_margin_m:
                    # BOTH warm endpoints clearly beyond the same face → pin the segment to it
                    for j, b in enumerate(bs):
                        _fix(b, 0 if j == i0 else 1)
                # else: a transition segment near the obstacle — leave it free for CBC (keeps it
                # feasible; the endpoints straddle faces exactly where the warm path turns a corner)
        return len(faces)

    def _nearby_obstacles(self, ledger, start, goal, t_lo, t_hi, cfg,
                          exempt_tids: frozenset = frozenset()) -> list[dict]:
        """Committed volumes + always-active walls that could constrain this flight → obstacle dicts.

        Keeps volumes whose time window overlaps [t_lo, t_hi] and whose footprint falls in an xy reach
        box around the route (capped at ``max_obstacles``). Each footprint is inflated by the new
        corridor's half-width (Minkowski) and its window clamped to the reachable horizon — so a
        permanent obstacle within the flight gets a finite ``t1`` the tight big-M can disable. Box →
        ``{R, c, half}``; cylinder → ``{cx, cy, cz, radius, hz}``; both carry ``t0``/``t1``.

        Own-hub exemption (``exempt_tids`` = the flight's origin/dest terminal ids): committed and
        static CYLINDERS with an exempt tag are skipped — this flight's tagged geometry may share its
        own hub's columns/wall (conflict.volumes_conflict same-tid+cylinder), and pad capacity is
        gated separately by ``TerminalCapacity``. Sibling corridor BOXES stay obstacles (box↔box is
        never exempt). Foreign hubs' static walls are harvested FIRST — they are few and permanent,
        and must not be crowded out of the ``max_obstacles`` cap by transient traffic.
        """
        reach = self.detour_allow * float(np.linalg.norm(goal[:2] - start[:2])) + cfg.corridor_width_m
        lo = np.minimum(start, goal) - reach
        hi = np.maximum(start, goal) + reach
        infl = cfg.corridor_width_m / 2.0
        out: list[dict] = []
        for vol in ledger.static_volumes():          # always-active foreign-hub walls (cylinders)
            if vol.terminal_id in exempt_tids:
                continue
            amin, amax = vol.aabb()
            if bool(np.any(amax < lo) or np.any(amin > hi)):
                continue
            s = vol.shape
            if isinstance(s, CylinderSpec) and len(out) < self.max_obstacles:
                out.append({"kind": "cyl", "cx": s.cx, "cy": s.cy, "cz": (s.z_lo + s.z_hi) / 2.0,
                            "radius": s.radius + infl, "hz": (s.z_hi - s.z_lo) / 2.0 + infl,
                            "t0": t_lo, "t1": t_hi})
        for _fid, vol in ledger.iter_committed():
            if len(out) >= self.max_obstacles:
                break
            if not (vol.t_start < t_hi and t_lo < vol.t_end):
                continue
            if vol.terminal_id in exempt_tids and isinstance(vol.shape, CylinderSpec):
                continue                             # own-hub column — shared, capacity-gated instead
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
