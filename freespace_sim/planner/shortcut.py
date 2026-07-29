"""Deterministic greedy shortcut post-pass — drop redundant knots without losing feasibility.

After any planner returns a corner polyline, repeatedly splice out an interior knot i (replace
``i-1 → i → i+1`` with ``i-1 → i+1``) and keep the removal iff the *rebuilt* reservation stays
conflict-free against the ledger and within the detour budget — the same build-then-check contract
every planner obeys. Triangle inequality guarantees a removal never lengthens the horizontal path,
so the result is always ≤ the input length and never invents a new conflict.

A deterministic single-knot fixpoint sweep, so a hex staircase collapses (each removal re-enables
the next) without depending on random shortcut draws. It re-checks against the *real committed
obstacles*, not A*'s conservative
inflated raster, so it can tighten A*'s over-wide berth toward the true continuous clearance — much
of what the MILP does, but greedy and solver-free. Wrap any planner with ``ShortcutRefiner``.
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig
from ..cost import endpoint_altitude_change_m, trajectory_cost
from ..ledger import ReservationLedger
from ..types import FlightRequest, IntentStatus, OperationalIntent
from ..volumes import (build_reservation_from_corners, enroute_detour_m,
                       enroute_flown_m, enroute_reference_m)

_EPS = 1e-9


def _rebuild(corners, origin, dest, t_depart, g_delay, cfg, ledger, straight_horiz,
             origin_term=None, dest_term=None, corridor_t0=None):
    """Resample corners → ≤120 m corridor boxes, then budget + ledger conflict check.

    Returns (volumes, centerline, cum_horiz, cum_dz) or None if it busts the detour budget or
    overlaps a committed reservation. This is the feasibility oracle the greedy sweep consults.
    ``origin_term``/``dest_term`` preserve the inner A*'s terminal tags through the rebuild;
    ``corridor_t0`` anchors the corridor at the inner planner's VERIFIED first-cruise stamp.
    """
    volumes, centerline, cum_horiz, cum_dz = build_reservation_from_corners(
        corners, origin, dest, t_depart, g_delay, cfg, origin_term=origin_term, dest_term=dest_term,
        corridor_t0=corridor_t0,
    )
    # Both sides span lane → lane via the same helper metrics uses (issue #50), so the gate enforces
    # exactly the ratio the caller will report as air_detour_m / stretch.
    flown = enroute_flown_m([p for p, _ in centerline], origin, dest, origin_term, dest_term, cfg)
    if straight_horiz > _EPS and flown / straight_horiz > cfg.max_detour_factor:
        return None
    if ledger.any_conflict(volumes):
        return None
    return volumes, centerline, cum_horiz, cum_dz


def shortcut_corners(corners, origin, dest, t_depart, g_delay, cfg: SimConfig,
                     ledger: ReservationLedger, origin_term=None, dest_term=None, corridor_t0=None):
    """Greedily drop interior knots whose removal stays conflict-free; return simplified corners.

    Deterministic single-knot fixpoint: sweep interior knots front-to-back, remove any whose removal
    rebuilds conflict-free, and repeat full sweeps until one removes nothing. Endpoints (the climb-top
    and descent-top) are never dropped. If the input path is itself infeasible to rebuild, it is
    returned unchanged (the caller keeps the planner's verified original).
    """
    corners = [np.asarray(c, float) for c in corners]
    if len(corners) <= 2:
        return corners
    straight_horiz = enroute_reference_m(origin, dest, origin_term, dest_term, cfg)
    if _rebuild(corners, origin, dest, t_depart, g_delay, cfg, ledger, straight_horiz,
                origin_term, dest_term, corridor_t0) is None:
        return corners
    changed = True
    while changed and len(corners) > 2:
        changed = False
        i = 1
        while i < len(corners) - 1:
            cand = corners[:i] + corners[i + 1:]
            if _rebuild(cand, origin, dest, t_depart, g_delay, cfg, ledger, straight_horiz,
                        origin_term, dest_term, corridor_t0) is not None:
                corners = cand           # removed knot i; re-test the same index (list shifted)
                changed = True
            else:
                i += 1
    return corners


class ShortcutRefiner:
    """Wrap any planner; greedily simplify its accepted path, keeping it only if cheaper.

    A pure post-process: it never makes a plan worse (rebuild is re-verified against the ledger and
    the result is returned only when its cost is ≤ the original). Collapses mid-air holds first, since
    the rebuild re-times at nominal speed — if a hold was load-bearing for temporal deconfliction the
    rebuilt path will conflict and the original is kept.
    """

    def __init__(self, inner, label: str | None = None):
        self.inner = inner
        self.label = label

    def plan(self, req: FlightRequest, ledger: ReservationLedger, cfg: SimConfig) -> OperationalIntent:
        intent = self.inner.plan(req, ledger, cfg)
        if not intent.accepted or not intent.centerline or len(intent.centerline) <= 3:
            return intent

        corners: list[np.ndarray] = []
        for p, _ in intent.centerline:                 # collapse repeated positions (holds)
            p = np.asarray(p, float)
            if not corners or not np.allclose(p, corners[-1]):
                corners.append(p)
        if len(corners) <= 2:
            return intent

        g_delay = intent.ground_delay_s
        # Read the takeoff time off the committed origin column rather than inverting the centerline.
        # The old inverse subtracted only the climb, but issue #52 made centerline[0] land at
        # `takeoff + climb + Lane.steps*dt`, so the recovered departure came out LATE by the traverse
        # (measured 15 s on a 180 m hub: 12 s of lane traverse plus 3 s of pre-existing takeoff-step
        # rounding, which the old inverse also dropped) and every rebuilt volume shifted with it —
        # leaving the origin column unreserved while the drone was still in it. volumes[0].t_start IS
        # the takeoff time in both builders, so this needs no knowledge of which lane was taken.
        t_depart = intent.volumes[0].t_start - g_delay
        # Anchor the rebuilt corridor at the inner planner's VERIFIED first-cruise stamp: a refiner
        # re-times splices, not the takeoff. Re-deriving the start inside the rebuild mixes its
        # continuous clock (climb_time_to + WORST lane) with A*'s quantised stamp (climb_steps*dt +
        # CHOSEN lane's steps) — measured -3..+1 s on every rebuilt volume, lane-dependent.
        t_first = float(intent.centerline[0][1])
        ot, dt = req.origin_terminal, req.dest_terminal
        simplified = shortcut_corners(corners, req.origin, req.dest, t_depart, g_delay, cfg, ledger,
                                      ot, dt, corridor_t0=t_first)
        if len(simplified) >= len(corners):
            return intent                              # nothing removed

        straight = enroute_reference_m(req.origin, req.dest, ot, dt, cfg)
        built = _rebuild(simplified, req.origin, req.dest, t_depart, g_delay, cfg, ledger, straight,
                         ot, dt, corridor_t0=t_first)
        if built is None:
            return intent
        volumes, centerline, cum_horiz, cum_dz = built
        # Lattice overhead: decrement by what the sweep actually removed rather than inheriting the
        # inner planner's figure (which would over-report staircase on an already-straightened path).
        # Splicing out a redundant knot removes hex staircase before it removes any real berth, so
        # attributing the removal to overhead first is the right first-order split — approximate for
        # the refiner, exact for bare A*. Floored at 0 and monotone, so it can never invent overhead.
        inner_horiz = float(np.linalg.norm(np.diff(np.array(corners)[:, :2], axis=0), axis=1).sum())
        removed = max(0.0, inner_horiz - cum_horiz)
        refined = OperationalIntent(
            request=req, status=IntentStatus.ACCEPTED, volumes=volumes, centerline=centerline,
            ground_delay_s=g_delay, air_hold_s=0.0,
            air_detour_m=enroute_detour_m(
                enroute_flown_m([p for p, _ in centerline], req.origin, req.dest, ot, dt, cfg),
                straight),                                                              # issue #50
            lattice_overhead_m=max(0.0, intent.lattice_overhead_m - removed),
            altitude_change_m=endpoint_altitude_change_m(
                float(np.asarray(centerline[0][0])[2]), float(np.asarray(centerline[-1][0])[2]),
                cum_dz, cfg),
            planner=self.label or f"{intent.planner}+sc",
        )
        refined.cost = trajectory_cost(refined, cfg)
        return refined if refined.cost <= intent.cost + _EPS else intent
