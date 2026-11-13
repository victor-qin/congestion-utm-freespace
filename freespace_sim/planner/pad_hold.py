"""Ground-level pad holds for an itinerary's parked aircraft, as the A* and SIPP searches see them.

A round trip parks its aircraft on the customer pad from the outbound's landing until the return's
takeoff (``volumes.ground_dwell_reservation``, a low box: traffic may overfly it, nothing may reach
the ground beside it). The two legs are planned separately, so each search must keep that box clear
itself: the outbound only lands where the pad stays clear through the turnaround, and the return
only holds on the ground until the first committed volume reaches the pad. Both are exact ledger
queries, made once per plan.
"""

from __future__ import annotations

import math

from ..config import SimConfig
from ..types import FlightRequest
from ..volumes import ground_dwell_reservation


def _ground_hits(ledger, center, t0: float, t1: float, cfg: SimConfig) -> list:
    """Sorted ``(t_start, t_end)`` of committed volumes that conflict with an aircraft parked at
    ``center`` over ``[t0, t1)``; empty when ``t1 <= t0``. Permanent terminal walls are skipped, as in
    ``ItineraryPlanner._compose``."""
    if t1 <= t0:
        return []
    probe = ground_dwell_reservation(center, t0, t1 - t0, cfg)
    return sorted((v.t_start, v.t_end) for fid, v in ledger.conflicts([probe])
                  if fid != ledger.STATIC_WALL_FID)


def first_ground_hit_s(ledger, center, since: float, until: float, cfg: SimConfig) -> float:
    """When committed traffic first reaches an aircraft parked at ``center`` from ``since``, looking
    up to ``until``; ``math.inf`` if it never does."""
    hits = _ground_hits(ledger, center, since, until, cfg)
    return min((max(a, since) for a, b in hits if b > since), default=math.inf)


def parked_ground_max_step(req: FlightRequest, ledger, cfg: SimConfig, base: int,
                           ground_max_step: int) -> int:
    """
    The last step a leg may still be on the ground at its origin.

    For a return leg whose aircraft is parked on an ordinary pad (``req.parked_since_s``), a ground
    hold keeps the parked box alive, so the hold must end — the takeoff column must start — before
    the first committed volume reaches the pad. Otherwise ``ground_max_step`` is returned unchanged.

    Parameters
    ------------
    - req (FlightRequest): the leg being planned.
    - ledger (ReservationLedger): committed traffic.
    - cfg (SimConfig): timing and the pad footprint.
    - base (int): the leg's first takeoff step; the result never drops below it.
    - ground_max_step (int): the configured ground-delay bound.

    Return
    --------
    - output (int): ``min(ground_max_step, last clear takeoff step)``, at least ``base``.
    """
    if req.parked_since_s is None or req.origin_terminal is not None:
        return ground_max_step
    first = first_ground_hit_s(ledger, req.origin, float(req.parked_since_s),
                               (ground_max_step + 1) * cfg.dt_s, cfg)
    if first == math.inf:
        return ground_max_step
    return max(base, min(ground_max_step, int(math.floor(first / cfg.dt_s + 1e-9))))


def landing_hold_hits(req: FlightRequest, ledger, cfg: SimConfig, base: int, max_step: int,
                      dwell_steps) -> list:
    """
    Committed ground traffic at an ordinary dest pad that an outbound landing must keep clear of.

    Parameters
    ------------
    - req (FlightRequest): the leg being planned; ``req.pad_hold_after_s`` > 0 marks an outbound.
    - ledger (ReservationLedger): committed traffic.
    - cfg (SimConfig): timing and the pad footprint.
    - base (int): the leg's first step.
    - max_step (int): the last step the search may land at.
    - dwell_steps (Sequence[int]): landing column length per level.

    Return
    --------
    - output (list[tuple[float, float]]): sorted hit intervals; empty when nothing is held.
    """
    if req.pad_hold_after_s <= 0.0 or req.dest_terminal is not None:
        return []
    t1 = (max_step + max(dwell_steps)) * cfg.dt_s + req.pad_hold_after_s
    return _ground_hits(ledger, req.dest, base * cfg.dt_s, t1, cfg)


def landing_hold_clear(hits, s: int, dwell: int, req: FlightRequest, cfg: SimConfig) -> bool:
    """True iff landing at step ``s`` keeps the pad clear of ``hits`` from the landing column's
    start through ``req.pad_hold_after_s`` past its end (a superset of the parked turnaround)."""
    if not hits:
        return True
    t0 = s * cfg.dt_s
    t1 = (s + dwell) * cfg.dt_s + req.pad_hold_after_s
    return not any(a < t1 and t0 < b for a, b in hits)
